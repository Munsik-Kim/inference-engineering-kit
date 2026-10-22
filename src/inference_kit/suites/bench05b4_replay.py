"""B4 CPU replay: existing SSE decoder and numeric probe math, task-specific joins."""
import collections
import json
from pathlib import Path
import statistics
from ..core.io import read, write, require, sha, digest, safe_name
from ..core.state import derived_output, preserve_failure
from ..core.evidence import open_bundle
from ..tasks import context_evidence as task
from ..tasks.base import compatible
from ..adapters.protocol import Adapter
from ..adapters.http import sha as wire_hash
from .bench05b1_replay import jsonl, verify_raw, generation_from_raw, probe_metrics, memory
from .bench05a_replay import csv_output, load_job
from .bench05b4_plan import validate_data


def write_jsonl(path,rows):
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def requests_for(plan,arm):
    prepared={s['sample_id']:s for s in plan['samples']['generation_rows']};adapter=task.ContextEvidenceAdapter()
    return {s['sample_id']:adapter.requests(s|{'generation_input_ids':prepared[s['sample_id']]['generation_input_ids'],'tokenizer_identity':plan['samples']['tokenizer_identity']},
        plan['plan_identity'][:16]+'|attempt1|'+arm['arm_id'],plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity'])[0] for s in plan['data']['rows']}


def decode_generations(root,job,plan,arm):
    if not job:return []
    directory=root/safe_name(job['directory']);p=directory/'task-generations.jsonl'
    records=jsonl(p) if p.is_file() else [];requests=requests_for(plan,arm);by_request={r.request_id:r for r in requests.values()}
    prepared={s['sample_id']:s for s in plan['samples']['generation_rows']};submits={};ends={}
    journal=directory/'request-journal.jsonl'
    for e in jsonl(journal) if journal.is_file() else []:
        require(e['request_id'] in by_request,'Unknown B4 journal request')
        req=by_request[e['request_id']];row=prepared[req.sample_id]
        require(e['identity']==req.identity and e['arm_id']==arm['arm_id'] and e['artifact_identity']==arm['artifact']['artifact_identity'] and e['prompt_hash']==row['prompt_hash'],'B4 journal identity')
        target=submits if e['event']=='http_submit' else ends if e['event']=='response_end' else None
        require(target is not None and e['request_id'] not in target,'Unknown/duplicate journal event')
        if e['event']=='http_submit':require(e['request_body']==Adapter('vllm').body({'token_ids':req.payload['token_ids']},'serving-lab',64,'natural_generation_v1'),'B4 journal wire changed')
        target[e['request_id']]=e
    require(set(ends)<=set(submits) and {r['request_id'] for r in records}<=set(submits),'B4 raw/journal coverage')
    require(len({r['sample_id'] for r in records})==len(records),'Duplicate B4 raw sample')
    for rid,begin in submits.items():
        if any(r['request_id']==rid for r in records):continue
        req=by_request[rid];sample=next(s for s in plan['data']['rows'] if s['sample_id']==req.sample_id);end=ends.get(rid,{})
        records.append(begin|{'task_id':task.TASK,'task_version':task.VERSION,'sample_id':req.sample_id,'sample_identity':sample['sample_identity'],'split':sample['split'],
            'request_kind':'generate_until','output_text':'','transport_complete':False,'termination':'INCOMPLETE_SUBMISSION','completed_monotonic':end.get('completed_monotonic'),
            'first_content_monotonic':None,'last_content_monotonic':None,'request_latency_http':None,'request_latency_task':None,'ttft_http':None,
            'evidence_kind':'synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_task_generation','recovered_from_submission_journal':True})
    normalized=[]
    for record in records:
        require(record['request_id'] in by_request,'Unknown B4 raw request')
        req=by_request[record['request_id']];row=prepared[req.sample_id]
        require(record['identity']==req.identity and record['task_id']==task.TASK and record['task_version']==task.VERSION,'B4 typed request mismatch')
        require(record['sample_id']==req.sample_id and record['artifact_identity']==arm['artifact']['artifact_identity'] and record['arm_id']==arm['arm_id'],'B4 treatment/sample identity')
        if record.get('recovered_from_submission_journal'):
            require(record['evidence_kind']==('synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_task_generation'),'B4 synthetic/live mismatch')
            normalized.append(record);continue
        require(record['sample_identity']==next(s for s in plan['data']['rows'] if s['sample_id']==req.sample_id)['sample_identity'],'B4 authored identity changed')
        normalized.append(generation_from_raw(record,row,arm,plan,task_id=task.TASK,task_version=task.VERSION,max_new_tokens=64))
    return normalized


def evidence_mass(root,plan,progress):
    rows={r['sample_id']:r for r in plan['samples']['rows']};result=[]
    for arm in ('bf16','rtn'):
        job=progress['jobs'].get('main-'+arm)
        if not job or job['status']!='COMPLETE':continue
        _,mapping=load_job(root,job,plan)
        for key,value in mapping.items():
            if key[0]!='attention_row':continue
            row=rows[key[1]];span=row['reference_evidence_span'];c=value['record']['coordinates'];vals=value['values']
            available=span['status']=='AVAILABLE'
            tokens=span['overlapping_token_positions'] if available else None
            result.append({'arm_id':arm,'sample_id':key[1],'query_condition':'last_prompt' if c['query_position']==row['prompt_token_boundary']-1 else 'reference_prefix',
              'layer':c['layer'],'head':c['head'],'query_position':c['query_position'],'status':'MEASURED' if available else 'NOT_APPLICABLE',
              'mass':sum(vals[t] for t in tokens if c['allowed_keys'][t]) if available else None,'overlap_policy':row['source_span_policy'],
              'passage_id':span['passage_id'],'content_token_positions':tokens,'observation_id':value['record']['observation_id'],
              'meaning':'full causal row mass on tokens overlapping gold passage CONTENT; diagnostic, not causal explanation'})
    return result


def analyze(root,out):
    plan=read(root/'PLAN_FROZEN.json');frozen=dict(plan);identity=frozen.pop('plan_identity')
    require(digest(frozen)==identity and plan['task']=='BENCH05-B4','B4 plan identity')
    validate_data(plan['data'],plan['samples']);require(plan['scorer_identity']==sha(Path(task.__file__)),'B4 frozen scorer source changed')
    verified=verify_raw(root,scope='bench05b4_context_raw');progress=read(root/'progress.json');samples=plan['data']['rows'];arms={};by_raw={};by_score={};errors={}
    for arm in plan['arms']:
        name=arm['arm_id'];job=progress['jobs'].get('serve-'+name);records=decode_generations(root,job,plan,arm);scores,aggregate=task.aggregate(samples,records)
        by_raw[name]={r['sample_id']:r for r in records};by_score[name]={r['sample_id']:r for r in scores};splits={}
        for split in ('SMOKE','EVAL_LOCKED'):
            subset=[s for s in samples if s['split']==split];r=[g for g in records if g['split']==split];_,a=task.aggregate(subset,r)
            a['transport_complete']=all(g['transport_complete'] for g in r) and len(r)==len(subset)
            def latency(key):
                v=[g[key] for g in r if g.get(key) is not None]
                return {'n':len(v),'p50':statistics.median(v) if v else None,'max':max(v) if v else None}
            a['request_latency_http_seconds']=latency('request_latency_http');a['request_latency_task_seconds']=latency('request_latency_task');a['ttft_http_seconds']=latency('ttft_http')
            a['actual_prompt_tokens']=sum(g.get('usage',{}).get('prompt_tokens',0) for g in r if isinstance(g.get('usage'),dict))
            a['actual_output_tokens']=sum(g.get('usage',{}).get('completion_tokens',0) for g in r if isinstance(g.get('usage'),dict));splits[split]=a
        directory=root/safe_name(job['directory']) if job else None
        arms[name]={'artifact_identity':arm['artifact']['artifact_identity'],'serving_status':job['status'] if job else 'NOT_RUN','splits':splits,
          'memory':memory(directory) if directory else {'scope':'whole_device_mib','observed_max':None},'generation_label':'DEPLOYED_PATH_OBSERVATION'}
        eval_ids={s['sample_id'] for s in samples if s['split']=='EVAL_LOCKED'}
        errors[name]={'counts_can_overlap':True,'termination':dict(collections.Counter(s['termination'] for s in scores if s['status']=='SCORED')),
            **{k:sum(s.get(k) is True for s in scores if s['sample_id'] in eval_ids) for k in ('false_abstention','false_answer','answer_correct_evidence_wrong','evidence_correct_answer_wrong','values_correct_format_wrong')}}
        write_jsonl(out/(name+'-generations-replayed.jsonl'),records);write_jsonl(out/(name+'-scores.jsonl'),scores)
    evaluation=[s for s in samples if s['split']=='EVAL_LOCKED'];paired={'planned_documents':24,'metrics':{}};outcomes=[]
    for s in evaluation:
        sid=s['sample_id'];a,b=(by_raw[n].get(sid) for n in ('bf16','rtn'));complete=bool(a and b and a['transport_complete'] and b['transport_complete'])
        r={'sample_id':sid,'status':'COMPLETE' if complete else 'INCOMPLETE','language':s['language'],'case_type':s['case_type']}
        if complete:
            compatible(a['identity'],b['identity'])
            for metric in ('answer_exact','evidence_exact','task_success'):
                x,y=by_score['bf16'][sid][metric],by_score['rtn'][sid][metric]
                r[metric]={'bf16':x,'rtn':y,'outcome':'both_correct' if x and y else 'loss' if x else 'gain' if y else 'both_wrong'}
        outcomes.append(r)
    paired['complete_documents']=sum(r['status']=='COMPLETE' for r in outcomes);paired['incomplete_documents']=24-paired['complete_documents']
    paired['status']='COMPLETE_PAIRED24_DESCRIPTIVE' if paired['complete_documents']==24 else 'PARTIAL_OR_NOT_RUN'
    for metric in ('answer_exact','evidence_exact','task_success'):
        paired['metrics'][metric]={name:sum(r.get(metric,{}).get('outcome')==name for r in outcomes) for name in ('gain','loss','both_correct','both_wrong')}
    probe=probe_metrics(root,plan,progress);probe['observer_off_controls']='NOT_RUN_ON_B4_INPUTS';linked=[];flat=[]
    for row in plan['samples']['rows']:
        sid=row['sample_id'];entries=[r for r in probe['records'] if r['sample_id']==sid]
        def mean(kind,key):
            values=[r[key] for r in entries if r['kind']==kind and r.get(key) is not None];return statistics.fmean(values) if values else None
        item={'sample_id':sid,'task_id':task.TASK,'task_version':task.VERSION,'prompt_hash':row['prompt_hash'],'reference_hash':row['reference_hash'],
          'reference_source':row['reference_source'],'source_evidence_span':row['reference_evidence_span'],'free_generation':{n:{'score':by_score[n][sid],
             'raw':progress['jobs'].get('serve-'+n,{}).get('directory','NOT_RUN')+'/task-generations.jsonl','request_id':by_raw[n].get(sid,{}).get('request_id')} for n in ('bf16','rtn')},
          'target_positions':sum(r['kind']=='target_logprob' for r in entries),'attention_rows':sum(r['kind']=='attention_row' for r in entries),
          'reference_nll_nat':mean('target_logprob','reference_nll_nat'),'candidate_nll_nat':mean('target_logprob','candidate_nll_nat'),'delta_nll_nat':mean('target_logprob','delta_nll_nat'),
          'attention_js_nat':mean('attention_row','js_nat'),'attention_l1':mean('attention_row','l1'),'attention_output_nmse':mean('attention_output','nmse'),
          'attention_output_cosine':mean('attention_output','cosine'),'RTN_probe_mode':'DECODED_WEIGHT_DIAGNOSTIC',
          'observation_ids':[r['reference_observation'] for r in entries]+[r['candidate_observation'] for r in entries]}
        linked.append(item);flat.append({k:v for k,v in item.items() if not isinstance(v,(dict,list))})
    masses=evidence_mass(root,plan,progress)
    write_jsonl(out/'paired-outcomes.jsonl',outcomes);write_jsonl(out/'sample-linked.jsonl',linked);write_jsonl(out/'evidence-content-mass.jsonl',masses)
    csv_output(out/'sample-linked.csv',flat);write(out/'paired.json',paired);write(out/'error-analysis.json',errors);write(out/'probe-comparisons.json',probe)
    real=plan.get('evidence_kind')!='synthetic_contract' and any(by_raw.values())
    summary={'schema_version':1,'task':'BENCH05-B4','task_id':task.TASK,'task_version':task.VERSION,'scorer':task.SCORER,'scope':'SYNTHETIC_DEV_LOCKED',
      'evidence_kind':plan['evidence_kind'],'live_status':'CURRENT_NATIVE_MEASUREMENT' if real else 'NOT_RUN_OR_SYNTHETIC','arms':arms,'paired':paired,
      'probe':{'status':probe['status'],'paired_samples':probe['paired_samples'],'target_positions':sum(r['kind']=='target_logprob' for r in probe['records']),
               'attention_rows':sum(r['kind']=='attention_row' for r in probe['records']),'evidence_mass_rows':len(masses)},'raw_files_verified':verified,
      'budget':read(root/'budget-continuation.json') if (root/'budget-continuation.json').is_file() else None,'metric_specs':task.ContextEvidenceAdapter().metric_specs(),
      'stage_status':{'TASK_ADAPTER':'THREE_TASKS_TWO_REQUEST_KINDS_IMPLEMENTED','TASK_LIVE':'COMPLETE_PAIR24' if real and paired['complete_documents']==24 else 'PARTIAL_OR_BLOCKED',
        'PROBE_LIVE':probe['status'],'CPU_REPLAY':'RAW_HTTP_AND_NUMERIC_BLOBS_RECOMPUTED','PORTABILITY':'EXTERNAL_RECEIPT_REQUIRED','QUALITY':'DESCRIPTIVE_SYNTHETIC_DEV_ONLY'},
      'bridge':'NOT_RUN_ON_B4_INPUTS','observer_controls':'NOT_RUN_ON_B4_INPUTS','diagnostic_deployment_equivalence':'NOT_CLAIMED',
      'worker_native_exit':'UNKNOWN','CURRENT_REMOTE_CI':'NOT_RUN','retained_research':'HOLD_CLIPPING_VARIANT; prior quality INCONCLUSIVE; B1/B2 strict0/24 preserved',
      'PERFORMANCE':'C1_SINGLE_SERVER_DESCRIPTIVE','DEPLOYMENT_CUSTOMER':'NOT_VALIDATED','replay_gpu_actions':0}
    write(out/'summary.json',summary)
    lines=['# BENCH05-B4 CPU 재계산','','합성 문맥의 답·근거 검색이며 일반 RAG/공식 benchmark/배포 인증이 아닙니다.','','| Arm | submitted | format | answer | evidence | strict |','|---|---:|---:|---:|---:|---:|']
    for n,a in arms.items():
        q=a['splits']['EVAL_LOCKED'];m=q['metrics'];lines.append(f"| {n} | {q['observed']}/24 | {m['format_valid']['correct']}/24 | {m['answer_exact']['correct']}/24 | {m['evidence_exact']['correct']}/24 | {m['task_success']['correct']}/24 |")
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary


def replay(bundle,output):
    out=derived_output(bundle,output)
    with preserve_failure(out),open_bundle(bundle) as root:
        candidates=[root] if (root/'PLAN_FROZEN.json').is_file() else [p.parent for p in root.rglob('PLAN_FROZEN.json') if (p.parent/'RAW_MANIFEST.json').is_file() and read(p)['task']=='BENCH05-B4']
        require(len(candidates)==1,'Expected one B4 run root')
        return analyze(candidates[0],out)
