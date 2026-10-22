"""Pure CPU B1 raw replay and sample-linked views. No model/runtime imports."""
import json
import math
from pathlib import Path
import statistics
from ..core.io import read,write,require,digest,sha,safe_name,loads
from ..core.state import derived_output,preserve_failure
from ..core.evidence import open_bundle
from ..probes.contracts import MetricSpec
from ..tasks.tech_record import TASK,VERSION,aggregate,score
from ..adapters.protocol import Adapter
from ..adapters.http import sha as wire_hash
from ..metrics.token import nll,distribution_metrics,log_softmax
from ..metrics.attention import row_metrics,vector_metrics
from .bench05a_replay import load_job,matching,csv_output
from .bench05b1_plan import validate_prepared

SPECS=[MetricSpec('request_latency_http','b1-v1',('generation_http',),('sample','profile'),
 'response_complete-http_submit','seconds','lower','observed requests; p50/max and n'),
 MetricSpec('request_latency_task','b1-v1',('generation_http',),('sample','profile'),
 'response_complete-queued','seconds','lower','observed requests; includes client queue'),
 MetricSpec('ttft_http','b1-v1',('generation_http',),('sample','profile'),
 'first_valid_content-http_submit','seconds','lower','observed nonempty requests'),
 MetricSpec('task_success_within_10s','b1-v1',('generation_http','cpu_task_score'),('task','sample','scorer'),
 'transport & schema & three fields & two evidence correct & request_latency_http<=10','fraction','higher','24 authored DEV documents; demo threshold, no SLA')]


def jsonl(path):return [loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_raw(root,scope='bench05b1_raw'):
    manifest=read(root/'RAW_MANIFEST.json');require(manifest['scope']==scope,'Wrong raw scope')
    for name,value in manifest['files'].items():
        p=root/safe_name(name);require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'Missing/unsafe B1 raw member')
        require(p.stat().st_size==value['bytes'] and sha(p)==value['sha256'],'B1 raw identity differs: '+name)
    return len(manifest['files'])


def generation_from_raw(record,prepared,arm,plan,*,task_id=TASK,task_version=VERSION,max_new_tokens=192):
    expected_kind='synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_task_generation'
    require(record.get('evidence_kind')==expected_kind,'Synthetic/live task generation mismatch')
    require(record['task_id']==task_id and record['task_version']==task_version and record['sample_id']==prepared['sample_id'],'Generation task/sample mismatch')
    require(record['artifact_identity']==arm['artifact']['artifact_identity'] and record['tokenizer_identity']==plan['samples']['tokenizer_identity'],'Generation artifact/tokenizer mismatch')
    require(record['prompt_hash']==prepared['prompt_hash'] and not record['reference_used_for_generation'],'Generation prompt/reference leak')
    body=record['request_body'];expected=Adapter('vllm').body({'token_ids':prepared['generation_input_ids']},'serving-lab',max_new_tokens,'natural_generation_v1')
    require(body==expected and record['request_identity']==wire_hash(body),'Request differs from frozen natural generation contract')
    text=[];ids=[];usage=None;finish=None;done=False;prompt_ids=None
    for frame in record['raw_sse_frames']:
        if frame.strip()=='[DONE]':require(not done,'Duplicate DONE');done=True;continue
        require(not done,'Data after DONE');event=Adapter.event(loads(frame));text.append(event['text']);ids.extend(event['ids'] or [])
        if event['prompt_ids'] is not None:
            require(prompt_ids is None or prompt_ids==event['prompt_ids'],'Prompt IDs changed within raw SSE')
            prompt_ids=event['prompt_ids']
        if event['usage'] is not None:usage=event['usage']
        if event['finish'] is not None:finish=event['finish']
    require(''.join(text)==record['output_text'],'Raw SSE/text mismatch')
    if record['transport_complete']:
        require(done and finish in ('stop','length') and usage is not None and usage==record['usage'],'Completed generation missing raw termination/usage')
        require(type(usage['completion_tokens']) is int and 0<=usage['completion_tokens']<=max_new_tokens and len(ids)==usage['completion_tokens'],'Natural token count mismatch')
        require(usage['prompt_tokens']==len(prepared['generation_input_ids']),'Generation prompt length differs')
        require(prompt_ids==prepared['generation_input_ids'],'Raw echoed prompt IDs differ')
    result=dict(record)
    submit=record['http_submit_monotonic'];completed=record['completed_monotonic'];queued=record['queued_monotonic'];first=record['first_content_monotonic']
    result['request_latency_http']=completed-submit if submit is not None else None
    result['request_latency_task']=completed-queued;result['ttft_http']=first-submit if first is not None and submit is not None else None
    for name in ('request_latency_http','request_latency_task','ttft_http'):
        a,b=record[name],result[name];require(a==b if a is None or b is None else math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-9),'Timing boundary changed: '+name)
    return result


def probe_metrics(root,plan,progress):
    loaded={};executions={};records=[]
    for arm in ('bf16','rtn'):
        job=progress['jobs'].get('main-'+arm)
        if job and job['status']=='COMPLETE':executions[arm],loaded[arm]=load_job(root,job,plan)
    if set(loaded)!={'bf16','rtn'}:return {'status':'NOT_RUN_OR_PARTIAL','paired_samples':0,'records':[],'executions':executions}
    a,b=loaded['bf16'],loaded['rtn'];require(a.keys()==b.keys(),'Probe observation coverage differs')
    for key,left in a.items():
        right=b[key];matching(left,right);require(key[2]=='ON','B1 does not include OFF repetitions')
        x,y=left['values'],right['values'];require(x is not None and y is not None,'Missing probe numeric evidence')
        value={'sample_id':key[1],'kind':key[0],'coordinates':left['record']['coordinates'],
               'reference_observation':left['record']['observation_id'],'candidate_observation':right['record']['observation_id'],
               'reference_raw':left['raw_directory']+'/observations.jsonl','candidate_raw':right['raw_directory']+'/observations.jsonl'}
        if key[0]=='target_logprob':value.update(reference_nll_nat=nll(x[0]),candidate_nll_nat=nll(y[0]),delta_nll_nat=nll(y[0])-nll(x[0]))
        elif key[0]=='attention_row':value.update(row_metrics(x,y,value['coordinates']['allowed_keys']))
        elif key[0] in ('attention_output','output_projection','reconstructed_AV'):value.update(vector_metrics(x,y))
        elif key[0]=='full_logits':
            require(len(x)==len(y)==plan['samples']['vocabulary_size'] and left['record']['details']['full_vocabulary'] and right['record']['details']['full_vocabulary'],'Full vocabulary axis mismatch')
            target_key=('target_logprob',*key[1:]);target=value['coordinates']['target_token_id']
            for mapping,logits in ((a,x),(b,y)):
                require(abs(log_softmax(logits)[target]-mapping[target_key]['values'][0])<=2e-5,'Saved full logits/target likelihood mismatch (frozen 2e-5 tolerance)')
            value.update(distribution_metrics(x,y))
        records.append(value)
    require(sum(r['kind']=='target_logprob' for r in records)==plan['counts']['probe_target_positions_per_arm'],'Probe target coverage differs')
    require(sum(r['kind']=='attention_row' for r in records)==plan['counts']['probe_attention_rows_per_arm'],'Probe row coverage differs')
    kinds={e['evidence_kind'] for e in executions.values()};require(len(kinds)==1,'Mixed synthetic/live probe executions')
    synthetic=kinds=={'synthetic_contract'}
    require(synthetic==(plan.get('evidence_kind')=='synthetic_contract'),'Synthetic/live probe plan mismatch')
    return {'status':'SYNTHETIC_CONTRACT_ONLY' if synthetic else 'MEASURED_SAME_TASK_DIAGNOSTIC_ONLY','paired_samples':4,'records':records,'executions':executions,
            'rtn_mode':'DECODED_WEIGHT_DIAGNOSTIC','deployment_equivalence':'NOT_CLAIMED','observer_off_controls':'NOT_RUN_IN_B1'}


def memory(directory):
    path=directory/'memory.jsonl';values=[]
    if path.exists():
        for r in jsonl(path):
            if r['returncode']==0:
                fields=r.get('values','').split(',')
                if len(fields)>3:values.append(float(fields[3].strip()))
    return {'scope':'whole_device_mib','observed_max':max(values) if values else None,'samples':len(values),'sampling_seconds':1,'phases_file':'phases.jsonl' if (directory/'phases.jsonl').exists() else None,
            'meaning':'Diagnostic and serving executions separated; not process/allocator memory or production performance'}


def analyze(root,out):
    plan=read(root/'PLAN_FROZEN.json');frozen=dict(plan);expected=frozen.pop('plan_identity');require(digest(frozen)==expected,'B1 plan integrity changed')
    validate_prepared(plan['samples'],plan['data']);verified=verify_raw(root);progress=read(root/'progress.json')
    data=plan['data']['rows'];prepared={r['sample_id']:r for r in plan['samples']['generation_rows']};scores_by={};raw_by={};arms={}
    for arm in plan['arms']:
        name=arm['arm_id'];job=progress['jobs'].get('serve-'+name);raw=[];directory=None
        if job:
            directory=root/safe_name(job['directory']);path=directory/'generations.jsonl'
            if path.exists():raw=[generation_from_raw(r,prepared[r['sample_id']],arm,plan) for r in jsonl(path)]
        require(len({r['sample_id'] for r in raw})==len(raw),'Duplicate generated sample ID')
        raw_by[name]={r['sample_id']:r for r in raw};scores_by[name]={};splits={}
        for split in ('SMOKE','DEV_LOCKED'):
            chosen=[s for s in data if s['split']==split];observed=[r for r in raw if r['split']==split]
            scores,summary=aggregate(chosen,observed);scores_by[name].update({r['sample_id']:r for r in scores});splits[split]=summary
            with (out/(name+'-'+split+'-scores.jsonl')).open('x',encoding='utf-8') as f:
                for r in scores:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        lat=[r['request_latency_http'] for r in raw if r['split']=='DEV_LOCKED' and r['request_latency_http'] is not None]
        arms[name]={'artifact_identity':arm['artifact']['artifact_identity'],'splits':splits,'execution_status':job['status'] if job else 'NOT_RUN',
                    'dev_transport_errors':sum(not r['transport_complete'] for r in raw if r['split']=='DEV_LOCKED'),
                    'latency_http_seconds':{'n':len(lat),'p50':statistics.median(lat) if lat else None,'max':max(lat) if lat else None},
                    'memory':memory(directory) if directory else {'scope':'whole_device_mib','observed_max':None,'samples':0}}
    probe=probe_metrics(root,plan,progress);linked=[];flat=[]
    mean=lambda rows,key:statistics.fmean(r[key] for r in rows if r.get(key) is not None) if any(r.get(key) is not None for r in rows) else None
    for sample in [s for s in data if s['split']=='DEV_LOCKED']:
        sid=sample['sample_id'];values=[r for r in probe['records'] if r['sample_id']==sid]
        token=[r for r in values if r['kind']=='target_logprob'];attention=[r for r in values if r['kind']=='attention_row'];outputs=[r for r in values if r['kind']=='attention_output']
        diagnostics={'status':'MEASURED_DIAGNOSTIC_ONLY' if token else 'NOT_SELECTED' if sid not in plan['samples']['probe_selection']['sample_ids'] else 'NOT_RUN',
          'target_positions':len(token),'attention_rows':len(attention),'reference_nll_nat':mean(token,'reference_nll_nat'),'candidate_nll_nat':mean(token,'candidate_nll_nat'),'delta_nll_nat':mean(token,'delta_nll_nat'),
          'attention_js_nat':mean(attention,'js_nat'),'attention_output_nmse':mean(outputs,'nmse'),'attention_output_cosine':mean(outputs,'cosine'),
          'reference_hash':prepared[sid]['reference_hash'],'prompt_hash':prepared[sid]['prompt_hash'],'tokenizer_identity':plan['samples']['tokenizer_identity'],
          'reference_source':'task_author_gold_json','execution_links':{k:{'execution_id':v['execution_id'],'artifact_identity':v['artifact_identity'],
              'observations':progress['jobs']['main-'+k]['directory']+'/'+v['raw_path'],'comparison':'probe-comparisons.json'} for k,v in probe['executions'].items()},
          'observation_ids':[r['reference_observation'] for r in values]+[r['candidate_observation'] for r in values]}
        linked.append({'sample_id':sid,'sample_identity':sample['sample_identity'],'task_id':TASK,'task_version':VERSION,'case_type':sample['case_type'],'language':sample['language'],
           'free_generation':{arm:{'score':scores_by[arm][sid],'request_id':raw_by[arm].get(sid,{}).get('request_id'),
             'raw':progress['jobs'].get('serve-'+arm,{}).get('directory','NOT_RUN')+'/generations.jsonl','artifact_identity':arms[arm]['artifact_identity']} for arm in ('bf16','rtn')},'teacher_forced_probe':diagnostics})
        flat.append({'sample_id':sid,'bf16_task_success':scores_by['bf16'][sid]['task_success'],'rtn_task_success':scores_by['rtn'][sid]['task_success'],**{k:v for k,v in diagnostics.items() if k in ('status','target_positions','attention_rows','reference_nll_nat','candidate_nll_nat','delta_nll_nat','attention_js_nat','attention_output_nmse','attention_output_cosine')}})
    paired_complete=all(arms[a]['splits']['DEV_LOCKED']['complete'] for a in ('bf16','rtn'))
    paired={'status':'COMPLETE_24_DOCUMENT_DESCRIPTIVE' if paired_complete else 'PARTIAL_OR_NOT_RUN','planned_documents':24,
            'paired_observed_documents':sum(all(scores_by[a][r['sample_id']]['status']=='SCORED' for a in ('bf16','rtn')) for r in data if r['split']=='DEV_LOCKED')}
    if paired_complete:
        left=[scores_by['bf16'][r['sample_id']]['task_success'] for r in data if r['split']=='DEV_LOCKED'];right=[scores_by['rtn'][r['sample_id']]['task_success'] for r in data if r['split']=='DEV_LOCKED']
        paired.update(correct_retained=sum(a and b for a,b in zip(left,right)),correct_lost=sum(a and not b for a,b in zip(left,right)),correct_gained=sum(not a and b for a,b in zip(left,right)),incorrect_retained=sum(not a and not b for a,b in zip(left,right)))
    with (out/'sample-linked.jsonl').open('x',encoding='utf-8') as f:
        for row in linked:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    csv_output(out/'sample-linked.csv',flat);write(out/'probe-comparisons.json',probe);write(out/'paired.json',paired)
    summary={'schema_version':1,'task':'BENCH05-B1','task_id':TASK,'task_version':VERSION,'scorer':plan['scorer'],'scope':'synthetic_dev_pilot','evidence_kind':plan.get('evidence_kind','UNKNOWN'),
      'arms':arms,'paired':paired,'probe':{'status':probe['status'],'paired_samples':probe['paired_samples'],'token_positions':sum(r['kind']=='target_logprob' for r in probe['records']),'attention_rows':sum(r['kind']=='attention_row' for r in probe['records'])},
      'raw_files_verified':verified,'metric_specs':[s.record() for s in SPECS],'run_process_status':progress['status'],
      'budget':read(root/'budget-continuation.json') if (root/'budget-continuation.json').exists() else None,
      'CURRENT_REMOTE_CI':'NOT_RUN','historical_base_ci':{'run_id':38830709923,'head':'4758fb34843cd8bb692cd859ec3b7adbbb2e079e','status':'success'},
      'PERFORMANCE_CLAIM':'DESCRIPTIVE_C1_SINGLE_ROUND_ONLY','DEPLOYMENT_CUSTOMER':'NOT_VALIDATED',
      'retained_research':'HOLD_CLIPPING_VARIANT; prior quality INCONCLUSIVE','diagnostic_deployment_equivalence':'NOT_CLAIMED; BENCH05-A bridge is historical provenance only',
      'replay_gpu_actions':0}
    write(out/'summary.json',summary)
    lines=['# BENCH05-B1 CPU 재계산','','합성 업무 DEV24의 관측값이며 일반 품질·고객 SLA·배포 동등성 인증이 아닙니다.','','| Arm | submitted DEV | JSON | schema | task success | ≤10s |','|---|---:|---:|---:|---:|---:|']
    for name,result in arms.items():
        q=result['splits']['DEV_LOCKED'];m=q['metrics'];lines.append(f"| {name} | {q['submitted']}/24 | {m['json_valid']['correct']} | {m['schema_valid']['correct']} | {m['task_success']['correct']} | {m['task_success_within_10s']['correct']} |")
    lines+=['','RTN 서비스는 실제 packed artifact, 내부 진단은 DECODED_WEIGHT_DIAGNOSTIC입니다. 서로 다른 실행의 같은 문항을 연결하며 인과 효과를 주장하지 않습니다.']
    (out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary


def replay(bundle,output):
    out=derived_output(bundle,output)
    with preserve_failure(out),open_bundle(bundle) as root:
        if not (root/'RAW_MANIFEST.json').exists():
            matches=[p.parent for p in root.rglob('RAW_MANIFEST.json') if read(p).get('scope')=='bench05b1_raw'];require(len(matches)==1,'Ambiguous/missing B1 evidence root');root=matches[0]
        return analyze(root,out)
