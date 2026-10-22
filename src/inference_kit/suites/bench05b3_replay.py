"""Recompute ARC scores and every continuation likelihood from saved HTTP; CPU only."""
import json
import statistics
from pathlib import Path
from ..core.io import read,write,require,digest,sha,safe_name,loads
from ..core.state import derived_output,preserve_failure
from ..core.evidence import open_bundle
from ..tasks.arc_challenge import ArcChallengeAdapter,decode_record,TASK,VERSION
from ..tasks.base import compatible
from .bench05b3_plan import validate_data
from .bench05b1_replay import probe_metrics,memory,jsonl
from .bench05a_replay import csv_output


def write_jsonl(path,rows):
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def verify_raw(root):
    manifest=read(root/'RAW_MANIFEST.json');require(manifest['scope']=='bench05b3_arc_raw','Wrong ARC raw scope')
    for name,entry in manifest['files'].items():
        p=root/safe_name(name);require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),'Unsafe/missing ARC raw')
        require(sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],'ARC raw hash/size mismatch: '+name)
    return len(manifest['files'])

def summarize(samples,documents,choices,records):
    by={s['sample_id']:s for s in samples};docs=[r for r in documents if r['sample_id'] in by];complete=[r for r in docs if r['status']=='COMPLETE']
    raw=[r for r in records if r['sample_id'] in by];submitted={r['sample_id'] for r in raw};goldtokens=[]
    for r in choices:
        if r['status']=='COMPLETE' and r['tokens'] and r['tokens'][0]['sample_id'] in by and r['choice_index']==by[r['tokens'][0]['sample_id']]['gold']:goldtokens.extend(r['tokens'])
    lat=[r['response_complete']-r['http_submit'] for r in raw if r['response_complete'] is not None]
    return {'planned_documents':len(samples),'submitted_documents':len(submitted),'complete_documents':len(complete),'failed_or_incomplete_documents':len(submitted)-len(complete),'not_run_documents':len(samples)-len(submitted),
      'complete_subset':len(complete)==len(samples),'acc_correct':sum(r['acc'] for r in complete),'acc_norm_correct':sum(r['acc_norm'] for r in complete),
      'observed_complete_acc':statistics.fmean(r['acc'] for r in complete) if complete else None,'observed_complete_acc_norm':statistics.fmean(r['acc_norm'] for r in complete) if complete else None,
      'planned_denominator_correct_lower_bound_acc_norm':sum(r['acc_norm'] for r in complete)/len(samples),'gold_mean_nll_document_macro_nat':statistics.fmean(r['gold_mean_nll_nat'] for r in complete) if complete else None,
      'gold_mean_nll_token_weighted_nat':statistics.fmean(r['nll_nat'] for r in goldtokens) if goldtokens else None,'gold_token_count':len(goldtokens),
      'submitted_posts':len(raw),'transport_complete_posts':sum(r['transport_complete'] for r in raw),'http_errors':sum(not r['transport_complete'] for r in raw),
      'likelihood_contract_failures':sum(r['status']!='COMPLETE' for r in choices if r['request_id'] in {q['request_id'] for q in raw}),
      'actual_input_tokens':sum(r['response'].get('usage',{}).get('prompt_tokens',0) for r in raw if r['transport_complete']),
      'actual_audit_generation_tokens':sum(r['response'].get('usage',{}).get('completion_tokens',0) for r in raw if r['transport_complete']),
      'choice_http_latency_seconds':{'n':len(lat),'p50':statistics.median(lat) if lat else None,'max':max(lat) if lat else None,'meaning':'non-stream full response, NOT TTFT; C1 one server'},
      'document_serial_elapsed_seconds':{'n':len(complete),'p50':statistics.median(r['document_serial_elapsed_seconds'] for r in complete) if complete else None,'max':max((r['document_serial_elapsed_seconds'] for r in complete),default=None),'meaning':'all sequential choice requests for one document, distinct from one natural generation'},
      'independent_unit':'original ARC document; choices/tokens not independent documents'}

def analyze(root,out):
    plan=read(root/'PLAN_FROZEN.json');frozen=dict(plan);identity=frozen.pop('plan_identity');require(digest(frozen)==identity,'B3 frozen plan integrity')
    validate_data(plan['data'],plan['samples']);verified=verify_raw(root);progress=read(root/'progress.json');adapter=ArcChallengeAdapter();samples=plan['data']['rows']
    arm_results={};all_choices={};all_docs={};all_tokens={};raw_by={}
    for arm in plan['arms']:
        name=arm['arm_id'];job=progress['jobs'].get('serve-'+name);records=[];directory=None
        if job:
            directory=root/safe_name(job['directory']);file=directory/'http-responses.jsonl'
            if file.exists():records=jsonl(file)
        execution=plan['plan_identity'][:16]+'|attempt1|'+name
        requests={r.request_id:r for s in samples for r in adapter.requests(s,execution,plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity'])}
        if directory and (directory/'request-journal.jsonl').is_file():
            events=jsonl(directory/'request-journal.jsonl');submits={};ends={}
            for event in events:
                require(event['request_id'] in requests,'Unknown journal request')
                req=requests[event['request_id']]
                from ..tasks.arc_challenge import body
                require(event['request_body']==body(req) and event['identity']==req.identity and event['request_body_identity']==digest(event['request_body']),'Journal request identity mismatch')
                target=submits if event['event']=='http_submit' else ends if event['event']=='response_end' else None
                require(target is not None and event['request_id'] not in target,'Unknown/duplicate journal event');target[event['request_id']]=event
            require(set(ends)<=set(submits) and {r['request_id'] for r in records}<=set(submits),'Journal/raw submission coverage mismatch')
            for request_id,start in submits.items():
                if any(r['request_id']==request_id for r in records):continue
                end=ends.get(request_id,{})
                records.append(start|{'request_kind':'choice_loglikelihood','http_status':None,'raw_response_text':None,'response':None,
                  'transport_complete':False,'response_complete':end.get('response_complete'),'error':end.get('error') or {'type':'IncompleteRaw','message':'Client submission journal exists; completed response absent'},
                  'evidence_kind':'synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_native_likelihood','recovered_from_submission_journal':True})
        require(len({r['request_id'] for r in records})==len(records),'Duplicate ARC request ID')
        choices=[];by_doc={s['sample_id']:[] for s in samples}
        for raw in records:
            require(raw['request_id'] in requests and raw['artifact_identity']==arm['artifact']['artifact_identity'] and raw['arm_id']==name,'ARC execution/treatment mismatch')
            require(raw['evidence_kind']==('synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_native_likelihood'),'Synthetic/historical raw cannot be promoted to current live')
            req=requests[raw['request_id']];require(raw['sample_id']==req.sample_id and raw['choice_index']==req.payload['choice_index'],'Raw sample/choice mismatch')
            value=decode_record(req,raw);choices.append(value);by_doc[req.sample_id].append(value)
        docs=[adapter.score(s,by_doc[s['sample_id']]) for s in samples];tokens=[t for c in choices for t in c['tokens']]
        for part in ('SMOKE','REGRESSION'):
            subset=[s for s in samples if s['usage_partition']==part]
            arm_results.setdefault(name,{})[part]=summarize(subset,docs,choices,records)
        arm_results[name].update(artifact_identity=arm['artifact']['artifact_identity'],serving_status=job['status'] if job else 'NOT_RUN',
                                memory=memory(directory) if directory else {'scope':'whole_device_mib','observed_max':None})
        write_jsonl(out/(name+'-choice-scores.jsonl'),choices);write_jsonl(out/(name+'-document-scores.jsonl'),docs);write_jsonl(out/(name+'-native-tokens.jsonl'),tokens)
        all_choices[name]=choices;all_docs[name]={r['sample_id']:r for r in docs};all_tokens[name]={(t['sample_id'],t['choice_index'],t['token_position']):t for t in tokens};raw_by[name]={r['request_id']:r for r in records}
    regression=[s for s in samples if s['usage_partition']=='REGRESSION'];deltas=[];outcomes=[]
    for s in regression:
        sid=s['sample_id'];a,b=(all_docs[k][sid] for k in ('bf16','rtn'))
        outcome={'sample_id':sid,'status':'COMPLETE' if a['status']==b['status']=='COMPLETE' else 'INCOMPLETE'}
        if outcome['status']=='COMPLETE':
            for metric in ('acc','acc_norm'):outcome.update({metric+'_bf16':a[metric],metric+'_rtn':b[metric],metric+'_lost':bool(a[metric] and not b[metric]),metric+'_gained':bool(not a[metric] and b[metric])})
            outcome['choice_argmax_agreement_acc']=a['prediction_acc']==b['prediction_acc'];outcome['choice_argmax_agreement_acc_norm']=a['prediction_acc_norm']==b['prediction_acc_norm']
        outcomes.append(outcome)
        for choice in range(len(s['choices'])):
            left=[r for r in raw_by['bf16'].values() if r['sample_id']==sid and r['choice_index']==choice]
            right=[r for r in raw_by['rtn'].values() if r['sample_id']==sid and r['choice_index']==choice]
            if left and right:compatible(left[0]['identity'],right[0]['identity'])
        for key,x in all_tokens['bf16'].items():
            if key[0]!=sid or key not in all_tokens['rtn']:continue
            y=all_tokens['rtn'][key];require(all(x[k]==y[k] for k in ('token_id','prefix_hash','token_id_space','logit_position')),'Native token/prefix axis mismatch')
            deltas.append({'sample_id':sid,'choice_index':key[1],'gold_choice':key[1]==s['gold'],'token_position':key[2],'logit_position':x['logit_position'],'token_id':x['token_id'],'prefix_hash':x['prefix_hash'],
                'bf16_nll_nat':x['nll_nat'],'rtn_nll_nat':y['nll_nat'],'delta_nll_nat':y['nll_nat']-x['nll_nat'],'bf16_request':x['request_id'],'rtn_request':y['request_id']})
    paired={'status':'COMPLETE_REGRESSION64' if all(r['status']=='COMPLETE' for r in outcomes) else 'PARTIAL','paired_documents':sum(r['status']=='COMPLETE' for r in outcomes),'planned_documents':64}
    for metric in ('acc','acc_norm'):
        paired[metric]={'lost':sum(r.get(metric+'_lost',False) for r in outcomes),'gained':sum(r.get(metric+'_gained',False) for r in outcomes),
            'correct_retained':sum(r.get(metric+'_bf16',False) and r.get(metric+'_rtn',False) for r in outcomes),
            'choice_argmax_agreement':sum(r.get('choice_argmax_agreement_'+metric,False) for r in outcomes)}
    paired['native_token_positions']=len(deltas);paired['delta_nll_token_weighted_nat']=statistics.fmean(r['delta_nll_nat'] for r in deltas) if deltas else None
    paired['delta_nll_document_macro_nat']=statistics.fmean(statistics.fmean(t['delta_nll_nat'] for t in deltas if t['sample_id']==s['sample_id']) for s in regression if any(t['sample_id']==s['sample_id'] for t in deltas)) if deltas else None
    probe=probe_metrics(root,plan,progress);probe['observer_off_controls']='NOT_RUN_ON_ARC_INPUTS; historical A only';bridges=[];linked=[]
    for row in plan['samples']['rows']:
        sid=row['sample_id'];gold=row['gold_choice_index'];entries=[r for r in probe['records'] if r['sample_id']==sid];token_entries=[r for r in entries if r['kind']=='target_logprob']
        for entry in token_entries:
            c=entry['coordinates'];position=c['token_position']
            require(c['prefix_hash']==digest({'token_ids':row['input_ids'][:position],'token_id_space':plan['samples']['token_id_space']}),'Native/diagnostic prefix identity mismatch')
            for name,field in (('bf16','reference_nll_nat'),('rtn','candidate_nll_nat')):
                native=all_tokens[name].get((sid,gold,position));require(native and native['token_id']==c['target_token_id'],'Native/diagnostic gold token missing')
                bridges.append({'sample_id':sid,'arm_id':name,'choice_index':gold,'token_position':position,'native_nll_nat':native['nll_nat'],'diagnostic_nll_nat':entry[field],
                    'absolute_delta_nll_nat':abs(native['nll_nat']-entry[field]),'native_request_id':native['request_id'],'diagnostic_observation_id':entry['reference_observation' if name=='bf16' else 'candidate_observation']})
        mean=lambda kind,k:statistics.fmean(r[k] for r in entries if r['kind']==kind and r.get(k) is not None) if any(r['kind']==kind and r.get(k) is not None for r in entries) else None
        linked.append({'sample_id':sid,'task_id':TASK,'reference_source':'dataset_answer_key_choice','gold_choice':gold,'prompt_hash':row['prompt_hash'],'reference_hash':row['reference_hash'],
          'native_documents':{a:all_docs[a][sid] for a in ('bf16','rtn')},'probe_target_positions':len(token_entries),'attention_rows':sum(r['kind']=='attention_row' for r in entries),
          'probe_delta_nll_nat':mean('target_logprob','delta_nll_nat'),'attention_js_nat':mean('attention_row','js_nat'),'attention_output_nmse':mean('attention_output','nmse'),
          'attention_output_cosine':mean('attention_output','cosine'),'probe_label_rtn':'DECODED_WEIGHT_DIAGNOSTIC','raw_observation_links':{k:progress['jobs'].get('main-'+k,{}).get('directory') for k in ('bf16','rtn')},
          'native_raw_links':{k:(progress['jobs']['serve-'+k]['directory']+'/http-responses.jsonl') if 'serve-'+k in progress['jobs'] and (root/progress['jobs']['serve-'+k]['directory']/'http-responses.jsonl').is_file() else None for k in ('bf16','rtn')}})
    bridge={'status':'BRIDGE_MEASURED_NO_EQUIVALENCE_CLAIM' if bridges else 'NOT_RUN','records':bridges,'observer_control':'NOT_RUN_ON_ARC_INPUTS'}
    for name in ('bf16','rtn'):
        vals=[r['absolute_delta_nll_nat'] for r in bridges if r['arm_id']==name];bridge[name]={'coverage':len(vals),'mean_abs_delta_nll_nat':statistics.fmean(vals) if vals else None,'max_abs_delta_nll_nat':max(vals) if vals else None}
    write_jsonl(out/'native-token-deltas.jsonl',deltas);write_jsonl(out/'paired-outcomes.jsonl',outcomes);write_jsonl(out/'sample-linked.jsonl',linked)
    write(out/'paired.json',paired);write(out/'probe-comparisons.json',probe);write(out/'bridge.json',bridge);csv_output(out/'sample-linked.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in linked])
    summary={'schema_version':1,'task':'BENCH05-B3','task_id':TASK,'task_version':VERSION,'request_kind':'choice_loglikelihood','usage_scope':'REGRESSION_REUSE_NOT_NEW_TEST','evidence_kind':plan['evidence_kind'],'live_status':'NOT_LIVE_SYNTHETIC_CONTRACT' if plan.get('evidence_kind')=='synthetic_contract' else 'CURRENT_NATIVE_MEASUREMENT' if any(raw_by.values()) else 'NOT_RUN_OR_BLOCKED_NO_NATIVE_REQUESTS','arms':arm_results,'paired':paired,
      'probe':{'status':probe['status'],'paired_samples':probe['paired_samples'],'target_positions':sum(r['kind']=='target_logprob' for r in probe['records']),'attention_rows':sum(r['kind']=='attention_row' for r in probe['records'])},
      'bridge':{k:v for k,v in bridge.items() if k!='records'},'raw_files_verified':verified,'budget':read(root/'budget-continuation.json') if (root/'budget-continuation.json').exists() else None,
      'metric_specs':adapter.metric_specs(),'replay_gpu_actions':0,'REMOTE_CI':'NOT_RUN','PERFORMANCE':'C1_SINGLE_SERVER_DESCRIPTIVE_NON_STREAM_NO_TTFT','DEPLOYMENT_CUSTOMER':'NOT_VALIDATED',
      'retained_research':'HOLD_CLIPPING_VARIANT; prior ARC quality INCONCLUSIVE','native_full_vocabulary':'NOT_COLLECTED','worker_native_exit':'UNKNOWN','diagnostic_memory':{a:memory(root/safe_name(progress['jobs']['main-'+a]['directory'])) if 'main-'+a in progress['jobs'] else {'status':'NOT_RUN'} for a in ('bf16','rtn')}}
    summary['stage_status']={'TASK_ADAPTER_ENGINEERING':'TWO_TYPED_REQUEST_KINDS_IMPLEMENTED','ARC_NATIVE_LIKELIHOOD':'COMPLETE_REGRESSION64_PAIR' if paired['status']=='COMPLETE_REGRESSION64' and summary['live_status']=='CURRENT_NATIVE_MEASUREMENT' else 'PARTIAL_OR_NOT_LIVE',
      'TOKEN_DIAGNOSTICS':'NATIVE_CONTINUATION_TOKENS_MEASURED' if deltas and summary['live_status']=='CURRENT_NATIVE_MEASUREMENT' else 'PARTIAL_OR_SYNTHETIC','ATTENTION_LINK':probe['status'],'PORTABILITY_REPLAY':'CPU_RECOMPUTED_INSTALLED_ORIGIN_REQUIRES_EXTERNAL_RECEIPT','MODEL_QUALITY':'DESCRIPTIVE_REGRESSION_REUSE','DEPLOYMENT_CUSTOMER':'NOT_VALIDATED'}
    write(out/'summary.json',summary)
    text=['# BENCH05-B3 ARC CPU 재계산','','기존 test 자료의 고정 REGRESSION64 재사용이며 새 TEST·전체 benchmark·배포 인증이 아닙니다.','','| Arm | complete | acc | acc_norm | gold NLL macro | POST | device MiB |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in arm_results.items():
        q=r['REGRESSION'];a=str(q['acc_correct'])+'/64' if q['complete_subset'] else 'NOT_RUN/PARTIAL';n=str(q['acc_norm_correct'])+'/64' if q['complete_subset'] else 'NOT_RUN/PARTIAL';text.append(f"| {name} | {q['complete_documents']}/64 | {a} | {n} | {q['gold_mean_nll_document_macro_nat']} | {q['submitted_posts']} | {r['memory']['observed_max']} |")
    text.extend(['','RTN serving은 실제 packed 경로이며 내부 attention은 decoded BF16 진단입니다. Attention/bridge 차이는 정답 변화의 인과 근거나 경로 동등성 인증이 아닙니다.'])
    (out/'REPORT_KO.md').write_text('\n'.join(text)+'\n',encoding='utf-8');return summary

def replay(bundle,output):
    out=derived_output(bundle,output)
    with preserve_failure(out),open_bundle(bundle) as root:
        candidates=[root] if (root/'PLAN_FROZEN.json').is_file() else [p.parent for p in root.rglob('PLAN_FROZEN.json') if (p.parent/'RAW_MANIFEST.json').is_file() and read(p)['task']=='BENCH05-B3']
        require(len(candidates)==1,'Expected one explicit B3 run root')
        return analyze(candidates[0],out)
