"""Recompute BENCH05-A from numeric blobs/raw HTTP. Standard library only."""
import csv
import json
import math
from pathlib import Path
import statistics
from ..core.io import read,write,require,digest,sha,safe_name,loads
from ..core.state import derived_output,preserve_failure
from ..core.evidence import open_bundle,verify_manifest
from ..probes.blobs import load
from ..probes.contracts import Observation,MetricResult
from ..metrics.token import nll,distribution_metrics,log_softmax,SPECS as TOKEN_SPECS
from ..metrics.attention import row_metrics,vector_metrics,SPECS as ATTENTION_SPECS
from .bench05a_plan import validate_samples


def verify_raw(root):
    m=read(root/'RAW_MANIFEST.json');require(m['schema_version']==1 and m['scope']=='bench05a_raw','Missing raw evidence manifest')
    for name,entry in m['files'].items():
        name=safe_name(name);p=root/name
        require(p.resolve().is_relative_to(root.resolve()) and not p.is_symlink() and p.is_file(),'Missing/unsafe raw reference')
        require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Raw manifest identity mismatch: '+name)
    return len(m['files'])


def load_job(root,job,plan):
    directory=root/safe_name(job['directory']);result=read(directory/'result.json');execution=read(directory/'execution.json')
    require(result['status']==execution['status']=='MEASURED','Failed/partial diagnostic is not measured')
    require(execution['phase']=='main' and execution['evidence_kind'] in ('live_model_diagnostic','synthetic_contract'),'Wrong phase/evidence kind')
    arm=next(a for a in plan['arms'] if a['arm_id']==execution['arm_id'])
    require(execution['artifact_identity']==arm['artifact']['artifact_identity'] and execution['model_revision']==plan['pair']['revision'],'Execution treatment identity mismatch')
    require(execution['recipe_identity']==arm['artifact']['recipe_identity'] and execution['model_id']==plan['pair']['model_id'],'Execution model/recipe mismatch')
    require(execution['tokenizer_identity']==plan['samples']['tokenizer_identity'] and execution['runtime_identity']==plan['runtime_identity'],'Execution tokenizer/runtime differs')
    require(execution['execution_mode']==arm['execution_mode'],'Execution label differs from frozen plan')
    rows={r['sample_id']:r for r in plan['samples']['rows']};mapping={}
    for line in (directory/'observations.jsonl').read_text(encoding='utf-8').splitlines():
        record=loads(line);Observation(**record).validate();c=record['coordinates'];row=rows[record['sample_id']]
        require(record['execution_id']==execution['execution_id'] and record['evidence_kind']==execution['evidence_kind'],'Observation execution/kind changed')
        require(c['token_id_space']==plan['samples']['token_id_space'] and c['tokenizer_identity']==plan['samples']['tokenizer_identity'] and c['sequence_identity']==row['sequence_identity'],'Observation tokenizer/sequence mismatch')
        prefix=c['token_position'] if 'token_position' in c else c['query_position']+1
        require(c['prefix_hash']==digest({'token_ids':row['input_ids'][:prefix],'token_id_space':c['token_id_space']}),'Observed prefix differs')
        if 'token_position' in c:
            require(c['logit_position']==c['token_position']-1 and c['target_token_id']==row['input_ids'][c['token_position']],'Token next-step shift or target identity mismatch')
            require(c['token_position'] in [p['token_position'] for p in row['targets']],'Token coordinate outside frozen selection')
            if record['kind']=='full_logits':require(c['token_position'] in row['full_vocab_target_positions'],'Full vocabulary position outside selection')
        else:
            expected=row['attention'];require(c['layer']==expected['layer'] and c['query_position'] in expected['queries'],'Attention layer/query outside frozen selection')
            if record['kind']!='output_projection':
                require(c['head'] in expected['heads'] and c['kv_head']==expected['query_to_kv_head'][str(c['head'])],'Head/KV mapping outside frozen selection')
                require(c['key_positions']==list(range(len(row['input_ids']))) and c['allowed_keys']==[k<=c['query_position'] for k in range(len(row['input_ids']))],'Full native causal key axis differs')
        require(c['pass_id'] in ('ON','OFF1','OFF2'),'Unknown observer pass')
        if c['pass_id']!='ON':require(row['sample_id'] in [r['sample_id'] for r in plan['samples']['rows'][:2]],'OFF control outside fixed samples')
        key=(record['kind'],record['sample_id'],c['pass_id'],c.get('token_position'),c.get('layer'),c.get('head'),c.get('query_position'))
        require(key not in mapping,'Duplicate observation coordinate')
        mapping[key]={'record':record,'values':load(directory,record['numeric']) if record['status']=='MEASURED' else None,'raw_directory':job['directory']}
    return execution,mapping


def matching(left,right):
    a,b=left['record'],right['record'];require(a['kind']==b['kind'] and a['sample_id']==b['sample_id'],'Observation kind/sample mismatch')
    require(a['evidence_kind']==b['evidence_kind'],'Synthetic/live comparison cannot be promoted')
    require(a['coordinates']==b['coordinates'],'Prefix/token/attention axes or mask mismatch')
    for name in ('scaling','normalization_dtype','probability_source','qk_point','mask_source','key_scope'):
        require(a['details'].get(name)==b['details'].get(name),'Observation computation policy mismatch: '+name)


def csv_output(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def observer_effects(mapping,plan):
    pairs=[]
    for key,left in mapping.items():
        if key[0] not in ('target_logprob','full_logits') or key[2]!='OFF1':continue
        for tag in ('OFF2','ON'):
            other=(key[0],key[1],tag,*key[3:]);require(other in mapping,'Observer OFF/ON coverage missing')
            right=mapping[other]
            require(left['values'] is not None and right['values'] is not None,'Missing observer control')
            delta=max(abs(a-b) for a,b in zip(left['values'],right['values']))
            pairs.append({'sample_id':key[1],'kind':key[0],'token_position':key[3],'comparison':tag+'-OFF1','max_abs_delta':delta,
               'raw_top1_match':left['record']['details'].get('raw_argmax_token_id')==right['record']['details'].get('raw_argmax_token_id') if key[0]=='target_logprob' else max(range(len(left['values'])),key=left['values'].__getitem__)==max(range(len(right['values'])),key=right['values'].__getitem__)})
    target=max((r['max_abs_delta'] for r in pairs if r['kind']=='target_logprob'),default=None)
    logits=max((r['max_abs_delta'] for r in pairs if r['kind']=='full_logits'),default=None)
    required={r['sample_id'] for r in plan['samples']['rows'][:plan['suite']['observer']['off_samples']]}
    require({r['sample_id'] for r in pairs}==required,'Observer sample coverage differs')
    passed=target is not None and logits is not None and target<=plan['suite']['observer']['max_target_logprob_abs_delta'] and logits<=plan['suite']['observer']['max_selected_logit_abs_delta'] and all(r['raw_top1_match'] for r in pairs)
    return {'status':'INSUFFICIENT_OBSERVER_CONTROL' if target is None or logits is None else 'NO_CHANGE_WITHIN_FROZEN_TOLERANCE' if passed else 'OBSERVER_IMPACT_OBSERVED','max_target_logprob_abs_delta':target,
      'max_selected_logit_abs_delta':logits,'evidence_pairs':pairs,'backend_off_on':'same SDPA interface/original call; no output_attentions backend switch',
      'scope':'Same model/process small controls only; not cross-backend equivalence'}


def analyze(root,output):
    plan=read(root/'PLAN_FROZEN.json');frozen=dict(plan);identity=frozen.pop('plan_identity');require(digest(frozen)==identity,'Frozen plan changed')
    rows=validate_samples(plan['samples']);verified=verify_raw(root);progress=read(root/'progress.json');loaded={};executions={}
    for arm in ('bf16','rtn'):
        job=progress['jobs'].get('main-'+arm)
        if job and job['status']=='COMPLETE':executions[arm],loaded[arm]=load_job(root,job,plan)
    tokens=[];attention=[];outputs=[];full=[];reconstruction=[];observer={};projections=[];bridge=[]
    for arm,mapping in loaded.items():
        observer[arm]=observer_effects(mapping,plan)
        for key,entry in mapping.items():
            if key[0]!='attention_output' or key[2]!='ON':continue
            other=('reconstructed_AV',*key[1:]);require(other in mapping,'Native/reconstructed AV coverage missing')
            reconstruction.append({'arm_id':arm,'sample_id':key[1],'layer':key[4],'head':key[5],'query_position':key[6],**vector_metrics(entry['values'],mapping[other]['values'])})
    if set(loaded)=={'bf16','rtn'}:
        left,right=loaded['bf16'],loaded['rtn'];require(left.keys()==right.keys(),'Pair observation coverage differs')
        for key,a in left.items():
            b=right[key];matching(a,b)
            if key[2]!='ON':continue
            require(a['values'] is not None and b['values'] is not None,'Unmeasured pair coordinate cannot become a numeric success')
            c=a['record']['coordinates'];e={'sample_id':key[1],'reference_observation':a['record']['observation_id'],'candidate_observation':b['record']['observation_id'],
              'reference_raw':a['raw_directory']+'/observations.jsonl','candidate_raw':b['raw_directory']+'/observations.jsonl'}
            if key[0]=='target_logprob':
                av,bv=a['values'][0],b['values'][0];arg_a=a['record']['details'].get('raw_argmax_token_id');arg_b=b['record']['details'].get('raw_argmax_token_id')
                tokens.append(e|{'token_position':c['token_position'],'logit_position':c['logit_position'],'target_token_id':c['target_token_id'],
                    'reference_logprob':av,'candidate_logprob':bv,'reference_nll':nll(av),'candidate_nll':nll(bv),'delta_nll_candidate_minus_reference':nll(bv)-nll(av),
                    'reference_raw_top1':arg_a,'candidate_raw_top1':arg_b,'raw_top1_match':None if arg_a is None or arg_b is None else arg_a==arg_b,'prefix_hash':c['prefix_hash']})
            elif key[0]=='full_logits':
                require(a['record']['details']['full_vocabulary'] is True and b['record']['details']['full_vocabulary'] is True,'Top-k cannot be full distribution')
                require(len(a['values'])==len(b['values'])==plan['samples']['vocabulary_size'],'Full vocabulary axis differs')
                consistency=[]
                for arm,mapping,entry in (('bf16',left,a),('rtn',right,b)):
                    p=log_softmax(entry['values']);target=('target_logprob',key[1],'ON',key[3],None,None,None);expected=mapping[target]['values'][0]
                    error=abs(p[c['target_token_id']]-expected);require(error<=plan['suite']['observer']['raw_logit_to_logprob_abs_tolerance'],'Saved full logits and target logprob disagree')
                    require(max(range(len(p)),key=p.__getitem__)==mapping[target]['record']['details']['raw_argmax_token_id'],'Saved raw argmax differs')
                    consistency.append(error)
                full.append(e|{'token_position':c['token_position'],**distribution_metrics(a['values'],b['values']),
                               'max_saved_target_vs_cpu_logsoftmax_abs_error':max(consistency)})
            elif key[0]=='attention_row':attention.append(e|{'layer':c['layer'],'head':c['head'],'kv_head':c['kv_head'],'query_position':c['query_position'],**row_metrics(a['values'],b['values'],c['allowed_keys'])})
            elif key[0]=='attention_output':outputs.append(e|{'layer':c['layer'],'head':c['head'],'query_position':c['query_position'],**vector_metrics(a['values'],b['values'])})
            elif key[0]=='output_projection':projections.append(e|{'layer':c['layer'],'query_position':c['query_position'],**vector_metrics(a['values'],b['values'])})
        require(len(tokens)==plan['counts']['main_reference_positions_per_arm'] and len(attention)==plan['counts']['main_attention_rows_per_arm'],'Requested/main coordinate coverage differs')
    # Serving bridge, if present: re-parse echo token logprobs, not a stored summary.
    for arm,mapping in loaded.items():
        job=progress['jobs'].get('bridge-'+arm)
        if not job or job['status']!='COMPLETE':continue
        directory=root/safe_name(job['directory'])
        for line in (directory/'http-responses.jsonl').read_text(encoding='utf-8').splitlines():
            record=loads(line);sample=next(r for r in rows if r['sample_id']==record['sample_id']);response=record['response']
            require(record['http_status']==200 and record['request']['prompt']==sample['input_ids'],'Native bridge HTTP/token pairing differs')
            lp=response['choices'][0]['logprobs'];require(response['usage']['prompt_tokens']==len(sample['input_ids']) and lp['tokens'][:len(sample['input_ids'])]==['token_id:'+str(i) for i in sample['input_ids']],'Native bridge token-ID echo mismatch')
            require(len(lp['token_logprobs'])==len(sample['input_ids'])+1,'Native prompt likelihood missing')
            for pos in sample['targets']:
                key=('target_logprob',sample['sample_id'],'ON',pos['token_position'],None,None,None)
                native=lp['token_logprobs'][pos['token_position']];require(native is not None and math.isfinite(native),'Missing native target logprob')
                diag=mapping[key]['values'][0];tops=lp.get('top_logprobs');top=tops[pos['token_position']] if isinstance(tops,list) and len(tops)>pos['token_position'] else None
                raw_top=None
                if isinstance(top,dict) and top:
                    winner=max(top,key=top.__getitem__)
                    if winner.startswith('token_id:'):raw_top=int(winner.split(':')[1])
                bridge.append({'arm_id':arm,'sample_id':sample['sample_id'],'token_position':pos['token_position'],'native_logprob':native,'diagnostic_logprob':diag,
                  'nll_diagnostic_minus_native':-diag+native,'native_raw_top1':raw_top,'diagnostic_raw_top1':mapping[key]['record']['details'].get('raw_argmax_token_id'),
                  'raw_top1_match':None if raw_top is None else raw_top==mapping[key]['record']['details'].get('raw_argmax_token_id'),
                  'status':'BRIDGE_MEASURED_NO_EQUIVALENCE_CLAIM'})
    counts={'requested_samples_per_arm':4,'measured_main_arms':list(loaded),'requested_token_positions_per_arm':plan['counts']['main_reference_positions_per_arm'],
            'paired_token_positions':len(tokens),'paired_attention_rows':len(attention),'paired_native_attention_output_vectors':len(outputs),'paired_projection_vectors':len(projections),
            'paired_full_vocab_positions':len(full),'native_bridge_positions':len(bridge)}
    mean=lambda rows,key:statistics.fmean(r[key] for r in rows if r[key] is not None) if any(r[key] is not None for r in rows) else None
    numeric={'mean_reference_nll_nat':mean(tokens,'reference_nll'),'mean_candidate_nll_nat':mean(tokens,'candidate_nll'),
      'mean_delta_nll_nat':mean(tokens,'delta_nll_candidate_minus_reference'),'raw_top1_match_fraction':mean(tokens,'raw_top1_match'),
      'mean_attention_l1':mean(attention,'l1'),'mean_attention_js_nat':mean(attention,'js_nat'),'mean_attention_output_nmse':mean(outputs,'nmse'),
      'mean_attention_output_cosine':mean(outputs,'cosine'),'mean_selected_full_kl_nat':mean(full,'kl_ref_candidate_nat'),'mean_selected_full_js_nat':mean(full,'js_nat')}
    complete=set(loaded)=={'bf16','rtn'} and len(tokens)==plan['counts']['main_reference_positions_per_arm'] and len(attention)==plan['counts']['main_attention_rows_per_arm']
    evidence_kind=executions['bf16']['evidence_kind'] if complete else None
    summary={'schema_version':1,'task':'BENCH05-A','evidence_kind':evidence_kind,'stages':{'engineering':'CPU_REPLAY_IMPLEMENTED',
      'live_token':'MEASURED_DIAGNOSTIC_ONLY' if complete and evidence_kind=='live_model_diagnostic' else 'SYNTHETIC_CONTRACT_ONLY' if complete else 'BLOCKED_OR_INCOMPLETE',
      'live_attention':'MEASURED_SELECTED_NATIVE_QK_AND_OUTPUT' if complete and evidence_kind=='live_model_diagnostic' else 'SYNTHETIC_CONTRACT_ONLY' if complete else 'BLOCKED_OR_INCOMPLETE',
      'bridge':'BRIDGE_MEASURED_NO_EQUIVALENCE_CLAIM' if bridge else progress.get('bridge_status','NOT_RUN'),
      'research_production':'NOT_EVALUATED'},'counts':counts,'metrics':numeric,'executions':executions,'observer':observer,
      'raw_files_verified':verified,'scope':'Selected synthetic-text positions, not benchmark accuracy/corpus perplexity/throughput or layer causality',
      'retained_research_status':'HOLD_CLIPPING_VARIANT; prior quality INCONCLUSIVE unchanged','replay_gpu_actions':0}
    comparisons={'metric_specs':[s.record() for s in TOKEN_SPECS+ATTENTION_SPECS],'token':tokens,'attention':attention,'attention_output':outputs,
                 'output_projection':projections,'selected_full_vocab':full,'native_AV_vs_reconstructed_AV':reconstruction,'native_bridge':bridge,'observer':observer}
    comparisons['metric_results']=[MetricResult(name,value,'MEASURED' if value is not None else 'NOT_RUN',[],
       len(tokens) if 'nll' in name or 'top1' in name else len(attention) if 'attention' in name else len(full),
       plan['counts']['main_reference_positions_per_arm'] if 'nll' in name or 'top1' in name else plan['counts']['main_attention_rows_per_arm'] if 'attention' in name else plan['counts'].get('main_full_vocab_rows_per_arm',len(full)),
       'paired selected coordinates; not independent documents','BENCH05-A diagnostic positions only').record() for name,value in numeric.items()]
    write(output/'comparisons.json',comparisons);write(output/'summary.json',summary)
    for name,records in (('token',tokens),('attention',attention),('attention-output',outputs),('full-vocabulary',full),('native-bridge',bridge)):csv_output(output/(name+'.csv'),records)
    lines=['# BENCH05-A CPU 재계산','',json.dumps(counts,ensure_ascii=False),'','| Metric | Value |','|---|---:|']+[f'| {k} | {v} |' for k,v in numeric.items()]
    lines+=['','국소/최종 정답률·언어 능력·production 성능을 평가하지 않습니다. 원시 좌표/경로는 comparisons.json을 따릅니다.']
    (output/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary


def replay(run,output):
    out=derived_output(run,output)
    with preserve_failure(out),open_bundle(run) as root:
        verify_manifest(root);return analyze(root,out)
