"""Read-only adapters from verified B2/B3/B4 raw into common token records.

Only target scalars are decoded here. Full logits are transferred through the
explicit sink; unrelated attention arrays are never loaded into Python lists.
"""
import json
import math
from pathlib import Path

from ..core.io import read, require, digest, sha, safe_name, loads
from ..probes.blobs import load
from ..probes.contracts import Observation

ADAPTERS={'bench05b2':('BENCH05-B2-CONTRACT','bench05b2_contract_raw'),
          'bench05b3':('BENCH05-B3','bench05b3_arc_raw'),
          'bench05b4':('BENCH05-B4','bench05b4_context_raw')}


def _lines(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield loads(line)


def _plan(path):
    value=read(path);body=dict(value);identity=body.pop('plan_identity')
    require(digest(body)==identity,'Source frozen plan identity mismatch')
    return value


def _verify(root,expected_scope):
    manifest=read(root/'RAW_MANIFEST.json')
    require(manifest['scope']==expected_scope,'Source raw manifest scope mismatch')
    names=set()
    for name,entry in manifest['files'].items():
        safe=safe_name(name);require(safe.casefold() not in names,'Duplicate source raw member');names.add(safe.casefold())
        path=root/safe
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),'Unsafe/missing source member')
        require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'Source raw size/hash mismatch: '+name)
    require('PLAN_FROZEN.json' in manifest['files'] and 'progress.json' in manifest['files'],'Source manifest missing plan/progress')
    return manifest


def _selected(spec,sample):
    if 'sample_ids' in spec and sample['sample_id'] not in spec['sample_ids']:return False
    split=sample.get('usage_partition',sample.get('split'))
    return 'splits' not in spec or split in spec['splits']


def _sequence(source_id,row,policy_id,choice_index):
    ids=row['input_ids'];require(ids and all(type(t) is int and t>=0 for t in ids),'Invalid source sequence token IDs')
    identifier=digest({'source_id':source_id,'policy_id':policy_id,'sample_id':row['sample_id'],
                       'choice_index':choice_index,'input_ids':ids})
    return {'sequence_id':identifier,'source_id':source_id,'sample_id':row['sample_id'],'policy_id':policy_id,
            'policy_identity':row.get('policy_identity'),'reference_source':row.get('reference_source'),
            'reference_role':row.get('reference_role','GOLD' if row.get('gold_choice_index') is not None else 'AUTHOR_GOLD_REFERENCE'),
            'choice_index':choice_index,'input_ids':ids,'offsets_codepoints':row.get('offsets_codepoints'),
            'offsets_utf8_bytes':row.get('offsets_utf8_bytes'),'field_spans':row.get('value_spans',{}),
            'offset_mapping_status':'AVAILABLE' if row.get('offsets_codepoints') is not None and row.get('offsets_utf8_bytes') is not None else 'NOT_COLLECTED',
            'field_spans_coordinate_space':'reference_codepoints','selection_policy':row.get('selection_policy'),
            'source_tokenization_policy':row.get('tokenization'),'reference_evidence_span':row.get('reference_evidence_span'),
            'passage_content_spans':row.get('passage_content_spans'),
            'prompt_token_boundary':row.get('prompt_token_boundary'),
            'special_token_ids':row.get('special_token_ids'),'special_token_mask':row.get('special_token_mask'),
            'loss_mask':row.get('loss_mask'),'prompt':row.get('prompt'),'reference_text':row.get('reference_text'),
            'full_text':row.get('full_text'),'source_sequence_identity':row.get('sequence_identity',digest(ids))}


def _merge_sequence(sequences,value):
    old=sequences.get(value['sequence_id'])
    if old is None:sequences[value['sequence_id']]=value;return
    require(old['input_ids']==value['input_ids'],'Common sequence ID collision')
    for key,item in value.items():
        if key=='offset_mapping_status':continue
        if item is None or item=={}:continue
        if old.get(key) is None or old.get(key)=={}:old[key]=item
        else:require(old[key]==item,'Common sequence metadata mismatch: '+key)
    old['offset_mapping_status']='AVAILABLE' if old.get('offsets_codepoints') is not None and old.get('offsets_utf8_bytes') is not None else 'NOT_COLLECTED'


def _base(spec,root,plan,arm,row,sequence,*,policy_id,choice_index,reference_scope,acquisition,execution):
    return {'source_id':spec['source_id'],'bundle_identity':sha(root/'RAW_MANIFEST.json'),
       'plan_identity':plan['plan_identity'],'execution_id':execution,'arm_id':arm['arm_id'],
       'task_id':plan['task_id'],'task_version':plan['task_version'],'dataset_identity':digest(plan['data']),
       'split':row.get('usage_partition',row.get('split')),'original_split':row.get('split'),
       'sample_id':row['sample_id'],'policy_id':policy_id,'choice_index':choice_index,'reference_scope':reference_scope,
       'policy_identity':row.get('policy_identity'),'reference_source':row.get('reference_source'),
       'prompt_hash':row.get('prompt_hash',digest(row.get('prompt',''))),'reference_hash':row.get('reference_hash'),
       'sequence_id':sequence['sequence_id'],'tokenizer_identity':plan['samples']['tokenizer_identity'],
       'token_id_space':plan['samples']['token_id_space'],'vocabulary_size':plan['samples']['vocabulary_size'],
       'acquisition_source':acquisition,'runtime_identity':plan['serving_runtime_identity'] if acquisition=='native_likelihood' else plan['runtime_identity'],
       'backend_policy':'native' if acquisition=='native_likelihood' else 'diagnostic',
       'artifact_identity':arm['artifact']['artifact_identity'],'recipe_identity':arm['artifact']['recipe_identity'],
       'model_id':plan['pair']['model_id'],'model_revision':plan['pair']['revision'],
       'probability_stage':'raw_pre_sampling','metric_contract':'target_logprob_v1','language':row.get('language'),
       'field_tags':[],'requested':True}


def _coordinate(row,coordinates,plan):
    ids=row['input_ids'];position=coordinates['token_position']
    require(type(position) is int and 0<position<len(ids),'Target position outside source sequence')
    require(coordinates['logit_position']==position-1 and coordinates['target_token_id']==ids[position],'Source next-token shift/ID mismatch')
    require(coordinates['prefix_hash']==digest({'token_ids':ids[:position],'token_id_space':plan['samples']['token_id_space']}),'Source prefix hash mismatch')
    return {key:coordinates[key] for key in ('token_position','logit_position','target_token_id','prefix_hash')}


def _diagnostic(spec,root,plan,job_root,job_plan,progress,policy_id,blob_sink,sequences,tokens,coverage):
    rows={r['sample_id']:r for r in job_plan['samples']['rows']}
    authored={r['sample_id']:r for r in plan['data']['rows']}
    for arm in job_plan['arms']:
        name=arm['arm_id'];job=progress.get('jobs',{}).get('main-'+name)
        expected=[row for row in rows.values() if _selected(spec,authored[row['sample_id']])]
        execution=None;directory=None;observed={};full={}
        if job and job['status']=='COMPLETE':
            directory=job_root/safe_name(job['directory']);execution=read(directory/'execution.json');result=read(directory/'result.json')
            require(execution['status']==result['status']=='MEASURED' and execution['phase']=='main','Incomplete diagnostic source')
            require(execution['artifact_identity']==arm['artifact']['artifact_identity'] and execution['recipe_identity']==arm['artifact']['recipe_identity'],'Diagnostic treatment identity mismatch')
            require(execution['model_id']==plan['pair']['model_id'] and execution['model_revision']==plan['pair']['revision'],'Diagnostic model identity mismatch')
            require(execution['tokenizer_identity']==plan['samples']['tokenizer_identity'] and execution['runtime_identity']==plan['runtime_identity'],'Diagnostic tokenizer/runtime mismatch')
            require(execution['execution_mode']==arm['execution_mode'],'Diagnostic path label changed')
            for observation in _lines(directory/'observations.jsonl'):
                if observation['kind'] not in ('target_logprob','full_logits'):continue
                Observation(**observation).validate();sid=observation['sample_id'];require(sid in rows,'Unknown diagnostic sample')
                row=rows[sid];c=observation['coordinates']
                require(observation['execution_id']==execution['execution_id'] and observation['evidence_kind']==execution['evidence_kind'],'Diagnostic observation/execution mismatch')
                require(c['pass_id']=='ON','Only frozen ON diagnostic records accepted')
                require(c['token_id_space']==plan['samples']['token_id_space'] and c['tokenizer_identity']==plan['samples']['tokenizer_identity'] and c['sequence_identity']==row['sequence_identity'],'Diagnostic token-space/sequence mismatch')
                _coordinate(row,c,plan);require(c['token_position'] in {t['token_position'] for t in row['targets']},'Diagnostic target outside frozen selection')
                key=(sid,c['token_position']);mapping=observed if observation['kind']=='target_logprob' else full
                require(key not in mapping,'Duplicate diagnostic observation coordinate')
                if observation['kind']=='full_logits':
                    require(c['token_position'] in row['full_vocab_target_positions'],'Selected full-vocabulary source required')
                    if observation['status']=='MEASURED':
                        require(observation['details'].get('full_vocabulary') is True,'Measured full-vocabulary source claim required')
                        require(observation['numeric']['shape']==[plan['samples']['vocabulary_size']],'Full-vocabulary blob axis mismatch')
                mapping[key]=observation
        for row in expected:
            coverage['full_vocab_requested']+=len(row['full_vocab_target_positions'])
            sample=authored[row['sample_id']];choice=row.get('gold_choice_index');sequence=_sequence(spec['source_id'],row,policy_id,choice);_merge_sequence(sequences,sequence)
            enriched=row|{'usage_partition':sample.get('usage_partition',sample.get('split')),'split':sample.get('split',row.get('split'))}
            mode=execution['execution_mode'] if execution else arm['execution_mode']
            acquisition='decoded_diagnostic' if mode=='DECODED_WEIGHT_DIAGNOSTIC' else 'instrumented_diagnostic'
            for target in row['targets']:
                coverage['diagnostic_requested']+=1;key=(row['sample_id'],target['token_position']);observation=observed.get(key)
                position=target['token_position'];coords=target|{'prefix_hash':digest({'token_ids':row['input_ids'][:position],'token_id_space':plan['samples']['token_id_space']})}
                value=_base(spec,root,plan,arm,enriched,sequence,policy_id=policy_id,choice_index=choice,
                    reference_scope='GOLD' if choice is not None else 'AUTHOR_GOLD_REFERENCE',acquisition=acquisition,execution=execution['execution_id'] if execution else None)
                value.update(_coordinate(row,coords,plan));value.update(observation_id=observation['observation_id'] if observation else None,
                    target_logprob=None,raw_argmax_token_id=None,full_logits=None,status='NOT_RUN' if observation is None else observation['status'],
                    evidence_kind=execution['evidence_kind'] if execution else 'NOT_RUN',field_tags=['continuation'])
                for field,selection in row.get('field_token_coverage',{}).items():
                    if position in selection['overlapping_positions']:value['field_tags'].append(field)
                if choice is not None:value['field_tags'].append('gold_choice')
                if observation and observation['status']=='MEASURED':
                    numeric=load(directory,observation['numeric']);require(len(numeric)==1 and math.isfinite(numeric[0]) and numeric[0]<=0,'Invalid target logprob scalar')
                    value.update(target_logprob=numeric[0],raw_argmax_token_id=observation['details'].get('raw_argmax_token_id'),
                        normalization_dtype=observation['details'].get('normalization_dtype'),raw_logits_dtype=observation['details'].get('raw_logits_dtype'),
                        source_observation=observation,source_observations_path=str((directory/'observations.jsonl').relative_to(root)),
                        source_numeric_sha256=observation['numeric']['sha256'],effective_weight_dtype=execution['effective_weight_dtype'],
                        backend=execution['backend'],execution_mode=mode)
                    coverage['diagnostic_measured']+=1
                    full_observation=full.get(key)
                    if full_observation and full_observation['status']=='MEASURED':
                        value['full_logits']=blob_sink(directory,full_observation['numeric'])
                        value['source_full_observation']=full_observation;coverage['full_vocab_measured']+=1
                    tolerance=job_plan['suite'].get('observer',{}).get('raw_logit_to_logprob_abs_tolerance')
                    if tolerance is not None:value['logit_logprob_tolerance']=tolerance
                value['record_id']=digest({**{k:value[k] for k in ('source_id','bundle_identity','plan_identity','execution_id','observation_id','arm_id','sample_id','policy_id','choice_index','acquisition_source')},
                    'coordinates':{k:value[k] for k in ('token_position','logit_position','target_token_id','prefix_hash')}})
                tokens.append(value)


def _native(spec,root,plan,progress,sequences,tokens,coverage):
    from ..tasks.arc_challenge import ArcChallengeAdapter,decode_record,body
    adapter=ArcChallengeAdapter()
    for arm in plan['arms']:
        name=arm['arm_id'];execution=plan['plan_identity'][:16]+'|attempt1|'+name
        requests={r.request_id:r for sample in plan['data']['rows'] for r in adapter.requests(sample,execution,plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity'])}
        job=progress.get('jobs',{}).get('serve-'+name);raw={};submits={};ends={}
        if job:
            directory=root/safe_name(job['directory']);path=directory/'http-responses.jsonl'
            for record in _lines(path) if path.is_file() else []:
                require(record['request_id'] in requests and record['request_id'] not in raw,'Unknown/duplicate native likelihood request');raw[record['request_id']]=record
            journal=directory/'request-journal.jsonl'
            for event in _lines(journal) if journal.is_file() else []:
                require(event['request_id'] in requests,'Unknown native journal request');request=requests[event['request_id']]
                require(event['identity']==request.identity and event['request_body']==body(request) and event['request_body_identity']==digest(body(request)),'Native journal identity mismatch')
                target=submits if event['event']=='http_submit' else ends if event['event']=='response_end' else None
                require(target is not None and event['request_id'] not in target,'Duplicate/unknown native journal event');target[event['request_id']]=event
            if journal.is_file():require(set(ends)<=set(submits) and set(raw)<=set(submits),'Native journal/raw coverage mismatch')
        for sample in plan['data']['rows']:
            if not _selected(spec,sample):continue
            for request in adapter.requests(sample,execution,plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity']):
                payload=request.payload;choice=payload['choice_index'];ids=payload['input_ids'];start=len(payload['context_ids']);record=raw.get(request.request_id)
                decoded=None
                if record:
                    require(record['arm_id']==name and record['artifact_identity']==arm['artifact']['artifact_identity'],'Native artifact/arm identity mismatch')
                    require(record['sample_id']==sample['sample_id'] and record['choice_index']==choice,'Native sample/choice identity mismatch')
                    require(record['evidence_kind']==('synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_native_likelihood'),'Historical/synthetic native source mismatch')
                    decoded=decode_record(request,record)
                effective_prompt=payload['context'].rstrip()
                effective_reference=payload['context'][len(effective_prompt):]+payload['continuation']
                row={'sample_id':sample['sample_id'],'input_ids':ids,'prompt_token_boundary':start,
                     'prompt':effective_prompt,'reference_text':effective_reference,'prompt_hash':digest(effective_prompt),
                     'reference_hash':digest(effective_reference),'full_text':effective_prompt+effective_reference,
                     'reference_source':'dataset_answer_key_choice' if choice==sample['gold'] else 'dataset_choice_continuation',
                     'reference_role':'GOLD' if choice==sample['gold'] else 'NON_GOLD',
                     'usage_partition':sample['usage_partition'],'split':sample['split']}
                sequence=_sequence(spec['source_id'],row,None,choice);_merge_sequence(sequences,sequence)
                observed={t['token_position']:t for t in decoded['tokens']} if decoded and decoded['status']=='COMPLETE' else {}
                for position in range(start,len(ids)):
                    coverage['native_requested']+=1;target=observed.get(position)
                    value=_base(spec,root,plan,arm,row,sequence,policy_id=None,choice_index=choice,
                        reference_scope='GOLD' if choice==sample['gold'] else 'NON_GOLD',acquisition='native_likelihood',execution=execution)
                    value.update(token_position=position,logit_position=position-1,target_token_id=ids[position],
                         prefix_hash=digest({'token_ids':ids[:position],'token_id_space':plan['samples']['token_id_space']}),
                         observation_id=None,request_id=request.request_id,target_logprob=target['logprob'] if target else None,
                         raw_argmax_token_id=None,full_logits=None,status='MEASURED' if target else 'FAILED' if record or request.request_id in submits else 'NOT_RUN',
                         field_tags=['continuation','gold_choice' if choice==sample['gold'] else 'non_gold_choice'],
                         evidence_kind=record['evidence_kind'] if record else 'NOT_RUN',source_request_identity=request.identity,
                         source_request_path=str((directory/'http-responses.jsonl').relative_to(root)) if job else None,
                         source_response_identity=digest(record['response']) if record and record.get('response') is not None else None,
                         normalization_dtype='SERVER_NOT_EXPOSED',effective_weight_dtype='bfloat16' if name=='bf16' else 'packed_int4',
                         native_full_vocabulary='NOT_COLLECTED',native_argmax_status='NOT_COLLECTED_ACTUAL_ARGMAX',
                         offset_mapping_status='NOT_COLLECTED_NATIVE_SOURCE',audit_generated_token_excluded=True)
                    if target:coverage['native_measured']+=1
                    _coordinate(row,value,plan)
                    value['record_id']=digest({**{k:value[k] for k in ('source_id','bundle_identity','plan_identity','execution_id','observation_id','request_id','arm_id','sample_id','policy_id','choice_index','acquisition_source')},
                        'coordinates':{k:value[k] for k in ('token_position','logit_position','target_token_id','prefix_hash')}})
                    tokens.append(value)


def collect_source(spec,blob_sink):
    """Collect verified historical raw; filters never bypass full manifest checks."""
    require(spec.get('adapter') in ADAPTERS and isinstance(spec.get('source_id'),str) and spec['source_id'],'Unsupported/missing token source adapter/ID')
    root=Path(spec['path']);require(root.is_absolute() and root.is_dir() and not root.is_symlink(),'Source must be explicit local run directory')
    if 'raw_manifest_sha256' in spec:require(sha(root/'RAW_MANIFEST.json')==spec['raw_manifest_sha256'],'Pinned source manifest identity mismatch')
    expected_task,scope=ADAPTERS[spec['adapter']];manifest=_verify(root,scope);plan=_plan(root/'PLAN_FROZEN.json');progress=read(root/'progress.json')
    require(plan['task']==expected_task,'Source adapter/task mismatch')
    coverage={'diagnostic_requested':0,'diagnostic_measured':0,'native_requested':0,'native_measured':0,'full_vocab_requested':0,'full_vocab_measured':0}
    sequences={};tokens=[];policies=spec.get('policies',['P00','P11'])
    if spec['adapter']=='bench05b2':
        require(isinstance(policies,list) and policies and len(set(policies))==len(policies) and all(p in ('P00','P11') for p in policies),'B2 token source only frozen P00/P11')
        for policy in policies:
            entry=progress.get('diagnostics',{}).get(policy)
            from ..suites.bench05b2_run import diagnostic_plan
            if entry:
                subroot=root/safe_name(entry['directory']);subplan=_plan(subroot/'PLAN_FROZEN.json')
                require(subplan==diagnostic_plan(plan,policy),'B2 diagnostic plan ancestry mismatch')
                subprogress=read(subroot/'progress.json')
            else:subroot=root;subplan=diagnostic_plan(plan,policy);subprogress={'jobs':{}}
            _diagnostic(spec,root,plan,subroot,subplan,subprogress,policy,blob_sink,sequences,tokens,coverage)
    else:
        require('policies' not in spec or spec['policies'] in ([],[None]),'Prompt policies unsupported for this source')
        if spec['adapter']=='bench05b3' and spec.get('include_native',True):_native(spec,root,plan,progress,sequences,tokens,coverage)
        _diagnostic(spec,root,plan,root,plan,progress,None,blob_sink,sequences,tokens,coverage)
    require(len({t['record_id'] for t in tokens})==len(tokens),'Duplicate common token record ID')
    require(all(t<plan['samples']['vocabulary_size'] for row in sequences.values() for t in row['input_ids']),'Source token outside declared vocabulary')
    source={'source_id':spec['source_id'],'adapter':spec['adapter'],'task_id':plan['task_id'],'task_version':plan['task_version'],
        'source_bundle_path':str(root),'raw_manifest_sha256':sha(root/'RAW_MANIFEST.json'),'bundle_identity':sha(root/'RAW_MANIFEST.json'),
        'raw_manifest_scope':scope,'verified_raw_members':len(manifest['files']),'plan_identity':plan['plan_identity'],
        'runtime_identity':plan['runtime_identity'],'serving_runtime_identity':plan.get('serving_runtime_identity'),
        'source_acquisition_code_identity':digest(plan['source_identity']),'source_acquisition_code_files':plan['source_identity'],
        'dataset_identity':digest(plan['data']),'evidence_kind':plan['evidence_kind'],'usage':'historical_raw_reanalysis_not_new_measurement',
        'filters':{k:spec[k] for k in ('policies','sample_ids','splits','include_native') if k in spec},'coverage':coverage,
        'backend_equivalence':'NOT_CLAIMED','GPU_actions':0,'tokenizer_loads':0}
    return {'source':source,'sequences':list(sequences.values()),'tokens':tokens}
