"""Task-neutral CPU adapter for configured token_probe_raw_v1 acquisitions."""
import math
from pathlib import Path

from ..core.io import read,require,digest,sha,safe_name
from ..probes.blobs import load
from ..probes.contracts import Observation
from ..metrics.token import log_softmax
from .token_sources import _verify,_plan,_lines,_coordinate,_sequence,_merge_sequence


FIELD_MEMBERSHIP_POLICY = {
    'id':'configured_flat_field_membership_v2',
    'namespace':'token_probe_raw_v1.field_tags',
    'identity':'exact field key within source/task/dataset/sample/reference; case and explicit key namespace are preserved',
    'order':'continuation first; literal field keys lexicographically; choice role last unless already present',
    'duplicates':'same exact legacy flat field key contributes once; all contributing origins are retained',
    'unsupported':'structured namespace/role metadata cannot be projected into the legacy flat key',
    'denominator':'physical token records unchanged; overlapping distinct fields have independent denominators',
}


def _field_membership(row,position,choice):
    """Project the legacy flat field contract, preserving each membership origin.

    A literal named ``gold_choice`` and its choice-role annotation describe the
    same legacy membership.  They are two origins, not two token observations.
    Explicitly qualified field keys remain distinct; structured namespaces need
    a new contract and are refused rather than collapsed by their display name.
    """
    candidates=[('continuation',{'kind':'reference_continuation','reference_role':row['reference_role']})]
    fields=row.get('field_token_coverage',{})
    require(isinstance(fields,dict),'Configured field coverage must be a mapping')
    for field in fields:
        require(isinstance(field,str) and field and field not in ('__all__','continuation'),'Invalid/reserved configured field key')
    for field in sorted(fields):
        info=fields[field]
        require(isinstance(info,dict),'Configured field coverage must be an object')
        require(not any(key in info for key in ('namespace','field_namespace','role','reference_role','field_key','label')),
                'Structured field namespace/role unsupported by legacy flat field membership')
        positions=info.get('overlapping_positions')
        require(isinstance(positions,list) and all(type(t) is int and 0<t<len(row['input_ids']) for t in positions),
                'Invalid configured field overlap positions')
        occurrences=positions.count(position)
        for occurrence in range(occurrences):
            candidates.append((field,{'kind':'field_token_coverage','field_key':field,
                'position_occurrence':occurrence,'coverage':info,'value_span':row.get('value_spans',{}).get(field)}))
    if choice is not None:
        field='gold_choice' if row['reference_role'] in ('GOLD','GOLD_CHOICE','dataset_gold_choice') else 'non_gold_choice'
        candidates.append((field,{'kind':'choice_reference_role','reference_role':row['reference_role'],'choice_index':choice}))
    memberships={}
    for field,origin in candidates:
        memberships.setdefault(field,[]).append(origin)
    tags=list(memberships)
    return tags,[{'field_key':field,'namespace':FIELD_MEMBERSHIP_POLICY['namespace'],
                 'origins':origins} for field,origins in memberships.items()],len(candidates)


def _contract(plan):
    spec=plan['probe_spec'];normalization=spec['normalization'];policy=spec['consistency_policy']
    require(normalization=={'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'},'Unsupported configured token normalization')
    require(policy['id']=='same_pass_saved_f32_v1','Unsupported same-pass consistency policy')
    tolerance=policy['absolute_tolerance_nat']
    require(type(tolerance) in (int,float) and math.isfinite(tolerance) and tolerance>=0,'Invalid frozen consistency tolerance')
    require(isinstance(policy.get('scope'),str) and policy['scope'],'Missing consistency scope')
    return normalization,policy


def _token_coordinate(row,coordinates,plan):
    require(all(type(coordinates.get(key)) is int for key in ('token_position','logit_position','target_token_id')),'Configured token coordinates must be integer indices, not bool')
    return _coordinate(row,coordinates,plan)


def _row_identity(row,plan):
    for key in ('task_id','task_version','dataset_identity','sample_id','split','reference_source','reference_role'):
        require(isinstance(row.get(key),str) and row[key],'Missing generic probe row identity: '+key)
    ids=row['input_ids'];vocabulary=plan['samples']['vocabulary_size']
    require(type(vocabulary) is int and vocabulary>1 and ids and all(type(i) is int and 0<=i<vocabulary for i in ids),'Invalid configured token-ID space')
    require(row['sequence_identity']==digest(ids),'Configured sequence identity mismatch')
    require(row['task_version'] and row.get('tokenizer_identity',plan['samples']['tokenizer_identity'])==plan['samples']['tokenizer_identity'],'Configured tokenizer identity mismatch')
    targets=row['targets'];require(targets and len({t['token_position'] for t in targets})==len(targets),'Empty/duplicate configured targets')
    for target in targets:
        position=target['token_position'];_token_coordinate(row,target|{'prefix_hash':digest({'token_ids':ids[:position],'token_id_space':plan['samples']['token_id_space']})},plan)
    full=row['full_vocab_target_positions'];require(isinstance(full,list) and len(set(full))==len(full) and set(full)<={t['token_position'] for t in targets},'Configured full-logit selection differs')
    for key in ('offsets_codepoints','offsets_utf8_bytes'):
        offsets=row.get(key)
        require(offsets is None or isinstance(offsets,list) and len(offsets)==len(ids),'Configured offset axis mismatch')
    for key in ('loss_mask','attention_mask','special_token_mask'):
        values=row.get(key);require(values is None or len(values)==len(ids),'Configured mask axis mismatch')


def _selected(spec,row):
    return ('task_ids' not in spec or row['task_id'] in spec['task_ids']) and ('sample_ids' not in spec or row['sample_id'] in spec['sample_ids']) and ('splits' not in spec or row['split'] in spec['splits'])


def collect_source(spec,blob_sink):
    """Normalize configured raw without a task registry, model, or tokenizer."""
    require(spec.get('adapter')=='token_probe_raw_v1' and isinstance(spec.get('source_id'),str) and spec['source_id'],'Unsupported configured token source')
    root=Path(spec['path']);require(root.is_absolute() and root.is_dir() and not root.is_symlink(),'Explicit local configured raw required')
    if 'raw_manifest_sha256' in spec:require(sha(root/'RAW_MANIFEST.json')==spec['raw_manifest_sha256'],'Configured raw manifest pin mismatch')
    plan=_plan(root/'PLAN_FROZEN.json');require(plan['task'] in ('BENCH05-C2','BENCH05-D1'),'Configured raw/plan type mismatch')
    is_structure=plan['task']=='BENCH05-D1'
    manifest=_verify(root,'structure_probe_raw_v1' if is_structure else 'token_probe_raw_v1')
    normalization,consistency=_contract(plan);progress=read(root/'progress.json');all_rows=plan['samples']['rows']
    require(all_rows and len({r['sample_id'] for r in all_rows})==len(all_rows),'Configured sample IDs must be globally unique')
    rows={r['sample_id']:r for r in all_rows}
    for row in all_rows:_row_identity(row,plan)
    selected={sid:r for sid,r in rows.items() if _selected(spec,r)}
    bundle=sha(root/'RAW_MANIFEST.json');tokens=[];sequences={};panels={}
    coverage={'target_requested':0,'target_measured':0,'full_vocab_requested':0,'full_vocab_measured':0,'same_pass_consistency_verified':0,
              'field_membership_origin_entries':0,'field_membership_unique_entries':0,'field_membership_repeated_entries':0,
              'field_membership_reconciled_records':0}
    reconciled_records=[]
    for arm in plan['arms']:
        name=arm['arm_id'];job=progress.get('jobs',{}).get('main-'+name);execution=None;directory=None;targets={};full={}
        if job and job['status']=='COMPLETE':
            directory=root/safe_name(job['directory']);execution=read(directory/'execution.json');result=read(directory/'result.json')
            require(execution['status']==result['status']=='MEASURED' and execution['phase']=='main','Configured diagnostic completion mismatch')
            for key,want in (('arm_id',name),('model_id',plan['pair']['model_id']),('model_revision',plan['pair']['revision']),
                 ('artifact_identity',arm['artifact']['artifact_identity']),('recipe_identity',arm['artifact']['recipe_identity']),
                 ('tokenizer_identity',plan['samples']['tokenizer_identity']),('runtime_identity',plan['runtime_identity']),('execution_mode',arm['execution_mode'])):
                require(execution.get(key)==want,'Configured execution identity mismatch: '+key)
            require(execution['evidence_kind']==('synthetic_contract' if plan.get('evidence_kind')=='synthetic_contract' else 'live_model_diagnostic'),'Configured synthetic/live source mismatch')
            for observation in _lines(directory/'observations.jsonl'):
                if observation['kind'] not in ('target_logprob','full_logits'):continue
                Observation(**observation).validate();sid=observation['sample_id'];require(sid in rows,'Unknown configured observation sample')
                if is_structure and observation['coordinates'].get('pass_id') in ('OFF1','OFF2'):continue
                row=rows[sid];coordinates=observation['coordinates'];_token_coordinate(row,coordinates,plan)
                require(observation['execution_id']==execution['execution_id'] and observation['evidence_kind']==execution['evidence_kind'] and observation['label']==execution['execution_mode'],'Configured observation/execution mismatch')
                for key,want in (('sample_id',sid),('pass_id','ON'),('token_id_space',plan['samples']['token_id_space']),('tokenizer_identity',plan['samples']['tokenizer_identity']),('sequence_identity',row['sequence_identity'])):
                    require(coordinates.get(key)==want,'Configured observation coordinate mismatch: '+key)
                position=coordinates['token_position'];require(position in {t['token_position'] for t in row['targets']},'Observation outside configured target selection')
                mapping=targets if observation['kind']=='target_logprob' else full;key=(sid,position)
                require(key not in mapping,'Duplicate configured observation coordinate')
                if observation['kind']=='full_logits':require(position in row['full_vocab_target_positions'],'Full logits outside configured selection')
                if observation['status']=='MEASURED':
                    details=observation['details']
                    require(details.get('normalization_dtype')==normalization['dtype'] and details.get('probability_stage')==normalization['probability_stage'] and details.get('log_base')==normalization['log_base'],'Configured observation normalization/probability stage mismatch')
                    require(details.get('consistency_policy')==consistency,'Configured observation consistency declaration mismatch')
                mapping[key]=observation
        for row in selected.values():
            panel_identity={key:row.get(key) for key in ('task_id','task_version','dataset_identity','split','policy_id','reference_role')}
            panel=digest(panel_identity);panels[panel]=panel_identity
            choice=row.get('choice_index',row.get('gold_choice_index')) if is_structure else row.get('choice_index')
            sequence=_sequence(spec['source_id'],row,row.get('policy_id'),choice);sequence['task_id']=row['task_id'];sequence['task_version']=row['task_version'];sequence['dataset_identity']=row['dataset_identity']
            _merge_sequence(sequences,sequence)
            coverage['full_vocab_requested']+=len(row['full_vocab_target_positions'])
            for target in row['targets']:
                coverage['target_requested']+=1;position=target['token_position'];observation=targets.get((row['sample_id'],position));full_observation=full.get((row['sample_id'],position))
                mode=execution['execution_mode'] if execution else arm['execution_mode']
                coordinates=target|{'prefix_hash':digest({'token_ids':row['input_ids'][:position],'token_id_space':plan['samples']['token_id_space']})}
                value={'record_id':None,'source_id':spec['source_id'],'panel':panel,'source_panel_id':row['task_id']+'/'+row['task_version']+':'+str(row.get('policy_id'))+':'+row['split'],
                  'bundle_identity':bundle,'plan_identity':plan['plan_identity'],'execution_id':execution['execution_id'] if execution else None,
                  'observation_id':observation['observation_id'] if observation else None,'arm_id':name,'artifact_identity':arm['artifact']['artifact_identity'],
                  'recipe_identity':arm['artifact']['recipe_identity'],'model_id':plan['pair']['model_id'],'model_revision':plan['pair']['revision'],
                  'task_id':row['task_id'],'task_version':row['task_version'],'dataset_identity':row['dataset_identity'],'split':row['split'],
                  'sample_id':row['sample_id'],'policy_id':row.get('policy_id'),'policy_identity':row.get('policy_identity'),'choice_index':choice,
                  'reference_scope':row['reference_role'],'reference_source':row['reference_source'],'prompt_hash':row['prompt_hash'],'reference_hash':row['reference_hash'],
                  'sequence_id':sequence['sequence_id'],'tokenizer_identity':plan['samples']['tokenizer_identity'],'token_id_space':plan['samples']['token_id_space'],
                  'vocabulary_size':plan['samples']['vocabulary_size'],'acquisition_source':'decoded_diagnostic' if mode=='DECODED_WEIGHT_DIAGNOSTIC' else 'instrumented_diagnostic',
                  'runtime_identity':plan['runtime_identity'],'backend_policy':'diagnostic','probability_stage':normalization['probability_stage'],
                  'metric_contract':'target_logprob_v1',**_token_coordinate(row,coordinates,plan),'target_logprob':None,'raw_argmax_token_id':None,'full_logits':None,
                  'full_logits_status':full_observation['status'] if full_observation else 'NOT_RUN' if position in row['full_vocab_target_positions'] else 'NOT_SELECTED',
                  'field_tags':['continuation'],'requested':True,'status':observation['status'] if observation else 'FAILED' if job and job['status']!='COMPLETE' else 'NOT_RUN',
                  'source_job_status':job['status'] if job else 'NOT_RUN','execution_mode':mode,'evidence_kind':execution['evidence_kind'] if execution else 'NOT_RUN',
                  'normalization_dtype':normalization['dtype'],'logit_logprob_tolerance':consistency['absolute_tolerance_nat'],'consistency_policy_id':consistency['id'],
                  'consistency_policy':consistency,'full_logit_consistency':{'status':'UNKNOWN','reason':'FULL_OR_TARGET_NOT_MEASURED'}}
                tags,memberships,origins=_field_membership(row,position,choice)
                value.update(field_tags=tags,field_memberships=memberships,field_membership_policy=FIELD_MEMBERSHIP_POLICY['id'])
                coverage['field_membership_origin_entries']+=origins
                coverage['field_membership_unique_entries']+=len(tags)
                coverage['field_membership_repeated_entries']+=origins-len(tags)
                if observation and observation['status']=='MEASURED':
                    scalar=load(directory,observation['numeric']);require(len(scalar)==1 and math.isfinite(scalar[0]) and scalar[0]<=0,'Invalid configured target scalar')
                    value.update(target_logprob=scalar[0],raw_argmax_token_id=observation['details'].get('raw_argmax_token_id'),source_observation=observation,
                        source_observations_path=str((directory/'observations.jsonl').relative_to(root)),source_numeric_sha256=observation['numeric']['sha256'],
                        effective_weight_dtype=execution['effective_weight_dtype'],backend=execution['backend'])
                    coverage['target_measured']+=1
                    if full_observation and full_observation['status']=='MEASURED':
                        require(full_observation['details'].get('full_vocabulary') is True and full_observation['numeric']['shape']==[plan['samples']['vocabulary_size']],'Configured full vocabulary claim/axis mismatch')
                        logits=load(directory,full_observation['numeric']);normalized=log_softmax(logits)
                        error=abs(normalized[target['target_token_id']]-scalar[0]);require(error<=consistency['absolute_tolerance_nat'],'Configured same-pass saved scalar/full consistency violation')
                        argmax=value['raw_argmax_token_id']
                        if argmax is not None:require(argmax==max(range(len(logits)),key=logits.__getitem__),'Configured saved argmax/full mismatch')
                        value['full_logits']=blob_sink(directory,full_observation['numeric']);value['source_full_observation']=full_observation
                        value['full_logit_consistency']={'status':'MEASURED','absolute_error_nat':error,'absolute_tolerance_nat':consistency['absolute_tolerance_nat'],'within_declared_tolerance':True}
                        coverage['full_vocab_measured']+=1;coverage['same_pass_consistency_verified']+=1
                value['record_id']=digest({key:value[key] for key in ('source_id','panel','bundle_identity','plan_identity','execution_id','observation_id','arm_id','sample_id','token_position','logit_position','target_token_id','prefix_hash')})
                if origins!=len(tags):
                    coverage['field_membership_reconciled_records']+=1
                    reconciled_records.append({'record_id':value['record_id'],'observation_id':value['observation_id'],
                        'arm_id':name,'task_id':row['task_id'],'sample_id':row['sample_id'],'token_position':position,
                        'origin_entries':origins,'unique_memberships':len(tags)})
                if is_structure:value.update(pass_id='ON',observer_consistency_policy=plan['structure_spec']['observer_consistency_policy'])
                tokens.append(value)
    require(len({r['record_id'] for r in tokens})==len(tokens),'Duplicate generic token record IDs')
    source={'source_id':spec['source_id'],'adapter':'token_probe_raw_v1','task_id':None,'task_ids':sorted({r['task_id'] for r in selected.values()}),
        'source_bundle_path':str(root),'raw_manifest_sha256':bundle,'bundle_identity':bundle,'raw_manifest_scope':manifest['scope'],'verified_raw_members':len(manifest['files']),
        'plan_identity':plan['plan_identity'],'runtime_identity':plan['runtime_identity'],'source_acquisition_code_identity':digest(plan['source_identity']),
        'source_acquisition_code_files':plan['source_identity'],'evidence_kind':plan['evidence_kind'],'coverage':coverage,'panel_definitions':panels,
        'probe_spec':plan['probe_spec'],'filters':{k:spec[k] for k in ('task_ids','sample_ids','splits') if k in spec},
        'field_membership_policy':FIELD_MEMBERSHIP_POLICY,'field_membership_reconciliation':reconciled_records,
        'backend_equivalence':'NOT_CLAIMED','GPU_actions':0,'model_loads':0,'tokenizer_loads':0}
    if is_structure:source.update(token_pass='ON',excluded_control_passes=['OFF1','OFF2'],structure_spec=plan['structure_spec'],structure_selection_identity=plan['structure_selection']['selection_identity'])
    return {'source':source,'sequences':list(sequences.values()),'tokens':tokens}
