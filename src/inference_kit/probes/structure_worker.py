"""Explicit D1 worker: one model, selected tokens and temporary structure hooks.

Import is stdlib-only. OFF1/OFF2 retain the C2 selected-token path; ON wraps
the same backbone once with the reviewed multi-layer observer. No task content
is interpreted and no new tokenizer or serving path is invoked here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from ..core.io import read,write,require,digest
from .contracts import Execution,Observation
from .local_transformer import selected_token_forward
from .token_worker import (CONSISTENCY_POLICY,NORMALIZATION,MAX_BYTES,
                           validate_worker_row,observation_details,store_numeric,
                           check_saved_consistency)

PASSES=('OFF1','OFF2','ON')
OBSERVER_POLICY={'id':'d1_off_repeat_and_on_v1','absolute_tolerance':1e-5,
                 'passes':list(PASSES)}
STRUCTURE_PATHS={'attention_row':'reconstructed_from_native_qk',
                 'attention_output':'native_returned_SDPA_AV',
                 'reconstructed_AV':'reconstructed_AV_from_FP32_row_and_native_V',
                 'output_projection':'native_o_proj_forward_hook_before_residual'}


def forward_pass(model,ids,mask,targets,buffer_cap,pass_id,*,observer_factory=None,functional=None):
    """Small shared dispatch; an OFF pass never constructs an observer."""
    require(pass_id in PASSES,'Unknown D1 observer pass')
    require(pass_id!='ON' or observer_factory is not None,'ON pass requires explicit observer')
    observer=observer_factory() if pass_id=='ON' else None
    result=selected_token_forward(model,ids,mask,targets,buffer_cap,
                                  context=observer,functional=functional)
    return result,observer


def state_versions(model):
    """Capture object identity/version as well as shape and dtype, without copies."""
    return {group:{name:(id(value),value._version,tuple(value.shape),str(value.dtype))
                   for name,value in iterator()}
            for group,iterator in (('parameters',model.named_parameters),
                                   ('buffers',model.named_buffers))}


def validate_selection(row,selection,config):
    """Validate this prepared sequence against frozen physical query coordinates."""
    layers=selection['layers'];heads=selection['heads'];queries=selection['queries']
    for name,values,limit in (('layers',layers,config['num_hidden_layers']),
                              ('heads',heads,config['num_attention_heads']),
                              ('queries',queries,len(row['input_ids']))):
        require(isinstance(values,list) and values and all(type(value) is int and 0<=value<limit for value in values)
                and values==sorted(set(values)),'Invalid/duplicate D1 '+name)
    roles=selection['query_roles']
    require(set(roles)=={str(q) for q in queries},'D1 query role axes differ')
    require(all(isinstance(roles[str(q)],list) and roles[str(q)] and
                len(set(roles[str(q)]))==len(roles[str(q)]) and
                set(roles[str(q)])<={'last_prompt','before_last_reference'} for q in queries),'Invalid D1 role aliases')
    expected={'last_prompt':row['prompt_token_boundary']-1,
              'before_last_reference':max(t['token_position'] for t in row['targets'])-1}
    # The planner may use the final eligible reference rather than the selected
    # target subset. It records that exact coordinate, checked from the loss mask.
    if row.get('loss_mask'):
        eligible=[t for t,loss in enumerate(row['loss_mask'])
                  if loss and t>0 and (not row.get('special_token_mask') or not row['special_token_mask'][t])]
        require(eligible,'No eligible D1 reference query')
        expected['before_last_reference']=max(eligible)-1
    for query in queries:
        require(row.get('attention_mask',[1]*len(row['input_ids']))[query]==1,'D1 query is masked')
        for role in roles[str(query)]:require(query==expected[role],'D1 query/role coordinate differs')
    return selection


def coordinates_for(row,prepared,pass_id,coordinates):
    """One global prefix and task identity contract for tokens and structure."""
    require(pass_id in PASSES,'Unknown D1 observer pass')
    result={'sample_id':row['sample_id'],'token_id_space':prepared['token_id_space'],
            'tokenizer_identity':prepared['tokenizer_identity'],'sequence_identity':row['sequence_identity'],
            'pass_id':pass_id,**coordinates}
    for key in ('task_id','task_version','source_id','split','reference_role','reference_source'):
        if key in row:result[key]=row[key]
    prefix_length=result['token_position'] if 'token_position' in result else result['query_position']+1
    require(type(prefix_length) is int and 0<prefix_length<=len(row['input_ids']),'D1 prefix outside prepared sequence')
    result['prefix_hash']=digest({'token_ids':row['input_ids'][:prefix_length],
                                  'token_id_space':prepared['token_id_space']})
    return result


def run(plan_file,arm_id,output):
    import torch
    import torch.nn.functional as F
    from importlib import metadata
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .structure_observer import StructureObserver
    from .weights import load_model
    from ..suites.bench05a_plan import source_identity

    plan=read(Path(plan_file));directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    require(plan['task']=='BENCH05-D1','Wrong structure worker task')
    require(source_identity()==plan['source_identity'],'Installed structure worker source differs from frozen plan')
    frozen=dict(plan);identity=frozen.pop('plan_identity');require(digest(frozen)==identity,'Structure worker plan integrity differs')
    probe=plan['probe_spec'];structure=plan['structure_spec']
    require(probe['consistency_policy']==CONSISTENCY_POLICY and probe.get('normalization')==NORMALIZATION,'Unapproved token normalization/consistency contract')
    require(structure['observer_consistency_policy']==OBSERVER_POLICY,'Unapproved D1 OFF/ON consistency policy')
    budget=plan['suite']['budget'];buffer_cap=budget['probe_gpu_buffer_bytes'];raw_cap=budget['raw_bytes_per_job']
    require(0<buffer_cap<=MAX_BYTES and 0<raw_cap<=MAX_BYTES,'D1 buffer/raw cap exceeds256MiB')
    arm=next(a for a in plan['arms'] if a['arm_id']==arm_id);prepared=plan['samples'];rows=prepared['rows'];chosen=plan['structure_selection']
    require(rows and prepared.get('rows_identity',digest(rows))==digest(rows),'Structure worker rows changed')
    selections={}
    for row in rows:
        validate_worker_row(row,probe,prepared['vocabulary_size'])
        selections[row['sample_id']]=validate_selection(row,{'layers':chosen['layers'],'heads':chosen['heads'],
            **chosen['rows'][row['sample_id']]},prepared['model_config'])
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(12,0),'Explicit approved RTX5080 diagnostic unavailable')
    torch.set_num_threads(2);torch.manual_seed(plan['suite']['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();model=None;storage_bytes=0;observations=[];consistency=[];active={};observers=[];passes=[];peak_buffer=0
    execution=Execution(directory.parent.parent.name,directory.name,arm_id,plan['pair']['model_id'],plan['pair']['revision'],
      arm['artifact']['artifact_identity'],arm['artifact']['recipe_identity'],prepared['tokenizer_identity'],plan['runtime_identity'],
      arm['execution_mode'],'torch.bfloat16','transformers5.14.1/torchSDPA/native-selected-lm-head',
      'live_model_diagnostic','MEASURED','observations.jsonl').validate()
    execution.update(phase='main',observation_profile='structure_probe_raw_v1',pass_ids=list(PASSES),
                     observer_consistency_policy=OBSERVER_POLICY)
    execution['execution_id']=digest(execution);write(directory/'execution.json',execution)
    write(directory/'environment.json',{'pid':os.getpid(),'gpu_name':torch.cuda.get_device_name(),
      'capability':list(torch.cuda.get_device_capability()),'torch':torch.__version__,'python':sys.executable,
      'packages':{name:{'version':metadata.version(name),'package_root':str(metadata.distribution(name).locate_file(''))}
                  for name in ('transformers','compressed-tensors','torch','safetensors')},
      'package_import_origin':__file__,'observer':'One temporary SDPA dispatch; selected module identity; o_proj hooks',
      'attention_passes':['ON'],'no_grad':True,'eval':True,'tf32':False,'batch':1,'use_cache':False,
      'normalization':NORMALIZATION,'consistency_policy':CONSISTENCY_POLICY,
      'observer_consistency_policy':OBSERVER_POLICY,'source_identity':plan['source_identity']})
    raw=(directory/'observations.jsonl').open('x',encoding='utf-8')

    def emit(kind,coordinates,tensor,details):
        nonlocal storage_bytes
        row=active['row'];coords=coordinates_for(row,prepared,active['pass_id'],coordinates)
        identifier=digest({'execution':execution['execution_id'],'kind':kind,'coordinates':coords})[:28]
        numeric=None;failed=None
        if tensor is not None:
            value=tensor.detach().float().cpu().contiguous();shape=list(value.shape) or [1]
            numeric,failed,storage_bytes=store_numeric(directory,identifier,value.reshape(-1).tolist(),shape,storage_bytes,raw_cap)
        details=dict(details,observer_consistency_policy=OBSERVER_POLICY)
        if kind in ('target_logprob','full_logits'):details=observation_details(details)
        else:
            require(active['pass_id']=='ON' and kind in STRUCTURE_PATHS,'Structure observation outside ON pass')
            details['measurement_path']=STRUCTURE_PATHS[kind]
            details['structure_spec_identity']=digest(structure)
        if failed:details.update(failure='NONFINITE_NUMERIC',preserved_failed_numeric=failed)
        status='MEASURED' if numeric else 'FAILED' if failed else details.get('status','NOT_APPLICABLE')
        observation=Observation(identifier,kind,execution['execution_id'],row['sample_id'],'live_model_diagnostic',
                                arm['execution_mode'],status,coords,numeric,details).validate()
        raw.write(json.dumps(observation,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();observations.append(observation)
        require(failed is None,'Nonfinite structure/token numeric preserved at '+identifier)
        return observation

    try:
        reference=next(a['artifact']['path'] for a in plan['arms'] if a['arm_id']=='bf16')
        model,receipt=load_model(arm,reference,reference);model.eval()
        write(directory/'loader.json',receipt|{'loading_seconds':time.monotonic()-started})
        versions=state_versions(model)
        modules={layer:model.model.layers[layer].self_attn for layer in chosen['layers']}
        for row in rows:
            require(row['sequence_identity']==digest(row['input_ids']),'Structure worker sequence identity differs')
            selection=selections[row['sample_id']]
            for pass_id in PASSES:
                active.update(row=row,pass_id=pass_id)
                ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda')
                mask=torch.tensor([row.get('attention_mask',[1]*len(row['input_ids']))],dtype=torch.long,device='cuda')
                input_versions=(ids._version,mask._version);pass_start=time.monotonic();context=None
                def observer_factory():
                    observer=StructureObserver(ALL_ATTENTION_FUNCTIONS,modules,selection,emit,buffer_limit_bytes=buffer_cap)
                    observers.append(observer);return observer
                with torch.no_grad():
                    selected,context=forward_pass(model,ids,mask,row['targets'],buffer_cap,pass_id,
                                                  observer_factory=observer_factory,functional=F)
                    hidden,hidden_rows,logits,logprobs=(selected[key] for key in ('hidden','chosen','logits','logprobs'))
                    peak_buffer=max(peak_buffer,selected['required_buffer_bytes']);del selected
                    require(logits.shape[-1]==prepared['vocabulary_size'],'Native LM head vocabulary differs')
                    for j,target in enumerate(row['targets']):
                        scalar=emit('target_logprob',target,logprobs[j,target['target_token_id']].reshape(1),{
                          'raw_argmax_token_id':int(logits[j].argmax()),'raw_logits_dtype':str(logits.dtype),
                          'sampling':'raw pre-sampling','causal_mask':'native model SDPA; selected ON rows separately checked',
                          'lm_head_path':'native final norm model.model -> native tied lm_head on selected hidden rows only'})
                        if target['token_position'] in row['full_vocab_target_positions']:
                            full=emit('full_logits',target,logits[j],{'full_vocabulary':True,'vocabulary_size':logits.shape[-1],
                              'raw_dtype':str(logits.dtype),'lm_head_path':'native selected hidden rows only','softcap':'not present in approved Smol config'})
                            check=check_saved_consistency(directory,scalar['numeric'],full['numeric'],target['target_token_id'],CONSISTENCY_POLICY)
                            check.update(sample_id=row['sample_id'],pass_id=pass_id,token_position=target['token_position'],
                              scalar_observation_id=scalar['observation_id'],full_observation_id=full['observation_id'])
                            consistency.append(check)
                            if check['status']!='PASS':
                                write(directory/'failed-consistency-coordinate.json',check)
                                require(False,'Saved same-pass scalar/full consistency violation; raw preserved')
                    require(input_versions==(ids._version,mask._version),'D1 input/mask changed in-place during forward')
                    del hidden,hidden_rows,logits,logprobs,ids,mask
                require(state_versions(model)==versions,'Model parameter/buffer identity/version changed during D1 observation')
                if context:
                    require(context.restored and context.hooks_removed and context.no_inplace_changes,'D1 hooks/interface restoration or native mutation failure')
                    expected_calls={layer:1 for layer in selection['layers']}
                    require(context.calls_by_layer==expected_calls and context.projection_calls_by_layer==expected_calls,
                            'Selected layer native SDPA/projection call coverage differs; no repeated forward is allowed')
                torch.cuda.synchronize()
                passes.append({'sample_id':row['sample_id'],'pass_id':pass_id,'wall_seconds':time.monotonic()-pass_start,
                  'target_positions':len(row['targets']),'full_vocab_positions':len(row['full_vocab_target_positions']),
                  'observer_installed':context is not None,'restored':context.restored if context else True})
        stats={file.name:{'bytes':file.stat().st_size,'mtime_ns':file.stat().st_mtime_ns}
               for file in Path(arm['artifact']['path']).glob('*.safetensors')}
        require(stats==arm['weight_file_stat'],'Source artifact file stat changed')
        counts={pass_id:{'target_positions':sum(o['kind']=='target_logprob' and o['coordinates']['pass_id']==pass_id for o in observations),
                        'full_vocab_positions':sum(o['kind']=='full_logits' and o['coordinates']['pass_id']==pass_id for o in observations)} for pass_id in PASSES}
        kinds={kind:sum(o['kind']==kind and o['coordinates']['pass_id']=='ON' for o in observations)
               for kind in ('attention_row','attention_output','reconstructed_AV','output_projection')}
        observer_peak=max((observer.max_selected_temporary_bytes for observer in observers),default=0)
        result={'schema_version':1,'status':'MEASURED','evidence_kind':'live_model_diagnostic','phase':'main','arm_id':arm_id,
          'execution_mode':arm['execution_mode'],'samples':len(rows),'forward_calls':len(passes),
          'main_target_positions':counts['ON']['target_positions'],'full_vocab_positions':counts['ON']['full_vocab_positions'],
          'main_attention_rows':kinds['attention_row'],'structure_counts':kinds,'pass_counts':counts,'passes':passes,
          'observation_records':len(observations),'numeric_bytes':storage_bytes,'wall_seconds':time.monotonic()-started,
          'observer_contexts':len(observers),'observer_calls':sum(sum(observer.calls_by_layer.values()) for observer in observers),
          'observer_calls_by_context':[dict(observer.calls_by_layer) for observer in observers],
          'all_hooks_interfaces_restored':all(observer.restored and observer.hooks_removed for observer in observers),
          'no_inplace_changes':all(observer.no_inplace_changes for observer in observers),
          'weight_version_unchanged':True,'buffer_version_unchanged':True,'source_weight_stat_unchanged':True,
          'allocator_max_allocated_bytes':torch.cuda.max_memory_allocated(),'allocator_max_reserved_bytes':torch.cuda.max_memory_reserved(),
          'allocator_scope':'Whole diagnostic worker including native/dense decoded load, not INT4 serving memory',
          'selected_probe_buffer_actual_max_bytes':max(peak_buffer,observer_peak),'selected_token_buffer_actual_max_bytes':peak_buffer,
          'selected_structure_temporary_max_bytes':observer_peak,'selected_probe_buffer_cap_bytes':buffer_cap,
          'selected_buffer_scope':'Peak token selection or observer temporaries; native backbone QKV/SDPA workspaces excluded; contexts do not overlap LM head',
          'selected_probe_buffer_estimate_upper_bytes':plan['resource_estimates']['extra_probe_gpu_buffer_upper_bytes'],
          'no_full_T_by_V_logits_created':True,'no_all_layer_attention_created':True,
          'kernel_observation':'Original diagnostic torch SDPA once per native attention invocation, no vLLM/Marlin/profiler claim',
          'observer_consistency_policy':OBSERVER_POLICY,
          'same_pass_consistency':{'policy':CONSISTENCY_POLICY,'coordinates':consistency,'checked':len(consistency),'violations':0}}
        write(directory/'result.json',result);return result
    except BaseException as exc:
        if not (directory/'failure.json').exists():
            write(directory/'failure.json',{'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),
              'sample_id':active.get('row',{}).get('sample_id'),'pass_id':active.get('pass_id'),
              'restored_contexts':[observer.restored for observer in observers],
              'hooks_removed':[observer.hooks_removed for observer in observers],
              'numeric_bytes':storage_bytes,'preserved_observation_records':len(observations),'consistency_coordinates':consistency})
        raise
    finally:
        raw.close();del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('plan','arm','output'):parser.add_argument('--'+name,required=True,type=Path if name!='arm' else str)
    args=parser.parse_args();result=run(args.plan,args.arm,args.output)
    print(json.dumps(result,ensure_ascii=True,allow_nan=False))
