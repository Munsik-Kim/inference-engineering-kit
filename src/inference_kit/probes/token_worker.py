"""Opt-in token-only worker over frozen selected positions; no attention hooks.

Module import is CPU/stdlib-only. The explicit ``run`` entrypoint imports an
existing diagnostic runtime, loads exactly one native/decoded model, and creates
LM-head logits for selected hidden rows rather than full sequence by vocabulary.
"""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import sys
import time

from ..core.io import read,write,require,digest,sha
from ..metrics.token import log_softmax
from .contracts import Execution,Observation
from .blobs import save,load,size
from .local_transformer import selected_token_forward

CONSISTENCY_POLICY={'id':'same_pass_saved_f32_v1','absolute_tolerance_nat':2e-5,
 'scope':'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'}
NORMALIZATION={'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'}
MAX_BYTES=256*1024*1024


def validate_worker_row(row,probe_spec,vocabulary_size=None):
    """Pure CPU bound/coordinate validation from the frozen C2 ProbeSpec."""
    sequence_cap=probe_spec['max_sequence_tokens'];target_cap=probe_spec['max_target_positions_per_sample'];full_cap=probe_spec['max_full_vocab_positions_per_sample']
    require(type(sequence_cap) is int and 1<=sequence_cap<=640,'Configured token sequence cap')
    require(type(target_cap) is int and 1<=target_cap<=32,'Configured target cap')
    require(type(full_cap) is int and 0<=full_cap<=min(8,target_cap),'Configured full-vocabulary cap')
    require(probe_spec.get('observe_attention') is False,'Token-only worker must not observe attention')
    ids=row['input_ids'];targets=row['targets'];full=row['full_vocab_target_positions']
    require(isinstance(ids,list) and 0<len(ids)<=sequence_cap and all(type(t) is int and t>=0 for t in ids),'Input exceeds configured sequence cap or invalid IDs; never truncate')
    if vocabulary_size is not None:require(all(t<vocabulary_size for t in ids),'Configured input token outside vocabulary')
    require(isinstance(targets,list) and 0<len(targets)<=target_cap,'Configured target count cap')
    positions=[]
    for target in targets:
        t=target['token_position']
        require(type(t) is int and 0<t<len(ids) and type(target['logit_position']) is int and target['logit_position']==t-1 and type(target['target_token_id']) is int and target['target_token_id']==ids[t],'Configured next-token coordinate differs')
        if 'prompt_token_boundary' in row:require(t>=row['prompt_token_boundary'],'Configured target before reference boundary')
        if 'loss_mask' in row:require(len(row['loss_mask'])==len(ids) and row['loss_mask'][t]==1,'Configured target outside reference loss mask')
        positions.append(t)
    require(len(set(positions))==len(positions),'Duplicate configured target position')
    require(isinstance(full,list) and len(full)<=full_cap and all(type(t) is int for t in full) and len(set(full))==len(full) and set(full)<=set(positions),'Configured full-vocabulary selection cap or coordinate differs')
    if 'attention_mask' in row:require(row['attention_mask']==[1]*len(ids),'Token-only capture requires unpadded sequence')
    return {'sequence_tokens':len(ids),'target_positions':len(targets),'full_vocab_positions':len(full),'attention_rows':0}


def observation_details(details):
    """One explicit metadata contract for scalar AND full-logit observations."""
    return dict(details,normalization=NORMALIZATION,normalization_dtype='float32',
                consistency_policy=CONSISTENCY_POLICY,
                absolute_tolerance_nat=CONSISTENCY_POLICY['absolute_tolerance_nat'],
                probability_stage='raw_pre_sampling',log_base='e')


def store_numeric(directory,identifier,values,shape,used_bytes,raw_cap):
    """Preserve failed nonfinite bytes separately, never as measured blobs."""
    n=size(shape);require(len(values)==n,'Token numeric shape differs')
    require(0<raw_cap<=MAX_BYTES and used_bytes+n*4<=raw_cap,'Token-only raw storage cap')
    used_bytes+=n*4
    if all(math.isfinite(value) for value in values):
        numeric=save(directory,identifier,values,shape)
        return numeric,None,used_bytes
    value=array('f',values)
    if sys.byteorder!='little':value.byteswap()
    path=Path(directory)/'failed-numeric'/(identifier+'.f32');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:value.tofile(stream)
    failed={'path':path.relative_to(directory).as_posix(),'dtype':'float32','shape':shape,
            'endianness':'little','bytes':n*4,'sha256':sha(path),'numeric_status':'NONFINITE_NOT_VALIDATED',
            'nan_count':sum(math.isnan(v) for v in values),'positive_infinity_count':sum(v==math.inf for v in values),
            'negative_infinity_count':sum(v==-math.inf for v in values)}
    return None,failed,used_bytes


def check_saved_consistency(directory,scalar_numeric,full_numeric,target_id,policy):
    """Check serialized f32 evidence, with a frozen absolute nat tolerance."""
    require(policy==CONSISTENCY_POLICY,'Unapproved saved-f32 consistency policy')
    scalar=load(directory,scalar_numeric);full=load(directory,full_numeric)
    require(len(scalar)==1 and type(target_id) is int and 0<=target_id<len(full),'Saved consistency coordinate differs')
    computed=log_softmax(full)[target_id];delta=abs(computed-scalar[0])
    return {'status':'PASS' if delta<=policy['absolute_tolerance_nat'] else 'FAILED',
            'saved_target_logprob_nat':scalar[0],'saved_full_logits_cpu_logprob_nat':computed,
            'absolute_difference_nat':delta,'policy':policy,'target_token_id':target_id}


def run(plan_file,arm_id,output):
    import torch
    import torch.nn.functional as F
    from importlib import metadata
    from .weights import load_model
    from ..suites.bench05a_plan import source_identity
    plan=read(Path(plan_file));directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    require(source_identity()==plan['source_identity'],'Installed token worker source differs from frozen plan')
    frozen=dict(plan);identity=frozen.pop('plan_identity');require(digest(frozen)==identity,'Token worker plan integrity differs')
    probe=plan['probe_spec'];require(probe['consistency_policy']==CONSISTENCY_POLICY,'Unapproved token consistency policy')
    require(probe.get('normalization')==NORMALIZATION,'Token normalization contract differs')
    budget=plan['suite']['budget'];buffer_cap=budget['probe_gpu_buffer_bytes'];raw_cap=budget['raw_bytes_per_job']
    require(0<buffer_cap<=MAX_BYTES and 0<raw_cap<=MAX_BYTES,'Token-only buffer/raw cap exceeds256MiB')
    arm=next(a for a in plan['arms'] if a['arm_id']==arm_id);prepared=plan['samples'];rows=prepared['rows']
    require(rows and prepared.get('rows_identity',digest(rows))==digest(rows),'Token worker rows changed')
    for row in rows:validate_worker_row(row,probe,prepared['vocabulary_size'])
    require(torch.cuda.is_available() and torch.cuda.get_device_capability()==(12,0),'Explicit approved RTX5080 token diagnostic unavailable')
    torch.set_num_threads(2);torch.manual_seed(plan['suite']['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();model=None;storage_bytes=0;observations=[];consistency=[];active={};peak_buffer=0
    execution=Execution(directory.parent.parent.name,directory.name,arm_id,plan['pair']['model_id'],plan['pair']['revision'],
      arm['artifact']['artifact_identity'],arm['artifact']['recipe_identity'],prepared['tokenizer_identity'],plan['runtime_identity'],
      arm['execution_mode'],'torch.bfloat16','transformers5.14.1/torchSDPA/native-selected-lm-head',
      'live_model_diagnostic','MEASURED','observations.jsonl').validate()
    execution['phase']='main';execution['observation_profile']='token_only_configured_v1';execution['execution_id']=digest(execution)
    write(directory/'execution.json',execution)
    write(directory/'environment.json',{'pid':os.getpid(),'gpu_name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
      'torch':torch.__version__,'python':sys.executable,'packages':{name:{'version':metadata.version(name),'package_root':str(metadata.distribution(name).locate_file(''))} for name in ('transformers','compressed-tensors','torch','safetensors')},'package_import_origin':__file__,'observer':'NONE_TOKEN_ONLY','attention_hooks_installed':0,
      'all_attention_functions_imported':False,'no_grad':True,'eval':True,'tf32':False,'batch':1,'use_cache':False,
      'normalization':NORMALIZATION,'consistency_policy':CONSISTENCY_POLICY,'source_identity':plan['source_identity']})
    raw=(directory/'observations.jsonl').open('x',encoding='utf-8')
    def emit(kind,target,tensor,details):
        nonlocal storage_bytes
        row=active['row'];coords={'sample_id':row['sample_id'],'token_id_space':prepared['token_id_space'],
          'tokenizer_identity':prepared['tokenizer_identity'],'sequence_identity':row['sequence_identity'],'pass_id':'ON',**target}
        coords['prefix_hash']=digest({'token_ids':row['input_ids'][:target['token_position']],'token_id_space':prepared['token_id_space']})
        identifier=digest({'execution':execution['execution_id'],'kind':kind,'coordinates':coords})[:28]
        value=tensor.detach().float().cpu().contiguous();shape=list(value.shape) or [1]
        numeric,failed,storage_bytes=store_numeric(directory,identifier,value.reshape(-1).tolist(),shape,storage_bytes,raw_cap)
        details=observation_details(details)
        if failed:details.update(failure='NONFINITE_NUMERIC',preserved_failed_numeric=failed)
        observation=Observation(identifier,kind,execution['execution_id'],row['sample_id'],'live_model_diagnostic',
          arm['execution_mode'],'MEASURED' if numeric else 'FAILED',coords,numeric,details).validate()
        raw.write(json.dumps(observation,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();observations.append(observation)
        require(failed is None,'Nonfinite token numeric preserved at '+identifier)
        return observation
    try:
        reference=next(a['artifact']['path'] for a in plan['arms'] if a['arm_id']=='bf16')
        model,receipt=load_model(arm,reference,reference);model.eval()
        write(directory/'loader.json',receipt|{'loading_seconds':time.monotonic()-started})
        versions={name:parameter._version for name,parameter in model.named_parameters()}
        for row in rows:
            active['row']=row
            require(row['sequence_identity']==digest(row['input_ids']),'Token worker sequence identity differs')
            ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda');mask=torch.ones_like(ids)
            targets=row['targets'];full_positions=row['full_vocab_target_positions']
            validate_worker_row(row,probe,prepared['vocabulary_size'])
            with torch.no_grad():
                selected=selected_token_forward(model,ids,mask,targets,buffer_cap,functional=F)
                hidden,chosen,logits,logprobs=(selected[key] for key in ('hidden','chosen','logits','logprobs'))
                peak_buffer=max(peak_buffer,selected['required_buffer_bytes']);del selected
                require(logits.shape[-1]==prepared['vocabulary_size'],'Native LM head vocabulary differs')
                for j,target in enumerate(targets):
                    require(target['target_token_id']==row['input_ids'][target['token_position']],'Configured target token differs from input')
                    lp=logprobs[j,target['target_token_id']]
                    scalar=emit('target_logprob',target,lp.reshape(1),{'raw_argmax_token_id':int(logits[j].argmax()),
                      'normalization_dtype':'float32','raw_logits_dtype':str(logits.dtype),
                      'sampling':'raw pre-sampling','causal_mask':'native model SDPA; no observer installed',
                      'lm_head_path':'native final norm model.model -> native tied lm_head on selected hidden rows only'})
                    if target['token_position'] in full_positions:
                        full=emit('full_logits',target,logits[j],{'full_vocabulary':True,'vocabulary_size':logits.shape[-1],
                          'raw_dtype':str(logits.dtype),'lm_head_path':'native final norm model.model -> native tied lm_head on selected hidden rows only',
                          'softcap':'not present in approved Smol config'})
                        check=check_saved_consistency(directory,scalar['numeric'],full['numeric'],target['target_token_id'],CONSISTENCY_POLICY)
                        check.update(sample_id=row['sample_id'],token_position=target['token_position'],scalar_observation_id=scalar['observation_id'],full_observation_id=full['observation_id'])
                        consistency.append(check)
                        if check['status']!='PASS':
                            write(directory/'failed-consistency-coordinate.json',check)
                            require(False,'Saved same-pass consistency violation; raw scalar/full logits preserved')
                del hidden,chosen,logits,logprobs,ids,mask
            require(all(parameter._version==versions[name] for name,parameter in model.named_parameters()),'Model weights changed during token observation')
            torch.cuda.synchronize()
        stats={file.name:{'bytes':file.stat().st_size,'mtime_ns':file.stat().st_mtime_ns} for file in Path(arm['artifact']['path']).glob('*.safetensors')}
        require(stats==arm['weight_file_stat'],'Source artifact file stat changed')
        result={'schema_version':1,'status':'MEASURED','evidence_kind':'live_model_diagnostic','phase':'main','arm_id':arm_id,
          'execution_mode':arm['execution_mode'],'samples':len(rows),'main_target_positions':sum(o['kind']=='target_logprob' for o in observations),
          'main_attention_rows':0,'full_vocab_positions':sum(o['kind']=='full_logits' for o in observations),'observation_records':len(observations),
          'numeric_bytes':storage_bytes,'wall_seconds':time.monotonic()-started,'observer_contexts':0,'observer_calls':0,
          'all_hooks_interfaces_restored':True,'attention_hooks_installed':0,'weight_version_unchanged':True,'source_weight_stat_unchanged':True,
          'allocator_max_allocated_bytes':torch.cuda.max_memory_allocated(),'allocator_max_reserved_bytes':torch.cuda.max_memory_reserved(),
          'allocator_scope':'Whole token-only diagnostic worker including native/dense decoded load; not INT4 serving memory',
          'selected_probe_buffer_actual_max_bytes':peak_buffer,'selected_probe_buffer_cap_bytes':buffer_cap,
          'selected_probe_buffer_estimate_upper_bytes':plan['resource_estimates']['extra_probe_gpu_buffer_upper_bytes'],
          'no_full_T_by_V_logits_created':True,'no_attention_observations_created':True,'kernel_observation':'native torch SDPA; no profiler claim',
          'same_pass_consistency':{'policy':CONSISTENCY_POLICY,'coordinates':consistency,'checked':len(consistency),'violations':0}}
        write(directory/'result.json',result);return result
    except BaseException as exc:
        if not (directory/'failure.json').exists():
            write(directory/'failure.json',{'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),
              'sample_id':active.get('row',{}).get('sample_id'),'attention_hooks_installed':0,'numeric_bytes':storage_bytes,
              'preserved_observation_records':len(observations),'consistency_coordinates':consistency})
        raise
    finally:
        raw.close();del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('plan','arm','output'):parser.add_argument('--'+name,required=True,type=Path if name!='arm' else str)
    args=parser.parse_args();result=run(args.plan,args.arm,args.output)
    print(json.dumps(result,ensure_ascii=True,allow_nan=False))
