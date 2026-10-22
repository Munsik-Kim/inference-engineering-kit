"""Explicit GPU worker for frozen Smol BF16/exact decoded-RTN diagnostics.

Invoked by bench run only; base imports and CPU replay do not import Torch.
"""
import argparse
from pathlib import Path
import json
import os
import time
from ..core.io import read,write,require,digest,sha
from .contracts import Execution,Observation
from .blobs import save


def selected_token_forward(model,input_ids,attention_mask,targets,buffer_cap,*,context=None,functional=None):
    """Run native backbone, then native LM head on selected hidden rows only.

    The optional attention context wraps only the backbone, matching the original
    BENCH05-A call order. Supplying ``functional`` makes the shape/dispatch
    contract testable without importing any model or numerical SDK.
    """
    if functional is None:
        import torch.nn.functional as functional
    require(targets and len(input_ids.shape)==2 and input_ids.shape[0]==1,'Selected forward requires batch1 and targets')
    positions=[target['logit_position'] for target in targets]
    require(all(type(t) is int and 0<=t<input_ids.shape[1] for t in positions) and len(set(positions))==len(positions),'Invalid/duplicate selected logit position')
    require(all(type(target['token_position']) is int and 0<target['token_position']<input_ids.shape[1] and target['token_position']==target['logit_position']+1 for target in targets),'Selected target/logit shift differs')
    if context:
        with context:hidden=model.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
    else:hidden=model.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
    chosen=hidden[0,positions,:]
    logits=model.lm_head(chosen)
    require(len(logits.shape)==2 and logits.shape[0]==len(targets),'Native LM head selected-row shape differs')
    require(all(type(target['target_token_id']) is int and 0<=target['target_token_id']<logits.shape[-1] for target in targets),'Selected target outside vocabulary')
    logprobs=functional.log_softmax(logits.float(),dim=-1)
    required=logits.numel()*logits.element_size()+logprobs.numel()*logprobs.element_size()+chosen.numel()*chosen.element_size()
    require(required<=buffer_cap,'Selected LM head buffer cap')
    return {'hidden':hidden,'chosen':chosen,'logits':logits,'logprobs':logprobs,'required_buffer_bytes':required}


def run(plan_file,arm_id,stage,output):
    import torch
    import torch.nn.functional as F
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .attention import AttentionObserver
    from .weights import load_model
    plan=read(Path(plan_file));directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    from ..suites.bench05a_plan import source_identity
    require(source_identity()==plan['source_identity'],'Installed worker source differs from frozen plan')
    arm=next(a for a in plan['arms'] if a['arm_id']==arm_id);suite=plan['suite'];prepared=plan['samples']
    require(stage in ('smoke','main') and torch.cuda.is_available(),'Explicit GPU diagnostic unavailable')
    require(torch.cuda.get_device_capability()==(12,0),'Device capability differs from approved RTX5080 profile')
    torch.set_num_threads(2);torch.manual_seed(suite['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();model=None;observers=[];storage_bytes=0;observations=[];active={}
    execution=Execution(directory.parent.parent.name,directory.name,arm_id,plan['pair']['model_id'],plan['pair']['revision'],
       arm['artifact']['artifact_identity'],arm['artifact']['recipe_identity'],prepared['tokenizer_identity'],plan['runtime_identity'],
       arm['execution_mode'],'torch.bfloat16','transformers5.14.1/torchSDPA', 'live_model_diagnostic','MEASURED','observations.jsonl').validate()
    execution['phase']=stage;execution['execution_id']=digest(execution);write(directory/'execution.json',execution)
    write(directory/'environment.json',{'pid':os.getpid(),'gpu_name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
       'torch':torch.__version__,'package_import_origin':__file__,'observer':'Original SDPA call preserved; selected Q/K/V reconstruction only',
       'no_grad':True,'eval':True,'tf32':False,'batch':1,'use_cache':False})
    raw=(directory/'observations.jsonl').open('x',encoding='utf-8')
    def emit(kind,coordinates,tensor,details):
        nonlocal storage_bytes
        row=active['row'];coords={'sample_id':row['sample_id'],'token_id_space':prepared['token_id_space'],
             'tokenizer_identity':prepared['tokenizer_identity'],'sequence_identity':row['sequence_identity'],'pass_id':active['pass_id'],**coordinates}
        prefix_length=coords['token_position'] if 'token_position' in coords else coords['query_position']+1
        coords['prefix_hash']=digest({'token_ids':row['input_ids'][:prefix_length],'token_id_space':prepared['token_id_space']})
        identity=digest({'execution':execution['execution_id'],'kind':kind,'coordinates':coords})[:28]
        numeric=None
        if tensor is not None:
            value=tensor.detach().float().cpu().contiguous();shape=list(value.shape) or [1]
            numeric=save(directory,identity,value.reshape(-1).tolist(),shape);storage_bytes+=numeric['bytes']
            require(storage_bytes<=suite['budget']['raw_bytes_per_job'],'Selected raw storage cap')
        observation=Observation(identity,kind,execution['execution_id'],row['sample_id'],'live_model_diagnostic',arm['execution_mode'],
              'MEASURED' if numeric else details.get('status','NOT_APPLICABLE'),coords,numeric,details).validate()
        raw.write(json.dumps(observation,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();observations.append(observation)
    try:
        reference=next(a['artifact']['path'] for a in plan['arms'] if a['arm_id']=='bf16')
        model,receipt=load_model(arm,reference,reference)
        write(directory/'loader.json',receipt|{'loading_seconds':time.monotonic()-started})
        model.eval();versions={name:p._version for name,p in model.named_parameters()}
        rows=prepared['rows'][:1] if stage=='smoke' else prepared['rows']
        def forward(row,pass_id,observe):
            active.update(row=row,pass_id=pass_id)
            targets=row['targets'][:1] if stage=='smoke' else row['targets'];positions=[r['logit_position'] for r in targets]
            selection=dict(row['attention'])
            if stage=='smoke':selection=selection|{'heads':selection['heads'][:1],'queries':selection['queries'][:1]}
            ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda');mask=torch.ones_like(ids)
            context=AttentionObserver(ALL_ATTENTION_FUNCTIONS,model.model.layers[selection['layer']].self_attn,selection,emit) if observe else None
            if context:observers.append(context)
            with torch.no_grad():
                selected=selected_token_forward(model,ids,mask,targets,suite['budget']['probe_gpu_buffer_bytes'],context=context,functional=F)
                hidden,chosen,logits,logprobs=(selected[key] for key in ('hidden','chosen','logits','logprobs'))
                del selected
                for j,target in enumerate(targets):
                    t=target['token_position'];argmax=int(logits[j].argmax());logprob=logprobs[j,target['target_token_id']]
                    emit('target_logprob',target,logprob.reshape(1),{'raw_argmax_token_id':argmax,'stored_nll':-float(logprob),
                         'normalization_dtype':'float32','raw_logits_dtype':str(logits.dtype),'log_base':'e','sampling':'raw pre-sampling','causal_mask':'native model SDPA; future mask checked by observer'})
                    if t in row['full_vocab_target_positions']:
                        emit('full_logits',target,logits[j],{'full_vocabulary':True,'vocabulary_size':logits.shape[-1],'raw_dtype':str(logits.dtype),
                             'lm_head_path':'native final norm model.model -> native tied lm_head on selected hidden rows only','softcap':'not present in verified Smol config'})
                del hidden,chosen,logits,logprobs,ids,mask
            require(all(p._version==versions[name] for name,p in model.named_parameters()),'Model weights changed during observation')
            torch.cuda.synchronize()
        for index,row in enumerate(rows):
            if stage=='main' and index<suite['observer']['off_samples']:
                for repeat in range(suite['observer']['off_repeats']):forward(row,'OFF'+str(repeat+1),False)
            forward(row,'ON',True)
        raw.close()
        stats={f.name:{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in Path(arm['artifact']['path']).glob('*.safetensors')}
        require(stats==arm['weight_file_stat'],'Source artifact file changed')
        result={'schema_version':1,'status':'MEASURED','evidence_kind':'live_model_diagnostic','phase':stage,'arm_id':arm_id,'execution_mode':arm['execution_mode'],
            'samples':len(rows),'main_target_positions':sum(o['kind']=='target_logprob' and o['coordinates']['pass_id']=='ON' for o in observations),
            'main_attention_rows':sum(o['kind']=='attention_row' and o['coordinates']['pass_id']=='ON' for o in observations),
            'observation_records':len(observations),'numeric_bytes':storage_bytes,'wall_seconds':time.monotonic()-started,
            'observer_contexts':len(observers),'observer_calls':sum(o.calls for o in observers),'all_hooks_interfaces_restored':all(o.restored for o in observers),
            'weight_version_unchanged':True,'source_weight_stat_unchanged':True,'allocator_max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'allocator_max_reserved_bytes':torch.cuda.max_memory_reserved(),'allocator_scope':'Whole worker including dense diagnostic load/forward; not INT4 serving memory',
            'selected_probe_buffer_estimate_upper_bytes':plan['resource_estimates']['extra_probe_gpu_buffer_upper_bytes'],
            'no_full_T_by_V_logits_created':True,'no_all_layer_attention_created':True,'kernel_observation':'native torch SDPA call; no profiler claim'}
        write(directory/'result.json',result);return result
    except BaseException as exc:
        if not (directory/'failure.json').exists():write(directory/'failure.json',{'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),'restored_contexts':[o.restored for o in observers]})
        raise
    finally:
        if not raw.closed:raw.close()
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--arm',required=True);p.add_argument('--stage',choices=('smoke','main'),required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=run(a.plan,a.arm,a.stage,a.output);print(json.dumps(result))
