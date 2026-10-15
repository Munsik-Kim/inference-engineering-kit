"""Explicit model, recipe, workload, runtime and profile planning; never loads a model."""
from pathlib import Path
import re
from .io import read,require,digest,sha
from .paths import Paths,masked


def make_plan(spec_file,local_file=None):
    spec_file=Path(spec_file).expanduser().absolute();s=read(spec_file)
    require(set(s)=={'schema_version','backend','model','method','profile','paths','budget'},'Unknown/missing spec fields')
    require(s.get('schema_version')==1,'Unsupported kit spec schema')
    require(s['backend'] in ('recorded','vllm'),'Unsupported backend; no automatic fallback')
    require(re.fullmatch(r'[0-9a-f]{40}',s['model']['revision'] or ''),'Pinned model revision required')
    require(s['method']['activation_dtype']==s['method']['kv_dtype']=='bfloat16','Unsupported activation/KV dtype')
    require(s['method']['format'] in ('native','compressed-tensors/pack-quantized'),'Unsupported artifact format')
    if s['backend']=='vllm':
        require(s['model']['revision']!='0'*40 and not s['model']['model_id'].startswith('REPLACE'), 'Replace example model identity before planning a GPU path')
        require(s['method']['name'] in ('bf16','rtn_w4','gptq_w4','clip_rtn_w4','packed_composition'),'Unregistered method recipe; no precision fallback')
        require((s['method']['name']=='bf16')==(s['method']['format']=='native'),'Method/format mismatch')
    p=s['profile']
    require(set(p)=={'context','max_num_seqs','max_num_batched_tokens','kv_tokens','output_tokens','concurrency',
                    'shutdown_timeout','cleanup_grace','chunked_prefill','eager','prefix_caching','async_scheduling','seed'},'Unknown/missing profile fields')
    require(p['eager'] is True and p['prefix_caching'] is False and p['async_scheduling'] is False
            and type(p['chunked_prefill']) is bool,'Unregistered eager/cache/scheduler policy')
    for key in ('context','max_num_seqs','max_num_batched_tokens','kv_tokens','output_tokens','concurrency'):
        require(type(p[key]) is int and p[key]>0,'Invalid profile '+key)
    require(p['concurrency']<=p['max_num_seqs'] and p['output_tokens']<p['context'],'Profile capacity mismatch')
    require(p['shutdown_timeout']>0 and p['cleanup_grace']>p['shutdown_timeout'],'Invalid cleanup budgets')
    require(s['budget']['automatic_retries']==0 and s['budget']['wall_seconds']>s['budget']['cleanup_reserve_seconds']>=60,'Explicit bounded budget required')
    require(s['budget']['cleanup_reserve_seconds']>=2*p['cleanup_grace']+5,'Cleanup grace exceeds reserve')
    paths=Paths.load(spec_file,local_file);local=read(Path(local_file)) if local_file else {}
    result={'schema_version':1,'status':'CPU_PLANNED_NOT_EXECUTED','gpu_inference':False,'downloads':False,
            'spec':masked(s),'spec_identity':digest(s),'spec_sha256':sha(spec_file),
            'runtime_support':'RECORDED_CPU' if s['backend']=='recorded' else 'REFACTORED_CPU_CONTRACT_VERIFIED_GPU_NOT_RERUN',
            'profile_identity':digest(p),'model_identity':digest(s['model']),'recipe_identity':digest(s['method']),
            'paths':{},'local_snapshot':masked(local)}
    for key,value in s['paths'].items():result['paths'][key]=str(paths.resolve(value))
    for key in ('server_python',):
        if key in local:result['paths'][key]=str(paths.resolve(local[key]))
    if s['backend']=='vllm':
        require(s['model']['architecture'] in ('qwen3','llama'),'Unsupported transformer structure adapter')
        require(local.get('server_python'),'Missing explicit runtime Python')
        from ..adapters.vllm import launch
        result['launch']=launch(s,result['paths'],local)
    else:require(s['model']['architecture']=='synthetic','Recorded demo architecture must be explicit')
    return result
