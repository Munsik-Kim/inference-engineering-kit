"""Existing vLLM0.29 policy with explicit inputs; no engine SDK imported here."""
import json
import os
from pathlib import Path
from ..core.io import read,require,digest,sha


def launch(spec,paths,local):
    artifact=Path(paths['artifact']);tokenizer=Path(paths['tokenizer']);python=Path(paths['server_python'])
    require(python.is_file() and os.access(python,os.X_OK),'Missing explicit runtime Python')
    require((artifact/'config.json').is_file() and tokenizer.is_dir(),'Missing local artifact/tokenizer; downloads disabled')
    cfg=read(artifact/'config.json');require(cfg['model_type']==spec['model']['architecture'],'Model architecture identity differs')
    require('artifact_receipt' in paths,'Explicit artifact identity receipt required')
    receipt=read(Path(paths['artifact_receipt']))
    require(receipt.get('model_id')==spec['model']['model_id'] and receipt.get('model_revision')==spec['model']['revision'], 'Artifact model/revision receipt mismatch')
    require(receipt.get('recipe_identity')==digest(spec['method']) and receipt.get('artifact_identity'),'Artifact recipe/identity receipt mismatch')
    require(receipt.get('artifact_config_sha256')==sha(artifact/'config.json'),'Artifact configuration bytes differ')
    tokenizer_files=receipt.get('tokenizer_files',{})
    require(tokenizer_files and all(name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json','tokenizer.model') for name in tokenizer_files),'Explicit tokenizer file identities required')
    require(all((tokenizer/name).is_file() and sha(tokenizer/name)==identity for name,identity in tokenizer_files.items()),'Tokenizer file identity differs')
    quant=cfg.get('quantization_config');fmt=spec['method']['format']
    if fmt=='native':require(not quant,'Native precision request points to quantized artifact')
    else:
        from .tensors import check_scheme
        check_scheme(cfg)
    require(cfg.get('dtype',cfg.get('torch_dtype'))=='bfloat16','BF16 activation/native metadata required')
    p=spec['profile'];per_token=2*cfg['num_hidden_layers']*cfg['num_key_value_heads']*cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])*2
    port=local.get('port',18160);require(type(port) is int and 1024<=port<=65534,'Invalid loopback port')
    require(type(local.get('gpu')) is int and local['gpu']>=0 and local.get('gpu_uuid'),'Explicit GPU index and UUID required')
    argv=[str(python),'-B','-m','vllm.entrypoints.cli.main','serve',str(artifact),
          '--host','127.0.0.1','--port',str(port),'--served-model-name','serving-lab','--tokenizer',str(tokenizer),
          '--dtype','bfloat16','--kv-cache-dtype','bfloat16','--tensor-parallel-size','1',
          '--max-model-len',str(p['context']),'--max-num-seqs',str(p['max_num_seqs']),
          '--max-num-batched-tokens',str(p['max_num_batched_tokens']),'--kv-cache-memory-bytes',str(p['kv_tokens']*per_token),
          '--gpu-memory-utilization','0.80','--no-enable-prefix-caching',
          '--enable-chunked-prefill' if p['chunked_prefill'] else '--no-enable-chunked-prefill',
          '--no-async-scheduling','--enforce-eager','--generation-config','vllm','--stream-interval','1',
          '--attention-config','{"backend":"FLASH_ATTN"}','--seed',str(p['seed']),
          '--shutdown-timeout',str(p['shutdown_timeout'])]
    env={'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','DO_NOT_TRACK':'1',
         'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':str(local['gpu']),'VLLM_NO_USAGE_STATS':'1',
         'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_USE_V2_MODEL_RUNNER':'0','VLLM_USE_FLASHINFER_SAMPLER':'0',
         'VLLM_CACHE_ROOT':str(Path(paths['output'])/'runtime-cache')}
    return {'argv':argv,'environment':env,'url':f'http://127.0.0.1:{port}',
            'kv_bytes_per_token':per_token,'kv_bytes':p['kv_tokens']*per_token,
            'artifact_config_sha256':sha(artifact/'config.json'),'profile_identity':digest(p),
            'artifact_identity':receipt['artifact_identity'],'artifact_receipt_sha256':sha(Path(paths['artifact_receipt'])),
            'tokenizer_identity':digest(tokenizer_files),'execution_identity':digest({'artifact':receipt['artifact_identity'],'receipt':sha(Path(paths['artifact_receipt'])),'model':spec['model'],'method':spec['method'],'profile':p}),
            'support':'REFACTORED_CPU_CONTRACT_VERIFIED_GPU_NOT_RERUN'}


import subprocess
def runtime_snapshot(python):
    code = '''import importlib.metadata as m,json,sys
names=('torch','vllm','transformers','compressed-tensors','triton','aiohttp')
packages={}
for name in names:
 try:
  d=m.distribution(name);packages[name]={'version':d.version,'package_root':str(d.locate_file(''))}
 except m.PackageNotFoundError:packages[name]=None
print(json.dumps({'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'packages':packages}))
'''
    result = subprocess.run([python, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, "BLOCKED_RUNTIME: snapshot failed: " + result.stderr[:500])
    return json.loads(result.stdout)


def runtime_gate(snapshot):
    packages = snapshot["packages"]
    missing = [name for name in ("vllm", "torch", "transformers", "compressed-tensors") if packages.get(name) is None]
    if missing:
        return {"status": "BLOCKED_RUNTIME_DEPENDENCY", "missing_packages": missing,
                "runtime": snapshot, "gpu_measurement": "NOT_RUN", "actual_requests": 0,
                "reason": "Selected runtime cannot load the existing vLLM/compressed-tensors artifact; no installation or alternate environment used"}
    if packages["vllm"]["version"] != "0.29.0" or packages["compressed-tensors"]["version"] != "0.17.0":
        return {"status": "BLOCKED_RUNTIME_VERSION", "runtime": snapshot, "gpu_measurement": "NOT_RUN", "actual_requests": 0,
                "reason": "Review installed engine/API and compressed loader before changing the source-checked pin"}
    if packages["torch"]["version"].split("+")[0] != "2.13.0" or packages["transformers"]["version"] != "5.17.0":
        return {"status": "BLOCKED_RUNTIME_VERSION", "runtime": snapshot, "gpu_measurement": "NOT_RUN", "actual_requests": 0,
                "reason": "Selected runtime differs from the retained serving package tuple; compatibility review required"}
    return None

