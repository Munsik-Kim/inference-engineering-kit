"""Shared local artifact and resource inspection; never loads models or queries GPU."""
from pathlib import Path
import shutil
from ..core.io import read,sha,require
from ..adapters.tensors import check_scheme

def inspect_artifacts(pair,refs,prepared):
    arms=[]
    for declaration in pair['arms']:
        ref=refs['arms'][declaration['receipt_arm']];p=Path(ref['path']);exists=(p/'config.json').is_file();stats={}
        require(ref['model_id']==pair['model_id'] and ref['model_revision']==pair['revision'],'Artifact revision mismatch')
        if exists:
            cfg=read(p/'config.json');require(sha(p/'config.json')==ref['config_sha256'],'Artifact config changed')
            for k in ('model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size','vocab_size'):
                require(cfg[k]==prepared['model_config'][k],'Structure mismatch: '+k)
            if pair['model_id']=='Qwen/Qwen3-4B-Instruct-2507':
                from ..models.capabilities import qwen_serving_capability
                evidence=declaration|{'artifact':ref,'model_config':cfg,'quantization_config':cfg.get('quantization_config')}
                require(qwen_serving_capability(pair,cfg,evidence), 'UNSUPPORTED_CAPABILITY: Qwen artifact/packing receipt')
                if cfg.get('quantization_config'):check_scheme(cfg)
            elif declaration['arm_id']=='rtn':
                check_scheme(cfg);require(ref['recipe']['algorithm']=='RTN','Original RTN required')
            else:require(not cfg.get('quantization_config'),'BF16 artifact quantized')
            require(all((Path(ref['tokenizer_path'])/n).is_file() and sha(Path(ref['tokenizer_path'])/n)==h for n,h in prepared['tokenizer_files'].items()),'Tokenizer changed')
            stats={f.name:{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in p.glob('*.safetensors')};require(stats,'Local weights absent')
        arms.append(declaration|{'artifact':ref,'artifact_available':exists,'weight_file_stat':stats})
    return arms

def resources(directory,budget):
    mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if s.startswith('MemAvailable:')}
    free=shutil.disk_usage(directory).free;available=mem.get('MemAvailable',0)
    return {'cpu_available_bytes':available,'disk_free_bytes':free,'ready':available>=budget['minimum_cpu_available_bytes'] and free>=budget['minimum_disk_free_bytes']}
