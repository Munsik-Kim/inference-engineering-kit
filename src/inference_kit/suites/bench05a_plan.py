"""CPU-only frozen plan for the first token/attention diagnostic, no SDK imports."""
from dataclasses import fields as dataclass_fields
from importlib.resources import files
import os
from pathlib import Path
import shutil
from ..core.io import read,write,require,sha,digest
from ..core.paths import Paths
from ..core.state import new_output,cumulative
from ..probes.contracts import Sample,target_positions,attention_coordinates
from ..models.capabilities import inspect_capabilities
from ..adapters.tensors import check_scheme


def source_identity():
    package=files('inference_kit')
    return {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts}


def validate_samples(prepared):
    require(prepared['kind']=='prepared_diagnostic_samples' and prepared['schema_version']==1,'Unsupported prepared samples')
    rows=prepared['rows'];require(len(rows)==4 and len({r['sample_id'] for r in rows})==4,'Four unique samples required')
    require(digest(rows)==prepared['rows_identity'],'Prepared row identity differs')
    cfg=prepared['model_config'];sample_fields={f.name for f in dataclass_fields(Sample)}
    for row in rows:
        Sample(**{k:row[k] for k in sample_fields}).validate()
        ids=row['input_ids'];boundary=row['prompt_token_boundary']
        require(ids and len(ids)<=128 and all(type(i) is int and 0<=i<prepared['vocabulary_size'] for i in ids),'Invalid token IDs/length')
        require(row['sequence_identity']==digest(ids) and row['full_text_hash']==digest(row['prompt']+row['reference_text']),'Input sequence/text changed')
        require(row['targets']==target_positions(ids,boundary,row['special_token_ids'],16),'Next-token target/logit shift differs')
        require(row['attention_mask']==[1]*len(ids),'This profile is unpadded batch1 only')
        reference=[t for t in range(boundary,len(ids)) if ids[t] not in row['special_token_ids']]
        require(row['loss_mask']==[int(i in reference) for i in range(len(ids))],'Loss mask special/boundary differs')
        require(row['attention']==attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],boundary,[{'token_position':reference[-1]}]),'Frozen attention coordinates differ')
        require(row['tokenization']['tokenizer_identity']==prepared['tokenizer_identity'] and row['tokenization']['token_id_space']==prepared['token_id_space'],'Tokenizer/prefix space differs')
        positions=row['full_vocab_target_positions'];require(positions==sorted(set(positions)) and 1<=len(positions)<=4 and positions[0]==row['targets'][0]['token_position'] and set(positions)<={p['token_position'] for p in row['targets']},'Selected full-vocabulary coordinates invalid')
        require(len(row['offsets_codepoints'])==len(row['offsets_utf8_bytes'])==len(ids),'Offset coverage differs')
    return rows


def ledger_evidence(path,root):
    path,root=Path(path),Path(root);records={}
    def resolver(reference):
        p=Path(reference)
        require(p.resolve().is_relative_to(root.resolve()) and p.is_file(),'Ledger outside explicit campaign root')
        records[str(p)]={'sha256':sha(p),'record':read(p)};return p
    resolver(str(path));used=cumulative(read(path),resolver)
    return {'path':str(path),'sha256':sha(path),'used_seconds':used,'chain':records}


def plan(suite_file,models_file,local_file,output):
    suite_file,models_file,local_file=map(lambda p:Path(p).absolute(),(suite_file,models_file,local_file))
    suite,pair,local=read(suite_file),read(models_file),read(local_file)
    require(suite['schema_version']==pair['schema_version']==1 and suite['suite_id']=='BENCH05-A','Unsupported bench suite')
    require(suite['task_id']=='diagnostic_continuation_v1' and suite['task_version']=='1','Unregistered task')
    require(suite['tokenization']=={'policy':'plain_completion_joint_no_special_tokens','boundary_rule':'first joint token whose end offset crosses prompt end; score whole boundary token','max_sequence_tokens':128,'max_reference_positions':16,'full_vocab_positions':4},'Unregistered token policy')
    require(suite['attention']['backend']=='sdpa' and suite['attention']['layer_rule']=='middle' and suite['attention']['head_rule']=='first_and_middle' and suite['attention']['query_rule']=='last_prompt_and_before_last_reference' and suite['attention']['output_attentions'] is False,'Unregistered attention profile')
    require(suite['budget']['automatic_retries']==0 and suite['budget']['max_gpu_processes']<=8 and suite['budget']['task_gpu_wall_seconds']<=1200 and suite['budget']['cleanup_reserve_seconds']>=120,'Unbounded diagnostic budget')
    require(suite['observer']['off_repeats']==2 and suite['observer']['off_samples']==2,'Observer control profile changed')
    require([a['arm_id'] for a in pair['arms']]==['bf16','rtn'] and [a['execution_mode'] for a in pair['arms']]==['INSTRUMENTED_PATH','DECODED_WEIGHT_DIAGNOSTIC'],'Pair mode must be explicit; no automatic precision fallback')
    paths=Paths.load(suite_file,local_file)
    resolved={k:str(paths.resolve(local[k])) for k in ('artifact_receipt','prepared_samples','runtime_metadata','runtime_python','server_python','package_site','prior_ledger','campaign_root')}
    require(Path(resolved['runtime_python']).is_file(),'Missing explicit diagnostic runtime Python')
    prepared=read(Path(resolved['prepared_samples']));runtime=read(Path(resolved['runtime_metadata']));rows=validate_samples(prepared)
    require(prepared['authored_sha256']==sha(paths.resolve(suite['samples'])),'Authored samples changed before freeze')
    authored=read(paths.resolve(suite['samples']))
    require(len(authored)==len(rows) and all(all(a[k]==r[k] for k in ('sample_id','source_id','prompt','reference_text','language')) for a,r in zip(authored,rows)),'Prepared text/order differs from authored samples')
    references=read(Path(resolved['artifact_receipt']));arms=[];available=True
    for declaration in pair['arms']:
        ref=references['arms'][declaration['receipt_arm']]
        require(ref['model_id']==pair['model_id'] and ref['model_revision']==pair['revision'],'Artifact model/revision mismatch')
        p=Path(ref['path']);exists=p.is_dir() and (p/'config.json').is_file()
        available=available and exists
        if exists:
            require(sha(p/'config.json')==ref['config_sha256'],'Artifact config identity differs')
            cfg=read(p/'config.json')
            if declaration['arm_id']=='rtn':
                check_scheme(cfg);require(sha(Path(ref['manifest_path']))==ref['manifest_sha256'],'RTN manifest changed')
                require(ref['recipe']['algorithm']=='RTN' and ref['recipe']['ignore']==['lm_head'],'Not the original RTN/exclusion recipe')
            for key in ('model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size','intermediate_size','vocab_size','tie_word_embeddings','attention_bias','mlp_bias'):
                require(cfg[key]==prepared['model_config'][key],'Model structure differs: '+key)
            require(all((Path(ref['tokenizer_path'])/name).is_file() and sha(Path(ref['tokenizer_path'])/name)==identity for name,identity in prepared['tokenizer_files'].items()),'Pinned tokenizer files changed')
            stats={f.name:{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in p.glob('*.safetensors')}
            require(stats,'No local safetensors; downloads disabled')
        else:stats={}
        arms.append(declaration|{'artifact':ref,'artifact_available':exists,'weight_file_stat':stats})
    history=ledger_evidence(resolved['prior_ledger'],resolved['campaign_root'])
    remaining=suite['budget']['campaign_cap_seconds']-history['used_seconds'];cap=min(suite['budget']['task_gpu_wall_seconds'],remaining)
    require(cap>suite['budget']['cleanup_reserve_seconds'],'Campaign has insufficient cleanup-safe allowance')
    cfg=prepared['model_config'];max_positions=max(len(r['targets']) for r in rows)
    head_dim=cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
    extra=max_positions*cfg['vocab_size']*4*3+128*head_dim*4*12
    require(extra<suite['budget']['probe_gpu_buffer_bytes'],'Probe buffer bound exceeded')
    disk=shutil.disk_usage(Path(output).absolute().parent if Path(output).absolute().parent.exists() else suite_file.parent)
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))} if Path('/proc/meminfo').exists() else {}
    capability=inspect_capabilities(pair['model_id'],cfg,runtime,available)
    require(all(v['status']!='UNSUPPORTED' for v in capability['features'].values()),'Installed runtime/structure not supported by the initial adapter')
    healthy=available and disk.free>=suite['budget']['minimum_disk_free_bytes'] and mem.get('MemAvailable',0)>=suite['budget']['minimum_cpu_available_bytes']
    result={'schema_version':1,'task':'BENCH05-A','status':'CPU_PLANNED_NOT_EXECUTED' if healthy else 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT',
      'gpu_actions':0,'model_loads':0,'network_actions':0,'suite':suite,'pair':pair,'arms':arms,'samples':prepared,
      'runtime':runtime,'runtime_identity':digest(runtime),'paths':resolved,'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'port':local['port'],
      'capability':capability,'campaign':history,'gpu_allowance_seconds':cap,'campaign_remaining_before_seconds':remaining,
      'counts':{'samples':4,'main_reference_positions_per_arm':sum(len(r['targets']) for r in rows),'main_attention_rows_per_arm':sum(len(r['attention']['heads'])*len(r['attention']['queries']) for r in rows),
                'main_full_vocab_rows_per_arm':sum(len(r['full_vocab_target_positions']) for r in rows),'smoke_jobs':2,'main_jobs':2,'optional_bridge_servers':2,'observer_off_repeats':2,'observer_off_samples':2},
      'resource_estimates':{'extra_probe_gpu_buffer_upper_bytes':extra,'cpu_available_bytes':mem.get('MemAvailable'),'disk_free_bytes':disk.free,
                            'bf16_weight_file_bytes':sum(v['bytes'] for v in arms[0]['weight_file_stat'].values()),'raw_storage_cap_per_job':suite['budget']['raw_bytes_per_job']},
      'source_identity':source_identity(),'suite_sha256':sha(suite_file),'pair_sha256':sha(models_file),'artifact_receipt_sha256':sha(Path(resolved['artifact_receipt'])),
      'treatment_axes':['weight_artifact'],'held_equal':['model_revision','tokenizer','input_token_sequence','prefix','selected_coordinates','backend','dtype_policy','observer_policy'],
      'native_bridge':'Optional C1 echo likelihood; full cross-backend KL is unavailable','scope':'research_diagnostic_only; production/accuracy/speed NOT_EVALUATED'}
    result['plan_identity']=digest(result);out=new_output(output);write(out/'PLAN_FROZEN.json',result);write(out/'capability.json',capability)
    return result
