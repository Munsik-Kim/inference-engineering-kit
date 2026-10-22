"""B1 CPU freeze; bounded task generation and A collector inputs have separate policies."""
from dataclasses import fields
from pathlib import Path
import random
import shutil
import time
from ..core.io import read,write,require,sha,digest
from ..core.paths import Paths
from ..core.state import new_output
from ..tasks.tech_record import TASK,VERSION,SCORER,CASES,verify_gold,canonical_reference,select_probes,render_prompt
from ..probes.contracts import Sample,attention_coordinates
from ..probes.task_tokens import POLICY,SELECTION,field_targets
from ..models.capabilities import inspect_capabilities
from ..adapters.tensors import check_scheme
from .bench05a_plan import source_identity,ledger_evidence


def validate_data(data):
    require(data['schema_version']==1 and data['task_id']==TASK and data['task_version']==VERSION and data['scorer']==SCORER,'B1 task/scorer identity differs')
    rows=data['rows'];require(len(rows)==28 and len({r['sample_id'] for r in rows})==28,'B1 needs SMOKE4 and DEV24 unique samples')
    require(digest(rows)==data['rows_identity'],'Task rows changed')
    dev=[r for r in rows if r['split']=='DEV_LOCKED'];smoke=[r for r in rows if r['split']=='SMOKE']
    require(len(dev)==24 and len(smoke)==4 and len({r['document_hash'] for r in rows})==28,'Wrong split/document coverage')
    for case in CASES:
        for language in ('en','ko'):require(sum(r['case_type']==case and r['language']==language for r in dev)==2,'DEV design changed')
    for row in rows:
        identity=dict(row);recorded=identity.pop('sample_identity');require(digest(identity)==recorded,'Task sample identity changed')
        require(row['task_id']==TASK and row['task_version']==VERSION,'Wrong row task');verify_gold(row)
    return rows


def validate_prepared(prepared,data):
    require(prepared['kind']=='prepared_task_samples_v1' and prepared['preparation_id']==POLICY,'Wrong B1 preparation; A is not an alias')
    require(prepared['task_id']==TASK and prepared['task_version']==VERSION,'Prepared task mismatch')
    authored={r['sample_id']:r for r in validate_data(data)};rows=prepared['generation_rows'];cfg=prepared['model_config']
    require(len(rows)==28 and digest(rows)==prepared['generation_rows_identity'] and {r['sample_id'] for r in rows}==set(authored),'Generation preparation changed')
    for row in rows:
        Sample(**{f.name:row[f.name] for f in fields(Sample)}).validate();source=authored[row['sample_id']]
        require(row['sample_identity']==source['sample_identity'] and row['reference_text']==canonical_reference(source)['text'],'Task/gold diagnostic pairing mismatch')
        require(row['value_spans']==canonical_reference(source)['value_spans'],'B1 value spans must cover values, not JSON keys')
        require(row['task_prompt_hash']==digest(render_prompt(source)) and row['prompt'].count(render_prompt(source))==1 and row['reference_text'] not in row['prompt'],'Task prompt or gold isolation differs')
        require(row['reference_source']=='task_author_gold_json' and row['tokenization']['policy']==POLICY and row['tokenization']['chat_template_applied']==1,'Chat/reference policy mismatch')
        require(row['tokenization']['tokenizer_identity']==prepared['tokenizer_identity'] and row['tokenization']['token_id_space']==prepared['token_id_space'],'Tokenizer identity differs')
        ids=row['input_ids'];boundary=row['prompt_token_boundary']
        require(row['generation_input_ids']==row['prompt_separate_ids'],'Serving prompt token IDs changed')
        require(len(row['offsets_codepoints'])==len(row['offsets_utf8_bytes'])==len(ids),'B1 offset coverage changed')
        require(0<boundary<len(ids)<=512 and len(row['generation_input_ids'])<=384 and len(row['reference_separate_ids'])<=128,'B1 token bounds')
        require(all(type(v) is int and 0<=v<prepared['vocabulary_size'] for v in ids+row['generation_input_ids']),'Invalid token IDs')
        require(row['sequence_identity']==digest(ids) and row['full_text_hash']==digest(row['prompt']+row['reference_text']),'Token/text identity changed')
        require(row['reference_hash']==digest(row['reference_text']) and row['prompt_hash']==digest(row['prompt']),'Prompt/reference identity changed')
        targets,coverage=field_targets(ids,boundary,row['offsets_codepoints'],len(row['prompt']),row['value_spans'],row['special_token_ids'])
        require(row['targets']==targets and row['field_token_coverage']==coverage and row['selection_policy']==SELECTION,'Value span selection changed')
        valid=[i for i in range(boundary,len(ids)) if ids[i] not in row['special_token_ids']]
        require(row['attention_mask']==[1]*len(ids) and row['loss_mask']==[int(i in valid) for i in range(len(ids))],'Causal/padding/loss boundary differs')
        require(row['attention']==attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],boundary,[{'token_position':valid[-1]}]),'A attention policy changed')
        require(set(row['full_vocab_target_positions'])<={p['token_position'] for p in targets} and len(row['full_vocab_target_positions'])<=4,'Full vocabulary selection invalid')
    selected=select_probes(data['rows']);require([r['sample_id'] for r in prepared['rows']]==selected and digest(prepared['rows'])==prepared['rows_identity'],'Probe choice/order changed')
    require(prepared['rows']==[next(r for r in rows if r['sample_id']==sid) for sid in selected],'Probe/generation content mismatch')
    return rows


def plan(suite_file,models_file,local_file,output):
    suite_file,models_file,local_file=map(lambda p:Path(p).absolute(),(suite_file,models_file,local_file))
    suite,pair,local=read(suite_file),read(models_file),read(local_file)
    require(suite['suite_id']=='BENCH05-B1' and suite['schema_version']==1 and suite['task_id']==TASK and suite['task_version']==VERSION,'Unsupported B1 profile')
    require(suite['generation']=={'profile':'natural_generation_v1','max_tokens':192,'concurrency':1,'smoke_samples':4,'dev_samples':24,'warmup_posts':0},'Natural generation contract changed')
    b=suite['budget'];require(0<b['task_gpu_wall_seconds']<=600 and b['max_gpu_processes']<=6 and b['automatic_retries']==0 and b['cleanup_reserve_seconds']>=60,'B1 budget must be bounded')
    require(suite['observer']=={'off_samples':0,'off_repeats':0},'B1 does not add OFF/bridge GPU jobs')
    require(suite['serving']=={'context':2048,'max_num_seqs':1,'max_num_batched_tokens':2048,'kv_tokens':4096,'shutdown_timeout':10,'cleanup_grace':25,'chunked_prefill':True,'eager':True,'prefix_caching':False,'async_scheduling':False},'B1 serving profile changed')
    require(pair['model_id']=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and pair['revision']=='31b70e2e869a7173562077fd711b654946d38674','Only pinned Smol pair initially supported')
    require([a['arm_id'] for a in pair['arms']]==['bf16','rtn'] and [a['execution_mode'] for a in pair['arms']]==['INSTRUMENTED_PATH','DECODED_WEIGHT_DIAGNOSTIC'],'No automatic precision fallback')
    paths=Paths.load(suite_file,local_file);resolved={k:str(paths.resolve(local[k])) for k in ('authored_data','artifact_receipt','prepared_samples','runtime_metadata','runtime_python','server_python','package_site','prior_ledger','campaign_root','task_clock')}
    data=read(Path(resolved['authored_data']));prepared=read(Path(resolved['prepared_samples']));rows=validate_prepared(prepared,data)
    require(prepared['authored_sha256']==sha(Path(resolved['authored_data'])),'Authored data bytes changed')
    runtime=read(Path(resolved['runtime_metadata']));references=read(Path(resolved['artifact_receipt']));arms=[];available=True
    for declaration in pair['arms']:
        ref=references['arms'][declaration['receipt_arm']];p=Path(ref['path']);exists=(p/'config.json').is_file()
        require(ref['model_id']==pair['model_id'] and ref['model_revision']==pair['revision'],'Artifact revision mismatch')
        available=available and exists
        stats={}
        if exists:
            cfg=read(p/'config.json');require(sha(p/'config.json')==ref['config_sha256'],'Artifact config differs')
            for key in ('model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','hidden_size','vocab_size'):
                require(cfg[key]==prepared['model_config'][key],'Structure mismatch: '+key)
            if declaration['arm_id']=='rtn':check_scheme(cfg);require(ref['recipe']['algorithm']=='RTN','Wrong original RTN recipe')
            else:require(not cfg.get('quantization_config'),'BF16 points to quantized weights')
            require(all((Path(ref['tokenizer_path'])/name).is_file() and sha(Path(ref['tokenizer_path'])/name)==value for name,value in prepared['tokenizer_files'].items()),'Tokenizer changed')
            stats={f.name:{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in p.glob('*.safetensors')};require(stats,'No local weights')
        arms.append(declaration|{'artifact':ref,'artifact_available':exists,'weight_file_stat':stats})
    history=ledger_evidence(resolved['prior_ledger'],resolved['campaign_root']);clock=read(Path(resolved['task_clock']))
    origin_used=local['task_origin_used_seconds'];require(history['used_seconds']>=origin_used,'Campaign origin changed')
    allowance=min(b['task_gpu_wall_seconds']-(history['used_seconds']-origin_used),7200-history['used_seconds'])
    require(allowance>b['cleanup_reserve_seconds'],'Insufficient cleanup-safe GPU allowance')
    require(clock['deadline_monotonic']==clock['start_monotonic']+3600,'Task deadline changed')
    cpu={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if line.startswith(('MemAvailable:','MemTotal:'))} if Path('/proc/meminfo').exists() else {}
    disk=shutil.disk_usage(suite_file.parent);runtime_ok=all(Path(resolved[k]).is_file() for k in ('runtime_python','server_python'))
    healthy=available and runtime_ok and cpu.get('MemAvailable',0)>=b['minimum_cpu_available_bytes'] and disk.free>=b['minimum_disk_free_bytes']
    capability=inspect_capabilities(pair['model_id'],prepared['model_config'],runtime,available)
    require(not any(v['status']=='UNSUPPORTED' for v in capability['features'].values()),'Diagnostic runtime/structure unsupported')
    c=prepared['model_config'];head=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);max_len=max(len(r['input_ids']) for r in prepared['rows']);max_positions=max(len(r['targets']) for r in prepared['rows'])
    extra=max_positions*c['vocab_size']*12+max_len*head*4*12;require(extra<=b['probe_gpu_buffer_bytes'],'Probe buffer upper bound')
    order=['bf16','rtn'];random.Random(suite['seed']).shuffle(order)
    result={'schema_version':1,'task':'BENCH05-B1','task_id':TASK,'task_version':VERSION,'evidence_kind':'planned_live_task_pipeline','status':'CPU_PLANNED_NOT_EXECUTED' if healthy else 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT',
      'gpu_actions':0,'model_loads':0,'network_actions':0,'suite':suite,'pair':pair,'data':data,'samples':prepared,'arms':arms,'arm_order':order,
      'paths':resolved,'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'port':local['port'],'runtime':runtime,'runtime_identity':digest(runtime),'capability':capability,
      'campaign':history,'task_origin_used_seconds':origin_used,'gpu_allowance_seconds':allowance,'clock':{k:clock[k] for k in ('start_utc','start_monotonic','deadline_monotonic','no_new_gpu_after')},
      'counts':{'smoke':4,'dev':24,'natural_requests_per_arm':28,'natural_requests_total':56,'warmup_posts':0,'serving_jobs':2,'diagnostic_jobs':2,'max_gpu_jobs':6,
                'maximum_natural_generated_tokens':56*192,'probe_samples':4,'probe_target_positions_per_arm':sum(len(r['targets']) for r in prepared['rows']),'probe_attention_rows_per_arm':sum(len(r['attention']['heads'])*len(r['attention']['queries']) for r in prepared['rows'])},
      'resource_estimates':{'cpu_available_bytes':cpu.get('MemAvailable'),'disk_free_bytes':disk.free,'extra_probe_gpu_buffer_upper_bytes':extra,'maximum_sequence_tokens':max_len},
      'source_identity':source_identity(),'input_file_sha256':{k:sha(Path(resolved[k])) for k in ('authored_data','prepared_samples','artifact_receipt','runtime_metadata')},
      'suite_sha256':sha(suite_file),'pair_sha256':sha(models_file),'scorer':SCORER,'selection_policy':SELECTION,'scope':'synthetic_dev_pilot; research/production/customer NOT_VALIDATED',
      'retained_bridge':'BENCH05-A dev2 measured diagnostic/deployment differences; no new bridge or equivalence claim'}
    result['plan_identity']=digest(result);out=new_output(output);write(out/'PLAN_FROZEN.json',result);return result


def check_frozen(plan):
    frozen=dict(plan);expected=frozen.pop('plan_identity');require(digest(frozen)==expected,'Frozen B1 plan changed')
    require(plan['status']=='CPU_PLANNED_NOT_EXECUTED','Plan has unresolved local artifacts/resources')
    require(plan['source_identity']==source_identity(),'Source changed after B1 freeze; create new wheel/plan/attempt')
    require(Path(__file__).absolute().parents[2]==Path(plan['paths']['package_site']).absolute(),'B1 run must import the explicitly installed package_site')
    for key,value in plan['input_file_sha256'].items():require(sha(Path(plan['paths'][key]))==value,'B1 input changed: '+key)
    require(sha(Path(plan['campaign']['path']))==plan['campaign']['sha256'],'Prior ledger changed')
    require(time.monotonic()<plan['clock']['no_new_gpu_after'],'No new GPU job after task cutoff')
    validate_prepared(plan['samples'],plan['data'])
