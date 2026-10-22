"""CPU-only frozen closed-context task plan, using existing artifact/resource gates."""
from pathlib import Path
import random
import time
from ..core.io import read, write, sha, digest, require
from ..core.paths import Paths
from ..core.state import new_output
from ..tasks import context_evidence as task
from ..probes.contracts import Sample, attention_coordinates
from ..probes.retrieval_tokens import POLICY, BOUNDS, validate_content_spans
from .bench05a_plan import source_identity, ledger_evidence
from .task_environment import inspect_artifacts, resources

COLLECTOR = {'request_kind':'generate_until','stream':True,'max_new_tokens':64,
             'ignore_eos':False,'audit_generation_tokens':0}
SERVING = {'context':2048,'max_num_seqs':1,'max_num_batched_tokens':2048,'kv_tokens':4096,
           'shutdown_timeout':10,'cleanup_grace':25,'chunked_prefill':True,'eager':True,
           'prefix_caching':False,'async_scheduling':False}


def validate_data(data, prepared):
    require(data['task_id']==task.TASK and data['task_version']==task.VERSION and data['scorer']==task.SCORER, 'B4 data task/scorer mismatch')
    rows=data['rows'];require(len(rows)==28 and len({r['sample_id'] for r in rows})==28 and digest(rows)==data['rows_identity'], 'B4 authored identity/count')
    require(sum(s['split']=='SMOKE' for s in rows)==4 and sum(s['split']=='EVAL_LOCKED' for s in rows)==24,'B4 split count')
    for sample in rows:task.verify_gold(sample)
    evaluation=[s for s in rows if s['split']=='EVAL_LOCKED']
    require(sum(s['gold']['answerable'] for s in evaluation)==18,'B4 answerability count')
    require(all(sum(s['gold']['evidence']==f'P{i:02}' for s in evaluation)==3 for i in range(1,7)), 'B4 evidence position balance')
    require(prepared['kind']=='prepared_retrieval_samples_v1' and prepared['preparation_id']==POLICY and prepared['bounds']==BOUNDS, 'B4 preparation policy/bounds')
    require(prepared['task_rows_identity']==data['rows_identity'] and digest(prepared['generation_rows'])==prepared['generation_rows_identity'] and digest(prepared['rows'])==prepared['rows_identity'], 'B4 prepared identity')
    by={s['sample_id']:s for s in rows};generation=prepared['generation_rows']
    require(len(generation)==28 and {s['sample_id'] for s in generation}==set(by),'B4 generation coverage')
    require([r['sample_id'] for r in prepared['rows']]==task.select_probes(rows),'B4 frozen probe selection changed')
    require(len(prepared['rows'])==4 and all(p in generation for p in prepared['rows']),'B4 probes must use identical generation input')
    cfg=prepared['model_config']
    for row in generation:
        sample=by[row['sample_id']]
        Sample(**{k:row[k] for k in Sample.__dataclass_fields__}).validate()
        require(row['task_id']==task.TASK and row['task_version']==task.VERSION and row['reference_source']=='task_author_gold_answer_evidence_v1','B4 diagnostic identity')
        require(row['sample_identity']==sample['sample_identity'] and row['task_prompt_hash']==digest(task.render_prompt(sample)), 'B4 source/prompt changed')
        reference=task.build_probe_sample(sample)
        require(row['reference_text']==reference['reference_text'] and row['reference_hash']==reference['reference_hash'],'B4 author reference changed')
        require(row['tokenization']['chat_template_applied']==1 and row['tokenization']['policy']==POLICY,'B4 chat policy')
        require(row['full_text']==row['prompt']+row['reference_text'] and row['sequence_identity']==digest(row['input_ids']),'B4 joint identity')
        require(0<len(row['generation_input_ids'])<=512 and 0<len(row['reference_separate_ids'])<=64 and len(row['input_ids'])<=576,'B4 token limit; no runtime truncation')
        require(len(row['generation_input_ids'])+64<=SERVING['context'],'B4 serving context overflow')
        require(0<len(row['targets'])<=16 and len(row['full_vocab_target_positions'])<=4,'B4 target bounds')
        for t in row['targets']:
            require(t['token_position']>=row['prompt_token_boundary'] and t['logit_position']==t['token_position']-1 and t['target_token_id']==row['input_ids'][t['token_position']],'B4 target shift/boundary')
        last=max(i for i,v in enumerate(row['loss_mask']) if v)
        require(row['attention']==attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],row['prompt_token_boundary'],[{'token_position':last}]),'B4 attention coordinates changed')
        validate_content_spans(sample,row)
    return rows


def plan(suite_file,models_file,local_file,output):
    suite_file,models_file,local_file=[Path(p).absolute() for p in (suite_file,models_file,local_file)]
    suite,pair,local=read(suite_file),read(models_file),read(local_file)
    require(suite['suite_id']=='BENCH05-B4' and suite['task_id']==task.TASK and suite['task_version']==task.VERSION and suite['request_kind']=='generate_until','Unsupported B4 task')
    require(suite['serving']==SERVING and suite['generation']=={'profile':'natural_generation_v1','max_new_tokens':64,'ignore_eos':False,'stream':True,'temperature':0,'top_p':1},'B4 serving/generation contract')
    b=suite['budget'];require(0<b['task_gpu_wall_seconds']<=600 and b['cleanup_reserve_seconds']>=120 and b['max_gpu_processes']<=6 and b['automatic_retries']==0,'B4 budget contract')
    require(suite['observer']=={'off_samples':0,'off_repeats':0},'B4 adds no observer control jobs')
    require(pair['model_id']=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and pair['revision']=='31b70e2e869a7173562077fd711b654946d38674' and [a['arm_id'] for a in pair['arms']]==['bf16','rtn'],'B4 only pinned Smol pair')
    paths=Paths.load(suite_file,local_file)
    keys=('authored_data','prepared_samples','artifact_receipt','runtime_metadata','runtime_python','server_python','package_site','prior_ledger','campaign_root','task_clock')
    resolved={k:str(paths.resolve(local[k])) for k in keys}
    data,prepared=read(Path(resolved['authored_data'])),read(Path(resolved['prepared_samples']));rows=validate_data(data,prepared)
    runtime=read(Path(resolved['runtime_metadata']));arms=inspect_artifacts(pair,read(Path(resolved['artifact_receipt'])),prepared)
    history=ledger_evidence(resolved['prior_ledger'],resolved['campaign_root']);origin=local['task_origin_used_seconds']
    require(origin<=history['used_seconds'],'B4 campaign origin invalid')
    allowance=min(600-(history['used_seconds']-origin),7200-history['used_seconds']);require(allowance>b['cleanup_reserve_seconds'],'B4 cleanup-safe budget unavailable')
    clock=read(Path(resolved['task_clock']));require(clock['deadline_monotonic']==clock['start_monotonic']+3600,'B4 task clock invalid')
    resource=resources(suite_file.parent,b);healthy=resource['ready'] and all(a['artifact_available'] for a in arms) and all(Path(resolved[k]).is_file() for k in ('runtime_python','server_python'))
    cfg=prepared['model_config'];maximum=max(len(r['input_ids']) for r in prepared['rows']);head=cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
    buffer=max(len(r['targets']) for r in prepared['rows'])*cfg['vocab_size']*12+maximum*head*4*12
    require(buffer<=b['probe_gpu_buffer_bytes'],'B4 probe buffer bound')
    counts={'eval_documents':24,'smoke_documents':4,'total_posts':56,'eval_posts_total':48,'smoke_posts_total':8,'max_output_tokens':3584,
      'manual_code_fix_max_posts_all_attempts':112,'manual_code_fix_max_output_tokens':7168,'serving_jobs':2,'optional_diagnostic_jobs':2,'max_gpu_jobs':6,
      'probe_target_positions_per_arm':sum(len(r['targets']) for r in prepared['rows']),
      'probe_attention_rows_per_arm':sum(len(r['attention']['heads'])*len(r['attention']['queries']) for r in prepared['rows']),
      'prompt_tokens_per_arm':sum(len(r['generation_input_ids']) for r in prepared['generation_rows'])}
    order=['bf16','rtn'];random.Random(suite['seed']).shuffle(order)
    requests=[];rng=random.Random(suite['request_seed'])
    for split in ('SMOKE','EVAL_LOCKED'):
        ids=[s['sample_id'] for s in rows if s['split']==split];rng.shuffle(ids);requests.extend(ids)
    result={'schema_version':1,'task':'BENCH05-B4','task_id':task.TASK,'task_version':task.VERSION,'scorer':task.SCORER,'request_kind':'generate_until',
      'evidence_kind':'planned_live_context_retrieval','scope':'SYNTHETIC_DEV_LOCKED','status':'CPU_PLANNED_NOT_EXECUTED' if healthy else 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT',
      'gpu_actions':0,'model_loads':0,'network_actions':0,'suite':suite,'pair':pair,'data':data,'samples':prepared,'arms':arms,'arm_order':order,'request_order':requests,
      'paths':resolved,'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'port':local['port'],'runtime':runtime,'runtime_identity':digest(runtime),
      'serving_runtime_identity':local['serving_runtime_identity'],'campaign':history,'task_origin_used_seconds':origin,'gpu_allowance_seconds':allowance,
      'clock':{k:clock[k] for k in ('start_utc','start_monotonic','deadline_monotonic','no_new_gpu_after')},'counts':counts,
      'resource_estimates':resource|{'maximum_sequence_tokens':maximum,'extra_probe_gpu_buffer_upper_bytes':buffer},'collector_contract':COLLECTOR,'collector_output_tokens':64,
      'source_identity':source_identity(),'scorer_identity':sha(Path(task.__file__)),
      'input_file_sha256':{k:sha(Path(resolved[k])) for k in ('authored_data','prepared_samples','artifact_receipt','runtime_metadata')},
      'spec_files':{str(p):sha(p) for p in (suite_file,models_file)},'metric_specs':task.ContextEvidenceAdapter().metric_specs()}
    result['plan_identity']=digest(result);out=new_output(output);write(out/'PLAN_FROZEN.json',result);return result


def check_frozen(plan):
    frozen=dict(plan);identity=frozen.pop('plan_identity');require(digest(frozen)==identity and plan['task']=='BENCH05-B4','B4 frozen identity')
    require(plan['status']=='CPU_PLANNED_NOT_EXECUTED','B4 resources/artifacts unresolved')
    require(resources(Path(plan['paths']['authored_data']).parent,plan['suite']['budget'])['ready'],'B4 CPU/disk gate before GPU/query/budget')
    require(plan['source_identity']==source_identity(),'B4 source/scorer changed after freeze')
    require(Path(__file__).absolute().parents[2]==Path(plan['paths']['package_site']).absolute(),'B4 controller must use installed package_site')
    for k,h in plan['input_file_sha256'].items():require(sha(Path(plan['paths'][k]))==h,'B4 input changed: '+k)
    for p,h in plan['spec_files'].items():require(sha(Path(p))==h,'B4 spec changed')
    require(sha(Path(plan['campaign']['path']))==plan['campaign']['sha256'],'B4 parent ledger changed')
    for arm in plan['arms']:
        now={p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in Path(arm['artifact']['path']).glob('*.safetensors')}
        require(now==arm['weight_file_stat'],'B4 source artifact file stat changed')
    require(time.monotonic()<plan['clock']['no_new_gpu_after'],'B4 GPU launch cutoff')
    validate_data(plan['data'],plan['samples'])
