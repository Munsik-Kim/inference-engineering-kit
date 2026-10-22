"""Explicit ARC regression plan, no network/GPU/model loading."""
from pathlib import Path
import random
import time
from ..core.io import read,write,sha,digest,require
from ..core.paths import Paths
from ..core.state import new_output
from ..tasks.arc_challenge import TASK,VERSION,SCORER,validate_sample,ArcChallengeAdapter
from ..probes.contracts import Sample,attention_coordinates
from ..probes.arc_tokens import POLICY
from ..models.capabilities import inspect_capabilities
from .bench05a_plan import source_identity,ledger_evidence
from .task_environment import inspect_artifacts,resources

def validate_data(data,prepared):
    rows=data['rows'];require(data['task_id']==TASK and data['task_version']==VERSION and data['scorer_version']==SCORER,'ARC task/scorer mismatch')
    require(len(rows)==68 and len({r['sample_id'] for r in rows})==68 and digest(rows)==data['rows_identity'],'ARC frozen data identity/count')
    require(sum(r['usage_partition']=='SMOKE' for r in rows)==4 and sum(r['usage_partition']=='REGRESSION' for r in rows)==64,'ARC split coverage')
    for row in rows:validate_sample(row)
    require(prepared['kind']=='prepared_arc_regression_v1' and prepared['preparation_id']==POLICY and prepared['data_identity']==data['rows_identity'],'ARC preparation mismatch')
    require(digest(prepared['rows'])==prepared['rows_identity'] and len(prepared['rows'])<=4,'ARC probe identity/count')
    by={r['sample_id']:r for r in rows};cfg=prepared['model_config']
    require(len(prepared['native_boundary_checks'])==sum(len(r['choices']) for r in rows),'Native boundary verification coverage')
    for row in prepared['rows']:
        Sample(**{k:row[k] for k in Sample.__dataclass_fields__}).validate()
        src=by[row['sample_id']];req=src['requests'][src['gold']]
        require(row['input_ids']==req['input_ids'] and row['prompt_token_boundary']==len(req['context_ids']),'Native/probe tokens differ')
        require(row['prompt']+row['reference_text']==req['context']+req['continuation'] and digest(row['input_ids'])==row['sequence_identity'],'ARC probe prefix changed')
        require(row['reference_source']=='dataset_answer_key_choice' and len(row['input_ids'])<=512,'ARC reference/length')
        require(0<len(row['targets'])<=16 and len(row['full_vocab_target_positions'])<=4,'ARC target cap')
        for t in row['targets']:require(t['token_position']>=len(req['context_ids']) and t['logit_position']==t['token_position']-1 and t['target_token_id']==row['input_ids'][t['token_position']],'ARC shift/target mismatch')
        require(row['attention']==attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],row['prompt_token_boundary'],[{'token_position':len(row['input_ids'])-1}]),'ARC attention policy changed')
    return rows

def plan(suite_file,models_file,local_file,output):
    suite_file,models_file,local_file=[Path(p).absolute() for p in (suite_file,models_file,local_file)]
    suite,pair,local=read(suite_file),read(models_file),read(local_file)
    require(suite['suite_id']=='BENCH05-B3' and suite['request_kind']=='choice_loglikelihood' and suite['task_id']==TASK and suite['task_version']==VERSION,'Unsupported B3 suite/task')
    require(suite['serving']=={'context':2048,'max_num_seqs':1,'max_num_batched_tokens':2048,'kv_tokens':4096,'shutdown_timeout':10,'cleanup_grace':25,'chunked_prefill':True,'eager':True,'prefix_caching':False,'async_scheduling':False},'B3 serving contract changed')
    b=suite['budget'];require(0<b['task_gpu_wall_seconds']<=600 and b['max_gpu_processes']<=6 and b['automatic_retries']==0 and b['cleanup_reserve_seconds']>=120,'B3 unbounded budget')
    require(suite['observer']=={'off_samples':0,'off_repeats':0},'No extra observer controls in B3')
    require(pair['model_id']=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and pair['revision']=='31b70e2e869a7173562077fd711b654946d38674' and [a['arm_id'] for a in pair['arms']]==['bf16','rtn'],'Only pinned Smol pair supported')
    require([a['execution_mode'] for a in pair['arms']]==['INSTRUMENTED_PATH','DECODED_WEIGHT_DIAGNOSTIC'],'B3 diagnostic labels changed; no fallback')
    paths=Paths.load(suite_file,local_file);resolved={k:str(paths.resolve(local[k])) for k in ('arc_data','artifact_receipt','prepared_samples','runtime_metadata','runtime_python','server_python','package_site','prior_ledger','campaign_root','task_clock')}
    data,prepared=read(Path(resolved['arc_data'])),read(Path(resolved['prepared_samples']));rows=validate_data(data,prepared)
    runtime=read(Path(resolved['runtime_metadata']));arms=inspect_artifacts(pair,read(Path(resolved['artifact_receipt'])),prepared)
    history=ledger_evidence(resolved['prior_ledger'],resolved['campaign_root']);origin=local['task_origin_used_seconds'];require(origin<=history['used_seconds'],'Campaign origin exceeds current use')
    allowance=min(b['task_gpu_wall_seconds']-(history['used_seconds']-origin),7200-history['used_seconds']);require(allowance>b['cleanup_reserve_seconds'],'Insufficient cleanup-safe budget')
    clock=read(Path(resolved['task_clock']));require(clock['deadline_monotonic']==clock['start_monotonic']+3600,'Task clock changed')
    resource=resources(suite_file.parent,b);healthy=resource['ready'] and all(a['artifact_available'] for a in arms) and all(Path(resolved[k]).is_file() for k in ('runtime_python','server_python'))
    cfg=prepared['model_config'];maximum=max((len(r['input_ids']) for r in prepared['rows']),default=0);head=cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
    extra=max((len(r['targets']) for r in prepared['rows']),default=0)*cfg['vocab_size']*12+maximum*head*4*12;require(extra<=b['probe_gpu_buffer_bytes'],'Probe buffer bound')
    counts={'unique_regression_documents':64,'smoke_documents':4,'regression_posts_total':2*sum(len(r['choices']) for r in rows if r['usage_partition']=='REGRESSION'),
      'smoke_posts_total':2*sum(len(r['choices']) for r in rows if r['usage_partition']=='SMOKE'),'total_posts':2*sum(len(r['choices']) for r in rows),'audit_generation_token_cap':2*sum(len(r['choices']) for r in rows),
      'serving_jobs':2,'optional_diagnostic_jobs':2,'max_gpu_jobs':b['max_gpu_processes'],'probe_target_positions_per_arm':sum(len(r['targets']) for r in prepared['rows']),
      'probe_attention_rows_per_arm':sum(len(r['attention']['heads'])*len(r['attention']['queries']) for r in prepared['rows'])}
    order=['bf16','rtn'];random.Random(suite['seed']).shuffle(order)
    result={'schema_version':1,'task':'BENCH05-B3','task_id':TASK,'task_version':VERSION,'request_kind':'choice_loglikelihood','scorer':SCORER,
      'evidence_kind':'planned_live_arc_regression','status':'CPU_PLANNED_NOT_EXECUTED' if healthy else 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT','gpu_actions':0,'model_loads':0,'network_actions':0,
      'suite':suite,'pair':pair,'data':data,'samples':prepared,'arms':arms,'arm_order':order,'paths':resolved,'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'port':local['port'],
      'runtime':runtime,'runtime_identity':digest(runtime),'serving_runtime_identity':local['serving_runtime_identity'],'capability':inspect_capabilities(pair['model_id'],cfg,runtime,all(a['artifact_available'] for a in arms)),
      'campaign':history,'task_origin_used_seconds':origin,'gpu_allowance_seconds':allowance,'clock':{k:clock[k] for k in ('start_utc','start_monotonic','deadline_monotonic','no_new_gpu_after')},
      'counts':counts,'resource_estimates':resource|{'maximum_sequence_tokens':maximum,'extra_probe_gpu_buffer_upper_bytes':extra},
      'source_identity':source_identity(),'scorer_identity':sha(Path(__import__('inference_kit.tasks.arc_challenge',fromlist=['x']).__file__)),
      'input_file_sha256':{k:sha(Path(resolved[k])) for k in ('arc_data','prepared_samples','artifact_receipt','runtime_metadata')},'spec_files':{str(p):sha(p) for p in (suite_file,models_file)},
      'scope':'REGRESSION_REUSE_NOT_NEW_TEST','collector_output_tokens':1,'task_contract':{'adapter':ArcChallengeAdapter().metric_specs(),'transport':'non-stream prompt-token echo logprobs; audit1 excluded',
        'prompt_policy':'original prepared chat-rendered prompt; apply template zero additional times','native_full_vocabulary':'NOT_COLLECTED','character_normalization':'original choice string length, excludes delimiter'}}
    result['plan_identity']=digest(result);out=new_output(output);write(out/'PLAN_FROZEN.json',result);return result

def check_frozen(plan):
    frozen=dict(plan);identity=frozen.pop('plan_identity');require(digest(frozen)==identity and plan['task']=='BENCH05-B3','Frozen B3 plan changed')
    require(plan['status']=='CPU_PLANNED_NOT_EXECUTED','B3 resources/artifacts unresolved')
    require(resources(Path(plan['paths']['arc_data']).parent,plan['suite']['budget'])['ready'],'B3 CPU/disk resource gate before GPU/budget')
    require(plan['source_identity']==source_identity(),'B3 source/scorer changed after freeze')
    require(Path(__file__).absolute().parents[2]==Path(plan['paths']['package_site']).absolute(),'B3 controller must use installed package_site')
    for k,h in plan['input_file_sha256'].items():require(sha(Path(plan['paths'][k]))==h,'B3 input changed: '+k)
    for p,h in plan['spec_files'].items():require(sha(Path(p))==h,'B3 spec changed')
    for arm in plan['arms']:
        current={p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in Path(arm['artifact']['path']).glob('*.safetensors')}
        require(current==arm['weight_file_stat'],'B3 artifact file stat changed; no model rehash required')
    require(sha(Path(plan['campaign']['path']))==plan['campaign']['sha256'],'B3 parent ledger changed')
    require(time.monotonic()<plan['clock']['no_new_gpu_after'],'B3 GPU cutoff')
    validate_data(plan['data'],plan['samples'])
