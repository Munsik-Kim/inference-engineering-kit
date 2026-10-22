"""Configured CPU plan and bounded owned token-only capture; no task scoring."""
import asyncio
from pathlib import Path
import time
from ..core.io import read,write,sha,digest,require
from ..core.paths import Paths
from ..core.state import new_output,atomic
from ..probes.token_inputs import prepare_inputs,select_inputs,validate_probe_inputs,validate_spec
from .bench05a_plan import source_identity,ledger_evidence
from .task_environment import inspect_artifacts,resources
from .bench05a_run import check_gpu,launch_job
from .bench05b1_run import seal
from ..models.capabilities import inspect_capabilities

WORKER={'module':'inference_kit.probes.token_worker','token_only':True}


def _record(spec_file,inputs_file,models_file,local_file,*,prepared=None,spec_override=None,task_id='BENCH05-C2',worker_contract=WORKER):
    """Shared frozen context; only explicit built-in recipes supply prepared rows."""
    require(task_id in ('BENCH05-C2','BENCH05-D1'),'Unsupported configured recipe')
    spec_file,inputs_file,models_file,local_file=[Path(p).expanduser().absolute() for p in (spec_file,inputs_file,models_file,local_file)]
    spec=validate_spec(read(spec_file) if spec_override is None else spec_override);raw=read(inputs_file)
    if prepared is None:
        inputs=validate_probe_inputs(raw,verify_sources=True) if raw.get('kind')=='common_probe_inputs_v1' else prepare_inputs(inputs_file)
        selected=select_inputs(inputs,spec)
    else:
        selected=validate_probe_inputs(prepared,verify_sources=True)
        require(selected==raw and selected['spec']==spec,'Prepared configured inputs changed or reselected')
    pair=read(models_file);local=read(local_file);paths=Paths.load(spec_file,local_file)
    require(pair['model_id']=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and pair['revision']=='31b70e2e869a7173562077fd711b654946d38674','Only prepared pinned Smol diagnostic pair supported')
    require([a['arm_id'] for a in pair['arms']]==['bf16','rtn'] and [a['execution_mode'] for a in pair['arms']]==['INSTRUMENTED_PATH','DECODED_WEIGHT_DIAGNOSTIC'],'Explicit native/decoded pair required')
    keys=('artifact_receipt','runtime_metadata','runtime_python','package_site','prior_ledger','campaign_root','task_clock')
    resolved={key:str(paths.resolve(local[key])) for key in keys}
    b=spec['budget'];require(b['minimum_cpu_available_bytes']>=10*1024**3 and b['minimum_disk_free_bytes']>=4*1024**3,'Live CPU RAM/disk floor must not be lowered')
    require(b['probe_gpu_buffer_bytes']<=256*1024**2 and b['raw_bytes_total']<=256*1024**2,'C2 raw/buffer upper bound')
    require(len({row['sample_id'] for row in selected['rows']})==len(selected['rows']),'Configured observation sample IDs must be globally unique')
    arms=inspect_artifacts(pair,read(Path(resolved['artifact_receipt'])),selected)
    identities={'bf16':'0c668d5b8e51322d63898007d3d4a4ea488f21168208c3f3fc0d0f9643ac371b','rtn':'c8debb3188403ab9d332149969e6e0f9b7b689dc06e77e882fde07a9cb8da761'}
    require(all(a['artifact']['artifact_identity']==identities[a['arm_id']] for a in arms),'Original BF16/R00 artifact identity required')
    history=ledger_evidence(resolved['prior_ledger'],resolved['campaign_root']);origin=local['task_origin_used_seconds']
    require(type(origin) in (int,float) and 0<=origin<=history['used_seconds'],'C2 task campaign origin invalid')
    allowance=min(b['task_gpu_wall_seconds']-(history['used_seconds']-origin),7200-history['used_seconds'])
    require(allowance>b['cleanup_reserve_seconds'],'C2 cleanup-safe allowance unavailable')
    clock=read(Path(resolved['task_clock']));require(clock['deadline_monotonic']==clock['start_monotonic']+3600 and clock['no_new_gpu_after']==clock['start_monotonic']+3000,'C2 requires a fresh task clock')
    runtime=read(Path(resolved['runtime_metadata']));capability=inspect_capabilities(pair['model_id'],selected['model_config'],runtime,all(a['artifact_available'] for a in arms))
    for key,feature in capability['features'].items():
        if key not in ('target_logprobs','selected_full_vocab'):feature.update(status='NOT_REQUESTED',evidence='Configured token-only profile; no attention observer')
    require(all(capability['features'][k]['status']!='UNSUPPORTED' for k in ('target_logprobs','selected_full_vocab')),'Existing diagnostic runtime/structure unsupported')
    resource=resources(spec_file.parent,b);healthy=resource['ready'] and all(a['artifact_available'] for a in arms) and Path(resolved['runtime_python']).is_file()
    task_jobs=sum(len(entry['record'].get('jobs',[])) for entry in history['chain'].values() if entry['record'].get('task_id')==task_id and entry['record'].get('task_origin_used_seconds')==origin)
    require(task_jobs<b['max_gpu_processes'],'C2 total job cap already exhausted')
    counts=selected['counts']|{'planned_gpu_jobs':2,'prior_task_gpu_jobs':task_jobs,'forwards_per_arm':len(selected['rows']),
      'forward_calls_total':2*len(selected['rows']),'target_observations_total':2*selected['counts']['target_positions_per_arm'],
      'full_logit_observations_total':2*selected['counts']['full_vocab_positions_per_arm'],'serving_POSTs':0,'tokenizer_loads':0}
    suite={'seed':spec['seed'],'budget':b|{'max_gpu_processes':b['max_gpu_processes']-task_jobs,'raw_bytes_per_job':b['raw_bytes_total']},'observer':{'off_samples':0,'off_repeats':0}}
    result={'schema_version':1,'task':task_id,'raw_format':'token_probe_raw_v1','status':'CPU_PLANNED_NOT_EXECUTED' if healthy else 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT',
      'scope':'engineering_regression_reuse; no new quality test','evidence_kind':'planned_live_token_probe','probe_spec':spec,'samples':selected,'pair':pair,'arms':arms,'arm_order':['bf16','rtn'],
      'suite':suite,'worker_contract':worker_contract,'paths':resolved,'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'capability':capability,
      'runtime':runtime,'runtime_identity':digest(runtime),'campaign':history,'task_origin_used_seconds':origin,'gpu_allowance_seconds':allowance,
      'clock':{key:clock[key] for key in ('start_utc','start_monotonic','deadline_monotonic','no_new_gpu_after')},'counts':counts,
      'resource_estimates':resource|selected['resource_estimates'],'source_identity':source_identity(),
      'input_files':{str(p):sha(p) for p in (spec_file,inputs_file,models_file,Path(resolved['artifact_receipt']),Path(resolved['runtime_metadata']))},
      'treatment_axes':['weight_artifact'],'held_equal':['prepared_input_ids','prefix','task_reference','selected_coordinates','tokenizer','vocabulary','diagnostic_backend','activation_dtype','normalization'],
      'GPU_actions':0,'gpu_actions':0,'model_loads':0,'tokenizer_loads':0,'network_actions':0,'attention_observations':0,
      'native_bridge':'NOT_REQUESTED','quality':'NOT_EVALUATED_NEW','deployment_customer':'NOT_VALIDATED'}
    return result


def plan(spec_file,inputs_file,models_file,local_file,output):
    result=_record(spec_file,inputs_file,models_file,local_file)
    result['plan_identity']=digest(result);out=new_output(output);write(out/'PLAN_FROZEN.json',result);write(out/'PROBE_INPUTS_FROZEN.json',result['samples']);write(out/'SELECTION_RECEIPT.json',result['samples']['selection_receipt']);write(out/'PROBE_SPEC.json',result['probe_spec'])
    return result


def check_frozen(plan,*,task_id='BENCH05-C2',worker_contract=WORKER):
    value=dict(plan);identity=value.pop('plan_identity');require(digest(value)==identity,'Configured plan identity changed')
    require(plan['task']==task_id and plan['worker_contract']==worker_contract,'Unsupported configured capture')
    require(plan['status']=='CPU_PLANNED_NOT_EXECUTED','Plan has blocked resource/artifact dependencies')
    require(source_identity()==plan['source_identity'],'Source changed after token capture freeze; new attempt required')
    validate_spec(plan['probe_spec']);validate_probe_inputs(plan['samples'],verify_sources=True)
    require(plan['samples']['spec']==plan['probe_spec'],'ProbeSpec and selected inputs differ')
    for name,identity in plan['input_files'].items():require(sha(Path(name))==identity,'Configured input file changed after freeze')
    require(sha(Path(plan['campaign']['path']))==plan['campaign']['sha256'],'Prior continuation ledger changed')
    require(time.monotonic()<plan['clock']['no_new_gpu_after'] and time.monotonic()<plan['clock']['deadline_monotonic']-plan['suite']['budget']['cleanup_reserve_seconds'],'Task clock expired or cleanup reserve unavailable')
    for arm in plan['arms']:
        actual={p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in Path(arm['artifact']['path']).glob('*.safetensors')}
        require(actual==arm['weight_file_stat'],'Original model file stat changed since freeze')
    return True


async def managed(plan_file,output,*,task_id='BENCH05-C2',worker_contract=WORKER,raw_scope='token_probe_raw_v1',extra_validator=None):
    require((task_id,raw_scope) in (('BENCH05-C2','token_probe_raw_v1'),('BENCH05-D1','structure_probe_raw_v1')),'Unsupported managed capture recipe')
    plan=read(Path(plan_file));check_frozen(plan,task_id=task_id,worker_contract=worker_contract)
    if extra_validator is not None:extra_validator(plan)
    # Live resource gate precedes GPU queries, worker start and budget mutation.
    resource=resources(Path(output).absolute().parent,plan['suite']['budget'])
    pre=check_gpu(plan) if resource['ready'] else {'status':'BLOCKED_CPU_OR_DISK_RESOURCE','resources':resource}
    root=new_output(output);write(root/'PLAN_FROZEN.json',plan);write(root/'gpu-preflight.json',pre)
    import inference_kit,sys
    write(root/'controller-environment.json',{'python':sys.executable,'package_version':inference_kit.__version__,'package_import_origin':inference_kit.__file__,'source_identity':source_identity()})
    progress={'schema_version':1,'task':task_id,'status':pre['status'],'jobs':{},'automatic_retries':0}
    ledger=root/'budget-continuation.json'
    if pre['status']!='READY_WITHIN_OBSERVATION_LIMITS':
        atomic(root/'progress.json',progress);seal(root,scope=raw_scope);return progress
    previous=plan['campaign']['used_seconds']
    atomic(ledger,{'schema_version':1,'task_id':task_id,'campaign_cap_seconds':7200,'cap_seconds':previous+plan['gpu_allowance_seconds'],'used_seconds':previous,
      'task_origin_used_seconds':plan['task_origin_used_seconds'],'running':None,'jobs':[],
      'inherited_from':{'path':plan['campaign']['path'],'sha256':plan['campaign']['sha256'],'used_seconds':previous}})
    progress['status']='RUNNING';atomic(root/'progress.json',progress)
    try:
        for name in plan['arm_order']:
            if time.monotonic()>=plan['clock']['no_new_gpu_after']:
                progress['status']='BUDGET_SKIPPED';break
            arm=next(a for a in plan['arms'] if a['arm_id']==name)
            okay=await launch_job(plan,root,ledger,progress,'main',arm,worker_contract=worker_contract)
            atomic(root/'progress.json',progress)
            actual=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            require(actual<=plan['probe_spec']['budget']['raw_bytes_total'],'Configured total raw bytes exceeded')
            if not okay:progress['status']='PARTIAL_OR_BLOCKED';break
        else:progress['status']='COMPLETE'
    except BaseException as exc:
        progress['status']='FAILED_OR_PARTIAL';write(root/'controller-failure.json',{'type':type(exc).__name__,'error':str(exc)});raise
    finally:
        atomic(root/'progress.json',progress);seal(root,scope=raw_scope)
    return {'status':progress['status'],'root':str(root),'new_gpu_job_wall_seconds':read(ledger)['used_seconds']-previous,'jobs':progress['jobs'],'attention_rows':0 if task_id=='BENCH05-C2' else plan['structure_selection']['counts']['attention_rows_per_arm']*sum(j['status']=='COMPLETE' for j in progress['jobs'].values())}


def run(plan,output):return asyncio.run(managed(plan,output))
