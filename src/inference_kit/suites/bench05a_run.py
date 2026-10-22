"""Bounded sequential owned GPU jobs. No automatic retry or foreign PID cleanup."""
import asyncio
import datetime
import json
import os
from pathlib import Path
import shutil
import time
from ..core.io import read,write,require,digest,sha
from ..core.state import new_output,atomic,Budget
from ..adapters import owned
from ..adapters.nvml import gpu_query,gpu_monitor,proc_identity
from ..adapters.vllm import launch,runtime_snapshot,runtime_gate
from ..adapters.lifecycle_evidence import observed_shutdown
from .bench05a_plan import source_identity

DIAGNOSTIC_WORKERS=({'module':'inference_kit.probes.token_worker','token_only':True},
                    {'module':'inference_kit.probes.structure_worker','structure':True})


def raw_manifest(root):
    include=[]
    for p in sorted(Path(root).rglob('*')):
        rel=p.relative_to(root)
        if p.is_file() and not any(x in rel.parts for x in ('metrics','private','runtime-cache','worker-site','source-snapshot','__pycache__')) and p.name not in ('RAW_MANIFEST.json','manifest.json','REPORT_KO.md','HANDOFF_KO.md','summary.json','VERIFY_CPU.py'):
            include.append((str(rel),{'bytes':p.stat().st_size,'sha256':sha(p)}))
    write(Path(root)/'RAW_MANIFEST.json',{'schema_version':1,'scope':'bench05a_raw','files':dict(include),'raw_immutable':True})


def check_gpu(plan):
    if Path('/proc/meminfo').exists():
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if available<plan['suite']['budget']['minimum_cpu_available_bytes']:return {'status':'BLOCKED_CPU_MEMORY','available_bytes':available}
    if shutil.disk_usage(Path(plan['paths']['package_site']).parent).free<plan['suite']['budget']['minimum_disk_free_bytes']:return {'status':'BLOCKED_DISK_SPACE'}
    device=gpu_query(plan['gpu']['index']);apps=gpu_query(plan['gpu']['index'],True)
    if device['returncode']!=0 or apps['returncode']!=0:return {'status':'BLOCKED_GPU_OBSERVATION','device':device,'applications':apps}
    if apps.get('values'):return {'status':'BLOCKED_SHARED_GPU','device':device,'applications':apps,'meaning':'Foreign/unverified compute PID; nothing terminated'}
    require(device['values'].split(',')[1].strip()==plan['gpu']['uuid'],'GPU UUID mismatch')
    minimum_free=plan['suite']['budget'].get('minimum_gpu_free_mib',8192)
    require(type(minimum_free) is int and minimum_free>=8192,'GPU free-memory floor cannot be relaxed')
    require(float(device['values'].split(',')[4].strip())>=minimum_free,'Insufficient device free MiB for frozen model profile and headroom')
    return {'status':'READY_WITHIN_OBSERVATION_LIMITS','device':device,'applications':apps,'limitation':'NVML/local /proc cannot exhaustively exclude WSL/graphics/brief work'}


async def native_bridge(plan,arm,directory,launch_plan,process,abort):
    import aiohttp
    readiness=[]
    await owned.ready(process,launch_plan['url'],timeout=180,observations=readiness)
    write(directory/'readiness-http.json',readiness)
    with (directory/'http-responses.jsonl').open('x',encoding='utf-8') as raw:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            for sample in plan['samples']['rows']:
                require(not abort.is_set(),'Shared GPU appeared before bridge request')
                body={'model':'serving-lab','prompt':sample['input_ids'],'max_tokens':1,'temperature':0,'top_p':1,'ignore_eos':True,
                      'n':1,'echo':True,'stream':False,'logprobs':1,'prompt_logprobs':1,'add_special_tokens':False,'return_tokens_as_token_ids':True,'return_token_ids':True}
                async with session.post(launch_plan['url']+'/v1/completions',json=body) as response:
                    payload=await response.json();record={'task_id':'diagnostic_continuation_v1','sample_id':sample['sample_id'],'request_identity':digest(body),
                      'request':body,'http_status':response.status,'response':payload,'evidence_kind':'live_model_diagnostic','label':'DEPLOYED_PATH_OBSERVATION'}
                    raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');raw.flush()
                    require(response.status==200 and not payload.get('error'),'Native likelihood unavailable; no alternate scoring task')
                    from ..core.quality import likelihood_from_response
                    boundary=sample['prompt_token_boundary']
                    likelihood_from_response(payload,{'input_ids':sample['input_ids'],'context_ids':sample['input_ids'][:boundary],'continuation_ids':sample['input_ids'][boundary:]})
    write(directory/'result.json',{'status':'MEASURED','phase':'native_bridge','samples':4,'http_requests':4,'generation_tokens':4,
              'classification':'BRIDGE_MEASURED_NO_EQUIVALENCE_CLAIM','deployed_path':True,'quality_benchmark':'NOT_EVALUATED','full_logits':'UNSUPPORTED_TRANSPORT'})


def diagnostic_launch(plan,root,phase,arm,directory,worker_contract=None):
    """Explicit built-in worker choice; legacy diagnostic argv remains the default."""
    if worker_contract is not None:
        require(worker_contract in DIAGNOSTIC_WORKERS,'Unsupported owned diagnostic worker contract')
        require(phase=='main','Token-only worker has no smoke/bridge phase')
    module='inference_kit.probes.local_transformer' if worker_contract is None else worker_contract['module']
    argv=[plan['paths']['runtime_python'],'-B','-m',module,'--plan',str(root/'PLAN_FROZEN.json'),'--arm',arm['arm_id']]
    if worker_contract is None:argv+=['--stage',phase]
    argv+=['--output',str(directory)]
    return {'argv':argv,'environment':{'PYTHONPATH':plan['paths']['package_site'],'PYTHONDONTWRITEBYTECODE':'1',
        'CUDA_VISIBLE_DEVICES':str(plan['gpu']['index']),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
        'TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','DO_NOT_TRACK':'1','TOKENIZERS_PARALLELISM':'false'}}


async def launch_job(plan,root,ledger,progress,phase,arm,budget_namespace="",*,worker_contract=None):
    if worker_contract is not None:
        require(worker_contract in DIAGNOSTIC_WORKERS and phase=='main','Unsupported owned diagnostic worker contract')
    key=phase+'-'+arm['arm_id'];identifier=budget_namespace+key+'-attempt1';directory=root/'attempts'/identifier;directory.mkdir(parents=True,exist_ok=False)
    require(len(read(ledger)['jobs'])<plan['suite']['budget']['max_gpu_processes'],'GPU launch cap reached')
    pre=check_gpu(plan);write(directory/'pre-gpu.json',pre)
    if pre['status']!='READY_WITHIN_OBSERVATION_LIMITS':
        progress['jobs'][key]={'status':pre['status'],'directory':str(directory.relative_to(root))};return False
    r=read(ledger);remaining=r['cap_seconds']-r['used_seconds'];reserve=plan['suite']['budget']['cleanup_reserve_seconds']
    if remaining<=reserve+30:
        progress['jobs'][key]={'status':'BUDGET_SKIPPED','directory':str(directory.relative_to(root))};return False
    timeout=min(240 if phase=='bridge' else 180,remaining-reserve,plan.get('clock',{}).get('deadline_monotonic',float('inf'))-time.monotonic()-reserve)
    if timeout<30:
        progress['jobs'][key]={'status':'BUDGET_SKIPPED','directory':str(directory.relative_to(root))};return False
    if phase=='bridge':
        snapshot=runtime_snapshot(plan['paths']['server_python']);require(runtime_gate(snapshot) is None,'Existing native bridge runtime unsupported')
        write(directory/'runtime.json',snapshot)
        artifact=arm['artifact'];method={'name':'bf16' if arm['arm_id']=='bf16' else 'rtn_w4','format':arm['source_format'],'activation_dtype':'bfloat16','kv_dtype':'bfloat16'}
        allowed=('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja')
        receipt={'model_id':plan['pair']['model_id'],'model_revision':plan['pair']['revision'],'artifact_identity':artifact['artifact_identity'],
          'recipe_identity':digest(method),'artifact_config_sha256':artifact['config_sha256'],'tokenizer_files':{k:v for k,v in plan['samples']['tokenizer_files'].items() if k in allowed}}
        write(directory/'identity-receipt.json',receipt)
        b=plan['suite']['bridge'];profile={'context':b['context'],'max_num_seqs':1,'max_num_batched_tokens':b['max_num_batched_tokens'],'kv_tokens':b['kv_tokens'],
          'output_tokens':1,'concurrency':1,'shutdown_timeout':b['shutdown_timeout'],'cleanup_grace':b['cleanup_grace'],'chunked_prefill':True,'eager':True,'prefix_caching':False,'async_scheduling':False,'seed':plan['suite']['seed']}
        spec={'model':{'architecture':'llama','model_id':plan['pair']['model_id'],'revision':plan['pair']['revision']},'method':method,'profile':profile}
        paths={'artifact':artifact['path'],'tokenizer':artifact['tokenizer_path'],'artifact_receipt':str(directory/'identity-receipt.json'),'server_python':plan['paths']['server_python'],'output':str(directory)}
        launch_plan=launch(spec,paths,{'port':plan['port'],'gpu':plan['gpu']['index'],'gpu_uuid':plan['gpu']['uuid']})
        await owned.ensure_port_free(launch_plan['url'])
    else:
        launch_plan=diagnostic_launch(plan,root,phase,arm,directory,worker_contract)
    write(directory/'launch.json',launch_plan);budget=Budget(ledger);budget.begin(identifier,timeout+reserve)
    started=time.monotonic();process=log=None;stop=asyncio.Event();abort=asyncio.Event();owner={};status='FAILED';cleanup=None
    monitor=asyncio.create_task(gpu_monitor(stop,abort,directory,{'runtime':{'gpu':plan['gpu']['index']},'instrumentation':{'ownership_poll_seconds':2,'memory_poll_seconds':1}},owner))
    try:
        process,log=owned.start(launch_plan,directory);owner.update(parent_pid=process.pid,parent_proc_identity=proc_identity(process.pid));write(directory/'owned.json',owner)
        async def execute():
            if phase=='bridge':await native_bridge(plan,arm,directory,launch_plan,process,abort)
            else:
                while process.poll() is None:
                    require(not abort.is_set(),'Foreign/unknown GPU compute appeared; stop owned work')
                    await asyncio.sleep(.1)
                require(process.returncode==0,'Diagnostic worker failed; see immutable attempt/server.log')
            require(not abort.is_set(),'GPU ownership could not be verified')
        await asyncio.wait_for(execute(),timeout=timeout)
        status='COMPLETE'
    except BaseException as exc:
        if not (directory/'controller-failure.json').exists():write(directory/'controller-failure.json',{'status':'FAILED','error_type':type(exc).__name__,'error':str(exc)})
    finally:
        try:
            if process is not None:
                cleanup=await asyncio.to_thread(owned.stop,process,log,grace=25)
                if phase=='bridge':
                    evidence=observed_shutdown(cleanup,(directory/'server.log').read_text(encoding='utf-8'),10)
                    try:await owned.ensure_port_free(launch_plan['url']);evidence['owned_port_cleanup']='FREE'
                    except ValueError:evidence['owned_port_cleanup']='OCCUPIED';status='FAILED'
                    write(directory/'lifecycle.json',evidence)
                else:write(directory/'lifecycle.json',cleanup|{'native_main_returncode':process.returncode,'kernel_observation':'torch SDPA API call, no profiler'})
                if cleanup['owned_process_cleanup']!='CLEANED':status='FAILED'
        except BaseException as exc:
            status='FAILED';write(directory/'cleanup-failure.json',{'status':'UNRESOLVED_OWNED_RESOURCE','error':str(exc)})
        finally:
            stop.set();await monitor
            after=gpu_query(plan['gpu']['index'],True);write(directory/'post-gpu.json',after)
            if after['returncode']!=0 or after.get('values'):status='FAILED'
            budget.end(time.monotonic()-started,label=identifier,returncode=process.returncode if process else None,
                       owner_pid=process.pid if process else None,status=status,post_gpu=after)
    progress['jobs'][key]={'status':status,'directory':str(directory.relative_to(root)),'source_identity':plan['source_identity']}
    return status=='COMPLETE'


async def managed(plan_file,output):
    plan=read(Path(plan_file));frozen=dict(plan);expected=frozen.pop('plan_identity');require(digest(frozen)==expected,'Plan integrity changed')
    require(plan['status']=='CPU_PLANNED_NOT_EXECUTED','Plan has unresolved local artifacts/resources')
    require(plan['source_identity']==source_identity(),'Maintained source changed after freeze; make a new plan/attempt')
    parent=Path(plan['campaign']['path']);require(sha(parent)==plan['campaign']['sha256'],'Prior ledger changed since plan')
    require(plan['gpu_allowance_seconds']>plan['suite']['budget']['cleanup_reserve_seconds'],'No cleanup-safe GPU budget')
    root=new_output(output);write(root/'PLAN_FROZEN.json',plan);write(root/'data.json',plan['samples']);write(root/'capability.json',plan['capability'])
    chain_map={}
    for name,meta in plan['campaign']['chain'].items():
        d=root/'campaign-history'/Path(name).parent.name/'gpu_budget.json';write(d,meta['record']);chain_map[name]=str(d.relative_to(root))
    write(root/'campaign-history/bindings.json',chain_map)
    ledger=root/'budget-continuation.json';previous=plan['campaign']['used_seconds']
    atomic(ledger,{'schema_version':1,'campaign_cap_seconds':7200,'task_cap_seconds':plan['gpu_allowance_seconds'],
      'cap_seconds':previous+plan['gpu_allowance_seconds'],'used_seconds':previous,'running':None,'jobs':[],
      'inherited_from':{'path':str(parent),'sha256':plan['campaign']['sha256'],'used_seconds':previous},'task_id':'BENCH05-A'})
    progress={'schema_version':1,'jobs':{},'status':'RUNNING','bridge_status':'NOT_RUN','automatic_retries':0}
    atomic(root/'progress.json',progress)
    pre=check_gpu(plan);write(root/'gpu-preflight.json',pre)
    if pre['status']!='READY_WITHIN_OBSERVATION_LIMITS':progress['status']=pre['status'];progress['bridge_status']=pre['status']
    else:
        okay=True
        for phase in ('smoke','main','bridge'):
            if not okay:break
            if phase=='bridge' and not plan['suite']['bridge']['requested']:continue
            for arm in plan['arms']:
                okay=await launch_job(plan,root,ledger,progress,phase,arm);atomic(root/'progress.json',progress)
                if not okay:break
        progress['status']='COMPLETE' if okay else 'PARTIAL_OR_BLOCKED'
        progress['bridge_status']='BRIDGE_MEASURED_NO_EQUIVALENCE_CLAIM' if all(progress['jobs'].get('bridge-'+a['arm_id'],{}).get('status')=='COMPLETE' for a in plan['arms']) else 'BLOCKED_OR_NOT_RUN'
    atomic(root/'progress.json',progress);raw_manifest(root)
    from .bench05a_replay import analyze
    summary=analyze(root,new_output(root/'metrics'))
    measured={}
    for arm in plan['arms']:
        key='main-'+arm['arm_id'];job=progress['jobs'].get(key,{})
        if job.get('status')=='COMPLETE':
            result=read(root/job['directory']/'result.json')
            measured[arm['arm_id']]={'evidence_kind':'live_model_diagnostic','execution_mode':arm['execution_mode'],
              'features':{name:{'status':'MEASURED','evidence':job['directory']+'/observations.jsonl'} for name in ('target_logprobs','selected_full_vocab','selected_attention_rows','attention_output','output_projection')},
              'main_target_positions':result['main_target_positions'],'main_attention_rows':result['main_attention_rows']}
        else:measured[arm['arm_id']]={'features':{name:{'status':job.get('status',progress['status']),'evidence':job.get('directory')} for name in ('target_logprobs','selected_full_vocab','selected_attention_rows','attention_output','output_projection')}}
    write(root/'capability-measured.json',{'schema_version':1,'arms':measured,'scope':'Only actual execution/backend/artifact combination; no generic live GQA/old suite re-certification'})
    return {'status':progress['status'],'root':str(root),'stages':summary['stages'],'new_gpu_job_wall_seconds':read(ledger)['used_seconds']-previous}


def run(plan,output):return asyncio.run(managed(plan,output))
