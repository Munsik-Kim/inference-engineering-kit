"""B1 orchestration: two natural-generation servers and two existing diagnostic workers."""
import asyncio
import json
from pathlib import Path
import time
import sys
import re
from ..core.io import read,write,require,digest
from ..core.state import new_output,atomic,Budget
from ..adapters import owned,http
from ..adapters.protocol import Adapter
from ..adapters.vllm import launch,runtime_snapshot,runtime_gate
from ..adapters.nvml import gpu_monitor,gpu_query,proc_identity
from ..adapters.lifecycle_evidence import observed_shutdown
from ..adapters.run import guarded
from ..tasks.tech_record import TASK,VERSION,score
from .bench05a_run import check_gpu,launch_job
from .bench05b1_plan import check_frozen


def seal(root,scope="bench05b1_raw"):
    from ..core.io import sha
    names={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*'))
           if p.is_file() and 'runtime-cache' not in p.relative_to(root).parts and p.name not in ('RAW_MANIFEST.json',)}
    write(root/'RAW_MANIFEST.json',{'schema_version':1,'scope':scope,'files':names,'immutable':True})


def normalize_collector_contract(plan,request_driver=None,explicit=None):
    """Bind collector wire semantics without using driver presence as its identity.

    The omitted contract retains the frozen B1/B2 and B3 call conventions.
    New injected collectors must supply their actual wire contract explicitly.
    """
    if isinstance(explicit,dict) and explicit.get('request_kind')=='typed_mixed_v1':
        require(request_driver is not None,'Mixed collector requires an explicit typed driver')
        require(set(explicit)=={'request_kind','stream','max_new_tokens','ignore_eos','audit_generation_tokens','contracts'},'Mixed collector contract keys differ')
        require(explicit['stream']==explicit['ignore_eos']==explicit['audit_generation_tokens']=='per_request','Mixed wire flags must be per_request')
        contracts=explicit['contracts']
        require(isinstance(contracts,list) and len(contracts)>1 and all(isinstance(v,dict) and v.get('request_kind')!='typed_mixed_v1' for v in contracts),'Mixed collector requires finite nonnested contracts')
        values=[normalize_collector_contract({},request_driver,entry) for entry in contracts]
        require(len({digest(v) for v in values})==len(values) and type(explicit['max_new_tokens']) is int and
                explicit['max_new_tokens']==max(v['max_new_tokens'] for v in values),'Mixed contract duplicate/token cap mismatch')
        return dict(explicit,contracts=values)
    legacy={'request_kind':'generate_until','stream':True,'max_new_tokens':192,
            'ignore_eos':False,'audit_generation_tokens':0} if request_driver is None else {
            'request_kind':'choice_loglikelihood','stream':False,'max_new_tokens':1,
            'ignore_eos':True,'audit_generation_tokens':1}
    value=dict(legacy if explicit is None else explicit)
    require(set(value)==set(legacy),'Collector contract keys differ')
    require(value['request_kind'] in ('generate_until','choice_loglikelihood'),'Unsupported collector request kind')
    require(type(value['stream']) is bool and type(value['ignore_eos']) is bool,'Collector stream/EOS flags must be bool')
    require(type(value['max_new_tokens']) is int and value['max_new_tokens']>0,'Collector output token bound must be positive int')
    require(type(value['audit_generation_tokens']) is int,'Collector audit token count must be int')
    if value['request_kind']=='generate_until':
        require(value['stream'] and not value['ignore_eos'] and value['audit_generation_tokens']==0,'Natural collector must stream, honor EOS, and exclude audit generation')
    else:
        require(not value['stream'] and value['ignore_eos'] and value['max_new_tokens']==value['audit_generation_tokens']==1,'Likelihood collector requires non-stream audit1 contract')
    if request_driver is None:require(value==legacy,'The frozen default B1 collector requires natural192; supply an explicit driver for another policy')
    if 'request_kind' in plan:require(plan['request_kind']==value['request_kind'],'Plan/collector request kind mismatch')
    if 'collector_output_tokens' in plan:require(plan['collector_output_tokens']==value['max_new_tokens'],'Plan/collector output bound mismatch')
    return value


def collector_metadata(contract):
    result={'collector_contract':dict(contract),'request_kind':contract['request_kind'],'stream':contract['stream'],'ignore_eos':contract['ignore_eos']}
    if contract['request_kind']=='typed_mixed_v1':
        return result|{'wire_contracts':contract['contracts'],'max_new_tokens_upper':contract['max_new_tokens']}
    if contract['request_kind']=='generate_until':result['natural_max_tokens']=contract['max_new_tokens']
    else:result['audit_generation_tokens']=contract['audit_generation_tokens']
    return result


def validate_collector_records(records,contract):
    """Validate the actual captured payload, independently of metadata labels."""
    if contract['request_kind']=='typed_mixed_v1':
        require(records,'Mixed collector produced no request records')
        for row in records:
            matches=[v for v in contract['contracts'] if v['request_kind']==row.get('request_kind') and
                     v['max_new_tokens']==row.get('request_body',{}).get('max_tokens')]
            require(len(matches)==1,'Mixed record has undeclared/ambiguous wire contract')
            validate_collector_records([row],matches[0])
        return
    for row in records:
        body=row.get('request_body',{})
        require(type(body.get('stream')) is bool and body['stream']==contract['stream'],'Collector/wire stream mismatch')
        require(type(body.get('ignore_eos')) is bool and body['ignore_eos']==contract['ignore_eos'],'Collector/wire EOS mismatch')
        require(type(body.get('max_tokens')) is int and body['max_tokens']==contract['max_new_tokens'],'Collector/wire token bound mismatch')
        if 'request_kind' in row:require(row['request_kind']==contract['request_kind'],'Collector/wire request kind mismatch')


def server_launch(plan,arm,directory,collector_contract=None):
    artifact=arm['artifact'];method={'name':'bf16' if arm['arm_id']=='bf16' else 'rtn_w4','format':arm['source_format'],'activation_dtype':'bfloat16','kv_dtype':'bfloat16'}
    architecture='llama'
    if plan.get('binding_profile')=='prepared_qwen_serve_v1':
        from ..models.capabilities import qwen_serving_capability
        cfg=read(Path(artifact['path'])/'config.json')
        require(qwen_serving_capability(plan['pair'],cfg,arm), 'UNSUPPORTED_CAPABILITY: Qwen serving artifact')
        architecture=cfg['model_type'];method=method|{'name':'bf16' if arm['source_format']=='native' else 'gptq_w4'}
    receipt={'model_id':plan['pair']['model_id'],'model_revision':plan['pair']['revision'],'artifact_identity':artifact['artifact_identity'],
      'recipe_identity':digest(method),'artifact_config_sha256':artifact['config_sha256'],
      'tokenizer_files':{k:v for k,v in plan['samples']['tokenizer_files'].items() if k in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja')}}
    write(directory/'identity-receipt.json',receipt)
    output_tokens=collector_contract['max_new_tokens'] if collector_contract is not None else plan.get('collector_output_tokens',192)
    profile=plan['suite']['serving']|{'output_tokens':output_tokens,'concurrency':1,'seed':plan['suite']['seed']}
    spec={'model':{'architecture':architecture,'model_id':plan['pair']['model_id'],'revision':plan['pair']['revision']},'method':method,'profile':profile}
    return launch(spec,{'artifact':artifact['path'],'tokenizer':artifact['tokenizer_path'],'artifact_receipt':str(directory/'identity-receipt.json'),
                       'server_python':plan['paths']['server_python'],'output':str(directory)},
                  {'port':plan['port'],'gpu':plan['gpu']['index'],'gpu_uuid':plan['gpu']['uuid']})


async def serving_job(plan,root,ledger,progress,arm,score_function=score,request_driver=None,collector_contract=None,*,job_key=None):
    contract=normalize_collector_contract(plan,request_driver,collector_contract)
    # A caller may explicitly namespace multiple owned serving units for one arm.
    # The retained wrappers keep their original key and ledger meaning by default.
    key='serve-'+arm['arm_id'] if job_key is None else job_key
    require(isinstance(key,str) and key and all(c.isalnum() or c in '-_' for c in key),'Unsafe serving execution unit key')
    directory=root/'attempts'/(key+'-attempt1');directory.mkdir(parents=True,exist_ok=False)
    require(len(read(ledger)['jobs'])<plan['suite']['budget']['max_gpu_processes'],'GPU launch limit')
    pre=check_gpu(plan);write(directory/'pre-gpu.json',pre)
    if pre['status']!='READY_WITHIN_OBSERVATION_LIMITS':
        progress['jobs'][key]={'status':pre['status'],'directory':str(directory.relative_to(root))};return False
    current=read(ledger);remaining=current['cap_seconds']-current['used_seconds'];reserve=plan['suite']['budget']['cleanup_reserve_seconds']
    timeout=min(plan['suite'].get('serving_job_seconds',180),remaining-reserve,plan['clock']['deadline_monotonic']-time.monotonic()-reserve)
    if timeout<30:
        progress['jobs'][key]={'status':'BUDGET_SKIPPED','directory':str(directory.relative_to(root))};return False
    snapshot=runtime_snapshot(plan['paths']['server_python']);require(runtime_gate(snapshot) is None,'Existing serving tuple unsupported; no fallback')
    if 'serving_runtime_identity' in plan:require(digest(snapshot)==plan['serving_runtime_identity'],'Serving runtime identity changed after B2 freeze')
    write(directory/'runtime.json',snapshot);launch_plan=server_launch(plan,arm,directory,contract);write(directory/'launch.json',launch_plan)
    await owned.ensure_port_free(launch_plan['url'])
    budget=Budget(ledger);budget.begin(key+'-attempt1',timeout+reserve);started=time.monotonic()
    stop,abort=asyncio.Event(),asyncio.Event();owner={};process=log=None;status='FAILED';readiness=[];rows=[]
    monitor=asyncio.create_task(gpu_monitor(stop,abort,directory,{'runtime':{'gpu':plan['gpu']['index']},'instrumentation':{'ownership_poll_seconds':2,'memory_poll_seconds':1}},owner))
    authored={r['sample_id']:r for r in plan['data']['rows']}
    with (directory/'generations.jsonl').open('x',encoding='utf-8') as raw,(directory/'scores.jsonl').open('x',encoding='utf-8') as scored,(directory/'phases.jsonl').open('x',encoding='utf-8') as phases:
        def phase(name):phases.write(json.dumps({'phase':name,'monotonic':time.monotonic()})+'\n');phases.flush()
        async def exercise():
            await guarded(owned.ready(process,launch_plan['url'],timeout=min(120,timeout),observations=readiness),abort)
            log_text=(directory/'server.log').read_text(encoding='utf-8')
            capacities=[int(n.replace(',','')) for n in re.findall(r'GPU KV cache size: ([\d,]+) tokens',log_text)]
            require(capacities and all(n==4096 for n in capacities),'Effective KV capacity not verified at requested4096')
            selected=[line for line in log_text.splitlines() if 'Marlin' in line or 'Using ' in line and 'Kernel' in line]
            if arm['arm_id']=='rtn' or plan.get('binding_profile')=='prepared_qwen_serve_v1' and arm['source_format']!='native':require(any('Marlin' in line for line in selected),'Packed RTN kernel selection log missing; no decoded serving fallback')
            effective={'kv_token_capacities':capacities,'kv_bytes':launch_plan['kv_bytes'],'kv_bytes_per_token':launch_plan['kv_bytes_per_token'],
                'kernel_observation':'kernel_selected_log','profiler_verified':False,'selection_lines':selected}
            effective.update(collector_metadata(contract))
            write(directory/'effective-serving.json',effective)
            phase('loaded_idle');write(directory/'loaded-idle-gpu.json',gpu_query(plan['gpu']['index']))
            if request_driver is not None:
                records=await request_driver(plan,arm,directory,launch_plan['url'],process,abort,phase)
                validate_collector_records(records,contract)
                rows.extend(records)
                phase('shutdown')
                kind='live_typed_task_requests' if contract['request_kind']=='typed_mixed_v1' else 'live_native_likelihood' if contract['request_kind']=='choice_loglikelihood' else 'live_task_generation'
                write(directory/'result.json',{'status':'COMPLETE','request_records':len(rows),'evidence_kind':kind,'deployed_path':True,'collector_contract':contract})
                return
            for prepared in plan['samples']['generation_rows']:
                require(process.poll() is None and not abort.is_set(),'Owned server stopped or GPU ownership changed')
                require(time.monotonic()<plan['clock']['deadline_monotonic']-reserve,'Session cleanup reserve')
                sample=authored[prepared['sample_id']];phase(sample['split']+(':'+prepared['policy_id'] if 'policy_id' in prepared else ''))
                row={'id':key+'-'+(prepared['policy_id']+'-' if 'policy_id' in prepared else '')+sample['sample_id'],'prompt_hash':prepared['prompt_hash'],'text_hash':prepared['document_hash'],
                     'token_ids':prepared['generation_input_ids'],'input_length':len(prepared['generation_input_ids'])}
                block=await guarded(http.block(launch_plan['url'],Adapter('vllm'),[row],1,output_tokens=192,
                    profile='natural_generation_v1',timeout=min(30,timeout),abort_event=abort),abort)
                wire=block['records'][0]
                validate_collector_records([wire],contract)
                record=wire|{'request_identity':http.sha(wire['request_body']),'task_id':TASK,'task_version':VERSION,'sample_id':sample['sample_id'],'sample_identity':sample['sample_identity'],
                    'arm_id':arm['arm_id'],'split':sample['split'],'evidence_kind':'live_task_generation','execution_mode':'DEPLOYED_PATH_OBSERVATION',
                    'artifact_identity':arm['artifact']['artifact_identity'],'tokenizer_identity':plan['samples']['tokenizer_identity'],
                    'task_prompt_hash':prepared['task_prompt_hash'],'reference_used_for_generation':False}
                if 'policy_id' in prepared:
                    record.update(policy_id=prepared['policy_id'],policy_identity=prepared['policy_identity'],prepared_identity=digest(prepared),scorer_identity=plan['scorer_identity'],runtime_identity=plan['serving_runtime_identity'])
                raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();rows.append(record)
                begin=time.monotonic();value=score_function(sample,record);value['score_wall_seconds']=time.monotonic()-begin
                scored.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');scored.flush()
                if len(rows)==plan['counts'].get('smoke_posts_per_arm',4):require(all(r['transport_complete'] for r in rows),'SMOKE transport not validated; DEV remains NOT_RUN, no retry')
            phase('shutdown');write(directory/'result.json',{'status':'COMPLETE','request_records':len(rows),'quality_status':'SCORED_SEPARATELY','generation_posts':len(rows),'warmup_posts':0,'evidence_kind':'live_task_generation','deployed_path':True})
        try:
            phase('startup');process,log=owned.start(launch_plan,directory);owner.update(parent_pid=process.pid,parent_proc_identity=proc_identity(process.pid));write(directory/'owned.json',owner)
            await asyncio.wait_for(exercise(),timeout=timeout);status='COMPLETE'
        except BaseException as exc:write(directory/'failure.json',{'status':'FAILED_OR_PARTIAL','error_type':type(exc).__name__,'error':str(exc)})
        finally:
            try:
                if process is not None:
                    cleanup=await asyncio.to_thread(owned.stop,process,log,grace=25)
                    life=observed_shutdown(cleanup,(directory/'server.log').read_text(encoding='utf-8'),10)
                    await owned.ensure_port_free(launch_plan['url']);life['owned_port_cleanup']='FREE';write(directory/'lifecycle.json',life)
                    if cleanup['owned_process_cleanup']!='CLEANED':status='FAILED_UNRESOLVED_OWNED_RESOURCE'
                write(directory/'readiness-http.json',readiness)
            finally:
                stop.set();await monitor;after=gpu_query(plan['gpu']['index'],True);write(directory/'post-gpu.json',after)
                if after['returncode']!=0 or after.get('values'):status='BLOCKED_POST_GPU_OR_INTERFERENCE'
                budget.end(time.monotonic()-started,status=status,owner_pid=process.pid if process else None,parent_returncode=process.returncode if process else None)
    progress['jobs'][key]={'status':status,'directory':str(directory.relative_to(root)),'requests':len(rows)};return status=='COMPLETE'


async def managed(plan_file,output):
    plan=read(Path(plan_file));check_frozen(plan)  # Resource/identity gate before GPU query, outputs, or budget changes.
    root=new_output(output);write(root/'PLAN_FROZEN.json',plan)
    import inference_kit
    write(root/'controller-environment.json',{'python':sys.executable,'package_version':inference_kit.__version__,
          'package_import_origin':inference_kit.__file__,'package_site':plan['paths']['package_site'],'source_identity':plan['source_identity']})
    ledger=root/'budget-continuation.json';previous=plan['campaign']['used_seconds']
    atomic(ledger,{'schema_version':1,'task_id':'BENCH05-B1','campaign_cap_seconds':7200,'cap_seconds':previous+plan['gpu_allowance_seconds'],
        'used_seconds':previous,'task_origin_used_seconds':plan['task_origin_used_seconds'],
        'inherited_from':{'path':plan['campaign']['path'],'sha256':plan['campaign']['sha256'],'used_seconds':previous},'running':None,'jobs':[]})
    progress={'status':'RUNNING','jobs':{},'automatic_retries':0};atomic(root/'progress.json',progress)
    try:
        for phase in ('serve','main'):
            for arm_id in plan['arm_order']:
                if time.monotonic()>=plan['clock']['no_new_gpu_after']:
                    progress['status']='BUDGET_SKIPPED';return progress
                arm=next(a for a in plan['arms'] if a['arm_id']==arm_id)
                success=await serving_job(plan,root,ledger,progress,arm) if phase=='serve' else await launch_job(plan,root,ledger,progress,'main',arm)
                atomic(root/'progress.json',progress)
                if not success:progress['status']='PARTIAL_OR_BLOCKED';return progress
        progress['status']='COMPLETE';return progress
    except BaseException as exc:
        progress['status']='FAILED_OR_PARTIAL';write(root/'controller-failure.json',{'status':progress['status'],'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        atomic(root/'progress.json',progress);seal(root)


def run(plan,output):return asyncio.run(managed(plan,output))
