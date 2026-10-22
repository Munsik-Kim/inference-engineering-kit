"""Explicit E3 live bindings and typed drivers over retained owned lifecycles.

Import and bind are CPU-only. Acquisition occurs only through the async entry
called by the authorized suite executor. No scheduler, retries or tokenizer.
"""
import copy
import errno
import json
from pathlib import Path
import socket
import time

from ..core.io import read, write, require, digest, sha, fields
from ..core.state import atomic
from ..tasks.registry import get_task
from .suite_inputs import load_sources
from .suite_plan import check_frozen
from .task_environment import inspect_artifacts, resources
from .bench05a_plan import source_identity, ledger_evidence

TOKEN_WORKER = {'module':'inference_kit.probes.token_worker','token_only':True}


def mixed_contract():
    entries=[{'request_kind':'generate_until','stream':True,'max_new_tokens':n,'ignore_eos':False,'audit_generation_tokens':0} for n in (192,64)]
    entries.append({'request_kind':'choice_loglikelihood','stream':False,'max_new_tokens':1,'ignore_eos':True,'audit_generation_tokens':1})
    return {'request_kind':'typed_mixed_v1','stream':'per_request','max_new_tokens':192,
            'ignore_eos':'per_request','audit_generation_tokens':'per_request','contracts':entries}


def bind_live_plan(frozen_suite, sources_manifest, local_binding, new_clock):
    """Resolve existing receipts/stat/config only. Never query GPU or load ML SDKs."""
    check_frozen(frozen_suite)
    sources=load_sources(sources_manifest)
    require(sources['data_identity']==frozen_suite['semantic']['inputs']['data_identity'],'Live prepared sources differ from suite')
    local=read(Path(local_binding)) if isinstance(local_binding,(str,Path)) else copy.deepcopy(local_binding)
    required={'artifact_receipt','runtime_metadata','runtime_python','server_python','package_site','prior_ledger','campaign_root','historical_plan','historical_plan_sha256','gpu','gpu_uuid','port'}
    fields(local,required,('schema_version','task_clock','parent_plan_files','historical_index','serving_runtime_identity','roots','task_origin_used_seconds'))
    paths={k:str(Path(local[k]).expanduser().absolute()) for k in required if k not in ('gpu','gpu_uuid','port','historical_plan_sha256')}
    for key in ('artifact_receipt','runtime_metadata','runtime_python','server_python','prior_ledger','historical_plan'):
        require(Path(paths[key]).is_file(),'Missing explicit existing live binding: '+key)
    require(type(local['gpu']) is int and local['gpu']>=0 and type(local['port']) is int and 1024<=local['port']<=65535 and
            isinstance(local['gpu_uuid'],str) and local['gpu_uuid'].startswith('GPU-'),'Invalid explicit GPU/port binding')
    historical_path=Path(paths['historical_plan']);require(sha(historical_path)==local['historical_plan_sha256'],'Historical metadata plan changed')
    historical=read(historical_path);require(digest({k:v for k,v in historical.items() if k!='plan_identity'})==historical['plan_identity'],'Historical plan integrity changed')
    selected=copy.deepcopy(historical['samples']);expected=[s['diagnostic'] for s in sources['samples']]
    require(selected['rows']==expected,'E3 must preserve the same six prepared inputs and exact selected positions')
    require(len(expected)==6 and selected['counts']['target_positions_per_arm']==110 and selected['counts']['full_vocab_positions_per_arm']==44,'Preserved E3 diagnostic coverage changed')
    metadata=sources['model_metadata'];require(selected['model_config']==metadata['model_config'] and
        selected['tokenizer_identity']==metadata['tokenizer_identity'] and selected['token_id_space']==metadata['token_id_space'],'Historical model/tokenizer metadata changed')
    spec=copy.deepcopy(metadata['token_spec']);budget=spec['budget']|{'max_gpu_processes':4,'raw_bytes_per_job':spec['budget']['raw_bytes_total']}
    require(budget['minimum_cpu_available_bytes']>=10*1024**3 and budget['minimum_disk_free_bytes']>=4*1024**3 and
            budget['cleanup_reserve_seconds']>=120 and budget['task_gpu_wall_seconds']<=600,'Live resource or cleanup floor changed')
    resource=resources(Path(paths['package_site']).parent,budget)
    require(resource['ready'],'BLOCKED_CPU_OR_DISK_RESOURCE')
    pair=copy.deepcopy(historical['pair']);refs=read(Path(paths['artifact_receipt']))
    arms=inspect_artifacts(pair,refs,selected)
    require(all(a['artifact_available'] for a in arms),'BLOCKED_MISSING_LOCAL_ARTIFACT')
    planned={m['artifact_ref']:m for m in frozen_suite['semantic']['models']}
    require(set(planned)=={a['arm_id'] for a in arms},'Live pair differs from explicit suite models')
    for arm in arms:
        require(arm['artifact']['artifact_identity']==planned[arm['arm_id']]['artifact']['artifact_identity'],'Live artifact differs from frozen suite')
    runtime=read(Path(paths['runtime_metadata']));require(digest(runtime)==metadata['runtime_identities']['diagnostic'],'Diagnostic runtime receipt changed')
    history=ledger_evidence(paths['prior_ledger'],paths['campaign_root'])
    allowance=min(600,7200-history['used_seconds']);require(allowance>budget['cleanup_reserve_seconds'],'No cleanup-safe campaign remainder')
    clock=copy.deepcopy(new_clock)
    require(clock['deadline_monotonic']==clock['start_monotonic']+3600 and clock['no_new_gpu_after']==clock['start_monotonic']+3000,'Fresh E3 task clock required')
    require(time.monotonic()<clock['no_new_gpu_after'] and time.monotonic()<clock['deadline_monotonic']-120,'E3 task clock expired')
    code=source_identity();package=Path(paths['package_site'])/'inference_kit'
    require(package.is_dir(),'Own installed worker package-site missing')
    installed={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts}
    require(installed==code,'Controller/worker Python source identity differs')
    model_by_arm={a['artifact_ref']:a['arm_id'] for a in frozen_suite['semantic']['models']}
    serving_groups=[]
    for arm in arms:
        logical_arm=model_by_arm[arm['arm_id']]
        nodes=[n for n in frozen_suite['semantic']['graph']['nodes'] if n['arm_id']==logical_arm and n['operation_kind'] in ('serve_generate_until','serve_choice_loglikelihood')]
        require(len(nodes)==3 and all(n['state']=='NOT_EXECUTED' for n in nodes),'E3 requires the three validated task serving nodes per arm')
        policies=[n['output_contract']['serving_policy'] for n in nodes];runtimes=[n['output_contract']['runtime_identity'] for n in nodes]
        require(all(p==policies[0] for p in policies) and all(r==runtimes[0] for r in runtimes),'Serving recipes differ; grouping not permitted')
        require(runtimes[0]==metadata['runtime_identities']['serving'],'Serving runtime differs across source contracts')
        serving_groups.append({'group_id':'serve-'+arm['arm_id'],'arm_id':arm['arm_id'],'logical_arm_id':logical_arm,
            'node_ids':[n['node_id'] for n in nodes],'requests':[r for n in nodes for r in n['output_contract']['requests']],
            'serving_policy':policies[0],'serving_runtime_identity':runtimes[0]})
    common={'schema_version':1,'task':'BENCH05-E3','status':'CPU_BOUND_NOT_EXECUTED','pair':pair,'arms':arms,'arm_order':[a['arm_id'] for a in arms],
        'paths':paths,'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'port':local['port'],'runtime':runtime,'runtime_identity':digest(runtime),
        'serving_runtime_identity':metadata['runtime_identities']['serving'],'clock':clock,'campaign':history,'task_origin_used_seconds':history['used_seconds'],
        'gpu_allowance_seconds':allowance,'source_identity':code,'samples':selected,'probe_spec':spec,'worker_contract':TOKEN_WORKER,
        'suite':{'seed':frozen_suite['semantic']['suite']['seed'],'budget':budget,'observer':{'off_samples':0,'off_repeats':0},
                 'serving':copy.deepcopy(serving_groups[0]['serving_policy']),'serving_job_seconds':180},
        'data':{'rows':[copy.deepcopy(s['authored_sample']) for s in sources['samples']]},'evidence_kind':'planned_live_suite_execution',
        'parent_suite_plan_identity':frozen_suite['plan_identity'],'parent_task':'BENCH05-E3','resources':resource,
        'resource_estimates':copy.deepcopy(selected['resource_estimates'])}
    token_plan=copy.deepcopy(common);token_plan.update(task='BENCH05-C2',raw_format='token_probe_raw_v1',evidence_kind='planned_live_token_probe',
        scope='E3 new acquisition; retained common C2 token-only worker recipe, not historical C2 replay')
    token_plan['plan_identity']=digest(token_plan)
    return {'schema_version':1,'kind':'suite_live_context_v1','suite_plan':copy.deepcopy(frozen_suite),'sources':sources,
        'runtime_plan':common,'token_plan':token_plan,'serving_groups':serving_groups,'campaign':history,
        'serving_jobs':{model_by_arm[a['arm_id']]:{'source_arm_id':a['arm_id'],'raw_path':'acquisitions/serve/attempts/serve-'+a['arm_id']+'-attempt1/task-responses.jsonl',
            'journal_path':'acquisitions/serve/attempts/serve-'+a['arm_id']+'-attempt1/request-journal.jsonl'} for a in arms},
        'diagnostic_run':'acquisitions/token','historical_index':local.get('historical_index'),
        'binding_files':{paths[k]:sha(Path(paths[k])) for k in ('artifact_receipt','runtime_metadata','prior_ledger','historical_plan')},
        'source_manifest_path':str(Path(sources_manifest).absolute()),'source_manifest_sha256':sha(Path(sources_manifest)),
        'execution_count':0,'gpu_queries':0,'tokenizer_calls':0,'model_loads':0}


def _verify_bound_files_context(context):
    """Revalidate CPU identities before an owned launch, without GPU queries."""
    for name,want in context['binding_files'].items():require(sha(Path(name))==want,'Live bound file changed: '+Path(name).name)
    require(sha(Path(context['source_manifest_path']))==context['source_manifest_sha256'],'Suite source manifest changed')
    require(load_sources(context['source_manifest_path'])['data_identity']==context['sources']['data_identity'],'Suite source slice changed')
    plan=context['runtime_plan'];require(source_identity()==plan['source_identity'],'Source changed after E3 live binding')
    package=Path(plan['paths']['package_site'])/'inference_kit'
    installed={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts}
    require(package.is_dir() and installed==plan['source_identity'],'Installed worker source changed after E3 live binding')
    for arm in plan['arms']:
        artifact=arm['artifact'];path=Path(artifact['path']);require(sha(path/'config.json')==artifact['config_sha256'],'Artifact config changed')
        actual={p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in path.glob('*.safetensors')}
        require(actual==arm['weight_file_stat'],'Artifact weight stat changed')
        for name,want in plan['samples']['tokenizer_files'].items():require(sha(Path(artifact['tokenizer_path'])/name)==want,'Tokenizer file changed')
    require(time.monotonic()<plan['clock']['no_new_gpu_after'] and time.monotonic()<plan['clock']['deadline_monotonic']-120,'E3 clock/reserve expired')
    return True


def verify_live_context(context):
    if context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1'):
        from .prepared_live import verify_live_context as verify_prepared
        verify_prepared(context)
    return _verify_bound_files_context(context)


def preflight(context,runroot,*,query_gpu=True):
    """Resource/identity gate precedes every optional GPU query and ledger write."""
    plan=context['runtime_plan'];target=Path(runroot).absolute()
    directory=target if target.is_dir() else target.parent
    resource=resources(directory,plan['suite']['budget'])
    if not resource['ready']:
        return {'status':'BLOCKED_CPU_OR_DISK_RESOURCE','resources':resource,'gpu_queries':0,'ledger_changes':0}
    try:verify_live_context(context)
    except (ValueError,OSError) as error:
        return {'status':'BLOCKED_LOCAL_IDENTITY','reason':str(error),'resources':resource,'gpu_queries':0,'ledger_changes':0}
    if not query_gpu:
        return {'status':'CPU_PREFLIGHT_VERIFIED_GPU_NOT_QUERIED','resources':resource,'gpu_queries':0,'ledger_changes':0}
    if context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1'):
        loopback=loopback_preflight(plan['port'])
        if loopback['status']!='LOOPBACK_AVAILABLE':
            return loopback|{'resources':resource,'gpu_queries':0,'ledger_changes':0,
                            'runtime_import_check':'NOT_RUN_BLOCKED_BEFORE_MODEL_LAUNCH'}
    from .bench05a_run import check_gpu
    result=check_gpu(plan)
    return result|{'resources':resource,'gpu_queries':'EXPLICIT_PREFLIGHT_ONLY','ledger_changes':0}


verify_context=verify_live_context


def loopback_preflight(port):
    """Check the selected localhost port in current permissions; never escalate."""
    require(type(port) is int and 1024<=port<=65535,'Invalid explicit loopback port')
    stage='socket_create';handle=None
    try:
        handle=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        stage='bind';handle.bind(('127.0.0.1',port))
        stage='listen';handle.listen(1)
        return {'status':'LOOPBACK_AVAILABLE','host':'127.0.0.1','port':port,
                'stage':'listen','inference_POSTs':0,'server_launches':0}
    except OSError as error:
        reason='PERMISSION_DENIED' if error.errno in (errno.EPERM,errno.EACCES) else 'PORT_IN_USE' if error.errno==errno.EADDRINUSE else 'SOCKET_ERROR'
        return {'status':'BLOCKED_LOOPBACK','reason':reason,'errno':error.errno,'message':str(error),
                'host':'127.0.0.1','port':port,'stage':stage,'bind_attempted':stage!='socket_create',
                'listen_attempted':stage=='listen','inference_POSTs':0,'server_launches':0}
    finally:
        if handle is not None:handle.close()


def initialize_ledger(context,path):
    """Create only this E3 continuation; preserves campaign use and predecessor."""
    plan=context['runtime_plan'];history=context['campaign'];path=Path(path)
    require(not path.exists(),'Refuse overwrite continuation ledger')
    atomic(path,{'schema_version':1,'task_id':context.get('ledger_task_id','BENCH05-E3'),'campaign_cap_seconds':7200,
        'cap_seconds':history['used_seconds']+plan['gpu_allowance_seconds'],'used_seconds':history['used_seconds'],
        'task_origin_used_seconds':history['used_seconds'],'running':None,'jobs':[],
        'inherited_from':{'path':history['path'],'sha256':history['sha256'],'used_seconds':history['used_seconds']}})
    return read(path)


async def mixed_request_driver(plan,arm,directory,url,process,abort,phase):
    """One C1 server handles explicit task requests, preserving each wire contract."""
    import aiohttp
    from ..adapters import http
    from ..adapters.protocol import Adapter
    from ..adapters.task_executor import execute
    from ..adapters.run import guarded
    from ..adapters.likelihood import request_body
    from ..tasks.arc_challenge import decode_record
    source_samples={(s['task_id'],s['sample_id']):s for s in plan['e3_sources']['samples']}
    execution=plan['parent_suite_plan_identity']+'|'+plan['e3_run_name']+'|attempt1|'+arm['arm_id']
    prepared_profile=plan.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1')
    if prepared_profile:execution+='|'+plan['e3_group']['group_id']
    logical_arm=plan['e3_group'].get('logical_arm_id',arm['arm_id'])
    records=[]
    with (directory/'task-responses.jsonl').open('x',encoding='utf-8') as raw,(directory/'request-journal.jsonl').open('x',encoding='utf-8') as journal:
        async with aiohttp.ClientSession() as session:
            for entry in plan['e3_group']['requests']:
                sample=source_samples[(entry['task_id'],entry['sample_id'])] if 'task_id' in entry else next(s for (task,sid),s in source_samples.items() if sid==entry['sample_id'])
                adapter=get_task(sample['task_id'],sample['task_version']);authored=sample['authored_sample'];phase(sample['task_id'])
                requests=adapter.requests(authored|{'generation_input_ids':sample['generation']['wire_payload']['prompt']},execution,plan['samples']['tokenizer_identity']) if adapter.request_kind=='generate_until' else adapter.requests(authored,execution,plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity'])
                request=next(r for r in requests if r.request_kind=='generate_until' or r.payload['choice_index']==entry['choice_index'])
                require(process.poll() is None and not abort.is_set(),'Server/ownership unsafe before E3 submission')
                require(time.monotonic()<plan['clock']['deadline_monotonic']-plan['suite']['budget']['cleanup_reserve_seconds'],'E3 cleanup reserve before submission')
                def sink(event):
                    event=event|{'task_id':sample['task_id'],'task_version':sample['task_version'],'sample_id':sample['sample_id'],
                        'arm_id':logical_arm,'source_arm_id':arm['arm_id'],'execution_prefix':execution,'artifact_identity':arm['artifact']['artifact_identity']}
                    if prepared_profile:
                        from .prepared_live import sample_namespace
                        namespace=sample_namespace(sample)
                        event.update(execution_unit_id=plan['e3_group']['group_id'],source_sample_key=namespace,
                            qualified_sample_id=digest(namespace),qualified_sample_identity=digest(namespace))
                    journal.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
                async def natural(req):
                    prepared=sample['generation']['prepared'];payload={'id':req.request_id,'prompt_hash':prepared['prompt_hash'],
                        'text_hash':authored['document_hash'],'token_ids':req.payload['token_ids'],'input_length':len(req.payload['token_ids'])}
                    block=await http.block(url,Adapter('vllm'),[payload],1,output_tokens=entry['wire_body']['max_tokens'],
                        profile='natural_generation_v1',timeout=30,abort_event=abort,event_sink=sink)
                    return block['records'][0]
                wire=await guarded(execute(request,session=session,url=url,natural_collector=natural,timeout=30,event_sink=sink),abort)
                scorer=sample['diagnostic'].get('source_scorer_identity')
                if prepared_profile:
                    source=next(s for s in plan['e3_sources']['sources'] if s['source_id']==sample['source_id'])
                    scorer=digest({'contract':'task_scorer_identity_v1','task_id':sample['task_id'],
                        'task_version':sample['task_version'],'scorer_version':source['preparation']['scorer_version']})
                require(scorer,'Missing verified task scorer identity')
                record=wire|{'identity':request.identity,'request_kind':request.request_kind,'task_id':sample['task_id'],'task_version':sample['task_version'],
                    'sample_id':sample['sample_id'],'sample_identity':authored.get('sample_identity'),'split':authored['split'],'arm_id':logical_arm,'source_arm_id':arm['arm_id'],
                    'artifact_identity':arm['artifact']['artifact_identity'],'tokenizer_identity':plan['samples']['tokenizer_identity'],
                    'scorer_version':adapter.scorer_version,'scorer_identity':scorer,
                    'execution_prefix':execution,'execution_id':execution,'execution_mode':'DEPLOYED_PATH_OBSERVATION',
                    'runtime_identity':plan['serving_runtime_identity'],'reference_used_for_generation':False,
                    'policy_id':sample['diagnostic'].get('policy_id'),'reference_source':sample['diagnostic']['reference_source'],
                    'frozen_wire_contract_match':wire.get('request_body')==entry['wire_body']}
                if request.request_kind=='generate_until':
                    record.update(request_identity=http.sha(wire['request_body']),prompt_hash=sample['generation']['prepared']['prompt_hash'],
                        task_prompt_hash=sample['generation']['prepared'].get('task_prompt_hash',sample['diagnostic'].get('task_prompt_hash')),evidence_kind='live_task_generation')
                else:
                    record.update(choice_index=request.payload['choice_index'],usage_partition='E3_SMOKE_REGRESSION_REUSE',evidence_kind='live_native_likelihood')
                if prepared_profile:
                    from .prepared_live import sample_namespace
                    key=sample_namespace(sample)
                    record.update(source_sample_key=key,qualified_sample_identity=digest(key),qualified_sample_id=digest(key),
                        execution_unit_id=plan['e3_group']['group_id'],planned_request_id=entry['request_id'],
                        policy_id=sample['raw_record']['policy_id'],usage_scope=sample['raw_record']['usage_scope'])
                    if request.request_kind=='choice_loglikelihood':record['usage_partition']=sample['raw_record']['usage_scope']
                    if plan.get('binding_profile')=='prepared_qwen_serve_v1':
                        original_scope=sample['raw_record']['usage_scope']
                        effective='ENGINEERING_REGRESSION_REUSE' if original_scope=='ENGINEERING_INPUT_ONLY' else original_scope
                        record.update(source_usage_scope=original_scope,usage_scope=effective)
                        if request.request_kind=='choice_loglikelihood':record['usage_partition']=effective
                # Persist/flush first. Content errors never erase a submitted result.
                raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');raw.flush();records.append(record)
                require(record['frozen_wire_contract_match'],'Actual E3 payload differs from frozen per-request contract')
                if request.request_kind=='choice_loglikelihood':require(decode_record(request,record)['status']=='COMPLETE','Likelihood echo/transport failed; no retry')
                else:require(record['transport_complete'],'Generation transport failed; no retry')
    return records


def _lines(path):
    from ..core.io import loads
    return [loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_serving_attempt(directory,plan,arm):
    """Recheck complete saved responses against exact planned request coverage."""
    from .bench05b1_replay import generation_from_raw
    from ..tasks.arc_challenge import decode_record
    directory=Path(directory);records=_lines(directory/'task-responses.jsonl')
    expected={(r['sample_id'],r.get('choice_index')):r for r in plan['e3_group']['requests']}
    require(expected and len(expected)==len(plan['e3_group']['requests']),'Empty/duplicate planned task requests')
    actual={(r['sample_id'],r.get('choice_index')):r for r in records}
    require(len(actual)==len(records) and actual.keys()==expected.keys(),'Actual mixed request coverage differs')
    require(len({r['request_id'] for r in records})==len(records),'Duplicate actual request identity')
    samples={s['sample_id']:s for s in plan['e3_sources']['samples']}
    submits={e['request_id']:e for e in _lines(directory/'request-journal.jsonl') if e.get('event')=='http_submit'}
    submit_rows=[e for e in _lines(directory/'request-journal.jsonl') if e.get('event')=='http_submit']
    require(len(submits)==len(submit_rows)==len(records) and set(submits)=={r['request_id'] for r in records},'Submission journal coverage differs')
    for key,record in actual.items():
        sample=samples[record['sample_id']];adapter=get_task(sample['task_id'],sample['task_version'])
        require(record['task_id']==sample['task_id'] and record['task_version']==sample['task_version'] and
                record['arm_id']==plan['e3_group'].get('logical_arm_id',arm['arm_id']) and record.get('source_arm_id')==arm['arm_id'] and
                record['runtime_identity']==plan['serving_runtime_identity'] and record['artifact_identity']==arm['artifact']['artifact_identity'] and
                record['request_body']==expected[key]['wire_body'] and record['transport_complete'] is True,'Saved task response identity/transport differs')
        requests=adapter.requests(sample['authored_sample']|{'generation_input_ids':sample['generation']['wire_payload']['prompt']},record['execution_prefix'],plan['samples']['tokenizer_identity']) if adapter.request_kind=='generate_until' else adapter.requests(sample['authored_sample'],record['execution_prefix'],plan['samples']['tokenizer_identity'],plan['samples']['token_id_space'],plan['serving_runtime_identity'])
        request=next(r for r in requests if r.request_kind=='generate_until' or r.payload['choice_index']==key[1])
        require(record['request_id']==request.request_id and record['identity']==request.identity,'Saved typed request identity differs')
        require(submits[request.request_id]['request_body']==record['request_body'],'Submission/body mismatch')
        if adapter.request_kind=='generate_until':generation_from_raw(record,sample['generation']['prepared'],arm,plan,
            task_id=sample['task_id'],task_version=sample['task_version'],max_new_tokens=expected[key]['wire_body']['max_tokens'])
        else:require(decode_record(request,record)['status']=='COMPLETE','Saved choice likelihood response incomplete')
    return {'inference_POSTs':len(records),'main_target_observations':0,'main_full_observations':0,'forward_calls':0}


def validate_token_attempt(directory,plan,arm):
    """Validate saved numeric coverage, finiteness and existing same-pass policy."""
    from ..probes.contracts import Observation
    from ..probes.blobs import load
    from ..probes.token_worker import check_saved_consistency,CONSISTENCY_POLICY
    directory=Path(directory);result=read(directory/'result.json');execution=read(directory/'execution.json')
    original_rows=plan['samples']['rows']
    rows={r['sample_id']:r for r in original_rows}
    require(rows and len(rows)==len(original_rows),'Empty/duplicate token worker input')
    expected={(kind,sid,t):target for sid,row in rows.items() for target in row['targets'] for t in [target['token_position']]
              for kind in (['target_logprob','full_logits'] if t in row['full_vocab_target_positions'] else ['target_logprob'])}
    require(result['status']==execution['status']=='MEASURED' and result['samples']==len(rows) and execution['phase']=='main','Token worker result/forward coverage differs')
    for key,want in (('arm_id',arm['arm_id']),('artifact_identity',arm['artifact']['artifact_identity']),
        ('recipe_identity',arm['artifact']['recipe_identity']),('tokenizer_identity',plan['samples']['tokenizer_identity']),
        ('runtime_identity',plan['runtime_identity']),('execution_mode',arm['execution_mode'])):
        require(execution[key]==want,'Token execution identity changed: '+key)
    observations=_lines(directory/'observations.jsonl');actual={};ids=set()
    for value in observations:
        Observation(**value).validate();c=value['coordinates'];key=(value['kind'],value['sample_id'],c['token_position'])
        require(key in expected and key not in actual and value['observation_id'] not in ids,'Unexpected/duplicate token observation')
        require(value['status']=='MEASURED' and value['execution_id']==execution['execution_id'] and value['label']==arm['execution_mode'] and
            value['evidence_kind']=='live_model_diagnostic','Invalid token observation acquisition identity')
        target=expected[key];row=rows[key[1]];t=key[2]
        require(all(type(c.get(name)) is int for name in ('token_position','logit_position','target_token_id')) and
            c['logit_position']==t-1 and c['target_token_id']==target['target_token_id'] and c['sequence_identity']==row['sequence_identity'] and
            c['prefix_hash']==digest({'token_ids':row['input_ids'][:t],'token_id_space':plan['samples']['token_id_space']}) and
            c['tokenizer_identity']==plan['samples']['tokenizer_identity'] and c['token_id_space']==plan['samples']['token_id_space'] and c['pass_id']=='ON','Token observation coordinate changed')
        require(value['details'].get('normalization')==plan['probe_spec']['normalization'] and
            value['details'].get('consistency_policy')==CONSISTENCY_POLICY,'Token observation probability/consistency metadata changed')
        values=load(directory,value['numeric']);require(value['numeric']['shape']==([1] if key[0]=='target_logprob' else [plan['samples']['vocabulary_size']]),'Token observation numeric shape changed')
        if key[0]=='target_logprob':require(len(values)==1 and values[0]<=0,'Target logprob must be finite nonpositive scalar')
        actual[key]=value;ids.add(value['observation_id'])
    require(actual.keys()==expected.keys(),'Actual token observation set incomplete')
    for key,value in actual.items():
        if key[0]=='full_logits':
            scalar=actual[('target_logprob',key[1],key[2])]
            check=check_saved_consistency(directory,scalar['numeric'],value['numeric'],expected[key]['target_token_id'],CONSISTENCY_POLICY)
            require(check['status']=='PASS','Saved scalar/full consistency violation')
    target_count=sum(k[0]=='target_logprob' for k in actual);full_count=sum(k[0]=='full_logits' for k in actual)
    require(target_count==result['main_target_positions'] and full_count==result['full_vocab_positions'] and
        result['main_attention_rows']==result['attention_hooks_installed']==0 and result['no_attention_observations_created'] is True and
        result['no_full_T_by_V_logits_created'] is True,'Token-only result/count mismatch')
    require(result['same_pass_consistency']['checked']==full_count and result['same_pass_consistency']['violations']==0,'Worker saved consistency coverage differs')
    return {'inference_POSTs':0,'main_target_observations':target_count,'main_full_observations':full_count,'forward_calls':result['samples']}


async def execute_acquisition(kind,node,context,runroot,ledger,progress):
    """Invoke retained owned lifecycle only after the suite executor authorizes it."""
    verify_live_context(context)
    from .bench05b1_run import serving_job
    from .bench05a_run import launch_job
    runroot=Path(runroot);arm_id=node['arm_id'];base=context['runtime_plan']
    arm=next(a for a in base['arms'] if a['arm_id']==arm_id)
    serve=kind in ('serve_group','serve','serve_generate_until','serve_choice_loglikelihood')
    require(serve or kind in ('token_group','token','collect_token_diagnostic'),'Unsupported E3 acquisition operation')
    prepared_profile=context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1')
    root=runroot/'acquisitions'/('serve' if serve else 'token')
    if serve and prepared_profile:root=root/node['node_id']
    root.mkdir(parents=True,exist_ok=True)
    plan=copy.deepcopy(base if serve else context['token_plan'])
    if serve:
        groups=[g for g in context['serving_groups'] if (g['group_id']==node['node_id'] if prepared_profile else g['arm_id']==arm_id)]
        require(len(groups)==1,'Ambiguous/missing serving execution unit')
        group=groups[0]
        plan.update(e3_group=group,e3_sources=context['sources'],e3_run_name=runroot.name)
        if prepared_profile:plan['suite']['serving']=copy.deepcopy(group['serving_policy'])
        plan['plan_identity']=digest(plan)
    else:group={'node_ids':node.get('provides',[node['node_id']])}
    # Shared C2 child plan is identical for both arms; it explicitly names E3.
    frozen=root/'PLAN_FROZEN.json'
    frozen_record=plan if not serve or prepared_profile else base|{'plan_identity':digest(base)}
    if not frozen.exists():write(frozen,frozen_record)
    else:require(read(frozen)==frozen_record,'Acquisition root frozen plan changed')
    state_file=root/'progress.json';state=read(state_file) if state_file.exists() else {'schema_version':1,'status':'RUNNING','jobs':{},'automatic_retries':0}
    extra={'job_key':group['group_id']} if serve and prepared_profile else {}
    okay=await serving_job(plan,root,Path(ledger),state,arm,request_driver=mixed_request_driver,collector_contract=mixed_contract(),**extra) if serve else await launch_job(plan,root,Path(ledger),state,'main',arm,worker_contract=TOKEN_WORKER)
    key=group['group_id'] if serve and prepared_profile else ('serve-' if serve else 'main-')+arm_id
    job=state['jobs'].get(key,{})
    directory=root/job['directory'] if job.get('directory') else root
    counts={'inference_POSTs':0,'main_target_observations':0,'main_full_observations':0,'forward_calls':0}
    if okay:
        life=read(directory/'lifecycle.json')
        try:
            require(life.get('owned_process_cleanup')=='CLEANED','Owned cleanup was not verified')
            if not serve:require(life.get('native_main_returncode')==0,'Token worker exit was not zero')
            counts=validate_serving_attempt(directory,plan,arm) if serve else validate_token_attempt(directory,plan,arm)
            write(directory/'raw-validation-receipt.json',{'status':'VERIFIED','counts':counts,'validation':'Exact expected IDs/coverage and saved raw payload/numeric contracts; no model reload'})
        except (ValueError,KeyError,TypeError,OSError) as error:
            okay=False;job['status']='FAILED_RAW_CONTRACT';write(directory/'raw-validation-failure.json',{'status':job['status'],'error':str(error)})
    expected_jobs={key} if serve and prepared_profile else {'main-'+a['arm_id'] for a in base['arms']} if prepared_profile else None
    complete=(set(state['jobs'])==expected_jobs if expected_jobs is not None else len(state['jobs'])==2)
    state['status']='COMPLETE' if complete and all(j['status']=='COMPLETE' for j in state['jobs'].values()) else 'PARTIAL_OR_PENDING'
    atomic(state_file,state)
    receipt={'status':'COMPLETE' if okay else job.get('status','FAILED'),'arm_id':arm_id,'kind':kind,'directory':str(directory.relative_to(runroot)),
        'child_node_ids':group['node_ids'],'counts':counts,
        'evidence_files':[str(p.relative_to(runroot)) for p in sorted(directory.rglob('*')) if p.is_file() and 'runtime-cache' not in p.parts],
        'cleanup':'RECORDED_IN_OWNED_LIFECYCLE','native_worker_exit':'UNKNOWN_NOT_PROMOTED',
        'owned_cleanup_verified':okay and (directory/'lifecycle.json').is_file() and read(directory/'lifecycle.json').get('owned_process_cleanup')=='CLEANED'}
    progress.setdefault('jobs',{})[key]=job
    return receipt


def finalize_acquisitions(runroot):
    """Seal retained raw after acquisition stages; never overwrite an old seal."""
    from .bench05b1_run import seal
    runroot=Path(runroot)
    for name,scope in (('serve','suite_native_task_raw_v1'),('token','token_probe_raw_v1')):
        root=runroot/'acquisitions'/name
        if root.is_dir() and not (root/'RAW_MANIFEST.json').exists():seal(root,scope=scope)


def finalize_token_raw(context,runroot):
    finalize_acquisitions(runroot)
