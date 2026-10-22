"""E3 sequential, evidence-committed execution of the prepared Suite graph.

Only built-in operations dispatch. Completed acquisition reuse belongs to this
run, never to a historical cache. Safe boundaries do not promise exactly-once
submission recovery after an interrupted HTTP/GPU operation.
"""
from contextlib import contextmanager
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from ..core.io import read,write,sha,digest,require,safe_name
from ..core.state import new_output,atomic
from ..analysis.tokens import source_identity
from .suite_plan import check_frozen

GPU_KINDS={'serve_group','token_group'}
CPU_KINDS={'validate_prepared_inputs','score_task','analyze_tokens','index_add','report'}


def _utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def _lock(root):
    with (root/'.controller.lock').open('a+',encoding='utf-8') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('RUN_LOCKED: another controller owns this run')
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def _event(root,kind,**payload):
    events=_events(root)
    value={'sequence':len(events),'previous_hash':digest(events[-1]) if events else None,'kind':kind,
           'utc':_utc(),'monotonic':time.monotonic(),'controller_pid':os.getpid(),**payload}
    with (root/'events.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return value


def _events(root):
    path=root/'events.jsonl';rows=[]
    if not path.exists():return rows
    with path.open(encoding='utf-8') as stream:
        from ..core.io import loads
        for line in stream:
            value=loads(line);require(value['sequence']==len(rows) and value['previous_hash']==(digest(rows[-1]) if rows else None),'Journal sequence/hash changed');rows.append(value)
    return rows


def _path(root,relative):
    name=safe_name(relative);path=root/name
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),'Evidence path escapes run')
    for parent in path.parents:
        if parent==root:break
        require(not parent.is_symlink(),'Evidence symlink ancestor')
    return path


def _inventory(root,directory):
    directory=_path(root,str(Path(directory).relative_to(root))) if Path(directory).is_absolute() else _path(root,str(directory))
    require(directory.is_dir(),'Missing node output directory');files={}
    for p in sorted(directory.rglob('*')):
        require(not p.is_symlink(),'Symlink node output')
        if p.is_file() and 'runtime-cache' not in p.relative_to(directory).parts:
            files[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    require(files,'EMPTY_REQUIRED_OUTPUT: no evidence files')
    return files


def _verify_files(root,files):
    require(isinstance(files,dict) and files,'MISSING_EVIDENCE: empty output manifest')
    for name,expected in files.items():
        path=_path(root,name);require(path.is_file(),'MISSING_EVIDENCE: '+name)
        require(path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'],'STALE_EVIDENCE: '+name)


def _unit_id(node):return node['node_id']


def execution_units(compiled,context):
    nodes=compiled['semantic']['graph']['nodes'];require(nodes,'INVALID_EMPTY_PLAN')
    require(compiled['status']=='VALIDATED_NOT_EXECUTED','Required Suite operations blocked')
    require(not any(n['operation_kind']=='collect_structure_diagnostic' for n in nodes),'EXECUTION_NOT_IMPLEMENTED: structure collection in E3')
    units=[];validates=[n['node_id'] for n in nodes if n['operation_kind']=='validate_prepared_inputs']
    require(validates,'INVALID_EMPTY_PLAN: no prepared validation')
    units.append({'node_id':'validate_inputs','operation_kind':'validate_prepared_inputs','dependencies':[], 'provides':validates,'required':True,'counts':{'documents':compiled['semantic']['counts']['documents']}})
    acquisition=[]
    for group in context['serving_groups']:
        name=group['group_id'];acquisition.append(name)
        units.append({'node_id':name,'operation_kind':'serve_group','dependencies':['validate_inputs'],'provides':group['node_ids'], 'arm_id':group['arm_id'],'required':True,'counts':{'inference_POSTs':len(group['requests'])}})
    tokennodes=[n for n in nodes if n['operation_kind']=='collect_token_diagnostic']
    for arm in ([] if context.get('binding_profile')=='prepared_qwen_serve_v1' else context['runtime_plan']['arms']):
        model=next(m for m in compiled['semantic']['models'] if m['artifact_ref']==arm['arm_id'])
        selected=[n for n in tokennodes if n['arm_id']==model['arm_id']];require(len(selected)==1,'E3 requires one token-only collector per arm')
        name='token-'+arm['arm_id'];acquisition.append(name)
        units.append({'node_id':name,'operation_kind':'token_group','dependencies':['validate_inputs'],'provides':[selected[0]['node_id']],'arm_id':arm['arm_id'],'required':True,'counts':selected[0]['counts']})
    if context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1'):
        expected=context['execution_unit_ids']
        require(acquisition and len(acquisition)==len(set(acquisition)) and
                set(acquisition)==set(expected) and len(acquisition)==context['expected_gpu_jobs'],
                'Prepared execution unit coverage differs from frozen binding')
    else:
        require(len(acquisition)==4,'E3 frozen live recipe requires two serving and two token acquisitions')
    for kind in ('score_task','analyze_tokens','index_add','report'):
        selected=[n['node_id'] for n in nodes if n['operation_kind']==kind];require(selected,'Missing required E3 downstream operation: '+kind)
        deps=acquisition[:] if kind in ('score_task','analyze_tokens') else ['score_task','analyze_tokens'] if kind=='index_add' else ['index_add']
        units.append({'node_id':kind,'operation_kind':kind,'dependencies':deps,'provides':selected,'required':True,'counts':{'documents':compiled['semantic']['counts']['documents']}})
    produced=[p for u in units for p in u['provides']]
    require(len(produced)==len(set(produced)) and set(produced)=={n['node_id'] for n in nodes},'Execution units must cover each typed node exactly once')
    return units


def _frozen(root,*,current_source=True):
    record=read(root/'EXECUTION_PLAN.json');value=dict(record);identity=value.pop('execution_plan_identity')
    require(identity==digest(value),'Execution plan changed')
    require(isinstance(record.get('units'),list) and record['units'] and any(n.get('required') is True for n in record['units']),'INVALID_EMPTY_PLAN: no required execution units')
    names=[n['node_id'] for n in record['units']];require(len(names)==len(set(names)),'Duplicate execution unit ID')
    preceding=set()
    for node in record['units']:
        require(node['operation_kind'] in GPU_KINDS|CPU_KINDS and set(node['dependencies'])<=preceding,'Invalid operation/dependency execution order')
        require(node.get('counts') and all(type(v) is int and v>=0 for v in node['counts'].values()) and any(node['counts'].values()),'EMPTY_REQUIRED_OUTPUT: missing planned coverage')
        preceding.add(node['node_id'])
    if current_source:require(record['executor_source']==source_identity(),'Executor source changed; automatic resume forbidden')
    _verify_files(root,record['immutable_inputs']);_events(root)
    return record


def _dependency_identity(root,plan,state,node):
    values={}
    for parent in node['dependencies']:
        require(state['nodes'][parent]['status']=='COMPLETE_VALIDATED','BLOCKED_DEPENDENCY: '+parent)
        result=_completion(root,plan,state,next(n for n in plan['units'] if n['node_id']==parent))
        values[parent]=result['completion_identity']
    return digest({'execution_plan_identity':plan['execution_plan_identity'],'node':node,'dependencies':values})


def _completion(root,plan,state,node):
    name=node['node_id'];entry=state['nodes'][name];path=_path(root,entry.get('completion','nodes/'+name+'/COMPLETION.json'))
    require(path.is_file(),'MISSING_EVIDENCE: completion receipt absent')
    value=read(path);check=dict(value);identity=check.pop('completion_identity')
    require(identity==digest(check) and value['execution_plan_identity']==plan['execution_plan_identity'] and value['node_id']==name,'Completion identity differs')
    require(value['status']=='COMPLETE_VALIDATED' and value['input_identity']==_dependency_identity(root,plan,state,node),'Completion dependency/input identity differs')
    _verify_files(root,value['files'])
    if node['operation_kind'] == 'index_add':
        from ..analysis.suite_results import validate_index_completion
        validate_index_completion(_path(root, value['evidence_directory']), root)
    if 'ledger_prefix' in value:
        current=_verify_ledger(read(root/'suite-execution-context.json'),root);prefix=value['ledger_prefix']
        require(current is not None and current['jobs'][:len(prefix['jobs'])]==prefix['jobs'] and prefix['jobs'],'Committed GPU budget prefix changed')
        for key in ('inherited_from','cap_seconds','task_origin_used_seconds','campaign_cap_seconds'):require(current[key]==prefix[key],'Committed GPU budget identity changed')
    return value


def _commit(root,plan,state,node,receipt,output):
    require(receipt.get('status') in ('COMPLETE','PASS','MEASURED','COMPLETE_VALIDATED'),'Node did not produce validated complete output')
    if node['operation_kind'] in GPU_KINDS:
        require(receipt.get('owned_cleanup_verified') is True,'GPU cleanup not verified')
        actual=receipt['counts'];expected=node['counts']
        if node['operation_kind']=='serve_group':require(actual['inference_POSTs']==expected['inference_POSTs']>0,'Incomplete serving submission/response coverage')
        else:
            prepared=read(root/'suite-execution-context.json').get('binding_profile')=='prepared_smol_live_v1'
            for key in ('main_target_observations','main_full_observations','forward_calls'):
                valid_count=expected[key]>0 or prepared and key=='main_full_observations' and expected[key]==0
                require(type(actual.get(key)) is int and actual[key]==expected[key] and valid_count,'Incomplete diagnostic coverage: '+key)
    files=_inventory(root,output)
    value={'schema_version':1,'node_id':node['node_id'],'execution_plan_identity':plan['execution_plan_identity'],
      'input_identity':_dependency_identity(root,plan,state,node),'evidence_directory':str(Path(output).relative_to(root)),'status':'COMPLETE_VALIDATED','receipt':receipt,'files':files,'completed_utc':_utc()}
    if node['operation_kind'] in GPU_KINDS:value['ledger_prefix']=read(root/'budget-continuation.json')
    value['completion_identity']=digest(value);directory=root/'nodes'/node['node_id'];directory.mkdir(parents=True,exist_ok=True)
    target=directory/'COMPLETION.json';write(target,value)
    _event(root,'NODE_COMMITTED',node_id=node['node_id'],completion_identity=value['completion_identity'])
    return str(target.relative_to(root))


def _seal_raw(root):
    target=root/'RUN_MANIFEST.json'
    if target.exists():_verify_files(root,read(target)['files']);return
    files={}
    for name in ('EXECUTION_PLAN.json','SUITE_PLAN.json','suite-execution-context.json','budget-continuation.json'):
        p=root/name;files[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    for folder in ('prepared','acquisitions'):
        files.update(_inventory(root,root/folder))
    write(target,{'schema_version':1,'kind':'suite_e3_raw_v1','files':files,'immutable':True,'scope':'New E3 acquisitions; root mutable checkpoint/eventlog and derived outputs excluded'})


def _verify_ledger(context,root):
    path=root/'budget-continuation.json'
    if not path.exists():return None
    record=read(path);parent=context['campaign'];origin=parent['used_seconds']
    prepared=context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1')
    require(record.get('task_id')==(context['ledger_task_id'] if prepared else 'BENCH05-E3') and record.get('campaign_cap_seconds')==7200 and
        record.get('task_origin_used_seconds')==origin and record.get('cap_seconds')==origin+min(600,7200-origin),'GPU ledger cap/origin changed')
    require(record.get('inherited_from')=={'path':parent['path'],'sha256':parent['sha256'],'used_seconds':origin},'GPU ledger parent changed')
    require(sha(Path(parent['path']))==parent['sha256'],'Prior campaign ledger changed')
    require(record.get('running') is None,'UNRESOLVED_GPU_ACCOUNTING; no retry')
    maximum=context['expected_gpu_jobs'] if prepared else 4
    jobs=record.get('jobs');require(isinstance(jobs,list) and len(jobs)<=maximum,'GPU ledger job cap changed')
    allowed=set(context['allowed_ledger_job_ids']) if prepared else {'serve-bf16-attempt1','serve-rtn-attempt1','main-bf16-attempt1','main-rtn-attempt1'}
    require(allowed and len(allowed)==maximum,'Invalid frozen ledger job coverage')
    require(len({j['id'] for j in jobs})==len(jobs) and all(j['id'] in allowed for j in jobs),'GPU ledger duplicate/unknown job')
    require(all(type(j['wall_seconds']) in (int,float) and math.isfinite(j['wall_seconds']) and j['wall_seconds']>0 for j in jobs),'GPU wall cost missing/nonfinite/zero')
    used=origin+math.fsum(j['wall_seconds'] for j in jobs)
    require(type(record.get('used_seconds')) in (int,float) and math.isfinite(record['used_seconds']) and
        abs(record['used_seconds']-used)<1e-6 and origin<=used<=record['cap_seconds'],'GPU ledger cumulative usage changed')
    return record


def _reserve_campaign(context,root):
    location=context.get('campaign_reservation_root')
    require(isinstance(location,str) and Path(location).is_absolute(),'Explicit shared E3 reservation location required')
    directory=Path(location);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(context['campaign']['sha256']+'.json')
    record={'schema_version':1,'parent_ledger_sha256':context['campaign']['sha256'],'run_root':str(root),
            'execution_plan_identity':read(root/'EXECUTION_PLAN.json')['execution_plan_identity'],
            'reserved_task_wall_upper_seconds':min(600,7200-context['campaign']['used_seconds'])}
    if path.exists():require(read(path)==record,'CAMPAIGN_RESERVED_OR_PARENT_ALREADY_CONSUMED: resolve latest continuation before another run')
    else:
        try:write(path,record)
        except (ValueError,FileExistsError):require(path.is_file() and read(path)==record,'CAMPAIGN_RESERVATION_CONFLICT')
    return record


def _dispatch(root,context,node,state,allow_gpu,output):
    kind=node['operation_kind']
    if kind=='validate_prepared_inputs':
        from .suite_inputs import load_sources
        loaded=load_sources(root/'prepared/SOURCES.json')
        require(loaded['data_identity']==context['sources']['data_identity'],'Prepared content changed')
        write(output/'VALIDATED.json',{'status':'PASS','documents':len(loaded['samples']),'data_identity':loaded['data_identity'],'GPU_actions':0});return {'status':'PASS'},output
    if kind in GPU_KINDS:
        require(allow_gpu,'Explicit --allow-gpu required for pending acquisition')
        from .suite_live import execute_acquisition,preflight
        _verify_ledger(context,root)
        gate=preflight(context,root)
        write(output/'PREFLIGHT.json',gate)
        if context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1') and gate['status'].startswith('BLOCKED_'):
            return gate|{'owned_cleanup_verified':False,'acquisition_started':False},output
        require(gate['status']=='READY_WITHIN_OBSERVATION_LIMITS',gate['status'])
        ledger=root/'budget-continuation.json'
        _reserve_campaign(context,root)
        if not ledger.exists():
            campaign=context['campaign'];previous=campaign['used_seconds'];cap=min(600,7200-previous)
            require(cap>120,'Cleanup-safe campaign allowance unavailable')
            atomic(ledger,{'schema_version':1,'task_id':context.get('ledger_task_id','BENCH05-E3'),'campaign_cap_seconds':7200,'task_origin_used_seconds':previous,
                'cap_seconds':previous+cap,'used_seconds':previous,'running':None,'jobs':[],
                'inherited_from':{'path':campaign['path'],'sha256':campaign['sha256'],'used_seconds':previous}})
        require(read(ledger)['running'] is None,'UNRESOLVED_GPU_ACCOUNTING; no retry')
        import asyncio
        progress={'jobs':{}}
        receipt=asyncio.run(execute_acquisition('serve' if kind=='serve_group' else 'token',node,context,root,ledger,progress))
        write(output/'HANDLER_RECEIPT.json',receipt)
        return receipt,root/receipt['directory']
    require(kind in CPU_KINDS,'EXECUTION_NOT_IMPLEMENTED: '+kind)
    if not (root/'RUN_MANIFEST.json').exists():
        from .suite_live import finalize_token_raw
        finalize_token_raw(context,root)
    _seal_raw(root)
    from ..analysis.suite_results import execute_cpu
    stage_outputs={k:str(root/v['output']) for k,v in state['nodes'].items() if v.get('output') and v['status']=='COMPLETE_VALIDATED'}
    if context.get('historical_index'):stage_outputs['historical_index']=context['historical_index']
    receipt=execute_cpu(kind,root,output,stage_outputs=stage_outputs)
    if not isinstance(receipt,dict):receipt={'status':'COMPLETE','result':receipt}
    receipt.setdefault('status','COMPLETE')
    return receipt,output


def _initial(plan_file,local_file,output):
    compiled=check_frozen(read(Path(plan_file)));require(compiled['status']=='VALIDATED_NOT_EXECUTED','Blocked Suite cannot execute')
    local=read(Path(local_file));binding=dict(local);reservation_root=binding.pop('campaign_reservation_root',None);clock=read(Path(local['task_clock']))
    if binding.get('kind')=='prepared_suite_live_binding_v1':
        from .prepared_live import bind_live_plan
    else:
        from .suite_live import bind_live_plan
    context=bind_live_plan(compiled,Path(compiled['bindings']['sources_path']),binding,clock)
    root=new_output(output);prepared=root/'prepared';prepared.mkdir()
    sources=Path(compiled['bindings']['sources_path']);manifest=read(sources)
    shutil.copyfile(sources,prepared/'SOURCES.json')
    for source in manifest['sources']:
        relative=safe_name(source['path']);target=prepared/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(sources.parent/relative,target)
    context['sources']=context.get('sources',context.get('input_sources'));context['suite_plan']=compiled
    prepared_profile=context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1')
    if not prepared_profile:
        context['serving_jobs']={m['arm_id']:{'raw_path':'acquisitions/serve/attempts/serve-'+m['artifact_ref']+'-attempt1/task-responses.jsonl','journal_path':'acquisitions/serve/attempts/serve-'+m['artifact_ref']+'-attempt1/request-journal.jsonl','source_arm_id':m['artifact_ref']} for m in compiled['semantic']['models']}
    if context.get('binding_profile')!='prepared_qwen_serve_v1':context['diagnostic_run']='acquisitions/token'
    context['historical_index']=local.get('historical_index')
    context['historical_index_sha256']=sha(Path(local['historical_index'])) if local.get('historical_index') else None
    context['campaign_reservation_root']=reservation_root
    context['local_binding_sha256']=sha(Path(local_file));context['evidence_kind']='live_prepared_suite_v1' if prepared_profile else 'live_suite_e3'
    write(root/'SUITE_PLAN.json',compiled);write(root/'suite-execution-context.json',context)
    units=execution_units(compiled,context)
    immutable={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in prepared.rglob('*') if p.is_file()}
    for name in ('SUITE_PLAN.json','suite-execution-context.json'):
        p=root/name;immutable[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    parents=[]
    for file in local.get('parent_plan_files',[]):
        old=read(Path(file));require(old['semantic']['inputs']['data_identity']==compiled['semantic']['inputs']['data_identity'],'Parent E2 input lineage differs')
        parents.append({'file':str(file),'sha256':sha(Path(file)),'plan_identity':old['plan_identity'],'compiler':old['semantic']['compiler'],
            'relation':'Prepared input/intent parent; freshly compiled execution source' if prepared_profile else 'E2 read-only intent; E3 freshly compiled combined task+token plan'})
    jobs=context['expected_gpu_jobs'] if prepared_profile else 4
    value={'schema_version':1,'kind':'suite_execution_plan_v1','executor_source':source_identity(),'compiled_plan_identity':compiled['plan_identity'],
        'units':units,'immutable_inputs':immutable,'parent_E2_plans':parents,'grouping':('One serving unit per typed task/arm; diagnostic NOT_REQUESTED' if context.get('binding_profile')=='prepared_qwen_serve_v1' else 'One serving unit per typed task/arm; one token worker per arm') if prepared_profile else 'Verified identical serving recipe per arm; original typed serve outputs provided explicitly',
        'expected_actual_gpu_jobs':jobs,'execution_count_at_freeze':0,'authorization_required':'GPU scoped to this content/output/source; not inherited from E2','created_utc':_utc()}
    if prepared_profile:value['parent_prepared_plans']=value.pop('parent_E2_plans')
    value['execution_plan_identity']=digest(value);write(root/'EXECUTION_PLAN.json',value)
    write(root/'LIVE_AUTHORIZATION.json',{'execution_plan_identity':value['execution_plan_identity'],'executor_source':source_identity(),'output_root':str(root),'operations':sorted({n['operation_kind'] for n in units if n['operation_kind'] in GPU_KINDS}),'max_gpu_jobs':jobs,'task_gpu_wall_cap':600,'automatic_retries':0,'scope':'Explicit --allow-gpu for this run; resume only verified completed nodes'})
    state={'schema_version':1,'run_id':uuid.uuid4().hex,'execution_plan_identity':value['execution_plan_identity'],'status':'PENDING','started_utc':_utc(),'started_monotonic':time.monotonic(),'active_controller_seconds':0.,'paused_seconds':0.,'pause_started_monotonic':None,'controller_sessions':0,
      'nodes':{u['node_id']:{'status':'PENDING','attempts':0} for u in units}}
    atomic(root/'checkpoint.json',state);_event(root,'RUN_CREATED',execution_plan_identity=value['execution_plan_identity'])
    return root


def _drive(root,local_file,allow_gpu,stop_after_node=None):
    with _lock(root):
        plan=_frozen(root);state=read(root/'checkpoint.json');context=read(root/'suite-execution-context.json')
        _verify_ledger(context,root)
        require(state['execution_plan_identity']==plan['execution_plan_identity'],'Checkpoint/plan mismatch')
        require(stop_after_node is None or stop_after_node in {n['node_id'] for n in plan['units'] if n['operation_kind']=='serve_group'},'Pause only at an owned serving acquisition boundary')
        checkpoint_before=digest(state);completed_before=[];reconciled_nodes=[]
        # Commit receipts are durable evidence; reconcile the commit→checkpoint gap.
        for node in plan['units']:
            name=node['node_id'];entry=state['nodes'][name];commit=root/'nodes'/name/'COMPLETION.json'
            if entry['status']=='COMPLETE_VALIDATED' or commit.exists():
                entry_before=digest(entry)
                if commit.exists() and entry['status']!='COMPLETE_VALIDATED':entry['completion']=str(commit.relative_to(root));entry['status']='COMPLETE_VALIDATED';entry['output']=read(commit)['evidence_directory']
                verified=_completion(root,plan,state,node)
                entry.setdefault('completion',str(commit.relative_to(root)));entry['output']=verified['evidence_directory']
                if digest(entry)!=entry_before:reconciled_nodes.append(name)
                completed_before.append(name)
            elif entry['status']=='RUNNING':
                entry['status']='INTERRUPTED' if node['operation_kind'] in GPU_KINDS else 'FAILED'
                entry['reason']='No durable completion; GPU submission unknown, never automatically resubmit' if node['operation_kind'] in GPU_KINDS else 'CPU stage may derive again from validated immutable inputs'
        pending=[n for n in plan['units'] if state['nodes'][n['node_id']]['status']!='COMPLETE_VALIDATED']
        if not pending:
            require(set(state['nodes'])=={n['node_id'] for n in plan['units']} and state['nodes'] and
                all(e['status']=='COMPLETE_VALIDATED' for e in state['nodes'].values()),'EMPTY_OR_INCOMPLETE_REQUIRED_PLAN')
            before=sha(root/'budget-continuation.json') if (root/'budget-continuation.json').exists() else None
            state['status']='COMPLETE_VALIDATED'
            checkpoint_reconciled=digest(state)!=checkpoint_before
            # No new execution: persist only a verified terminal reconciliation.
            # An already complete checkpoint keeps its original bytes/timestamps.
            if checkpoint_reconciled:atomic(root/'checkpoint.json',state)
            _event(root,'NOOP_RESUME',skipped=completed_before,GPU_queries=0,inference_POSTs=0,worker_launches=0,ledger_debit=0,
                checkpoint_reconciled=checkpoint_reconciled,reconciled_nodes=reconciled_nodes)
            require(before==(sha(root/'budget-continuation.json') if (root/'budget-continuation.json').exists() else None),'No-op mutated ledger')
            return {'status':'COMPLETE_VALIDATED','run':str(root),'noop':True,'skipped':completed_before,'new_GPU_jobs':0,'new_POSTs':0,'ledger_debit_seconds':0,'GPU_queries':0,
                'checkpoint_reconciled':checkpoint_reconciled,'reconciled_nodes':reconciled_nodes}
        if any(n['operation_kind'] in GPU_KINDS for n in pending):
            require(allow_gpu,'Explicit --allow-gpu required for pending GPU nodes')
            require(local_file is not None and sha(Path(local_file))==context['local_binding_sha256'],'Live local binding changed')
        for node in pending:
            require(not(node['operation_kind'] in GPU_KINDS and state['nodes'][node['node_id']]['status'] in ('FAILED','INTERRUPTED')),'GPU failure/interruption requires explicit new acquisition; automatic retry forbidden')
        started=time.monotonic();state['controller_sessions']+=1
        if state['pause_started_monotonic'] is not None:
            state['paused_seconds']+=time.monotonic()-state['pause_started_monotonic'];state['pause_started_monotonic']=None
        state['status']='RUNNING';_event(root,'CONTROLLER_STARTED',skipped=completed_before);atomic(root/'checkpoint.json',state)
        try:
            for node in plan['units']:
                name=node['node_id'];entry=state['nodes'][name]
                if entry['status']=='COMPLETE_VALIDATED':continue
                identity=_dependency_identity(root,plan,state,node)
                entry.update(status='RUNNING',attempts=entry['attempts']+1,input_identity=identity)
                _event(root,'NODE_STARTED',node_id=name,input_identity=identity,counts=node['counts'],attempt=entry['attempts']);atomic(root/'checkpoint.json',state)
                directory=root/'nodes'/name/('attempt'+str(entry['attempts']))
                if node['operation_kind'] in GPU_KINDS or node['operation_kind']=='validate_prepared_inputs':directory.mkdir(parents=True,exist_ok=False)
                receipt,evidence=_dispatch(root,context,node,state,allow_gpu,directory)
                if context.get('binding_profile') in ('prepared_smol_live_v1','prepared_qwen_serve_v1') and receipt.get('status','').startswith('BLOCKED_'):
                    require(receipt.get('acquisition_started') is False,'Blocked acquisition state is ambiguous')
                    entry.update(status='BLOCKED',reason=receipt['status'],preflight=str(evidence.relative_to(root)))
                    state['status']=receipt['status']
                    _event(root,'NODE_BLOCKED_BEFORE_ACQUISITION',node_id=name,reason=receipt['status'],GPU_queries=receipt.get('gpu_queries',0),inference_POSTs=0)
                    break
                completion=_commit(root,plan,state,node,receipt,evidence)
                entry.update(status='COMPLETE_VALIDATED',completion=completion,output=str(evidence.relative_to(root)))
                atomic(root/'checkpoint.json',state)
                if name==stop_after_node:
                    state.update(status='PAUSED_BOUNDARY',pause_started_monotonic=time.monotonic());_event(root,'PAUSED_BOUNDARY',after_node=name,owned_cleanup_verified=True);break
            else:
                require(all(e['status']=='COMPLETE_VALIDATED' for e in state['nodes'].values()) and state['nodes'],'EMPTY_OR_INCOMPLETE_REQUIRED_PLAN')
                state['status']='COMPLETE_VALIDATED';_event(root,'RUN_COMPLETE',nodes=len(state['nodes']))
        except BaseException as exc:
            if 'entry' in locals() and entry['status']=='RUNNING':entry.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
            state['status']='FAILED_OR_BLOCKED';_event(root,'CONTROLLER_STOPPED',error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            state['active_controller_seconds']+=time.monotonic()-started;atomic(root/'checkpoint.json',state)
        return {'status':state['status'],'run':str(root),'skipped':completed_before,'completed_nodes':[k for k,v in state['nodes'].items() if v['status']=='COMPLETE_VALIDATED'],'controller_sessions':state['controller_sessions']}


def run(plan,local,output,allow_gpu=False,stop_after_node=None):
    require(allow_gpu,'Explicit --allow-gpu required; no launch or GPU query performed')
    return _drive(_initial(plan,local,output),local,allow_gpu,stop_after_node)


def bind(plan,local,output,metadata_only=False):
    """Inspect a prepared execution layout; never authorize or launch a job."""
    from .prepared_live import bind_live_plan,binding_layout,contract
    from .suite_inputs import load_sources
    compiled=check_frozen(read(Path(plan)));out=Path(output)
    require(not out.exists(),'Refusing existing binding output')
    sources=Path(compiled['bindings']['sources_path']);loaded=load_sources(sources)
    if metadata_only:
        value=binding_layout(compiled,loaded)
    else:
        local_value=read(Path(local));clock=read(Path(local_value['task_clock']))
        selected={k:v for k,v in local_value.items() if k!='campaign_reservation_root'}
        context=bind_live_plan(compiled,sources,selected,clock)
        value={'schema_version':1,'status':'CPU_BOUND_NOT_EXECUTED','binding_profile':context['binding_profile'],
            'counts':compiled['semantic']['counts'],'execution_units':execution_units(compiled,context),
            'context':context,'artifact_presence':'RESOLVED_LOCAL_CONFIG_PIN_AND_WEIGHT_STAT_ONLY',
            'runtime_import_check':'NOT_RUN','loopback_check':'NOT_RUN','GPU_queries':0,'model_loads':0,'tokenizer_calls':0}
    root=new_output(output);write(root/'BINDING_PLAN.json',value);write(root/'LIVE_BINDING_CONTRACT.json',contract(value.get('binding_profile',value.get('profile','prepared_smol_live_v1'))))
    return {k:v for k,v in value.items() if k!='context'}


def resume(run,local=None,allow_gpu=False,stop_after_node=None):
    """Resume validated evidence; noop means no dispatch, not no checkpoint repair."""
    return _drive(Path(run).absolute(),local,allow_gpu,stop_after_node)


def status(run,output=None):
    root=Path(run).absolute();plan=_frozen(root,current_source=False);state=read(root/'checkpoint.json');verified=[]
    for node in plan['units']:
        if state['nodes'][node['node_id']]['status']=='COMPLETE_VALIDATED':_completion(root,plan,state,node);verified.append(node['node_id'])
    result={'status':state['status'],'run_id':state['run_id'],'nodes':state['nodes'],'verified_completed':verified,'controller_sessions':state['controller_sessions'],'active_controller_seconds':state['active_controller_seconds'],'paused_seconds':state['paused_seconds'],'GPU_queries':0,'model_loads':0,'new_POSTs':0}
    if output:write(new_output(output)/'STATUS.json',result)
    return result


def replay(run,output):
    from ..analysis.suite_results import replay as calculate
    return calculate(run,output)
