# Derived from inference-lab selector; path verification moved to evidence adapter.
"""Local constraints: evidence gaps and incompatible workloads remain visible."""
import argparse
import math
from pathlib import Path

from .io import fields, read, require, write


def requirements(value):
    fields(value, ('schema_version','policy','model_id','model_revision','identity','memory','quality','latency','evidence_level'))
    require(value['schema_version']==2 and value['policy']=='DEMO','Unsupported requirements version/policy')
    require(value['evidence_level'] in ('exploratory_point','screen_with_interval','profiler_and_native_exit'),'Unsupported evidence level')
    fields(value['identity'],('gpu_uuid','runtime_identity','profile_identity','workload_identity','quality_task_identity','quality_split_identity','tokenizer_identity'))
    fields(value['memory'],('max_mib','scope'))
    require(value['memory']['scope'] in ('whole_device_mib','process_mib','allocator_mib'),'Unknown memory scope')
    fields(value['quality'],('delta_floor',),('absolute_floor',))
    fields(value['latency'],('metric','statistic','max_seconds','input_length','concurrency'))
    require(value['latency']['metric'] in ('ttft_seconds','tpot_stream_estimate_seconds','http_complete_seconds'),'Unknown latency metric')
    require(value['latency']['statistic'] in ('p50','p95'),'Unknown latency statistic')
    def number(x):return type(x) in (int,float) and math.isfinite(x)
    require(number(value['memory']['max_mib']) and value['memory']['max_mib']>0,'Invalid memory limit')
    require(number(value['latency']['max_seconds']) and value['latency']['max_seconds']>0,'Invalid latency limit')
    require(number(value['quality']['delta_floor']) and -1<=value['quality']['delta_floor']<=0,'Invalid quality delta floor')
    if 'absolute_floor' in value['quality']:
        require(number(value['quality']['absolute_floor']) and 0<=value['quality']['absolute_floor']<=1,'Invalid absolute quality floor')
    for name in ('input_length','concurrency'):
        require(type(value['latency'][name]) is int and value['latency'][name]>0,'Invalid latency workload dimension')
    require(all(isinstance(x,str) and x for x in value['identity'].values()),'Incomplete evidence identity')
    return value


def constraint(status, observed=None, reason=None, evidence=None):
    return {'status':status,'observed':observed,'reason':reason,'evidence':evidence}


def assess_candidate(row, req):
    checks={}; identity=row.get('identity',{}); live=row.get('kind')=='live_measurement'
    matches=row.get('model_id')==req['model_id'] and row.get('model_revision')==req['model_revision']
    mismatches=[k for k,v in req['identity'].items() if identity.get(k)!=v]
    comparable=matches and not mismatches
    checks['comparability']=constraint('PASS' if comparable else 'UNKNOWN', reason=None if comparable else {'different_model':not matches,'identity_mismatches':mismatches})
    validity=row.get('validity',{})
    safe=live and row.get('measurement_status')=='VALID_LIVE_SUBSET_AND_PROFILE' and validity.get('requests_complete') is True and validity.get('owned_resources_clean') is True and bool(validity.get('parent_returncodes')) and all(x==0 for x in validity['parent_returncodes'])
    checks['live_execution']=constraint('PASS' if safe else 'UNKNOWN',reason=None if safe else 'Historical/synthetic/failed/incomplete execution cannot be promoted',evidence=row.get('evidence_files'))
    memory=row.get('memory',{}); mem=memory.get('max_mib')
    checks['memory']=constraint('UNKNOWN',reason='Memory missing or scope differs')
    if memory.get('scope')==req['memory']['scope'] and mem is not None:
        checks['memory']=constraint('PASS' if mem<=req['memory']['max_mib'] else 'FAIL',mem,evidence=memory.get('evidence'))
    if memory.get('comparison_status')=='SHARED_GPU_INTERFERENCE':
        checks['memory']=constraint('UNKNOWN',mem,'Observed whole-device maximum includes unresolved shared-GPU interference; artifact comparison is uncertified',memory.get('evidence'))
    quality=row.get('quality',{}); delta=quality.get('delta');ci=quality.get('delta_ci95');score=quality.get('acc_norm')
    floor=req['quality']['delta_floor']
    checks['quality_metric_available']=constraint('PASS' if score is not None and 0<=score<=1 else 'UNKNOWN',score,'Official subset acc_norm point value required')
    observed_quality=delta is not None and delta>=floor
    if delta is None:
        checks['quality_loss']=constraint('UNKNOWN',reason='Live paired official acc_norm missing')
    elif req['evidence_level']=='exploratory_point':
        checks['quality_loss']=constraint('PASS' if observed_quality else 'FAIL',delta,'Point estimate screen only',quality.get('evidence'))
    elif ci is None:
        checks['quality_loss']=constraint('UNKNOWN',delta,'Paired interval missing',quality.get('evidence'))
    else:
        checks['quality_loss']=constraint('PASS' if ci[0]>=floor else 'FAIL' if ci[1]<floor else 'UNKNOWN',
                                           {'delta':delta,'ci95':ci},'Subset interval screen; crossing boundary is INCONCLUSIVE',quality.get('evidence'))
    absolute=req['quality'].get('absolute_floor')
    if absolute is not None:
        interval=quality.get('absolute_acc_norm_ci95')
        if req['evidence_level']=='exploratory_point':state='UNKNOWN' if score is None else 'PASS' if score>=absolute else 'FAIL'
        else:state='UNKNOWN' if interval is None else 'PASS' if interval[0]>=absolute else 'FAIL' if interval[1]<absolute else 'UNKNOWN'
        checks['absolute_quality']=constraint(state,{'acc_norm':score,'ci95':interval},'Absolute subset floor; interval required above point evidence level',quality.get('evidence'))
        observed_quality=observed_quality and score is not None and score>=absolute
    cells=row.get('performance',{}).get('cells',[])
    cell=next((c for c in cells if c['input_length']==req['latency']['input_length'] and c['concurrency']==req['latency']['concurrency']),None)
    latency=cell.get(req['latency']['metric'],{}).get(req['latency']['statistic']) if cell else None
    checks['latency']=constraint('UNKNOWN' if latency is None else 'PASS' if latency<=req['latency']['max_seconds'] else 'FAIL',latency,
                                 'Worst observed server-round statistic; small closed-loop sample, not production tail SLO',row.get('performance',{}).get('evidence'))
    if cell and cell.get('comparison_status')=='SHARED_GPU_INTERFERENCE':
        checks['latency']=constraint('UNKNOWN',latency,'A requested server-round cell overlaps unresolved shared-GPU interference',row.get('performance',{}).get('evidence'))
    if req['evidence_level']=='profiler_and_native_exit':
        verified=validity.get('worker_native_exit')=='VERIFIED_ZERO' and validity.get('kernel_level')=='profiler_verified'
        checks['strict_runtime_evidence']=constraint('PASS' if verified else 'UNKNOWN',reason='Native worker exits and profiler required')
    feasible=safe and comparable and all(c['status']=='PASS' for c in checks.values())
    observed=safe and comparable and observed_quality and checks['memory']['status']=='PASS' and checks['latency']['status']=='PASS'
    return {'configuration_id':row['configuration_id'],'checks':checks,'observed_feasible':observed,'required_evidence_satisfied':feasible,
            'artifact':row.get('artifact'),'reexecute_command':row.get('reexecute_command'),'limitations':row.get('limitations',[]),
            'objectives':{'memory_mib':mem,'latency_seconds':latency,'quality_acc_norm':score}}


def assess(index, req):
    if req.get('schema_version') == 3:
        from .assess_multi import assess as assess_step03
        return assess_step03(index, req)
    req=requirements(req)
    require(index.get('schema_version')==2 and index.get('kind')=='verified_result_index','Unsupported/unverified result index')
    results=[assess_candidate(row,req) for row in index['results']]
    feasible=[r for r in results if r['required_evidence_satisfied']]
    pareto=[]
    for candidate in feasible:
        c=candidate['objectives'];dominated=False
        for other in feasible:
            if other is candidate:continue
            o=other['objectives']
            no_worse=o['memory_mib']<=c['memory_mib'] and o['latency_seconds']<=c['latency_seconds'] and o['quality_acc_norm']>=c['quality_acc_norm']
            strict=o!=c
            if no_worse and strict:dominated=True;break
        if not dominated:pareto.append(candidate['configuration_id'])
    considered=[r for r in results if r['checks']['comparability']['status']=='PASS']
    if feasible:status='FEASIBLE_WITH_REQUIRED_EVIDENCE'
    elif considered and all(r['checks']['live_execution']['status']=='PASS' and any(c['status']=='FAIL' for c in r['checks'].values()) for r in considered):status='NO_FEASIBLE_CANDIDATE'
    else:status='INSUFFICIENT_EVIDENCE'
    return {'schema_version':2,'policy':'DEMO','status':status,'requirements':req,'candidates':results,'pareto_candidates':pareto,
            'selection_reason':'Only comparable candidates satisfying all requested evidence enter Pareto set' if feasible else 'No recommendation from speed alone',
            'business_usability':'NOT_VALIDATED; relative quality constraints do not establish task usability','production_slo':'NOT_VALIDATED'}

