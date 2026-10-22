"""D1 structure extension of configured capture, reusing its owned orchestration."""
import asyncio
from pathlib import Path
from ..core.io import read,write,require,sha,digest
from ..core.state import new_output
from ..core.paths import Paths
from ..probes.structure_inputs import select,validate_spec
from ..probes.token_inputs import validate_probe_inputs
from . import token_probe

WORKER={'module':'inference_kit.probes.structure_worker','structure':True}


def plan(spec_file,inputs_file,models_file,local_file,output):
    spec_file,inputs_file,models_file,local_file=[Path(p).expanduser().absolute() for p in (spec_file,inputs_file,models_file,local_file)]
    spec=validate_spec(read(spec_file));inputs=validate_probe_inputs(read(inputs_file),verify_sources=True)
    require(inputs['kind']=='selected_probe_inputs_v1' and len(inputs['rows'])==6 and len({r['task_id'] for r in inputs['rows']})==3,'D1 requires exactly the six existing prepared C2 inputs')
    local=read(local_file);paths=Paths.load(spec_file,local_file);prior=paths.resolve(local['prior_input_run'])
    manifest=read(prior/'RAW_MANIFEST.json');require(sha(prior/'RAW_MANIFEST.json')==local['prior_input_manifest_sha256'] and manifest['scope']=='token_probe_raw_v1','C2 input source manifest changed')
    prior_plan=read(prior/'PLAN_FROZEN.json');body=dict(prior_plan);identity=body.pop('plan_identity')
    require(digest(body)==identity and sha(prior/'PLAN_FROZEN.json')==manifest['files']['PLAN_FROZEN.json']['sha256'],'C2 input frozen plan changed')
    require(prior_plan['task']=='BENCH05-C2' and prior_plan['samples']==inputs,'D1 input/order/targets differ from exact C2 MAIN')
    selection=select(inputs,spec)
    result=token_probe._record(spec_file,inputs_file,models_file,local_file,prepared=inputs,spec_override=inputs['spec'],task_id='BENCH05-D1',worker_contract=WORKER)
    jobs=result['counts']['prior_task_gpu_jobs'];budget=spec['budget']
    require(jobs<budget['max_gpu_processes'],'D1 GPU job cap exhausted')
    origin=result['task_origin_used_seconds'];used=result['campaign']['used_seconds']
    result['gpu_allowance_seconds']=min(budget['task_gpu_wall_seconds']-(used-origin),7200-used)
    require(result['gpu_allowance_seconds']>budget['cleanup_reserve_seconds'],'D1 cleanup-safe allowance unavailable')
    result['suite']['budget']=budget|{'max_gpu_processes':budget['max_gpu_processes']-jobs,'raw_bytes_per_job':budget['raw_bytes_total']}
    result.update(raw_format='structure_probe_raw_v1',structure_spec=spec,structure_selection=selection,
      evidence_kind='planned_live_structure_probe',scope='D1 same prepared six-document engineering regression reuse',
      input_provenance={'C2_run_as_metadata':str(prior),'C2_raw_manifest_sha256':sha(prior/'RAW_MANIFEST.json'),
        'C2_plan_identity':prior_plan['plan_identity'],'C2_source_identity':digest(prior_plan['source_identity']),
        'C2_selection_identity':inputs['selection_identity'],'unchanged_prepared_rows_identity':inputs['rows_identity'],
        'same_input_order_targets_full_masks_offsets':True,'tokenizer_loads':0,'new_task_scores':False})
    result['input_files'].update({str(prior/'RAW_MANIFEST.json'):sha(prior/'RAW_MANIFEST.json'),str(prior/'PLAN_FROZEN.json'):sha(prior/'PLAN_FROZEN.json')})
    result['counts'].update(selection['counts'])
    result['counts'].update(forwards_per_arm=18,forward_calls_total=36,passes=['OFF1','OFF2','ON'],
      token_target_records_all_passes=6*inputs['counts']['target_positions_per_arm'],full_logit_records_all_passes=6*inputs['counts']['full_vocab_positions_per_arm'])
    result['resource_estimates'].update(selection['resource_estimates'])
    for key in ('selected_attention_rows','attention_output','output_projection'):
        if key in result['capability']['features']:result['capability']['features'][key].update(status='PLANNED_NOT_EXECUTED',evidence='Explicit D1 multi-layer selection; verified runtime structure required')
    result['attention_observations']='PLANNED_ON_ONLY';result['held_equal']+=['structure_coordinates','native_mask','observer_policy']
    result['plan_identity']=digest(result);out=new_output(output)
    for name,value in (('PLAN_FROZEN.json',result),('PROBE_INPUTS_FROZEN.json',inputs),('STRUCTURE_SPEC.json',spec),('SELECTION_RECEIPT.json',selection),('INPUT_PROVENANCE.json',result['input_provenance']),('CAPABILITY.json',result['capability'])):write(out/name,value)
    return result


def validate_frozen(plan):
    require(plan['raw_format']=='structure_probe_raw_v1' and plan['worker_contract']==WORKER,'D1 raw/worker contract mismatch')
    require(select(plan['samples'],plan['structure_spec'])==plan['structure_selection'],'Frozen structure selection/spec changed')
    require(plan['counts']['passes']==['OFF1','OFF2','ON'] and plan['counts']['forward_calls_total']==36,'D1 controls/forward count changed')
    return True


def run(plan,output):
    return asyncio.run(token_probe.managed(plan,output,task_id='BENCH05-D1',worker_contract=WORKER,
      raw_scope='structure_probe_raw_v1',extra_validator=validate_frozen))
