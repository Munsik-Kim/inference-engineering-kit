# Derived from inference-lab; import/path boundaries changed for internal release01.
"""Several workload constraints, explicit treatment axes and retained quality policy."""
import math

from .io import fields, require
from .assess import constraint


def requirements(req):
    fields(req, ('schema_version', 'policy', 'model_id', 'model_revision', 'comparison_group_identity', 'identity',
                 'treatment_axes', 'memory', 'latency_constraints', 'quality', 'allow_retained_quality', 'quality_evidence', 'evidence_level'))
    require(req['schema_version'] == 3 and req['policy'] == 'DEMO', 'Unsupported multi-workload requirements')
    require(req['treatment_axes'] == ['weight_artifact', 'kv_pool_capacity'], 'Only two declared treatment axes are comparable')
    fields(req['identity'], ('gpu_uuid', 'runtime_identity', 'workload_identity', 'instrumentation_identity', 'fixed_profile_identity', 'tokenizer_identity'))
    require(all(isinstance(v, str) and v for v in req['identity'].values()), 'Incomplete comparison identities')
    require(req['evidence_level'] in ('exploratory_point', 'screen_with_interval'), 'Unknown quality evidence level')
    require(req['quality_evidence'] in ('retained_checkpoint', 'exact_profile'), 'Unknown quality transfer policy')
    require(type(req['allow_retained_quality']) is bool, 'Retained quality must be explicitly allowed or refused')
    fields(req['memory'], ('scope', 'phase_scope', 'max_mib'))
    require(req['memory']['scope'] in ('whole_device_mib', 'process_mib', 'allocator_mib'), 'Unknown memory scope')
    def finite(v): return type(v) in (int, float) and math.isfinite(v)
    require(finite(req['memory']['max_mib']) and req['memory']['max_mib'] > 0, 'Invalid memory limit')
    require(req['latency_constraints'], 'Need at least one workload latency constraint')
    seen = set()
    for cell in req['latency_constraints']:
        fields(cell, ('input_length', 'output_tokens', 'concurrency', 'metric', 'statistic', 'max_seconds'))
        require(all(type(cell[k]) is int and cell[k] > 0 for k in ('input_length', 'output_tokens', 'concurrency')), 'Invalid workload dimensions')
        require(cell['metric'] in ('ttft_seconds', 'tpot_stream_estimate_seconds', 'http_complete_seconds') and cell['statistic'] in ('p50', 'p95'), 'Unknown latency measure')
        require(finite(cell['max_seconds']) and cell['max_seconds'] > 0, 'Invalid latency upper bound')
        identity = tuple(cell[k] for k in ('input_length', 'output_tokens', 'concurrency', 'metric', 'statistic'))
        require(identity not in seen, 'Duplicate latency constraint')
        seen.add(identity)
    fields(req['quality'], ('delta_floor', 'absolute_floor'))
    require(finite(req['quality']['delta_floor']) and -1 <= req['quality']['delta_floor'] <= 0, 'Invalid quality loss floor')
    require(finite(req['quality']['absolute_floor']) and 0 <= req['quality']['absolute_floor'] <= 1, 'Invalid absolute quality floor')
    return req


def combined(checks):
    states = [c['status'] for c in checks.values()]
    return 'FAIL' if 'FAIL' in states else 'PASS' if states and all(v == 'PASS' for v in states) else 'UNKNOWN'


def candidate(row, req):
    checks = {}
    matches = row.get('model_id') == req['model_id'] and row.get('model_revision') == req['model_revision']
    mismatches = [k for k, v in req['identity'].items() if row.get('identity', {}).get(k) != v]
    matches = matches and not mismatches and row.get('comparison_group_identity') == req['comparison_group_identity']
    matches = matches and row.get('treatment_axes') == req['treatment_axes']
    checks['comparability'] = constraint('PASS' if matches else 'UNKNOWN', reason=None if matches else {'identity_mismatches': mismatches, 'comparison_group_or_model_mismatch': True})
    validity = row.get('validity', {})
    safe = (row.get('kind') == 'live_measurement' and row.get('measurement_status') == 'VALID_LIVE_PROFILE'
            and validity.get('requests_complete') is True and validity.get('owned_resources_clean') is True
            and validity.get('server_rounds') == 3 and len(validity.get('parent_returncodes', [])) == 3
            and all(code == 0 for code in validity['parent_returncodes']))
    checks['live_execution'] = constraint('PASS' if safe else 'UNKNOWN', reason=None if safe else 'Partial/failed/historical evidence is not new complete profile evidence', evidence=row.get('evidence_files'))
    isolated = validity.get('comparison_status') == 'NO_INTERFERENCE_OBSERVED_WITHIN_OBSERVATION_LIMITS'
    checks['device_condition'] = constraint('PASS' if isolated else 'UNKNOWN', reason=validity.get('comparison_status'))
    memory = row.get('memory', {})
    correct_scope = memory.get('scope') == req['memory']['scope'] and memory.get('phase_scope') == req['memory']['phase_scope']
    observed = memory.get('max_mib')
    checks['memory'] = constraint('UNKNOWN', reason='Memory scope/phase missing or different', evidence=memory.get('evidence'))
    if correct_scope and observed is not None and isolated:
        checks['memory'] = constraint('PASS' if observed <= req['memory']['max_mib'] else 'FAIL', observed, evidence=memory.get('evidence'))
    vector = []
    for number, limit in enumerate(req['latency_constraints']):
        cell = next((c for c in row.get('performance', {}).get('cells', []) if all(c.get(k) == limit[k] for k in ('input_length', 'output_tokens', 'concurrency'))), None)
        value = cell.get(limit['metric'], {}).get(limit['statistic']) if cell else None
        state = 'UNKNOWN' if value is None or not isolated else 'PASS' if value <= limit['max_seconds'] else 'FAIL'
        checks['latency_' + str(number)] = constraint(state, value, 'Worst independent server-round statistic; homogeneous cells measured separately', row.get('performance', {}).get('evidence'))
        vector.append(value)
    operational = {'status': combined(checks), 'checks': checks}
    qchecks = {}
    q = row.get('retained_quality', {})
    transfer = row.get('quality_transfer', {})
    sanity = row.get('cache_sanity', {})
    transferable = (req['allow_retained_quality'] and req['quality_evidence'] == 'retained_checkpoint'
                    and transfer.get('identity_status') == 'MATCHED_ARTIFACT_RUNTIME_TOKENIZER_SCORING'
                    and sanity.get('status') == 'COMPLETE' and sanity.get('same_weight_prediction_match') is True)
    qchecks['quality_source'] = constraint('PASS' if transferable else 'UNKNOWN', reason='Conditional reuse of earlier checkpoint quality' if transferable else 'Exact-profile sufficient quality or permitted compatible retained quality missing', evidence=q.get('evidence'))
    delta, score, ci, absolute_ci = q.get('delta'), q.get('acc_norm'), q.get('delta_ci95'), q.get('absolute_acc_norm_ci95')
    if req['evidence_level'] == 'exploratory_point':
        qchecks['quality_loss'] = constraint('UNKNOWN' if delta is None else 'PASS' if delta >= req['quality']['delta_floor'] else 'FAIL', delta, 'DEMO point only', q.get('evidence'))
        qchecks['absolute_quality'] = constraint('UNKNOWN' if score is None else 'PASS' if score >= req['quality']['absolute_floor'] else 'FAIL', score, 'Retained subset point, not task usability', q.get('evidence'))
    else:
        qchecks['quality_loss'] = constraint('UNKNOWN' if ci is None else 'PASS' if ci[0] >= req['quality']['delta_floor'] else 'FAIL' if ci[1] < req['quality']['delta_floor'] else 'UNKNOWN', {'delta': delta, 'ci95': ci}, 'Boundary crossing remains INCONCLUSIVE', q.get('evidence'))
        qchecks['absolute_quality'] = constraint('UNKNOWN' if absolute_ci is None else 'PASS' if absolute_ci[0] >= req['quality']['absolute_floor'] else 'FAIL' if absolute_ci[1] < req['quality']['absolute_floor'] else 'UNKNOWN', {'acc_norm': score, 'ci95': absolute_ci}, 'BF16 self-delta zero does not certify absolute quality', q.get('evidence'))
    overall_checks = checks | qchecks
    state = combined(overall_checks) if safe and matches and isolated else 'UNKNOWN'
    return {'configuration_id': row['configuration_id'], 'arm': row['arm'], 'operational_constraints': operational,
            'overall_deployment_evidence': {'status': 'CONDITIONAL_OBSERVED_CANDIDATE' if state == 'PASS' else 'FAIL' if state == 'FAIL' else 'INSUFFICIENT_EVIDENCE',
                                            'checks': qchecks, 'quality_source_profile': transfer.get('quality_source_profile'),
                                            'transfer_assumptions': transfer.get('assumptions'), 'cache_sanity': sanity,
                                            'exact_profile_quality': row.get('exact_profile_quality', 'NOT_VERIFIED'),
                                            'evidence_label': 'DEMO_POINT_ONLY' if req['evidence_level'] == 'exploratory_point' else 'RETAINED_SUBSET_INTERVAL_ONLY'},
            'metric_vector': {'memory_mib': observed, 'latency_seconds': vector}, 'artifact': row.get('artifact'),
            'execution': row.get('execution'), 'reexecute_commands': row.get('reexecute_commands'), 'limitations': row.get('limitations', [])}


def assess(index, req):
    requirements(req)
    require(index.get('schema_version') == 3 and index.get('kind') == 'verified_result_index', 'STEP03 requires its verified result schema')
    rows = [candidate(r, req) for r in index['results']]
    eligible = [r for r in rows if r['overall_deployment_evidence']['status'] == 'CONDITIONAL_OBSERVED_CANDIDATE']
    pareto = []
    for row in eligible:
        vector = [row['metric_vector']['memory_mib']] + row['metric_vector']['latency_seconds']
        dominated = False
        for other in eligible:
            other_vector = [other['metric_vector']['memory_mib']] + other['metric_vector']['latency_seconds']
            if all(a <= b for a, b in zip(other_vector, vector)) and any(a < b for a, b in zip(other_vector, vector)):
                dominated = True
        if not dominated:
            pareto.append(row['configuration_id'])
    if eligible:
        status = 'CONDITIONAL_OBSERVED_CANDIDATE'
    elif rows and all(r['overall_deployment_evidence']['status'] == 'FAIL' for r in rows if r['operational_constraints']['checks']['comparability']['status'] == 'PASS') and any(r['operational_constraints']['checks']['comparability']['status'] == 'PASS' for r in rows):
        status = 'NO_FEASIBLE_CANDIDATE'
    else:
        status = 'INSUFFICIENT_EVIDENCE'
    return {'schema_version': 3, 'policy': 'DEMO', 'status': status, 'requirements': req, 'candidates': rows,
            'pareto_candidates': pareto, 'selection_reason': 'All workload conditions AND; each memory/latency objective stays separate',
            'workload_scope': 'Two homogeneous workloads measured independently; simultaneous mixture NOT_VERIFIED',
            'business_usability': 'NOT_VALIDATED', 'production_slo': 'NOT_VALIDATED'}
