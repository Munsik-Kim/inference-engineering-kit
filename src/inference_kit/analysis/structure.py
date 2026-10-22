"""Task-neutral, data-only CPU replay of selected structure observations.

The acquisition retains native SDPA outputs and reconstructs only selected
probability rows. This reader never imports a model/runtime or executes bundle
code. Observer controls and model-pair contrasts are separate panels.
"""
import collections
import math
from pathlib import Path

from ..core.evidence import open_bundle
from ..core.io import read, write, require, digest, sha, safe_name
from ..core.state import derived_output, preserve_failure
from ..metrics.attention import row_metrics, vector_metrics
from ..metrics.token import log_softmax
from ..probes.blobs import load
from ..probes.contracts import Observation
from .token_sources import _verify, _plan, _lines, _coordinate
from .tokens import write_jsonl, csv_output, source_identity

KINDS = ('attention_row', 'attention_output', 'reconstructed_AV', 'output_projection')
PASSES = ('OFF1', 'OFF2', 'ON')
CONTROL_POLICY = {'id': 'd1_off_repeat_and_on_v1', 'absolute_tolerance': 1e-5,
                  'passes': list(PASSES)}


def metric_contract():
    return {'schema_version': 1, 'metric_family': 'selected_structure_d1', 'version': '1',
        'attention': 'Existing attention.row_metrics v1: full-key L1, JS, entropy; no subset renormalization',
        'vectors': 'Existing attention.vector_metrics v1: SSE/reference energy NMSE; cosine; energy<=1e-12 undefined',
        'native_AV': 'Native SDPA return in diagnostic runtime, not deployed Marlin output',
        'reconstructed_AV': 'Same-arm reconstructed AV vs native SDPA AV is a separate implementation diagnostic',
        'output_projection': 'Native o_proj output before residual; one layer/query vector with no head axis',
        'controls': CONTROL_POLICY,
        'aggregation': 'Row mean and source-document macro mean separately per task/layer/kind; no combined task score',
        'missing': 'Unknown/incomparable/integrity-unverified produce null metrics; no zero imputation',
        'evidence_mass': 'Sum saved full-key probabilities over validated content-overlap tokens; labels excluded; no causal claim',
        'cpu_replay_tolerance': {'absolute': 1e-10, 'relative': 1e-9},
        'GPU_actions': 0, 'model_loads': 0, 'tokenizer_loads': 0, 'archive_code_executed': False}


def _member(root, manifest, path):
    relative = Path(path).relative_to(root).as_posix()
    require(relative in manifest['files'], 'Read outside frozen raw manifest: ' + relative)
    return Path(path)


def _numeric(root, manifest, directory, observation):
    metadata = observation['numeric']
    _member(root, manifest, directory / safe_name(metadata['path']))
    return load(directory, metadata)


def _control_selection(row):
    """Reject vacuous/ambiguous observer controls before any PASS reduction."""
    targets = row.get('targets'); full = row.get('full_vocab_target_positions')
    require(isinstance(targets, list) and targets, 'Empty observer target selection')
    require(isinstance(full, list) and full, 'Empty observer full-logit selection')
    ids = row.get('input_ids'); require(isinstance(ids, list) and ids, 'Missing observer input sequence')
    positions = []
    for target in targets:
        require(isinstance(target, dict), 'Invalid observer target coordinate')
        position = target.get('token_position')
        require(type(position) is int and 0 < position < len(ids), 'Observer target position outside sequence')
        require(type(target.get('logit_position')) is int and target['logit_position'] == position-1
                and type(target.get('target_token_id')) is int and target['target_token_id'] == ids[position],
                'Observer target/logit shift or token ID differs')
        positions.append(position)
    require(len(set(positions)) == len(positions), 'Duplicate observer target selection')
    require(all(type(position) is int for position in full) and len(set(full)) == len(full)
            and set(full) <= set(positions), 'Invalid/duplicate observer full-logit selection')
    return positions


def _selection(plan):
    selection = plan['structure_selection']; rows = plan['samples']['rows']
    require(rows and len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate/empty structure sample IDs')
    config = plan['samples']['model_config']; layers = selection['layers']; heads = selection['heads']
    require(layers and len(set(layers)) == len(layers) and all(type(i) is int and 0 <= i < config['num_hidden_layers'] for i in layers), 'Invalid structure layer selection')
    require(heads and len(set(heads)) == len(heads) and all(type(i) is int and 0 <= i < config['num_attention_heads'] for i in heads), 'Invalid structure head selection')
    require(config['num_attention_heads'] % config['num_key_value_heads'] == 0, 'Invalid structure head mapping')
    require(set(selection['rows']) == {r['sample_id'] for r in rows}, 'Structure selection/sample identity differs')
    for row in rows:
        ids = row['input_ids']; require(ids and row['sequence_identity'] == digest(ids), 'Structure sequence identity differs')
        positions = _control_selection(row)
        require(all(type(i) is int and 0 <= i < plan['samples']['vocabulary_size'] for i in ids), 'Invalid structure token IDs')
        mask = row['attention_mask']; require(len(mask) == len(ids) and all(type(i) in (int, bool) and i in (0, 1) for i in mask), 'Invalid structure attention mask')
        require(len(row['loss_mask']) == len(row['special_token_mask']) == len(ids), 'Structure loss/special mask axis differs')
        require(all(row['loss_mask'][t] and mask[t] and not row['special_token_mask'][t]
                    and ids[t] not in row.get('special_token_ids', []) for t in positions),
                'Observer target is masked or special')
        selected = selection['rows'][row['sample_id']]; queries = selected['queries']
        require(queries and len(set(queries)) == len(queries) and all(type(q) is int and 0 <= q < len(ids) and mask[q] for q in queries), 'Invalid structure query selection')
        roles = selected['query_roles']; require(set(roles) == {str(q) for q in queries}, 'Structure query role axis differs')
        for q in queries:
            require(isinstance(roles[str(q)], list) and roles[str(q)] and len(set(roles[str(q)])) == len(roles[str(q)]), 'Empty/duplicate query role aliases')
            require(all(role in ('last_prompt', 'before_last_reference') for role in roles[str(q)]), 'Unknown structure query role')
            valid = [t for t, loss in enumerate(row['loss_mask']) if t > 0 and loss and not row['special_token_mask'][t]
                     and row['input_ids'][t] not in row.get('special_token_ids', [])]
            require(valid, 'No valid reference query in frozen structure input')
            expected = {'last_prompt': row['prompt_token_boundary']-1, 'before_last_reference': valid[-1]-1}
            require(all(q == expected[role] for role in roles[str(q)]), 'Structure query role/global position differs')
            if 'last_prompt' in roles[str(q)]:
                require(not row['loss_mask'][q], 'last_prompt query is not prompt-only')
                offsets = row.get('offsets_codepoints')
                if offsets is not None: require(offsets[q][1] <= len(row['prompt']), 'Prompt query crosses reference boundary')
    return {r['sample_id']: r for r in rows}


def _identity(row, layer, head, query, roles):
    return {**{key: row.get(key) for key in ('task_id', 'task_version', 'dataset_identity', 'sample_id', 'source_id', 'split', 'policy_id', 'reference_role')},
        'layer': layer, 'head': head, 'query_position': query, 'query_roles': roles,
        'sequence_identity': row['sequence_identity']}


def _expected(plan, rows):
    selection = plan['structure_selection']
    for sid, row in rows.items():
        queries = selection['rows'][sid]
        for layer in selection['layers']:
            for query in queries['queries']:
                roles = queries['query_roles'][str(query)]
                for kind in KINDS:
                    for head in [None] if kind == 'output_projection' else selection['heads']:
                        yield (sid, kind, layer, head, query), _identity(row, layer, head, query, roles)


def _structure_coordinate(observation, row, plan):
    c = observation['coordinates']; kind = observation['kind']; selection = plan['structure_selection']
    require(c.get('pass_id') == 'ON', 'Structure observations must belong to ON pass')
    for key, want in (('sample_id', row['sample_id']), ('task_id', row['task_id']), ('task_version', row['task_version']),
        ('source_id', row['source_id']), ('split', row['split']), ('reference_role', row['reference_role']),
        ('sequence_identity', row['sequence_identity']), ('token_id_space', plan['samples']['token_id_space']),
        ('tokenizer_identity', plan['samples']['tokenizer_identity'])):
        require(c.get(key) == want, 'Structure coordinate identity mismatch: ' + key)
    layer, query = c.get('layer'), c.get('query_position')
    require(type(layer) is int and layer in selection['layers'], 'Unselected structure layer')
    selected = selection['rows'][row['sample_id']]
    require(type(query) is int and query in selected['queries'], 'Unselected structure query')
    require(c.get('query_roles') == selected['query_roles'][str(query)], 'Structure query alias mismatch')
    require(c.get('prefix_hash') == digest({'token_ids': row['input_ids'][:query+1], 'token_id_space': plan['samples']['token_id_space']}), 'Structure prefix differs')
    expected_module = 'model.layers.' + str(layer) + '.self_attn' + ('.o_proj' if kind == 'output_projection' else '')
    require(c.get('module_path') == expected_module, 'Structure module/layer mismatch')
    head = None if kind == 'output_projection' else c.get('head')
    if kind == 'output_projection':
        require(c.get('head') is None and c.get('kv_head') is None, 'Output projection must have no head axis')
    else:
        require(type(head) is int and head in selection['heads'], 'Unselected structure head')
        config = plan['samples']['model_config']; expected = head // (config['num_attention_heads'] // config['num_key_value_heads'])
        require(c.get('kv_head') == expected and type(c.get('kv_head')) is int, 'Structure query/KV head mapping mismatch')
    if kind != 'output_projection':
        axis = list(range(len(row['input_ids'])))
        allowed = [i <= query and bool(row['attention_mask'][i]) for i in axis]
        require(c.get('key_positions') == axis, 'Structure full key axis differs')
        require(c.get('allowed_keys') == allowed and all(type(m) is bool for m in c['allowed_keys']), 'Structure global causal/padding mask differs')
    return row['sample_id'], kind, layer, head, query


def _execution(directory, arm, plan, root, manifest):
    path = directory / 'execution.json'
    if not path.is_file(): return None
    _member(root, manifest, path); execution = read(path)
    for key, want in (('arm_id', arm['arm_id']), ('artifact_identity', arm['artifact']['artifact_identity']),
        ('recipe_identity', arm['artifact']['recipe_identity']), ('tokenizer_identity', plan['samples']['tokenizer_identity']),
        ('runtime_identity', plan['runtime_identity']), ('execution_mode', arm['execution_mode']),
        ('model_id', plan['pair']['model_id']), ('model_revision', plan['pair']['revision'])):
        require(execution.get(key) == want, 'Structure execution identity mismatch: ' + key)
    expected_kind = 'synthetic_contract' if plan.get('evidence_kind') == 'synthetic_contract' else 'live_model_diagnostic'
    require(execution['evidence_kind'] == expected_kind, 'Structure synthetic/live source mismatch')
    return execution


def _read_arm(root, manifest, plan, arm, rows, progress):
    job = progress.get('jobs', {}).get('main-' + arm['arm_id'])
    directory = root / safe_name(job['directory']) if job else None
    execution = _execution(directory, arm, plan, root, manifest) if directory else None
    result = None
    if directory and (directory / 'result.json').is_file():
        _member(root, manifest, directory / 'result.json'); result = read(directory / 'result.json')
    if job and job['status'] == 'COMPLETE':
        require(execution is not None and result is not None and execution['status'] == result['status'] == 'MEASURED', 'Structure completion/result mismatch')
        require(execution.get('phase') == 'main', 'Structure execution phase differs')
    restoration = True
    if execution and execution['evidence_kind'] == 'live_model_diagnostic':
        restoration = result is not None and all(result.get(k) is True for k in ('all_hooks_interfaces_restored', 'no_inplace_changes',
            'weight_version_unchanged', 'buffer_version_unchanged', 'source_weight_stat_unchanged'))
    structural, tokens, ids = {}, {}, set()
    if directory and (directory / 'observations.jsonl').is_file():
        require(execution is not None, 'Observation without execution identity')
        _member(root, manifest, directory / 'observations.jsonl')
        for observation in _lines(directory / 'observations.jsonl'):
            Observation(**observation).validate(); sid = observation['sample_id']
            require(sid in rows, 'Unknown structure observation sample')
            require(observation['observation_id'] not in ids, 'Duplicate structure observation ID'); ids.add(observation['observation_id'])
            require(observation['execution_id'] == execution['execution_id'] and observation['label'] == execution['execution_mode'] and observation['evidence_kind'] == execution['evidence_kind'], 'Structure observation/execution mismatch')
            kind = observation['kind']; row = rows[sid]
            if kind in KINDS:
                key = _structure_coordinate(observation, row, plan); mapping = structural
                if observation['status'] == 'MEASURED':
                    shape = observation['numeric']['shape']; config = plan['samples']['model_config']
                    wanted = len(row['input_ids']) if kind == 'attention_row' else config['hidden_size'] if kind == 'output_projection' else config['hidden_size'] // config['num_attention_heads']
                    require(shape == [wanted], 'Structure numeric axis mismatch')
                    paths = {'attention_row': ('reconstructed_from_native_qk',),
                        'attention_output': ('native_returned_SDPA_AV',),
                        'reconstructed_AV': ('reconstructed_AV_from_FP32_row_and_native_V',),
                        'output_projection': ('native_o_proj_forward_hook', 'native_o_proj_forward_hook_before_residual')}
                    require(observation['details'].get('measurement_path') in paths[kind], 'Unsupported structure measurement path')
                    values = _numeric(root, manifest, directory, observation)
                    if kind == 'attention_row': row_metrics(values, values, observation['coordinates']['allowed_keys'])
            else:
                require(kind in ('target_logprob', 'full_logits'), 'Unknown structure raw observation kind')
                c = observation['coordinates']; require(c.get('pass_id') in PASSES, 'Missing/unknown observer control pass')
                for field, want in (('sample_id', sid), ('token_id_space', plan['samples']['token_id_space']), ('tokenizer_identity', plan['samples']['tokenizer_identity']), ('sequence_identity', row['sequence_identity'])):
                    require(c.get(field) == want, 'Control token identity mismatch: ' + field)
                require(all(type(c.get(k)) is int for k in ('token_position', 'logit_position', 'target_token_id')), 'Control token indices must be integer')
                _coordinate(row, c, plan); position = c['token_position']
                require(position in {t['token_position'] for t in row['targets']}, 'Control token outside frozen targets')
                if kind == 'full_logits': require(position in row['full_vocab_target_positions'], 'Control logits outside frozen full selection')
                key = (sid, c['pass_id'], kind, position); mapping = tokens
                if observation['status'] == 'MEASURED':
                    details = observation['details']; normalization = plan['probe_spec']['normalization']
                    require(details.get('normalization_dtype') == normalization['dtype'] and details.get('probability_stage') == normalization['probability_stage'] and details.get('log_base') == normalization['log_base'], 'Control probability contract mismatch')
                    require(details.get('consistency_policy') == plan['probe_spec']['consistency_policy'], 'Control scalar/full policy mismatch')
                    wanted = 1 if kind == 'target_logprob' else plan['samples']['vocabulary_size']
                    require(observation['numeric']['shape'] == [wanted], 'Control token numeric axis mismatch')
                    if kind == 'full_logits': require(details.get('full_vocabulary') is True, 'Control full vocabulary flag missing')
                    values = _numeric(root, manifest, directory, observation)
                    if kind == 'target_logprob': require(values[0] <= 0, 'Positive control target logprob')
            require(key not in mapping, 'Duplicate structure/control coordinate'); mapping[key] = observation
    return {'job': job, 'directory': directory, 'execution': execution, 'structure': structural, 'tokens': tokens,
        'native_preservation_receipt_verified': restoration}


def _value(root, manifest, side, observation):
    if observation is None or observation['status'] != 'MEASURED': return None
    return _numeric(root, manifest, side['directory'], observation)


def observer_controls(root, manifest, plan, rows, sides):
    policy = plan['structure_spec']['observer_consistency_policy']
    require(policy == CONTROL_POLICY, 'Observer control policy differs from registered pre-execution contract')
    tolerance = policy['absolute_tolerance']; control_rows, consistency = [], []
    integrity = {}
    for arm, side in sides.items():
        integrity[arm] = {}
        for sid, row in rows.items():
            _control_selection(row)
            sample_controls = []; sample_consistency = []
            for target in row['targets']:
                position = target['token_position']
                for kind in ('target_logprob', 'full_logits'):
                    if kind == 'full_logits' and position not in row['full_vocab_target_positions']: continue
                    values = {p: _value(root, manifest, side, side['tokens'].get((sid, p, kind, position))) for p in PASSES}
                    for other in ('OFF2', 'ON'):
                        a, b = values['OFF1'], values[other]
                        if a is not None and b is not None:
                            require(len(a) == len(b) and len(a) > 0, 'Observer control numeric axis mismatch/empty')
                            require(all(math.isfinite(x) for x in a+b), 'Non-finite observer control numeric value')
                            error = max(abs(a[i]-b[i]) for i in range(len(a)))
                        else: error = None
                        status = 'UNKNOWN' if error is None else 'MEASURED' if error <= tolerance else 'TOLERANCE_VIOLATION'
                        item = {'arm_id': arm, 'task_id': row['task_id'], 'sample_id': sid, 'kind': kind, 'token_position': position,
                            'reference_pass': 'OFF1', 'candidate_pass': other, 'maximum_absolute_difference': error,
                            'absolute_tolerance': tolerance, 'status': status, 'within_tolerance': None if error is None else error <= tolerance,
                            'reference': _reference(side['tokens'].get((sid, 'OFF1', kind, position)), side, root),
                            'candidate_observation': _reference(side['tokens'].get((sid, other, kind, position)), side, root)}
                        control_rows.append(item); sample_controls.append(item)
                if position in row['full_vocab_target_positions']:
                    for pass_id in PASSES:
                        scalar = _value(root, manifest, side, side['tokens'].get((sid, pass_id, 'target_logprob', position)))
                        logits = _value(root, manifest, side, side['tokens'].get((sid, pass_id, 'full_logits', position)))
                        error = abs(log_softmax(logits)[target['target_token_id']]-scalar[0]) if scalar is not None and logits is not None else None
                        allowed = plan['probe_spec']['consistency_policy']['absolute_tolerance_nat']
                        item = {'arm_id': arm, 'task_id': row['task_id'], 'sample_id': sid, 'pass_id': pass_id, 'token_position': position,
                            'absolute_error_nat': error, 'absolute_tolerance_nat': allowed,
                            'status': 'UNKNOWN' if error is None else 'MEASURED' if error <= allowed else 'TOLERANCE_VIOLATION',
                            'within_tolerance': None if error is None else error <= allowed,
                            'scalar_observation': _reference(side['tokens'].get((sid, pass_id, 'target_logprob', position)), side, root),
                            'full_observation': _reference(side['tokens'].get((sid, pass_id, 'full_logits', position)), side, root)}
                        consistency.append(item); sample_consistency.append(item)
            completed = side['job'] is not None and side['job']['status'] == 'COMPLETE' and side['execution'] is not None and side['execution']['status'] == 'MEASURED'
            good = completed and side['native_preservation_receipt_verified'] and all(r['status'] == 'MEASURED' for r in sample_controls + sample_consistency)
            off_good = all(r['status'] == 'MEASURED' for r in sample_controls if r['candidate_pass'] == 'OFF2')
            on_good = all(r['status'] == 'MEASURED' for r in sample_controls if r['candidate_pass'] == 'ON')
            integrity[arm][sid] = {'status': 'VERIFIED' if good else 'INTEGRITY_UNVERIFIED',
                'OFF_repeat': 'WITHIN_TOLERANCE' if off_good else 'VARIATION_OR_MISSING',
                'OFF_ON': 'WITHIN_TOLERANCE' if on_good else 'OBSERVER_EFFECT_POSSIBLE' if off_good else 'VARIATION_NOT_SEPARATED',
                'job_status': side['job']['status'] if side['job'] else 'NOT_RUN',
                'native_preservation_receipt_verified': side['native_preservation_receipt_verified'],
                'requested_controls': len(sample_controls), 'measured_controls': sum(r['status'] == 'MEASURED' for r in sample_controls)}
    return control_rows, consistency, integrity


def _evidence_positions(row):
    span = row.get('reference_evidence_span')
    if span is None: return {'status': 'MISSING_SPAN_METADATA', 'positions': None}
    if span.get('status') == 'NOT_APPLICABLE': return {'status': 'NOT_APPLICABLE', 'positions': None}
    require(span.get('status') == 'AVAILABLE', 'Unknown preserved evidence span status')
    require(span.get('coordinate_space') == 'joint_input_ids', 'Evidence span coordinate space differs')
    text = row['prompt']; cp = span['prompt_codepoint_span']; byte = span['prompt_utf8_byte_span']; content = span['content']
    require(isinstance(cp, list) and len(cp) == 2 and 0 <= cp[0] < cp[1] <= len(text), 'Invalid evidence codepoint span')
    require(text[cp[0]:cp[1]] == content and digest(content) == span['content_hash'], 'Evidence content/codepoint/hash mismatch')
    require(text.encode('utf-8')[byte[0]:byte[1]].decode('utf-8') == content, 'Evidence UTF8 span mismatch')
    offsets = row.get('offsets_codepoints')
    if offsets is None: return {'status': 'MISSING_SPAN_METADATA', 'positions': None}
    positions = [i for i, (a, b) in enumerate(offsets) if a < b and a < cp[1] and cp[0] < b]
    require(positions == span['overlapping_token_positions'] and positions and max(positions) < row['prompt_token_boundary'], 'Evidence token overlap/prompt boundary mismatch')
    return {'status': 'MEASURED', 'positions': positions, 'passage_id': span.get('passage_id'),
        'content_hash': span['content_hash'], 'overlap_rule': span.get('overlap_rule')}


def _reference(observation, side, root):
    if observation is None: return None
    return {'observation_id': observation['observation_id'], 'execution_id': observation['execution_id'],
        'observations_path': str((side['directory'] / 'observations.jsonl').relative_to(root)),
        'numeric': observation['numeric'], 'measurement_path': observation['details'].get('measurement_path')}


def _compatible(left, right, plan):
    a, b = left['coordinates'], right['coordinates']
    matching = ('task_id', 'task_version', 'source_id', 'split', 'reference_role', 'sample_id', 'tokenizer_identity',
        'token_id_space', 'sequence_identity', 'prefix_hash', 'layer', 'module_path', 'head', 'kv_head',
        'query_position', 'query_roles', 'key_positions', 'allowed_keys')
    reasons = [key for key in matching if a.get(key) != b.get(key)]
    require(left['kind'] == right['kind'], 'Structure observation kind mismatch')
    if left['details'].get('measurement_path') != right['details'].get('measurement_path'): reasons.append('measurement_path')
    for key in ('scaling', 'qk_point', 'causal_index_policy', 'key_scope', 'normalization_dtype', 'mask_source'):
        if left['details'].get(key) != right['details'].get(key): reasons.append(key)
    return reasons


def compare_structure(root, manifest, plan, rows, sides, integrity, baseline, candidate):
    results, reconstructed, masses = [], [], []
    spans = {sid: _evidence_positions(row) for sid, row in rows.items()}
    for key, identity in _expected(plan, rows):
        sid, kind, layer, head, query = key; left, right = sides[baseline], sides[candidate]
        a, b = left['structure'].get(key), right['structure'].get(key)
        result = {'identity': identity, 'kind': kind, 'baseline': baseline, 'candidate': candidate,
            'status': 'UNKNOWN', 'reasons': [], 'metrics': None,
            'reference': _reference(a, left, root), 'candidate_observation': _reference(b, right, root)}
        if a is None or b is None: result['reasons'] = ['MISSING_OBSERVATION_PAIR']
        elif a['status'] != 'MEASURED' or b['status'] != 'MEASURED': result['reasons'] = ['UNMEASURED_OBSERVATION_PAIR']
        else:
            reasons = _compatible(a, b, plan)
            if reasons: result.update(status='INCOMPARABLE', reasons=reasons)
            elif any(integrity[name][sid]['status'] != 'VERIFIED' for name in (baseline, candidate)):
                result.update(status='INTEGRITY_UNVERIFIED', reasons=['OBSERVER_CONTROL_OR_JOB_NOT_VERIFIED'])
            else:
                av, bv = _value(root, manifest, left, a), _value(root, manifest, right, b)
                metrics = row_metrics(av, bv, a['coordinates']['allowed_keys']) if kind == 'attention_row' else vector_metrics(av, bv)
                if kind == 'attention_row': metrics['entropy_delta_nat'] = metrics['entropy_candidate_nat']-metrics['entropy_reference_nat']
                result.update(status='MEASURED', metrics=metrics)
        results.append(result)
        if kind == 'attention_row':
            for name, side, observation in ((baseline, left, a), (candidate, right, b)):
                span = spans[sid]; value = _value(root, manifest, side, observation)
                item = {'identity': identity, 'arm_id': name, 'status': span['status'], 'mass': None,
                    'evidence_token_count': len(span['positions']) if span['positions'] is not None else None,
                    'valid_key_count': sum(observation['coordinates']['allowed_keys']) if observation else None,
                    'passage_id': span.get('passage_id'), 'evidence': _reference(observation, side, root)}
                if span['status'] == 'MEASURED':
                    if value is None: item['status'] = 'UNKNOWN'
                    elif integrity[name][sid]['status'] != 'VERIFIED': item['status'] = 'INTEGRITY_UNVERIFIED'
                    else: item['mass'] = math.fsum(value[i] for i in span['positions'])
                masses.append(item)
        if kind == 'attention_output':
            for name, side in sides.items():
                native = side['structure'].get(key); recon = side['structure'].get((sid, 'reconstructed_AV', layer, head, query))
                item = {'identity': identity, 'arm_id': name, 'kind': 'same_arm_reconstructed_vs_native_AV', 'status': 'UNKNOWN', 'metrics': None,
                    'native_reference': _reference(native, side, root), 'reconstructed_candidate': _reference(recon, side, root)}
                av, bv = _value(root, manifest, side, native), _value(root, manifest, side, recon)
                if av is not None and bv is not None:
                    if integrity[name][sid]['status'] != 'VERIFIED': item['status'] = 'INTEGRITY_UNVERIFIED'
                    else: item.update(status='MEASURED', metrics=vector_metrics(av, bv))
                reconstructed.append(item)
    return results, reconstructed, masses


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        identity = row['identity']; key = tuple(identity.get(k) for k in ('task_id', 'task_version', 'layer')) + (row['kind'],)
        groups[key].append(row)
    result = []
    for key, values in sorted(groups.items()):
        good = [v for v in values if v['status'] == 'MEASURED']; metrics = {}; documents = collections.defaultdict(list)
        for value in good: documents[value['identity']['sample_id']].append(value)
        names = sorted({k for v in good for k, x in v['metrics'].items() if type(x) in (int, float)})
        for name in names:
            numbers = [v['metrics'][name] for v in good if type(v['metrics'].get(name)) in (int, float)]
            per_document = [[v['metrics'][name] for v in group if type(v['metrics'].get(name)) in (int, float)] for group in documents.values()]
            means = [math.fsum(v)/len(v) for v in per_document if v]
            metrics[name] = {'row_mean': math.fsum(numbers)/len(numbers) if numbers else None,
                'document_macro_mean': math.fsum(means)/len(means) if means else None,
                'valid_rows': len(numbers), 'valid_documents': len(means)}
        result.append({'task_id': key[0], 'task_version': key[1], 'layer': key[2], 'kind': key[3],
            'requested_rows': len(values), 'measured_rows': len(good), 'documents': len({v['identity']['sample_id'] for v in values}),
            'status_counts': dict(collections.Counter(v['status'] for v in values)), 'metrics': metrics})
    return result


def replay_structure(run, output):
    """Verify an explicit run or safe ZIP, then recompute selected raw metrics."""
    out = derived_output(run, output)
    with preserve_failure(out), open_bundle(run) as root:
        if not (root / 'RAW_MANIFEST.json').is_file() or read(root / 'PLAN_FROZEN.json').get('task') != 'BENCH05-D1':
            candidates = [p.parent for p in root.rglob('RAW_MANIFEST.json') if (p.parent / 'PLAN_FROZEN.json').is_file() and read(p.parent / 'PLAN_FROZEN.json').get('task') == 'BENCH05-D1']
            require(len(candidates) == 1, 'Expected one explicit D1 raw run'); root = candidates[0]
        manifest = _verify(root, 'structure_probe_raw_v1'); plan = _plan(root / 'PLAN_FROZEN.json')
        require(plan['task'] == 'BENCH05-D1', 'Structure raw/plan type mismatch')
        rows = _selection(plan); progress = read(root / 'progress.json'); arms = plan['arms']
        require(len(arms) == 2 and len({a['arm_id'] for a in arms}) == 2, 'D1 requires two distinct explicit arm labels')
        baseline, candidate = [a['arm_id'] for a in arms]
        sides = {a['arm_id']: _read_arm(root, manifest, plan, a, rows, progress) for a in arms}
        controls, consistency, integrity = observer_controls(root, manifest, plan, rows, sides)
        results, reconstructed, masses = compare_structure(root, manifest, plan, rows, sides, integrity, baseline, candidate)
        projections = [r for r in results if r['kind'] == 'output_projection']
        structure_rows = [r for r in results if r['kind'] != 'output_projection']
        write_jsonl(out / 'structure-comparisons.jsonl', structure_rows); csv_output(out / 'structure-comparisons.csv', structure_rows)
        write_jsonl(out / 'output-projection-comparisons.jsonl', projections); csv_output(out / 'output-projection-comparisons.csv', projections)
        write_jsonl(out / 'reconstructed-native-AV.jsonl', reconstructed); csv_output(out / 'reconstructed-native-AV.csv', reconstructed)
        write_jsonl(out / 'evidence-content-mass.jsonl', masses); csv_output(out / 'evidence-content-mass.csv', masses)
        write_jsonl(out / 'observer-controls.jsonl', controls); write_jsonl(out / 'scalar-full-consistency.jsonl', consistency)
        control_receipt = {'policy': CONTROL_POLICY, 'status': 'VERIFIED' if all(v['status'] == 'VERIFIED' for arm in integrity.values() for v in arm.values()) else 'INTEGRITY_UNVERIFIED',
            'requested_comparisons': len(controls), 'status_counts': dict(collections.Counter(r['status'] for r in controls)),
            'maximum_absolute_difference': max((r['maximum_absolute_difference'] for r in controls if r['maximum_absolute_difference'] is not None), default=None),
            'per_arm_sample': integrity, 'not_deployment_equivalence': True}
        consistency_receipt = {'policy': plan['probe_spec']['consistency_policy'], 'requested_coordinates': len(consistency),
            'status_counts': dict(collections.Counter(r['status'] for r in consistency)),
            'maximum_absolute_error_nat': max((r['absolute_error_nat'] for r in consistency if r['absolute_error_nat'] is not None), default=None)}
        groups = aggregate(results); write(out / 'task-layer-aggregates.json', groups); csv_output(out / 'task-layer-aggregates.csv', groups)
        write(out / 'OBSERVER_CONTROL_RECEIPT.json', control_receipt); write(out / 'SCALAR_FULL_CONSISTENCY_RECEIPT.json', consistency_receipt)
        coverage = {'documents': len(rows), 'arms': len(arms), 'ON_structure_only': True, 'passes': list(PASSES),
            'per_kind': {kind: {'requested_pairs': sum(r['kind'] == kind for r in results), 'status_counts': dict(collections.Counter(r['status'] for r in results if r['kind'] == kind))} for kind in KINDS},
            'same_arm_reconstruction_pairs': len(reconstructed), 'observer_comparisons': len(controls), 'same_pass_scalar_full': len(consistency)}
        from .tokens import export_tokens, compare_tokens
        sources = {'schema_version': 1, 'kind': 'token_input_sources', 'sources': [{'source_id': 'd1-main',
            'adapter': 'token_probe_raw_v1', 'path': str(root), 'raw_manifest_sha256': sha(root / 'RAW_MANIFEST.json')}]}
        write(out / 'TOKEN_INPUT_SOURCES.json', sources)
        token_summary = {'coverage': {'status': 'NOT_COMPARED_INTEGRITY_UNVERIFIED', 'reason': 'Observer controls or source job are not verified'}}
        if not any(r['status'] == 'TOLERANCE_VIOLATION' for r in consistency):
            export_tokens(out / 'TOKEN_INPUT_SOURCES.json', out / 'token-view')
            if control_receipt['status'] == 'VERIFIED':
                token_summary = compare_tokens(out / 'token-view', baseline, candidate, out / 'token-metrics')
        write(out / 'TOKEN_COMPARISON_STATUS.json', {'status': 'MEASURED' if control_receipt['status'] == 'VERIFIED' else 'INTEGRITY_UNVERIFIED',
            'coverage': token_summary['coverage'], 'token_pass': 'ON', 'control_passes_pooled': False})
        summary = {'schema_version': 1, 'task': 'BENCH05-D1', 'analysis_source_digest': source_identity(),
            'raw_manifest_sha256': sha(root / 'RAW_MANIFEST.json'), 'plan_identity': plan['plan_identity'], 'acquisition_source_identity': plan['source_identity'],
            'baseline': baseline, 'candidate': candidate, 'coverage': coverage, 'aggregates': groups,
            'ON_token_coverage': token_summary['coverage'],
            'OBSERVER_INTEGRITY': control_receipt['status'], 'CPU_REPLAY': 'RAW_RECALCULATED',
            'DEPLOYMENT_EQUIVALENCE': 'NOT_ESTABLISHED', 'TASK_QUALITY': 'NOT_EVALUATED', 'CAUSAL_MECHANISM': 'NOT_ESTABLISHED',
            'GPU_actions': 0, 'model_loads': 0, 'tokenizer_loads': 0, 'downloads': 0, 'ledger_changes': 0, 'REMOTE_CI': 'NOT_RUN'}
        write(out / 'COVERAGE.json', coverage); write(out / 'METRIC_CONTRACT.json', metric_contract()); write(out / 'summary.json', summary)
        (out / 'REPORT_KO.md').write_text('# BENCH05-D1 구조 관측 CPU 재계산\n\n선택한 저장 행·출력만 재계산했습니다. 모델 정확도·배포 동등성·오류의 인과 기전은 평가하지 않았습니다.\n\n'
            + '문서 ' + str(len(rows)) + ', observer integrity: ' + control_receipt['status'] + '.\n'
            + 'o_proj는 head 축 없는 별도 표이며, native SDPA AV와 같은-arm reconstructed AV를 구분합니다.\n', encoding='utf-8')
        return summary
