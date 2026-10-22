"""CPU-only common token inputs and deterministic, configurable capture selection.

Only these explicit ancestry adapters know historical B2/B3/B4 preparation.
No tokenizer, model, network, runtime SDK or CUDA operation is performed here.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

from ..core.io import digest, read, require, sha, safe_name
from ..tasks.registry import get_task

SELECTION = 'preserved_field_round_robin_uniform_v1'
ROLES = ('author_gold_json', 'dataset_gold_choice', 'author_answer_evidence')
ROLE_SOURCES = {
    'author_gold_json': ('tech_record_extract', '0.1', 'task_author_gold_json'),
    'dataset_gold_choice': ('arc_challenge', '1.0', 'dataset_answer_key_choice'),
    'author_answer_evidence': ('context_evidence_retrieval', '0.1', 'task_author_gold_answer_evidence_v1'),
}
ADAPTERS = {
    'bench05b2': ('BENCH05-B2-CONTRACT', 'bench05b2_contract_raw', 'author_gold_json'),
    'bench05b3': ('BENCH05-B3', 'bench05b3_arc_raw', 'dataset_gold_choice'),
    'bench05b4': ('BENCH05-B4', 'bench05b4_context_raw', 'author_answer_evidence'),
}


@dataclass(frozen=True)
class ProbeSpec:
    schema_version: int
    selection_policy: str
    selection_version: str
    reference_role_filter: list
    max_sequence_tokens: int
    max_target_positions_per_sample: int
    max_full_vocab_positions_per_sample: int
    observe_attention: bool
    normalization: dict
    consistency_policy: dict
    budget: dict
    seed: int = 909274

    @classmethod
    def from_record(cls, value):
        require(isinstance(value, dict), 'ProbeSpec must be a JSON object')
        require(set(value) <= set(cls.__dataclass_fields__), 'Unknown ProbeSpec fields')
        try:
            return cls(**deepcopy(value))
        except TypeError as exc:
            raise ValueError('Missing ProbeSpec fields') from exc

    from_dict = from_record

    def validate(self):
        value = asdict(self)
        require(type(self.schema_version) is int and self.schema_version == 1, 'Unsupported ProbeSpec schema')
        require(self.selection_policy == SELECTION and self.selection_version == '1', 'Unknown target selection contract')
        require(isinstance(self.reference_role_filter, list) and self.reference_role_filter and
                len(set(self.reference_role_filter)) == len(self.reference_role_filter) and
                set(self.reference_role_filter) <= set(ROLES), 'Unknown/duplicate/empty external-reference role filter')
        require(type(self.max_sequence_tokens) is int and 1 <= self.max_sequence_tokens <= 640, 'Probe sequence bound')
        require(type(self.max_target_positions_per_sample) is int and 1 <= self.max_target_positions_per_sample <= 32, 'Probe target cap')
        require(type(self.max_full_vocab_positions_per_sample) is int and
                0 <= self.max_full_vocab_positions_per_sample <= min(8, self.max_target_positions_per_sample), 'Probe full-vocabulary cap')
        require(self.observe_attention is False, 'C2 capture does not observe attention')
        require(self.normalization == {'dtype': 'float32', 'probability_stage': 'raw_pre_sampling', 'log_base': 'e'}, 'Probe normalization contract')
        require(self.consistency_policy == {'id': 'same_pass_saved_f32_v1', 'absolute_tolerance_nat': 2e-5,
                'scope': 'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'}, 'Declared same-pass consistency policy changed')
        required = {'task_gpu_wall_seconds', 'cleanup_reserve_seconds', 'max_gpu_processes', 'automatic_retries',
                    'probe_gpu_buffer_bytes', 'raw_bytes_total', 'minimum_cpu_available_bytes', 'minimum_disk_free_bytes'}
        require(isinstance(self.budget, dict) and set(self.budget) == required, 'Probe budget fields')
        require(all(type(v) is int for v in self.budget.values()), 'Probe budget values must be integers')
        b = self.budget
        require(0 < b['task_gpu_wall_seconds'] <= 600 and b['cleanup_reserve_seconds'] >= 120 and
                b['cleanup_reserve_seconds'] < b['task_gpu_wall_seconds'] and 1 <= b['max_gpu_processes'] <= 3 and
                b['automatic_retries'] == 0, 'Probe execution/cleanup budget')
        require(all(b[k] > 0 for k in ('probe_gpu_buffer_bytes', 'raw_bytes_total',
                'minimum_cpu_available_bytes', 'minimum_disk_free_bytes')), 'Probe resource budget must be positive')
        require(type(self.seed) is int, 'Probe seed must be an integer')
        return value


def validate_spec(value):
    return (value if isinstance(value, ProbeSpec) else ProbeSpec.from_record(value)).validate()


def _source(spec):
    require(spec.get('adapter') in ADAPTERS and isinstance(spec.get('source_id'), str) and spec['source_id'], 'Unknown/missing prepared source')
    require(isinstance(spec.get('path'), str) and spec['path'] and
            isinstance(spec.get('raw_manifest_sha256'), str) and len(spec['raw_manifest_sha256']) == 64, 'Prepared source needs explicit path and full manifest SHA256')
    root = Path(spec['path'])
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'Prepared source must be an explicit local directory')
    manifest_path = root / 'RAW_MANIFEST.json'
    require(sha(manifest_path) == spec['raw_manifest_sha256'], 'Pinned input source manifest changed')
    manifest = read(manifest_path)
    expected, scope, role = ADAPTERS[spec['adapter']]
    require(manifest['scope'] == scope, 'Prepared source scope mismatch')
    name = safe_name('PLAN_FROZEN.json'); path = root / name
    require(name in manifest['files'] and path.is_file() and not path.is_symlink(), 'Source manifest has no safe frozen plan')
    entry = manifest['files'][name]
    require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Frozen input plan size/hash changed')
    plan = read(path); body = dict(plan); identity = body.pop('plan_identity')
    require(digest(body) == identity and plan['task'] == expected, 'Frozen input plan identity/task changed')
    if spec['adapter'] == 'bench05b2':
        require(spec.get('policies', ['P00']) == ['P00'], 'C2 uses B2 P00 only, not a new prompt policy')
        rows = plan['samples']['by_policy']['P00']
    else:
        require('policies' not in spec or spec['policies'] in ([], [None]), 'Unexpected input prompt policy')
        rows = plan['samples']['rows']
    require(len(rows) == 4 and len({r['sample_id'] for r in rows}) == 4, 'Expected four original diagnostic inputs per task')
    authored = {r['sample_id']: r for r in plan['data']['rows']}
    adapter = get_task(plan['task_id'], plan['task_version'])
    for row in rows:
        require(row['sample_id'] in authored, 'Prepared source sample is absent from task data')
        original = authored[row['sample_id']]
        reference = adapter.reference(original)
        text = reference.get('text', reference.get('reference_text'))
        if spec['adapter'] == 'bench05b3':
            request = original['requests'][original['gold']]
            context = request['context']; prompt = context.rstrip()
            text = context[len(prompt):] + text
            require(row['prompt'] == prompt and row['input_ids'] == request['input_ids'] and
                    row['prompt_token_boundary'] == len(request['context_ids']), 'ARC original likelihood boundary/token IDs changed')
            require(row['gold_choice_index'] == original['gold'], 'ARC reference is not the external gold choice')
        require(row['task_id'] == plan['task_id'] and row['task_version'] == plan['task_version'] and
                row['reference_text'] == text, 'TaskAdapter external reference differs from frozen preparation')
    receipt = {'source_id': spec['source_id'], 'adapter': spec['adapter'], 'path': str(root),
      'raw_manifest_sha256': sha(manifest_path), 'plan_sha256': sha(path), 'plan_identity': identity,
      'source_acquisition_code_identity': digest(plan['source_identity']), 'dataset_identity': digest(plan['data']),
      'scorer_identity': plan.get('scorer_identity', plan['scorer']), 'model_id': plan['pair']['model_id'],
      'model_revision': plan['pair']['revision'], 'reference_role': role,
      'verified_scope': 'pinned manifest and its frozen prepared plan; no response/blob/model re-execution'}
    return plan, rows, receipt


def _row_identity(row):
    return digest({k: v for k, v in row.items() if k != 'probe_input_identity'})


def prepare_inputs(input_manifest_or_dict):
    if isinstance(input_manifest_or_dict, (str, Path)):
        path = Path(input_manifest_or_dict)
        manifest = read(path); manifest_identity = sha(path)
    else:
        manifest = deepcopy(input_manifest_or_dict); manifest_identity = digest(manifest)
    require(manifest.get('schema_version') == 1 and manifest.get('kind') == 'token_input_sources', 'Unsupported input manifest')
    sources = manifest['sources']
    require(len(sources) == 3 and len({s['source_id'] for s in sources}) == 3 and
            {s['adapter'] for s in sources} == set(ADAPTERS), 'C2 requires one explicit source for each of three tasks')
    rows, receipts, selections = [], [], []
    common = None
    for source in sources:
        plan, prepared, receipt = _source(source)
        metadata = {key: deepcopy(plan['samples'][key]) for key in
                    ('model_config', 'tokenizer_files', 'tokenizer_identity', 'token_id_space', 'vocabulary_size')}
        if common is None:
            common = metadata
        else:
            require(common == metadata, 'Prepared model/tokenizer/vocabulary differ across tasks')
        ranked = sorted(prepared, key=lambda row: hashlib.sha256((plan['task_id'] + '|' + row['sample_id']).encode('utf-8')).hexdigest())
        chosen = ranked[:2]
        selections.append({'source_id': source['source_id'], 'task_id': plan['task_id'],
          'original_sample_ids': [r['sample_id'] for r in prepared], 'selected_sample_ids': [r['sample_id'] for r in chosen],
          'rule': 'sha256(task_id + "|" + sample_id) ascending, first two among original four diagnostic rows'})
        for original in chosen:
            row = deepcopy(original)
            row.update(common_source_id=source['source_id'], source_bundle_identity=receipt['raw_manifest_sha256'],
              source_plan_identity=receipt['plan_identity'], dataset_identity=receipt['dataset_identity'],
              source_scorer_identity=receipt['scorer_identity'], reference_role=receipt['reference_role'],
              original_targets=deepcopy(original['targets']),
              original_full_vocab_target_positions=deepcopy(original['full_vocab_target_positions']),
              original_selection_policy=original.get('selection_policy'),
              original_field_token_coverage=deepcopy(original.get('field_token_coverage', {})),
              offset_mapping_status=('MAPPING_UNAVAILABLE' if original.get('offsets_codepoints') is None and
                                     original.get('offsets_utf8_bytes') is None else 'AVAILABLE'),
              full_text=original['prompt'] + original['reference_text'])
            row['probe_input_identity'] = _row_identity(row)
            rows.append(row)
        receipts.append(receipt)
    result = {'schema_version': 1, 'kind': 'common_probe_inputs_v1', 'rows': rows, 'rows_identity': digest(rows),
      'sources': receipts, 'input_manifest_identity': manifest_identity,
      'selection_receipt': {'sample_selection': selections, 'samples_per_task': 2, 'model_results_used_for_selection': False},
      'scope': 'preserved external references; historical inputs reused, not new held-out evaluation',
      'tokenizer_loads': 0, 'model_loads': 0, 'gpu_actions': 0, **common}
    result['input_identity'] = digest(result)
    validate_probe_inputs(result)
    return result


def _eligible(row):
    special = set(row.get('special_token_ids', [])); mask = row['special_token_mask']
    return [i for i, value in enumerate(row['input_ids']) if i > 0 and row['loss_mask'][i] == 1 and
            not mask[i] and value not in special]


def validate_probe_inputs(inputs, *, verify_sources=False):
    require(isinstance(inputs, dict) and inputs.get('schema_version') == 1, 'Invalid common ProbeInputs schema')
    kind = inputs.get('kind')
    require(kind in ('common_probe_inputs_v1', 'selected_probe_inputs_v1'), 'Unknown ProbeInputs kind')
    identity_name = 'input_identity' if kind == 'common_probe_inputs_v1' else 'selection_identity'
    require(inputs.get(identity_name) == digest({k: v for k, v in inputs.items() if k != identity_name}), 'Frozen ProbeInputs identity changed')
    rows = inputs['rows']
    expected_count = 6 if kind == 'common_probe_inputs_v1' else len(rows)
    require(0 < len(rows) <= 6 and len(rows) == expected_count and digest(rows) == inputs['rows_identity'], 'Invalid frozen ProbeInputs row count/identity')
    require(len({(r['task_id'], r['sample_id']) for r in rows}) == len(rows), 'Duplicate common task/sample input')
    if kind == 'common_probe_inputs_v1':
        require(len({r['task_id'] for r in rows}) == 3, 'Expected three task types')
    else:
        require(all(r['reference_role'] in inputs['spec']['reference_role_filter'] for r in rows), 'Selected input violates reference role filter')
    require(all(sum(r['task_id'] == task for r in rows) == 2 for task in {r['task_id'] for r in rows}), 'Two inputs per selected task required')
    vocabulary = inputs['vocabulary_size']
    require(type(vocabulary) is int and vocabulary > 1, 'Invalid common vocabulary')
    for row in rows:
        require(row.get('probe_input_identity') == _row_identity(row), 'Frozen probe row identity changed')
        ids = row['input_ids']; text = row['prompt'] + row['reference_text']; boundary = row['prompt_token_boundary']
        require(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < vocabulary for i in ids), 'Invalid common input token IDs')
        require(row['reference_role'] in ROLES and row['reference_hash'] == digest(row['reference_text']) and
                row['prompt_hash'] == digest(row['prompt']) and row['full_text_hash'] == digest(text) and
                row['full_text'] == text and row['sequence_identity'] == digest(ids), 'Common text/sequence/reference identity changed')
        require((row['task_id'], row['task_version'], row['reference_source']) == ROLE_SOURCES[row['reference_role']], 'Common task/external-reference role changed')
        require(row['tokenization']['tokenizer_identity'] == inputs['tokenizer_identity'] and
                row['tokenization']['token_id_space'] == inputs['token_id_space'], 'Common tokenizer identity changed')
        offsets, byte_offsets = row.get('offsets_codepoints'), row.get('offsets_utf8_bytes')
        require((offsets is None) == (byte_offsets is None), 'Both codepoint/UTF-8 offsets must be available or missing together')
        require(type(boundary) is int and 0 < boundary < len(ids), 'Common prompt/reference boundary changed')
        if offsets is None:
            require(row.get('offset_mapping_status') == 'MAPPING_UNAVAILABLE', 'Missing offsets need explicit MAPPING_UNAVAILABLE status')
        else:
            require(isinstance(offsets, list) and isinstance(byte_offsets, list) and len(offsets) == len(byte_offsets) == len(ids), 'Common offset coverage changed')
            previous = -1
            for (a, b), actual_bytes in zip(offsets, byte_offsets):
                require(type(a) is int and type(b) is int and 0 <= a <= b <= len(text) and a >= previous, 'Invalid common codepoint offset')
                require(actual_bytes == [len(text[:a].encode('utf-8')), len(text[:b].encode('utf-8'))], 'Common UTF-8/codepoint offset mismatch')
                previous = a
            require(boundary == next((i for i, (_, b) in enumerate(offsets) if b > len(row['prompt'])), None), 'Common prompt/reference offset boundary changed')
        require(row['attention_mask'] == [1] * len(ids), 'C2 uses unpadded C1 sequences only')
        require(len(row['special_token_mask']) == len(ids) and all(type(v) is int and v in (0, 1) for v in row['special_token_mask']), 'Common special-token mask changed')
        require(row['loss_mask'] == [int(i >= boundary and ids[i] not in row['special_token_ids']) for i in range(len(ids))], 'Common causal reference loss mask changed')
        eligible = set(_eligible(row)); seen = set()
        for key in ('targets', 'original_targets'):
            seen_for_key = set()
            for target in row[key]:
                t = target['token_position']
                require(t in eligible and t not in seen_for_key and target['logit_position'] == t - 1 and target['target_token_id'] == ids[t], 'Invalid/duplicate common target coordinate')
                if 'prefix_hash' in target:
                    require(target['prefix_hash'] == digest({'token_ids': ids[:t], 'token_id_space': inputs['token_id_space']}), 'Common target prefix changed')
                seen_for_key.add(t)
            if key == 'targets':
                seen = seen_for_key
        full = row['full_vocab_target_positions']
        require(isinstance(full, list) and len(set(full)) == len(full) and set(full) <= seen, 'Full-vocabulary positions must be unique selected targets')
        original_positions = {t['token_position'] for t in row['original_targets']}
        require(set(row['original_full_vocab_target_positions']) <= original_positions and
                len(set(row['original_full_vocab_target_positions'])) == len(row['original_full_vocab_target_positions']), 'Original full-vocabulary ancestry changed')
        spans = row.get('value_spans', {})
        for field, (a, b) in spans.items():
            require(type(a) is int and type(b) is int and 0 <= a < b <= len(row['reference_text']), 'Invalid reference field span')
            expected = ([t for t in sorted(eligible) if offsets[t][0] < len(row['prompt']) + b and offsets[t][1] > len(row['prompt']) + a]
                        if offsets is not None else None)
            if expected is not None:
                require(expected, 'Reference field has no token overlap')
            for coverage_key, selected_set in (('field_token_coverage', seen), ('original_field_token_coverage', original_positions)):
                coverage = row.get(coverage_key, {}).get(field)
                if expected is None and coverage is None:
                    continue
                positions = expected if expected is not None else coverage['overlapping_positions']
                require(coverage is not None and set(positions) <= eligible and coverage['overlapping_positions'] == positions and
                        coverage['selected_positions'] == [t for t in positions if t in selected_set], 'Reference field/selection coordinate mismatch')
    if verify_sources:
        for source in inputs['sources']:
            root = Path(source['path'])
            require(sha(root / 'RAW_MANIFEST.json') == source['raw_manifest_sha256'] and
                    sha(root / 'PLAN_FROZEN.json') == source['plan_sha256'], 'Input source ancestry changed after freeze')
    return inputs


def _uniform(values, count):
    count = min(count, len(values))
    # Integer floor, including both endpoints when count>1; no floating rounding.
    return [values[(i * (len(values) - 1)) // max(1, count - 1)] for i in range(count)]


def select_inputs(inputs, spec):
    validate_probe_inputs(inputs)
    spec = validate_spec(spec)
    rows, receipt = [], []
    for original in inputs['rows']:
        if original['reference_role'] not in spec['reference_role_filter']:
            continue
        row = deepcopy(original); eligible = _eligible(row)
        require(len(row['input_ids']) <= spec['max_sequence_tokens'], 'Input exceeds configured sequence cap; never truncate')
        cap = spec['max_target_positions_per_sample']; selected = []
        def add(position):
            if position in eligible and position not in selected and len(selected) < cap:
                selected.append(position)
        for target in row['original_targets']:
            add(target['token_position'])
        add(eligible[0])
        fields = row.get('original_field_token_coverage', {})
        field_positions = [fields[name]['overlapping_positions'] for name in sorted(fields)]
        for positions in field_positions:
            if positions:
                add(positions[0]); add(positions[-1])
        for i in range(max((len(v) for v in field_positions), default=0)):
            for positions in field_positions:
                if i < len(positions):
                    add(positions[i])
        remaining = [t for t in eligible if t not in selected]
        for position in _uniform(remaining, cap - len(selected)):
            add(position)
        selected.sort()
        targets = [{'token_position': t, 'logit_position': t - 1, 'target_token_id': row['input_ids'][t],
          'prefix_hash': digest({'token_ids': row['input_ids'][:t], 'token_id_space': inputs['token_id_space']})} for t in selected]
        full, full_cap = [], spec['max_full_vocab_positions_per_sample']
        for t in row['original_full_vocab_target_positions']:
            if t in selected and t not in full and len(full) < full_cap:
                full.append(t)
        for t in _uniform([t for t in selected if t not in full], full_cap - len(full)):
            if t not in full:
                full.append(t)
        full.sort()
        row.update(targets=targets, full_vocab_target_positions=full, selection_policy=SELECTION,
          field_token_coverage={name: {**value, 'selected_positions': [t for t in value['overlapping_positions'] if t in selected]}
                                for name, value in fields.items()})
        row['probe_input_identity'] = _row_identity(row)
        rows.append(row)
        receipt.append({'task_id': row['task_id'], 'sample_id': row['sample_id'],
          'eligible_target_positions': len(eligible), 'requested_target_cap': cap, 'actual_target_positions': len(targets),
          'requested_full_vocab_cap': full_cap, 'actual_full_vocab_positions': len(full),
          'preserved_original_positions': [t['token_position'] for t in original['original_targets'] if t['token_position'] in selected],
          'selected_positions': selected, 'selected_full_vocab_positions': full,
          'padding_or_repeated_positions_added': False})
    cfg = inputs['model_config']; vocabulary = inputs['vocabulary_size']
    max_t = max(len(r['targets']) for r in rows); max_seq = max(len(r['input_ids']) for r in rows)
    raw_target_bytes = 2 * sum(len(r['targets']) for r in rows) * 4
    raw_logits_bytes = 2 * sum(len(r['full_vocab_target_positions']) for r in rows) * vocabulary * 4
    buffer = max_t * vocabulary * 12 + max_seq * cfg['hidden_size'] * 4
    require(buffer <= spec['budget']['probe_gpu_buffer_bytes'], 'Configured selected-position buffer exceeds explicit budget')
    require(raw_target_bytes + raw_logits_bytes <= spec['budget']['raw_bytes_total'], 'Configured raw numeric bytes exceed explicit budget')
    result = {key: deepcopy(inputs[key]) for key in ('sources', 'model_config', 'tokenizer_files', 'tokenizer_identity', 'token_id_space', 'vocabulary_size')}
    result.update(schema_version=1, kind='selected_probe_inputs_v1', rows=rows, rows_identity=digest(rows),
      input_identity=inputs['input_identity'], source_input_identity=inputs['input_identity'], spec=spec, spec_identity=digest(spec),
      selection_receipt={'policy': SELECTION, 'version': '1', 'rows': receipt,
        'priority': 'preserved original targets; first reference; field endpoints; sorted-field round robin; integer-floor uniform fill',
        'full_priority': 'preserved original full positions within selected targets; integer-floor uniform fill',
        'uniform_rule': 'index=floor(i*(n-1)/(count-1)), singleton takes first; no replacement'},
      counts={'samples': len(rows), 'tasks': len({r['task_id'] for r in rows}), 'target_positions_per_arm': sum(len(r['targets']) for r in rows),
        'full_vocab_positions_per_arm': sum(len(r['full_vocab_target_positions']) for r in rows), 'attention_rows_per_arm': 0},
      resource_estimates={'maximum_sequence_tokens': max_seq, 'maximum_selected_targets_per_sample': max_t,
        'extra_probe_gpu_buffer_upper_bytes': buffer, 'raw_target_scalar_bytes_both_arms': raw_target_bytes,
        'raw_full_logits_bytes_both_arms': raw_logits_bytes,
        'raw_numeric_bytes_both_arms': raw_target_bytes + raw_logits_bytes,
        'raw_estimate_scope': 'numeric scalar/full-logit blobs only; JSON/log metadata tracked separately'})
    result['selection_identity'] = digest(result)
    validate_probe_inputs(result)
    return result
