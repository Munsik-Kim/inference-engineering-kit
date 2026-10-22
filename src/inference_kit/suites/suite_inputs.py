"""Prepared-only Suite input validation. Never renders, tokenizes or executes a model.

Portable slices pin their own bytes and preserve parent identities. Validation of a
slice is not verification of every file in its parent acquisition manifest.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from ..core.io import digest, fields, loads, read, require, safe_name, sha
from ..tasks.registry import get_task
from ..probes.token_inputs import ROLE_SOURCES

SCOPE = 'SLICE_CONTRACT_VALIDATED_ONLY'


def _hash(value, label):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), label + ' SHA256 missing')


def _ids(value, vocabulary, label):
    require(isinstance(value, list) and value and all(type(t) is int and 0 <= t < vocabulary for t in value), label + ' invalid token IDs')


def validate_diagnostic(row, metadata):
    """Validate one preserved ProbeInput without changing C2's frozen 6-row wrapper."""
    vocabulary = metadata['vocabulary_size']
    _ids(row['input_ids'], vocabulary, 'diagnostic')
    ids = row['input_ids']; prompt = row['prompt']; reference = row['reference_text']; text = prompt + reference
    require(row['probe_input_identity'] == digest({k: v for k, v in row.items() if k != 'probe_input_identity'}), 'ProbeInput identity changed')
    require(row['prompt_hash'] == digest(prompt) and row['reference_hash'] == digest(reference) and
            row['full_text_hash'] == digest(text) and row['full_text'] == text and row['sequence_identity'] == digest(ids), 'Diagnostic text/prefix identity changed')
    require(row.get('reference_role') in ROLE_SOURCES and (row['task_id'], row['task_version'], row.get('reference_source')) == ROLE_SOURCES[row['reference_role']], 'Diagnostic reference role/source mismatch')
    require(row['tokenization']['tokenizer_identity'] == metadata['tokenizer_identity'] and
            row['tokenization']['token_id_space'] == metadata['token_id_space'], 'Diagnostic tokenizer/vocabulary identity changed')
    boundary = row['prompt_token_boundary']
    require(type(boundary) is int and 0 < boundary < len(ids), 'Invalid diagnostic boundary')
    require(row['attention_mask'] == [1] * len(ids) and all(type(v) is int for v in row['attention_mask']), 'Invalid attention mask')
    special = row['special_token_mask']; special_ids = row['special_token_ids']
    require(isinstance(special_ids, list) and all(type(x) is int for x in special_ids) and len(special_ids) == len(set(special_ids)), 'Invalid special IDs')
    require(isinstance(special, list) and len(special) == len(ids) and all(type(v) is int and v in (0, 1) for v in special), 'Invalid special mask')
    require(row['loss_mask'] == [int(t >= boundary and ids[t] not in special_ids) for t in range(len(ids))] and
            all(type(v) is int for v in row['loss_mask']), 'Invalid loss mask')
    cp = row.get('offsets_codepoints'); byte = row.get('offsets_utf8_bytes')
    require((cp is None) == (byte is None), 'Offset availability mismatch')
    if cp is not None:
        require(len(cp) == len(byte) == len(ids), 'Offset lengths mismatch')
        previous = -1
        for i in range(len(ids)):
            pair = cp[i]
            require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair), 'Invalid offset type')
            a, b = pair
            require(isinstance(byte[i], list) and len(byte[i]) == 2 and all(type(v) is int for v in byte[i]), 'Invalid byte offset type')
            require(0 <= a <= b <= len(text) and a >= previous and byte[i] == [len(text[:a].encode('utf-8')), len(text[:b].encode('utf-8'))], 'Invalid codepoint/byte offset')
            previous = a
        require(boundary == next((i for i, (_, b) in enumerate(cp) if b > len(prompt)), None), 'Joint boundary is not preserved offset boundary')
    else:
        require(row.get('offset_mapping_status') == 'MAPPING_UNAVAILABLE', 'Missing offset status')
    eligible = {t for t in range(1, len(ids)) if row['loss_mask'][t] == 1 and not special[t] and ids[t] not in special_ids}
    positions = []
    require(isinstance(row['targets'], list) and row['targets'], 'INVALID_EMPTY_PLAN: no diagnostic targets')
    for target in row['targets']:
        t = target['token_position']
        require(type(t) is int and t in eligible and t not in positions and type(target['logit_position']) is int and
                target['logit_position'] == t - 1 and type(target['target_token_id']) is int and target['target_token_id'] == ids[t], 'Invalid/duplicate target shift')
        if 'prefix_hash' in target:
            require(target['prefix_hash'] == digest({'token_ids': ids[:t], 'token_id_space': metadata['token_id_space']}), 'Target prefix identity changed')
        positions.append(t)
    full = row['full_vocab_target_positions']
    require(isinstance(full, list) and all(type(t) is int for t in full) and len(full) == len(set(full)) and set(full) <= set(positions), 'Invalid full positions')
    for field, span in row.get('value_spans', {}).items():
        require(isinstance(span, list) and len(span) == 2, 'Invalid literal span shape')
        a, b = span
        require(type(a) is int and type(b) is int and 0 <= a < b <= len(reference), 'Invalid literal span')
        if cp is not None:
            overlap = [t for t in sorted(eligible) if cp[t][0] < len(prompt) + b and cp[t][1] > len(prompt) + a]
            coverage = row.get('field_token_coverage', {}).get(field)
            require(overlap and isinstance(coverage, dict) and coverage.get('overlapping_positions') == overlap and
                    coverage.get('selected_positions') == [t for t in overlap if t in positions], 'Literal field token coverage mismatch')
    return row


def _generation(sample, source):
    generation = sample['generation']
    fields(generation, ('prepared', 'wire_payload', 'proof'))
    row = generation['prepared']; diagnostic = sample['diagnostic']; proof = generation['proof']
    fields(proof, ('parent_file', 'parent_file_sha256', 'line_number', 'line_sha256', 'raw_line'))
    _hash(proof['parent_file_sha256'], 'raw parent'); _hash(proof['line_sha256'], 'raw line')
    safe_name(proof['parent_file'])
    require(type(proof['line_number']) is int and proof['line_number'] > 0 and isinstance(proof['raw_line'], str), 'Invalid raw line proof')
    require(hashlib.sha256(proof['raw_line'].encode('utf-8')).hexdigest() == proof['line_sha256'], 'Raw wire line SHA changed')
    wire_record = loads(proof['raw_line'])
    body = generation['wire_payload']
    fields(body, ('prompt', 'max_tokens', 'temperature', 'top_p', 'n', 'stream', 'echo', 'ignore_eos', 'add_special_tokens'),
           ('model', 'stream_options', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'return_token_ids'))
    require(wire_record['request_body'] == body and wire_record['sample_id'] == sample['sample_id'] and
            wire_record['task_id'] == source['task_id'] and wire_record['task_version'] == source['task_version'] and
            wire_record['tokenizer_identity'] == source['model_metadata']['tokenizer_identity'] and
            wire_record['reference_used_for_generation'] is False, 'Generation raw provenance mismatch')
    require(body.get('prompt') == row['generation_input_ids'] and body.get('prompt') == diagnostic.get('generation_input_ids'), 'Generation payload differs from independently preserved prompt-only wire')
    _ids(body['prompt'], source['model_metadata']['vocabulary_size'], 'generation')
    require(row['sample_id'] == sample['sample_id'] and row['prompt_hash'] == diagnostic['prompt_hash'] and
            row['prompt'] == diagnostic['prompt'] and row['prompt_hash'] == digest(row['prompt']), 'Generation prepared prompt changed')
    require(body['prompt'] != diagnostic['input_ids'] and body['stream'] is True and body['echo'] is False and body['ignore_eos'] is False and
            type(body['max_tokens']) is int and body['max_tokens'] > 0 and type(body['temperature']) in (int, float) and
            body['temperature'] == 0 and type(body['top_p']) in (int, float) and body['top_p'] == 1 and
            type(body['n']) is int and body['n'] == 1 and body['add_special_tokens'] is False, 'Generation policy/gold suffix leakage')
    adapter = get_task(source['task_id'], source['task_version'])
    request = adapter.requests(sample['authored_sample'] | {'generation_input_ids': body['prompt']}, 'suite-static-validation', source['model_metadata']['tokenizer_identity'])[0]
    require(request.request_kind == 'generate_until' and request.payload == {'token_ids': body['prompt']}, 'Wrong generation adapter/payload')


def validate_slice(value):
    fields(value, ('schema_version', 'kind', 'source_id', 'task_id', 'task_version', 'request_kind', 'ancestry', 'model_metadata', 'samples', 'validation_scope'))
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and value['kind'] == 'prepared_suite_slice_v1', 'Unsupported prepared slice')
    require(value['validation_scope'] == SCOPE, 'Prepared slice must disclose limited parent scope')
    adapter = get_task(value['task_id'], value['task_version'])
    require(value['request_kind'] == adapter.request_kind, 'Wrong task request kind')
    require(isinstance(value['source_id'], str) and value['source_id'], 'Missing source ID')
    metadata = value['model_metadata']; require(type(metadata['vocabulary_size']) is int and metadata['vocabulary_size'] > 1, 'Invalid vocabulary')
    for key in ('tokenizer_identity', 'token_id_space'):
        _hash(metadata[key], key)
    ancestry = value['ancestry']
    fields(ancestry, ('parent_raw_manifest_sha256', 'parent_plan_sha256', 'parent_plan_identity', 'parent_acquisition_source_identity', 'parent_dataset_identity', 'scorer_identity', 'scorer_version', 'probe_inputs_sha256', 'probe_selection_identity', 'included_sample_ids', 'excluded_sample_count', 'scope'))
    for key in ('parent_raw_manifest_sha256', 'parent_plan_sha256', 'parent_plan_identity', 'parent_acquisition_source_identity', 'parent_dataset_identity', 'scorer_identity', 'probe_inputs_sha256', 'probe_selection_identity'):
        _hash(ancestry[key], key)
    require(type(ancestry['excluded_sample_count']) is int and ancestry['excluded_sample_count'] >= 0, 'Invalid excluded sample count')
    require(ancestry['scorer_version'] == adapter.scorer_version, 'Scorer contract changed')
    samples = value['samples']; require(isinstance(samples, list) and samples, 'INVALID_EMPTY_PLAN: prepared samples empty')
    require([s['sample_id'] for s in samples] == ancestry['included_sample_ids'] and len({s['sample_id'] for s in samples}) == len(samples), 'Duplicate/reordered sample selection')
    for sample in samples:
        optional = ('generation',) if adapter.request_kind == 'generate_until' else ('likelihood',)
        fields(sample, ('sample_id', 'authored_sample', 'diagnostic'), optional)
        require(optional[0] in sample, 'Missing prepared request payload')
        row = validate_diagnostic(sample['diagnostic'], metadata); authored = sample['authored_sample']
        require((row['task_id'], row['task_version'], row['sample_id']) == (value['task_id'], value['task_version'], sample['sample_id']) and
                authored['sample_id'] == sample['sample_id'], 'Prepared task/sample identity mismatch')
        require(row.get('source_bundle_identity') == ancestry['parent_raw_manifest_sha256'] and
                row.get('source_plan_identity') == ancestry['parent_plan_identity'] and
                row.get('source_scorer_identity') == ancestry['scorer_identity'] and
                row.get('dataset_identity') == ancestry['parent_dataset_identity'], 'Diagnostic/parent acquisition lineage mismatch')
        reference = adapter.reference(authored); text = reference.get('text', reference.get('reference_text'))
        if adapter.request_kind == 'choice_loglikelihood':
            likelihood = sample['likelihood']; fields(likelihood, ('requests', 'choice_labels', 'gold_index'))
            require(likelihood == {'requests': authored['requests'], 'choice_labels': authored['choice_labels'], 'gold_index': authored['gold']}, 'ARC all-choice order or source changed')
            requests = adapter.requests(authored, 'suite-static-validation', metadata['tokenizer_identity'], metadata['token_id_space'])
            require(len(requests) == len(authored['choices']) and [r.payload['choice_index'] for r in requests] == list(range(len(requests))), 'Incomplete ARC evaluation')
            gold = requests[authored['gold']].payload
            prompt = gold['context'].rstrip(); text = gold['context'][len(prompt):] + text
            require(row['prompt'] == prompt and row['input_ids'] == gold['input_ids'] and row['prompt_token_boundary'] == len(gold['context_ids']) and
                    row['gold_choice_index'] == authored['gold'], 'ARC gold diagnostic vs all-choice contract mismatch')
            for request in requests:
                _ids(request.payload['input_ids'], metadata['vocabulary_size'], 'choice')
        else:
            require(authored.get('document_hash') == digest(authored['document']) and row.get('document_hash') == authored['document_hash'], 'Source document identity changed')
            require(authored.get('sample_identity') == digest({k: v for k, v in authored.items() if k != 'sample_identity'}) and row.get('sample_identity') == authored['sample_identity'], 'Authored sample identity changed')
            _generation(sample, value)
        require(row['reference_text'] == text, 'External gold reference changed')
    return value


def load_sources(manifest_or_path, base=None):
    """Load explicitly pinned standalone slices; physical locator is not data identity."""
    if isinstance(manifest_or_path, (str, Path)):
        path = Path(manifest_or_path)
        require(path.is_file() and not path.is_symlink(), 'Missing/unsafe source manifest')
        manifest = read(path); root = path.parent.resolve(); manifest_sha = sha(path)
    else:
        manifest = deepcopy(manifest_or_path); root = Path(base or '.').resolve(); manifest_sha = digest(manifest)
    fields(manifest, ('schema_version', 'kind', 'sources'))
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and manifest['kind'] == 'suite_prepared_sources_v1', 'Unsupported Suite sources manifest')
    sources = manifest['sources']; require(isinstance(sources, list) and sources, 'INVALID_EMPTY_PLAN: no sources')
    require(all(isinstance(s, dict) for s in sources) and len({s.get('source_id') for s in sources}) == len(sources), 'Duplicate source ID')
    loaded, samples, receipts = [], [], []
    for source in sources:
        fields(source, ('source_id', 'path', 'sha256')); _hash(source['sha256'], 'slice')
        name = safe_name(source['path']); candidate = root / name
        require(candidate.is_file() and not candidate.is_symlink() and candidate.resolve().is_relative_to(root), 'Unsafe/missing prepared slice')
        require(sha(candidate) == source['sha256'], 'Prepared slice content hash changed')
        value = read(candidate)
        if value.get('kind') == 'prepared_task_source_v1':
            from .prepared_inputs import validate_source
            value = validate_source(value)
        else:
            value = validate_slice(value)
        require(value['source_id'] == source['source_id'], 'Source alias mismatch')
        loaded.append(value)
        samples.extend({'source_id': value['source_id'], 'task_id': value['task_id'], 'task_version': value['task_version'], 'request_kind': value['request_kind'], **deepcopy(s)} for s in value['samples'])
        receipts.append({'source_id': source['source_id'], 'sha256': source['sha256'], 'physical_locator': str(candidate.resolve()), 'scope': value['validation_scope'], 'artifact_presence': 'ARTIFACT_PRESENCE_NOT_CHECKED'})
    require(len({(s['task_id'], s['sample_id']) for s in samples}) == len(samples), 'Duplicate task/sample across Suite sources')
    identities = {digest({k: v['model_metadata'][k] for k in ('pair', 'model_config', 'tokenizer_identity', 'token_id_space', 'vocabulary_size')}) for v in loaded}
    require(len(identities) == 1, 'Suite input model/tokenizer/vocabulary mismatch')
    content = [{'source_id': s['source_id'], 'sha256': s['sha256']} for s in sources]
    scopes = {v['validation_scope'] for v in loaded}
    scope = next(iter(scopes)) if len(scopes) == 1 else 'MIXED_EXPLICIT_PREPARED_SCOPES'
    return {'schema_version': 1, 'kind': 'loaded_suite_sources_v1', 'sources': loaded, 'samples': samples,
            'model_metadata': deepcopy(loaded[0]['model_metadata']), 'receipts': receipts,
            'data_identity': digest(content), 'manifest_sha256': manifest_sha,
            'validation_scope': scope, 'gpu_actions': 0, 'model_loads': 0, 'tokenizer_loads': 0}
