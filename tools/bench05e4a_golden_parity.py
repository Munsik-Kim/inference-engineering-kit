#!/usr/bin/env python3
"""Explicit, stdlib-only exact check of three retokenized historical samples.

This checker neither tokenizes nor executes archive code. It compares only the
named stored sequence/request contracts, while reporting changed selection as a
separate policy. New source/runtime/creation provenance is never used to excuse
text, token ID, boundary, or mask mismatches.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

DIAGNOSTIC_EXACT = ('task_id', 'task_version', 'sample_id', 'source_id', 'prompt', 'prompt_hash',
    'reference_text', 'reference_hash', 'reference_source', 'reference_role', 'full_text', 'full_text_hash',
    'input_ids', 'sequence_identity', 'prompt_token_boundary', 'attention_mask', 'loss_mask',
    'special_token_ids', 'special_token_mask', 'offsets_codepoints', 'offsets_utf8_bytes')
NATURAL_EXACT = ('generation_input_ids', 'prompt_separate_ids', 'reference_separate_ids',
    'joint_equals_separate_concat', 'boundary_crossing_token', 'token_strings', 'value_spans', 'document_hash', 'sample_identity')
TOKENIZATION_EXACT = ('chat_template_applied', 'bos_added_by_encoder', 'eos_added_by_encoder',
    'padding_added', 'tokenizer_identity', 'token_id_space')
CHOICE_EXACT = ('choice_index', 'context', 'continuation', 'context_ids', 'continuation_ids', 'input_ids')
POLICY_FIELDS = ('max_tokens', 'temperature', 'top_p', 'n', 'stream', 'echo', 'ignore_eos', 'add_special_tokens')
RETRIEVAL_SPANS = ('source_span_policy', 'passage_content_spans', 'reference_evidence_span')


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key: ' + key); result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite: ' + value)))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(a, b):
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(exact(a[i], b[i]) for i in range(len(a)))
    return a == b


def _field(checks, name, baseline, actual):
    require(name in baseline and name in actual, 'Required golden field missing: ' + name)
    good = exact(baseline[name], actual[name]); checks[name] = good
    require(good, 'GOLDEN_PARITY_MISMATCH: ' + name)


def compare_sample(expected, actual):
    old = expected['diagnostic']; new = actual['diagnostic']; checks = {}
    for key in DIAGNOSTIC_EXACT: _field(checks, key, old, new)
    for key in TOKENIZATION_EXACT: _field(checks, 'tokenization.' + key, {'tokenization.' + key: old['tokenization'][key]}, {'tokenization.' + key: new['tokenization'][key]})
    if 'generation' in expected:
        require('generation' in actual and 'likelihood' not in actual, 'Natural/likelihood role changed')
        for key in NATURAL_EXACT: _field(checks, key, old, new)
        a = expected['generation']['wire_payload']; b = actual['generation']['wire_payload']
        for key in ('prompt',) + POLICY_FIELDS: _field(checks, 'generation.' + key, {'generation.' + key: a[key]}, {'generation.' + key: b[key]})
        require(b['prompt'] == new['generation_input_ids'] and b['prompt'] != new['input_ids'], 'Generation contains reference joint sequence')
        require(actual['generation']['preparation_proof']['reference_used_for_generation'] is False,
                'Generation reference role changed')
    else:
        require('likelihood' in actual and 'generation' not in actual, 'Likelihood/natural role changed')
        a = expected['likelihood']; b = actual['likelihood']
        for key in ('choice_labels', 'gold_index'): _field(checks, 'likelihood.' + key, {'likelihood.' + key: a[key]}, {'likelihood.' + key: b[key]})
        require(len(a['requests']) == len(b['requests']) >= 2, 'ARC choice coverage changed')
        require([r['choice_index'] for r in a['requests']] == [r['choice_index'] for r in b['requests']] == list(range(len(a['requests']))), 'ARC choice order changed')
        choices = actual['authored_sample']['choices']
        for index in range(len(a['requests'])):
            for key in CHOICE_EXACT:
                name = 'choice' + str(index) + '.' + key
                _field(checks, name, {name: a['requests'][index][key]}, {name: b['requests'][index][key]})
            require(b['requests'][index]['normalization_char_length'] == len(choices[index]), 'Original ARC character denominator changed')
        for key in ('gold_choice_index', 'original_context', 'original_continuation', 'moved_trailing_context_whitespace'):
            _field(checks, key, old, new)
    span_scope = {'reference_literal': 'EXACT' if 'value_spans' in old else 'NEW_MAPPING_NO_HISTORICAL_COUNTERPART',
                  'source_evidence_content': 'NOT_PROVIDED_BY_THIS_TASK', 'source_evidence_label': 'NOT_EXPLICITLY_MAPPED'}
    if old['task_id'] == 'context_evidence_retrieval':
        for key in RETRIEVAL_SPANS: _field(checks, key, old, new)
        span_scope['source_evidence_content'] = new['reference_evidence_span']['status']
        # Labels and content are distinct. Existing content mapping explicitly excludes [Pxx].
        span_scope['source_evidence_label'] = 'NO_SEPARATE_TOKEN_SPAN; content mapping excludes label'
    old_positions = [t['token_position'] for t in old['targets']]
    new_positions = [t['token_position'] for t in new['targets']]
    return {'task_id': old['task_id'], 'sample_id': old['sample_id'], 'status': 'EXACT_SEQUENCE_REQUEST_PARITY',
      'exact_field_count': len(checks), 'exact_fields': checks, 'joint_tokens': len(new['input_ids']),
      'selection': {'status': 'SEPARATE_DECLARED_POLICY', 'old_target_count': len(old_positions), 'new_target_count': len(new_positions),
        'common_target_count': len(set(old_positions) & set(new_positions)),
        'old_only_target_positions': sorted(set(old_positions) - set(new_positions)), 'new_only_target_positions': sorted(set(new_positions) - set(old_positions)),
        'old_full_positions': old['full_vocab_target_positions'], 'new_full_positions': new['full_vocab_target_positions']},
      'span_scope': span_scope,
      'new_preparation_provenance': 'New policy/source/runtime/locator/time retained separately; no sequence mismatch waived'}


def run(prepared_sources, expected_path):
    manifest_path = Path(prepared_sources); manifest = read(manifest_path); root = manifest_path.parent.resolve()
    require(manifest.get('kind') == 'suite_prepared_sources_v1' and manifest.get('sources'), 'Missing prepared source list')
    samples = {}; source_pins = []
    for source in manifest['sources']:
        name = source['path']; p = PurePosixPath(name)
        require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and str(p) == name, 'Unsafe prepared source path')
        path = root / name; require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe prepared source')
        require(sha(path) == source['sha256'], 'Prepared source SHA changed')
        value = read(path)
        for sample in value['samples']:
            key = (value['task_id'], sample['sample_id']); require(key not in samples, 'Duplicate prepared task/sample'); samples[key] = sample
        source_pins.append({'source_id': source['source_id'], 'sha256': source['sha256']})
    expected = read(expected_path); rows = expected['rows']; require(rows, 'No golden expectations')
    keys = [(row['task_id'], row['sample_id']) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == set(samples), 'Golden sample coverage mismatch')
    results = [compare_sample(row['expected_sample'], samples[(row['task_id'], row['sample_id'])]) for row in rows]
    return {'schema_version': 1, 'kind': 'e4a_golden_sequence_parity_v1', 'status': 'VERIFIED_EXACT_CHECKED_FIELDS',
      'scope': 'three historical raw documents retokenized through new preparer; no model or quality evaluation',
      'prepared_sources_sha256': sha(manifest_path), 'expected_sha256': sha(expected_path), 'source_pins': source_pins,
      'documents': len(results), 'rows': results, 'exact_field_checks': sum(row['exact_field_count'] for row in results),
      'new_tokenizer_invocations_in_this_checker': 0, 'model_loads': 0, 'GPU_actions': 0,
      'not_claimed': ['old complete acquisition replay', 'selection policy equality', 'evidence ID label token-span parity when absent upstream']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepared-sources', required=True); parser.add_argument('--expected', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); output = Path(args.output); require(not output.exists(), 'Refusing existing golden parity output')
    result = run(args.prepared_sources, args.expected); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'documents': result['documents'], 'exact_field_checks': result['exact_field_checks']}, ensure_ascii=True))

if __name__ == '__main__': main()
