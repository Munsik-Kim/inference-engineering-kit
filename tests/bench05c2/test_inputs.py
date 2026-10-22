import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import digest, read, sha, write
from inference_kit.probes.token_inputs import (
    ADAPTERS, ProbeSpec, ROLES, SELECTION, prepare_inputs, select_inputs,
    validate_probe_inputs, validate_spec, _uniform,
)
from inference_kit.probes.task_tokens import field_targets
from inference_kit.tasks import tech_record, context_evidence


def spec_fixture(targets=32, full=8):
    return {'schema_version': 1, 'selection_policy': SELECTION, 'selection_version': '1',
      'reference_role_filter': list(ROLES), 'max_sequence_tokens': 640,
      'max_target_positions_per_sample': targets, 'max_full_vocab_positions_per_sample': full,
      'observe_attention': False,
      'normalization': {'dtype': 'float32', 'probability_stage': 'raw_pre_sampling', 'log_base': 'e'},
      'consistency_policy': {'id': 'same_pass_saved_f32_v1', 'absolute_tolerance_nat': 2e-5,
        'scope': 'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'},
      'budget': {'task_gpu_wall_seconds': 600, 'cleanup_reserve_seconds': 120, 'max_gpu_processes': 3,
        'automatic_retries': 0, 'probe_gpu_buffer_bytes': 256 * 2 ** 20, 'raw_bytes_total': 256 * 2 ** 20,
        'minimum_cpu_available_bytes': 10 * 2 ** 30, 'minimum_disk_free_bytes': 4 * 2 ** 30}}


CFG = {'model_type': 'llama', 'num_hidden_layers': 2, 'num_attention_heads': 2,
       'num_key_value_heads': 2, 'hidden_size': 32, 'vocab_size': 512}


def encoded(prompt, reference):
    text = prompt + reference
    offsets = [[i, min(i + 2, len(text))] for i in range(0, len(text), 2)]
    ids = [1 + sum(map(ord, text[a:b])) % 510 for a, b in offsets]
    boundary = next(i for i, (_, end) in enumerate(offsets) if end > len(prompt))
    return text, ids, offsets, boundary


def prepared_row(sample, task, version, reference, source, spans=None, prompt=None):
    prompt = prompt or ('Frozen 한글 input ' + sample['sample_id'] + '\nAssistant\n')
    text, ids, offsets, boundary = encoded(prompt, reference)
    if spans:
        targets, coverage = field_targets(ids, boundary, offsets, len(prompt), spans)
    else:
        targets = [{'token_position': i, 'logit_position': i - 1, 'target_token_id': ids[i]}
                   for i in range(boundary, len(ids))][:16]
        coverage = {}
    return {'task_id': task, 'task_version': version, 'sample_id': sample['sample_id'],
      'source_id': sample.get('source_id', sample['sample_id']), 'prompt': prompt, 'reference_text': reference,
      'reference_source': source, 'prompt_hash': digest(prompt), 'reference_hash': digest(reference),
      'full_text': text, 'full_text_hash': digest(text), 'input_ids': ids, 'sequence_identity': digest(ids),
      'prompt_token_boundary': boundary, 'attention_mask': [1] * len(ids),
      'loss_mask': [int(i >= boundary) for i in range(len(ids))],
      'special_token_mask': [0] * len(ids), 'special_token_ids': [],
      'offsets_codepoints': offsets,
      'offsets_utf8_bytes': [[len(text[:a].encode('utf-8')), len(text[:b].encode('utf-8'))] for a, b in offsets],
      'tokenization': {'tokenizer_identity': 'synthetic-tokenizer', 'token_id_space': 'synthetic-space'},
      'targets': targets, 'full_vocab_target_positions': [t['token_position'] for t in targets[:4]],
      'value_spans': spans or {}, 'field_token_coverage': coverage, 'selection_policy': 'original-v1'}


def arc_sample(index):
    sample_id = 'arc-' + str(index)
    choices = ['x', 'answer-' + str(index), 'cc', 'dddd']
    context = 'Question: Which value?\nAnswer:\n'
    requests = []
    for i, choice in enumerate(choices):
        prompt = context.rstrip(); reference = context[len(prompt):] + ' ' + choice
        _, ids, _, boundary = encoded(prompt, reference)
        requests.append({'choice_index': i, 'context': context, 'continuation': ' ' + choice,
          'context_ids': ids[:boundary], 'continuation_ids': ids[boundary:], 'input_ids': ids})
    original = {'sample_id': sample_id, 'split': 'test', 'question': 'Which value?', 'choices': choices,
      'choice_labels': ['A', 'B', 'C', 'D'], 'gold': 1, 'prompt': context, 'requests': requests,
      'document': {'id': sample_id, 'question': 'Which value?', 'answerKey': 'B',
                   'choices': {'text': choices, 'label': ['A', 'B', 'C', 'D']}}}
    ref = context[len(context.rstrip()):] + requests[1]['continuation']
    row = prepared_row(original, 'arc_challenge', '1.0', ref, 'dataset_answer_key_choice', prompt=context.rstrip())
    row.update(gold_choice_index=1, original_context=context, original_continuation=requests[1]['continuation'])
    return original, row


def source_fixture(root):
    declarations = []
    for adapter in ADAPTERS:
        directory = root / adapter
        directory.mkdir()
        if adapter == 'bench05b2':
            authored = tech_record.make_dataset()['rows'][:4]
            rows = []
            for sample in authored:
                ref = tech_record.canonical_reference(sample)
                row = prepared_row(sample, tech_record.TASK, tech_record.VERSION, ref['text'],
                                   'task_author_gold_json', ref['value_spans'])
                row['policy_id'] = 'P00'; row['policy_identity'] = 'frozen-P00'
                rows.append(row)
            samples = {'by_policy': {'P00': rows}}
            task, version, scorer = tech_record.TASK, tech_record.VERSION, tech_record.SCORER
        elif adapter == 'bench05b3':
            pairs = [arc_sample(i) for i in range(4)]
            authored, rows = [s for s, _ in pairs], [r for _, r in pairs]
            samples = {'rows': rows}
            task, version, scorer = 'arc_challenge', '1.0', 'arc_choice_sum_character_norm_v1'
        else:
            authored = context_evidence.make_dataset()['rows'][:4]
            rows = []
            for sample in authored:
                ref = context_evidence.canonical_reference(sample)
                rows.append(prepared_row(sample, context_evidence.TASK, context_evidence.VERSION,
                    ref['text'], 'task_author_gold_answer_evidence_v1', ref['value_spans']))
            samples = {'rows': rows}
            task, version, scorer = context_evidence.TASK, context_evidence.VERSION, context_evidence.SCORER
        samples.update(model_config=copy.deepcopy(CFG), tokenizer_files={'tokenizer.json': 'frozen-metadata'},
          tokenizer_identity='synthetic-tokenizer', token_id_space='synthetic-space', vocabulary_size=512)
        plan = {'task': ADAPTERS[adapter][0], 'task_id': task, 'task_version': version,
          'samples': samples, 'data': {'rows': authored}, 'source_identity': {'fixture.py': 'synthetic'},
          'scorer': scorer, 'pair': {'model_id': 'synthetic/model', 'revision': 'pinned-synthetic'}}
        plan['plan_identity'] = digest(plan)
        write(directory / 'PLAN_FROZEN.json', plan)
        manifest = {'scope': ADAPTERS[adapter][1], 'files': {'PLAN_FROZEN.json': {
          'bytes': (directory / 'PLAN_FROZEN.json').stat().st_size, 'sha256': sha(directory / 'PLAN_FROZEN.json')}}}
        write(directory / 'RAW_MANIFEST.json', manifest)
        declarations.append({'source_id': adapter, 'adapter': adapter, 'path': str(directory),
          'raw_manifest_sha256': sha(directory / 'RAW_MANIFEST.json'), **({'policies': ['P00']} if adapter == 'bench05b2' else {})})
    return {'schema_version': 1, 'kind': 'token_input_sources', 'sources': declarations}


def refresh(inputs):
    for row in inputs['rows']:
        row['probe_input_identity'] = digest({k: v for k, v in row.items() if k != 'probe_input_identity'})
    inputs['rows_identity'] = digest(inputs['rows'])
    key = 'input_identity' if inputs['kind'] == 'common_probe_inputs_v1' else 'selection_identity'
    inputs[key] = digest({k: v for k, v in inputs.items() if k != key})
    return inputs


class ConfiguredInputTests(unittest.TestCase):
    def prepared(self, root):
        return prepare_inputs(source_fixture(root))

    def test_three_tasks_six_preserved_inputs_hash_selected_without_runtime(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); manifest = source_fixture(root)
            before = {str(p): sha(p) for p in root.rglob('*.json')}
            with patch('subprocess.run', side_effect=AssertionError('No CPU preparation process permitted')):
                result = prepare_inputs(manifest)
            self.assertEqual(len(result['rows']), 6)
            self.assertEqual({r['reference_role'] for r in result['rows']}, set(ROLES))
            for selection in result['selection_receipt']['sample_selection']:
                expected = sorted(selection['original_sample_ids'], key=lambda sid:
                    hashlib.sha256((selection['task_id'] + '|' + sid).encode('utf-8')).hexdigest())[:2]
                self.assertEqual(selection['selected_sample_ids'], expected)
            self.assertEqual(before, {str(p): sha(p) for p in root.rglob('*.json')})
            self.assertEqual((result['tokenizer_loads'], result['model_loads'], result['gpu_actions']), (0, 0, 0))

    def test_spec_main_small_configure_different_positions_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            main = select_inputs(inputs, spec_fixture())
            small = select_inputs(inputs, spec_fixture(8, 2))
            self.assertGreater(main['counts']['target_positions_per_arm'], small['counts']['target_positions_per_arm'])
            self.assertGreater(main['counts']['full_vocab_positions_per_arm'], small['counts']['full_vocab_positions_per_arm'])
            self.assertGreater(main['resource_estimates']['raw_numeric_bytes_both_arms'], small['resource_estimates']['raw_numeric_bytes_both_arms'])
            self.assertNotEqual(main['spec_identity'], small['spec_identity'])
            self.assertEqual(main['source_input_identity'], small['source_input_identity'])
            self.assertTrue(all(len(r['targets']) <= 8 and len(r['full_vocab_target_positions']) <= 2 for r in small['rows']))

    def test_small_preserves_original_first_positions_and_main_fill_is_deterministic(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            small = select_inputs(inputs, spec_fixture(8, 2)); main = select_inputs(inputs, spec_fixture())
            self.assertEqual(main, select_inputs(inputs, spec_fixture()))
            for original, selected in zip(inputs['rows'], small['rows']):
                expected = sorted(t['token_position'] for t in original['targets'][:8])
                self.assertEqual([t['token_position'] for t in selected['targets']], expected)
                self.assertEqual(selected['original_targets'], original['targets'])

    def test_arc_gold_choice_preserves_harness_whitespace_without_new_chat(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            arc = [r for r in inputs['rows'] if r['task_id'] == 'arc_challenge']
            self.assertEqual(len(arc), 2)
            self.assertTrue(all(r['reference_text'].startswith('\n ') and not r['prompt'].endswith('\n') for r in arc))
            self.assertTrue(all(r['gold_choice_index'] == 1 for r in arc))

    def test_utf8_offsets_reference_field_spans_and_loss_mask(self):
        with tempfile.TemporaryDirectory() as name:
            result = select_inputs(self.prepared(Path(name)), spec_fixture())
            validate_probe_inputs(result)
            self.assertTrue(any(byte != cp for row in result['rows'] for byte, cp in zip(row['offsets_utf8_bytes'], row['offsets_codepoints'])))
            for row in result['rows']:
                for target in row['targets']:
                    t = target['token_position']
                    self.assertGreater(t, 0)
                    self.assertEqual(row['loss_mask'][t], 1)
                    self.assertEqual(target['logit_position'], t - 1)
                    self.assertEqual(target['prefix_hash'], digest({'token_ids': row['input_ids'][:t], 'token_id_space': result['token_id_space']}))

    def test_uniform_rule_is_integer_floor_without_duplicates(self):
        self.assertEqual(_uniform(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(_uniform(list(range(10)), 1), [0])
        self.assertEqual(_uniform([2, 3], 8), [2, 3])
        self.assertEqual(_uniform([2, 3], 0), [])

    def test_bad_spec_caps_attention_normalization_and_policy_rejected(self):
        changes = [('max_target_positions_per_sample', 0), ('max_target_positions_per_sample', 33),
          ('max_target_positions_per_sample', True), ('max_full_vocab_positions_per_sample', 9),
          ('max_sequence_tokens', 641), ('observe_attention', True), ('selection_policy', 'unknown'),
          ('normalization', {'dtype': 'float16', 'probability_stage': 'raw_pre_sampling', 'log_base': 'e'}),
          ('reference_role_filter', ['unknown-role'])]
        for key, value in changes:
            record = spec_fixture(); record[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_spec(record)
        record = spec_fixture(2, 8)
        with self.assertRaises(ValueError): validate_spec(record)

    def test_spec_dataclass_and_unknown_fields(self):
        self.assertEqual(ProbeSpec.from_record(spec_fixture()).validate(), validate_spec(spec_fixture()))
        self.assertEqual(ProbeSpec.from_dict(spec_fixture()).validate()['selection_version'], '1')
        with self.assertRaises(ValueError): validate_spec(spec_fixture() | {'unknown': True})

    def test_bad_budget_retry_cleanup_and_boolean_rejected(self):
        for key, value in (('automatic_retries', 1), ('cleanup_reserve_seconds', 119),
                           ('max_gpu_processes', 4), ('task_gpu_wall_seconds', 601),
                           ('raw_bytes_total', True), ('probe_gpu_buffer_bytes', 0)):
            spec = spec_fixture(); spec['budget'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_spec(spec)

    def test_raw_or_gpu_buffer_budget_cannot_be_exceeded(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            for key in ('raw_bytes_total', 'probe_gpu_buffer_bytes'):
                spec = spec_fixture(); spec['budget'][key] = 1
                with self.subTest(key=key), self.assertRaises(ValueError): select_inputs(inputs, spec)

    def test_over_length_is_blocked_not_truncated(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name)); spec = spec_fixture(); spec['max_sequence_tokens'] = 1
            with self.assertRaises(ValueError): select_inputs(inputs, spec)

    def test_changed_frozen_inputs_or_selected_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            changed = copy.deepcopy(inputs); changed['rows'][0]['input_ids'][0] += 1
            with self.assertRaises(ValueError): validate_probe_inputs(changed)
            selected = select_inputs(inputs, spec_fixture()); selected['rows'][0]['targets'][0]['logit_position'] += 1
            with self.assertRaises(ValueError): validate_probe_inputs(selected)

    def test_rehashed_shift_prefix_and_utf8_corruption_still_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            for key in ('shift', 'utf8', 'prefix', 'field'):
                changed = copy.deepcopy(select_inputs(inputs, spec_fixture())); row = changed['rows'][0]
                if key == 'shift': row['targets'][0]['logit_position'] += 1
                elif key == 'utf8': row['offsets_utf8_bytes'][0][1] += 1
                elif key == 'prefix': row['targets'][0]['prefix_hash'] = 'wrong'
                else: row['field_token_coverage']['status']['overlapping_positions'] = [0]
                refresh(changed)
                with self.subTest(key=key), self.assertRaises(ValueError): validate_probe_inputs(changed)

    def test_duplicate_targets_full_positions_and_original_ancestry_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            for key in ('targets', 'full_vocab_target_positions', 'original_targets'):
                changed = copy.deepcopy(inputs); changed['rows'][0][key].append(copy.deepcopy(changed['rows'][0][key][0]))
                refresh(changed)
                with self.subTest(key=key), self.assertRaises(ValueError): validate_probe_inputs(changed)

    def test_source_model_tokenizer_mismatch_and_wrong_policy_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); manifest = source_fixture(root)
            wrong = copy.deepcopy(manifest); wrong['sources'][0]['policies'] = ['P11']
            with self.assertRaises(ValueError): prepare_inputs(wrong)
            source = manifest['sources'][2]; path = Path(source['path'])
            plan = read(path / 'PLAN_FROZEN.json'); plan['samples']['token_id_space'] = 'other'
            plan.pop('plan_identity'); plan['plan_identity'] = digest(plan)
            (path / 'PLAN_FROZEN.json').write_text(json.dumps(plan), encoding='utf-8')
            raw = read(path / 'RAW_MANIFEST.json'); raw['files']['PLAN_FROZEN.json'] = {'bytes': (path / 'PLAN_FROZEN.json').stat().st_size, 'sha256': sha(path / 'PLAN_FROZEN.json')}
            (path / 'RAW_MANIFEST.json').write_text(json.dumps(raw), encoding='utf-8'); source['raw_manifest_sha256'] = sha(path / 'RAW_MANIFEST.json')
            with self.assertRaises(ValueError): prepare_inputs(manifest)

    def test_manifest_pin_source_plan_and_external_reference_mismatch_rejected(self):
        for key in ('pin', 'plan', 'reference'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as name:
                manifest = source_fixture(Path(name)); source = manifest['sources'][0]; path = Path(source['path'])
                if key == 'pin': source['raw_manifest_sha256'] = '0' * 64
                else:
                    plan = read(path / 'PLAN_FROZEN.json')
                    if key == 'plan': plan['task_version'] = 'changed-without-digest'
                    else:
                        plan['samples']['by_policy']['P00'][0]['reference_text'] = 'not external gold'
                        plan.pop('plan_identity'); plan['plan_identity'] = digest(plan)
                    (path / 'PLAN_FROZEN.json').write_text(json.dumps(plan), encoding='utf-8')
                    raw = read(path / 'RAW_MANIFEST.json'); raw['files']['PLAN_FROZEN.json'] = {'bytes': (path / 'PLAN_FROZEN.json').stat().st_size, 'sha256': sha(path / 'PLAN_FROZEN.json')}
                    (path / 'RAW_MANIFEST.json').write_text(json.dumps(raw), encoding='utf-8'); source['raw_manifest_sha256'] = sha(path / 'RAW_MANIFEST.json')
                with self.assertRaises(ValueError): prepare_inputs(manifest)

    def test_optional_source_revalidation_and_portable_saved_input_validation(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name)); validate_probe_inputs(inputs, verify_sources=True)
            source = inputs['sources'][0]
            (Path(source['path']) / 'PLAN_FROZEN.json').write_text('{}', encoding='utf-8')
            validate_probe_inputs(inputs)
            with self.assertRaises(ValueError): validate_probe_inputs(inputs, verify_sources=True)

    def test_role_subset_actually_filters_selected_rows_and_counts(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name)); spec = spec_fixture()
            spec['reference_role_filter'] = ['dataset_gold_choice']
            selected = select_inputs(inputs, spec)
            self.assertEqual(selected['counts']['samples'], 2)
            self.assertEqual(selected['counts']['tasks'], 1)
            self.assertTrue(all(r['task_id'] == 'arc_challenge' for r in selected['rows']))
            validate_probe_inputs(selected)

    def test_missing_offsets_stay_unavailable_without_retokenization(self):
        with tempfile.TemporaryDirectory() as name:
            inputs = self.prepared(Path(name))
            for row in inputs['rows']:
                row.update(offsets_codepoints=None, offsets_utf8_bytes=None,
                           offset_mapping_status='MAPPING_UNAVAILABLE',
                           field_token_coverage={}, original_field_token_coverage={})
            refresh(inputs)
            validate_probe_inputs(inputs)
            selected = select_inputs(inputs, spec_fixture())
            self.assertTrue(all(r['offsets_codepoints'] is None and r['offsets_utf8_bytes'] is None and
                                r['offset_mapping_status'] == 'MAPPING_UNAVAILABLE' for r in selected['rows']))
            self.assertGreater(selected['counts']['target_positions_per_arm'], 0)
            selected['rows'][0]['offsets_codepoints'] = []
            refresh(selected)
            with self.assertRaises(ValueError): validate_probe_inputs(selected)


if __name__ == '__main__':
    unittest.main()
