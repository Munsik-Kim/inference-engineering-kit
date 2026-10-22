"""Closed CPU tests for the explicit tokenizer worker (no tokenizer SDK needed)."""
from copy import deepcopy
import hashlib
import re
import unittest

from inference_kit.core.io import digest
from inference_kit.probes.input_preparation_worker import prepare_rows, _offsets, _uniform, tokenizer_offset_probe
from inference_kit.tasks import tech_record, context_evidence
from inference_kit.tasks.input_records import arc_context, render_pinned_chat, render_record


class ToyFastTokenizer:
    """Deterministic whitespace/word tokenizer; synthetic contract test only."""
    is_fast = True
    all_special_ids = []
    def get_vocab(self): return {str(i): i for i in range(4096)}
    def _parts(self, text): return list(re.finditer(r'\S+|\s+', text))
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [int(hashlib.sha256(p.group().encode()).hexdigest()[:8], 16) % 4096 for p in self._parts(text)]
    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text), 'offset_mapping': [p.span() for p in self._parts(text)],
                'special_tokens_mask': [0] * len(self._parts(text))}
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return render_pinned_chat(messages[0]['content'])
    def convert_ids_to_tokens(self, ids): return [str(i) for i in ids]


def worker_fixture():
    records = []
    for sample in (tech_record.make_dataset()['rows'][0], context_evidence.make_dataset()['rows'][0]):
        task = sample['task_id']
        records.append({'task_id': task, 'task_version': sample['task_version'], 'dataset_id': 'synthetic-contract',
          'source_id': sample['source_id'], 'sample_id': sample['sample_id'], 'split': sample['split'],
          'usage_scope': 'ENGINEERING_INPUT_ONLY', 'policy_id': 'P00' if task == tech_record.TASK else 'context_evidence_fixed_v1',
          'reference_source': 'task_author_gold_json' if task == tech_record.TASK else 'task_author_gold_answer_evidence_v1',
          'sample': sample})
    document = {'id': 'arc-synthetic-contract', 'question': 'Which controlled choice?',
                'choices': {'text': ['short', 'the longer choice', 'third'], 'label': ['1', '2', '3']}, 'answerKey': '2'}
    records.append({'task_id': 'arc_challenge', 'task_version': '1.0', 'dataset_id': 'synthetic-contract',
      'source_id': 'synthetic-arc-source', 'sample_id': document['id'], 'split': 'test', 'usage_scope': 'REGRESSION_REUSE',
      'policy_id': 'arc_original_completion_v1', 'reference_source': 'dataset_answer_key_choice',
      'sample': {'document': document, 'original_index': 1, 'prompt_context': arc_context(document['question']),
                 'prompt_contract_id': 'arc_original_completion_v1'}})
    raw = {'schema_version': 1, 'kind': 'raw_task_manifest_v1', 'dataset_id': 'synthetic-contract',
           'usage_scope': 'SYNTHETIC_CONTRACT_TEST', 'rows': records}
    spec = {'schema_version': 1, 'kind': 'input_preparation_spec_v1', 'policy_id': 'existing_tasks_local_tokenizer_v1',
      'limits': {'max_prompt_tokens': 512, 'max_reference_tokens': 128, 'max_joint_tokens': 640, 'context_tokens': 2048},
      'selection': {'policy_id': 'field_round_robin_uniform_v1', 'max_targets': 16, 'max_full': 4}, 'seed': 1}
    tokenizer = ToyFastTokenizer()
    metadata = {'model_config': {'num_hidden_layers': 24, 'num_attention_heads': 32, 'num_key_value_heads': 32},
      'tokenizer_identity': 'a' * 64, 'token_id_space': digest(tokenizer.get_vocab()), 'vocabulary_size': len(tokenizer.get_vocab())}
    return raw, spec, metadata, tokenizer


class TokenPreparationTests(unittest.TestCase):
    def test_three_task_roles_and_all_arc_choices(self):
        raw, spec, metadata, tok = worker_fixture(); result = prepare_rows(raw, spec, metadata, tok)
        self.assertEqual(len(result['sources']), 3)
        by = {source['task_id']: source for source in result['sources']}
        for task in ('tech_record_extract', 'context_evidence_retrieval'):
            sample = by[task]['samples'][0]
            self.assertEqual(sample['generation']['wire_payload']['prompt'], sample['diagnostic']['generation_input_ids'])
            self.assertNotEqual(sample['generation']['wire_payload']['prompt'], sample['diagnostic']['input_ids'])
            self.assertFalse(sample['generation']['preparation_proof']['reference_used_for_generation'])
            self.assertFalse(sample['generation']['wire_payload']['ignore_eos'])
            self.assertEqual(sample['diagnostic']['tokenization']['chat_template_applied'], 1)
        arc = by['arc_challenge']['samples'][0]
        self.assertNotIn('generation', arc)
        self.assertEqual([request['choice_index'] for request in arc['likelihood']['requests']], [0, 1, 2])
        self.assertEqual(arc['diagnostic']['tokenization']['chat_template_applied'], 0)
        self.assertEqual(arc['diagnostic']['input_ids'], arc['likelihood']['requests'][1]['input_ids'])

    def test_arc_whitespace_and_character_denominator_preserved(self):
        raw, spec, metadata, tok = worker_fixture(); sample = prepare_rows(raw, spec, metadata, tok)['sources'][2]['samples'][0]
        for request, choice in zip(sample['likelihood']['requests'], raw['rows'][2]['sample']['document']['choices']['text']):
            self.assertEqual(request['continuation'], ' ' + choice)
            self.assertEqual(request['normalization_char_length'], len(choice))
            self.assertEqual(request['moved_trailing_context_whitespace'], '\n')
            self.assertEqual(request['input_ids'], request['context_ids'] + request['continuation_ids'])
        self.assertTrue(sample['diagnostic']['reference_text'].startswith('\n '))

    def test_offsets_unicode_bytes_and_overlap_are_explicit(self):
        text = '한e\u0301🙂\n끝'; offsets = [[0, 1], [1, 3], [3, 4], [4, 5], [5, 6]]
        cp, byte, overlaps = _offsets(text, [1] * 5, offsets, [0] * 5)
        self.assertEqual(cp, offsets)
        self.assertEqual(byte[0], [0, 3]); self.assertEqual(byte[2][1] - byte[2][0], 4)
        self.assertEqual(overlaps, [])
        self.assertEqual(_offsets('한', [1, 2], [[0, 1], [0, 1]], [0, 0])[2], [1])
        with self.assertRaises(ValueError): _offsets(text, [1], [[0, 100]], [0])
        with self.assertRaises(ValueError): _offsets(text, [1], [[0, 0]], [0])

    def test_selection_limits_and_full_off(self):
        raw, spec, metadata, tok = worker_fixture()
        main = prepare_rows(raw, spec, metadata, tok)
        small_spec = deepcopy(spec); small_spec['selection'].update(max_targets=4, max_full=0)
        small = prepare_rows(raw, small_spec, metadata, tok)
        self.assertGreater(sum(r['targets'] for r in main['receipt']['rows']), sum(r['targets'] for r in small['receipt']['rows']))
        for source in small['sources']:
            row = source['samples'][0]['diagnostic']
            self.assertLessEqual(len(row['targets']), 4)
            self.assertEqual(row['full_vocab_target_positions'], [])
            self.assertEqual(row['selection_receipt']['full_status'], 'NOT_REQUESTED')
            self.assertEqual(len({t['token_position'] for t in row['targets']}), len(row['targets']))
            for target in row['targets']:
                t = target['token_position']; self.assertGreater(t, 0)
                self.assertEqual(target['logit_position'], t - 1)
                self.assertEqual(target['target_token_id'], row['input_ids'][t])

    def test_repeat_content_identity_and_no_mutation(self):
        raw, spec, metadata, tok = worker_fixture(); original = deepcopy((raw, spec, metadata))
        first = prepare_rows(raw, spec, metadata, tok); second = prepare_rows(raw, spec, metadata, tok)
        self.assertEqual(first, second); self.assertEqual((raw, spec, metadata), original)
        self.assertEqual(_uniform([2, 4, 6], 8), [2, 4, 6])

    def test_pins_empty_and_length_rejections(self):
        raw, spec, metadata, tok = worker_fixture()
        changed = deepcopy(metadata); changed['token_id_space'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'vocabulary pin'): prepare_rows(raw, spec, changed, tok)
        empty = deepcopy(raw); empty['rows'] = []
        with self.assertRaisesRegex(ValueError, 'EMPTY'): prepare_rows(empty, spec, metadata, tok)
        short = deepcopy(spec); short['limits']['max_prompt_tokens'] = 2
        with self.assertRaisesRegex(ValueError, 'PROMPT_TOO_LONG'): prepare_rows(raw, short, metadata, tok)
        long_ref = deepcopy(spec); long_ref['limits']['max_reference_tokens'] = 1
        with self.assertRaisesRegex(ValueError, 'REFERENCE_TOO_LONG'): prepare_rows(raw, long_ref, metadata, tok)

    def test_reference_spans_and_content_spans_not_confused(self):
        raw, spec, metadata, tok = worker_fixture(); source = prepare_rows(raw, spec, metadata, tok)['sources'][1]
        row = source['samples'][0]['diagnostic']; span = row['reference_evidence_span']
        self.assertEqual(span['status'], 'AVAILABLE')
        a, b = span['prompt_codepoint_span']
        self.assertEqual(row['prompt'][a:b], span['content'])
        self.assertEqual(len(row['passage_content_spans']), 6)
        self.assertTrue(all(t < row['prompt_token_boundary'] for t in span['overlapping_token_positions']))
        self.assertIn('answer', row['value_spans']); self.assertIn('evidence', row['value_spans'])

    def test_hidden_gold_changes_reference_but_not_generation_renderer(self):
        raw, _, _, _ = worker_fixture(); original = raw['rows'][0]; changed = deepcopy(original)
        changed['sample']['gold']['temperature_c'] = 987.5
        left, right = render_record(original), render_record(changed)
        self.assertEqual(left['content'], right['content'])
        self.assertNotEqual(left['reference_text'], right['reference_text'])

    def test_fixed_unicode_offset_probe_and_document_mapping(self):
        raw, spec, metadata, tok = worker_fixture()
        receipt = tokenizer_offset_probe(tok)
        self.assertEqual(receipt['status'], 'VERIFIED_FOR_FIXED_PROBE')
        self.assertFalse(receipt['preparer_unicode_normalization_applied'])
        self.assertEqual(len(receipt['authored_spans']), 6)
        prepared = prepare_rows(raw, spec, metadata, tok)
        for source in prepared['sources'][:2]:
            row = source['samples'][0]['diagnostic']; span = row['source_document_span']
            a, b = span['prompt_codepoint_span']
            self.assertEqual(row['prompt'][a:b], source['samples'][0]['authored_sample']['document'])
        self.assertEqual(prepared['sources'][2]['samples'][0]['diagnostic']['source_document_span']['status'], 'NOT_APPLICABLE')

    def test_copied_token_spec_matches_preparation_selection(self):
        raw, spec, metadata, tok = worker_fixture()
        metadata['token_spec'] = {'max_sequence_tokens': 640, 'max_target_positions_per_sample': 32,
                                  'max_full_vocab_positions_per_sample': 8, 'observe_attention': False}
        spec['selection'].update(max_targets=4, max_full=0)
        result = prepare_rows(raw, spec, metadata, tok)
        self.assertEqual(result['sources'][0]['model_metadata']['token_spec']['max_target_positions_per_sample'], 4)
        self.assertEqual(result['sources'][0]['model_metadata']['token_spec']['max_full_vocab_positions_per_sample'], 0)
        self.assertEqual(metadata['token_spec']['max_target_positions_per_sample'], 32)

    def test_changed_or_double_chat_template_rejected(self):
        raw, spec, metadata, tok = worker_fixture()
        class DoubleTemplateTokenizer(ToyFastTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return render_pinned_chat(render_pinned_chat(messages[0]['content']))
        with self.assertRaisesRegex(ValueError, 'chat template'):
            prepare_rows(raw, spec, metadata, DoubleTemplateTokenizer())

    def test_arc_joint_boundary_failure_not_repaired(self):
        raw, spec, metadata, tok = worker_fixture()
        class BrokenBoundaryTokenizer(ToyFastTokenizer):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                if '\n short' in text and '<|im_start|>' in text:
                    ids[-1] = (ids[-1] + 1) % 4096
                return ids
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_TOKEN_BOUNDARY'):
            prepare_rows(raw, spec, metadata, BrokenBoundaryTokenizer())


if __name__ == '__main__': unittest.main()
