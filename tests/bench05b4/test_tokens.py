"""Synthetic offset oracles; no tokenizer package, model, or CUDA is imported."""
import copy
import unittest

from inference_kit.core.io import digest
from inference_kit.probes.contracts import Sample
from inference_kit.probes.retrieval_tokens import (
    POLICY, REFERENCE_SOURCE, content_span_records, prepare_retrieval_row,
    validate_content_spans,
)
from inference_kit.probes.task_tokens import prepare_row
from inference_kit.tasks import context_evidence as task


class TinyTokenizer:
    """Four-codepoint synthetic tokens deliberately straddle textual boundaries."""
    is_fast = True
    all_special_ids = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt and len(messages) == 1
        return 'USER\n' + messages[0]['content'] + '\nASSISTANT\n'

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [1 + sum(map(ord, text[i:i + 4])) % 254 for i in range(0, len(text), 4)]

    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        return {'input_ids': ids, 'offset_mapping': [(i, min(i + 4, len(text)))
          for i in range(0, len(text), 4)], 'special_tokens_mask': [0] * len(ids)}

    def convert_ids_to_tokens(self, ids):
        return ['synthetic-' + str(i) for i in ids]


CFG = {'num_hidden_layers': 24, 'num_attention_heads': 32, 'num_key_value_heads': 32}


class RetrievalTokenTests(unittest.TestCase):
    def setUp(self):
        self.data = task.make_dataset()
        self.tok = TinyTokenizer()

    def prepared(self, sample):
        return prepare_retrieval_row(sample, self.tok, CFG, 'fake-tokenizer', 'fake-space')

    def test_all_28_rows_have_new_task_policy_and_fixed_limits(self):
        for sample in self.data['rows']:
            with self.subTest(sample=sample['sample_id']):
                row = self.prepared(sample)
                self.assertEqual((row['task_id'], row['task_version'], row['reference_source']),
                                 (task.TASK, task.VERSION, REFERENCE_SOURCE))
                self.assertEqual(row['tokenization']['policy'], POLICY)
                self.assertEqual(row['tokenization']['chat_template_applied'], 1)
                self.assertLessEqual(len(row['generation_input_ids']), 512)
                self.assertLessEqual(len(row['reference_separate_ids']), 64)
                self.assertLessEqual(len(row['input_ids']), 576)
                self.assertEqual(row['prompt'].count(task.render_prompt(sample)), 1)
                self.assertNotIn(task.canonical_reference(sample)['text'], row['prompt'])

    def test_target_shift_value_literals_and_loss_mask(self):
        row = self.prepared(self.data['rows'][0])
        for item in row['targets']:
            self.assertEqual(item['logit_position'], item['token_position'] - 1)
            self.assertEqual(item['target_token_id'], row['input_ids'][item['token_position']])
            self.assertEqual(row['loss_mask'][item['token_position']], 1)
        self.assertLessEqual(len(row['targets']), 16)
        self.assertEqual(set(row['field_token_coverage']), {'answer', 'evidence'})
        for name, span in row['value_spans'].items():
            self.assertEqual(row['reference_text'][slice(*span)],
                             self.data['rows'][0]['gold'][name])
            self.assertTrue(row['field_token_coverage'][name]['selected_positions'])
        self.assertTrue(all(v == 0 for v in row['loss_mask'][:row['prompt_token_boundary']]))

    def test_gold_content_has_exact_cp_utf8_and_prompt_token_coordinates(self):
        sample = next(r for r in self.data['rows'] if r['language'] == 'ko' and r['gold']['answerable'])
        row = self.prepared(sample)
        validate_content_spans(sample, row)
        span = row['reference_evidence_span']
        self.assertEqual(span['status'], 'AVAILABLE')
        a, b = span['prompt_codepoint_span']
        byte_a, byte_b = span['prompt_utf8_byte_span']
        self.assertEqual(row['prompt'][a:b], span['content'])
        self.assertEqual(row['prompt'].encode('utf-8')[byte_a:byte_b].decode('utf-8'), span['content'])
        self.assertGreater(byte_b, b)
        self.assertNotIn('[' + span['passage_id'] + ']', span['content'])
        self.assertTrue(all(p < row['prompt_token_boundary'] for p in span['overlapping_token_positions']))
        self.assertNotIn(row['prompt_token_boundary'], span['overlapping_token_positions'])

    def test_no_answer_span_is_not_applicable_and_never_zero_mass(self):
        sample = next(r for r in self.data['rows'] if not r['gold']['answerable'])
        row = self.prepared(sample)
        span = row['reference_evidence_span']
        self.assertEqual(span['status'], 'NOT_APPLICABLE')
        self.assertIsNone(span['overlapping_token_positions'])
        self.assertIsNone(span['prompt_codepoint_span'])
        self.assertEqual(len(row['passage_content_spans']), 6)

    def test_corrupt_content_span_or_metadata_rejected(self):
        source = next(r for r in self.data['rows'] if r['gold']['answerable'])
        row = self.prepared(source)
        for key in ('prompt_codepoint_span', 'overlapping_token_positions', 'content_hash'):
            changed = copy.deepcopy(row)
            changed['reference_evidence_span'][key] = 'corrupt'
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_content_spans(source, changed)
        changed = copy.deepcopy(source)
        changed['passages'][0]['content_span'][0] += 1
        with self.assertRaises(ValueError):
            content_span_records(changed, row)

    def test_document_duplication_and_missing_offsets_rejected(self):
        sample = self.data['rows'][0]
        row = self.prepared(sample)
        changed = copy.deepcopy(row)
        changed['prompt'] += sample['document']
        with self.assertRaises(ValueError):
            content_span_records(sample, changed)
        changed = copy.deepcopy(row)
        changed['offsets_codepoints'].pop()
        with self.assertRaises(ValueError):
            content_span_records(sample, changed)

    def test_reference_max64_and_legacy_default128_are_distinct(self):
        sample = self.data['rows'][0]
        contract = task.build_probe_sample(sample)
        content = task.render_prompt(sample)
        contract['reference_text'] = 'x' * 260
        contract['reference_hash'] = digest(contract['reference_text'])
        contract['value_spans'] = {'answer': [0, 2], 'evidence': [3, 5]}
        with self.assertRaisesRegex(ValueError, 'prompt/reference bounds'):
            prepare_row(sample, contract, content, self.tok, CFG, 'tid', 'space',
                        POLICY, 512, 576, task_id=task.TASK, task_version=task.VERSION,
                        reference_source=REFERENCE_SOURCE, max_reference=64)
        # Explicit new keywords default to precisely the old B1 contract.
        legacy = prepare_row(sample, contract, content, self.tok, CFG, 'tid', 'space')
        explicit = prepare_row(sample, contract, content, self.tok, CFG, 'tid', 'space',
                               task_id='tech_record_extract', task_version='0.1',
                               reference_source='task_author_gold_json', max_reference=128)
        self.assertEqual(legacy, explicit)
        self.assertEqual(legacy['task_id'], 'tech_record_extract')

    def test_registered_source_rejects_wrong_task_and_version(self):
        sample = self.prepared(self.data['rows'][0])
        fields = ('task_id', 'task_version', 'sample_id', 'source_id', 'prompt',
                  'prompt_hash', 'reference_text', 'reference_source', 'tokenization',
                  'correctness_meaning')
        valid = {k: sample[k] for k in fields}
        Sample(**valid).validate()
        for key, value in (('task_id', 'tech_record_extract'), ('task_version', '9'),
                           ('reference_source', 'task_author_gold_json')):
            changed = valid | {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'reference source'):
                Sample(**changed).validate()


if __name__ == '__main__':
    unittest.main()
