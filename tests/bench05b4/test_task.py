import copy
import json
import unittest
from collections import Counter
from inference_kit.core.io import digest
from inference_kit.tasks.context_evidence import (
    TASK, VERSION, SCORER, REFERENCE_SOURCE, ContextEvidenceAdapter, make_dataset,
    verify_gold, parse_rendered, render_prompt, canonical_reference, build_probe_sample,
    select_probes, parse_output, score, aggregate, baseline_generations,
)


class RetrievalTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset()
        cls.rows = cls.data['rows']
        cls.eval = [r for r in cls.rows if r['split'] == 'EVAL_LOCKED']
        cls.sample = next(r for r in cls.eval if r['gold']['answerable'])
        cls.absent = next(r for r in cls.eval if not r['gold']['answerable'])

    def generation(self, sample=None, text=None, **overrides):
        s = sample or self.sample
        value = {'task_id': TASK, 'task_version': VERSION, 'sample_id': s['sample_id'],
                 'sample_identity': s['sample_identity'], 'request_id': 'run|attempt|bf16|' + s['sample_id'],
                 'request_kind': 'generate_until', 'output_text': canonical_reference(s)['text'] if text is None else text,
                 'transport_complete': True, 'termination': 'NATURAL_STOP',
                 'request_latency_http': .2, 'request_latency_task': 12.2}
        value.update(overrides)
        return value

    def test_dataset_counts_and_balance(self):
        self.assertEqual(len(self.rows), 28)
        self.assertEqual(len(self.eval), 24)
        self.assertEqual(Counter(r['language'] for r in self.eval), {'en': 12, 'ko': 12})
        self.assertEqual(Counter(r['case_type'] for r in self.eval), {k: 6 for k in ('direct', 'similar_entity', 'relation_distractor', 'not_found')})
        self.assertEqual(sum(r['gold']['answerable'] for r in self.eval), 18)
        self.assertEqual(Counter(r['gold']['evidence'] for r in self.eval if r['gold']['answerable']), {'P0' + str(i): 3 for i in range(1, 7)})

    def test_deterministic_unique_disjoint_documents(self):
        self.assertEqual(make_dataset(), self.data)
        self.assertEqual(len({r['sample_id'] for r in self.rows}), 28)
        self.assertEqual(len({r['document_hash'] for r in self.rows}), 28)
        smoke = {r['entity'] for r in self.rows if r['split'] == 'SMOKE'}
        self.assertFalse(smoke & {r['entity'] for r in self.eval})
        self.assertEqual(len({f['value'] for r in self.rows for f in r['authored_facts']}), 168)

    def test_independent_gold_validation_and_spans(self):
        for r in self.rows:
            self.assertTrue(verify_gold(r))
            self.assertEqual(parse_rendered(r['document'], r['question']), r['gold'])
            if r['gold']['answerable']:
                a, b = r['gold']['answer_span']
                self.assertEqual(r['document'][a:b], r['gold']['answer'])

    def test_source_parser_does_not_accept_hidden_gold(self):
        r = copy.deepcopy(self.sample)
        r['gold']['answer'] = 'ZZZZ-00'
        with self.assertRaisesRegex(ValueError, 'Gold/source'):
            verify_gold(r)
        self.assertEqual(parse_rendered(r['document'], r['question']), self.sample['gold'])

    def test_conflicting_or_duplicate_fact_rejected(self):
        r = self.sample
        source = r['document'].splitlines()
        target = next(p for p in r['passages'] if p['passage_id'] == r['gold']['evidence'])['content']
        other = 0 if r['gold']['evidence'] != 'P01' else 1
        source[other] = '[P0' + str(other + 1) + '] ' + target
        with self.assertRaisesRegex(ValueError, 'Multiple/conflicting'):
            parse_rendered('\n'.join(source), r['question'])

    def test_missing_passage_or_bad_span_rejected(self):
        r = copy.deepcopy(self.sample); r['passages'][0]['content_span'][0] += 1
        with self.assertRaisesRegex(ValueError, 'span'):
            verify_gold(r)
        with self.assertRaises(ValueError):
            parse_rendered('\n'.join(self.sample['document'].splitlines()[:5]), self.sample['question'])

    def test_baselines_use_same_scorer(self):
        _, rule = aggregate(self.eval, baseline_generations(self.eval))
        _, abstain = aggregate(self.eval, baseline_generations(self.eval, 'always_not_found'))
        self.assertEqual(rule['metrics']['task_success']['correct'], 24)
        self.assertEqual(abstain['metrics']['task_success']['correct'], 6)
        self.assertEqual(rule['metrics']['task_success_within_10s']['correct'], 0)

    def test_all_gold_positive_controls(self):
        for row in self.rows:
            s = score(row, self.generation(row))
            self.assertTrue(s['answer_exact'] and s['evidence_exact'] and s['format_valid'] and s['task_success'])
            self.assertTrue(s['task_success_within_10s'])

    def test_reference_and_probe_identity(self):
        for r in self.rows:
            ref = canonical_reference(r); probe = build_probe_sample(r)
            self.assertEqual(probe['reference_source'], REFERENCE_SOURCE)
            self.assertEqual(ref['reference_hash'], digest(ref['text']))
            self.assertEqual(probe['task_id'], TASK)
            for key, span in ref['value_spans'].items():
                self.assertEqual(ref['text'][span[0]:span[1]], r['gold'][key])

    def test_probe_selection_before_outcomes(self):
        ids = select_probes(self.rows)
        self.assertEqual(ids, select_probes(self.rows))
        chosen = [next(r for r in self.rows if r['sample_id'] == sid) for sid in ids]
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(Counter(r['language'] for r in chosen), {'en': 2, 'ko': 2})
        self.assertEqual({r['case_type'] for r in chosen}, {'direct', 'similar_entity', 'relation_distractor', 'not_found'})
        self.assertEqual(sum(not r['gold']['answerable'] for r in chosen), 1)

    def test_prompt_has_source_but_no_hidden_fields(self):
        for row in self.rows:
            prompt = render_prompt(row)
            self.assertIn(row['document'], prompt); self.assertIn(row['question'], prompt)
            for hidden in ('sample_identity', 'authored_facts', 'gold_answer', 'answer_span', 'source_id'):
                self.assertNotIn(hidden, prompt)
            self.assertNotIn('ANSWER: ' + row['gold']['answer'] + '\nEVIDENCE: ' + row['gold']['evidence'], prompt)

    def test_wrong_code_exact_not_substring(self):
        text = canonical_reference(self.sample)['text'].replace(self.sample['gold']['answer'], self.sample['gold']['answer'] + '0')
        s = score(self.sample, self.generation(text=text))
        self.assertFalse(s['answer_exact']); self.assertFalse(s['answer_format_valid']); self.assertFalse(s['task_success'])

    def test_other_source_value_not_answer(self):
        wrong = next(f['value'] for f in self.sample['authored_facts'] if f['value'] != self.sample['gold']['answer'])
        text = canonical_reference(self.sample)['text'].replace(self.sample['gold']['answer'], wrong)
        s = score(self.sample, self.generation(text=text))
        self.assertTrue(s['answer_format_valid']); self.assertFalse(s['answer_exact']); self.assertTrue(s['evidence_exact'])
        self.assertTrue(s['evidence_correct_answer_wrong'])

    def test_existing_wrong_evidence(self):
        wrong = next(p['passage_id'] for p in self.sample['passages'] if p['passage_id'] != self.sample['gold']['evidence'])
        text = canonical_reference(self.sample)['text'].replace(self.sample['gold']['evidence'], wrong)
        s = score(self.sample, self.generation(text=text))
        self.assertTrue(s['cited_id_exists']); self.assertFalse(s['evidence_exact']); self.assertTrue(s['answer_correct_evidence_wrong'])

    def test_nonexistent_evidence(self):
        text = canonical_reference(self.sample)['text'].replace(self.sample['gold']['evidence'], 'P99')
        s = score(self.sample, self.generation(text=text))
        self.assertTrue(s['evidence_format_valid']); self.assertFalse(s['cited_id_exists']); self.assertFalse(s['evidence_exact'])

    def test_no_answer_is_not_evidence_retrieval(self):
        s = score(self.absent, self.generation(self.absent))
        self.assertTrue(s['task_success']); self.assertIsNone(s['cited_id_exists']); self.assertEqual(s['cited_id_status'], 'NOT_APPLICABLE')

    def test_wrong_abstention_and_false_answer(self):
        s = score(self.sample, self.generation(text='ANSWER: NOT_FOUND\nEVIDENCE: NONE'))
        self.assertTrue(s['false_abstention']); self.assertFalse(s['answer_exact'])
        s = score(self.absent, self.generation(self.absent, 'ANSWER: ABCD-12\nEVIDENCE: P01'))
        self.assertTrue(s['false_answer']); self.assertFalse(s['task_success'])

    def test_sentinel_mixup(self):
        s = score(self.absent, self.generation(self.absent, 'ANSWER: NONE\nEVIDENCE: NOT_FOUND'))
        self.assertFalse(s['answer_format_valid']); self.assertFalse(s['evidence_format_valid'])

    def test_missing_fields_and_duplicate_same_value(self):
        ref = canonical_reference(self.sample)['text']
        s = score(self.sample, self.generation(text=ref.splitlines()[0]))
        self.assertFalse(s['evidence_field_present']); self.assertTrue(s['answer_exact']); self.assertFalse(s['task_success'])
        s = score(self.sample, self.generation(text=ref + '\n' + ref.splitlines()[0]))
        self.assertTrue(s['answer_field_present']); self.assertFalse(s['answer_exact']); self.assertTrue(s['evidence_exact'])
        self.assertTrue(s['parsed']['fields']['answer']['ambiguous'])

    def test_multiple_answers_invalid(self):
        ref = canonical_reference(self.sample)['text']
        s = score(self.sample, self.generation(text=ref.replace(self.sample['gold']['answer'], self.sample['gold']['answer'] + ', ABCD-12')))
        self.assertFalse(s['answer_format_valid'])

    def test_fence_prose_and_order_strict_fields_independent(self):
        ref = canonical_reference(self.sample)['text']
        for text in ('```\n' + ref + '\n```', ref + '\nExplanation.', '\n'.join(reversed(ref.splitlines()))):
            with self.subTest(text=text):
                s = score(self.sample, self.generation(text=text))
                self.assertTrue(s['answer_exact']); self.assertTrue(s['evidence_exact'])
                self.assertFalse(s['format_valid']); self.assertFalse(s['task_success']); self.assertTrue(s['values_correct_format_wrong'])

    def test_crlf_ascii_whitespace(self):
        ref = canonical_reference(self.sample)['text']
        for text in (' \t\r\n' + ref.replace('\n', '\r\n') + '\r\n ', ref.replace(': ', ':\t'), ref.replace('\n', '\r')):
            self.assertTrue(score(self.sample, self.generation(text=text))['task_success'])

    def test_unicode_whitespace_and_case_are_not_normalized(self):
        ref = canonical_reference(self.sample)['text']
        for text in ('\u00a0' + ref, ref.replace('ANSWER', 'answer'), ref.replace(self.sample['gold']['answer'], self.sample['gold']['answer'].lower())):
            self.assertFalse(score(self.sample, self.generation(text=text))['task_success'])

    def test_empty_and_truncated_output(self):
        for text in ('', 'ANSWER: ', 'ANSWER: ' + self.sample['gold']['answer'][:4] + '\nEVIDENCE:'):
            self.assertFalse(score(self.sample, self.generation(text=text))['task_success'])

    def test_length_cap_not_task_success_even_complete_text(self):
        s = score(self.sample, self.generation(termination='LENGTH_CAP'))
        self.assertTrue(s['answer_exact']); self.assertTrue(s['format_valid']); self.assertTrue(s['length_cap'])
        self.assertFalse(s['natural_stop']); self.assertFalse(s['task_success'])

    def test_transport_and_termination_remain_separate(self):
        for termination in ('HTTP_ERROR', 'TIMEOUT', 'TRUNCATED_OR_EOF', 'EMPTY_OUTPUT'):
            s = score(self.sample, self.generation(transport_complete=False, termination=termination))
            self.assertTrue(s['answer_exact']); self.assertFalse(s['task_success']); self.assertEqual(s['termination'], termination)

    def test_http_demo_boundary_not_client_queue(self):
        self.assertTrue(score(self.sample, self.generation(request_latency_http=10., request_latency_task=100.))['task_success_within_10s'])
        for value in (10.0001, -1, None, float('inf'), True):
            self.assertFalse(score(self.sample, self.generation(request_latency_http=value))['task_success_within_10s'])

    def test_not_run_is_unknown_not_wrong(self):
        s = score(self.sample, None)
        self.assertEqual(s['status'], 'NOT_RUN'); self.assertIsNone(s['answer_exact']); self.assertIsNone(s['task_success'])

    def test_denominator_preserves_failed_and_unsubmitted(self):
        gens = [self.generation(self.eval[0]), self.generation(self.eval[1], text='', transport_complete=False, termination='TIMEOUT')]
        scores, summary = aggregate(self.eval, gens)
        self.assertEqual(len(scores), 24); self.assertEqual(summary['expected'], 24); self.assertEqual(summary['submitted'], 2)
        self.assertEqual(summary['not_run'], 22); self.assertFalse(summary['complete'])
        self.assertEqual(summary['metrics']['task_success']['planned_denominator'], 24)
        self.assertEqual(summary['metrics']['task_success']['correct'], 1)

    def test_duplicate_and_unknown_sample_rejected(self):
        generation = self.generation()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(self.eval, [generation, generation])
        generation['sample_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            aggregate(self.eval, [generation])

    def test_task_sample_version_prompt_scorer_mismatch(self):
        for key, value in (('task_id', 'tech_record_extract'), ('task_version', '9'), ('sample_id', 'bad'),
                           ('sample_identity', 'bad'), ('request_kind', 'choice_loglikelihood'),
                           ('prompt_identity', 'bad'), ('scorer_version', 'bad')):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    score(self.sample, self.generation(**{key: value}))

    def test_scoring_does_not_depend_on_arm(self):
        a = score(self.sample, self.generation(arm_id='bf16'))
        b = score(self.sample, self.generation(arm_id='rtn'))
        self.assertEqual(a, b)

    def test_typed_adapter_rich_identity_no_gold_payload(self):
        adapter = ContextEvidenceAdapter(); sample = dict(self.sample, generation_input_ids=[1, 2, 3])
        req = adapter.requests(sample, 'run|attempt1|rtn', 'tok', 'space', 'runtime')[0]
        self.assertEqual(req.request_kind, 'generate_until')
        self.assertEqual(req.payload, {'token_ids': [1, 2, 3]})
        self.assertEqual(req.identity['prompt_identity'], digest(render_prompt(sample)))
        self.assertEqual(req.identity['scorer_version'], SCORER)
        self.assertEqual(req.identity['runtime_identity'], 'runtime')
        self.assertIn('run|attempt1|rtn', req.request_id)
        self.assertNotIn('gold', json.dumps(req.payload))
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            adapter.requests(dict(sample, tokenizer_identity='wrong'), 'exec', 'tok')


if __name__ == '__main__':
    unittest.main()
