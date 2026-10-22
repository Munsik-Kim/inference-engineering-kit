import copy
import math
import unittest

from inference_kit.core.io import digest
from inference_kit.metrics.token import log_softmax
from inference_kit.metrics.token_diagnostics import aggregate_token_comparisons, compare_token


def token(*, sample='sample-1', source='source-1', acquisition='instrumented_diagnostic',
          target=2, logits=None, tags=None, choice=None, reference='author_gold'):
    ids = [1, target, 3]
    sequence = {'sequence_id': digest({'ids': ids, 'sample': sample}), 'input_ids': ids}
    argmax = min(range(len(logits)), key=lambda i: (-logits[i], i)) if logits is not None else None
    row = {'record_id': sample + '-record', 'source_id': source, 'bundle_identity': 'raw-provenance',
      'plan_identity': 'plan-provenance', 'execution_id': 'execution-provenance',
      'observation_id': 'observation-provenance', 'arm_id': 'bf16', 'task_id': 'task-a',
      'task_version': '1', 'dataset_identity': 'data-a', 'split': 'selected', 'sample_id': sample,
      'panel': None, 'policy_id': 'policy-a', 'choice_index': choice, 'reference_scope': reference,
      'prompt_hash': 'prompt-a', 'reference_hash': 'reference-a', 'sequence_id': sequence['sequence_id'],
      'tokenizer_identity': 'tokenizer-a', 'token_id_space': 'vocabulary-a',
      'vocabulary_size': len(logits) if logits is not None else 4,
      'acquisition_source': acquisition, 'runtime_identity': 'runtime-a',
      'backend_policy': 'native' if acquisition == 'native_likelihood' else 'diagnostic',
      'probability_stage': 'raw_pre_sampling', 'metric_contract': 'target_logprob_v1',
      'token_position': 1, 'logit_position': 0, 'target_token_id': target,
      'prefix_hash': digest({'token_ids': ids[:1], 'token_id_space': 'vocabulary-a'}),
      'target_logprob': log_softmax(logits)[target] if logits is not None else -2.,
      'raw_argmax_token_id': argmax, 'full_logits': None,
      'field_tags': ['answer'] if tags is None else tags, 'status': 'MEASURED', 'requested': True}
    return row, sequence


def comparison(**kwargs):
    left, seq = token(**kwargs)
    right = copy.deepcopy(left)
    right.update(record_id=left['record_id'] + '-rtn', arm_id='rtn', acquisition_source=(
      'decoded_diagnostic' if left['backend_policy'] == 'diagnostic' else 'native_likelihood'))
    return compare_token(left, right, seq, seq)


class TokenDiagnosticMetricTests(unittest.TestCase):
    def test_identity_target_only_is_zero_delta_without_full_fabrication(self):
        result = comparison()
        self.assertEqual(result['status'], 'MEASURED')
        self.assertEqual(result['metrics']['delta_nll_nat'], 0.)
        self.assertEqual(result['metrics']['reference_nll_nat'], 2.)
        self.assertIsNone(result['metrics']['raw_top1_match'])
        self.assertEqual(result['metrics']['full_distribution']['status'], 'UNKNOWN')
        self.assertFalse(result['coverage']['full_logits'])
        self.assertTrue(all(r['overlap_fraction'] is None for r in result['metrics']['top_k'].values()))

    def test_candidate_minus_reference_nll_sign(self):
        left, seq = token(); right = copy.deepcopy(left)
        right['target_logprob'] = -3.25
        result = compare_token(left, right, seq, seq)
        self.assertEqual(result['metrics']['delta_nll_nat'], 1.25)

    def test_identity_full_kl_js_entropy_and_argmax(self):
        logits = [0., 1., 2., -1.]
        left, seq = token(logits=logits)
        result = compare_token(left, left, seq, seq, logits, logits)
        self.assertEqual(result['status'], 'MEASURED')
        full = result['metrics']['full_distribution']
        self.assertEqual(full['kl_ref_candidate_nat'], 0.)
        self.assertEqual(full['js_nat'], 0.)
        self.assertGreater(full['entropy_reference_nat'], 0.)
        self.assertTrue(result['metrics']['raw_top1_match'])
        self.assertEqual(result['metrics']['reference_target_rank'], 1)
        self.assertEqual(result['metrics']['reference_target_margin'], 1.)

    def test_kl_direction_js_symmetry_and_range(self):
        a, b = [0., 1., 4., -2.], [0., 2., 1., -1.]
        left, seq = token(logits=a); right, _ = token(logits=b)
        forward = compare_token(left, right, seq, seq, a, b)['metrics']['full_distribution']
        reverse = compare_token(right, left, seq, seq, b, a)['metrics']['full_distribution']
        self.assertNotAlmostEqual(forward['kl_ref_candidate_nat'], reverse['kl_ref_candidate_nat'])
        self.assertAlmostEqual(forward['js_nat'], reverse['js_nat'])
        self.assertGreaterEqual(forward['js_nat'], 0.)
        self.assertLessEqual(forward['js_nat'], math.log(2.))

    def test_deterministic_ties_rank_topk_and_effective_vocab_cap(self):
        logits = [1., 1., 1., 0.]
        left, seq = token(logits=logits)
        result = compare_token(left, left, seq, seq, logits, logits)
        self.assertEqual(result['metrics']['reference_target_rank'], 3)
        self.assertEqual(result['metrics']['reference_target_margin'], 0.)
        self.assertEqual(result['metrics']['top_k']['1']['reference_ids'], [0])
        self.assertEqual(result['metrics']['top_k']['5']['requested_k'], 5)
        self.assertEqual(result['metrics']['top_k']['5']['effective_k'], 4)
        self.assertEqual(result['metrics']['top_k']['10']['effective_k'], 4)
        self.assertEqual(result['metrics']['top_k']['5']['overlap_fraction'], 1.)

    def test_topk_overlap_uses_declared_denominator(self):
        a, b = [3., 2., 1., 0.], [1., 2., 3., 0.]
        left, seq = token(logits=a); right, _ = token(logits=b)
        result = compare_token(left, right, seq, seq, a, b)
        self.assertEqual(result['metrics']['top_k']['1']['overlap_count'], 0)
        self.assertEqual(result['metrics']['top_k']['1']['overlap_fraction'], 0.)

    def test_missing_record_or_logprob_is_unknown(self):
        left, seq = token()
        for a, b, reason in ((None, left, 'MISSING_BASELINE'), (left, None, 'MISSING_CANDIDATE')):
            result = compare_token(a, b, seq, seq)
            self.assertEqual(result['status'], 'UNKNOWN')
            self.assertIn(reason, result['reasons'])
            self.assertIsNone(result['metrics']['delta_nll_nat'])
        right = copy.deepcopy(left); right['target_logprob'] = None
        result = compare_token(left, right, seq, seq)
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIn('CANDIDATE:MISSING_TARGET_LOGPROB', result['reasons'])

    def test_budget_skipped_and_not_requested_never_measured(self):
        left, seq = token()
        for field, value in (('status', 'BUDGET_SKIPPED'), ('requested', False)):
            right = copy.deepcopy(left); right[field] = value
            result = compare_token(left, right, seq, seq)
            self.assertEqual(result['status'], 'UNKNOWN')
            self.assertFalse(result['coverage']['target_logprob'])

    def test_all_required_matching_axes_and_panel_reject_mismatch(self):
        left, seq = token()
        for field, value in (('source_id', 'different'), ('panel', 'B'), ('task_id', 'other-task'),
          ('task_version', '2'), ('dataset_identity', 'other-data'), ('split', 'test'),
          ('sample_id', 'other'), ('policy_id', 'P11'), ('choice_index', 1),
          ('reference_scope', 'other'), ('prompt_hash', 'changed'), ('reference_hash', 'changed'),
          ('tokenizer_identity', 'other'), ('token_id_space', 'other'), ('runtime_identity', 'other')):
            right = copy.deepcopy(left); right[field] = value
            result = compare_token(left, right, seq, seq)
            with self.subTest(field=field):
                self.assertEqual(result['status'], 'INCOMPARABLE')

    def test_raw_probability_metric_and_acquisition_policy_required(self):
        left, seq = token()
        for field, value in (('probability_stage', 'post_top_p'), ('metric_contract', 'other'),
          ('acquisition_source', 'native_likelihood'), ('backend_policy', 'native')):
            right = copy.deepcopy(left); right[field] = value
            with self.subTest(field=field):
                self.assertEqual(compare_token(left, right, seq, seq)['status'], 'INCOMPARABLE')

    def test_artifact_recipe_source_hash_and_execution_can_differ(self):
        left, seq = token(); right = copy.deepcopy(left)
        right.update(artifact_identity='other-artifact', recipe_identity='other-recipe',
          bundle_identity='other-bundle-hash', plan_identity='other-plan-hash',
          execution_id='other-execution', observation_id='other-observation')
        self.assertEqual(compare_token(left, right, seq, seq)['status'], 'MEASURED')

    def test_native_versus_diagnostic_is_incomparable(self):
        left, seq = token(); right = copy.deepcopy(left)
        right.update(acquisition_source='native_likelihood', backend_policy='native')
        self.assertEqual(compare_token(left, right, seq, seq)['status'], 'INCOMPARABLE')

    def test_shift_target_prefix_and_sequence_validation(self):
        left, seq = token()
        for field, value in (('logit_position', 1), ('token_position', 0),
          ('target_token_id', 3), ('prefix_hash', 'wrong'), ('sequence_id', 'wrong')):
            right = copy.deepcopy(left); right[field] = value
            with self.subTest(field=field):
                self.assertEqual(compare_token(left, right, seq, seq)['status'], 'INCOMPARABLE')
        changed = copy.deepcopy(seq); changed['input_ids'][0] = 0
        self.assertEqual(compare_token(left, left, seq, changed)['status'], 'INCOMPARABLE')

    def test_future_sequence_does_not_change_same_current_prefix_metric(self):
        left, seq = token(); right = copy.deepcopy(left); future = copy.deepcopy(seq)
        future['input_ids'][-1] = 0
        future['sequence_id'] = 'future-sequence'
        right['sequence_id'] = future['sequence_id']
        self.assertEqual(compare_token(left, right, seq, future)['status'], 'MEASURED')

    def test_nonfinite_boolean_positive_logprob_and_input_ids_rejected(self):
        left, seq = token()
        for value in (float('nan'), float('inf'), -float('inf'), True, .1):
            right = copy.deepcopy(left); right['target_logprob'] = value
            with self.subTest(value=value):
                self.assertEqual(compare_token(left, right, seq, seq)['status'], 'INCOMPARABLE')
        bad = copy.deepcopy(seq); bad['input_ids'][0] = True
        self.assertEqual(compare_token(left, left, seq, bad)['status'], 'INCOMPARABLE')

    def test_topk_subset_is_not_full_distribution_and_nan_rejected(self):
        left, seq = token()
        for values in ([1., 2.], [0., 1., 2., float('nan')], [True, 1., 2., 3.]):
            result = compare_token(left, left, seq, seq, values, values)
            self.assertEqual(result['status'], 'INCOMPARABLE')
            self.assertFalse(result['coverage']['full_logits'])

    def test_one_sided_full_has_no_pair_kl_or_topk(self):
        logits = [0., 1., 2., -1.]; left, seq = token(logits=logits)
        result = compare_token(left, left, seq, seq, logits, None)
        self.assertEqual(result['status'], 'MEASURED')
        self.assertEqual(result['metrics']['reference_target_rank'], 1)
        self.assertIsNone(result['metrics']['candidate_target_rank'])
        self.assertEqual(result['metrics']['full_distribution']['status'], 'UNKNOWN')

    def test_consistency_without_acquisition_tolerance_remains_unknown(self):
        logits = [0., 1., 2., -1.]; left, seq = token(logits=logits)
        result = compare_token(left, left, seq, seq, logits, logits)
        self.assertEqual(result['metrics']['full_consistency']['reference']['status'], 'UNKNOWN')
        self.assertEqual(result['metrics']['full_distribution']['status'], 'MEASURED')

    def test_declared_full_target_tolerance_and_exact_argmax_mismatch(self):
        logits = [0., 1., 2., -1.]; left, seq = token(logits=logits)
        left['logit_logprob_tolerance'] = 1e-5
        result = compare_token(left, left, seq, seq, logits, logits)
        self.assertTrue(result['metrics']['full_consistency']['reference']['within_declared_tolerance'])
        right = copy.deepcopy(left); right['target_logprob'] -= .1
        self.assertEqual(compare_token(left, right, seq, seq, logits, logits)['status'], 'INCOMPARABLE')
        right = copy.deepcopy(left); right['raw_argmax_token_id'] = 0
        self.assertEqual(compare_token(left, right, seq, seq, logits, logits)['status'], 'INCOMPARABLE')

    def test_invalid_tolerance_and_argmax_ids_rejected(self):
        logits = [0., 1., 2., -1.]; left, seq = token(logits=logits)
        for key, value in (('logit_logprob_tolerance', -1.), ('logit_logprob_tolerance', True),
          ('raw_argmax_token_id', 5), ('raw_argmax_token_id', True)):
            right = copy.deepcopy(left); right[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(compare_token(left, right, seq, seq, logits, logits)['status'], 'INCOMPARABLE')

    def test_empty_aggregation_is_explicit_without_task_average(self):
        result = aggregate_token_comparisons([])
        self.assertEqual(result['groups'], [])
        self.assertTrue(result['no_global_task_average'])

    def test_token_weighted_and_document_macro_denominators_differ(self):
        rows = []
        for sample, values in (('A', [1.]), ('B', [3., 3., 3.])):
            for index, value in enumerate(values):
                left, seq = token(sample=sample); right = copy.deepcopy(left)
                right['target_logprob'] = -value
                result = compare_token(left, right, seq, seq)
                result['left_record_id'] += str(index)
                rows.append(result)
        group = next(g for g in aggregate_token_comparisons(rows)['groups'] if g['field_tag'] == '__all__')
        self.assertEqual(group['planned_comparison_tokens'], 4)
        self.assertEqual(group['measured_documents'], 2)
        self.assertEqual(group['nll']['candidate_nll_nat']['token_weighted'], 2.5)
        self.assertEqual(group['nll']['candidate_nll_nat']['document_macro'], 2.)

    def test_aggregation_keeps_tasks_policies_acquisition_and_reference_separate(self):
        rows = [comparison()]
        for key, value in (('task_id', 'task-b'), ('policy_id', 'P11'),
                           ('reference_scope', 'other-reference'), ('source_id', 'other-source')):
            row = copy.deepcopy(rows[0]); row['identity'][key] = value; rows.append(row)
        groups = [g for g in aggregate_token_comparisons(rows)['groups'] if g['field_tag'] == '__all__']
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(g['measured_comparison_tokens'] == 1 for g in groups))

    def test_unknown_and_incomparable_remain_in_denominator_not_zero(self):
        good = comparison(); unknown = copy.deepcopy(good); bad = copy.deepcopy(good)
        unknown['status'] = 'UNKNOWN'; unknown['metrics']['candidate_nll_nat'] = None
        bad['status'] = 'INCOMPARABLE'
        group = next(g for g in aggregate_token_comparisons([good, unknown, bad])['groups'] if g['field_tag'] == '__all__')
        self.assertEqual(group['planned_comparison_tokens'], 3)
        self.assertEqual(group['measured_comparison_tokens'], 1)
        self.assertEqual(group['status_counts'], {'MEASURED': 1, 'UNKNOWN': 1, 'INCOMPARABLE': 1})
        self.assertEqual(group['nll']['candidate_nll_nat']['token_weighted'], 2.)

    def test_fields_overlap_without_double_counting_all_group(self):
        result = comparison(tags=['answer', 'evidence'])
        groups = aggregate_token_comparisons([result])['groups']
        self.assertEqual({g['field_tag'] for g in groups}, {'__all__', 'answer', 'evidence'})
        self.assertTrue(all(g['planned_comparison_tokens'] == 1 for g in groups))

    def test_native_gold_non_gold_and_all_choices_macro_by_document(self):
        rows = []
        for choice, reference in ((0, 'GOLD'), (1, 'NON_GOLD'), (2, 'NON_GOLD')):
            rows.append(comparison(acquisition='native_likelihood', reference=reference, choice=choice))
        groups = [g for g in aggregate_token_comparisons(rows)['groups'] if g['field_tag'] == '__all__']
        by_scope = {g['identity']['reference_scope']: g for g in groups}
        self.assertEqual(set(by_scope), {'GOLD_ONLY', 'NON_GOLD', 'ALL_CHOICES'})
        self.assertEqual(by_scope['GOLD_ONLY']['measured_comparison_tokens'], 1)
        self.assertEqual(by_scope['NON_GOLD']['measured_comparison_tokens'], 2)
        self.assertEqual(by_scope['ALL_CHOICES']['measured_comparison_tokens'], 3)
        self.assertEqual(by_scope['ALL_CHOICES']['measured_documents'], 1)

    def test_diagnostic_gold_scope_is_not_native_all_choices(self):
        row = comparison(reference='GOLD')
        groups = [g for g in aggregate_token_comparisons([row])['groups'] if g['field_tag'] == '__all__']
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['identity']['reference_scope'], 'GOLD')

    def test_live_and_synthetic_evidence_cannot_mix_and_groups_stay_separate(self):
        left, seq = token(); right = copy.deepcopy(left)
        left['evidence_kind'] = right['evidence_kind'] = 'live_model_diagnostic'
        live = compare_token(left, right, seq, seq)
        self.assertEqual(live['status'], 'MEASURED')
        right['evidence_kind'] = 'synthetic_contract'
        self.assertEqual(compare_token(left, right, seq, seq)['status'], 'INCOMPARABLE')
        left['evidence_kind'] = 'synthetic_contract'
        synthetic = compare_token(left, right, seq, seq)
        groups = [g for g in aggregate_token_comparisons([live, synthetic])['groups'] if g['field_tag'] == '__all__']
        self.assertEqual(len(groups), 2)
        self.assertEqual({g['identity']['evidence_kind'] for g in groups},
                         {'live_model_diagnostic', 'synthetic_contract'})

    def test_field_tag_mismatch_does_not_silently_reuse_left_mapping(self):
        left, seq = token(tags=['answer']); right = copy.deepcopy(left)
        right['field_tags'] = ['evidence']
        result = compare_token(left, right, seq, seq)
        self.assertEqual(result['status'], 'INCOMPARABLE')
        self.assertIn('MISMATCH_FIELD_TAGS', result['reasons'])


if __name__ == '__main__':
    unittest.main()
