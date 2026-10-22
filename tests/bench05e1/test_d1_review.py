"""E1 counterexamples for vacuous or truncated D1 observer controls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.analysis.structure import _control_selection, _selection, observer_controls, CONTROL_POLICY
from inference_kit.core.io import read
from bench05d1.test_analysis import raw_fixture


def row():
    return {'task_id': 'arbitrary_task', 'sample_id': 'document', 'input_ids': [0, 1, 2],
            'targets': [{'token_position': 1, 'logit_position': 0, 'target_token_id': 1}],
            'full_vocab_target_positions': [1]}


def direct_controls(value):
    plan = {'structure_spec': {'observer_consistency_policy': CONTROL_POLICY}}
    side = {'job': {'status': 'COMPLETE'}, 'execution': {'status': 'MEASURED'},
            'native_preservation_receipt_verified': True, 'tokens': {}}
    return observer_controls(None, None, plan, {'document': value}, {'arm': side})


class D1ReviewTests(unittest.TestCase):
    def test_empty_targets_cannot_be_verified_with_zero_comparisons(self):
        value = row(); value['targets'] = []
        with self.assertRaisesRegex(ValueError, 'Empty observer target'):
            direct_controls(value)

    def test_empty_full_selection_cannot_vacuously_certify_same_pass_check(self):
        value = row(); value['full_vocab_target_positions'] = []
        with self.assertRaisesRegex(ValueError, 'Empty observer full'):
            direct_controls(value)

    def test_duplicate_targets_and_full_positions_rejected(self):
        for field in ('targets', 'full_vocab_target_positions'):
            with self.subTest(field=field):
                value = row(); value[field] *= 2
                with self.assertRaisesRegex(ValueError, '[Dd]uplicate'):
                    _control_selection(value)

    def test_invalid_target_shift_id_and_boundary_rejected(self):
        for key, bad in (('token_position', 0), ('token_position', 3), ('token_position', True),
                         ('logit_position', 1), ('target_token_id', 2)):
            with self.subTest(key=key, value=bad):
                value = row(); value['targets'][0][key] = bad
                with self.assertRaises(ValueError):
                    _control_selection(value)

    def test_full_coordinate_must_be_selected_target(self):
        value = row(); value['full_vocab_target_positions'] = [2]
        with self.assertRaisesRegex(ValueError, 'full-logit'):
            _control_selection(value)

    def test_direct_control_numeric_axes_cannot_truncate(self):
        with patch('inference_kit.analysis.structure._value', side_effect=[[0.], [0., 9.], [0.]]):
            with self.assertRaisesRegex(ValueError, 'numeric axis'):
                direct_controls(row())

    def test_direct_empty_numeric_vectors_cannot_certify(self):
        with patch('inference_kit.analysis.structure._value', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'numeric axis'):
                direct_controls(row())

    def test_direct_nonfinite_control_cannot_hide_after_equal_first_element(self):
        with patch('inference_kit.analysis.structure._value', side_effect=[[0., 0.], [0., float('nan')], [0., 0.]]):
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                direct_controls(row())

    def test_frozen_selection_rejects_masked_special_target_and_bad_mask_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = read(raw_fixture(Path(temp) / 'raw') / 'PLAN_FROZEN.json')
        for key, value in (('loss_mask', [0, 0, 0, 1]), ('special_token_mask', [0, 0, 1, 0]),
                           ('special_token_mask', [0])):
            with self.subTest(key=key, value=value):
                altered = copy.deepcopy(plan); altered['samples']['rows'][0][key] = value
                with self.assertRaises(ValueError):
                    _selection(altered)

    def test_valid_frozen_fixture_selection_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = read(raw_fixture(Path(temp) / 'raw') / 'PLAN_FROZEN.json')
        before = copy.deepcopy(plan)
        rows = _selection(plan)
        self.assertEqual(len(rows), 2)
        self.assertEqual(plan, before)
