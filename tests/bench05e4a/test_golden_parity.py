"""The independent comparator rejects sequence differences without broad ignore flags."""
from copy import deepcopy
import unittest
import importlib.util
from pathlib import Path
_CHECKER = Path(__file__).resolve().parents[2] / 'tools' / 'bench05e4a_golden_parity.py'
_SPEC = importlib.util.spec_from_file_location('reviewed_e4a_golden_checker', _CHECKER)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compare_sample, exact = _MODULE.compare_sample, _MODULE.exact
DIAGNOSTIC_EXACT, NATURAL_EXACT = _MODULE.DIAGNOSTIC_EXACT, _MODULE.NATURAL_EXACT


def paired_sample():
    text = 'prompt'; reference = 'answer'
    diag = {key: 'constant' for key in DIAGNOSTIC_EXACT + NATURAL_EXACT}
    diag.update(task_id='tech_record_extract', task_version='0.1', sample_id='s1', source_id='source1',
                prompt=text, reference_text=reference, input_ids=[1, 2, 3], prompt_token_boundary=2,
                generation_input_ids=[1, 2], attention_mask=[1, 1, 1], loss_mask=[0, 0, 1],
                special_token_ids=[1], special_token_mask=[1, 0, 0], offsets_codepoints=[[0, 0], [0, 6], [6, 12]],
                offsets_utf8_bytes=[[0, 0], [0, 6], [6, 12]], targets=[{'token_position': 2}], full_vocab_target_positions=[2],
                tokenization={'chat_template_applied': 1, 'bos_added_by_encoder': False, 'eos_added_by_encoder': False,
                    'padding_added': False, 'tokenizer_identity': 't', 'token_id_space': 'v'})
    wire = {'prompt': [1, 2], 'max_tokens': 192, 'temperature': 0, 'top_p': 1, 'n': 1,
            'stream': True, 'echo': False, 'ignore_eos': False, 'add_special_tokens': False}
    old = {'diagnostic': diag, 'generation': {'wire_payload': wire}}
    new = deepcopy(old); new['generation']['preparation_proof'] = {'reference_used_for_generation': False}
    return old, new


class GoldenParityTests(unittest.TestCase):
    def test_exact_contract_and_provenance_selection_are_separate(self):
        old, new = paired_sample(); new['diagnostic']['new_source_identity'] = 'new'
        new['diagnostic']['targets'] = [{'token_position': 1}, {'token_position': 2}]
        result = compare_sample(old, new)
        self.assertEqual(result['status'], 'EXACT_SEQUENCE_REQUEST_PARITY')
        self.assertEqual(result['selection']['new_only_target_positions'], [1])
        self.assertTrue(all(result['exact_fields'].values()))

    def test_exact_type_and_named_sequence_mutations_rejected(self):
        self.assertFalse(exact([1], [True])); self.assertFalse(exact([1], [1, 2]))
        for key, value in (('input_ids', [1, 2]), ('prompt', 'different'), ('prompt_token_boundary', True),
                           ('loss_mask', [0, 1, 1]), ('special_token_mask', [0, 0, 0]),
                           ('offsets_utf8_bytes', [[0, 0], [0, 5], [5, 12]])):
            old, new = paired_sample(); new['diagnostic'][key] = value
            with self.assertRaises(ValueError): compare_sample(old, new)
        old, new = paired_sample(); new['generation']['wire_payload']['ignore_eos'] = True
        with self.assertRaises(ValueError): compare_sample(old, new)

    def test_missing_field_never_compares_empty_or_intersection_only(self):
        old, new = paired_sample(); del new['diagnostic']['input_ids']
        with self.assertRaises(ValueError): compare_sample(old, new)
        old, new = paired_sample(); new['generation']['wire_payload']['prompt'] = [1, 2, 3]
        with self.assertRaises(ValueError): compare_sample(old, new)

if __name__ == '__main__': unittest.main()
