"""E2 consumer input guards require no model, network, or installed SDK."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parents[2] / 'tools' / 'bench05e2_consumer_check.py'
spec = importlib.util.spec_from_file_location('e2_consumer_check', P)
consumer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consumer)


class ConsumerToolTests(unittest.TestCase):
    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            consumer.strict_loads('{"문항":1,"문항":2}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError):
            consumer.strict_loads('{"limit":NaN}')

    def test_zero_execution_contract_rejects_bool_and_nonzero(self):
        for bad in (True, 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                consumer.validate_plan({'plan_identity': 'p', 'status': 'VALIDATED_NOT_EXECUTED', 'execution_count': bad, 'acquisition_count': 0})

    def test_empty_cases_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cases.json'
            path.write_text(json.dumps({'schema_version': 1, 'cases': []}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Nonempty'):
                consumer.load_cases(path, [])

    def test_explicit_file_cannot_escape_or_enter_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'data.json'; path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                consumer.explicit_file(path, root / 'other', [])
            with self.assertRaises(ValueError):
                consumer.explicit_file(path, root, [root])

    def test_actual_unsafe_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / 'data.json'; path.write_text('{}', encoding='utf-8')
            link = root / 'linked.json'; link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                consumer.explicit_file(link, root, [])

    def test_valid_unicode_case_and_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root / '한글.json'; path.write_text('{}', encoding='utf-8')
            cases = root / 'cases.json'; cases.write_text(json.dumps({'schema_version': 1, 'cases': [{'id': 'normal', 'suite_path': path.name}]}), encoding='utf-8')
            _, result = consumer.load_cases(cases, [])
            self.assertEqual(result[0]['suite_path'], path)
        value = consumer.validate_plan({'plan_identity': 'p', 'status': 'VALIDATED_NOT_EXECUTED', 'execution_count': 0, 'acquisition_count': 0}, 'VALIDATED_NOT_EXECUTED')
        self.assertEqual(value['acquisition_count'], 0)
