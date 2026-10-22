"""CPU-only input boundary checks for the explicit installed-wheel consumer."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    'e1_consumer_tool', Path(__file__).resolve().parents[2] / 'tools' / 'bench05e1_consumer_check.py')
consumer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consumer)


class ConsumerToolTests(unittest.TestCase):
    def fixture(self, root, source_path='recorded'):
        target = root / 'INPUT_SOURCES.json'
        target.write_text(json.dumps({'schema_version': 1, 'sources': [
            {'source_id': '한국어', 'adapter': 'structure_d1', 'path': source_path}]
        }, ensure_ascii=False), encoding='utf-8')
        return target

    def test_explicit_copied_manifest_preserves_utf8_source_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'recorded').mkdir()
            found = consumer.validate_inputs(self.fixture(root), [root / 'denied'])
            self.assertEqual(found[0]['source_id'], '한국어')
            self.assertEqual(found[0]['relative_locator'], 'recorded')

    def test_external_source_is_rejected_without_copying_or_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'external').mkdir(); (root / 'consumer').mkdir()
            manifest = self.fixture(root / 'consumer', '../external')
            with self.assertRaisesRegex(ValueError, 'explicitly copied manifest root'):
                consumer.validate_inputs(manifest, [])

    def test_denied_manifest_and_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'recorded').mkdir()
            manifest = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'denied checkout'):
                consumer.validate_inputs(manifest, [root])
            with self.assertRaisesRegex(ValueError, 'denied checkout'):
                consumer.validate_inputs(manifest, [root / 'recorded'])

    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'real').mkdir(); (root / 'recorded').symlink_to(root / 'real')
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                consumer.validate_inputs(self.fixture(root), [])

    def test_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Missing'):
                consumer.validate_inputs(self.fixture(Path(temp)), [])

    def test_json_duplicates_and_nonfinite_values_rejected(self):
        for value in ('{"sources":[],"sources":[]}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                consumer.strict_loads(value)

    def test_subprocess_environment_excludes_pythonpath_pythonhome(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/original/src', 'PYTHONHOME': '/original'}):
            cleaned = consumer.clean_environment()
        self.assertNotIn('PYTHONPATH', cleaned)
        self.assertNotIn('PYTHONHOME', cleaned)
        self.assertEqual(cleaned['PYTHONDONTWRITEBYTECODE'], '1')

    def test_consumer_starts_with_no_model_or_http_sdk_import(self):
        self.assertIn('transformers', consumer.SDK)
        self.assertIn('torch', consumer.SDK)
        self.assertIn('vllm', consumer.SDK)
        self.assertIn('aiohttp', consumer.SDK)
        self.assertNotIn('sqlite3', consumer.SDK)
