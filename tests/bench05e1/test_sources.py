"""Thin source adapters delegate metrics and preserve acquisition boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from inference_kit.analysis.index_sources import load_source
from inference_kit.core.io import read, write
from bench05d1.test_analysis import raw_fixture, lines, replace_lines, reseal


class IndexSourceTests(unittest.TestCase):
    def spec(self, path, **kw):
        return dict(source_id='synthetic-source', adapter='structure_d1', path=str(path), **kw)

    def test_directory_zip_alias_does_not_change_source_acquisition_or_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw'); archive = root / 'raw.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for p in sorted(raw.rglob('*')):
                    if p.is_file(): z.write(p, 'relocated/' + p.relative_to(raw).as_posix())
            a = load_source(self.spec(raw), root)
            b = load_source(dict(self.spec(archive), source_id='another-alias'), root)
            self.assertEqual(a['source_id'], b['source_id'])
            for key in ('acquisitions', 'analyses', 'samples', 'executions', 'measurements', 'evidence'):
                self.assertEqual(a[key], b[key], key)
            self.assertNotEqual(a['source_alias'], b['source_alias'])
            self.assertEqual(b['locator']['relative_root'], 'relocated')

    def test_structure_controls_projection_and_on_token_are_separate_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = load_source(self.spec(raw_fixture(root / 'raw')), root)
            scopes = {r['scope'] for r in out['measurements']}
            self.assertTrue({'structure_pair', 'projection_pair', 'observer_control', 'scalar_full_control', 'token_pair', 'same_arm_AV'} <= scopes)
            projections = [r for r in out['measurements'] if r['scope'] == 'projection_pair']
            self.assertTrue(projections)
            self.assertTrue(all(r['coordinates'].get('head') is None for r in projections))
            tokens = [r for r in out['measurements'] if r['scope'] == 'token_pair']
            self.assertEqual({r['coordinates']['pass_id'] for r in tokens}, {'ON'})
            self.assertTrue(any(r['metric'] == 'token.full_distribution.js_nat' for r in tokens))

    def test_changed_blob_is_rejected_before_any_normalized_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw')
            blob = next(raw.rglob('*.f32')); blob.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'raw member changed'):
                load_source(self.spec(raw), root)

    def test_unknown_source_adapter_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported E1 source adapter'):
            load_source({'source_id':'x', 'adapter':'dynamic.module', 'path':'does-not-exist'}, '.')

    def test_explicit_sample_subset_does_not_drop_other_sources_into_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw')
            out = load_source(self.spec(raw, sample_ids=['sample-0']), root)
            full = load_source(self.spec(raw), root)
            self.assertEqual([s['sample_id'] for s in out['samples']], ['sample-0'])
            self.assertEqual({r['coordinates'].get('sample_id') for r in out['measurements']}, {'sample-0'})
            self.assertEqual(out['acquisitions'], full['acquisitions'])
            self.assertNotEqual(out['analyses'][0]['id'], full['analyses'][0]['id'])
            with self.assertRaisesRegex(ValueError, 'Unknown E1 source sample'):
                load_source(self.spec(raw, sample_ids=['missing']), root)

    def test_missing_span_and_not_applicable_are_null_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw')
            out = load_source(self.spec(raw), root)
            unavailable = [r for r in out['measurements'] if r['scope'] == 'evidence_mass' and r['status'] == 'NOT_APPLICABLE']
            self.assertTrue(unavailable)
            self.assertTrue(all(r['value'] is None for r in unavailable))

    def test_raw_source_ids_have_explicit_acquisition_and_analyzer_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw')
            out = load_source(self.spec(raw), root)
            self.assertEqual(len(out['acquisitions']), 1)
            self.assertEqual(len(out['analyses']), 1)
            self.assertEqual(out['analyses'][0]['acquisition_id'], out['acquisitions'][0]['id'])
            self.assertIn('analyzer_source_identity', out['analyses'][0]['payload'])
            self.assertIn('RAW_MANIFEST.json', out['locator']['files'])
            self.assertFalse(out['acquisitions'][0]['payload']['new_measurement'])

    def test_no_source_name_task_or_layer_hardcoding_in_common_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = load_source(self.spec(raw_fixture(root / 'raw')), root)
            self.assertEqual({s['task_id'] for s in out['samples']}, {'future_task_0', 'future_task_1'})
            self.assertEqual({r['coordinates']['layer'] for r in out['measurements'] if r['scope'] == 'projection_pair'}, {0,2})

    def test_quarantined_acquisition_keeps_evidence_without_promoting_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw')
            out = load_source(self.spec(raw, integration_status='NEEDS_NEW_ACQUISITION', quarantine_reason='Synthetic invalid coordinate'), root)
            self.assertEqual(out['measurements'], [])
            self.assertTrue(out['evidence']); self.assertTrue(out['samples'])
            self.assertEqual(out['verification']['status'], 'NEEDS_NEW_ACQUISITION')
            with self.assertRaisesRegex(ValueError, 'explicit reason'):
                load_source(self.spec(raw, integration_status='INVALID_ACQUISITION'), root)

    def test_missing_structure_coordinate_remains_null_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = raw_fixture(root / 'raw', missing=('candidate','sample-0','attention_row',0,0,1))
            out = load_source(self.spec(raw), root)
            missing = [r for r in out['measurements'] if r['scope'] == 'structure_pair' and r['status'] == 'UNKNOWN']
            self.assertEqual(len(missing), 1)
            self.assertIsNone(missing[0]['value'])
            self.assertEqual(missing[0]['coordinates']['head'], 0)


if __name__ == '__main__':
    unittest.main()
