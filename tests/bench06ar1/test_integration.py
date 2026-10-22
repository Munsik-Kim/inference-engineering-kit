"""Synthetic CPU acquisition bytes through the common publication/index path.

No model source, compiler, scorer, C1, manifest validator or SQLite transaction
is replaced.  The fault injections below affect only an explicit CPU write or
dispatch boundary.  The old, mixed root is never silently validated as the new
raw-only locator.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from inference_kit import api
from inference_kit.analysis import result_index as index, suite_postprocess as cpu, suite_results
from inference_kit.core.evidence import Limits
from inference_kit.core.io import digest, read, sha
from bench05e3.test_results import seal
from bench05e4b.test_results import fixture as diagnostic_fixture
from bench06a.test_results import serving_fixture


def raw_pin(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, sha(p), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def add_large_derived(root):
    """Sparse and unmanifested: this must never be copied or authenticated."""
    path = root / 'derived' / 'growing-index.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.truncate(Limits().file_bytes + 1)
    return path


class RawPublicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = serving_fixture(self.root / 'mixed-run')
        self.source_id = digest({'adapter': 'suite_e3', 'raw_manifest': sha(self.run / 'RUN_MANIFEST.json')})

    def indexed(self, output, **stages):
        result = suite_results.execute_cpu('index_add', self.run, output, stage_outputs=stages)
        self.assertEqual(result['status'], 'COMPLETE')
        spec = read(output / 'INPUT_SOURCES.json')['sources'][0]
        published = Path(spec['path'])
        self.assertNotEqual(published, self.run)
        self.assertEqual(sha(published / 'RUN_MANIFEST.json'), sha(self.run / 'RUN_MANIFEST.json'))
        return result, published

    def test_mixed_native_root_auto_publishes_and_derived_growth_does_not_change_source(self):
        derived = add_large_derived(self.run)
        original = suite_results.index_source_spec(self.run)
        envelope = suite_results.load_index_source(original, self.root)
        self.assertEqual(index.validate_locator(envelope['locator'])['status'], 'STALE')

        first, published = self.indexed(self.root / 'first')
        manifest = read(self.run / 'RUN_MANIFEST.json')
        self.assertEqual(set(raw_pin(published)), set(manifest['files']) | {'RUN_MANIFEST.json'})
        self.assertFalse((published / derived.relative_to(self.run)).exists())
        pin = raw_pin(published)
        snapshot = index.canonical_export(first['db'])
        current = next(s for s in snapshot['sources'] if s['id'] == self.source_id)
        self.assertEqual(current['validation']['status'], 'RAW_HASH_VERIFIED')
        ranks = [r for r in snapshot['measurements'] if r['metric'] == 'token.reference_target_rank']
        self.assertTrue(ranks)
        self.assertTrue(all(r['value'] is None and r['status'] == 'NOT_COLLECTED' for r in ranks))

        report = suite_results.execute_cpu('report', self.run, self.root / 'report',
                                          stage_outputs={'index_add': self.root / 'first'})
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertEqual(raw_pin(published), pin)
        # The derived artifact grows again after the report has been written.
        with derived.open('r+b') as stream:
            stream.truncate(2 * Limits().file_bytes + 1)
        second, republished = self.indexed(self.root / 'second', historical_index=first['db'],
                                           historical_index_sha256=sha(Path(first['db'])))
        self.assertEqual(first['result']['after'], second['result']['after'])
        self.assertEqual(index.canonical_export(first['db']), index.canonical_export(second['db']))
        self.assertEqual(raw_pin(republished), pin)
        rebuilt, _ = self.indexed(self.root / 'rebuilt')
        self.assertEqual(snapshot, index.canonical_export(rebuilt['db']))

    def test_current_published_raw_tamper_blocks_report_without_success_receipt(self):
        registered, published = self.indexed(self.root / 'registered')
        member = published / 'suite-execution-context.json'
        member.write_bytes(member.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            suite_results.execute_cpu('report', self.run, self.root / 'bad-report',
                                      stage_outputs={'index_add': self.root / 'registered'})
        self.assertFalse((self.root / 'bad-report' / 'REPORT_RECEIPT.json').exists())
        state = index.canonical_export(registered['db'])['sources'][0]['validation']['status']
        self.assertEqual(state, 'STALE')

    def test_live_style_nested_derived_outputs_publish_outside_the_run(self):
        add_large_derived(self.run)
        indexed, published = self.indexed(self.run / 'nodes' / 'index_add' / 'attempt1')
        self.assertFalse(published.resolve().is_relative_to(self.run.resolve()))
        before = raw_pin(published)
        result = suite_results.execute_cpu('report', self.run,
            self.run / 'nodes' / 'report' / 'attempt1',
            stage_outputs={'index_add': self.run / 'nodes' / 'index_add' / 'attempt1'})
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(raw_pin(published), before)
        current = next(row for row in index.canonical_export(indexed['db'])['sources']
                       if row['id'] == self.source_id)
        self.assertEqual(current['validation']['status'], 'RAW_HASH_VERIFIED')

    def test_missing_manifested_raw_never_publishes_or_registers_success(self):
        context = read(self.run / 'suite-execution-context.json')
        member = self.run / next(iter(context['serving_jobs'].values()))['raw_path']
        member.unlink()
        with self.assertRaises(ValueError):
            suite_results.execute_cpu('index_add', self.run, self.root / 'missing')
        receipt = self.root / 'missing' / 'registration' / 'IMPORT_RECEIPT.json'
        if receipt.exists():
            self.assertNotIn('REGISTERED', [s['status'] for s in read(receipt)['sources']])

    def test_historical_stale_is_visible_but_does_not_mask_valid_current_source(self):
        # A different frozen context is a separate synthetic acquisition.  It
        # must not be merged with current data merely because response bytes match.
        historical = self.root / 'historical'
        shutil.copytree(self.run, historical)
        context = read(historical / 'suite-execution-context.json')
        context['synthetic_fixture_revision'] = 'historical-independent-source'
        (historical / 'suite-execution-context.json').write_text(
            json.dumps(context), encoding='utf-8')
        seal(historical)
        previous = suite_results.execute_cpu('index_add', historical, self.root / 'history-index')
        prior_spec = read(self.root / 'history-index' / 'INPUT_SOURCES.json')['sources'][0]
        prior_raw = Path(prior_spec['path']) / 'suite-execution-context.json'
        prior_raw.write_bytes(prior_raw.read_bytes() + b' ')
        current, _ = self.indexed(self.root / 'current-index', historical_index=previous['db'],
                                  historical_index_sha256=sha(Path(previous['db'])))
        export = index.canonical_export(current['db'])
        status = {s['id']: s['validation']['status'] for s in export['sources']}
        self.assertEqual(status[self.source_id], 'RAW_HASH_VERIFIED')
        self.assertEqual(list(status.values()).count('STALE'), 1)
        report = suite_results.execute_cpu('report', self.run, self.root / 'historical-report',
                                          stage_outputs={'index_add': self.root / 'current-index'})
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertIn('STALE', (self.root / 'historical-report' / 'index.html').read_text(encoding='utf-8'))

    def test_registration_fault_rolls_back_all_current_rows(self):
        original = index._store
        def fail_after_store(connection, *args, **kwargs):
            original(connection, *args, **kwargs)
            raise RuntimeError('SYNTHETIC_REGISTRATION_COMMIT_FAILURE')
        with patch.object(index, '_store', side_effect=fail_after_store), self.assertRaisesRegex(
                RuntimeError, 'SYNTHETIC_REGISTRATION'):
            suite_results.execute_cpu('index_add', self.run, self.root / 'rolled-back')
        db = self.root / 'rolled-back' / 'index.sqlite'
        with index._connect(db) as connection:
            self.assertTrue(all(n == 0 for n in index.counts(connection).values()))
            self.assertEqual(connection.execute('SELECT count(*) FROM locators').fetchone()[0], 0)
        receipt = read(self.root / 'rolled-back' / 'registration' / 'IMPORT_RECEIPT.json')
        self.assertEqual(receipt['sources'][0]['status'], 'IMPORT_REJECTED_ROLLED_BACK')

    def test_legacy_diagnostic_cpu_pause_resume_noop_and_current_guard(self):
        legacy_parent = self.root / 'legacy'; legacy_parent.mkdir()
        legacy = diagnostic_fixture(legacy_parent / 'mixed-run')
        add_large_derived(legacy)
        analysis = self.root / 'legacy-analysis'
        paused = api.suite_postprocess(legacy, analysis, stop_after='index_add')
        self.assertEqual(paused['status'], 'CPU_PAUSED_BOUNDARY')
        index_output = analysis / paused['nodes']['index_add']['output']
        published = Path(read(index_output / 'INPUT_SOURCES.json')['sources'][0]['path'])
        self.assertNotEqual(published, legacy)
        resumed = api.suite_postprocess_resume(analysis)
        self.assertEqual(resumed['status'], 'CPU_POSTPROCESS_COMPLETE')
        snapshot = index.canonical_export(index_output / 'index.sqlite')
        self.assertEqual(snapshot['sources'][0]['validation']['status'], 'RAW_HASH_VERIFIED')
        self.assertEqual(len(snapshot['acquisitions']), 8)
        before = raw_pin(analysis / 'stages'); checkpoint = (analysis / 'checkpoint.json').read_bytes()
        with patch.object(cpu, 'execute_cpu', side_effect=AssertionError('No completed CPU work repeats')):
            noop = api.suite_postprocess_resume(analysis)
        self.assertTrue(noop['noop'])
        self.assertEqual((analysis / 'checkpoint.json').read_bytes(), checkpoint)
        self.assertEqual(raw_pin(analysis / 'stages'), before)
        member = published / 'suite-execution-context.json'
        member.write_bytes(member.read_bytes() + b' ')
        with patch.object(cpu, 'execute_cpu') as dispatch, self.assertRaises(ValueError):
            api.suite_postprocess_resume(analysis)
        dispatch.assert_not_called()
        with self.assertRaises(ValueError):
            api.suite_postprocess_status(analysis)


if __name__ == '__main__':
    unittest.main()
