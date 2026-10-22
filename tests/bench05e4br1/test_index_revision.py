"""Real product source readers and SQLite transactions on closed CPU evidence.

These constructed request/TokenView fixtures are SYNTHETIC_REGRESSION, never
new model acquisitions. No source/hash/compiler validation is replaced.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from inference_kit.analysis import result_index as index, suite_results
from inference_kit.analysis.index_sources import load_source
from inference_kit.core.io import read, sha, write
from inference_kit.core.state import atomic
from bench05e3.test_results import fixture, view_fixture


class IndexRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        parent = fixture(self.root / 'parent')
        view = view_fixture(self.root / 'view', read(parent / 'suite-execution-context.json'))
        self.run = self.root / 'raw'
        suite_results.export_mini_slice(parent, view, self.run)

    def tearDown(self):
        self.temp.cleanup()

    def spec(self, run=None, revision=None):
        return suite_results.index_source_spec(run or self.run, analysis_revision=revision)

    def source_file(self, name, *specs):
        path = self.root / (name + '.json')
        write(path, {'schema_version': 1, 'sources': list(specs)})
        return path

    def second_source(self):
        target = self.root / 'second-raw'; shutil.copytree(self.run, target)
        path = target / 'suite-execution-context.json'; context = read(path)
        context['synthetic_fixture_revision'] = 'independent-acquisition-contract'
        atomic(path, context)
        manifest_path = target / 'RUN_MANIFEST.json'; manifest = read(manifest_path)
        manifest['files']['suite-execution-context.json'] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        atomic(manifest_path, manifest)
        return target

    def test_suite_display_alias_is_scoped_by_canonical_source_not_locator(self):
        moved = self.root / 'moved'; shutil.copytree(self.run, moved)
        left, right = self.spec(), self.spec(moved)
        self.assertEqual(left['source_id'], 'suite-current')
        self.assertEqual(left['alias_scope'], right['alias_scope'])
        self.assertNotEqual(left['path'], right['path'])
        second = self.spec(self.second_source())
        self.assertEqual(second['source_id'], left['source_id'])
        self.assertNotEqual(second['alias_scope'], left['alias_scope'])

    def test_distinct_sources_share_display_alias_without_identity_collision(self):
        second = self.second_source(); manifest = self.source_file('sources', self.spec(), self.spec(second))
        db = self.root / 'index.sqlite'; receipt = index.build_index(manifest, db, self.root / 'build')
        self.assertEqual(receipt['after']['sources'], 2)
        self.assertEqual(receipt['after']['acquisitions'], 8)
        self.assertEqual({r['source_alias'] for r in receipt['sources']}, {'suite-current'})
        self.assertEqual(len({r['alias_storage_key'] for r in receipt['sources']}), 2)
        again = index.add_sources(manifest, db, self.root / 'again')
        self.assertEqual(receipt['after'], again['after'])

    def test_same_scope_different_source_is_still_atomic_conflict(self):
        db = self.root / 'index.sqlite'
        before = index.build_index(self.source_file('first', self.spec()), db, self.root / 'build')['after']
        other = self.spec(self.second_source()); other['alias_scope'] = self.spec()['alias_scope']
        with self.assertRaisesRegex(ValueError, 'SOURCE_ID_CONFLICT: declared source alias'):
            index.add_sources(self.source_file('conflict', other), db, self.root / 'rejected')
        with index._connect(db) as connection:
            self.assertEqual(index.counts(connection), before)
        self.assertEqual(read(self.root / 'rejected/IMPORT_RECEIPT.json')['sources'][0]['status'], 'IMPORT_REJECTED_ROLLED_BACK')

    def test_legacy_global_alias_conflict_is_not_weakened(self):
        first = self.spec(); first.pop('alias_scope')
        db = self.root / 'index.sqlite'
        index.build_index(self.source_file('first', first), db, self.root / 'build')
        second = self.spec(self.second_source()); second.pop('alias_scope')
        with self.assertRaisesRegex(ValueError, 'SOURCE_ID_CONFLICT'):
            index.add_sources(self.source_file('second', second), db, self.root / 'rejected')

    def test_same_raw_new_revision_retains_acquisition_and_idempotent_complete(self):
        first = suite_results.execute_cpu('index_add', self.run, self.root / 'first')
        history = Path(first['db']); initial = first['result']['after']
        stage = {'historical_index': history, 'historical_index_sha256': sha(history), 'analysis_revision': 'cpu-r1'}
        second = suite_results.execute_cpu('index_add', self.run, self.root / 'second', stage_outputs=stage)
        self.assertEqual(second['status'], 'COMPLETE')
        self.assertEqual(second['result']['after']['acquisitions'], initial['acquisitions'])
        self.assertEqual(second['result']['after']['sources'], initial['sources'])
        self.assertEqual(second['result']['after']['analyses'], 2 * initial['analyses'])
        latest = Path(second['db'])
        stage.update(historical_index=latest, historical_index_sha256=sha(latest))
        third = suite_results.execute_cpu('index_add', self.run, self.root / 'third', stage_outputs=stage)
        self.assertEqual(third['status'], 'COMPLETE')
        self.assertEqual(third['result']['before'], third['result']['after'])
        self.assertEqual(index.canonical_export(latest), index.canonical_export(third['db']))

    def test_same_source_different_locator_revalidates_without_duplicate_metrics(self):
        db = self.root / 'index.sqlite'
        original = index.build_index(self.source_file('first', self.spec()), db, self.root / 'build')
        moved = self.root / 'relocated'; shutil.copytree(self.run, moved)
        repeated = index.add_sources(self.source_file('moved', self.spec(moved)), db, self.root / 'add')
        self.assertEqual(original['after'], repeated['after'])
        context = moved / 'suite-execution-context.json'; context.write_bytes(context.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            index.add_sources(self.source_file('tampered', self.spec(moved)), db, self.root / 'bad')

    def test_rebuild_reverse_registration_has_same_canonical_semantics(self):
        second = self.second_source()
        left = self.source_file('forward', self.spec(), self.spec(second))
        right = self.source_file('reverse', self.spec(second), self.spec())
        a, b = self.root / 'a.sqlite', self.root / 'b.sqlite'
        index.build_index(left, a, self.root / 'a'); index.build_index(right, b, self.root / 'b')
        self.assertEqual(index.canonical_export(a), index.canonical_export(b))

    def test_transaction_midwrite_rollback_preserves_alias_and_measurements(self):
        envelope = load_source(self.spec(), self.root)
        envelope = deepcopy(envelope); envelope['measurements'][0]['analysis_id'] = 'missing-analysis'
        db = index._create(self.root / 'broken.sqlite')
        with sqlite3.connect(db) as connection:
            connection.execute('PRAGMA foreign_keys=ON')
            with self.assertRaises(sqlite3.IntegrityError):
                with connection:
                    index._store(connection, envelope, 'suite-current', self.spec()['alias_scope'])
            self.assertTrue(all(value == 0 for value in index.counts(connection).values()))
            self.assertEqual(connection.execute('SELECT count(*) FROM aliases').fetchone()[0], 0)

    def test_invalid_scope_revision_and_history_pin_rejected(self):
        for invalid in ('', 7, False, 'x' * 129):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'revision'):
                self.spec(revision=invalid)
        for invalid in ('', 7, False):
            with self.subTest(scope=invalid), self.assertRaisesRegex(ValueError, 'alias scope'):
                index._alias_identity('display', invalid)
        first = suite_results.execute_cpu('index_add', self.run, self.root / 'first')
        with self.assertRaisesRegex(ValueError, 'Historical index differs'):
            suite_results.execute_cpu('index_add', self.run, self.root / 'bad-history', stage_outputs={
                'historical_index': first['db'], 'historical_index_sha256': '0' * 64})

    def test_report_evidence_paths_are_source_namespaced(self):
        second = self.second_source(); db = self.root / 'index.sqlite'
        index.build_index(self.source_file('sources', self.spec(), self.spec(second)), db, self.root / 'build')
        result = index.report_index(db, self.root / 'report')
        self.assertEqual(result['kind'], 'offline_evidence_report_v1')
        # Identical basenames under different source roots stay separate.
        report = self.root / 'report'
        links = result['portable_evidence']
        self.assertTrue(links)
        directories = {Path(row['path']).parts[1] for row in links.values()}
        self.assertEqual(len(directories), 2)


if __name__ == '__main__':
    unittest.main()
