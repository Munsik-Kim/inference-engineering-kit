"""A declared raw subset is independent of mutable derived file growth."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.analysis import evidence_publication as publication
from inference_kit.core.evidence import Limits, inspect_directory
from inference_kit.core.io import digest, read, sha, write
from bench05e3.test_results import seal
from bench06a.test_results import serving_fixture


class RawPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.run = serving_fixture(self.root / 'run')
        self.parent = self.root / 'published'

    def publish(self, **kwargs):
        return publication.publish_suite_evidence(self.run, self.parent, **kwargs)

    def rewrite_manifest(self, callback):
        path = self.run / 'RUN_MANIFEST.json'; value = read(path); callback(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_mixed_source_publishes_exact_raw_and_preserves_identity(self):
        # This reproduces the real bounded-reader failure with a sparse,
        # unmanifested derived DB. No product safety limit is increased.
        derived = self.run / 'derived/index.sqlite'; derived.parent.mkdir()
        with derived.open('wb') as stream:
            stream.truncate(Limits().file_bytes + 1)
        with self.assertRaisesRegex(ValueError, 'Evidence file limit'):
            inspect_directory(self.run)
        result = self.publish(manifest_sha256=sha(self.run / 'RUN_MANIFEST.json'))
        target = Path(result['evidence_root']); manifest = read(self.run / 'RUN_MANIFEST.json')
        self.assertEqual(result['status'], 'RAW_HASH_VERIFIED')
        from inference_kit import __version__
        from inference_kit.analysis.tokens import source_identity
        self.assertEqual(result['publisher_package_version'], __version__)
        self.assertEqual(result['publisher_source_digest'], source_identity())
        self.assertEqual(result['publication_scope'], 'FULL_SUITE_RAW_MANIFEST')
        self.assertFalse((target / 'derived').exists())
        self.assertEqual({p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()},
                         set(manifest['files']) | {'RUN_MANIFEST.json'})
        for name in (*manifest['files'], 'RUN_MANIFEST.json'):
            self.assertEqual((self.run / name).read_bytes(), (target / name).read_bytes())
            self.assertNotEqual((self.run / name).stat().st_ino, (target / name).stat().st_ino)

    def test_same_publication_is_idempotent_without_rewriting_members(self):
        first = self.publish(); target = Path(first['evidence_root'])
        pins = {p.relative_to(target).as_posix(): (sha(p), p.stat().st_mtime_ns)
                for p in target.rglob('*') if p.is_file()}
        second = self.publish()
        self.assertTrue(second['reused']); self.assertEqual(first['canonical_source_id'], second['canonical_source_id'])
        self.assertEqual(pins, {p.relative_to(target).as_posix(): (sha(p), p.stat().st_mtime_ns)
                               for p in target.rglob('*') if p.is_file()})

    def test_declared_raw_missing_or_tampered_is_rejected_despite_valid_old_copy(self):
        self.publish(); manifest = read(self.run / 'RUN_MANIFEST.json')
        name = next(n for n in manifest['files'] if n not in ('SUITE_PLAN.json', 'suite-execution-context.json'))
        item = self.run / name; saved = item.read_bytes(); item.write_bytes(saved + b'!')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            self.publish()
        item.unlink()
        with self.assertRaisesRegex(ValueError, 'MISSING_EVIDENCE'):
            self.publish()

    def test_existing_target_content_and_extra_members_conflict(self):
        result = self.publish(); target = Path(result['evidence_root'])
        extra = target / 'unmanifested.txt'; extra.write_text('cannot overwrite this')
        with self.assertRaisesRegex(ValueError, 'SOURCE_ID_CONFLICT'):
            self.publish()
        self.assertEqual(extra.read_text(), 'cannot overwrite this')

    def test_absolute_traversal_normalized_duplicate_and_symlink_are_refused(self):
        original = (self.run / 'RUN_MANIFEST.json').read_bytes()
        for bad in ('../escape', '/escape', 'a/../escape', 'a//escape', 'trailing/'):
            with self.subTest(bad=bad):
                self.rewrite_manifest(lambda m: m['files'].update({bad: {'bytes': 0, 'sha256': '0' * 64}}))
                with self.assertRaises(ValueError): self.publish()
                (self.run / 'RUN_MANIFEST.json').write_bytes(original)
        self.rewrite_manifest(lambda m: m['files'].update({'suite_plan.json': m['files']['SUITE_PLAN.json']}))
        with self.assertRaisesRegex(ValueError, 'Duplicate normalized raw member'):
            self.publish()
        (self.run / 'RUN_MANIFEST.json').write_bytes(original)
        item = self.run / 'SUITE_PLAN.json'; saved = item.read_bytes(); item.unlink()
        external = self.root / 'outside.json'; external.write_bytes(saved); item.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'escapes|Symlink'):
            self.publish()

    def test_output_inside_source_and_symlink_output_parent_refused(self):
        with self.assertRaisesRegex(ValueError, 'outside source'):
            publication.publish_suite_evidence(self.run, self.run / 'evidence')
        real = self.root / 'real'; real.mkdir(); link = self.root / 'link'; link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            publication.publish_suite_evidence(self.run, link / 'evidence')

    def test_invalid_empty_unknown_or_self_declared_manifest_refused(self):
        saved = (self.run / 'RUN_MANIFEST.json').read_bytes()
        changes = (
            lambda m: m.update(files={}),
            lambda m: m.update(schema_version=True),
            lambda m: m.update(unknown='not supported'),
            lambda m: m.update(kind='arbitrary_json'),
            lambda m: m['files'].update({'RUN_MANIFEST.json': {'bytes': 0, 'sha256': '0' * 64}}),
        )
        for change in changes:
            self.rewrite_manifest(change)
            with self.assertRaises(ValueError): self.publish()
            (self.run / 'RUN_MANIFEST.json').write_bytes(saved)

    def test_expected_pin_and_frozen_input_relationship_are_verified(self):
        with self.assertRaisesRegex(ValueError, 'manifest identity changed'):
            self.publish(manifest_sha256='0' * 64)
        context = read(self.run / 'suite-execution-context.json')
        context['sources']['data_identity'] = '0' * 64
        (self.run / 'suite-execution-context.json').write_text(json.dumps(context), encoding='utf-8'); seal(self.run)
        with self.assertRaisesRegex(ValueError, 'inputs differ'):
            self.publish()

    def test_nested_raw_manifest_cannot_hide_a_missing_dependency(self):
        nested = self.run / 'nested'; nested.mkdir(); payload = nested / 'raw.txt'; payload.write_text('raw')
        write(nested / 'RAW_MANIFEST.json', {'schema_version': 1, 'files': {
            'raw.txt': {'bytes': payload.stat().st_size, 'sha256': sha(payload)}}})
        seal(self.run); self.rewrite_manifest(lambda m: m['files'].pop('nested/raw.txt'))
        with self.assertRaisesRegex(ValueError, 'nested raw dependency'):
            self.publish()

    def test_declared_raw_limits_still_apply(self):
        with self.assertRaisesRegex(ValueError, 'file limit'):
            self.publish(limits=Limits(file_bytes=32))
        count = len(read(self.run / 'RUN_MANIFEST.json')['files'])
        with self.assertRaisesRegex(ValueError, 'member limit'):
            self.publish(limits=Limits(members=count))
        with self.assertRaisesRegex(ValueError, 'size limit'):
            self.publish(limits=Limits(total_bytes=1))

    def test_custom_limits_cannot_relax_defaults_or_use_bool(self):
        defaults = Limits()
        for name in ('members', 'file_bytes', 'total_bytes', 'compression_ratio'):
            for value in (True, 0, -1, getattr(defaults, name) + 1):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'positive integer bounds'):
                    self.publish(limits=Limits(**{name: value}))

    def test_copy_and_rename_failure_do_not_publish_partial_evidence(self):
        for target in ('_copy_member', 'os.rename'):
            with self.subTest(target=target), patch.object(publication if target == '_copy_member' else publication.os,
                    target.rsplit('.', 1)[-1], side_effect=OSError('injected CPU publication failure')):
                with self.assertRaisesRegex(OSError, 'injected CPU'):
                    self.publish()
            self.assertEqual(list(self.parent.iterdir()), [])
        self.assertEqual(self.publish()['status'], 'RAW_HASH_VERIFIED')

    def test_mutating_source_while_copying_is_not_published(self):
        original = publication._copy_member; mutated = False
        def changing(source, target, entry, fingerprint):
            nonlocal mutated
            original(source, target, entry, fingerprint)
            if not mutated and source.name != 'RUN_MANIFEST.json':
                source.write_bytes(source.read_bytes() + b'changed'); mutated = True
        with patch.object(publication, '_copy_member', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.publish()
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_derived_growth_after_publication_does_not_change_evidence(self):
        result = self.publish(); target = Path(result['evidence_root'])
        before = {p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*') if p.is_file()}
        derived = self.parent.parent / 'derived'; derived.mkdir()
        with (derived / 'index.sqlite').open('wb') as stream:
            stream.truncate(Limits().file_bytes + 1)
        self.assertTrue(self.publish()['reused'])
        self.assertEqual(before, {p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*') if p.is_file()})

    def test_same_key_publish_lock_is_exclusive_and_never_stolen(self):
        key = sha(self.run / 'RUN_MANIFEST.json'); self.parent.mkdir()
        lock = self.parent / ('.' + key + '.publishing.lock'); lock.write_text('another publisher')
        with self.assertRaisesRegex(ValueError, 'PUBLICATION_IN_PROGRESS'):
            self.publish()
        self.assertEqual(lock.read_text(), 'another publisher')
        lock.unlink(); self.assertEqual(self.publish()['status'], 'RAW_HASH_VERIFIED')

    def test_production_execution_plan_identity_and_dependency_closure_verified(self):
        compiled = read(self.run / 'SUITE_PLAN.json')
        inputs = {name: {'bytes': (self.run / name).stat().st_size, 'sha256': sha(self.run / name)}
                  for name in ('SUITE_PLAN.json', 'suite-execution-context.json')}
        plan = {'kind': 'suite_execution_plan_v1', 'compiled_plan_identity': compiled['plan_identity'],
                'immutable_inputs': inputs, 'executor_source': 'historical-acquisition-source',
                'units': [{'node_id': 'validate', 'operation_kind': 'validate_prepared_inputs',
                           'dependencies': [], 'required': True, 'counts': {'documents': 1}}]}
        plan['execution_plan_identity'] = digest(plan)
        write(self.run / 'EXECUTION_PLAN.json', plan); write(self.run / 'budget-continuation.json', {})
        seal(self.run)
        self.rewrite_manifest(lambda m: m.update(kind='suite_e3_raw_v1', immutable=True))
        self.assertEqual(self.publish()['status'], 'RAW_HASH_VERIFIED')
        # Resealing a construction fixture cannot authorize a changed frozen
        # plan or a dependency silently removed from the raw subset.
        plan['compiled_plan_identity'] = '0' * 64
        plan['execution_plan_identity'] = digest({k: v for k, v in plan.items() if k != 'execution_plan_identity'})
        (self.run / 'EXECUTION_PLAN.json').write_text(json.dumps(plan), encoding='utf-8'); seal(self.run)
        self.rewrite_manifest(lambda m: m.update(kind='suite_e3_raw_v1', immutable=True))
        with self.assertRaisesRegex(ValueError, 'frozen plan identity differ'):
            self.publish()

    def test_legacy_slice_is_disclosed_and_explicit_forbidden_slice_refused(self):
        self.rewrite_manifest(lambda m: m.update(scope='suite_e3_task_slice_v1'))
        self.assertEqual(self.publish()['publication_scope'], 'DECLARED_LEGACY_SLICE')
        self.rewrite_manifest(lambda m: m.update(derived_slice_kind='suite_native_task_slice_v1'))
        with self.assertRaisesRegex(ValueError, 'SLICE_ONLY'):
            self.publish()
