"""The reader owns a copied member's descriptor before target acquisition."""
import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.analysis import evidence_publication as publication
from inference_kit.core.io import sha
from bench06a.test_results import serving_fixture


class CopyDescriptorOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'; self.source.write_bytes(b'original raw evidence')
        self.target = self.root / 'target'

    def copy(self, **changes):
        entry = {'bytes': self.source.stat().st_size, 'sha256': sha(self.source)}
        entry.update(changes)
        publication._copy_member(self.source, self.target, entry, publication._fingerprint(self.source))

    def assert_descriptor_closed_once(self, callback):
        opened = []; explicitly_closed = []
        original_open = os.open; original_close = os.close
        def opening(*args, **kwargs):
            descriptor = original_open(*args, **kwargs); opened.append(descriptor); return descriptor
        def closing(descriptor):
            explicitly_closed.append(descriptor); return original_close(descriptor)
        with patch.object(publication.os, 'open', side_effect=opening), patch.object(
                publication.os, 'close', side_effect=closing):
            callback()
        self.assertEqual(len(opened), 1)
        with self.assertRaises(OSError) as caught: os.fstat(opened[0])
        self.assertEqual(caught.exception.errno, errno.EBADF)
        return explicitly_closed

    def test_existing_target_preserves_file_exists_error_and_both_files(self):
        self.target.write_bytes(b'existing target')
        before = (self.source.read_bytes(), self.target.read_bytes())
        def attempt():
            with self.assertRaises(FileExistsError) as caught: self.copy()
            self.assertEqual(caught.exception.errno, errno.EEXIST)
        self.assertEqual(self.assert_descriptor_closed_once(attempt), [])
        self.assertEqual((self.source.read_bytes(), self.target.read_bytes()), before)

    def test_target_open_enospc_and_eacces_are_not_masked_by_double_close(self):
        original_open = Path.open; before = self.source.read_bytes()
        for expected in (errno.ENOSPC, errno.EACCES):
            with self.subTest(errno=expected):
                def target_open(path, *args, **kwargs):
                    if path == self.target and args == ('xb',):
                        raise OSError(expected, os.strerror(expected))
                    return original_open(path, *args, **kwargs)
                def attempt():
                    with patch.object(Path, 'open', new=target_open), self.assertRaises(OSError) as caught:
                        self.copy()
                    self.assertEqual(caught.exception.errno, expected)
                self.assertEqual(self.assert_descriptor_closed_once(attempt), [])
                self.assertFalse(self.target.exists()); self.assertEqual(self.source.read_bytes(), before)

    def test_fdopen_failure_closes_still_owned_descriptor_once(self):
        def attempt():
            with patch.object(publication.os, 'fdopen', side_effect=OSError(errno.EMFILE, 'fdopen failure')):
                with self.assertRaises(OSError) as caught: self.copy()
            self.assertEqual(caught.exception.errno, errno.EMFILE)
        self.assertEqual(len(self.assert_descriptor_closed_once(attempt)), 1)
        self.assertFalse(self.target.exists())

    def test_normal_and_empty_member_preserve_exact_bytes(self):
        for content in (b'original raw evidence', b''):
            with self.subTest(content=content):
                self.source.write_bytes(content)
                self.copy()
                self.assertEqual(self.target.read_bytes(), content)
                self.assertNotEqual(self.source.stat().st_ino, self.target.stat().st_ino)
                self.target.unlink()

    def test_length_and_sha_mismatch_still_rejected(self):
        for changes in ({'bytes': 1}, {'bytes': 999}, {'sha256': '0' * 64}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, 'grew|changed'): self.copy(**changes)
                self.target.unlink(missing_ok=True)

    def test_stale_source_fingerprint_rejected_before_target_is_created(self):
        fingerprint = publication._fingerprint(self.source)
        self.source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'changed before copy'):
            publication._copy_member(self.source, self.target, {'bytes': 0, 'sha256': '0' * 64}, fingerprint)
        self.assertFalse(self.target.exists())

    def test_public_publisher_target_failure_cleans_only_own_temporary_and_lock(self):
        run = serving_fixture(self.root / 'run'); parent = self.root / 'published'
        parent.mkdir(); unrelated = parent / 'unrelated'; unrelated.write_bytes(b'do not change')
        pins = {p.relative_to(run).as_posix(): (sha(p), p.stat().st_mtime_ns)
                for p in run.rglob('*') if p.is_file()}
        original_open = Path.open
        for expected in (errno.ENOSPC, errno.EACCES):
            with self.subTest(errno=expected):
                def target_open(path, *args, **kwargs):
                    if args == ('xb',) and parent in path.parents:
                        raise OSError(expected, os.strerror(expected))
                    return original_open(path, *args, **kwargs)
                with patch.object(Path, 'open', new=target_open), self.assertRaises(OSError) as caught:
                    publication.publish_suite_evidence(run, parent)
                self.assertEqual(caught.exception.errno, expected)
                self.assertEqual(list(parent.iterdir()), [unrelated])
                self.assertEqual(unrelated.read_bytes(), b'do not change')
                self.assertEqual(pins, {p.relative_to(run).as_posix(): (sha(p), p.stat().st_mtime_ns)
                                      for p in run.rglob('*') if p.is_file()})
        receipt = publication.publish_suite_evidence(run, parent)
        self.assertEqual(receipt['status'], 'RAW_HASH_VERIFIED')
        self.assertFalse(receipt['reused'])
        self.assertTrue(publication.publish_suite_evidence(run, parent)['reused'])
