import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit import assess, compare, verify
from inference_kit.cli import main
from inference_kit.core.index import check_index
from inference_kit.core.io import write, digest, sha
from inference_kit.core.evidence import open_bundle, member_name
from inference_kit.core.quality import paired_quality


class BoundaryTests(unittest.TestCase):
    def test_cli_run_requires_opt_in_before_runtime_or_gpu(self):
        with patch('subprocess.run',side_effect=AssertionError('GPU command')),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['run','--spec','missing','--local','missing','--output','unused']),2)
    def test_compose_requires_opt_in_without_importing_tensor_runtime(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['compose','--base','missing','--intervention','missing','--base-receipt','a','--intervention-receipt','b','--modules','H','--output','unused']),2)
    def test_unbound_real_index_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'index.json';write(p,{'kind':'verified_result_index','schema_version':3,'results':[]})
            with self.assertRaisesRegex(ValueError,'binding'):assess(p,{})
    def test_index_rejects_corrupted_reference_and_summary(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);i={'kind':'verified_result_index','schema_version':3,'evidence_root':str(root),'results':[],
                          'results_identity':digest([]),'evidence_manifest':{'raw.json':{'sha256':'0'*64,'bytes':2}}}
            (root/'raw.json').write_text('{}')
            with self.assertRaises(ValueError):check_index(i,root,root)
            i['results_identity']='wrong'
            with self.assertRaises(ValueError):check_index(i,root,root)
    def test_directory_cannot_misreport_expected_zip_hash(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                with open_bundle(t,expected_sha256='0'*64):pass
    def test_dot_archive_name_rejected(self):
        with self.assertRaises(ValueError):member_name('.')
    def test_jsonl_duplicate_keys_not_silently_overridden(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'records.jsonl';p.write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):compare(p,p)
    def test_mock_does_not_acquire_live_subset_label(self):
        fake={'kind':'mock','status':'VALID_LIVE_SUBSET','records':[]}
        with self.assertRaises(ValueError):paired_quality(fake,fake,20)
    def test_frozen_directory_cannot_receive_derived_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.assertRaises(ValueError):verify(root,root/'new')
            self.assertFalse((root/'new').exists())
    def test_failed_derivation_leaves_new_failure_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with self.assertRaises(ValueError):verify(root/'missing.zip',root/'attempt1')
            self.assertTrue((root/'attempt1/failure.json').is_file())
    def test_archival_missing_reference_is_explicit_without_invalidating_primary(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'bundle';root.mkdir();current=root/'step03_fixture';current.mkdir()
            write(current/'PLAN_FROZEN.json',{'task':'step03'})
            raw=current/'raw.json';write(raw,{})
            index={'kind':'verified_result_index','schema_version':3,'evidence_root':str(root),'results':[],
                   'results_identity':digest([]),'evidence_manifest':{'step03_fixture/raw.json':{'sha256':sha(raw),'bytes':raw.stat().st_size}}}
            write(current/'results-index.json',index)
            old=root/'archived';old.mkdir();index['evidence_manifest']={'missing.json':{'sha256':'0'*64,'bytes':2}}
            write(old/'results-index.json',index)
            result=verify(root,Path(t)/'new')
            self.assertEqual(result['reference_coverage'],'PRIMARY_VERIFIED_ARCHIVAL_PARTIAL')
            self.assertEqual(len(result['archival_reference_issues']),1)
            self.assertEqual(result['indexes_verified'],1)


if __name__=='__main__':unittest.main()
