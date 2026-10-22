"""Index identity, transaction and evidence boundaries without models."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.analysis import result_index as idx
from inference_kit.core.io import write, sha, read


def fixture(root, acquisition='acq', revision='analysis'):
    evidence = root/'raw'; evidence.mkdir(exist_ok=True)
    f=evidence/'한국어.json'
    if not f.exists():write(f, {'문항':'보존'})
    return {'source_id':'source-'+acquisition, 'content_identity':'content-'+acquisition,
      'payload':{'run_id':acquisition, 'evidence_kind':'synthetic_contract'},
      'locator':{'path':str(evidence),'files':{'한국어.json':{'bytes':f.stat().st_size,'sha256':sha(f)}}},
      'acquisitions':[{'id':acquisition,'payload':{'scope':'synthetic'}}],
      'analyses':[{'id':revision,'acquisition_id':acquisition}],
      'samples':[{'id':acquisition+'-sample','acquisition_id':acquisition,'task_id':'task','sample_id':'문항'}],
      'executions':[{'id':acquisition+'-exec','acquisition_id':acquisition}],
      'measurements':[{'id':revision+'-metric','analysis_id':revision,'sample_id':acquisition+'-sample','execution_id':acquisition+'-exec','task_id':'task','metric':'strict_success','status':'MEASURED','scope':'historical_quality','value':False}],
      'evidence':[{'id':acquisition+'-evidence','relative_path':'한국어.json','sha256':sha(f),'bytes':f.stat().st_size}],
      'relations':[]}


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.db=self.root/'index.sqlite';self.sources=self.root/'sources.json'
        write(self.sources, {'schema_version':1,'sources':[{'source_id':'alias','adapter':'test','path':'raw'}]})
        self.env=fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def build(self):
        with patch('inference_kit.analysis.index_sources.load_source',return_value=self.env):
            return idx.build_index(self.sources,self.db,self.root/'build')
    def test_identical_registration_idempotent(self):
        first=self.build()
        with patch('inference_kit.analysis.index_sources.load_source',return_value=self.env):
            second=idx.add_sources(self.sources,self.db,self.root/'add')
        self.assertEqual(first['after'],second['after']);self.assertEqual(second['after']['measurements'],1)
    def test_second_locator_does_not_duplicate(self):
        self.build();other=copy.deepcopy(self.env);other['locator']['relative_root']=''
        with patch('inference_kit.analysis.index_sources.load_source',return_value=other):
            result=idx.add_sources(self.sources,self.db,self.root/'add')
        self.assertEqual(result['after']['acquisitions'],1)
    def test_same_input_new_acquisition_retained(self):
        self.build();other=fixture(self.root,'acq2','analysis2')
        write(self.root/'s2.json',{'schema_version':1,'sources':[{'source_id':'alias2'}]})
        with patch('inference_kit.analysis.index_sources.load_source',return_value=other):
            result=idx.add_sources(self.root/'s2.json',self.db,self.root/'add')
        self.assertEqual(result['after']['acquisitions'],2)
    def test_new_analysis_not_new_acquisition(self):
        self.build();other=copy.deepcopy(self.env)
        other['analyses'][0]['id']='a2';other['measurements'][0].update(id='m2',analysis_id='a2')
        with patch('inference_kit.analysis.index_sources.load_source',return_value=other):
            result=idx.add_sources(self.sources,self.db,self.root/'add')
        self.assertEqual(result['after']['acquisitions'],1);self.assertEqual(result['after']['analyses'],2)
    def test_source_alias_conflict(self):
        self.build();other=fixture(self.root,'acq2','a2')
        with patch('inference_kit.analysis.index_sources.load_source',return_value=other),self.assertRaisesRegex(ValueError,'SOURCE_ID_CONFLICT'):
            idx.add_sources(self.sources,self.db,self.root/'add')
        self.assertEqual(len(idx.canonical_export(self.db)['acquisitions']),1)
    def test_transaction_midway_rolls_back(self):
        self.env['measurements'][0]['analysis_id']='missing'
        with self.assertRaises(sqlite3.IntegrityError):self.build()
        self.assertEqual(len(idx.canonical_export(self.db)['sources']),0)
    def test_duplicate_id_rejected(self):
        self.env['samples']*=2
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.build()
    def test_tamper_is_stale_and_not_zero(self):
        self.build();(self.root/'raw'/'한국어.json').write_text('{}',encoding='utf-8')
        result=idx.canonical_export(self.db)
        self.assertEqual(result['sources'][0]['validation']['status'],'STALE')
        self.assertIsNone(result['measurements'][0]['value'])
    def test_missing_root_and_explicit_relocation(self):
        self.build();old=self.root/'raw';new=self.root/'moved';old.rename(new)
        self.assertEqual(idx.canonical_export(self.db)['sources'][0]['validation']['status'],'MISSING_EVIDENCE')
        self.assertEqual(idx.canonical_export(self.db,root_map={str(old):str(new)})['sources'][0]['validation']['status'],'RAW_HASH_VERIFIED')
    def test_path_traversal_rejected(self):
        self.env['evidence'][0]['relative_path']='../outside'
        with self.assertRaises(ValueError):self.build()
    def test_symlink_evidence_is_stale(self):
        self.build();f=self.root/'raw'/'한국어.json';f.rename(self.root/'outside');f.symlink_to(self.root/'outside')
        self.assertEqual(idx.canonical_export(self.db)['sources'][0]['validation']['status'],'STALE')
    def test_rebuild_canonical_semantic_identity(self):
        self.build()
        with patch('inference_kit.analysis.index_sources.load_source',return_value=self.env):
            idx.build_index(self.sources,self.root/'rebuild.sqlite',self.root/'rebuilt')
        self.assertEqual(idx.canonical_export(self.db),idx.canonical_export(self.root/'rebuild.sqlite'))
    def test_zero_task_score_preserved_as_boolean(self):
        self.build();self.assertIs(idx.canonical_export(self.db)['measurements'][0]['value'],False)
    def test_missing_not_imputed(self):
        self.env['measurements'][0].update(status='UNKNOWN',value=0)
        with self.assertRaisesRegex(ValueError,'null'):self.build()
    def test_existing_db_and_output_refused(self):
        self.build()
        with self.assertRaises(ValueError):idx.build_index(self.sources,self.db,self.root/'again')
    def test_triggers_refused(self):
        self.build()
        with sqlite3.connect(self.db) as c:c.execute('CREATE TRIGGER evil AFTER INSERT ON sources BEGIN SELECT 1; END')
        with self.assertRaisesRegex(ValueError,'triggers'):idx.canonical_export(self.db)
    def test_no_sql_task_injection(self):
        self.build()
        with self.assertRaisesRegex(ValueError,'Unknown task'):idx.canonical_export(self.db,"x';DROP TABLE samples;--")
        self.assertEqual(len(idx.canonical_export(self.db)['samples']),1)
    def test_compatibility_refuses_missing_axes_and_cross_run(self):
        self.assertFalse(idx.compatibility({}, {})['delta_permitted'])
        self.assertFalse(idx.compatibility({}, {},'CROSS_RUN_COMPARABLE')['delta_permitted'])
    def test_compatibility_accepts_declared_artifact_treatment(self):
        keys=('task_id','task_version','dataset_identity','sample_id','policy_id','reference_role','prompt_hash','reference_hash','sequence_identity','tokenizer_identity','token_id_space','metric','metric_version','scope','coordinate_identity','probability_kind','acquisition_id','runtime_identity','pass_id')
        left={k:k for k in keys};right=dict(left,artifact='different')
        self.assertTrue(idx.compatibility(left,right)['delta_permitted'])
        for key in keys:
            mismatch=dict(right);mismatch[key]='bad'
            self.assertFalse(idx.compatibility(left,mismatch)['delta_permitted'],key)
    def test_base_import_has_no_ml_or_network(self):
        code="import sys;from inference_kit.analysis import result_index;assert not any(x in sys.modules for x in ('torch','transformers','numpy','vllm','aiohttp'))"
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
