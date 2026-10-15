import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from inference_kit import demo, plan, compare, assess, replay, verify
from inference_kit.core.io import read, write, digest, sha
from inference_kit.core.state import atomic, Attempt, Budget, cumulative
from inference_kit.core.evidence import open_bundle, inspect_zip, Limits, verify_manifest
from inference_kit.core.paths import Paths, masked
from inference_kit.core.parity import equal
from inference_kit.core.quality import score_document, paired_quality, likelihood_from_response, replay_scores
from inference_kit.experiments.step04 import margin, paired_contrast, screen
from inference_kit.adapters.tensors import module_for_key, choose_keys, check_scheme, BUNDLE

HERE=Path(__file__).parent


class StateTests(unittest.TestCase):
    def test_raw_is_immutable_progress_is_atomic(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'raw.json';write(p,{'a':1})
            with self.assertRaises(ValueError):write(p,{'a':1})
            q=p.with_name('progress.json');atomic(q,1);atomic(q,2)
            self.assertEqual(read(q),2);self.assertEqual(read(p),{'a':1})
            self.assertFalse(list(Path(t).glob('*.tmp')))
    def test_new_attempt_records_parent_without_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            a=Attempt(Path(t)/'a',{'source':'frozen'});a.finish('PARTIAL')
            b=Attempt(Path(t)/'b',{'source':'frozen'},a.path);b.finish('COMPLETE')
            self.assertNotEqual(a.record['attempt_id'],b.record['attempt_id'])
            self.assertEqual(read(a.path/'attempt.json')['status'],'PARTIAL')
            with self.assertRaises(ValueError):Attempt(Path(t)/'c',{'source':'changed'},a.path)
    def test_budget_rejects_duplicate_attempt_and_unresolved_job(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'budget.json';write(p,{'cap_seconds':20,'used_seconds':0,'running':None,'jobs':[]});b=Budget(p)
            b.begin('a',10)
            with self.assertRaises(ValueError):b.begin('b',5)
            b.end(3)
            with self.assertRaises(ValueError):b.begin('a',5)
            self.assertEqual(read(p)['used_seconds'],3)
    def test_ledger_chain_cannot_double_count(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'prior.json';write(p,{'used_seconds':10,'running':None,'jobs':[{'wall_seconds':10}]})
            r={'used_seconds':15,'running':None,'jobs':[{'wall_seconds':5}],
               'inherited_from':{'path':'prior','sha256':sha(p),'used_seconds':10}}
            self.assertEqual(cumulative(r,lambda _:p),15)
            r['used_seconds']=25
            with self.assertRaises(ValueError):cumulative(r,lambda _:p)
    def test_strict_json_rejects_duplicate_nonfinite(self):
        with tempfile.TemporaryDirectory() as t:
            for i,body in enumerate(('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}')):
                p=Path(t)/str(i);p.write_text(body)
                with self.assertRaises(ValueError):read(p)
    def test_crlf_and_float_serialization_are_separate_from_real_changes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'v.json';p.write_bytes(b'{\r\n"x":0.125\r\n}\r\n');equal(read(p),{'x':.125})
        equal({'f':1.0},{'f':1+1e-11})
        for a,b in ((1,1.0),('UNKNOWN','PASS'),(False,0),(float('nan'),float('nan')), (.1,.101)):
            with self.assertRaises(ValueError):equal(a,b)


class ArchiveTests(unittest.TestCase):
    def make(self,p,name='safe.json',blob='{}',mode=None):
        with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
            info=zipfile.ZipInfo(name)
            if mode:info.external_attr=mode<<16
            z.writestr(info,blob)
    def test_rejects_traversal_absolute_windows_paths(self):
        with tempfile.TemporaryDirectory() as t:
            for i,name in enumerate(('../escape','/etc/file','C:/file','a\\b','a/../b','a//b')):
                p=Path(t)/f'{i}.zip';self.make(p,name)
                with self.assertRaises(ValueError):inspect_zip(p)
    def test_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.zip';self.make(p,'link','safe',stat.S_IFLNK|0o777)
            with self.assertRaises(ValueError):inspect_zip(p)
    def test_rejects_duplicate_casefold_member(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('A','');z.writestr('a','')
            with self.assertRaises(ValueError):inspect_zip(p)
    def test_bounded_archive_and_bad_zip(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.zip';self.make(p,blob='x'*20)
            with self.assertRaises(ValueError):inspect_zip(p,Limits(file_bytes=10))
            p.write_bytes(b'bad zip')
            with self.assertRaises(zipfile.BadZipFile):inspect_zip(p)
    def test_never_executes_archive_python(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.zip';self.make(p,'VERIFY_CPU.py','raise SystemExit("must not run")')
            r=verify(p,Path(t)/'out');self.assertFalse(r['python_executed'])
    def test_wrong_expected_hash_and_corrupted_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.zip';self.make(p)
            with self.assertRaises(ValueError):
                with open_bundle(p,expected_sha256='0'*64):pass
            root=Path(t)/'bundle';root.mkdir();(root/'a').write_text('modified')
            write(root/'archive-manifest.json',{'files':{'a':{'sha256':'0'*64,'bytes':8}}})
            with self.assertRaises(ValueError):verify_manifest(root)
    def test_manifest_missing_and_unsafe_reference_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'archive-manifest.json'
            for name in ('missing','../escape'):
                atomic(p,{'files':{name:{'sha256':'0'*64}}})
                with self.assertRaises(ValueError):verify_manifest(Path(t))
    def test_directory_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'b';root.mkdir();(root/'link').symlink_to(Path(t)/'elsewhere')
            with self.assertRaises(ValueError):
                with open_bundle(root):pass


class PathPlanTests(unittest.TestCase):
    def test_paths_use_config_roots_and_preserve_runtime_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);s=root/'settings/spec.json';write(s,{})
            python=root/'runtime/bin/python';python.parent.mkdir(parents=True);python.symlink_to(sys.executable)
            local=root/'local.json';write(local,{'roots':{'PROJECT_ROOT':'.'},'server_python':'${PROJECT_ROOT}/runtime/bin/python'})
            p=Paths.load(s,local)
            self.assertEqual(str(p.resolve('${PROJECT_ROOT}/runtime/bin/python')),str(python))
            self.assertNotEqual(p.resolve('${PROJECT_ROOT}/runtime/bin/python'),python.resolve())
            self.assertEqual(p.resolve('data.json'),s.parent/'data.json')
    def test_unknown_path_binding_and_remote_rejected(self):
        p=Paths({},Path('/tmp'))
        for value in ('${UNKNOWN}/x','https://example.invalid/a'):
            with self.assertRaises(ValueError):p.resolve(value)
    def test_masks_secrets_without_reading_environment(self):
        self.assertEqual(masked({'Authorization':'sensitive','inner':{'hf_token':'value'}}),{'Authorization':'<MASKED>','inner':{'hf_token':'<MASKED>'}})
    def test_plan_never_spawns_process(self):
        with patch('subprocess.run',side_effect=AssertionError('plan launched')):
            value=plan(HERE.parent/'configs/examples/cpu-spec.json',HERE.parent/'configs/examples/local.example.json')
        self.assertFalse(value['gpu_inference']);self.assertFalse(value['downloads'])
    def test_schema_and_dtype_cannot_fall_back(self):
        original=read(HERE.parent/'configs/examples/cpu-spec.json')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'
            for field,value in (('schema_version',99),('backend','other')):
                v=copy.deepcopy(original);v[field]=value;atomic(p,v)
                with self.assertRaises(ValueError):plan(p)
            v=copy.deepcopy(original);v['method']['kv_dtype']='fp8';atomic(p,v)
            with self.assertRaises(ValueError):plan(p)
    def test_launch_kv_axis_preserves_context_and_batch_with_no_duplicate_flags(self):
        from inference_kit.adapters.vllm import launch
        s=read(HERE.parent/'configs/examples/vllm.example.json')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);write(root/'config.json',{'model_type':'llama','dtype':'bfloat16','num_hidden_layers':24,'num_key_value_heads':32,'hidden_size':2048,'num_attention_heads':32})
            write(root/'tokenizer.json',{'synthetic':True})
            write(root/'receipt.json',{'model_id':s['model']['model_id'],'model_revision':s['model']['revision'],'recipe_identity':digest(s['method']),
                  'artifact_identity':'synthetic-only','artifact_config_sha256':sha(root/'config.json'),'tokenizer_files':{'tokenizer.json':sha(root/'tokenizer.json')}})
            paths={'artifact':str(root),'tokenizer':str(root),'artifact_receipt':str(root/'receipt.json'),'server_python':sys.executable,'output':str(root/'out')}
            rows=[]
            for capacity in (4096,8192):
                s['profile']['kv_tokens']=capacity;r=launch(s,paths,{'gpu':0,'gpu_uuid':'synthetic-uuid'});rows.append(r)
                for flag in ('--max-model-len','--max-num-batched-tokens','--kv-cache-memory-bytes'):
                    self.assertEqual(r['argv'].count(flag),1)
                self.assertEqual(r['argv'][r['argv'].index('--max-model-len')+1],'2048')
                self.assertEqual(r['argv'][r['argv'].index('--max-num-batched-tokens')+1],'2048')
            self.assertEqual(rows[1]['kv_bytes'],2*rows[0]['kv_bytes'])


class QualityAndRecipeTests(unittest.TestCase):
    def row(self):
        return {'sample_id':'d','original_index':0,'split':'train','choices':['x','long answer'],'gold':1,
                'question':'q','prompt':'p','request_identity':'i','source_text_hash':'h'}
    def test_acc_norm_and_acc_have_different_decisions(self):
        r=score_document(self.row(),[-1.,-2.]);self.assertEqual((r['acc'],r['acc_norm']),(0,1));self.assertGreater(margin(r),0)
    def test_exact_tie_keeps_first_option_and_no_epsilon(self):
        row=self.row();row['choices']=['x','y'];self.assertEqual(score_document(row,[-1.,-1.])['prediction_acc_norm'],0)
    def test_missing_nonfinite_and_variable_choice_scores(self):
        row=self.row();row['choices']=['a','b','c']
        self.assertEqual(len(score_document(row,[-1.,-2.,-3.])['choice_loglikelihood']),3)
        for scores in ([-1.,-2.],[-1.,-2.,float('nan')]):
            with self.assertRaises(ValueError):score_document(row,scores)
    def test_duplicate_quality_samples_and_unpaired_identity_rejected(self):
        r=score_document(self.row(),[-1.,-2.]);a={'kind':'live_quality','status':'VALID_LIVE_SUBSET','records':[r]};b=copy.deepcopy(a)
        b['records'][0]['source_text_hash']='other'
        with self.assertRaises(ValueError):paired_quality(a,b,20)
        a['records']*=2
        with self.assertRaises(ValueError):paired_quality(a,a,20)
    def test_task_and_split_mismatch_rejected(self):
        r=score_document(self.row(),[-1.,-2.]);a={'kind':'live_quality','status':'VALID_LIVE_SUBSET','records':[r],'task':'ARC','split':'train'}
        for field in ('task','split'):
            b=copy.deepcopy(a);b[field]='different'
            with self.assertRaises(ValueError):compare(a,b,replicates=20)
    def test_research_index_never_becomes_deployment(self):
        self.assertEqual(assess({'evidence_scope':'research_dev_only'},{} )['production_recommendation'],'NOT_ELIGIBLE')
    def records(self):
        base=[{'sample_id':str(i),'request_identity':str(i),'source_text_hash':str(i),'gold':0,'acc_norm':v,'acc':v,'choice_normalized_scores':[float(v),.5]} for i,v in enumerate([0,1,0,1])]
        rows={a:copy.deepcopy(base) for a in ('R00','R10','R01','R11')}
        for arm,values in [('R10',[1,1,0,1]),('R01',[0,1,1,1]),('R11',[1,1,1,1])]:
            for row,value in zip(rows[arm],values):row['acc_norm']=value
        return rows
    def test_joint_bootstrap_uses_same_document_indices(self):
        r=paired_contrast(self.records(),{'R11':1,'R10':-1,'R01':-1,'R00':1},'acc_norm',100)
        self.assertEqual((r['delta'],r['ci95']),(0,[0,0]))
    def test_contrast_sign_and_resource_screen_are_distinct(self):
        a=paired_contrast(self.records(),{'R10':1,'R00':-1},'acc_norm',100);self.assertEqual(a['delta'],.25)
        self.assertEqual(screen(.03,-.01,.001,[0,.06],[-.02,.08])['uncertainty'],'EXPLORATORY_SIGNAL_ONLY')
        self.assertEqual(screen(.01,0,.01,[0,.02],[-.02,.04])['status'],'NO_FOLLOWUP_SIGNAL')
    def test_historical_partial_mock_not_new_valid_scores(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            for kind,status in [('historical','VALID_LIVE_SUBSET'),('mock','VALID_LIVE_SUBSET'),('live_quality','PARTIAL')]:
                atomic(p/'scores.json',{'kind':kind,'status':status,'n_expected_documents':1});(p/'http-responses.jsonl').write_text('')
                with self.assertRaises(ValueError):replay_scores(p,[{'requests':[]}])
    def test_recorded_fixture_replay_and_numeric_corruption(self):
        fixture=HERE/'fixtures/recorded-step04'
        with tempfile.TemporaryDirectory() as t:
            r=replay('step04',fixture,Path(t)/'out');self.assertEqual(r['http_responses_recomputed'],16)
            with self.assertRaises(ValueError):replay('step04',fixture,Path(t)/'out')
    def test_demo_is_synthetic_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            r=demo(Path(t)/'out');self.assertEqual(r['assessment_status'],'INSUFFICIENT_EVIDENCE')
            with self.assertRaises(ValueError):demo(Path(t)/'out')


class TensorSelectionTests(unittest.TestCase):
    def test_exact_prefix_and_missing_bundle(self):
        h='model.layers.7.mlp.down_proj';keys={h+'.'+x for x in BUNDLE}
        self.assertEqual(choose_keys(keys,[h],[h]),keys)
        self.assertIsNone(module_for_key(h+'_extra.weight_scale',[h]))
        self.assertIsNone(module_for_key('model.layers.70.mlp.down_proj.weight_scale',[h]))
        with self.assertRaises(ValueError):choose_keys(keys-{h+'.weight_scale'},[h],[h])
    def test_duplicate_nonexistent_and_overlapping_prefix(self):
        for modules,known in ((['H','H'],['H']),(['missing'],['H'])):
            with self.assertRaises(ValueError):choose_keys(set(),modules,known)
        with self.assertRaises(ValueError):module_for_key('model.layer.weight',['model','model.layer'])


if __name__=='__main__':unittest.main()
