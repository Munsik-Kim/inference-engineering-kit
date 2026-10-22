"""Portable C1 view contracts, integrity, locale and CPU boundaries."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import inference_kit
from inference_kit.analysis.tokens import compare_tokens,verify_view,SCHEMA,write_jsonl,source_identity
from inference_kit.core.io import digest,sha,write


def view(root):
    root.mkdir();seq={'sequence_id':'seq','input_ids':[1,2,3]}
    base={'record_id':'a','source_id':'panel','bundle_identity':'oldsha','plan_identity':'plan',
      'execution_id':'exec','observation_id':'obs','arm_id':'left','task_id':'toy','task_version':'1','dataset_identity':'data',
      'split':'DEV','sample_id':'한글','policy_id':'p','choice_index':None,'reference_scope':'AUTHOR_GOLD',
      'prompt_hash':'prompt','reference_hash':'ref','sequence_id':'seq','tokenizer_identity':'tok','token_id_space':'space',
      'vocabulary_size':4,'acquisition_source':'instrumented_diagnostic','runtime_identity':'runtime',
      'backend_policy':'diagnostic','probability_stage':'raw_pre_sampling','metric_contract':'target_logprob_v1',
      'token_position':1,'logit_position':0,'target_token_id':2,'prefix_hash':digest({'token_ids':[1],'token_id_space':'space'}),
      'target_logprob':-1.,'raw_argmax_token_id':2,'full_logits':None,'field_tags':['answer'],'status':'MEASURED','requested':True}
    other=dict(base,record_id='b',arm_id='right',artifact_identity='changed',acquisition_source='decoded_diagnostic',target_logprob=-2.)
    write_jsonl(root/'token_records.jsonl',[base,other]);write_jsonl(root/'sequences.jsonl',[seq]);write_jsonl(root/'generation_records.jsonl',[])
    write(root/'COVERAGE.json',{'scope':'synthetic_contract_only'})
    inventory={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in root.iterdir()}
    write(root/'TOKEN_VIEW_MANIFEST.json',{'schema_version':1,'kind':SCHEMA,'analysis_source_digest':source_identity(),
      'source_panels':[],'files':inventory,'view_identity':digest(inventory)})
    return base,other


class ViewTests(unittest.TestCase):
    def test_self_contained_view_and_delta(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view');r=compare_tokens(p/'view','left','right',p/'out')
            self.assertEqual(r['coverage']['measured'],1)
            x=json.loads((p/'out/token-comparisons.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(x['metrics']['delta_nll_nat'],1.)
            self.assertIsNone(x['metrics']['full_distribution'].get('kl_ref_candidate_nat'))
    def test_view_corrupt_member_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view');(p/'view/token_records.jsonl').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):compare_tokens(p/'view','left','right',p/'out')
    def test_extra_unmanifested_member_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view');(p/'view/extra').write_text('extra',encoding='utf-8')
            with self.assertRaises(ValueError):verify_view(p/'view')
    def test_manifest_own_identity_changed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view');q=p/'view/TOKEN_VIEW_MANIFEST.json';x=json.loads(q.read_text(encoding='utf-8'));x['view_identity']='bad';q.write_text(json.dumps(x),encoding='utf-8')
            with self.assertRaises(ValueError):verify_view(p/'view')
    def test_immutable_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view');compare_tokens(p/'view','left','right',p/'out')
            with self.assertRaises(ValueError):compare_tokens(p/'view','left','right',p/'out')
    def test_output_inside_view_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view')
            with self.assertRaises(ValueError):compare_tokens(p/'view','left','right',p/'view/out')
    def test_archive_relocation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view')
            with zipfile.ZipFile(p/'view.zip','w') as z:
                for f in (p/'view').iterdir():z.write(f,'relocated/'+f.name)
            r=compare_tokens(p/'view.zip','left','right',p/'out');self.assertEqual(r['coverage']['measured'],1)
    def test_unsafe_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with zipfile.ZipFile(p/'bad.zip','w') as z:z.writestr('../escape','bad')
            with self.assertRaises(ValueError):compare_tokens(p/'bad.zip','left','right',p/'out')
    def test_duplicate_input_panel_rejected_before_output(self):
        from inference_kit import tokens_export
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write(p/'sources.json',{'schema_version':1,'kind':'token_input_sources','sources':[{'source_id':'a','adapter':'bench05b4','path':'missing'}]*2})
            with self.assertRaises(ValueError):tokens_export(p/'sources.json',p/'out')
            self.assertFalse((p/'out').exists())
    def test_distinct_arm_labels_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view')
            with self.assertRaises(ValueError):compare_tokens(p/'view','same','same',p/'out')
    def test_ascii_locale_cli_and_utf8_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);view(p/'view');package_root=str(Path(inference_kit.__file__).absolute().parent.parent)
            code='import sys,locale;sys.path.insert(0,'+repr(package_root)+');assert locale.getpreferredencoding(False).lower() not in ("utf-8","utf8");from inference_kit.cli import main;raise SystemExit(main(["bench","tokens","compare","--view","view","--baseline","left","--candidate","right","--output","out"]))'
            child=subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',code],cwd=p,env=dict(os.environ,LC_ALL='C',LANG='C'),capture_output=True,encoding='utf-8')
            self.assertEqual(child.returncode,0,child.stderr);self.assertEqual(json.loads(child.stdout)['GPU_actions'],0)
            self.assertIn('한글',(p/'out/per-document.csv').read_text(encoding='utf-8'))
    def test_base_cli_help_import_no_sdk(self):
        package_root=str(Path(inference_kit.__file__).absolute().parent.parent)
        code='import sys;sys.path.insert(0,'+repr(package_root)+');import inference_kit;from inference_kit.cli import main;main(["bench","tokens","--help"])'
        child=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,encoding='utf-8')
        self.assertEqual(child.returncode,0,child.stderr);self.assertIn('compare',child.stdout)


if __name__=='__main__':unittest.main()
