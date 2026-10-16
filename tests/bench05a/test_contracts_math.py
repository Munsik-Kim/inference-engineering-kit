import copy
import io
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr
from inference_kit.core.io import digest,read,write
from inference_kit.core.state import atomic
from inference_kit.probes.contracts import Sample,Execution,Observation,MetricResult,target_positions,attention_coordinates
from inference_kit.probes.blobs import save,load
from inference_kit.metrics.token import nll,distribution_metrics,log_softmax
from inference_kit.metrics.attention import row_metrics,vector_metrics,selected_row_oracle
from inference_kit.suites.bench05a_plan import validate_samples,ledger_evidence
from inference_kit import bench_replay
from inference_kit.cli import main
from .helpers import prepared,synthetic_run


class ContractTests(unittest.TestCase):
    def test_next_token_shift_and_first_continuation(self):
        p=target_positions([9,8,7,6],2);self.assertEqual(p[0],{'token_position':2,'logit_position':1,'target_token_id':7})
    def test_special_eos_padding_are_excluded(self):
        p=target_positions([1,5,8,2,0],2,[0,1,2]);self.assertEqual([x['token_position'] for x in p],[2])
    def test_invalid_boundary_and_limit(self):
        for boundary in (0,3,-1):
            with self.assertRaises(ValueError):target_positions([1,2,3],boundary)
        with self.assertRaises(ValueError):target_positions([1,2],1,maximum=17)
    def test_duplicate_samples_and_shift_mismatch(self):
        for mutate in ('id','shift'):
            p=prepared()
            if mutate=='id':p['rows'][1]['sample_id']=p['rows'][0]['sample_id']
            else:p['rows'][0]['targets'][0]['logit_position']+=1
            p['rows_identity']=digest(p['rows'])
            with self.assertRaises(ValueError):validate_samples(p)
    def test_selected_coordinates_and_gqa_mapping_are_metadata_only(self):
        p=attention_coordinates(8,4,2,2,[{'token_position':3}]);self.assertEqual(p['heads'],[0,2]);self.assertEqual(p['query_to_kv_head'],{'0':0,'2':1})
        self.assertEqual(attention_coordinates(2,1,1,2,[{'token_position':2}])['queries'],[1])
    def test_synthetic_cannot_use_live_label(self):
        value=Observation('id','target_logprob','exec','sample','synthetic_contract','INSTRUMENTED_PATH','MEASURED',{'prefix_hash':'p','token_id_space':'v'},{},{})
        with self.assertRaises(ValueError):value.validate()
    def test_unmeasured_metric_never_becomes_zero(self):
        with self.assertRaises(ValueError):MetricResult('x',0.,'UNSUPPORTED',[],0,1,'positions','toy').record()
        self.assertIsNone(MetricResult('x',None,'BUDGET_SKIPPED',[],0,1,'positions','toy').record()['value'])
    def test_no_opt_in_does_not_touch_runtime(self):
        with patch('subprocess.run',side_effect=AssertionError('runtime launch')),redirect_stderr(io.StringIO()):
            self.assertEqual(main(['bench','run','--plan','missing','--output','unused']),2)
    def test_loss_mask_and_token_id_space_mismatch(self):
        p=prepared();p['rows'][0]['loss_mask'][0]=1;p['rows_identity']=digest(p['rows'])
        with self.assertRaises(ValueError):validate_samples(p)
    def test_missing_budget_and_changed_ledger_hash(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prior=root/'prior.json';write(prior,{'running':None,'used_seconds':10,'jobs':[{'wall_seconds':10}]})
            nxt=root/'next.json';write(nxt,{'running':None,'used_seconds':15,'jobs':[{'wall_seconds':5}],'inherited_from':{'path':str(prior),'sha256':'0'*64,'used_seconds':10}})
            with self.assertRaises(ValueError):ledger_evidence(nxt,root)


class NumericTests(unittest.TestCase):
    def test_blob_roundtrip_hash_shape_and_endianness(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m=save(root,'n',[1.,2.,3.],[3]);self.assertEqual(load(root,m),[1.,2.,3.]);self.assertEqual((root/m['path']).read_bytes(),struct.pack('<3f',1.,2.,3.))
            wrong=dict(m,shape=[2]);
            with self.assertRaises(ValueError):load(root,wrong)
    def test_blob_missing_hash_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m=save(root,'n',[1.],[1])
            for change in ({'path':'../escape'},{'sha256':'0'*64},{'path':'missing.f32'},{'endianness':'big'}):
                with self.assertRaises(ValueError):load(root,m|change)
    def test_nan_inf_rejected_before_storage(self):
        with tempfile.TemporaryDirectory() as t:
            for x in (float('nan'),float('inf')):
                with self.assertRaises(ValueError):save(Path(t),'bad',[x],[1])
    def test_nll_sign_and_natural_log(self):
        self.assertAlmostEqual(nll(math.log(.25)),math.log(4))
        with self.assertRaises(ValueError):nll(.1)
    def test_identical_distributions_have_zero_divergence(self):
        r=distribution_metrics([0.,1.,2.],[0.,1.,2.]);self.assertEqual(r['kl_ref_candidate_nat'],0);self.assertEqual(r['js_nat'],0)
    def test_js_is_symmetric_and_bounded(self):
        a=distribution_metrics([0.,2.,-1.],[3.,-1.,0.]);b=distribution_metrics([3.,-1.,0.],[0.,2.,-1.])
        self.assertAlmostEqual(a['js_nat'],b['js_nat']);self.assertGreaterEqual(a['js_nat'],0);self.assertLessEqual(a['js_nat'],math.log(2))
        self.assertNotAlmostEqual(a['kl_ref_candidate_nat'],b['kl_ref_candidate_nat'])
    def test_top_k_and_vocabulary_mismatch_refused(self):
        with self.assertRaises(ValueError):distribution_metrics([0.,1.],[0.,1.],False)
        with self.assertRaises(ValueError):distribution_metrics([0.,1.],[0.,1.,2.])
    def test_toy_eager_selected_attention_rows_and_mask(self):
        q=[1.,2.];keys=[[1.,0.],[0.,1.],[3.,3.]];values=[[1.,2.],[3.,4.],[5.,6.]]
        r=selected_row_oracle(q,keys,values,1,[True,True,True],1.)
        self.assertEqual(r['row'][2],0);self.assertAlmostEqual(sum(r['row']),1)
        self.assertAlmostEqual(r['row'][0],1/(1+math.exp(1)))
        row_metrics(r['row'],r['row'],r['allowed'])
    def test_all_masked_row_is_not_distribution(self):
        r=selected_row_oracle([1.],[[1.]],[[2.]],0,[False],1.);self.assertIsNone(r['row'])
        with self.assertRaises(ValueError):row_metrics([0.],[0.],[False])
    def test_nonzero_padding_and_bad_row_sum_rejected(self):
        with self.assertRaises(ValueError):row_metrics([.5,.5],[.5,.5],[True,False])
        with self.assertRaises(ValueError):row_metrics([.3,.3],[.3,.3],[True,True])
    def test_output_identity_cosine_and_zero_policy(self):
        r=vector_metrics([2.,3.],[2.,3.]);self.assertEqual(r['nmse'],0);self.assertEqual(r['cosine'],1.)
        r=vector_metrics([0.,0.],[0.,0.]);self.assertIsNone(r['nmse']);self.assertIsNone(r['cosine'])
        self.assertIsNone(vector_metrics([1e-9],[1e-9])['nmse'])


class RawReplayTests(unittest.TestCase):
    def test_synthetic_identity_replay_remains_synthetic(self):
        with tempfile.TemporaryDirectory() as t:
            root=synthetic_run(Path(t)/'raw');result=bench_replay(root,Path(t)/'out')
            self.assertEqual(result['stages']['live_attention'],'SYNTHETIC_CONTRACT_ONLY');self.assertEqual(result['metrics']['mean_delta_nll_nat'],0.)
            self.assertEqual(result['metrics']['raw_top1_match_fraction'],1.);self.assertEqual(result['metrics']['mean_attention_l1'],0.)
    def test_corrupted_blob_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=synthetic_run(Path(t)/'raw');bench_replay(root,Path(t)/'out')
            with self.assertRaises(ValueError):bench_replay(root,Path(t)/'out')
            blob=next(root.rglob('*.f32'));blob.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):bench_replay(root,Path(t)/'out2')
    def test_different_treatment_artifact_is_allowed(self):
        with tempfile.TemporaryDirectory() as t:
            root=synthetic_run(Path(t)/'raw');result=bench_replay(root,Path(t)/'out');self.assertEqual(set(result['counts']['measured_main_arms']),{'bf16','rtn'})
    def test_partial_is_not_new_live_success(self):
        with tempfile.TemporaryDirectory() as t:
            root=synthetic_run(Path(t)/'raw');p=read(root/'progress.json');p['jobs']['main-rtn']['status']='FAILED';atomic(root/'progress.json',p)
            # Record a new test fixture manifest, never overwrite a historical manifest.
            (root/'RAW_MANIFEST.json').unlink()
            from inference_kit.suites.bench05a_run import raw_manifest
            raw_manifest(root);result=bench_replay(root,Path(t)/'out')
            self.assertEqual(result['stages']['live_token'],'BLOCKED_OR_INCOMPLETE');self.assertIsNone(result['metrics']['mean_delta_nll_nat'])
    def test_same_wrong_head_on_both_arms_is_still_outside_policy(self):
        with tempfile.TemporaryDirectory() as t:
            root=synthetic_run(Path(t)/'raw')
            for file in root.rglob('observations.jsonl'):
                records=[__import__('json').loads(s) for s in file.read_text().splitlines()]
                for r in records:
                    if 'head' in r['coordinates'] and r['coordinates']['head']==1:r['coordinates']['head']=2
                file.write_text(''.join(__import__('json').dumps(r)+'\n' for r in records))
            (root/'RAW_MANIFEST.json').unlink()
            from inference_kit.suites.bench05a_run import raw_manifest
            raw_manifest(root)
            with self.assertRaisesRegex(ValueError,'Head'):bench_replay(root,Path(t)/'out')
