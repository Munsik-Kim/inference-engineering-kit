"""Token-only selected LM head and serialized consistency, no ML SDK required."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from inference_kit.core.io import read
from inference_kit.probes.local_transformer import selected_token_forward
from inference_kit.probes.token_worker import store_numeric,check_saved_consistency,CONSISTENCY_POLICY,NORMALIZATION,validate_worker_row,observation_details
from inference_kit.probes.blobs import save,load
from inference_kit.metrics.token import log_softmax


class Tensor:
    def __init__(self,shape,element_size=2):self.shape=shape;self.byte_size=element_size;self.dtype='bfloat16' if element_size==2 else 'float32'
    def numel(self):return math.prod(self.shape)
    def element_size(self):return self.byte_size
    def float(self):return Tensor(self.shape,4)


class Hidden(Tensor):
    def __init__(self,shape,events):super().__init__(shape);self.events=events
    def __getitem__(self,key):
        batch,positions,columns=key
        assert batch==0 and isinstance(positions,list) and columns==slice(None)
        self.events.append(('selected_hidden',positions))
        return Tensor((len(positions),self.shape[-1]))


class FakeModel:
    def __init__(self,vocab=10):self.events=[];self.vocab=vocab;self.head_input_shapes=[]
    def model(self,**kwargs):
        assert kwargs['use_cache'] is False
        self.events.append(('backbone',kwargs['input_ids'].shape))
        return types.SimpleNamespace(last_hidden_state=Hidden((1,kwargs['input_ids'].shape[1],6),self.events))
    def lm_head(self,chosen):
        self.events.append(('native_lm_head',chosen.shape));self.head_input_shapes.append(chosen.shape)
        return Tensor((chosen.shape[0],self.vocab))
    def __call__(self,*args,**kwargs):raise AssertionError('Must never request full model [T,V] output')


class Functional:
    def __init__(self,events):self.events=events
    def log_softmax(self,logits,dim):
        assert dim==-1 and logits.dtype=='float32';self.events.append(('float32_log_softmax',logits.shape));return Tensor(logits.shape,4)


class SelectedWorkerTests(unittest.TestCase):
    def targets(self):return [{'token_position':2,'logit_position':1,'target_token_id':3},{'token_position':7,'logit_position':6,'target_token_id':4}]
    def test_token_only_backbone_then_selected_native_lm_head(self):
        model=FakeModel();result=selected_token_forward(model,Tensor((1,128)),Tensor((1,128)),self.targets(),256*1024**2,functional=Functional(model.events))
        self.assertEqual(model.head_input_shapes,[(2,6)])
        self.assertEqual(result['logits'].shape,(2,10));self.assertEqual(result['logprobs'].shape,(2,10))
        self.assertEqual(model.events,[('backbone',(1,128)),('selected_hidden',[1,6]),('native_lm_head',(2,6)),('float32_log_softmax',(2,10))])
        self.assertEqual(result['required_buffer_bytes'],144)

    def test_old_observer_wraps_backbone_only_call_order_preserved(self):
        model=FakeModel()
        @contextmanager
        def context():
            model.events.append('observer_enter')
            try:yield
            finally:model.events.append('observer_exit')
        selected_token_forward(model,Tensor((1,128)),Tensor((1,128)),self.targets(),9999,context=context(),functional=Functional(model.events))
        self.assertEqual(model.events[:3],['observer_enter',('backbone',(1,128)),'observer_exit'])
        self.assertEqual(model.events[4][0],'native_lm_head')

    def test_exception_during_backbone_restores_optional_old_context(self):
        state=[];model=FakeModel()
        @contextmanager
        def context():
            state.append('enter')
            try:yield
            finally:state.append('exit')
        def fail(**kwargs):raise RuntimeError('fixture error')
        model.model=fail
        with self.assertRaisesRegex(RuntimeError,'fixture error'):
            selected_token_forward(model,Tensor((1,128)),Tensor((1,128)),self.targets(),9999,context=context(),functional=Functional(model.events))
        self.assertEqual(state,['enter','exit'])

    def test_shared_buffer_cap_retained(self):
        model=FakeModel()
        with self.assertRaisesRegex(ValueError,'buffer cap'):
            selected_token_forward(model,Tensor((1,128)),Tensor((1,128)),self.targets(),143,functional=Functional(model.events))

    def test_invalid_duplicate_positions_batch_and_shift_rejected(self):
        for targets,shape in (([],(1,128)),(self.targets()*2,(1,128)),([{'token_position':128,'logit_position':127,'target_token_id':1}],(1,128)),([{'token_position':2,'logit_position':0,'target_token_id':1}],(1,128)),(self.targets(),(2,128))):
            model=FakeModel()
            with self.subTest(targets=targets,shape=shape),self.assertRaises(ValueError):
                selected_token_forward(model,Tensor(shape),Tensor(shape),targets,99999,functional=Functional(model.events))

    def test_target_outside_native_vocabulary_rejected(self):
        model=FakeModel();targets=[{'token_position':2,'logit_position':1,'target_token_id':10}]
        with self.assertRaisesRegex(ValueError,'outside vocabulary'):
            selected_token_forward(model,Tensor((1,128)),Tensor((1,128)),targets,99999,functional=Functional(model.events))

    def test_same_saved_f32_consistency_exact_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);logits=[1.,2.,3.];lp=log_softmax(logits)[1]
            scalar=save(root,'scalar',[lp],[1]);full=save(root,'full',logits,[3])
            result=check_saved_consistency(root,scalar,full,1,CONSISTENCY_POLICY)
            self.assertEqual(result['status'],'PASS');self.assertLessEqual(result['absolute_difference_nat'],2e-5)
            with self.assertRaisesRegex(ValueError,'Unapproved'):
                check_saved_consistency(root,scalar,full,1,dict(CONSISTENCY_POLICY,absolute_tolerance_nat=1.))

    def test_consistency_violation_keeps_original_scalar_and_full_blobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scalar=save(root,'bad',[-3.],[1]);full=save(root,'full',[1.,2.,3.],[3])
            result=check_saved_consistency(root,scalar,full,1,CONSISTENCY_POLICY)
            self.assertEqual(result['status'],'FAILED');self.assertEqual(load(root,scalar),[-3.]);self.assertEqual(load(root,full),[1.,2.,3.])

    def test_nonfinite_coordinate_bytes_preserved_not_measured_blob(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);numeric,failed,used=store_numeric(root,'failed',[math.nan,math.inf,-math.inf],[3],0,256)
            self.assertIsNone(numeric);self.assertEqual(used,12)
            self.assertEqual(failed['numeric_status'],'NONFINITE_NOT_VALIDATED')
            self.assertEqual((failed['nan_count'],failed['positive_infinity_count'],failed['negative_infinity_count']),(1,1,1))
            self.assertEqual((root/failed['path']).stat().st_size,12)
            self.assertIsInstance(json.dumps(failed,allow_nan=False),str)

    def test_raw_storage_bound_and_immutable_blob(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);numeric,failed,used=store_numeric(root,'once',[1.,2.],[2],0,8)
            self.assertIsNone(failed);self.assertEqual(load(root,numeric),[1.,2.]);self.assertEqual(used,8)
            with self.assertRaises(FileExistsError):store_numeric(root,'once',[1.,2.],[2],0,8)
            with self.assertRaisesRegex(ValueError,'storage cap'):store_numeric(root,'more',[1.],[1],used,8)

    def test_worker_import_and_help_do_not_import_torch_or_attention_interface(self):
        import inference_kit
        installed=Path(inference_kit.__file__).parent.parent
        code="""import sys
sys.path.insert(0,sys.argv[1])
sdk={'torch','transformers','vllm','numpy','aiohttp'}
def guard(event,args):
    if event=='import' and args and str(args[0]).split('.')[0] in sdk:raise AssertionError('forbidden SDK')
sys.addaudithook(guard)
import runpy
sys.argv=['token_worker','--help']
try:runpy.run_module('inference_kit.probes.token_worker',run_name='__main__')
except SystemExit as exc:assert exc.code==0
assert 'inference_kit.probes.attention' not in sys.modules
assert not any(k.split('.')[0] in sdk for k in sys.modules)
print('TOKEN_WORKER_CPU_IMPORT_PASS')
"""
        child=subprocess.run([sys.executable,'-I','-c',code,str(installed)],capture_output=True,encoding='utf-8',timeout=30)
        self.assertEqual(child.returncode,0,child.stderr);self.assertIn('TOKEN_WORKER_CPU_IMPORT_PASS',child.stdout)

    def test_frozen_main32_full8_accepted_and_small8_full2_rejects_overruns(self):
        spec={'max_sequence_tokens':640,'max_target_positions_per_sample':32,'max_full_vocab_positions_per_sample':8,'observe_attention':False}
        ids=list(range(100));targets=[{'token_position':t,'logit_position':t-1,'target_token_id':ids[t]} for t in range(20,52)]
        row={'input_ids':ids,'targets':targets,'full_vocab_target_positions':list(range(20,28)),'prompt_token_boundary':20,'loss_mask':[int(t>=20) for t in range(100)],'attention_mask':[1]*100}
        self.assertEqual(validate_worker_row(row,spec,100),{'sequence_tokens':100,'target_positions':32,'full_vocab_positions':8,'attention_rows':0})
        small=dict(spec,max_target_positions_per_sample=8,max_full_vocab_positions_per_sample=2)
        with self.assertRaisesRegex(ValueError,'target count cap'):validate_worker_row(row,small,100)
        smaller=dict(row,targets=targets[:8])
        with self.assertRaisesRegex(ValueError,'full-vocabulary selection'):validate_worker_row(smaller,small,100)
        self.assertEqual(validate_worker_row(dict(smaller,full_vocab_target_positions=[20,21]),small,100)['target_positions'],8)

    def test_worker_coordinate_ids_and_full_positions_reject_boolean_or_float(self):
        spec={'max_sequence_tokens':640,'max_target_positions_per_sample':32,'max_full_vocab_positions_per_sample':8,'observe_attention':False}
        original={'input_ids':[0,1,2,3],'targets':[{'token_position':1,'logit_position':0,'target_token_id':1}],'full_vocab_target_positions':[1]}
        for key,value in (('token_position',True),('logit_position',False),('logit_position',0.),('target_token_id',True),('target_token_id',1.)):
            row=dict(original,targets=[dict(original['targets'][0],**{key:value})])
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_worker_row(row,spec,4)
        for value in (True,1.):
            with self.subTest(full=value),self.assertRaises(ValueError):validate_worker_row(dict(original,full_vocab_target_positions=[value]),spec,4)

    def test_frozen_worker_sequence_and_attention_gates(self):
        spec={'max_sequence_tokens':640,'max_target_positions_per_sample':32,'max_full_vocab_positions_per_sample':8,'observe_attention':False}
        row={'input_ids':[0]*641,'targets':[{'token_position':1,'logit_position':0,'target_token_id':0}],'full_vocab_target_positions':[1]}
        with self.assertRaisesRegex(ValueError,'sequence cap'):validate_worker_row(row,spec)
        row['input_ids']=[0]*3
        with self.assertRaisesRegex(ValueError,'must not observe'):validate_worker_row(row,dict(spec,observe_attention=True))

    def test_worker_full_logits_metadata_accepted_by_generic_raw_adapter(self):
        from bench05c2.test_source import synthetic_raw,reseal
        from inference_kit.analysis.token_probe_source import collect_source
        with tempfile.TemporaryDirectory() as temp:
            raw=synthetic_raw(Path(temp)/'raw')
            for file in raw.rglob('observations.jsonl'):
                rows=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line]
                for row in rows:
                    details=dict(row['details']);details.pop('normalization_dtype')
                    row['details']=observation_details(details)
                    self.assertEqual(row['details']['normalization_dtype'],'float32')
                file.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
            reseal(raw)
            result=collect_source({'source_id':'worker-metadata','adapter':'token_probe_raw_v1','path':str(raw)},lambda directory,metadata:metadata)
            self.assertEqual(result['source']['coverage']['full_vocab_measured'],12)
            self.assertEqual(result['source']['coverage']['same_pass_consistency_verified'],12)

    def test_normalization_is_frozen_raw_f32_nat(self):
        self.assertEqual(NORMALIZATION,{'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'})
        self.assertEqual(CONSISTENCY_POLICY['absolute_tolerance_nat'],2e-5)


if __name__=='__main__':unittest.main()
