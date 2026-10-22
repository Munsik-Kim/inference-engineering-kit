"""D1 worker dispatch and metadata contracts without a numerical runtime."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from inference_kit.core.io import digest
from inference_kit.probes.structure_worker import (PASSES,OBSERVER_POLICY,STRUCTURE_PATHS,
    forward_pass,coordinates_for,validate_selection,state_versions)


class Tensor:
    def __init__(self,shape,bytes_per_element=2):
        self.shape=shape;self.bytes_per_element=bytes_per_element;self.dtype='bfloat16' if bytes_per_element==2 else 'float32';self._version=0
    def numel(self):
        result=1
        for n in self.shape:result*=n
        return result
    def element_size(self):return self.bytes_per_element
    def float(self):return Tensor(self.shape,4)


class Hidden(Tensor):
    def __getitem__(self,key):
        batch,positions,columns=key
        assert batch==0 and columns==slice(None)
        return Tensor((len(positions),self.shape[-1]))


class Model:
    def __init__(self):self.calls=[];self.fail=False;self.parameter=Tensor((2,3));self.buffer=Tensor((2,))
    def model(self,**kwargs):
        assert kwargs['use_cache'] is False
        self.calls.append(('backbone',kwargs['input_ids'].shape))
        if self.fail:raise RuntimeError('fixture native failure')
        return types.SimpleNamespace(last_hidden_state=Hidden((1,kwargs['input_ids'].shape[1],4)))
    def lm_head(self,selected):self.calls.append(('head',selected.shape));return Tensor((selected.shape[0],8))
    def named_parameters(self):return [('weight',self.parameter)]
    def named_buffers(self):return [('frequency',self.buffer)]


class Functional:
    def log_softmax(self,value,dim):assert value.dtype=='float32' and dim==-1;return Tensor(value.shape,4)


class StructureWorkerTests(unittest.TestCase):
    def row(self):
        ids=[1,2,3,4,5,6]
        return {'sample_id':'문항-1','task_id':'task','task_version':'v1','source_id':'source','split':'REUSE',
                'reference_role':'author_gold','reference_source':'author','input_ids':ids,'sequence_identity':digest(ids),
                'prompt_token_boundary':3,'loss_mask':[0,0,0,1,1,1],'special_token_mask':[0]*6,
                'attention_mask':[1]*6,'targets':[{'token_position':3,'logit_position':2,'target_token_id':4}]}
    def selection(self):return {'layers':[0,2,3],'heads':[0,2],'queries':[2,4],
                                'query_roles':{'2':['last_prompt'],'4':['before_last_reference']}}
    def config(self):return {'num_hidden_layers':4,'num_attention_heads':4}
    def call(self,model,pass_id,factory=None):
        return forward_pass(model,Tensor((1,6)),Tensor((1,6)),self.row()['targets'],1024,pass_id,
                            observer_factory=factory,functional=Functional())

    def test_off_passes_do_not_construct_observer(self):
        def forbidden():raise AssertionError('OFF observer must not be constructed')
        model=Model()
        for pass_id in PASSES[:2]:
            selected,observer=self.call(model,pass_id,forbidden)
            self.assertIsNone(observer);self.assertEqual(selected['logits'].shape,(1,8))
        self.assertEqual(model.calls,[('backbone',(1,6)),('head',(1,4))]*2)

    def test_on_wraps_single_backbone_before_selected_native_head(self):
        events=[];model=Model()
        @contextmanager
        def observer():
            events.append('enter')
            try:yield
            finally:events.append('exit')
        selected,context=self.call(model,'ON',observer)
        self.assertEqual(events,['enter','exit']);self.assertIsNotNone(context)
        self.assertEqual(model.calls,[('backbone',(1,6)),('head',(1,4))])
        self.assertEqual(selected['chosen'].shape,(1,4))

    def test_on_failure_restores_context_without_extra_native_call(self):
        events=[];model=Model();model.fail=True
        @contextmanager
        def observer():
            events.append('enter')
            try:yield
            finally:events.append('exit')
        with self.assertRaisesRegex(RuntimeError,'native failure'):self.call(model,'ON',observer)
        self.assertEqual(events,['enter','exit']);self.assertEqual(model.calls,[('backbone',(1,6))])

    def test_unknown_or_on_without_factory_rejected_before_forward(self):
        for pass_id in ('OFF3','ON'):
            model=Model()
            with self.assertRaises(ValueError):self.call(model,pass_id)
            self.assertEqual(model.calls,[])

    def test_selection_uses_final_eligible_reference_not_selected_subset(self):
        row=self.row();selection=validate_selection(row,self.selection(),self.config())
        self.assertEqual(selection['queries'],[2,4])
        self.assertEqual(max(t['token_position'] for t in row['targets'])-1,2)

    def test_selection_index_duplicates_role_axes_and_mask_rejected(self):
        changes=[{'layers':[0,2,4]},{'layers':[0,0,2]},{'heads':[True]},
                 {'queries':[]},{'queries':[2,3],'query_roles':{'2':['last_prompt'],'3':['before_last_reference']}},
                 {'query_roles':{'2':['last_prompt'],'4':['last_prompt']}},
                 {'query_roles':{'2':['last_prompt'],'4':['before_last_reference','before_last_reference']}}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_selection(self.row(),dict(self.selection(),**change),self.config())
        row=self.row();row['attention_mask'][4]=0
        with self.assertRaisesRegex(ValueError,'masked'):validate_selection(row,self.selection(),self.config())

    def test_physical_query_can_have_two_roles_without_duplication(self):
        row=self.row();row['loss_mask']=[0,0,0,1,0,0]
        selected={'layers':[2],'heads':[0],'queries':[2],
                  'query_roles':{'2':['last_prompt','before_last_reference']}}
        self.assertEqual(validate_selection(row,selected,self.config())['queries'],[2])

    def test_token_prefix_shift_structure_prefix_and_task_identity(self):
        row=self.row();prepared={'token_id_space':'ids','tokenizer_identity':'tok'}
        token=coordinates_for(row,prepared,'OFF1',row['targets'][0])
        structure=coordinates_for(row,prepared,'ON',{'layer':2,'head':0,'query_position':2})
        self.assertEqual(token['prefix_hash'],structure['prefix_hash'])
        self.assertEqual(token['prefix_hash'],digest({'token_ids':[1,2,3],'token_id_space':'ids'}))
        self.assertEqual(token['pass_id'],'OFF1');self.assertEqual(structure['pass_id'],'ON')
        for key in ('task_id','task_version','source_id','split','reference_role','reference_source'):
            self.assertEqual(token[key],row[key])

    def test_projection_coordinate_remains_headless_and_passes_distinct(self):
        row=self.row();prepared={'token_id_space':'ids','tokenizer_identity':'tok'}
        values=[coordinates_for(row,prepared,p,{'layer':2,'query_position':2}) for p in PASSES]
        self.assertTrue(all('head' not in value for value in values));self.assertEqual(len({digest(value) for value in values}),3)

    def test_parameter_and_buffer_mutation_or_replacement_detected(self):
        model=Model();before=state_versions(model);self.assertEqual(state_versions(model),before)
        model.buffer._version+=1;self.assertNotEqual(state_versions(model),before)
        before=state_versions(model);model.parameter=Tensor((2,3));self.assertNotEqual(state_versions(model),before)

    def test_policies_frozen_and_native_measurement_paths_distinct(self):
        self.assertEqual(OBSERVER_POLICY,{'id':'d1_off_repeat_and_on_v1','absolute_tolerance':1e-5,'passes':['OFF1','OFF2','ON']})
        self.assertEqual(STRUCTURE_PATHS['attention_row'],'reconstructed_from_native_qk')
        self.assertNotEqual(STRUCTURE_PATHS['attention_output'],STRUCTURE_PATHS['reconstructed_AV'])
        self.assertIn('before_residual',STRUCTURE_PATHS['output_projection'])

    def test_worker_help_imports_no_numerical_or_model_sdk(self):
        import inference_kit
        parent=Path(inference_kit.__file__).parent.parent
        code="""import sys,runpy
sys.path.insert(0,sys.argv[1])
blocked={'torch','transformers','numpy','vllm','aiohttp','compressed_tensors'}
def guard(event,args):
    if event=='import' and args and str(args[0]).split('.')[0] in blocked:raise AssertionError('SDK import')
sys.addaudithook(guard)
sys.argv=['structure_worker','--help']
try:runpy.run_module('inference_kit.probes.structure_worker',run_name='__main__')
except SystemExit as exc:assert exc.code==0
assert not blocked.intersection(sys.modules)
"""
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(parent)],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--plan',result.stdout)


if __name__=='__main__':unittest.main()
