"""Tiny CPU tensor contract, optional prepared runtime; never a model/GPU forward."""
import contextlib
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from inference_kit.core.io import write, sha, read
from inference_kit.adapters.tensors import compose, BUNDLE

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
except ImportError:
    torch=None

H='model.layers.7.mlp.down_proj'
R='model.layers.8.mlp.down_proj'
S='model.layers.9.mlp.down_proj'


@unittest.skipIf(torch is None,'Optional tensor runtime not installed; base CPU replay is independent')
class PackedTests(unittest.TestCase):
    def fixture(self,root,broken=None):
        group={'weights':{'num_bits':4,'type':'int','symmetric':True,'strategy':'group','group_size':128,'actorder':None},
               'targets':['Linear'],'input_activations':None,'output_activations':None}
        config={'model_type':'llama','quantization_config':{'quant_method':'compressed-tensors','format':'pack-quantized','config_groups':{'group_0':group},'ignore':['lm_head']}}
        for arm in ('RTN','CLIP'):
            d=root/arm;d.mkdir();cfg=copy.deepcopy(config)
            if arm=='CLIP' and broken=='scheme':cfg['quantization_config']['config_groups']['group_0']['weights']['group_size']=64
            write(d/'config.json',cfg);write(d/'tokenizer.json',{'tiny':'synthetic'});write(d/'receipt.json',{'model_id':'synthetic','model_revision':'1'*40})
            values={'model.norm.weight':torch.ones(2,dtype=torch.bfloat16)}
            for module in (H,R,S):
                shift=int(arm=='CLIP' and module!=S) # S: ratio=1.00, no added clipping.
                values[module+'.weight_packed']=torch.full((2,16),shift,dtype=torch.int32)
                values[module+'.weight_scale']=torch.ones((2,1),dtype=torch.bfloat16)*(1+shift)
                values[module+'.weight_shape']=torch.tensor([2,128],dtype=torch.int64)
            if arm=='CLIP':
                if broken=='missing':del values[H+'.weight_scale']
                if broken=='shape':values[H+'.weight_shape']=torch.tensor([3,128],dtype=torch.int64)
                if broken=='native':values['model.norm.weight']*=2
                if broken=='scale':values[H+'.weight_scale']*=0
            save_file(values,str(d/'model.safetensors'))
        return root/'RTN',root/'CLIP'
    def test_truth_table_bundle_provenance_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b=self.fixture(root);before={p:sha(p) for folder in (a,b) for p in folder.iterdir()}
            for arm,modules in [('R00',[]),('R10',[H]),('R01',[R,S]),('R11',[H,R,S])]:
                result=compose(a,b,modules,root/arm,a/'receipt.json',b/'receipt.json')
                with contextlib.ExitStack() as stack:
                    left=stack.enter_context(safe_open(a/'model.safetensors',framework='pt',device='cpu'))
                    right=stack.enter_context(safe_open(b/'model.safetensors',framework='pt',device='cpu'))
                    out=stack.enter_context(safe_open(root/arm/'model.safetensors',framework='pt',device='cpu'))
                    for key in out.keys():
                        source=right if any(key==m+'.'+s for m in modules for s in BUNDLE) else left
                        self.assertTrue(torch.equal(source.get_tensor(key),out.get_tensor(key)))
                self.assertFalse(result['writable_hardlinks']);self.assertEqual(result['gpu_loader_verification'],'NOT_RUN')
            self.assertEqual(before,{p:sha(p) for p in before});self.assertFalse(torch.cuda.is_initialized())
    def rejection(self,broken):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a,b=self.fixture(root,broken)
            with self.assertRaises(ValueError):compose(a,b,[H],root/'out',a/'receipt.json',b/'receipt.json')
    def test_missing_scale_rejected(self):self.rejection('missing')
    def test_wrong_shape_rejected(self):self.rejection('shape')
    def test_scheme_mismatch_rejected(self):self.rejection('scheme')
    def test_native_tensor_difference_rejected(self):self.rejection('native')
    def test_invalid_scale_rejected(self):self.rejection('scale')
