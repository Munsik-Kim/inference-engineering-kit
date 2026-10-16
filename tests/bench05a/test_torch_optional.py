"""Tiny CPU oracle only. GQA coverage here is not live GQA support."""
import contextlib
import copy
from pathlib import Path
import tempfile
import unittest
from inference_kit.probes.attention import selected_torch_row,AttentionObserver
from inference_kit.probes.weights import load_model
from inference_kit.core.io import write

try:
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig,LlamaForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import compressed_tensors
except ImportError:torch=None


@unittest.skipIf(torch is None,'Existing Torch/Transformers/compressed-tensors runtime required for tiny CPU oracles')
class TorchOracleTests(unittest.TestCase):
    def test_selected_matches_toy_full_eager_mha_and_gqa(self):
        for heads,kvheads in ((2,2),(4,2)):
            torch.manual_seed(3);q=torch.randn(1,heads,4,2);k=torch.randn(1,kvheads,4,2);v=torch.randn(1,kvheads,4,2)
            repeated=k.repeat_interleave(heads//kvheads,dim=1);values=v.repeat_interleave(heads//kvheads,dim=1)
            mask=torch.ones(4,4,dtype=torch.bool).tril()[None,None];mask[:,:,:,0]=False
            full=torch.softmax((q@repeated.transpose(-1,-2)).masked_fill(~mask,-torch.inf),dim=-1);full_out=full@values
            class Module:is_causal=True
            for h in range(heads):
                row,av,allowed,kv=selected_torch_row(Module(),q,k,v,mask,h,2,1.)
                self.assertTrue(torch.allclose(row,full[0,h,2]));self.assertTrue(torch.allclose(av,full_out[0,h,2]));self.assertEqual(kv,h//(heads//kvheads));self.assertFalse(bool(row[3]))
    def test_all_masked_and_future_visible_refused(self):
        q=k=v=torch.ones(1,1,3,2)
        class Module:is_causal=True
        self.assertIsNone(selected_torch_row(Module(),q,k,v,torch.zeros(1,1,3,3,dtype=torch.bool),0,1,1.)[0])
        with self.assertRaises(ValueError):selected_torch_row(Module(),q,k,v,None,0,1,1.,causal=False)
    def test_interface_and_projection_hook_restored_after_exception(self):
        class Attention:
            o_proj=torch.nn.Linear(2,2)
        previous=lambda *a,**k:None;mapping={'sdpa':previous};module=Attention();selection={'layer':0,'heads':[0],'queries':[0]}
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with AttentionObserver(mapping,module,selection,lambda *a:None) as observer:raise RuntimeError('intentional')
        self.assertIs(mapping['sdpa'],previous);self.assertTrue(observer.restored);self.assertFalse(module.o_proj._forward_hooks)
    def test_native_model_observer_returns_identical_logits(self):
        torch.manual_seed(5);cfg=LlamaConfig(vocab_size=16,hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)
        cfg._attn_implementation='sdpa';model=LlamaForCausalLM(cfg).eval();ids=torch.tensor([[1,3,4,5]])
        with torch.no_grad():
            before=model(input_ids=ids,use_cache=False).logits;seen=[]
            with AttentionObserver(ALL_ATTENTION_FUNCTIONS,model.model.layers[1].self_attn,{'layer':1,'heads':[0,1],'queries':[1,2]},lambda *a:seen.append(a[0])) as observer:
                after=model(input_ids=ids,use_cache=False).logits
        self.assertTrue(torch.equal(before,after));self.assertTrue(observer.restored);self.assertEqual(seen.count('attention_row'),4);self.assertEqual(seen.count('attention_output'),4)
        self.assertFalse(torch.cuda.is_initialized())
    def fixture(self,root,missing=False):
        torch.manual_seed(4);cfg=LlamaConfig(vocab_size=32,hidden_size=128,intermediate_size=256,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True)
        model=LlamaForCausalLM(cfg).to(dtype=torch.bfloat16);base=root/'bf16';rtn=root/'rtn';base.mkdir();rtn.mkdir();config=cfg.to_dict();write(base/'config.json',config)
        state={k:v.detach().clone().contiguous() for k,v in model.state_dict().items()};save_file(state,str(base/'model.safetensors'))
        group={'targets':['Linear'],'input_activations':None,'output_activations':None,'weights':{'num_bits':4,'type':'int','symmetric':True,'strategy':'group','group_size':128,'actorder':None}}
        write(rtn/'config.json',config|{'quantization_config':{'quant_method':'compressed-tensors','format':'pack-quantized','config_groups':{'group_0':group}}})
        modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n!='lm_head'};packed={}
        for name,module in modules.items():
            shape=list(module.weight.shape);packed[name+'.weight_packed']=torch.zeros((shape[0],shape[1]//8),dtype=torch.int32);packed[name+'.weight_scale']=torch.ones((shape[0],shape[1]//128),dtype=torch.bfloat16)*.01;packed[name+'.weight_shape']=torch.tensor(shape,dtype=torch.int64)
        for key,value in state.items():
            if key.removesuffix('.weight') not in modules:packed[key]=value
        if missing:del packed['model.layers.0.self_attn.q_proj.weight_scale']
        save_file(packed,str(rtn/'model.safetensors'))
        return base,rtn
    def test_exact_decode_strict_state_native_tie_and_coverage(self):
        with tempfile.TemporaryDirectory() as t:
            base,rtn=self.fixture(Path(t));arm={'arm_id':'rtn','artifact':{'path':str(rtn),'artifact_identity':'toy-packed'},'execution_mode':'DECODED_WEIGHT_DIAGNOSTIC'}
            model,receipt=load_model(arm,base,base,device='cpu')
            self.assertEqual(receipt['quantized_modules'],14);self.assertTrue(receipt['strict_loading']);self.assertTrue(all(x['decoded_differs_from_original_bf16'] for x in receipt['representative_checks']))
            self.assertEqual(model.lm_head.weight.data_ptr(),model.model.embed_tokens.weight.data_ptr());self.assertFalse(torch.cuda.is_initialized())
    def test_decode_rejects_incomplete_bundle(self):
        with tempfile.TemporaryDirectory() as t:
            base,rtn=self.fixture(Path(t),True);arm={'arm_id':'rtn','artifact':{'path':str(rtn),'artifact_identity':'toy'},'execution_mode':'DECODED_WEIGHT_DIAGNOSTIC'}
            with self.assertRaises(ValueError):load_model(arm,base,base,device='cpu')
