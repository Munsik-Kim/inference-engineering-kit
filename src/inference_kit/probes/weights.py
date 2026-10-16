"""Read-only native or exact packed-decode Smol loader. No quantizer fitting."""
import contextlib
from pathlib import Path
from ..core.io import read,require,digest
from ..adapters.tensors import Reader,check_scheme,BUNDLE


def load_model(arm,reference_path,config_path,device='cuda'):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig,AutoModelForCausalLM
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
    source=Path(arm['artifact']['path']);cfg=AutoConfig.from_pretrained(str(config_path),local_files_only=True,trust_remote_code=False)
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    require(cfg.model_type=='llama' and cfg.num_attention_heads==cfg.num_key_value_heads,'Live support is the verified Smol MHA structure only')
    with torch.device('meta'):model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False,dtype=torch.bfloat16)
    expected=model.state_dict();module_names={name for name,m in model.named_modules() if isinstance(m,torch.nn.Linear) and name!='lm_head'}
    evidence={'execution_mode':arm['execution_mode'],'source_artifact':arm['artifact']['artifact_identity'],'effective_weight_dtype':'torch.bfloat16',
              'strict_loading':True,'missing_keys':[],'unexpected_keys':[],'quantized_modules':0,'quantization_refit':False,'disk_dense_artifact_created':False,
              'tie_word_embeddings':cfg.tie_word_embeddings,'representative_checks':[],'native_tensors_checked_against_bf16':0}
    state={}
    with contextlib.ExitStack() as stack:
        source_reader=Reader(source,stack,safe_open)
        reference=Reader(reference_path,stack,safe_open) if arm['arm_id']=='rtn' else source_reader
        if arm['arm_id']=='rtn':
            qgroup=check_scheme(read(source/'config.json'));scheme=QuantizationScheme(**qgroup)
            require(set(PackedQuantizationCompressor.compression_param_names(scheme))==set(BUNDLE),'Unreviewed packed metadata')
            actual={name.removesuffix('.weight_packed') for name in source_reader.handles if name.endswith('.weight_packed')}
            require(actual==module_names,'Quantized module coverage differs from complete model structure')
            representatives={f'model.layers.{i}.mlp.down_proj' for i in (0,cfg.num_hidden_layers//2,cfg.num_hidden_layers-1)}
            consumed=set()
            for module in sorted(actual):
                names={suffix:module+'.'+suffix for suffix in BUNDLE};require(all(k in source_reader.handles for k in names.values()),'Incomplete quantization bundle')
                bundle={suffix:source_reader.get(k) for suffix,k in names.items()};consumed.update(names.values())
                shape=bundle['weight_shape'].tolist();require(shape==list(expected[module+'.weight'].shape),'Packed original shape mismatch')
                require(bundle['weight_packed'].dtype==torch.int32 and bundle['weight_scale'].dtype==torch.bfloat16,'Packed dtype mismatch')
                weight=PackedQuantizationCompressor.decompress(bundle,scheme)['weight'];require(list(weight.shape)==shape and weight.dtype==torch.bfloat16 and bool(torch.isfinite(weight).all()),'Decoded weight invalid')
                state[module+'.weight']=weight
                if module in representatives:
                    words=bundle['weight_packed'][0,:16].tolist()
                    codes=[((int(word)>>(4*j))&15)-8 for word in words for j in range(8)]
                    scale=bundle['weight_scale'][0,0]
                    manual=(torch.tensor(codes,dtype=torch.bfloat16)*scale).to(torch.bfloat16)
                    require(torch.equal(manual,weight[0,:128]),'Representative integer code/scale/decode check failed')
                    original=reference.get(module+'.weight');changed=not torch.equal(original[0,:128],weight[0,:128])
                    require(changed,'Candidate representative group equals BF16; possible wrong source')
                    evidence['representative_checks'].append({'module':module,'row':0,'group':0,'integer_codes':codes,'scale':float(scale),
                         'decoded_values_sha256':digest(weight[0,:128].float().tolist()),'decoded_differs_from_original_bf16':changed,'library_decode_equals_manual_codes_scale':True})
                evidence['quantized_modules']+=1
            for key in set(source_reader.handles)-consumed:
                require(key in expected,'Unexpected packed/native key: '+key)
                value=source_reader.get(key);require(key in reference.handles and torch.equal(value,reference.get(key)),'Native tensor differs from original BF16: '+key)
                evidence['native_tensors_checked_against_bf16']+=1;state[key]=value
        else:
            require(not read(source/'config.json').get('quantization_config'),'BF16 reference is not native')
            state={key:source_reader.get(key) for key in source_reader.handles}
        if cfg.tie_word_embeddings:
            require('model.embed_tokens.weight' in state,'Native embedding missing')
            if 'lm_head.weight' in state:require(torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight']),'Tied native LM head differs')
            state['lm_head.weight']=state['model.embed_tokens.weight']
        require(state.keys()==expected.keys(),'Strict state coverage: missing='+str(sorted(expected.keys()-state.keys()))+' unexpected='+str(sorted(state.keys()-expected.keys())))
        for key,value in state.items():require(value.shape==expected[key].shape and value.dtype==torch.bfloat16,'Native/decoded state shape/dtype mismatch: '+key)
        incompatible=model.load_state_dict(state,strict=True,assign=True)
        require(not incompatible.missing_keys and not incompatible.unexpected_keys,'Strict loading incomplete')
        # Nonpersistent RoPE buffers are deliberately absent from safetensors/state_dict.
        # Initialize them with the installed native implementation and original config;
        # do not turn native FP32 frequencies into BF16 by a model-wide dtype cast.
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        meta_buffers={name for name,buffer in model.named_buffers() if buffer.is_meta}
        require(meta_buffers=={'model.rotary_emb.inv_freq','model.rotary_emb.original_inv_freq'},'Unexpected nonpersistent meta buffers')
        model.model.rotary_emb=LlamaRotaryEmbedding(cfg,device='cpu')
        evidence['nonpersistent_rope_buffers']='Native LlamaRotaryEmbedding(original config), FP32 frequencies; no learned tensor missing'
        model.tie_weights();model.eval();model.requires_grad_(False);model.to(device=device);model.tie_weights()
        require(not any(p.is_meta for p in model.parameters()) and not any(b.is_meta for b in model.buffers()),'Incomplete meta model after strict assignment')
    require(model.model.embed_tokens.weight.data_ptr()==model.lm_head.weight.data_ptr(),'Embedding/head tie was lost')
    evidence['state_keys']=len(state);evidence['effective_parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
    evidence['dtype_cast']='No source BF16 value changed; CPU library decode returns BF16; model/activation BF16, normalization FP32'
    return model,evidence
