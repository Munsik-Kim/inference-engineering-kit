"""Evidence-specific capabilities; static support never certifies a live run."""
from ..core.io import require,digest

QWEN_SERVING_PROFILE = 'qwen3_4b_gptq_serving_v1'
QWEN_SERVING_RUNTIME = 'cc88384a71f03de82c3f130c62cfa134b73aa301e8218f17d8c20cf25a54445b'
QWEN_ARTIFACTS = {
    'bf16':'80df9fe761e587e62157895ed06e4811608b93f12c3cd1c2ef4720dfec3711e1',
    'gptq_w4':'ae127f55c20fcb9fc8aed6f461a1187695ca74c4d3b4e43cfcdbdd83258bfa5f',
}
QWEN_TOKENIZER_IDENTITY = '3fb415f03fcb8a59a538bce25028c5b0f618e8a3e96721645c278c7d79674dd8'
QWEN_TOKEN_ID_SPACE = 'f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d'


def qwen_input_capability(metadata):
    """Match the pinned tokenizer, distinguishing its length from embeddings."""
    return isinstance(metadata,dict) and metadata.get('tokenizer_identity')==QWEN_TOKENIZER_IDENTITY and \
        metadata.get('token_id_space')==QWEN_TOKEN_ID_SPACE and \
        type(metadata.get('vocabulary_size')) is int and metadata['vocabulary_size']==151669 and \
        type(metadata.get('embedding_vocabulary_size')) is int and metadata['embedding_vocabulary_size']==151936


def qwen_serving_capability(pair, model_config, arm, runtime_identity=None):
    """Recognize the pinned Qwen pair's native serving contract, without imports.

    The candidate was produced with GPTQModifier but is stored in compressed-
    tensors pack-quantized format. Neither a model name nor a GPTQ label is
    sufficient; the artifact receipt, architecture, packing and recipe must
    agree. This function grants no diagnostic/attention/decoded capability.
    Current file presence, runtime source and GPU ownership remain live gates.
    """
    if not all(isinstance(v,dict) for v in (pair,model_config,arm)):
        return False
    if pair.get('model_id')!='Qwen/Qwen3-4B-Instruct-2507' or pair.get('revision')!='cdbee75f17c01a7cc42f958dc650907174af0554':
        return False
    config=arm.get('model_config',model_config)
    if not isinstance(config,dict):return False
    expected={'architectures':['Qwen3ForCausalLM'],'model_type':'qwen3','num_hidden_layers':36,
              'num_attention_heads':32,'num_key_value_heads':8,'head_dim':128,
              'hidden_size':2560,'vocab_size':151936}
    if any(config.get(k)!=v for k,v in expected.items()):return False
    if config.get('dtype',config.get('torch_dtype'))!='bfloat16':return False
    if runtime_identity is not None and runtime_identity!=QWEN_SERVING_RUNTIME:return False
    artifact=arm.get('artifact',{})
    if not isinstance(artifact,dict):return False
    method=arm.get('quantization_method',artifact.get('method'))
    if method not in QWEN_ARTIFACTS or artifact.get('artifact_identity')!=QWEN_ARTIFACTS[method]:return False
    if artifact.get('model_id',pair['model_id'])!=pair['model_id'] or artifact.get('model_revision',pair['revision'])!=pair['revision']:return False
    quant=arm.get('quantization_config',config.get('quantization_config'))
    if method=='bf16':
        return arm.get('source_format')=='native' and quant is None
    recipe=arm.get('recipe',artifact.get('recipe',{}))
    if not isinstance(recipe,dict) or not isinstance(quant,dict):return False
    if arm.get('source_format')!='compressed-tensors/pack-quantized':return False
    if recipe.get('method')!='GPTQ' or recipe.get('num_bits')!=4 or recipe.get('group_size')!=128 or recipe.get('symmetric') is not True or 'actorder' not in recipe or recipe['actorder'] is not None:return False
    if quant.get('quant_method')!='compressed-tensors' or quant.get('format')!='pack-quantized' or quant.get('quantization_status')!='compressed' or 'kv_cache_scheme' not in quant or quant['kv_cache_scheme'] is not None or quant.get('ignore')!=['lm_head']:return False
    groups=quant.get('config_groups')
    if not isinstance(groups,dict) or set(groups)!={'group_0'}:return False
    group=groups['group_0']
    if not isinstance(group,dict) or group.get('format')!='pack-quantized' or group.get('targets')!=['Linear'] or group.get('input_activations') is not None or group.get('output_activations') is not None:return False
    weights=group.get('weights')
    return isinstance(weights,dict) and all(k in weights and type(weights[k]) is type(v) and weights[k]==v for k,v in {
        'num_bits':4,'group_size':128,'symmetric':True,'dynamic':False,'strategy':'group',
        'type':'int','actorder':None,'block_structure':None,'scale_dtype':None,'zp_dtype':None}.items())

def inspect_capabilities(model_id,config,runtime,available):
    structure=(model_id=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and config.get('model_type')=='llama'
               and config.get('num_attention_heads')==config.get('num_key_value_heads') and config.get('attention_dropout')==0)
    versions=runtime.get('packages',{})
    supported=structure and versions.get('transformers',{}).get('version')=='5.14.1' and versions.get('compressed-tensors',{}).get('version')=='0.18.0'
    state='PLANNED_NOT_VERIFIED' if available and supported else 'UNAVAILABLE' if not available else 'UNSUPPORTED'
    return {'schema_version':1,'model_id':model_id,'runtime_identity':digest(runtime),'architecture':'MHA' if structure else 'UNSUPPORTED_LIVE_STRUCTURE',
      'features':{name:{'status':state,'evidence':'Installed source/config inspection only; actual worker must separately mark MEASURED'} for name in ('target_logprobs','selected_full_vocab','selected_attention_rows','attention_output','output_projection')},
      'observation_policy':{'bf16':'INSTRUMENTED_PATH','rtn':'DECODED_WEIGHT_DIAGNOSTIC','probabilities':'reconstructed_from_native_qk','attention_output':'native_returned SDPA before output projection'},
      'native_packed_linear':'UNSUPPORTED_BY_INSTALLED_COMPRESSEDLINEAR','live_gqa':'UNSUPPORTED; toy head-mapping test is not a live claim','model_loads':0}
