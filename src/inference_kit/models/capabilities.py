"""Evidence-specific support for the initial Smol MHA/SDPA diagnostic adapter."""
from ..core.io import require,digest

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
