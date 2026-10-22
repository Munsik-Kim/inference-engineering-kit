"""Explicit CPU tokenizer preparation worker; never loads a model or uses CUDA.

Plan/replay do not import this worker. Execute only in an existing tokenizer runtime.
"""
import argparse
from pathlib import Path
import importlib.metadata as metadata
import sys
from ..core.io import read,write,sha,digest,require
from .contracts import Sample,target_positions,attention_coordinates


def joint_encoding(tokenizer,prompt,reference,max_tokens):
    """Shared offset primitive; each versioned wrapper supplies its own limit/policy."""
    require(tokenizer.is_fast,'Offset-aware fast tokenizer required; no boundary guess')
    encoded=tokenizer(prompt+reference,add_special_tokens=False,return_offsets_mapping=True,return_special_tokens_mask=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping']
    boundary=next(i for i,(a,b) in enumerate(offsets) if b>len(prompt))
    require(0<boundary<len(ids)<=max_tokens,'Authored text exceeds frozen sequence bounds; revise before any GPU run')
    return encoded,ids,offsets,boundary


def prepare(samples_file,tokenizer_path,config_path,output,runtime_output):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False)
    require(tokenizer.is_fast,'Offset-aware fast tokenizer required; no boundary guess')
    cfg=read(Path(config_path));vocab=tokenizer.get_vocab();token_space=digest(vocab)
    files={p.name:sha(p) for p in Path(tokenizer_path).iterdir() if p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt')}
    tokenizer_identity=digest(files);rows=[]
    for authored in read(Path(samples_file)):
        prompt,reference=authored['prompt'],authored['reference_text'];text=prompt+reference
        encoded,ids,offsets,boundary=joint_encoding(tokenizer,prompt,reference,128)
        targets=target_positions(ids,boundary,set(tokenizer.all_special_ids),16)
        all_reference=[i for i in range(boundary,len(ids)) if ids[i] not in tokenizer.all_special_ids]
        coordinates=attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],boundary,[{'token_position':all_reference[-1]}])
        sample=Sample('diagnostic_continuation_v1','1',authored['sample_id'],authored['source_id'],prompt,digest(prompt),reference,'author_provided_diagnostic_text',
          {'policy':'plain_completion_joint_no_special_tokens','bos_added':False,'eos_added':False,'padding_added':False,
           'boundary_rule':'first joint token with end>len(prompt); include whole boundary token even when it includes prompt suffix',
           'tokenizer_identity':tokenizer_identity,'token_id_space':token_space}).validate()
        prompt_ids=tokenizer.encode(prompt,add_special_tokens=False);ref_ids=tokenizer.encode(reference,add_special_tokens=False)
        require(all(0<=i<len(vocab) for i in ids),'Non-contiguous or out-of-range tokenizer IDs')
        full_positions=sorted(set(round(i*(len(targets)-1)/max(1,min(4,len(targets))-1)) for i in range(min(4,len(targets)))))
        rows.append(sample|{'language':authored['language'],'full_text':text,'full_text_hash':digest(text),'input_ids':ids,'prompt_separate_ids':prompt_ids,
          'reference_separate_ids':ref_ids,'joint_equals_separate_concat':ids==prompt_ids+ref_ids,'prompt_token_boundary':boundary,
          'boundary_crossing_token':offsets[boundary][0]<len(prompt),'offsets_codepoints':[list(o) for o in offsets],
          'offsets_utf8_bytes':[[len(text[:a].encode()),len(text[:b].encode())] for a,b in offsets],
          'token_strings':tokenizer.convert_ids_to_tokens(ids),'attention_mask':[1]*len(ids),'loss_mask':[int(i in all_reference) for i in range(len(ids))],
          'special_token_mask':encoded['special_tokens_mask'],'special_token_ids':list(tokenizer.all_special_ids),'reference_token_count':len(all_reference),
          'targets':targets,'full_vocab_target_positions':[targets[i]['token_position'] for i in full_positions],
          'attention':coordinates,'sequence_identity':digest(ids)})
    require(len(rows)==4 and len({r['sample_id'] for r in rows})==4,'Exactly four unique authored inputs required')
    result={'schema_version':1,'kind':'prepared_diagnostic_samples','task_id':'diagnostic_continuation_v1','task_version':'1',
       'authored_sha256':sha(Path(samples_file)),'model_config':cfg,'model_config_sha256':sha(Path(config_path)),
       'tokenizer_files':files,'tokenizer_identity':tokenizer_identity,'token_id_space':token_space,'vocabulary_size':len(vocab),
       'tokenization_policy':'joint_no_specials_with_explicit_offset_boundary','rows':rows,'rows_identity':digest(rows),'model_loaded':False,'gpu_used':False}
    write(Path(output),result)
    packages={n:{'version':metadata.version(n),'package_root':str(metadata.distribution(n).locate_file(''))} for n in ('torch','transformers','compressed-tensors','safetensors','accelerate','numpy')}
    tf=Path(packages['transformers']['package_root'])/'transformers';ct=Path(packages['compressed-tensors']['package_root'])/'compressed_tensors'
    sources=[tf/'models/llama/modeling_llama.py',tf/'integrations/sdpa_attention.py',tf/'quantizers/quantizer_compressed_tensors.py',ct/'compressors/pack_quantized/base.py',ct/'linear/compressed_linear.py']
    runtime={'python':sys.executable,'packages':packages,'source_files':{str(p):sha(p) for p in sources},'model_loads':0,'gpu_used':False,
             'transformers_model_signature':'LlamaAttention.forward -> ALL_ATTENTION_FUNCTIONS SDPA, post-RoPE Q/K/V, native [B,T,H,D] output',
             'compressed_direct_linear':'UNSUPPORTED_BY_INSTALLED_COMPRESSEDLINEAR','planned_rtn_mode':'DECODED_WEIGHT_DIAGNOSTIC'}
    write(Path(runtime_output),runtime);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',required=True,type=Path);p.add_argument('--tokenizer',required=True,type=Path);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--runtime-output',required=True,type=Path);a=p.parse_args()
    r=prepare(a.samples,a.tokenizer,a.config,a.output,a.runtime_output)
    print({'samples':len(r['rows']),'token_counts':[len(s['input_ids']) for s in r['rows']],'reference_tokens':[s['reference_token_count'] for s in r['rows']],'model_loads':0,'gpu_used':False})
