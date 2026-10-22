"""Local tokenizer-only verification; preserve harness trailing-whitespace boundary."""
import argparse
from dataclasses import fields
from pathlib import Path
import sys
from importlib import metadata
import hashlib
from ..core.io import read,write,sha,digest,require
from ..tasks.arc_challenge import select_pool,TASK,VERSION
from .prepare_tokens import joint_encoding
from .contracts import Sample,target_positions,attention_coordinates

POLICY='arc_existing_chat_echo_boundary_v1'

def effective_text(request):
    context=request['context'];trimmed=context.rstrip()
    return trimmed,context[len(trimmed):]+request['continuation']

def prepare(pool_file,tokenizer_path,config_path,data_output,prepared_output,runtime_output):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False)
    data=select_pool(read(Path(pool_file)));cfg=read(Path(config_path))
    files={p.name:sha(p) for p in Path(tokenizer_path).iterdir() if p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt')}
    tid=digest(files);space=digest(tok.get_vocab());checks=[]
    for row in data['rows']:
        for req in row['requests']:
            prompt,ref=effective_text(req)
            require(tok.encode(prompt+ref,add_special_tokens=False)==req['input_ids'],'Original joint token IDs differ')
            require(tok.encode(prompt,add_special_tokens=False)==req['context_ids'] and tok.encode(ref,add_special_tokens=False)==req['continuation_ids'],'Harness whitespace boundary differs')
            require(len(req['input_ids'])<2048,'Original context exceeds serving bound; no truncation')
            checks.append({'sample_id':row['sample_id'],'choice_index':req['choice_index'],'input_tokens':len(req['input_ids']),'continuation_tokens':len(req['continuation_ids']),
                'moved_trailing_context_whitespace':req['context'][len(prompt):],'input_identity':digest(req['input_ids'])})
    regression=[r for r in data['rows'] if r['usage_partition']=='REGRESSION']
    ranked=sorted(regression,key=lambda r:hashlib.sha256(('BENCH05-B3-probe-v1|'+r['sample_id']).encode('utf-8')).hexdigest())
    eligible=[r for r in ranked if len(r['requests'][r['gold']]['input_ids'])<=512]
    chosen=eligible[:4];rows=[]
    for sample in chosen:
        req=sample['requests'][sample['gold']];prompt,ref=effective_text(req)
        enc,ids,offsets,boundary=joint_encoding(tok,prompt,ref,512)
        require(ids==req['input_ids'] and boundary==len(req['context_ids']),'Native/probe continuation boundary mismatch')
        targets=target_positions(ids,boundary,set(tok.all_special_ids),16)
        valid=[i for i in range(boundary,len(ids)) if ids[i] not in tok.all_special_ids]
        att=attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],boundary,[{'token_position':valid[-1]}])
        contract=Sample(TASK,VERSION,sample['sample_id'],sample['document']['id'],prompt,digest(prompt),ref,'dataset_answer_key_choice',
            {'policy':POLICY,'tokenizer_identity':tid,'token_id_space':space,'chat_template_applied':0,'historical_template_already_applied':True,
             'bos_added_by_encoder':False,'eos_added_by_encoder':False,'padding_added':False},'External dataset answerKey; historical regression reuse').validate()
        rows.append(contract|{'split':sample['split'],'input_ids':ids,'prompt_token_boundary':boundary,'sequence_identity':digest(ids),'full_text_hash':digest(prompt+ref),
          'reference_hash':digest(ref),'targets':targets,'attention':att,'attention_mask':[1]*len(ids),'loss_mask':[int(i in valid) for i in range(len(ids))],
          'full_vocab_target_positions':[t['token_position'] for t in targets[:4]],'special_token_ids':list(tok.all_special_ids),
          'special_token_mask':enc['special_tokens_mask'],'offsets_codepoints':[list(v) for v in offsets],
          'offsets_utf8_bytes':[[len((prompt+ref)[:a].encode('utf-8')),len((prompt+ref)[:b].encode('utf-8'))] for a,b in offsets],
          'gold_choice_index':sample['gold'],'native_choice_identity':digest(req),'original_context':req['context'],'original_continuation':req['continuation'],
          'moved_trailing_context_whitespace':req['context'][len(prompt):]})
    prepared={'schema_version':1,'kind':'prepared_arc_regression_v1','preparation_id':POLICY,'task_id':TASK,'task_version':VERSION,
      'model_config':cfg,'model_config_sha256':sha(Path(config_path)),'tokenizer_files':files,'tokenizer_identity':tid,'token_id_space':space,
      'vocabulary_size':len(tok.get_vocab()),'rows':rows,'rows_identity':digest(rows),'native_boundary_checks':checks,
      'probe_selection':{'rule':'sha256(BENCH05-B3-probe-v1|sample_id) ascending among gold joint<=512, first4; before model outputs',
                         'sample_ids':[r['sample_id'] for r in chosen],'excluded_over_length':[r['sample_id'] for r in ranked if r not in eligible]},
      'pool_sha256':sha(Path(pool_file)),'data_identity':data['rows_identity'],'model_loaded':False,'gpu_used':False,'bounds':{'native_joint_exclusive':2048,'diagnostic_joint':512}}
    write(Path(data_output),data);write(Path(prepared_output),prepared)
    import torch
    require(not torch.cuda.is_initialized(),'Tokenizer initialized CUDA')
    packages={n:{'version':metadata.version(n),'package_root':str(metadata.distribution(n).locate_file(''))} for n in ('torch','transformers','compressed-tensors','safetensors','accelerate','numpy')}
    write(Path(runtime_output),{'python':sys.executable,'packages':packages,'model_loads':0,'gpu_used':False,'cuda_initialized':False,'preparation_id':POLICY})
    return {'documents':len(data['rows']),'choices':len(checks),'probe_documents':len(rows),'gpu_used':False}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('pool','tokenizer','config','data-output','prepared-output','runtime-output'):p.add_argument('--'+n,required=True,type=Path)
    a=p.parse_args();print(__import__('json').dumps(prepare(a.pool,a.tokenizer,a.config,a.data_output,a.prepared_output,a.runtime_output),ensure_ascii=True))
