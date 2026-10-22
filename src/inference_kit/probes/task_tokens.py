"""Explicit local-only tokenizer preparation for B1; never called by CPU plan/replay."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import sys
from ..core.io import read,write,sha,digest,require
from ..tasks.tech_record import TASK,VERSION,render_prompt,build_probe_sample,select_probes,verify_gold
from .prepare_tokens import joint_encoding
from .contracts import Sample,attention_coordinates


POLICY='chat_rendered_joint_field_spans_v1'
SELECTION='field_value_spans_then_uniform16_v1'


def field_targets(ids,boundary,offsets,prompt_length,value_spans,special_ids=(),maximum=16):
    valid=[t for t in range(boundary,len(ids)) if ids[t] not in special_ids]
    require(valid,'No continuation targets')
    by_field={k:[t for t in valid if offsets[t][0]<prompt_length+end and offsets[t][1]>prompt_length+start]
              for k,(start,end) in value_spans.items()}
    require(all(by_field.values()),'Value span has no token coverage')
    selected={valid[0]};priority=[]
    for i in range(max(map(len,by_field.values()))):
        for values in by_field.values():
            if i<len(values):priority.append(values[i])
    for t in priority:
        if len(selected)<maximum:selected.add(t)
    remaining=[t for t in valid if t not in selected];slots=min(maximum-len(selected),len(remaining))
    for i in range(slots):selected.add(remaining[round(i*(len(remaining)-1)/max(1,slots-1))])
    targets=[{'token_position':t,'logit_position':t-1,'target_token_id':ids[t]} for t in sorted(selected)]
    return targets,{k:{'overlapping_positions':v,'selected_positions':[t for t in v if t in selected]} for k,v in by_field.items()}


def prepare_row(sample,task_sample,content,tokenizer,cfg,tokenizer_id,space,policy=POLICY,max_prompt=384,max_joint=512,
                *,task_id=TASK,task_version=VERSION,reference_source='task_author_gold_json',max_reference=128):
    """Shared offset/field-span collector input; callers supply versioned bounds."""
    prompt=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    reference=task_sample['reference_text'];encoded,ids,offsets,boundary=joint_encoding(tokenizer,prompt,reference,max_joint)
    prompt_ids=tokenizer.encode(prompt,add_special_tokens=False);ref_ids=tokenizer.encode(reference,add_special_tokens=False)
    require(len(prompt_ids)<=max_prompt and len(ref_ids)<=max_reference,'Explicit task prompt/reference bounds; revise before freeze')
    targets,coverage=field_targets(ids,boundary,offsets,len(prompt),task_sample['value_spans'],set(tokenizer.all_special_ids))
    ref_positions=[i for i in range(boundary,len(ids)) if ids[i] not in tokenizer.all_special_ids]
    coordinates=attention_coordinates(cfg['num_hidden_layers'],cfg['num_attention_heads'],cfg['num_key_value_heads'],boundary,[{'token_position':ref_positions[-1]}])
    contract=Sample(task_id,task_version,sample['sample_id'],sample['source_id'],prompt,digest(prompt),reference,reference_source,
            {'policy':policy,'selection_policy':SELECTION,'chat_template_applied':1,'tokenizer_identity':tokenizer_id,'token_id_space':space,
             'bos_added_by_encoder':False,'eos_added_by_encoder':False,'padding_added':False},
            'Teacher-forced author gold for diagnostic positions, separate from free generation scoring').validate()
    positions=[t['token_position'] for t in targets];full=sorted(set(positions[round(i*(len(positions)-1)/max(1,min(4,len(positions))-1))] for i in range(min(4,len(positions)))))
    row=contract|{'language':sample['language'],'split':sample['split'],'sample_identity':sample['sample_identity'],'task_prompt_hash':digest(content),
      'document_hash':sample['document_hash'],'full_text':prompt+reference,'full_text_hash':digest(prompt+reference),'input_ids':ids,
      'generation_input_ids':prompt_ids,'prompt_separate_ids':prompt_ids,'reference_separate_ids':ref_ids,'joint_equals_separate_concat':ids==prompt_ids+ref_ids,
      'prompt_token_boundary':boundary,'boundary_crossing_token':offsets[boundary][0]<len(prompt),
      'offsets_codepoints':[list(v) for v in offsets],'offsets_utf8_bytes':[[len((prompt+reference)[:a].encode('utf-8')),len((prompt+reference)[:b].encode('utf-8'))] for a,b in offsets],
      'token_strings':tokenizer.convert_ids_to_tokens(ids),'attention_mask':[1]*len(ids),'loss_mask':[int(i in ref_positions) for i in range(len(ids))],
      'special_token_mask':encoded['special_tokens_mask'],'special_token_ids':list(tokenizer.all_special_ids),'reference_token_count':len(ref_positions),
      'targets':targets,'value_spans':task_sample['value_spans'],'field_token_coverage':coverage,'selection_policy':SELECTION,
      'full_vocab_target_positions':full,'attention':coordinates,'sequence_identity':digest(ids),'reference_hash':task_sample['reference_hash']}
    return row


def prepare(data_file,tokenizer_path,config_path,output,runtime_output):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False)
    data=read(Path(data_file));require(data['task_id']==TASK and data['task_version']==VERSION,'Task mismatch')
    require(digest(data['rows'])==data['rows_identity'],'Authored data changed')
    cfg=read(Path(config_path));vocab=tokenizer.get_vocab();space=digest(vocab)
    identities={p.name:sha(p) for p in Path(tokenizer_path).iterdir() if p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt')}
    tokenizer_id=digest(identities);chosen=select_probes(data['rows']);rows=[];probes=[]
    for sample in data['rows']:
        verify_gold(sample);task_sample=build_probe_sample(sample);content=render_prompt(sample)
        row=prepare_row(sample,task_sample,content,tokenizer,cfg,tokenizer_id,space)
        rows.append(row)
        if sample['sample_id'] in chosen:probes.append(row)
    probes.sort(key=lambda r:chosen.index(r['sample_id']))
    prepared={'schema_version':1,'kind':'prepared_task_samples_v1','task_id':TASK,'task_version':VERSION,'preparation_id':POLICY,
      'authored_sha256':sha(Path(data_file)),'task_rows_identity':data['rows_identity'],'model_config':cfg,'model_config_sha256':sha(Path(config_path)),
      'tokenizer_files':identities,'tokenizer_identity':tokenizer_id,'token_id_space':space,'vocabulary_size':len(vocab),'generation_rows':rows,
      'rows':probes,'rows_identity':digest(probes),'generation_rows_identity':digest(rows),'probe_selection':{'seed':909261,'sample_ids':chosen,'rule':'basic en, correction ko, missing en, similar-ID ko; choose within type before responses'},
      'model_loaded':False,'gpu_used':False}
    write(Path(output),prepared)
    packages={name:{'version':metadata.version(name),'package_root':str(metadata.distribution(name).locate_file(''))} for name in ('torch','transformers','compressed-tensors','safetensors','accelerate','numpy')}
    import torch
    require(not torch.cuda.is_initialized(),'Tokenizer preparation unexpectedly initialized CUDA')
    write(Path(runtime_output),{'python':sys.executable,'packages':packages,'model_loads':0,'gpu_used':False,'cuda_initialized':False,'preparation_id':POLICY})
    return prepared


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True,type=Path);parser.add_argument('--tokenizer',required=True,type=Path);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--runtime-output',required=True,type=Path);a=parser.parse_args()
    result=prepare(a.data,a.tokenizer,a.config,a.output,a.runtime_output)
    print(__import__('json').dumps({'samples':len(result['generation_rows']),'probe_samples':len(result['rows']),'max_prompt':max(len(r['generation_input_ids']) for r in result['generation_rows']),'max_sequence':max(len(r['input_ids']) for r in result['generation_rows']),'gpu_used':False},ensure_ascii=True))
