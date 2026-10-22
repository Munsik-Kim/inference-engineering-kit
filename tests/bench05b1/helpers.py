"""Explicit synthetic token/HTTP/observation contracts, never model measurements."""
import copy
import json
from pathlib import Path
import sys
import time
import inference_kit
from inference_kit.core.io import write,read,digest,sha
from inference_kit.tasks.tech_record import make_dataset,render_prompt,canonical_reference,select_probes
from inference_kit.probes.task_tokens import field_targets,POLICY,SELECTION
from inference_kit.probes.contracts import Sample,attention_coordinates,Execution,Observation
from inference_kit.probes.blobs import save
from inference_kit.metrics.token import log_softmax
from inference_kit.suites.bench05a_plan import source_identity
from inference_kit.suites.bench05b1_run import seal
from inference_kit.adapters.protocol import Adapter
from inference_kit.adapters.http import sha as wire_hash


def prepared_fixture():
    data=make_dataset()
    for row in data['rows']:
        if row['language']=='ko':
            row['sample_id']='한글-'+row['sample_id'];row.pop('sample_identity');row['sample_identity']=digest(row)
    data['rows_identity']=digest(data['rows']);rows=[]
    cfg={'model_type':'llama','num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':2,'hidden_size':128,'vocab_size':256,'attention_dropout':0.,'head_dim':64,'torch_dtype':'bfloat16'}
    for source in data['rows']:
        ref=canonical_reference(source);prompt='SYNTHETIC CHAT\n'+render_prompt(source)+'\nASSISTANT\n';text=prompt+ref['text']
        offsets=[[i,min(i+4,len(text))] for i in range(0,len(text),4)];ids=[1+i%254 for i in range(len(offsets))];boundary=next(i for i,(a,b) in enumerate(offsets) if b>len(prompt))
        targets,coverage=field_targets(ids,boundary,offsets,len(prompt),ref['value_spans'])
        row=Sample('tech_record_extract','0.1',source['sample_id'],source['source_id'],prompt,digest(prompt),ref['text'],'task_author_gold_json',
          {'policy':POLICY,'selection_policy':SELECTION,'chat_template_applied':1,'tokenizer_identity':'synthetic-tokenizer','token_id_space':'synthetic-space'},'Synthetic contract only').validate()
        prompt_ids=[1+i%254 for i in range((len(prompt)+3)//4)]
        row.update(language=source['language'],split=source['split'],sample_identity=source['sample_identity'],document_hash=source['document_hash'],
          task_prompt_hash=digest(render_prompt(source)),input_ids=ids,generation_input_ids=prompt_ids,prompt_separate_ids=prompt_ids,reference_separate_ids=[1]*((len(ref['text'])+3)//4),
          full_text=text,full_text_hash=digest(text),reference_hash=ref['reference_hash'],prompt_token_boundary=boundary,offsets_codepoints=offsets,offsets_utf8_bytes=[[len(text[:a].encode('utf-8')),len(text[:b].encode('utf-8'))] for a,b in offsets],
          attention_mask=[1]*len(ids),loss_mask=[int(i>=boundary) for i in range(len(ids))],special_token_ids=[],targets=targets,value_spans=ref['value_spans'],field_token_coverage=coverage,
          selection_policy=SELECTION,full_vocab_target_positions=[targets[0]['token_position']],sequence_identity=digest(ids),
          attention=attention_coordinates(2,2,2,boundary,[{'token_position':len(ids)-1}]))
        rows.append(row)
    selected=select_probes(data['rows']);probes=[next(r for r in rows if r['sample_id']==sid) for sid in selected]
    return data,{'schema_version':1,'kind':'prepared_task_samples_v1','task_id':'tech_record_extract','task_version':'0.1','preparation_id':POLICY,
      'generation_rows':rows,'generation_rows_identity':digest(rows),'rows':probes,'rows_identity':digest(probes),'model_config':cfg,'tokenizer_files':{},
      'tokenizer_identity':'synthetic-tokenizer','token_id_space':'synthetic-space','vocabulary_size':256,'probe_selection':{'sample_ids':selected,'seed':909261}}


def local_fixture(root):
    data,prepared=prepared_fixture();kit=Path(__file__).parents[2];suite=read(kit/'configs/bench05b1/suite.json');suite['budget'].update(minimum_cpu_available_bytes=1,minimum_disk_free_bytes=1)
    pair=read(kit/'configs/bench05a/pair.json');write(root/'suite.json',suite);write(root/'pair.json',pair);write(root/'data.json',data)
    prepared['authored_sha256']=sha(root/'data.json');write(root/'prepared.json',prepared)
    write(root/'runtime.json',{'packages':{'transformers':{'version':'5.14.1'},'compressed-tensors':{'version':'0.18.0'}},'synthetic_contract':True})
    refs={}
    for name,arm in (('BF16_REF','bf16'),('R00','rtn')):
        p=root/arm;p.mkdir();cfg=copy.deepcopy(prepared['model_config'])
        if arm=='rtn':cfg['quantization_config']={'quant_method':'compressed-tensors','format':'pack-quantized','config_groups':{'g':{'input_activations':None,'output_activations':None,'targets':['Linear'],'weights':{'num_bits':4,'type':'int','strategy':'group','group_size':128,'symmetric':True,'actorder':None}}}}
        write(p/'config.json',cfg);(p/'tiny.safetensors').write_bytes(b'synthetic-only-no-model');write(p/'artifact.json',{'synthetic':True})
        refs[name]={'model_id':pair['model_id'],'model_revision':pair['revision'],'path':str(p),'tokenizer_path':str(p),
                    'artifact_identity':'synthetic-'+arm,'recipe_identity':'toy-'+arm,'config_sha256':sha(p/'config.json'),'recipe':{'algorithm':'RTN'}}
    write(root/'receipt.json',{'arms':refs});write(root/'prior.json',{'used_seconds':10,'running':None,'jobs':[{'wall_seconds':10}]})
    now=time.monotonic();write(root/'clock.json',{'start_utc':'synthetic','start_monotonic':now,'deadline_monotonic':now+3600,'no_new_gpu_after':now+3000})
    local={'authored_data':str(root/'data.json'),'prepared_samples':str(root/'prepared.json'),'artifact_receipt':str(root/'receipt.json'),'runtime_metadata':str(root/'runtime.json'),
           'runtime_python':sys.executable,'server_python':sys.executable,'package_site':str(Path(inference_kit.__file__).absolute().parent.parent),'prior_ledger':str(root/'prior.json'),'campaign_root':str(root),'task_clock':str(root/'clock.json'),'task_origin_used_seconds':10,'gpu':0,'gpu_uuid':'synthetic','port':18188}
    write(root/'local.json',local);return root/'suite.json',root/'pair.json',root/'local.json'


def generation(sample,prepared,arm,output=None):
    text=canonical_reference(sample)['text'] if output is None else output;prompt=prepared['generation_input_ids'];body=Adapter('vllm').body({'token_ids':prompt},'serving-lab',192,'natural_generation_v1')
    usage={'prompt_tokens':len(prompt),'completion_tokens':2,'total_tokens':len(prompt)+2}
    frames=[json.dumps({'choices':[{'text':text,'token_ids':[11,12],'prompt_token_ids':prompt,'index':0,'finish_reason':'stop'}]},ensure_ascii=False),json.dumps({'choices':[],'usage':usage}),'[DONE]']
    return {'task_id':'tech_record_extract','task_version':'0.1','sample_id':sample['sample_id'],'sample_identity':sample['sample_identity'],'request_id':arm['arm_id']+'-'+sample['sample_id'],
      'evidence_kind':'synthetic_contract','artifact_identity':arm['artifact']['artifact_identity'],'tokenizer_identity':'synthetic-tokenizer','prompt_hash':prepared['prompt_hash'],
      'reference_used_for_generation':False,'request_body':body,'request_identity':wire_hash(body),'raw_sse_frames':frames,'output_text':text,'usage':usage,'transport_complete':True,'termination':'NATURAL_STOP',
      'http_submit_monotonic':1.2,'queued_monotonic':1.,'first_content_monotonic':1.3,'completed_monotonic':1.5,'request_latency_http':1.5-1.2,'request_latency_task':.5,'ttft_http':1.3-1.2,'split':sample['split']}


def synthetic_run(root):
    from inference_kit.suites.bench05b1_plan import plan
    root=Path(root);root.mkdir();inputs=root.parent/(root.name+'-inputs');inputs.mkdir();args=local_fixture(inputs);plan(*args,root/'plan')
    p=read(root/'plan/PLAN_FROZEN.json');p['evidence_kind']='synthetic_contract';p.pop('plan_identity');p['plan_identity']=digest(p);write(root/'PLAN_FROZEN.json',p)
    jobs={}
    for arm in p['arms']:
        d=root/'attempts'/('serve-'+arm['arm_id']+'-attempt1');d.mkdir(parents=True);rows=[]
        for sample,prepared in zip(p['data']['rows'],p['samples']['generation_rows']):rows.append(generation(sample,prepared,arm))
        (d/'generations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        jobs['serve-'+arm['arm_id']]={'status':'COMPLETE','directory':str(d.relative_to(root))}
        d=root/'attempts'/('main-'+arm['arm_id']+'-attempt1');d.mkdir(parents=True)
        execution=Execution('toy-run',d.name,arm['arm_id'],p['pair']['model_id'],p['pair']['revision'],arm['artifact']['artifact_identity'],arm['artifact']['recipe_identity'],
          'synthetic-tokenizer',p['runtime_identity'],'REIMPLEMENTED_OR_SYNTHETIC','float32','toy','synthetic_contract','MEASURED','observations.jsonl').validate()
        # The synthetic execution must retain its explicit label in the frozen arm.
        execution['phase']='main';execution['execution_id']=digest(execution);write(d/'execution.json',execution);observations=[]
        def emit(row,kind,coordinates,values,details):
            c={'sample_id':row['sample_id'],'pass_id':'ON','token_id_space':'synthetic-space','tokenizer_identity':'synthetic-tokenizer','sequence_identity':row['sequence_identity'],**coordinates}
            prefix=c['token_position'] if 'token_position' in c else c['query_position']+1;c['prefix_hash']=digest({'token_ids':row['input_ids'][:prefix],'token_id_space':'synthetic-space'})
            identifier=digest({'arm':arm['arm_id'],'kind':kind,'coordinates':c})[:28]
            observations.append(Observation(identifier,kind,execution['execution_id'],row['sample_id'],'synthetic_contract','REIMPLEMENTED_OR_SYNTHETIC','MEASURED',c,save(d,identifier,values,[len(values)]),details).validate())
        logits=[-i*.05 for i in range(256)];lp=log_softmax(logits)
        for row in p['samples']['rows']:
            for target in row['targets']:
                emit(row,'target_logprob',target,[lp[target['target_token_id']]],{'raw_argmax_token_id':0,'normalization_dtype':'float32'})
                if target['token_position'] in row['full_vocab_target_positions']:emit(row,'full_logits',target,logits,{'full_vocabulary':True})
            for h in row['attention']['heads']:
                for q in row['attention']['queries']:
                    coords={'layer':1,'head':h,'kv_head':h,'query_position':q,'key_positions':list(range(len(row['input_ids']))),'allowed_keys':[i<=q for i in range(len(row['input_ids']))]}
                    emit(row,'attention_row',coords,[1/(q+1) if i<=q else 0. for i in range(len(row['input_ids']))],{'probability_source':'synthetic_full_key'})
                    for kind in ('attention_output','reconstructed_AV'):emit(row,kind,coords,[1.]+[0.]*63,{})
        (d/'observations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in observations),encoding='utf-8');write(d/'result.json',{'status':'MEASURED'})
        jobs['main-'+arm['arm_id']]={'status':'COMPLETE','directory':str(d.relative_to(root))}
    for arm in p['arms']:arm['execution_mode']='REIMPLEMENTED_OR_SYNTHETIC'
    # Recreate the test-only freeze after selecting explicit synthetic labels.
    p.pop('plan_identity');p['plan_identity']=digest(p)
    (root/'PLAN_FROZEN.json').write_text(json.dumps(p,ensure_ascii=False),encoding='utf-8')
    write(root/'progress.json',{'status':'SYNTHETIC_CONTRACT_ONLY','jobs':jobs});seal(root);return root
