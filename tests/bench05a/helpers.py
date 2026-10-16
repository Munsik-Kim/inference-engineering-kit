import copy
import json
from pathlib import Path
from inference_kit.core.io import digest,write
from inference_kit.probes.contracts import Sample,target_positions,attention_coordinates,Execution,Observation
from inference_kit.probes.blobs import save
from inference_kit.metrics.token import log_softmax


def prepared():
    rows=[];cfg={'model_type':'llama','num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':2,'hidden_size':128,'intermediate_size':256,'vocab_size':4,'head_dim':64,'tie_word_embeddings':True,'attention_dropout':0.,'attention_bias':False,'mlp_bias':False}
    for i in range(4):
        prompt,reference=('ab','cd') if i<2 else ('가나','다라');full=prompt+reference;ids=[0,1,2,3];targets=target_positions(ids,2)
        row=Sample('diagnostic_continuation_v1','1','toy-'+str(i),'synthetic-'+str(i),prompt,digest(prompt),reference,'author_provided_diagnostic_text',
          {'policy':'synthetic_contract_token_ids_not_real_tokenization','tokenizer_identity':'toy-tokenizer','token_id_space':'toy-space'}).validate()
        rows.append(row|{'language':'en' if i<2 else 'ko','full_text':full,'full_text_hash':digest(full),'input_ids':ids,'prompt_separate_ids':[0,1],'reference_separate_ids':[2,3],
          'joint_equals_separate_concat':True,'prompt_token_boundary':2,'boundary_crossing_token':False,'offsets_codepoints':[[j,j+1] for j in range(4)],
          'offsets_utf8_bytes':[[len(full[:j].encode()),len(full[:j+1].encode())] for j in range(4)],'token_strings':list(full),'attention_mask':[1]*4,'loss_mask':[0,0,1,1],
          'special_token_mask':[0]*4,'special_token_ids':[],'reference_token_count':2,'targets':targets,'full_vocab_target_positions':[2,3],
          'attention':attention_coordinates(2,2,2,2,targets),'sequence_identity':digest(ids)})
    return {'schema_version':1,'kind':'prepared_diagnostic_samples','model_config':cfg,'model_config_sha256':'synthetic','rows':rows,'rows_identity':digest(rows),
            'tokenizer_identity':'toy-tokenizer','token_id_space':'toy-space','vocabulary_size':4,'tokenizer_files':{},'model_loaded':False,'gpu_used':False}


def synthetic_run(root):
    root=Path(root);root.mkdir()
    p=prepared();arms=[{'arm_id':a,'execution_mode':'REIMPLEMENTED_OR_SYNTHETIC','artifact':{'artifact_identity':a+'-toy','recipe_identity':'recipe-'+a}} for a in ('bf16','rtn')]
    plan={'task':'BENCH05-A','pair':{'revision':'toy-revision','model_id':'synthetic-only'},'arms':arms,'samples':p,'runtime_identity':'toy-runtime',
      'suite':{'observer':{'off_samples':2,'off_repeats':2,'max_target_logprob_abs_delta':1e-5,'max_selected_logit_abs_delta':1e-5,'raw_logit_to_logprob_abs_tolerance':2e-5}},
      'counts':{'main_reference_positions_per_arm':8,'main_attention_rows_per_arm':16}}
    plan['plan_identity']=digest(plan);write(root/'PLAN_FROZEN.json',plan)
    jobs={}
    for arm in arms:
        d=root/'attempts'/('main-'+arm['arm_id']+'-attempt1');d.mkdir(parents=True)
        execution=Execution('toy-run',d.name,arm['arm_id'],'synthetic-only','toy-revision',arm['artifact']['artifact_identity'],arm['artifact']['recipe_identity'],
           'toy-tokenizer','toy-runtime','REIMPLEMENTED_OR_SYNTHETIC','float32','toy','synthetic_contract','MEASURED','observations.jsonl').validate()
        execution['phase']='main';execution['execution_id']=digest(execution);write(d/'execution.json',execution);observations=[]
        def emit(row,kind,pass_id,coordinates,values,details):
            c={'sample_id':row['sample_id'],'pass_id':pass_id,'token_id_space':'toy-space','tokenizer_identity':'toy-tokenizer','sequence_identity':row['sequence_identity'],**coordinates}
            n=c['token_position'] if 'token_position' in c else c['query_position']+1;c['prefix_hash']=digest({'token_ids':row['input_ids'][:n],'token_id_space':'toy-space'})
            identifier=digest({'arm':arm['arm_id'],'kind':kind,'coordinates':c})[:28];blob=save(d,identifier,values,[len(values)])
            observations.append(Observation(identifier,kind,execution['execution_id'],row['sample_id'],'synthetic_contract','REIMPLEMENTED_OR_SYNTHETIC','MEASURED',c,blob,details).validate())
        logits=[0.,1.,2.,3.];lp=log_softmax(logits)
        for i,row in enumerate(p['rows']):
            for tag in (['OFF1','OFF2','ON'] if i<2 else ['ON']):
                for target in row['targets']:
                    emit(row,'target_logprob',tag,target,[lp[target['target_token_id']]],{'raw_argmax_token_id':3,'normalization_dtype':'float32'})
                    emit(row,'full_logits',tag,target,logits,{'full_vocabulary':True,'vocabulary_size':4})
                if tag=='ON':
                    for h in (0,1):
                        for q in row['attention']['queries']:
                            allowed=[j<=q for j in range(4)];coords={'layer':1,'head':h,'kv_head':h,'query_position':q,'key_positions':list(range(4)),'allowed_keys':allowed}
                            emit(row,'attention_row',tag,coords,[1/(q+1) if j<=q else 0. for j in range(4)],{'probability_source':'toy'})
                            for kind in ('attention_output','reconstructed_AV'):emit(row,kind,tag,coords,[2.,3.],{'output_source':'toy'})
                    for q in row['attention']['queries']:emit(row,'output_projection',tag,{'layer':1,'query_position':q},[2.,3.],{'output_source':'toy'})
        (d/'observations.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in observations));write(d/'result.json',{'status':'MEASURED'})
        jobs['main-'+arm['arm_id']]={'status':'COMPLETE','directory':str(d.relative_to(root))}
    write(root/'progress.json',{'jobs':jobs,'bridge_status':'NOT_RUN'})
    from inference_kit.suites.bench05a_run import raw_manifest
    raw_manifest(root);return root
