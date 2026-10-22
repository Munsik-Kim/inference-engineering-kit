"""CPU-only configuration and coordinate resolution over unchanged ProbeInputs."""
from ..core.io import require,digest
from .token_inputs import validate_probe_inputs
from .structure_observer import selected_temporary_upper_bytes

OBSERVER_POLICY={'id':'d1_off_repeat_and_on_v1','absolute_tolerance':1e-5,'passes':['OFF1','OFF2','ON']}


def validate_spec(spec):
    require(spec.get('schema_version')==1 and spec.get('kind')=='structure_probe_spec_v1','Unsupported StructureProbeSpec')
    require(spec.get('profile') in ('SMALL','MAIN') and spec.get('observe_attention') is True,'Explicit structure profile required')
    require(spec.get('max_sequence_tokens')==640,'D1 sequence contract is 640; no truncation')
    require(spec.get('observer_consistency_policy')==OBSERVER_POLICY,'Unapproved OFF/OFF/ON consistency policy')
    require(spec.get('numeric')=={'dtype':'float32','endianness':'little'},'Unsupported structure serialization')
    require(spec.get('backend')=='native_transformers_sdpa_smol_mha_v1','Unverified structure backend')
    roles=spec.get('query_roles');require(isinstance(roles,list) and roles and len(set(roles))==len(roles) and set(roles)<={'last_prompt','before_last_reference'},'Invalid/duplicate query role')
    b=spec['budget'];require(b['task_gpu_wall_seconds']<=600 and b['cleanup_reserve_seconds']>=120 and b['automatic_retries']==0 and 2<=b['max_gpu_processes']<=4,'Unbounded structure budget')
    require(b['minimum_cpu_available_bytes']>=10*1024**3 and b['minimum_disk_free_bytes']>=4*1024**3,'Live resource floors must not be lowered')
    require(0<b['probe_gpu_buffer_bytes']<=256*1024**2 and 0<b['raw_bytes_total']<=256*1024**2,'Structure buffer/raw limit')
    return spec


def indices(policy,count,axis):
    require(type(count) is int and count>0,'Invalid model '+axis+' count')
    policies={'middle':[count//2],'first_middle':[0,count//2],'first_middle_last':[0,count//2,count-1],'first':[0]}
    if isinstance(policy,str):
        require(policy in policies,'Unknown '+axis+' selection');result=sorted(set(policies[policy]))
    else:
        require(isinstance(policy,list) and policy and all(type(i) is int for i in policy) and len(set(policy))==len(policy),'Empty/duplicate/noninteger '+axis+' indices')
        result=sorted(policy)
    require(result and all(0<=i<count for i in result),'Out-of-range '+axis+' index')
    return result


def select(inputs,spec):
    """No model/tokenizer load and no target reselection; token rows stay exact."""
    validate_probe_inputs(inputs);spec=validate_spec(spec);cfg=inputs['model_config']
    require(cfg['model_type']=='llama' and cfg['num_attention_heads']==cfg['num_key_value_heads'],'Live D1 supports verified Smol MHA only; toy GQA is separate')
    require(cfg.get('rope_scaling') is None and not cfg.get('sliding_window') and not cfg.get('attention_logit_softcapping') and not cfg.get('qk_norm'),'UNSUPPORTED position/QK/softcap/window policy')
    require(cfg['hidden_size']%cfg['num_attention_heads']==0 and cfg.get('attention_dropout',0)==0,'Unverified head dimension/dropout')
    layers=indices(spec['layers'],cfg['num_hidden_layers'],'layer');heads=indices(spec['heads'],cfg['num_attention_heads'],'head')
    rows={};counts={'samples':len(inputs['rows']),'tasks':len({r['task_id'] for r in inputs['rows']}),
      'target_positions_per_arm':sum(len(r['targets']) for r in inputs['rows']),'full_vocab_positions_per_arm':sum(len(r['full_vocab_target_positions']) for r in inputs['rows']),
      'attention_rows_per_arm':0,'native_AV_per_arm':0,'reconstructed_AV_per_arm':0,'output_projection_vectors_per_arm':0,'role_aliases':0}
    numeric=0;maximum_temporary=0;dim=cfg['hidden_size']//cfg['num_attention_heads']
    for row in inputs['rows']:
        ids=row['input_ids'];boundary=row['prompt_token_boundary'];length=len(ids)
        require(length<=spec['max_sequence_tokens'] and row['attention_mask']==[1]*length,'Unsupported input length/padding; no truncation')
        valid=[t for t in range(1,length) if row['loss_mask'][t] and not row['special_token_mask'][t] and ids[t] not in row['special_token_ids']]
        require(valid,'No valid reference for structure query')
        aliases={}
        for role in spec['query_roles']:
            q=boundary-1 if role=='last_prompt' else valid[-1]-1
            require(0<=q<length and row['attention_mask'][q]==1 and not row['special_token_mask'][q],'Invalid '+role+' query')
            if role=='last_prompt':
                require(row['loss_mask'][q]==0,'last_prompt must be prompt-only')
                offsets=row.get('offsets_codepoints')
                if offsets is not None:require(offsets[q][1]<=len(row['prompt']),'last_prompt token crosses reference boundary')
            aliases.setdefault(str(q),[]).append(role)
        queries=sorted(int(q) for q in aliases)
        rows[row['sample_id']]={'queries':queries,'query_roles':aliases,'sequence_tokens':length,
          'sequence_identity':row['sequence_identity'],'query_prefix_hashes':{str(q):digest({'token_ids':ids[:q+1],'token_id_space':inputs['token_id_space']}) for q in queries},
          'module_paths':{str(layer):'model.layers.'+str(layer)+'.self_attn' for layer in layers},
          'selection_reason':'Frozen config rule + exact prepared boundary/last eligible reference; physical query aliases stored once'}
        n=len(layers)*len(heads)*len(queries);p=len(layers)*len(queries)
        for key in ('attention_rows_per_arm','native_AV_per_arm','reconstructed_AV_per_arm'):counts[key]+=n
        counts['output_projection_vectors_per_arm']+=p;counts['role_aliases']+=len(spec['query_roles'])-len(queries)
        numeric+=n*(length*4+2*dim*4)+p*cfg['hidden_size']*4
        # Native Q/K/V and SDPA workspace already exist; float K/V row-cast
        # temporaries are additional. Collector releases them for every row.
        maximum_temporary=max(maximum_temporary,selected_temporary_upper_bytes(length,dim))
    token_bytes=2*3*(counts['target_positions_per_arm']*4+counts['full_vocab_positions_per_arm']*inputs['vocabulary_size']*4)
    estimated=token_bytes+2*numeric
    # Conservative sum even though selected LM head follows observer teardown.
    buffer_bound=inputs['resource_estimates']['extra_probe_gpu_buffer_upper_bytes']+maximum_temporary
    require(buffer_bound<=spec['budget']['probe_gpu_buffer_bytes'],'Configured structure temporary buffer exceeds budget')
    require(estimated<=spec['budget']['raw_bytes_total'],'Configured structure/token raw exceeds budget')
    selection={'layers':layers,'heads':heads,'rows':rows,'counts':counts,'index_base':0,'head_mapping':'kv_head=floor(query_head/(num_query_heads/num_kv_heads))',
      'position_point':'native post-RoPE Q/K; no QK norm/softcap; full native key axis with actual mask',
      'output_projection_axis':'layer/query only; after o_proj, before residual; not duplicated across heads',
      'resource_estimates':{'structure_numeric_bytes_both_arms':2*numeric,'token_numeric_bytes_all_three_passes_both_arms':token_bytes,
        'raw_numeric_bytes_both_arms':estimated,'selected_structure_temporary_upper_bytes':maximum_temporary,
        'extra_probe_gpu_buffer_upper_bytes':buffer_bound,'native_QKV_and_SDPA_workspace':'Existing model tensors/workspace; excluded from additional selected buffer estimate',
        'host_copy_scope':'One selected numeric row/vector at a time; no GPU tensor collection retained',
        'raw_estimate_scope':'Numeric blobs only; JSON/log overhead measured separately'}}
    selection['selection_identity']=digest(selection)
    return selection
