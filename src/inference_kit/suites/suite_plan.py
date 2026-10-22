"""E2 prepared-only SuiteSpec compiler. This module has no execution entrypoint.

An operation is a typed contract for a future executor, never a shell command.
Physical bindings, authorization and current preflight are separate from intent.
"""
import copy
import datetime
from pathlib import Path

from ..core.io import read, write, require, fields, digest, sha
from ..core.state import new_output
from ..analysis.tokens import source_identity
from ..tasks.registry import get_task
from .suite_inputs import load_sources
from .suite_graph import validate_graph, ADAPTERS, GPU_OPERATIONS

KIND = 'suite_spec_v1'
PROFILE = 'preserved_smol_v1'
QWEN_PROFILE = 'qwen3_4b_gptq_serving_v1'
NATIVE_TOKEN_METRIC = 'native_arc_tokens_c1_v1'
METRICS = {'task_default_v1', 'common_tokens_c1_v1', 'selected_structure_d1_v1', NATIVE_TOKEN_METRIC}
BUDGETS = {'max_inference_posts', 'max_output_tokens', 'max_numeric_bytes',
           'max_probe_buffer_bytes', 'max_gpu_jobs', 'gpu_wall_seconds', 'cleanup_reserve_seconds'}


def contract():
    return {'schema_version':1,'kind':KIND,'version':'1',
      'input_mode':'prepared_manifest; raw_text -> NEEDS_PREPARATION, never tokenized',
      'prepared_source_kinds':['prepared_suite_slice_v1','prepared_task_source_v1'],
      'token_selection_contracts':['preserved_c2_main_v1','prepared_reference_positions_v1'],
      'new_prepared_live_binding':'LIVE_BINDING_NOT_VALIDATED; historical E3 binder unchanged',
      'execution_authorization':'NONE; all operations NOT_EXECUTED or explicitly blocked',
      'operation_kinds':list(ADAPTERS),'metrics':sorted(METRICS),
      'static_capability_profile':PROFILE,'new_model_support':'NOT_VALIDATED',
      'static_capability_profiles':[PROFILE,QWEN_PROFILE],
      'native_likelihood_metric':{'id':NATIVE_TOKEN_METRIC,'inputs':'Complete saved ARC choice continuation target logprobs',
          'scopes':['all_choice','gold_choice','non_gold_choice'],'full_distribution':'NOT_COLLECTED',
          'diagnostic_worker':'NOT_REQUESTED; serving-only is distinct from selected-logit diagnostics'},
      'identity':'Intent and compiler/data/operation hashes exclude created time, output folder and physical locator',
      'cache':'Historical related results never replace new acquisition operations',
      'structure_provides':'Only explicitly declared same ON pass, exact input/target/full coordinates; controls excluded from main token denominator',
      'empty':'Empty required task/model/sample/job is INVALID_EMPTY_PLAN; optional OFF is NOT_REQUESTED',
      'scope':'Compile/validate/describe only, not scheduler/resume/runtime execution'}


def validate_spec(spec):
    fields(spec, ('schema_version','kind','suite_id','version','purpose','tasks','models',
                  'observations','metrics','comparisons','budget','permissions','seed','outputs'))
    require(type(spec['schema_version']) is int and spec['schema_version']==1 and spec['kind']==KIND, 'Unsupported SuiteSpec schema')
    require(all(isinstance(spec[k],str) and spec[k] for k in ('suite_id','version')), 'Missing Suite identity')
    require(spec['purpose'] in ('smoke','regression','diagnostic'), 'Unknown Suite purpose')
    require(type(spec['seed']) is int, 'Suite seed must be integer')
    require(isinstance(spec['tasks'],list) and spec['tasks'], 'INVALID_EMPTY_PLAN: empty tasks')
    require(isinstance(spec['models'],list) and spec['models'], 'INVALID_EMPTY_PLAN: empty models')
    for task in spec['tasks']:
        fields(task, ('task_id','task_version','source_id','sample_ids','input_mode','execution_kind','required'))
        require(all(isinstance(task[k],str) and task[k] for k in ('task_id','task_version','source_id')), 'Missing task/source identity')
        get_task(task['task_id'],task['task_version'])
        ids=task['sample_ids'];require(isinstance(ids,list) and ids and all(isinstance(i,str) and i for i in ids), 'INVALID_EMPTY_PLAN: empty samples')
        require(len(set(ids))==len(ids), 'Duplicate sample selection')
        require(task['input_mode'] in ('prepared_manifest','raw_text') and task['execution_kind'] in ('serve','diagnostic','both'), 'Unsupported task input/execution kind')
        require(type(task['required']) is bool, 'required must be boolean')
    require(len({(t['task_id'],t['source_id']) for t in spec['tasks']})==len(spec['tasks']), 'Duplicate task/source')
    require(any(t['required'] for t in spec['tasks']), 'INVALID_EMPTY_PLAN: empty required jobs')
    for model in spec['models']:
        fields(model, ('arm_id','role','artifact_ref','profile'))
        require(all(isinstance(model[k],str) and model[k] for k in model), 'Missing model reference')
        require(model['role'] in ('baseline','candidate'), 'Unknown model role')
    require(len({m['arm_id'] for m in spec['models']})==len(spec['models']) and
            len({m['role'] for m in spec['models']})==len(spec['models']), 'Duplicate model arm/role')
    require({m['role'] for m in spec['models']}=={'baseline','candidate'}, 'Explicit baseline/candidate pair required')
    fields(spec['observations'],('token','structure'))
    token=spec['observations']['token'];structure=spec['observations']['structure']
    fields(token,('enabled','required','full_vocabulary','selection'))
    fields(structure,('enabled','required','selection','provides_token_on'))
    for value in (token['enabled'],token['required'],token['full_vocabulary'],structure['enabled'],structure['required'],structure['provides_token_on']):
        require(type(value) is bool, 'Observation switches must be booleans')
    require(token['selection'] in ('preserved_c2_main_v1','prepared_reference_positions_v1') and
            structure['selection']=='preserved_d1_main_v1', 'Unknown observation selection contract')
    require(not token['required'] or token['enabled'], 'Required token observation cannot be OFF')
    require(not structure['required'] or structure['enabled'], 'Required structure observation cannot be OFF')
    require(not structure['enabled'] or token['enabled'], 'Structure integrity requires explicit token observations')
    require(not structure['provides_token_on'] or structure['enabled'], 'Structure provides declaration without structure')
    require(isinstance(spec['metrics'],list) and spec['metrics'] and len(set(spec['metrics']))==len(spec['metrics']) and set(spec['metrics'])<=METRICS, 'Unknown/empty/duplicate metric contract')
    expected=set()
    if any(t['execution_kind'] in ('serve','both') for t in spec['tasks']):expected.add('task_default_v1')
    if any(t['execution_kind'] in ('diagnostic','both') for t in spec['tasks']):
        if token['enabled']:expected.add('common_tokens_c1_v1')
        if structure['enabled']:expected.add('selected_structure_d1_v1')
    if NATIVE_TOKEN_METRIC in spec['metrics']:
        require(any(t['task_id']=='arc_challenge' and t['execution_kind'] in ('serve','both') for t in spec['tasks']),
                'Native likelihood metric requires complete ARC serving requests')
        expected.add(NATIVE_TOKEN_METRIC)
    require(set(spec['metrics'])==expected, 'Requested metric contracts lack matching operation/observation inputs')
    models={m['arm_id'] for m in spec['models']}
    require(isinstance(spec['comparisons'],list) and spec['comparisons'], 'Explicit comparison contract required')
    for comparison in spec['comparisons']:
        fields(comparison,('baseline','candidate','mode'))
        require(comparison['baseline'] in models and comparison['candidate'] in models and comparison['baseline']!=comparison['candidate'] and comparison['mode']=='within_recipe', 'Invalid comparison model axes')
        require(next(m for m in spec['models'] if m['arm_id']==comparison['baseline'])['role']=='baseline' and
                next(m for m in spec['models'] if m['arm_id']==comparison['candidate'])['role']=='candidate', 'Comparison roles differ')
    fields(spec['budget'],BUDGETS)
    require(all(type(v) is int and v>=0 for v in spec['budget'].values()), 'Budget must contain nonnegative integer bounds')
    b=spec['budget'];require(120<=b['cleanup_reserve_seconds']<b['gpu_wall_seconds']<=600 and
        0<b['max_probe_buffer_bytes']<=256*1024**2 and 0<b['max_numeric_bytes']<=256*1024**2 and b['max_gpu_jobs']>0, 'Unbounded probe/cleanup budget')
    require(spec['permissions']=={'gpu_execution':'NONE','network':'NONE','tokenizer':'NONE'}, 'E2 execution permission must be NONE')
    fields(spec['outputs'],('index','report','historical_source_ids'))
    require(type(spec['outputs']['index']) is bool and type(spec['outputs']['report']) is bool and
            (not spec['outputs']['report'] or spec['outputs']['index']), 'Report requires index output')
    require(isinstance(spec['outputs']['historical_source_ids'],list) and all(isinstance(v,str) for v in spec['outputs']['historical_source_ids']) and
            len(set(spec['outputs']['historical_source_ids']))==len(spec['outputs']['historical_source_ids']), 'Invalid historical source references')
    return spec


def _local(local):
    fields(local,('schema_version','artifact_check','artifact_roots','runtime_check','authorization'),('ledger',))
    require(type(local['schema_version']) is int and local['schema_version']==1, 'Unknown local binding schema')
    require(local['artifact_check'] in ('not_requested','stat_only') and isinstance(local['artifact_roots'],dict), 'Unsupported artifact presence policy')
    require(local['runtime_check']=='not_run' and local['authorization']=='NONE', 'E2 cannot execute runtime preflight/authorize GPU')
    if 'ledger' in local:
        require(isinstance(local['ledger'],str) and Path(local['ledger']).is_file(), 'Explicit existing ledger snapshot path required')
    return local


def _artifact_bindings(models, metadata, local):
    # Static receipt identities of the already exercised Smol recipe. A new
    # runtime name/hash does not acquire support merely by retaining its model.
    preserved_runtime = {
        'diagnostic':'33c7cd42d0c4fd2bec331491685445fb8abff9c9536d9a9f6c496f32d037c595',
        'serving':'cc88384a71f03de82c3f130c62cfa134b73aa301e8218f17d8c20cf25a54445b',
    }
    preserved_artifact = {
        'native':'0c668d5b8e51322d63898007d3d4a4ea488f21168208c3f3fc0d0f9643ac371b',
        'compressed-tensors/pack-quantized':'c8debb3188403ab9d332149969e6e0f9b7b689dc06e77e882fde07a9cb8da761',
    }
    arms={a['arm_id']:a for a in metadata['arms']};resolved=[];bindings={}
    for model in models:
        original=arms.get(model['artifact_ref'])
        cfg=metadata['model_config'];pair=metadata['pair']
        qwen=model['profile']==QWEN_PROFILE
        if qwen:
            from ..models.capabilities import qwen_serving_capability,qwen_input_capability,QWEN_SERVING_RUNTIME
            capable=qwen_input_capability(metadata) and qwen_serving_capability(pair,cfg,original,metadata.get('runtime_identities',{}).get('serving'))
            supported_runtime={'serving':QWEN_SERVING_RUNTIME}
        else:
            capable=(model['profile']==PROFILE and original is not None and
            pair['model_id']=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and
            pair['revision']=='31b70e2e869a7173562077fd711b654946d38674' and
            cfg.get('model_type')=='llama' and cfg.get('num_attention_heads')==cfg.get('num_key_value_heads') and
            original.get('source_format') in preserved_artifact and
            original.get('execution_mode')==('INSTRUMENTED_PATH' if original['source_format']=='native' else 'DECODED_WEIGHT_DIAGNOSTIC') and
            original.get('artifact',{}).get('artifact_identity')==preserved_artifact[original['source_format']])
            supported_runtime=preserved_runtime
        evidence=copy.deepcopy(original) if original else {'artifact':{'artifact_identity':'UNRESOLVED'}}
        presence='ARTIFACT_PRESENCE_NOT_CHECKED'
        if local['artifact_check']=='stat_only':
            path=local['artifact_roots'].get(model['artifact_ref'])
            require(isinstance(path,str) and Path(path).is_absolute(), 'Explicit artifact root required for stat-only policy')
            root=Path(path);presence='ARTIFACT_PATH_PRESENT_IDENTITY_NOT_REHASHED' if root.is_dir() and not root.is_symlink() and (root/'config.json').is_file() else 'ARTIFACT_PATH_MISSING'
            bindings[model['arm_id']]={'path':str(root),'check':presence,'weight_bytes_read':0}
        resolved.append({'arm_id':model['arm_id'],'role':model['role'],'artifact_ref':model['artifact_ref'],
            'profile':model['profile'],'artifact':evidence['artifact'],'diagnostic_path':'UNSUPPORTED_CAPABILITY' if qwen else evidence.get('execution_mode'),
            'serving_path':'NATIVE_ARTIFACT' if evidence.get('source_format')=='native' else 'PACKED_ARTIFACT_RUNTIME',
            'effective_diagnostic_dtype':None if qwen else 'bfloat16','static_capability':'SUPPORTED_PRESERVED_RECIPE_ONLY' if capable else 'UNSUPPORTED_CAPABILITY',
            'runtime_capability':{kind:'SUPPORTED_PRESERVED_RUNTIME_RECEIPT_ONLY'
                if capable and kind in supported_runtime and metadata.get('runtime_identities',{}).get(kind)==supported_runtime[kind]
                else 'UNSUPPORTED_CAPABILITY' for kind in ('serving','diagnostic')},
            'preserved_runtime_identities':dict(supported_runtime),
            'artifact_presence':presence})
    return resolved,bindings


def _probe_context(samples, metadata, full, *, newly_prepared=False):
    """Reuse validated prepared rows and the existing pure CPU selection resolver."""
    from ..probes.token_inputs import validate_probe_inputs, validate_spec as validate_token_spec
    rows=[copy.deepcopy(s['diagnostic']) for s in samples];token_spec=copy.deepcopy(metadata['token_spec'])
    if not full:
        token_spec['max_full_vocab_positions_per_sample']=0
        for row in rows:
            row['full_vocab_target_positions']=[]
            row['probe_input_identity']=digest({k:v for k,v in row.items() if k!='probe_input_identity'})
    validate_token_spec(token_spec)
    maximum_targets=max(len(r['targets']) for r in rows);maximum_sequence=max(len(r['input_ids']) for r in rows)
    buffer=maximum_targets*metadata['vocabulary_size']*12+maximum_sequence*metadata['model_config']['hidden_size']*4
    prepared={k:copy.deepcopy(metadata[k]) for k in ('model_config','tokenizer_identity','token_id_space','vocabulary_size')}
    prepared.update(schema_version=1,kind='selected_probe_inputs_v1',rows=rows,rows_identity=digest(rows),sources=[],
        spec=token_spec,resource_estimates={'extra_probe_gpu_buffer_upper_bytes':buffer},
        scope='E2 derived prepared coordinate contract; parent raw not acquired again')
    prepared['selection_identity']=digest(prepared)
    if newly_prepared:
        # This plan-only input contract deliberately does not relax C2's fixed
        # six-row validator or assert that E3's historical live binder supports it.
        from .suite_inputs import validate_diagnostic
        require(rows and len({(r['task_id'],r['task_version'],r['sample_id']) for r in rows}) == len(rows),
                'INVALID_EMPTY_PLAN: duplicate/empty new diagnostic rows')
        for row in rows:
            validate_diagnostic(row, metadata)
            require(row['reference_role'] in token_spec['reference_role_filter'] and
                    len(row['input_ids']) <= token_spec['max_sequence_tokens'] and
                    0 < len(row['targets']) <= token_spec['max_target_positions_per_sample'] and
                    len(row['full_vocab_target_positions']) <= token_spec['max_full_vocab_positions_per_sample'],
                    'Prepared reference selection/length exceeds declared ProbeSpec')
        prepared['kind'] = 'prepared_reference_probe_inputs_v1'
        prepared['scope'] = 'New prepared reference coordinates; CPU plan only, live binding not validated'
        prepared['live_binding'] = 'LIVE_BINDING_NOT_VALIDATED'
        prepared['selection_identity'] = digest({k:v for k,v in prepared.items() if k!='selection_identity'})
    else:
        validate_probe_inputs(prepared,verify_sources=False)
    return prepared


def _ref(name,schema,arm=None,task=None,samples=None,identity=None,external=False):
    value={'ref_id':name,'schema':schema,'arm_id':arm,'task_id':task,'sample_ids':list(samples or []),'identity':identity or digest(name)}
    if external:value['external']=True
    return value


def _node(name,kind,dependencies,inputs,outputs,samples,*,arm=None,task=None,required=True,state='NOT_EXECUTED',counts=None,details=None):
    return {'node_id':name,'operation_kind':kind,'adapter':ADAPTERS[kind],'adapter_version':'1',
      'dependencies':dependencies,'inputs':inputs,'outputs':outputs,'arm_id':arm,'task_id':task,'sample_ids':samples,
      'required':required,'authorization':'NEEDS_PREPARATION' if state=='NEEDS_PREPARATION' else 'NEEDS_GPU_AUTHORIZATION' if kind in GPU_OPERATIONS else 'CPU',
      'state':state,'counts':counts or {'documents':len(samples)},'failure_policy':'BLOCKED_DEPENDENCY',
      'output_contract':{'minimum_coverage':len(samples), 'verification':'Required input identities and nonempty coverage; missing results remain explicit',**(details or {})}}


def compile_suite(spec,sources,local):
    """Compile data only. No worker, runtime, network, GPU or tokenizer call."""
    validate_spec(spec);_local(local)
    metadata=sources['model_metadata'];models,bindings=_artifact_bindings(spec['models'],metadata,local)
    by_source={s['source_id']:s for s in sources['sources']};selected=[];nodes=[];task_context=[];warnings=[]
    for i,task in enumerate(spec['tasks']):
        sid=task['source_id'];source=by_source.get(sid);state='NOT_EXECUTED'
        if task['input_mode']=='raw_text':state='NEEDS_PREPARATION'
        elif source is None:state='BLOCKED_INPUT'
        else:
            require((source['task_id'],source['task_version'])==(task['task_id'],task['task_version']), 'Task/source contract differs')
            available={s['sample_id']:s for s in source['samples']}
            require(set(task['sample_ids'])<=available.keys(),'Unknown requested sample')
        rows=[] if state!='NOT_EXECUTED' else [{**copy.deepcopy(available[s]),'task_id':task['task_id'],'task_version':task['task_version'],'source_id':sid,'request_kind':source['request_kind']} for s in task['sample_ids']]
        selected.extend(rows);prefix='task'+str(i);sample_ids=task['sample_ids'];identity=digest({'source_id':sid,'samples':rows,'input_mode':task['input_mode']})
        prepared_ref=_ref(prefix+'.prepared','validated_prepared_task_v1',task=task['task_id'],samples=sample_ids,identity=identity)
        nodes.append(_node(prefix+'.validate','validate_prepared_inputs',[],[_ref(prefix+'.source',source['kind'] if source else 'prepared_suite_slice_v1',task=task['task_id'],samples=sample_ids,identity=identity,external=True)],
            [prepared_ref],sample_ids,task=task['task_id'],required=task['required'],state=state,details={'validation_scope':sources['validation_scope'],'input_mode':task['input_mode']}))
        task_context.append((prefix,task,source,rows,prepared_ref,state))
    require(len({(s['task_id'],s['sample_id']) for s in selected})==len(selected),'Duplicate samples across tasks')
    all_ids=[sid for t in spec['tasks'] for sid in t['sample_ids']];require(len(set(all_ids))==len(all_ids),'Suite sample IDs must be globally unique in current adapter')
    tokens=spec['observations']['token'];structure=spec['observations']['structure'];terminal=[];totals={'documents':len(all_ids),'arms':len(models),'inference_POSTs':0,'natural_generation_token_upper':0,'audit_token_upper':0,'output_token_upper':0,
      'GPU_jobs':0,'forward_calls':0,'main_target_observations':0,'main_full_observations':0,'all_pass_target_observations':0,'all_pass_full_observations':0,'attention_rows':0,'native_AV':0,'reconstructed_AV':0,'output_projection_vectors':0,'numeric_bytes':0,'probe_buffer_upper_bytes':0}
    task_scores={};task_raw={}
    for model in models:
        arm=model['arm_id'];artifact=_ref('artifact.'+arm,'artifact_receipt_v1',arm=arm,identity=model['artifact'],external=True)
        for prefix,task,source,rows,prepared_ref,state in task_context:
            if task['execution_kind'] not in ('serve','both'):continue
            requests=[];adapter=get_task(task['task_id'],task['task_version'])
            for sample in rows:
                if adapter.request_kind=='generate_until':
                    body=sample['generation']['wire_payload'];request=adapter.requests(sample['authored_sample']|{'generation_input_ids':body['prompt']},spec['suite_id']+'|'+arm,metadata['tokenizer_identity'])[0]
                    requests.append({'sample_id':sample['sample_id'],'request_id':request.request_id,'request_kind':request.request_kind,'wire_body':body,'request_identity':digest(body),'reference_used_for_generation':False})
                    totals['natural_generation_token_upper']+=body['max_tokens']
                else:
                    from ..adapters.likelihood import request_body
                    for request in adapter.requests(sample['authored_sample'],spec['suite_id']+'|'+arm,metadata['tokenizer_identity'],metadata['token_id_space'],source['model_metadata']['runtime_identities']['serving']):
                        body=request_body(request);requests.append({'sample_id':sample['sample_id'],'choice_index':request.payload['choice_index'],'request_id':request.request_id,'request_kind':request.request_kind,'wire_body':body,'request_identity':digest(body),'reference_role':'gold_choice' if request.payload['choice_index']==sample['authored_sample']['gold'] else 'non_gold_choice','normalization_characters':len(sample['authored_sample']['choices'][request.payload['choice_index']])});totals['audit_token_upper']+=1
            name=prefix+'.'+arm+'.serve';ref=_ref(name+'.raw','native_task_raw_v1',arm,task['task_id'],task['sample_ids'],digest(requests))
            capable=(model['static_capability']!='UNSUPPORTED_CAPABILITY' and source is not None and
                source['model_metadata'].get('runtime_identities',{}).get('serving')==model['preserved_runtime_identities']['serving'])
            serve_state=state if state!='NOT_EXECUTED' else 'NOT_EXECUTED' if capable else 'UNSUPPORTED_CAPABILITY'
            kind='serve_generate_until' if adapter.request_kind=='generate_until' else 'serve_choice_loglikelihood'
            nodes.append(_node(name,kind,[prefix+'.validate'],[prepared_ref,artifact],[ref],task['sample_ids'],arm=arm,task=task['task_id'],required=task['required'],state=serve_state,
                counts={'documents':len(task['sample_ids']),'inference_POSTs':len(requests)},details={'requests':requests,'collector_contract':adapter.request_kind,'effective_execution_path':model['serving_path'],
                'runtime_identity':source['model_metadata']['runtime_identities']['serving'] if source else None,'serving_policy':source['model_metadata']['serving_policy'] if source else None,
                'recipe_grouping':'one arm/task recipe; no minimum-server claim','minimum_coverage':len(requests) if requests else len(task['sample_ids']),'result_roles':'task response; diagnostic reference is never submitted as generation suffix'}))
            nodes[-1]['output_contract']['request_identity_scope']='Local planned ID; future executor must qualify with run/attempt/plan/node/arm/task/sample/choice'
            task_raw[(prefix,arm)]=(name,ref)
            scored=_ref(name+'.scores','task_metric_results_v1',arm,task['task_id'],task['sample_ids'],digest([ref['identity'],prepared_ref['identity'],adapter.scorer_version]))
            nodes.append(_node(name+'.score','score_task',[name],[ref,prepared_ref],[scored],task['sample_ids'],arm=arm,task=task['task_id'],required=task['required'],
                details={'scoring_mode':'single_arm_task','scorer_version':adapter.scorer_version,'metric_specs':adapter.metric_specs(),'incomplete':'No argmax on partial choices; failed/not-run never removed from coverage'}));terminal.append((name+'.score',scored))
            task_scores[(prefix,arm)]=(name+'.score',scored)
            totals['inference_POSTs']+=len(requests);totals['GPU_jobs']+=1
    for prefix,task,source,rows,prepared_ref,state in task_context:
        if task['execution_kind'] not in ('serve','both'):continue
        for j,pair in enumerate(spec['comparisons']):
            left=task_scores[(prefix,pair['baseline'])];right=task_scores[(prefix,pair['candidate'])]
            name=prefix+'.paired'+str(j);out=_ref(name+'.scores','paired_task_metric_results_v1',task=task['task_id'],samples=task['sample_ids'],identity=digest([left[1]['identity'],right[1]['identity'],pair]))
            nodes.append(_node(name,'score_task',[left[0],right[0]],[left[1],right[1]],[out],task['sample_ids'],task=task['task_id'],required=task['required'],
                details={'scoring_mode':'paired_saved_document_scores','comparison':pair,'metric_semantics':'Existing task-specific paired gains/losses; distinct accuracy/strict metric definitions retained',
                    'missing_pair':'UNKNOWN; no zero imputation','required_equal':['task/version','source/sample','prompt','tokenizer','scorer','runtime policy'],'treatment':'artifact/recipe'}));terminal.append((name,out))
    if NATIVE_TOKEN_METRIC in spec['metrics']:
        native_targets=0;gold_targets=0
        for prefix,task,source,rows,prepared_ref,state in task_context:
            if task['task_id']!='arc_challenge' or task['execution_kind'] not in ('serve','both'):continue
            per_arm=sum(len(request['input_ids'])-len(request['context_ids'])
                        for row in rows for request in row['authored_sample']['requests'])
            gold_per_arm=sum(len(row['authored_sample']['requests'][row['authored_sample']['gold']]['input_ids'])-
                             len(row['authored_sample']['requests'][row['authored_sample']['gold']]['context_ids']) for row in rows)
            require(not rows or per_arm>0,'INVALID_EMPTY_PLAN: native ARC continuation targets empty')
            native_targets+=per_arm*len(models);gold_targets+=gold_per_arm*len(models)
            for j,pair in enumerate(spec['comparisons']):
                left=task_raw[(prefix,pair['baseline'])];right=task_raw[(prefix,pair['candidate'])]
                name=prefix+'.native.paired'+str(j)+'.analyze_tokens'
                out=_ref(name+'.metrics','common_token_comparison_v1',task=task['task_id'],samples=task['sample_ids'],
                         identity=digest([left[1]['identity'],right[1]['identity'],NATIVE_TOKEN_METRIC]))
                nodes.append(_node(name,'analyze_tokens',[left[0],right[0]],[left[1],right[1]],[out],task['sample_ids'],
                    task=task['task_id'],required=task['required'],counts={'documents':len(rows),
                        'paired_native_continuation_targets':per_arm,'paired_gold_choice_targets':gold_per_arm,
                        'full_vocabulary_positions':0},details={'existing_metric_contract':NATIVE_TOKEN_METRIC,
                        'pass_filter':'NATIVE_CHOICE_LIKELIHOOD','comparison':pair,
                        'probability_stage':'raw_pre_sampling','reference_scopes':['all_choice','gold_choice','non_gold_choice'],
                        'same_input_coordinates_verified':True,'all_choices_required':True,
                        'audit_generation_in_continuation':False,'full_distribution':'NOT_COLLECTED',
                        'diagnostic_worker':'NOT_REQUESTED','controls_in_main_token_aggregate':False,
                        'required_equal':['task/version','source/sample','choice/order','tokenizer/vocabulary','prefix/target/logit position','runtime/probability contract'],
                        'cross_tokenizer_comparison':'INCOMPARABLE'}));terminal.append((name,out))
        totals['native_continuation_target_observations']=native_targets
        totals['native_gold_choice_target_observations']=gold_targets
    diagnostic_tasks=[ctx for ctx in task_context if ctx[1]['execution_kind'] in ('diagnostic','both')]
    diagnostic_rows=[r for ctx in diagnostic_tasks for r in ctx[3]];selection=None;probe=None;diagnostic_error=None
    if diagnostic_tasks and (tokens['enabled'] or structure['enabled']):
        try:
            require(diagnostic_rows,'NEEDS_PREPARATION: no diagnostic prepared rows')
            require(not any(model['profile']==QWEN_PROFILE for model in models),
                    'UNSUPPORTED_CAPABILITY: Qwen profile supports native serving only; no diagnostic/structure worker')
            fresh = [ctx[2].get('kind') == 'prepared_task_source_v1' for ctx in diagnostic_tasks if ctx[2]]
            newly_prepared = bool(fresh) and all(fresh)
            require(not any(fresh) or newly_prepared, 'UNSUPPORTED_CAPABILITY: mixed historical/new diagnostic preparation')
            require(tokens['selection'] == ('prepared_reference_positions_v1' if newly_prepared else 'preserved_c2_main_v1'),
                    'UNSUPPORTED_CAPABILITY: diagnostic selection/source contract mismatch')
            require(not (newly_prepared and structure['enabled']),
                    'UNSUPPORTED_CAPABILITY: new prepared structure/live binding not validated')
            probe=_probe_context(diagnostic_rows,metadata,tokens['full_vocabulary'],newly_prepared=newly_prepared)
            if structure['enabled']:
                require(tokens['full_vocabulary'],'UNSUPPORTED_CAPABILITY: structure profile requires its preserved full coordinates')
                from ..probes.structure_inputs import select
                selection=select(probe,metadata['structure_spec'])
        except ValueError as error:diagnostic_error=str(error);warnings.append(diagnostic_error)
        target=sum(len(r['diagnostic']['targets']) for r in diagnostic_rows);full=sum(len(r['diagnostic']['full_vocab_target_positions']) for r in diagnostic_rows) if tokens['full_vocabulary'] else 0
        diag_ids=[sid for ctx in diagnostic_tasks for sid in ctx[1]['sample_ids']];diag_required=any(ctx[1]['required'] for ctx in diagnostic_tasks)
        deps=[ctx[0]+'.validate' for ctx in diagnostic_tasks];refs=[ctx[4] for ctx in diagnostic_tasks]
        provided=structure['enabled'] and structure['provides_token_on'] and diagnostic_error is None
        diagnostic_raw={}
        for model in models:
            arm=model['arm_id'];artifact=_ref('artifact.'+arm,'artifact_receipt_v1',arm=arm,identity=model['artifact'],external=True)
            state='UNSUPPORTED_CAPABILITY' if (model['static_capability']=='UNSUPPORTED_CAPABILITY' or
                model['runtime_capability']['diagnostic']=='UNSUPPORTED_CAPABILITY' or diagnostic_error) else 'NOT_EXECUTED'
            options=[]
            if structure['enabled']:options.append(('structure','collect_structure_diagnostic',3,structure['required']))
            if tokens['enabled'] and not provided:options.append(('token','collect_token_diagnostic',1,tokens['required']))
            for label,kind,passes,required in options:
                name='diagnostic.'+arm+'.'+label;out=_ref(name+'.raw','structure_probe_raw_v1' if label=='structure' else 'token_probe_raw_v1',arm,None,diag_ids,digest([probe,label,model['artifact']]))
                count={'documents':len(diag_ids),'forward_calls':passes*len(diag_ids),'main_target_observations':target,'main_full_observations':full,'all_pass_target_observations':target*passes,'all_pass_full_observations':full*passes}
                numeric=(target*4+full*metadata['vocabulary_size']*4)*passes
                if label=='structure' and selection:
                    sc=selection['counts'];count.update(attention_rows=sc['attention_rows_per_arm'],native_AV=sc['native_AV_per_arm'],reconstructed_AV=sc['reconstructed_AV_per_arm'],output_projection_vectors=sc['output_projection_vectors_per_arm'])
                    numeric+=selection['resource_estimates']['structure_numeric_bytes_both_arms']//2
                count['numeric_bytes']=numeric
                details={'effective_execution_path':model['diagnostic_path'],'effective_dtype':'bfloat16','runtime_identity':metadata['runtime_identities'].get('diagnostic'),
                    'passes':['OFF1','OFF2','ON'] if label=='structure' else ['TOKEN_ONLY'],'observer_control_not_main_samples':True,
                    'resolved_probe_inputs':probe,'selection':selection if label=='structure' else None,'blocker':diagnostic_error,
                    'provides':{'token_pass':'ON','exact_targets':True,'exact_full_positions':True,'same_sequence':True} if label=='structure' and provided else {},
                    'numeric_bytes_scope':'Selected float32 scalar/full/structure numeric only; metadata overhead and model/KV/SDPA workspace separate',
                    'scalar_full_policy':metadata.get('token_spec',{}).get('consistency_policy'),
                    'observer_policy':metadata.get('structure_spec',{}).get('observer_consistency_policy') if label=='structure' else 'NOT_REQUESTED'}
                nodes.append(_node(name,kind,deps,refs+[artifact],[out],diag_ids,arm=arm,required=diag_required and required,state=state,counts=count,details=details))
                diagnostic_raw[(label,arm)]=(name,out,required)
                totals['GPU_jobs']+=1;totals['forward_calls']+=count['forward_calls'];totals['numeric_bytes']+=numeric
                for key in ('all_pass_target_observations','all_pass_full_observations','attention_rows','native_AV','reconstructed_AV','output_projection_vectors'):totals[key]+=count.get(key,0)
                if label=='token' or provided:
                    totals['main_target_observations']+=target;totals['main_full_observations']+=full
        families=[]
        if structure['enabled']:families.append(('structure','analyze_structure'))
        if tokens['enabled']:families.append(('structure' if provided else 'token','analyze_tokens'))
        for label,analyze in families:
            for j,pair in enumerate(spec['comparisons']):
                left=diagnostic_raw[(label,pair['baseline'])];right=diagnostic_raw[(label,pair['candidate'])]
                name='diagnostic.paired'+str(j)+'.'+analyze
                aref=_ref(name+'.metrics','structure_metric_results_v1' if analyze=='analyze_structure' else 'common_token_comparison_v1',None,None,diag_ids,digest([left[1]['identity'],right[1]['identity'],analyze]))
                nodes.append(_node(name,analyze,[left[0],right[0]],[left[1],right[1]],[aref],diag_ids,required=diag_required and (left[2] or right[2]),
                    details={'pass_filter':'ON' if label=='structure' else 'TOKEN_ONLY','controls_in_main_token_aggregate':False,
                        'comparison':pair,'comparison_mode':'explicit_instrumented_decoded_diagnostic_pair','same_input_coordinates_verified':True,
                        'arm_raw_assembly':'Validate both worker execution/observation/blob manifests under this frozen plan; CPU-only assembly into existing raw reader',
                        'existing_metric_contract':'selected_structure_d1_v1' if analyze=='analyze_structure' else 'common_tokens_c1_v1'}));terminal.append((name,aref))
        totals['probe_buffer_upper_bytes']=selection['resource_estimates']['extra_probe_gpu_buffer_upper_bytes'] if selection else probe['resource_estimates']['extra_probe_gpu_buffer_upper_bytes'] if probe else 0
    elif diagnostic_tasks:
        warnings.append('Diagnostic observations NOT_REQUESTED; no diagnostic acquisition work emitted')
    require(terminal,'INVALID_EMPTY_PLAN: no requested executable operation/results')
    historical=[]
    for source_id in spec['outputs']['historical_source_ids']:
        original=by_source.get(source_id)
        historical.append({'source_id':source_id,'status':'PROVENANCE_RELATED_ONLY' if original else 'UNRESOLVED_RELATED_ONLY',
            'parent_provenance':copy.deepcopy(original.get('ancestry',original.get('preparation'))) if original else None,
            'source_status_freshness':'NOT_CHECKED_IN_E2','cache_eligible':False,'replaces_acquisition':False})
    if any(v['status']=='UNRESOLVED_RELATED_ONLY' for v in historical):
        warnings.append('Unresolved historical IDs are labels only; no paired join/cache substitution')
    if spec['outputs']['index']:
        iref=_ref('index.results','e1_index_revision_v1',samples=all_ids,identity=digest([ref['identity'] for _,ref in terminal]))
        nodes.append(_node('index.add','index_add',[n for n,_ in terminal],[ref for _,ref in terminal],[iref],all_ids,details={'new_acquisition_count_in_E2':0,'historical_sources':historical,'cache_reuse':'RELATED_ONLY_NO_SKIPPING','output_path':'logical:index/new.sqlite'}))
        if spec['outputs']['report']:
            nodes.append(_node('index.report','report',['index.add'],[iref],[_ref('index.report','offline_html_v1',samples=all_ids)],all_ids,details={'output_path':'logical:report/index.html','historical_outcome_not_same_execution':True}))
    totals['output_token_upper']=totals['natural_generation_token_upper']+totals['audit_token_upper']
    limits={'inference_POSTs':'max_inference_posts','output_token_upper':'max_output_tokens','numeric_bytes':'max_numeric_bytes','probe_buffer_upper_bytes':'max_probe_buffer_bytes','GPU_jobs':'max_gpu_jobs'}
    for count,limit in limits.items():require(totals[count]<=spec['budget'][limit], 'BUDGET_EXCEEDED: '+count)
    graph=validate_graph(nodes)
    input_contract={'data_identity':sources['data_identity'],'scope':sources['validation_scope'],'source_rows':[{k:r[k] for k in ('source_id','sha256','scope')} for r in sources['receipts']],
        'requested_samples':[{'task_id':t['task_id'],'source_id':t['source_id'],'sample_ids':t['sample_ids'],'input_mode':t['input_mode']} for t in spec['tasks']]}
    semantic={'suite':spec,'inputs':input_contract,'models':[{k:v for k,v in m.items() if k!='artifact_presence'} for m in models],
        'graph':graph,'counts':totals,'compiler':{'version':'e2-suite-v1','source_digest':source_identity()},
        'comparison_contract':{'declarations':spec['comparisons'],'held_equal':['task/version','source/sample','request token IDs','tokenizer/vocabulary','runtime per execution path','prefix/reference roles','probe/pass coordinates','metric/version/aggregation/tolerance'],
            'treatment':['artifact/recipe/effective weights'],'native_decoded':'Separate acquisition paths; no deployment equivalence','cross_task':'INCOMPARABLE','historical':'RELATED_ONLY, not cache hit'},
        'historical_links':historical,
        'optional_features':{'structure':'REQUESTED' if structure['enabled'] else 'NOT_REQUESTED','full_vocabulary':'REQUESTED' if tokens['enabled'] and tokens['full_vocabulary'] else 'NOT_REQUESTED'},'warnings':warnings}
    if NATIVE_TOKEN_METRIC in spec['metrics']:
        semantic['optional_features'].update(native_arc_tokens='REQUESTED',
            token_diagnostic='REQUESTED' if tokens['enabled'] else 'NOT_REQUESTED',
            native_full_distribution='NOT_COLLECTED')
    if any(source.get('kind') == 'prepared_task_source_v1' for source in sources['sources']):
        semantic['input_execution_scope'] = {'preparation': 'PREPARED_CONTRACT_VALIDATED_ONLY',
            'live_binding': 'LIVE_BINDING_NOT_VALIDATED', 'new_inputs_executed': False,
            'historical_e3_binder': 'UNCHANGED_FIXED_RECIPE'}
    identity=digest(semantic)
    return {'schema_version':1,'kind':'frozen_suite_plan_v1','plan_identity':identity,'semantic':semantic,
        'status':graph['status'],'execution_count':0,'acquisition_count':0,'GPU_queries':0,'model_loads':0,'tokenizer_calls':0,'network_requests':0,'ledger_changes':0,
        'preflight':{'runtime':'CHECK_NOT_RUN','GPU_free_busy':'CHECK_NOT_RUN','ports':'CHECK_NOT_RUN','authorization':'NONE','execution_readiness':'REQUIRES_RUNTIME_PREFLIGHT_AND_NEW_AUTHORIZATION',
            'artifacts':models,'new_model_support':'NOT_VALIDATED',
            'ledger_snapshot':{'sha256':sha(Path(local['ledger'])),'bytes':Path(local['ledger']).stat().st_size,'authorization':'NONE','consumed_or_reserved':0} if 'ledger' in local else 'NOT_REQUESTED',
            'live_resource_floors':({'minimum_cpu_available_bytes':10*1024**3,'minimum_disk_free_bytes':4*1024**3}
                if any(model['profile']==QWEN_PROFILE for model in models) else
                {k:metadata['token_spec']['budget'][k] for k in ('minimum_cpu_available_bytes','minimum_disk_free_bytes')}),
            'unmeasured':['loading_time','throughput','peak_VRAM','model_memory','SDPA_workspace','current_GPU_ownership']},'bindings':{'artifacts':bindings},'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}


def plan(suite,sources,local,output):
    suite_path,sources_path,local_path=[Path(p).absolute() for p in (suite,sources,local)]
    out=new_output(output)
    try:
        record=compile_suite(read(suite_path),load_sources(sources_path),read(local_path))
        record['bindings'].update(suite_path=str(suite_path),sources_path=str(sources_path),local_path=str(local_path),
            suite_file_sha256=sha(suite_path),sources_file_sha256=sha(sources_path),local_file_sha256=sha(local_path))
        write(out/'FROZEN_PLAN.json',record);write(out/'SUITE_CONTRACT.json',contract());write(out/'PLAN_SUMMARY.json',summary(record))
        write(out/'VALIDATION_RECEIPT.json',{'status':record['status'],'plan_identity':record['plan_identity'],'input_scope':record['semantic']['inputs']['scope'],
            'runtime_preflight':'CHECK_NOT_RUN','execution_count':0,'acquisition_count':0,'artifact_presence':[m['artifact_presence'] for m in record['preflight']['artifacts']]})
        return record
    except (ValueError,KeyError,TypeError,OSError) as error:
        write(out/'VALIDATION_RECEIPT.json',{'status':'INVALID_EMPTY_PLAN' if 'INVALID_EMPTY_PLAN' in str(error) else 'INVALID_INPUT_OR_CONTRACT','error':str(error),'execution_count':0,'acquisition_count':0})
        raise


def check_frozen(record):
    require(record.get('kind')=='frozen_suite_plan_v1' and type(record.get('schema_version')) is int and record['schema_version']==1,'Wrong frozen suite schema')
    require(record['plan_identity']==digest(record['semantic']),'Frozen plan identity changed')
    require(record['semantic']['compiler']['source_digest']==source_identity(),'Compiler/source changed after freeze')
    require(all(type(record[k]) is int and record[k]==0 for k in ('execution_count','acquisition_count','GPU_queries','model_loads','tokenizer_calls','network_requests','ledger_changes')) and record['preflight']['authorization']=='NONE','E2 plan is not execution evidence')
    require(record['status']==record['semantic']['graph']['status'] and record['preflight']['runtime']=='CHECK_NOT_RUN' and record['preflight']['GPU_free_busy']=='CHECK_NOT_RUN','Frozen plan reporting/preflight state changed')
    validate_spec(record['semantic']['suite']);graph=validate_graph(record['semantic']['graph']['nodes'])
    require(digest(graph)==digest(record['semantic']['graph']),'Frozen graph derived state changed')
    return record


def validate(plan_file,sources,output,local=None):
    record=check_frozen(read(Path(plan_file)));out=new_output(output);loaded=load_sources(Path(sources))
    require(loaded['data_identity']==record['semantic']['inputs']['data_identity'],'Frozen prepared inputs changed')
    bound=read(Path(local)) if local is not None else {'schema_version':1,'artifact_check':'not_requested','artifact_roots':{},'runtime_check':'not_run','authorization':'NONE'}
    current=compile_suite(record['semantic']['suite'],loaded,bound)
    require(current['plan_identity']==record['plan_identity'],'Resolved operation/coordinate contract changed')
    receipt={'status':record['status'],'plan_identity':record['plan_identity'],'input_scope':loaded['validation_scope'],'content_revalidated':True,
      'physical_relocation':'CONTENTS_VERIFIED_NO_NEW_ACQUISITION' if loaded['manifest_sha256']!=record['bindings']['sources_file_sha256'] or str(Path(sources).absolute())!=record['bindings']['sources_path'] else 'UNCHANGED_BINDING',
      'artifact_presence':[m['artifact_presence'] for m in current['preflight']['artifacts']], 'runtime_preflight':'CHECK_NOT_RUN','authorization':'NONE','execution_count':0,'acquisition_count':0}
    write(out/'VALIDATION_RECEIPT.json',receipt);return receipt


def summary(record):
    result = {'schema_version':1,'suite_id':record['semantic']['suite']['suite_id'],'status':record['status'],'plan_identity':record['plan_identity'],
      'tasks':record['semantic']['inputs']['requested_samples'],'models':[{'arm_id':m['arm_id'],'role':m['role'],'artifact_identity':m['artifact']['artifact_identity'],'diagnostic_path':m['diagnostic_path'],'serving_path':m['serving_path']} for m in record['semantic']['models']],
      'counts':record['semantic']['counts'],'node_count':len(record['semantic']['graph']['nodes']),'dependencies':{n['node_id']:n['dependencies'] for n in record['semantic']['graph']['nodes']},
      'blocked_required_nodes':record['semantic']['graph']['blocked_required_node_ids'],'optional_features':record['semantic']['optional_features'],
      'historical_links':record['semantic']['historical_links'],
      'input_scope':record['semantic']['inputs']['scope'],'runtime_preflight':'CHECK_NOT_RUN','authorization':'NONE','execution_count':0,'acquisition_count':0,'warnings':record['semantic']['warnings']}
    if 'input_execution_scope' in record['semantic']:
        result['input_execution_scope'] = copy.deepcopy(record['semantic']['input_execution_scope'])
    return result


def describe(plan_file,output):
    record=check_frozen(read(Path(plan_file)));out=new_output(output);result=summary(record)
    write(out/'PLAN_DESCRIPTION.json',result)
    lines=['# Suite 실행계획 설명','', '이 자료는 계획이며 모델 실행·GPU availability·신규 품질 검증이 아니다.',
      '', '상태: '+record['status'], '권한: NONE. 모든 작업 미실행. runtime/GPU/port preflight는 CHECK_NOT_RUN.',
      '', '입력 범위: '+result['input_scope'], '', '|Operation|요청/관측 수량|의존성|상태|','|---|---|---|---|']
    for node in record['semantic']['graph']['nodes']:
        lines.append('|'+node['node_id']+' ('+node['operation_kind']+')|'+str(node['counts'])+'|'+', '.join(node['dependencies'])+'|'+node['state']+'|')
    lines+=['','생성 prompt-only wire와 teacher-forced gold joint는 별도다. ARC는 원래 모든 choice를 계획한다.',
      'OFF1/OFF2는 integrity control이며 본 sample/ON token 분모에 더하지 않는다. o_proj는 head 없는 layer/query 축이다.',
      '과거 결과는 historical relation이며 새 작업을 CACHED로 건너뛰지 않는다. 실제 executor/resume/raw-text preparation은 미구현이다.']
    if 'input_execution_scope' in result:
        lines[-1] = '과거 결과는 related-only이며 새 준비 자료를 historical acquisition으로 취급하지 않는다.'
        lines += ['새 자료는 명시적 CPU prepare를 거쳤다. 공통 계획 검증은 모델 실행을 의미하지 않는다.',
                  'LIVE_BINDING_NOT_VALIDATED: 기존 E3의 고정 6문항 실행 binder는 변경하지 않았다.']
    (out/'DESCRIPTION_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
