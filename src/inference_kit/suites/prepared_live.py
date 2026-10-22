"""Versioned live binding for explicit prepared task manifests.

Only CPU validation/stat/hash work occurs here. Model work still belongs to the
existing owned serving and token lifecycles. Historical acquisition samples are
never used as authority for this input profile.
"""
from copy import deepcopy
import math
from pathlib import Path
import time

from ..core.io import digest, fields, read, require, sha
from ..probes.input_preparation import _load
from ..probes.token_inputs import validate_spec
from ..probes.token_worker import validate_worker_row
from ..tasks.registry import get_task
from .bench05a_plan import ledger_evidence, source_identity
from .suite_inputs import validate_diagnostic
from .suite_plan import check_frozen
from .task_environment import inspect_artifacts, resources

PROFILE = 'prepared_smol_live_v1'
SERVING_PROFILE = 'prepared_qwen_serve_v1'
PREPARED_PROFILES = (PROFILE, SERVING_PROFILE)
KINDS = {'serve_generate_until', 'serve_choice_loglikelihood'}
TOKEN_WORKER = {'module': 'inference_kit.probes.token_worker', 'token_only': True}


def contract(profile=PROFILE):
    require(profile in PREPARED_PROFILES, 'Unsupported prepared live contract profile')
    if profile==SERVING_PROFILE:
        return {'schema_version':1,'profile':SERVING_PROFILE,
            'input_authority':'prepared_input_bundle_v1 inventory and prepared_task_source_v1',
            'models':'Pinned Qwen3-4B-Instruct-2507 BF16 / compressed-tensors GPTQ W4 serving only',
            'operations':sorted(KINDS),'diagnostic':'NOT_REQUESTED','full_vocabulary':'NOT_COLLECTED',
            'native_tokens':'All ARC choice continuation scalars via existing C1',
            'counts':'Derived from prepared requests; no diagnostic collector',
            'serving_grouping':'One execution unit per typed task/arm serving node',
            'namespace_policy':'Qualified identity retained; duplicate bare sample ID explicitly unsupported',
            'structure':'NOT_IMPLEMENTED','preflight':'CPU binding is not runtime/GPU/loopback readiness',
            'budget_compatibility':'GPU wall600/cleanup120; conservative free-memory launch floor12288MiB is not a peak estimate',
            'historical_plan_required':False,'tokenizer_calls':0,'model_loads':0}
    return {'schema_version': 1, 'profile': PROFILE,
        'input_authority': 'prepared_input_bundle_v1 inventory and prepared_task_source_v1',
        'models': 'Existing pinned Smol BF16 / packed RTN serve / decoded RTN diagnostic only',
        'operations': sorted(KINDS | {'collect_token_diagnostic'}),
        'counts': 'Derived from selected manifest requests and exact target/full coordinate sets',
        'serving_grouping': 'One execution unit per typed task/arm serving node',
        'namespace_policy': 'Qualified source identity retained; duplicate bare sample ID explicitly unsupported by retained worker',
        'structure': 'NOT_IMPLEMENTED', 'preflight': 'CPU binding is not GPU or loopback readiness',
        'historical_plan_required': False,
        'budget_compatibility': 'Suite and retained ProbeSpec must both use approved GPU wall600/cleanup120 seconds',
        'tokenizer_calls': 0, 'model_loads': 0}


def validate_budget_compatibility(compiled, metadata):
    """The retained lifecycle accepts its approved wall/reserve policy exactly."""
    suite = compiled['semantic']['suite']['budget']
    if metadata.get('capability_profile')=='qwen3_4b_gptq_serving_v1':
        require(suite['gpu_wall_seconds']==600 and suite['cleanup_reserve_seconds']==120, 'UNSUPPORTED_BUDGET_POLICY: native serving wall/cleanup policy changed')
        return True
    probe = metadata['token_spec']['budget']
    require(suite['gpu_wall_seconds'] == probe['task_gpu_wall_seconds'] == 600 and
            suite['cleanup_reserve_seconds'] == probe['cleanup_reserve_seconds'] == 120,
            'UNSUPPORTED_BUDGET_POLICY: Suite GPU wall/cleanup reserve differs from retained ProbeSpec')
    return True


def sample_namespace(sample):
    raw = sample['raw_record']
    keys = ('task_id', 'task_version', 'dataset_id', 'split', 'sample_id')
    require(all(isinstance(raw.get(k), str) and raw[k] for k in keys), 'Missing qualified sample identity')
    return {k: raw[k] for k in keys}


def qualified_samples(samples):
    """Prevent retained sample-ID-only code from silently aliasing namespaces."""
    require(isinstance(samples, list) and samples, 'INVALID_EMPTY_PLAN: no selected prepared samples')
    qualified, seen = [], set()
    for sample in samples:
        identity = sample_namespace(sample); key = digest(identity)
        require(key not in seen, 'Duplicate qualified prepared sample namespace')
        seen.add(key)
        qualified.append({'qualified_sample_id': key, 'original_sample_id': sample['sample_id'], **identity,
                          'source_id': sample['source_id']})
    require(len({r['original_sample_id'] for r in qualified}) == len(qualified),
            'UNSUPPORTED_SAMPLE_ID_COLLISION: retained serving/worker requires unique sample_id across namespaces')
    return qualified


def selected_samples(compiled, loaded, *, execution_kind='both'):
    """Resolve the Suite's exact declared selection from validated new authority."""
    check_frozen(compiled)
    require(compiled['status'] == 'VALIDATED_NOT_EXECUTED', 'Required Suite operations blocked')
    require(loaded['data_identity'] == compiled['semantic']['inputs']['data_identity'],
            'Prepared sources differ from frozen Suite')
    require(all(s['kind'] == 'prepared_task_source_v1' and
                s['validation_scope'] == 'PREPARED_CONTRACT_VALIDATED_ONLY' for s in loaded['sources']),
            'Prepared live profile requires explicit new prepared source scope')
    by_source = {s['source_id']: s for s in loaded['sources']}; selected = []
    for declaration in compiled['semantic']['suite']['tasks']:
        require(declaration['input_mode'] == 'prepared_manifest' and declaration['execution_kind'] == execution_kind,
                'Prepared live profile task/observation operations differ')
        source = by_source.get(declaration['source_id'])
        require(source is not None and (source['task_id'], source['task_version']) ==
                (declaration['task_id'], declaration['task_version']), 'Prepared source/task selection differs')
        by_id = {s['sample_id']: s for s in source['samples']}
        require(declaration['sample_ids'] and set(declaration['sample_ids']) <= by_id.keys(),
                'Empty/unknown prepared sample selection')
        selected.extend(deepcopy(by_id[sid]) | {'source_id': source['source_id'], 'task_id': source['task_id'],
                        'task_version': source['task_version'], 'request_kind': source['request_kind']}
                        for sid in declaration['sample_ids'])
    qualified_samples(selected)
    require(len(selected) == compiled['semantic']['counts']['documents'], 'Prepared document count differs')
    return selected


def serving_groups(compiled, samples):
    """Freeze per-recipe unit, request and path identity before any submission."""
    models = {m['arm_id']: m for m in compiled['semantic']['models']}
    by_id = {s['sample_id']: s for s in samples}; groups = []
    require(len(by_id) == len(samples) and samples, 'Empty/ambiguous prepared request samples')
    for node in compiled['semantic']['graph']['nodes']:
        if node['operation_kind'] not in KINDS: continue
        require(node['state'] == 'NOT_EXECUTED' and node['arm_id'] in models, 'Serving node blocked/unknown arm')
        model = models[node['arm_id']]; requests = deepcopy(node['output_contract']['requests'])
        require(requests and len(requests) == node['counts']['inference_POSTs'], 'Empty/mismatched serving request coverage')
        expected = set(); actual = set()
        for sid in node['sample_ids']:
            sample = by_id.get(sid)
            require(sample is not None and sample['task_id'] == node['task_id'], 'Serving node sample/task mismatch')
            if sample['request_kind'] == 'generate_until': expected.add((sid, None))
            else: expected.update((sid, r['choice_index']) for r in sample['likelihood']['requests'])
        for request in requests:
            sample = by_id.get(request['sample_id'])
            require(sample is not None and sample['task_id'] == node['task_id'] and
                    request['request_kind'] == sample['request_kind'], 'Serving request task/role mismatch')
            key = (request['sample_id'], request.get('choice_index'))
            require(key not in actual, 'Duplicate serving request coordinate'); actual.add(key)
            if sample['request_kind'] == 'generate_until':
                require(request['wire_body'] == sample['generation']['wire_payload'] and
                        request['reference_used_for_generation'] is False, 'Generation prompt-only wire changed')
            else:
                from ..adapters.likelihood import request_body
                adapter = get_task(sample['task_id'], sample['task_version'])
                task_requests = adapter.requests(sample['authored_sample'], 'binding-static-validation',
                    sample['diagnostic']['tokenization']['tokenizer_identity'],
                    sample['diagnostic']['tokenization']['token_id_space'], node['output_contract']['runtime_identity'])
                require(type(request.get('choice_index')) is int and 0 <= request['choice_index'] < len(task_requests) and
                        request['wire_body'] == request_body(task_requests[request['choice_index']]) and
                        request.get('normalization_characters') == len(sample['authored_sample']['choices'][request['choice_index']]) and
                        request.get('reference_role') == ('gold_choice' if request['choice_index']==sample['authored_sample']['gold'] else 'non_gold_choice'),
                        'ARC all-choice request, normalization or continuation changed')
            request.update(task_id=sample['task_id'], task_version=sample['task_version'],
                           source_id=sample['source_id'], qualified_sample_id=digest(sample_namespace(sample)))
        require(actual == expected, 'Serving request exact coverage differs')
        group_id = 'serve-' + model['artifact_ref'] + '-' + digest(node['node_id'])[:12]
        groups.append({'group_id': group_id, 'arm_id': model['artifact_ref'], 'logical_arm_id': node['arm_id'],
            'task_id': node['task_id'], 'source_id': samples[next(i for i,s in enumerate(samples) if s['sample_id']==node['sample_ids'][0])]['source_id'],
            'node_ids': [node['node_id']], 'requests': requests,
            'serving_policy': deepcopy(node['output_contract']['serving_policy']),
            'serving_runtime_identity': node['output_contract']['runtime_identity'],
            'raw_root': 'acquisitions/serve/' + group_id,
            'job_id': group_id + '-attempt1'})
    require(groups and len({g['group_id'] for g in groups}) == len(groups), 'INVALID_EMPTY_PLAN: empty/duplicate serving groups')
    require(sum(len(g['requests']) for g in groups) == compiled['semantic']['counts']['inference_POSTs'],
            'Grouped serving POST count differs from Suite')
    return groups


def probe_inputs(compiled, samples, metadata, bundle_sha):
    """Derive execution provenance without changing any prepared token coordinate."""
    nodes = [n for n in compiled['semantic']['graph']['nodes'] if n['operation_kind'] == 'collect_token_diagnostic']
    require(nodes and len(nodes) == len(compiled['semantic']['models']), 'One token-only operation per model required')
    original = nodes[0]['output_contract']['resolved_probe_inputs']
    require(all(n['output_contract']['resolved_probe_inputs'] == original for n in nodes), 'Model probe inputs differ')
    require(original['kind'] == 'prepared_reference_probe_inputs_v1', 'Wrong prepared token input kind')
    require(original['rows'] == [s['diagnostic'] for s in samples], 'Suite probe selection differs from prepared manifest')
    spec = validate_spec(deepcopy(metadata['token_spec'])); rows = []
    for sample in samples:
        row = deepcopy(sample['diagnostic']); validate_diagnostic(row, metadata)
        validate_worker_row(row, spec, metadata['vocabulary_size'])
        raw = sample['raw_record']; adapter = get_task(sample['task_id'], sample['task_version'])
        original_identity = row['probe_input_identity']
        row.update(prepared_probe_input_identity=original_identity,
            dataset_identity=digest({'task_id':raw['task_id'],'task_version':raw['task_version'],
                                     'dataset_id':raw['dataset_id'],'raw_manifest_identity':bundle_sha}),
            policy_id=raw['policy_id'], policy_identity=digest({'schema_version':1,'policy_id':raw['policy_id'], 'task_id':raw['task_id']}),
            prepared_bundle_identity=bundle_sha, execution_parent_plan_identity=compiled['plan_identity'],
            source_preparation_identity=sample['diagnostic']['probe_input_identity'],
            source_scorer_identity=digest({'contract':'task_scorer_identity_v1','task_id':raw['task_id'],'task_version':raw['task_version'],
                                          'scorer_version':adapter.scorer_version}),
            qualified_sample_id=digest(sample_namespace(sample)))
        if 'gold_choice_index' in row: row['choice_index'] = row['gold_choice_index']
        row['probe_input_identity'] = digest({k:v for k,v in row.items() if k != 'probe_input_identity'})
        rows.append(row)
    vocabulary = metadata['vocabulary_size']; maximum_targets = max(len(r['targets']) for r in rows)
    maximum_sequence = max(len(r['input_ids']) for r in rows)
    targets = sum(len(r['targets']) for r in rows); full = sum(len(r['full_vocab_target_positions']) for r in rows)
    eligible = sum(sum(1 for t in range(1,len(r['input_ids'])) if r['loss_mask'][t] == 1 and
        not r['special_token_mask'][t] and r['input_ids'][t] not in r['special_token_ids']) for r in rows)
    arms = len(compiled['semantic']['models']); numeric = arms * (targets * 4 + full * vocabulary * 4)
    buffer = maximum_targets * vocabulary * 12 + maximum_sequence * metadata['model_config']['hidden_size'] * 4
    require(buffer <= spec['budget']['probe_gpu_buffer_bytes'] and numeric <= spec['budget']['raw_bytes_total'],
            'Prepared selected numeric/buffer bound exceeded')
    require(arms * targets == compiled['semantic']['counts']['main_target_observations'] and
            arms * full == compiled['semantic']['counts']['main_full_observations'] and
            arms * len(rows) == compiled['semantic']['counts']['forward_calls'], 'Prepared token count differs from Suite')
    result = {k:deepcopy(metadata[k]) for k in ('model_config','tokenizer_files','tokenizer_identity','token_id_space','vocabulary_size')}
    result.update(schema_version=1, kind='prepared_reference_probe_inputs_v1', rows=rows, rows_identity=digest(rows),
        spec=spec, spec_identity=digest(spec), sources=[],
        scope='Manifest-authorized prepared inputs; execution provenance derived without retokenization',
        selection_receipt={'policy':'exact_manifest_coordinates_v1','rows':[{'qualified_sample_id':r['qualified_sample_id'],
            'sample_id':r['sample_id'],'task_id':r['task_id'],'selected_positions':[t['token_position'] for t in r['targets']],
            'selected_full_vocab_positions':r['full_vocab_target_positions']} for r in rows]},
        counts={'samples':len(rows),'tasks':len({r['task_id'] for r in rows}),
            'target_positions_per_arm':targets,'full_vocab_positions_per_arm':full,'attention_rows_per_arm':0,
            'eligible_target_positions_per_arm':eligible},
        resource_estimates={'maximum_sequence_tokens':maximum_sequence,'maximum_selected_targets_per_sample':maximum_targets,
            'extra_probe_gpu_buffer_upper_bytes':buffer,'raw_numeric_bytes_both_arms':numeric,
            'raw_target_scalar_bytes_both_arms':arms*targets*4,
            'raw_full_logits_bytes_both_arms':arms*full*vocabulary*4,
            'raw_estimate_scope':'Selected float32 numeric only; model, native workspace and metadata separate'})
    result['selection_identity'] = digest(result)
    return result


def serving_inputs(samples, metadata, bundle_sha):
    """Preparation metadata for native requests; no diagnostic worker output."""
    result = {k:deepcopy(metadata[k]) for k in ('model_config','tokenizer_files','tokenizer_identity','token_id_space','vocabulary_size')}
    result.update(schema_version=1, kind='prepared_serving_inputs_v1',
        rows=[deepcopy(s['diagnostic']) for s in samples], rows_identity=digest([s['diagnostic'] for s in samples]),
        prepared_bundle_identity=bundle_sha, scope='Prepared reference metadata only; diagnostic collection NOT_REQUESTED')
    return result


def bind_live_plan(frozen_suite, sources_manifest, local_binding, new_clock):
    """Resolve explicit files/receipts, never GPU, network, tokenizer or model APIs."""
    check_frozen(frozen_suite)
    local = read(Path(local_binding)) if isinstance(local_binding, (str,Path)) else deepcopy(local_binding)
    required = {'prepared_manifest','prepared_manifest_sha256','artifact_receipt','runtime_metadata','runtime_python',
        'server_python','package_site','prior_ledger','campaign_root','gpu','gpu_uuid','port','kind','profile','schema_version'}
    fields(local, required, ('task_clock','parent_plan_files','historical_index',
        'serving_runtime_identity','roots','task_origin_used_seconds'))
    require(type(local['schema_version']) is int and local['schema_version'] == 1 and
            local['kind'] == 'prepared_suite_live_binding_v1' and local['profile'] in PREPARED_PROFILES, 'Unsupported prepared live binding profile')
    profile=local['profile']; serving_only=profile==SERVING_PROFILE
    path_keys = required - {'prepared_manifest_sha256','gpu','gpu_uuid','port','kind','profile','schema_version'}
    paths = {k:str(Path(local[k]).expanduser().absolute()) for k in path_keys}
    for key in ('prepared_manifest','artifact_receipt','runtime_metadata','runtime_python','server_python','prior_ledger'):
        require(Path(paths[key]).is_file(), 'Missing explicit live binding file: '+key)
    require(sha(Path(paths['prepared_manifest'])) == local['prepared_manifest_sha256'], 'Prepared inventory SHA changed')
    loaded, bundle = _load(paths['prepared_manifest'])
    require(bundle is not None and Path(sources_manifest).absolute() ==
            Path(paths['prepared_manifest']).parent / bundle['source_manifest'], 'Prepared authority/source locator differs')
    samples = selected_samples(frozen_suite, loaded, execution_kind='serve' if serving_only else 'both'); metadata = loaded['model_metadata']
    validate_budget_compatibility(frozen_suite, metadata)
    groups = serving_groups(frozen_suite, samples)
    require(not frozen_suite['semantic']['suite']['observations']['structure']['enabled'] and
            frozen_suite['semantic']['suite']['observations']['token']['enabled'] is (not serving_only), 'Requested live observation profile differs')
    require(all(m['static_capability'] == 'SUPPORTED_PRESERVED_RECIPE_ONLY' and
                all(v == 'SUPPORTED_PRESERVED_RUNTIME_RECEIPT_ONLY' for k,v in m['runtime_capability'].items() if not serving_only or k=='serving')
                for m in frozen_suite['semantic']['models']), 'UNSUPPORTED_CAPABILITY: preserved recipe required')
    selected = serving_inputs(samples, metadata, local['prepared_manifest_sha256']) if serving_only else probe_inputs(frozen_suite, samples, metadata, local['prepared_manifest_sha256'])
    spec = {'budget':{'minimum_cpu_available_bytes':10*1024**3,'minimum_disk_free_bytes':4*1024**3,'cleanup_reserve_seconds':120,'task_gpu_wall_seconds':600,'automatic_retries':0,'raw_bytes_total':256*1024**2}} if serving_only else deepcopy(selected['spec']); job_count = len(groups) + (0 if serving_only else len(frozen_suite['semantic']['models']))
    require(0 < job_count <= 8 and job_count <= frozen_suite['semantic']['suite']['budget']['max_gpu_jobs'], 'Prepared live GPU job cap exceeded')
    budget = spec['budget'] | {'max_gpu_processes':job_count, 'raw_bytes_per_job':spec['budget']['raw_bytes_total']}
    if serving_only:budget['minimum_gpu_free_mib']=12288  # conservative launch floor, not a peak-memory estimate
    require(budget['minimum_cpu_available_bytes'] >= 10*1024**3 and budget['minimum_disk_free_bytes'] >= 4*1024**3 and
            budget['cleanup_reserve_seconds'] >= 120 and budget['task_gpu_wall_seconds'] <= 600 and budget['automatic_retries'] == 0,
            'Live resource, retry or cleanup floor changed')
    resource = resources(Path(paths['package_site']).parent, budget); require(resource['ready'], 'BLOCKED_CPU_OR_DISK_RESOURCE')
    pair = deepcopy(metadata['pair']); arms = inspect_artifacts(pair, read(Path(paths['artifact_receipt'])), selected)
    require(arms and all(a['artifact_available'] for a in arms), 'BLOCKED_MISSING_LOCAL_ARTIFACT')
    models = {m['artifact_ref']:m for m in frozen_suite['semantic']['models']}
    require(set(models) == {a['arm_id'] for a in arms}, 'Live artifact arms differ from Suite')
    for arm in arms:
        require(arm['artifact']['artifact_identity'] == models[arm['arm_id']]['artifact']['artifact_identity'], 'Live artifact identity changed')
    require(type(local['gpu']) is int and local['gpu'] >= 0 and type(local['port']) is int and
            1024 <= local['port'] <= 65535 and isinstance(local['gpu_uuid'],str) and local['gpu_uuid'].startswith('GPU-'), 'Invalid explicit GPU/port binding')
    runtime = read(Path(paths['runtime_metadata']))
    require(digest(runtime) == metadata['runtime_identities']['serving' if serving_only else 'diagnostic'], 'Runtime receipt changed')
    require(all(g['serving_runtime_identity'] == metadata['runtime_identities']['serving'] for g in groups), 'Serving runtime changed')
    history = ledger_evidence(paths['prior_ledger'], paths['campaign_root'])
    allowance = min(budget['task_gpu_wall_seconds'], 7200-history['used_seconds'])
    require(allowance > budget['cleanup_reserve_seconds'], 'No cleanup-safe campaign remainder')
    clock = deepcopy(new_clock)
    require(all(type(clock.get(k)) in (int,float) and math.isfinite(clock[k]) for k in
                ('start_monotonic','deadline_monotonic','no_new_gpu_after')) and
            clock['deadline_monotonic'] == clock['start_monotonic']+3600 and
            clock['no_new_gpu_after'] == clock['start_monotonic']+3000, 'Fresh prepared-live task clock required')
    require(time.monotonic() < clock['no_new_gpu_after'] and time.monotonic() < clock['deadline_monotonic']-120,
            'Prepared live task clock expired')
    code = source_identity(); package = Path(paths['package_site'])/'inference_kit'
    require(package.is_dir(), 'Own installed worker package-site missing')
    installed = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts}
    require(installed == code, 'Controller/worker Python source identity differs')
    common = {'schema_version':1,'task':'BENCH06-A' if serving_only else 'BENCH05-E4B','status':'CPU_BOUND_NOT_EXECUTED','binding_profile':profile,
        'pair':pair,'arms':arms,'arm_order':[a['arm_id'] for a in arms],'paths':paths,
        'gpu':{'index':local['gpu'],'uuid':local['gpu_uuid']},'port':local['port'],'runtime':runtime,
        'runtime_identity':digest(runtime),'serving_runtime_identity':metadata['runtime_identities']['serving'],
        'clock':clock,'campaign':history,'task_origin_used_seconds':history['used_seconds'],'gpu_allowance_seconds':allowance,
        'source_identity':code,'samples':selected,'probe_spec':spec,'worker_contract':TOKEN_WORKER,
        'suite':{'seed':frozen_suite['semantic']['suite']['seed'],'budget':budget,'observer':{'off_samples':0,'off_repeats':0},
            'serving':deepcopy(groups[0]['serving_policy']),'serving_job_seconds':180},
        'data':{'rows':[deepcopy(s['authored_sample']) for s in samples]},'evidence_kind':'planned_live_suite_execution',
        'parent_suite_plan_identity':frozen_suite['plan_identity'],'parent_task':'BENCH06-A' if serving_only else 'BENCH05-E4B','resources':resource,
        'resource_estimates':deepcopy(selected.get('resource_estimates',{'diagnostic':'NOT_REQUESTED','full_vocabulary':'NOT_COLLECTED'}))}
    token = deepcopy(common);token.update(task='BENCH05-C2',raw_format='token_probe_raw_v1',
        evidence_kind='planned_live_token_probe', scope='New prepared acquisition; retained task-neutral token-only recipe')
    token['plan_identity'] = digest(token)
    if serving_only: token=None
    jobs = {g['group_id']:{'source_arm_id':g['arm_id'],'logical_arm_id':g['logical_arm_id'], 'task_id':g['task_id'],
        'source_id':g['source_id'],'node_ids':g['node_ids'],'raw_root':g['raw_root'],
        'raw_path':g['raw_root']+'/attempts/'+g['job_id']+'/task-responses.jsonl',
        'journal_path':g['raw_root']+'/attempts/'+g['job_id']+'/request-journal.jsonl'} for g in groups}
    loaded = deepcopy(loaded); loaded['samples'] = samples
    binding_keys = ('prepared_manifest','artifact_receipt','runtime_metadata','prior_ledger')
    files = {paths[k]:sha(Path(paths[k])) for k in binding_keys}
    return {'schema_version':1,'kind':'suite_live_context_v1','binding_profile':profile,
        'suite_plan':deepcopy(frozen_suite),'sources':loaded,'runtime_plan':common,'token_plan':token,
        'serving_groups':groups,'serving_jobs':jobs,'campaign':history,
        **({} if serving_only else {'diagnostic_run':'acquisitions/token'}),
        'historical_index':local.get('historical_index'),'binding_files':files,
        'source_manifest_path':str(Path(sources_manifest).absolute()),'source_manifest_sha256':sha(Path(sources_manifest)),
        'runtime_rows':{r['sample_id']:deepcopy(r) for r in selected['rows']},
        'qualified_samples':qualified_samples(samples),'prepared_manifest_sha256':local['prepared_manifest_sha256'],
        'ledger_task_id':'BENCH06-A' if serving_only else 'BENCH05-E4B','expected_gpu_jobs':job_count,
        'token_analysis':{'source':'native_choice_likelihood_v1','diagnostic':'NOT_REQUESTED','full_vocabulary':'NOT_COLLECTED'} if serving_only else None,
        'effective_usage_scope':{'ENGINEERING_INPUT_ONLY':'ENGINEERING_REGRESSION_REUSE','REGRESSION_REUSE':'REGRESSION_REUSE'} if serving_only else {},
        'execution_unit_ids':[g['group_id'] for g in groups]+([] if serving_only else ['token-'+a['arm_id'] for a in arms]),
        'allowed_ledger_job_ids':[g['job_id'] for g in groups]+([] if serving_only else ['main-'+a['arm_id']+'-attempt1' for a in arms]),
        'execution_count':0,'gpu_queries':0,'tokenizer_calls':0,'model_loads':0,
        'live_binding_status':'CPU_BOUND_NOT_EXECUTED_REQUIRES_RUNTIME_AND_LOOPBACK_PREFLIGHT'}


def verify_prepared_context(context):
    """Recheck the full prepared inventory and execution-only coordinate mapping.

    The caller also performs the retained shared source/artifact/stat/clock
    checks. This helper adds no GPU query and never changes an input file.
    """
    require(context.get('binding_profile') in PREPARED_PROFILES, 'Wrong prepared live profile')
    serving_only=context['binding_profile']==SERVING_PROFILE
    plan = context['runtime_plan']; path = Path(plan['paths']['prepared_manifest'])
    require(sha(path) == context['prepared_manifest_sha256'], 'Prepared inventory SHA changed')
    loaded, manifest = _load(path); require(manifest is not None, 'Prepared inventory authority missing')
    selected = selected_samples(context['suite_plan'], loaded, execution_kind='serve' if serving_only else 'both')
    require(context['sources']['samples'] == selected, 'Bound original prepared samples changed')
    require(context['qualified_samples'] == qualified_samples(selected), 'Qualified prepared identity changed')
    expected = serving_inputs(selected, loaded['model_metadata'], context['prepared_manifest_sha256']) if serving_only else probe_inputs(context['suite_plan'], selected, loaded['model_metadata'], context['prepared_manifest_sha256'])
    require(plan['samples'] == expected and (context['token_plan'] is None if serving_only else context['token_plan']['samples'] == expected) and
            context['runtime_rows'] == {r['sample_id']:r for r in expected['rows']}, 'Prepared execution coordinate/provenance changed')
    groups = serving_groups(context['suite_plan'], selected)
    require(context['serving_groups'] == groups, 'Prepared execution unit/request mapping changed')
    expected_units = [g['group_id'] for g in groups] + ([] if serving_only else ['token-'+a['arm_id'] for a in plan['arms']])
    jobs = [g['job_id'] for g in groups] + ([] if serving_only else ['main-'+a['arm_id']+'-attempt1' for a in plan['arms']])
    require(context['execution_unit_ids'] == expected_units and context['expected_gpu_jobs'] == len(expected_units) and
            context['allowed_ledger_job_ids'] == jobs and len(set(jobs)) == len(jobs), 'Prepared execution job coverage changed')
    return True


def verify_live_context(context):
    """Prepared-specific validation; shared runtime/artifact gate remains required."""
    return verify_prepared_context(context)


def binding_layout(compiled, loaded):
    """Portable contract-only layout, with no artifact/runtime/live assertion.

    A source manifest's bytes, not model files, are authority for this view.
    The full live binder separately requires the complete preparation inventory.
    """
    serving_only=loaded['model_metadata'].get('capability_profile')=='qwen3_4b_gptq_serving_v1'
    samples = selected_samples(compiled, loaded, execution_kind='serve' if serving_only else 'both')
    validate_budget_compatibility(compiled, loaded['model_metadata'])
    if serving_only:
        groups=serving_groups(compiled,samples)
        units=[{'node_id':g['group_id'],'operation_kind':'serve_group','arm_id':g['arm_id'],
                'provides':g['node_ids'],'raw_root':g['raw_root'],'ledger_job_id':g['job_id'],
                'inference_POSTs':len(g['requests'])} for g in groups]
        expected={n['node_id'] for n in compiled['semantic']['graph']['nodes'] if n['operation_kind'] in KINDS}
        actual=[p for u in units for p in u['provides']]
        require(actual and len(actual)==len(set(actual)) and set(actual)==expected and
                len(units)<=compiled['semantic']['suite']['budget']['max_gpu_jobs']<=6,
                'Serving-only execution units/coverage differ')
        return {'schema_version':1,'kind':'prepared_suite_binding_layout_v1','profile':SERVING_PROFILE,
            'status':'CONTRACT_VALIDATED_ONLY','plan_identity':compiled['plan_identity'],
            'data_identity':loaded['data_identity'],'qualified_samples':qualified_samples(samples),
            'serving_groups':groups,'execution_units':units,
            'counts':deepcopy(compiled['semantic']['counts'])|{'execution_units':len(units)},
            'diagnostic':'NOT_REQUESTED','full_vocabulary':'NOT_COLLECTED',
            'artifact_presence':'ARTIFACT_PRESENCE_NOT_CHECKED','runtime':'CHECK_NOT_RUN',
            'authorization':'NONE','live_status':'LIVE_BINDING_NOT_VALIDATED',
            'GPU_queries':0,'model_loads':0,'inference_POSTs':0,'tokenizer_calls':0,'ledger_changes':0}
    groups = serving_groups(compiled, samples)
    probe = probe_inputs(compiled, samples, loaded['model_metadata'], loaded['manifest_sha256'])
    models = compiled['semantic']['models']
    count = len(groups) + len(models)
    require(0 < count <= 8 and count <= compiled['semantic']['suite']['budget']['max_gpu_jobs'],
            'Prepared live GPU job cap exceeded')
    units = [{'node_id':g['group_id'],'operation_kind':'serve_group','arm_id':g['arm_id'],
              'provides':g['node_ids'],'raw_root':g['raw_root'],'ledger_job_id':g['job_id'],
              'inference_POSTs':len(g['requests'])} for g in groups]
    for model in models:
        node = [n for n in compiled['semantic']['graph']['nodes'] if
                n['operation_kind']=='collect_token_diagnostic' and n['arm_id']==model['arm_id']]
        require(len(node)==1, 'Token operation coverage differs')
        units.append({'node_id':'token-'+model['artifact_ref'],'operation_kind':'token_group',
            'arm_id':model['artifact_ref'],'provides':[node[0]['node_id']],
            'raw_root':'acquisitions/token','ledger_job_id':'main-'+model['artifact_ref']+'-attempt1',
            'forward_calls':len(samples)})
    provides = [name for unit in units for name in unit['provides']]
    expected = {n['node_id'] for n in compiled['semantic']['graph']['nodes']
                if n['operation_kind'] in KINDS | {'collect_token_diagnostic'}}
    require(provides and len(provides)==len(set(provides)) and set(provides)==expected,
            'Execution acquisition units must provide each typed collector exactly once')
    return {'schema_version':1,'kind':'prepared_suite_binding_layout_v1','profile':PROFILE,
        'status':'CONTRACT_VALIDATED_ONLY','plan_identity':compiled['plan_identity'],
        'data_identity':loaded['data_identity'],'qualified_samples':qualified_samples(samples),
        'serving_groups':groups,'probe_inputs':probe,'execution_units':units,
        'counts':deepcopy(compiled['semantic']['counts']) | {'execution_units':count,
            'eligible_target_positions_per_arm':probe['counts']['eligible_target_positions_per_arm']},
        'artifact_presence':'ARTIFACT_PRESENCE_NOT_CHECKED','runtime':'CHECK_NOT_RUN',
        'authorization':'NONE','live_status':'LIVE_BINDING_NOT_VALIDATED',
        'GPU_queries':0,'model_loads':0,'inference_POSTs':0,'tokenizer_calls':0,'ledger_changes':0}
