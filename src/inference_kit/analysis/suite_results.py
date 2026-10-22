"""CPU-only E3 task replay, existing token analysis and E1 result registration.

The immutable request/response journal is authoritative.  Scorers and token
formulas remain in their existing task adapters and C1 metrics.  Reading this
module never imports a model, tokenizer, network client or an execution runtime.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile

from ..core.io import digest, read, require, safe_name, sha, write
from ..core.state import new_output
from ..tasks.registry import get_task
from .tokens import jsonl, write_jsonl, source_identity


VERSION = 'suite_results_e3_v1'


def _native_only(context):
    """Versioned serving-only binding; absence of diagnostic raw is explicit."""
    native = context.get('binding_profile') == 'prepared_qwen_serve_v1'
    if native:
        require(context.get('token_analysis') == {'source': 'native_choice_likelihood_v1',
                'diagnostic': 'NOT_REQUESTED', 'full_vocabulary': 'NOT_COLLECTED'},
                'Serving-only native token analysis contract changed')
        require('diagnostic_run' not in context, 'Serving-only Suite cannot carry a diagnostic acquisition')
    return native


def _analysis_revision(value):
    require(value is None or isinstance(value, str) and 0 < len(value) <= 128,
            'Invalid explicit CPU analysis revision')
    return value


def index_source_spec(run, *, analysis_revision=None):
    """Bind a display alias to this raw source, independently of its locator."""
    root = Path(run).absolute(); raw_sha = sha(root / 'RUN_MANIFEST.json')
    source_id = digest({'adapter':'suite_e3','raw_manifest':raw_sha})
    result = {'source_id':'suite-current','alias_scope':'suite-source:' + source_id,
              'adapter':'suite_e3','path':str(root),'raw_manifest_sha256':raw_sha}
    if _analysis_revision(analysis_revision) is not None:
        result['analysis_revision'] = analysis_revision
    return result


def _file(root, name):
    root = Path(root).resolve(); relative = safe_name(name); path = root
    for component in Path(relative).parts:
        path = path / component
        require(not path.is_symlink(), 'Suite raw member symlink refused')
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            'Missing or unsafe suite raw member: ' + name)
    return path


def verify_run(run):
    """Verify the explicit immutable subset, never a mutable checkpoint."""
    root = Path(run).absolute(); require(root.is_dir() and not root.is_symlink(), 'Explicit suite run directory required')
    manifest = read(_file(root, 'RUN_MANIFEST.json'))
    files = manifest.get('files'); require(isinstance(files, dict) and files, 'Empty suite raw manifest')
    require({'SUITE_PLAN.json', 'suite-execution-context.json'} <= files.keys(), 'Suite manifest lacks frozen context/plan')
    for name, entry in files.items():
        path = _file(root, name)
        require(type(entry.get('bytes')) is int and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                'Suite raw member content changed: ' + name)
    context = read(root / 'suite-execution-context.json')
    sources = context.get('sources', context.get('input_sources'))
    require(isinstance(sources, dict) and sources.get('sources'), 'Missing inline prepared suite inputs')
    from ..suites.suite_inputs import validate_slice
    from ..suites.prepared_inputs import KIND as PREPARED_KIND, validate_source
    for source in sources['sources']:
        (validate_source if source.get('kind') == PREPARED_KIND else validate_slice)(source)
    compiled = context.get('suite_plan', read(root / 'SUITE_PLAN.json'))
    compiled = compiled.get('compiled_record', compiled.get('compiled', compiled))
    require(compiled.get('kind') == 'frozen_suite_plan_v1' and
            compiled['plan_identity'] == digest(compiled['semantic']), 'Suite compiled plan identity changed')
    if _native_only(context):
        suite=compiled['semantic']['suite'];observations=suite['observations']
        require(all(t['execution_kind']=='serve' for t in suite['tasks']) and
                observations['token']['enabled'] is False and observations['structure']['enabled'] is False and
                not any(n['operation_kind'].startswith('collect_') for n in compiled['semantic']['graph']['nodes']),
                'Serving-only Suite cannot substitute diagnostic/structure operations')
    require(sources['data_identity'] == compiled['semantic']['inputs']['data_identity'], 'Suite inputs differ from compiled plan')
    by_source={s['source_id']:s for s in sources['sources']}
    nodes={n['node_id']:n for n in compiled['semantic']['graph']['nodes']}
    for index, task in enumerate(compiled['semantic']['suite']['tasks']):
        require(task['source_id'] in by_source, 'Missing source frozen in suite')
        source=by_source[task['source_id']];by_sample={r['sample_id']:r for r in source['samples']}
        rows=[{**deepcopy(by_sample[s]),'task_id':task['task_id'],'task_version':task['task_version'],
               'source_id':task['source_id'],'request_kind':source['request_kind']} for s in task['sample_ids']]
        required_identity=digest({'source_id':task['source_id'],'samples':rows,'input_mode':task['input_mode']})
        require(nodes['task'+str(index)+'.validate']['outputs'][0]['identity']==required_identity,
                'Inline prepared samples/gold differ from frozen operation identity')
    for job in context.get('serving_jobs', {}).values():
        for key in ('raw_path', 'journal_path'):
            require(job[key] in files, 'Serving raw/journal not pinned in immutable run manifest')
    return root, context, sources, compiled, manifest


def _arms(sources, compiled):
    metadata = sources['model_metadata']; original = {a['arm_id']: a for a in metadata['arms']}
    semantic={m['arm_id']:m for m in compiled['semantic']['models']};result={}
    for model in compiled['semantic']['suite']['models']:
        arm=original[model['artifact_ref']] | {'arm_id':model['arm_id'],'source_arm_id':model['artifact_ref']};pinned=semantic[model['arm_id']]
        require(digest(arm['artifact'])==digest(pinned['artifact']), 'Suite model receipt differs from frozen artifact/path')
        if model['profile']=='qwen3_4b_gptq_serving_v1':
            expected='NATIVE_ARTIFACT' if arm['source_format']=='native' else 'PACKED_ARTIFACT_RUNTIME'
            require(arm['execution_mode']=='NATIVE_SERVING' and pinned['diagnostic_path']=='UNSUPPORTED_CAPABILITY' and
                    pinned['serving_path']==expected, 'Suite serving receipt differs from frozen native path')
        else:
            require(arm['execution_mode']==pinned['diagnostic_path'], 'Suite model receipt differs from frozen artifact/path')
        result[model['arm_id']]=arm
    return result


def _selected(sources, compiled):
    by_source = {s['source_id']: s for s in sources['sources']}; result = []
    for spec in compiled['semantic']['suite']['tasks']:
        if spec['execution_kind'] not in ('serve', 'both'): continue
        require(spec['input_mode'] == 'prepared_manifest' and spec['source_id'] in by_source, 'Task has no executable prepared source')
        source = by_source[spec['source_id']]; rows = {s['sample_id']: s for s in source['samples']}
        for sample_id in spec['sample_ids']:
            require(sample_id in rows, 'Unknown suite scoring sample')
            result.append((source, rows[sample_id]))
    require(result and len({(s['task_id'], r['sample_id']) for s, r in result}) == len(result), 'Empty/duplicate suite task samples')
    # The historical index emitter uses the original sample_id as its lookup.
    # Refuse ambiguous namespaces before constructing that map; never rename an
    # authored ID or silently attach one task's result to another task.
    require(len({r['sample_id'] for _, r in result}) == len(result),
            'UNSUPPORTED_AMBIGUOUS_SAMPLE_ID: qualified namespaces require distinct index lookup keys')
    return result


def _serving_jobs(context, arms):
    """Normalize physical serving units while retaining legacy arm-keyed jobs."""
    jobs = context.get('serving_jobs', {}); require(isinstance(jobs, dict), 'Invalid serving job mapping')
    result = []; paths = set()
    for key, job in jobs.items():
        require(isinstance(job, dict), 'Invalid serving job')
        arm = job.get('logical_arm_id', job.get('arm_id', key))
        require(arm in arms, 'Unknown serving job model arm')
        require(job.get('source_arm_id', arms[arm]['source_arm_id']) == arms[arm]['source_arm_id'],
                'Serving group artifact arm mapping differs')
        require(job.get('group_id', key) == key, 'Serving group key/identity mismatch')
        for name in ('raw_path', 'journal_path'):
            path = safe_name(job[name]); require(path not in paths, 'Serving groups share an evidence path'); paths.add(path)
        result.append((key, arm, job))
    return result


def _job_contains(job, source, sample):
    return (job.get('task_id', source['task_id']) == source['task_id'] and
            job.get('task_version', source['task_version']) == source['task_version'] and
            job.get('source_id', source['source_id']) == source['source_id'] and
            sample['sample_id'] in job.get('sample_ids', [sample['sample_id']]))


def _sample_jobs(context, arms, selected):
    jobs = _serving_jobs(context, arms); result = {}
    for arm in arms:
        for source, sample in selected:
            matches = [(key, job) for key, model, job in jobs if model == arm and _job_contains(job, source, sample)]
            # Missing evidence is scored as NOT_RUN. Ambiguous acquisition
            # ownership cannot be resolved by choosing an arbitrary group.
            require(len(matches) <= 1, 'Sample belongs to multiple serving execution units')
            result[(arm, source['task_id'], sample['sample_id'])] = matches[0] if matches else (None, None)
    return jobs, result


def _execution_prefix(record, compiled, arm, group_id, prepared):
    suffix = '|' + arm['source_arm_id'] + ('|' + group_id if prepared else '')
    require(record['execution_prefix'].startswith(compiled['plan_identity'] + '|') and
            record['execution_prefix'].endswith(suffix),
            'Request execution prefix differs from current suite/source arm/unit')
    if prepared and record.get('event') not in ('http_submit', 'response_end'):
        require(record.get('execution_unit_id') == group_id, 'Raw execution unit identity differs from serving group')


def _requests(source, sample, prefix, metadata):
    adapter = get_task(source['task_id'], source['task_version']); authored = sample['authored_sample']
    if adapter.request_kind == 'generate_until':
        return adapter.requests(authored | {'generation_input_ids': sample['generation']['wire_payload']['prompt']},
                                prefix, metadata['tokenizer_identity'])
    return adapter.requests(authored, prefix, metadata['tokenizer_identity'], metadata['token_id_space'],
                            metadata['runtime_identities']['serving'])


def _expected_record(source, sample, record, metadata):
    prefix = record.get('execution_prefix')
    require(isinstance(prefix, str) and prefix, 'Missing actual request execution prefix')
    choices = _requests(source, sample, prefix, metadata)
    index = record.get('choice_index', 0)
    require(type(index) is int and 0 <= index < len(choices), 'Unknown suite request choice')
    request = choices[index]
    journal=record.get('event') in ('http_submit','response_end')
    require(record['request_id'] == request.request_id and
            ('identity' not in record and journal or digest(record.get('identity')) == digest(request.identity)),
            'Suite typed request/sample/tokenizer identity mismatch')
    require(record['task_id'] == source['task_id'] and record['task_version'] == source['task_version'] and
            record['sample_id'] == sample['sample_id'] and (journal and 'request_kind' not in record or record.get('request_kind') == request.request_kind),
            'Suite raw task/request kind mismatch')
    if source.get('kind') == 'prepared_task_source_v1' and not journal:
        raw = sample['raw_record']
        key = {name: raw[name] for name in ('task_id', 'task_version', 'dataset_id', 'split', 'sample_id')}
        require(record.get('source_sample_key') == key and record.get('qualified_sample_identity') == digest(key) and
                record.get('qualified_sample_id') == digest(key),
                'Prepared request qualified sample namespace differs')
    return request


def _flatten(score):
    result = {k: v for k, v in score.items() if v is None or type(v) in (bool, int, float)}
    for family in ('field_correct', 'evidence_correct'):
        result.update({family + '.' + k: v for k, v in score.get(family, {}).items()})
    return result


def _binary_names(score, adapter):
    declared={spec['metric_id'] for spec in adapter.metric_specs() if isinstance(spec,dict) and 'fraction' in str(spec.get('unit',''))}
    return {name for name,value in _flatten(score).items() if type(value) is bool or
            name in declared and (value is None or type(value) in (int,float) and value in (0,1))}


def _latency(rows, key):
    values = [r[key] for r in rows if type(r.get(key)) in (int, float)]
    return {'n': len(values), 'p50': statistics.median(values) if values else None, 'max': max(values) if values else None}


def score_tasks(run, output):
    """Replay all three task kinds, preserving missing and submitted failures."""
    root, context, sources, compiled, manifest = verify_run(run); out = new_output(output)
    metadata = sources['model_metadata']; arms = _arms(sources, compiled); selected = _selected(sources, compiled)
    expected = {(s['task_id'], r['sample_id']): (s, r) for s, r in selected}
    jobs, sample_jobs = _sample_jobs(context, arms, selected)
    prepared = any(s.get('kind') == 'prepared_task_source_v1' for s in sources['sources'])
    documents = []; generations = []; choices = []; native = []; request_ids = set(); summaries = {};submitted_count=0
    for arm_id, arm in arms.items():
        records = []; events = []; raw_paths = {}; journal_groups = {}; record_groups = {}
        for group_id, model, job in jobs:
            if model != arm_id: continue
            for record in jsonl(_file(root, job['raw_path'])):
                key = (record['task_id'], record['sample_id']); require(key in expected, 'Unknown suite raw task/sample')
                source, sample = expected[key]
                require(_job_contains(job, source, sample), 'Raw response belongs to another serving execution unit')
                require(record['request_id'] not in raw_paths, 'Duplicate global suite request ID')
                raw_paths[record['request_id']] = job['raw_path']; record_groups[record['request_id']] = group_id; records.append(record)
            for event in jsonl(_file(root, job['journal_path'])):
                key = (event['task_id'], event['sample_id']); require(key in expected, 'Unknown journal task/sample')
                source, sample = expected[key]
                require(_job_contains(job, source, sample), 'Journal belongs to another serving execution unit')
                rid = event['request_id']
                require(rid not in journal_groups or journal_groups[rid] == group_id, 'Request journal crosses serving execution units')
                journal_groups[rid] = group_id; raw_paths.setdefault(rid, job['raw_path']); events.append(event)
        submissions = {}; responses = {}; ends={};by_sample = {}
        for record in records:
            rid = record['request_id']; require(rid not in request_ids, 'Duplicate global suite request ID'); request_ids.add(rid)
            key = (record['task_id'], record['sample_id']); require(key in expected, 'Unknown suite raw task/sample')
            source, sample = expected[key]; request = _expected_record(source, sample, record, metadata)
            _execution_prefix(record, compiled, arm, record_groups[rid], prepared)
            require(record['arm_id'] == arm_id and record['artifact_identity'] == arm['artifact']['artifact_identity'], 'Suite raw artifact/arm mismatch')
            responses[rid] = record; by_sample.setdefault(key, []).append((request, record))
        for event in events:
            require(event.get('event') in ('http_submit', 'response_end'), 'Unknown immutable request journal event')
            key=(event['task_id'],event['sample_id']);require(key in expected,'Unknown journal task/sample')
            source,sample=expected[key];request=_expected_record(source,sample,event,metadata)
            _execution_prefix(event, compiled, arm, journal_groups[event['request_id']], prepared)
            require(event['arm_id']==arm_id and event['artifact_identity']==arm['artifact']['artifact_identity'],'Journal artifact/arm differs')
            rid = event['request_id'];table = submissions if event['event'] == 'http_submit' else ends
            require(rid not in table, 'Duplicate immutable journal event')
            if event['event']=='http_submit':
                from ..adapters.likelihood import request_body
                body=sample['generation']['wire_payload'] if request.request_kind=='generate_until' else request_body(request)
                require(digest(event['request_body'])==digest(body),'Journal wire differs from frozen request')
                if rid in responses:require(digest(event['request_body'])==digest(responses[rid]['request_body']),'Journal/response wire mismatch')
            table[rid]=event
        require(set(ends)<=submissions.keys(),'Response-end journal lacks submission')
        require(set(responses) <= submissions.keys(), 'Raw response has no immutable HTTP submit journal')
        submitted_count+=len(submissions)
        # A submitted request without a response remains an incomplete attempt.
        for rid, event in submissions.items():
            if rid in responses: continue
            key = (event['task_id'], event['sample_id']); require(key in expected, 'Unknown submitted task/sample')
            source, sample = expected[key]; request = _expected_record(source, sample, event, metadata)
            incomplete = event | {'transport_complete': False, 'termination': 'INCOMPLETE_SUBMISSION',
                'output_text': '', 'request_latency_http': None, 'request_latency_task': None,
                'request_kind':request.request_kind,'identity':request.identity,
                'sample_identity':sample['authored_sample'].get('sample_identity'),
                'recovered_from_submission_journal': True}
            by_sample.setdefault(key, []).append((request, incomplete))
        for source, sample in selected:
            task_id = source['task_id']; sid = sample['sample_id']; adapter = get_task(task_id, source['task_version'])
            group_id, job = sample_jobs[(arm_id, task_id, sid)]
            raw = by_sample.get((task_id, sid), []); authored = sample['authored_sample']; references = []
            normalized = None
            if adapter.request_kind == 'generate_until':
                require(len(raw) <= 1, 'Duplicate suite natural generation')
                if raw:
                    request, record = raw[0]; normalized = dict(record)
                    if not record.get('recovered_from_submission_journal'):
                        from ..suites.bench05b1_replay import generation_from_raw
                        stub = {'samples': {'tokenizer_identity': metadata['tokenizer_identity']},
                                'evidence_kind': 'synthetic_contract' if context.get('evidence_kind') == 'synthetic_contract' else 'live_task_generation'}
                        normalized = generation_from_raw(record, sample['generation']['prepared'], arm, stub,
                            task_id=task_id, task_version=source['task_version'], max_new_tokens=sample['generation']['wire_payload']['max_tokens'])
                    require(digest(record['request_body']) == digest(sample['generation']['wire_payload']), 'Generation wire differs from preserved prompt-only payload')
                    normalized['raw_reference'] = raw_paths[request.request_id] + '#request_id=' + request.request_id
                    generations.append(normalized); references.append(normalized['raw_reference'])
                score = adapter.score(authored, normalized)
            else:
                decoded = []
                for request, record in raw:
                    if record.get('recovered_from_submission_journal'):
                        row = {'status':'INCOMPLETE','sample_id':sid,'identity':request.identity,'choice_index':request.payload['choice_index'],
                               'tokens':[],'request_id':request.request_id,'error':'Response missing after immutable submission'}
                    else:
                        from ..tasks.arc_challenge import decode_record
                        row = decode_record(request, record)
                    row.update(task_id=task_id, task_version=source['task_version'], arm_id=arm_id,
                               reference_scope='GOLD_CHOICE' if request.payload['choice_index'] == authored['gold'] else 'NON_GOLD_CHOICE',
                               raw_reference=raw_paths[request.request_id] + '#request_id=' + request.request_id)
                    choices.append(row); decoded.append(row); references.append(row['raw_reference'])
                    for token in row['tokens']:
                        native.append(token | {'task_id':task_id,'task_version':source['task_version'],'arm_id':arm_id,
                            'reference_scope':row['reference_scope'],'raw_reference':row['raw_reference']})
                score = adapter.score(authored, decoded)
            document = {'task_id':task_id,'task_version':source['task_version'],'sample_id':sid,'arm_id':arm_id,
                'request_kind':adapter.request_kind,'scorer_version':adapter.scorer_version,'score':score,
                'source_id':source['source_id'],'sample_identity':authored.get('sample_identity'),
                'parent_ancestry':source.get('ancestry', source.get('preparation')),'prompt_hash':sample['diagnostic']['prompt_hash'],
                'reference_hash':sample['diagnostic']['reference_hash'],'raw_references':references,
                'scope':'SELECTED_ENGINEERING_REGRESSION_REUSE','new_quality_scope':'Descriptive reused inputs; no independent TEST'}
            if prepared:
                document.update(serving_group_id=group_id,
                    usage_scope=sample['raw_record']['usage_scope'],
                    scope=sample['raw_record']['usage_scope'],
                    new_quality_scope='Descriptive supplied engineering inputs; no independent TEST or customer validation')
                if _native_only(context):
                    original_scope=sample['raw_record']['usage_scope']
                    effective=context.get('effective_usage_scope',{}).get(original_scope,original_scope)
                    require(isinstance(effective,str) and effective,'Missing effective usage scope')
                    document.update(source_usage_scope=original_scope,usage_scope=effective,scope=effective)
            documents.append(document)
        for source in sources['sources']:
            rows = [r for r in documents if r['arm_id'] == arm_id and r['task_id'] == source['task_id']]
            if not rows: continue
            adapter=get_task(source['task_id'],source['task_version'])
            names = sorted(set().union(*(_binary_names(r['score'],adapter) for r in rows)))
            summaries.setdefault(source['task_id'], {})[arm_id] = {'planned_documents':len(rows),
                'observed_documents':sum(r['score']['status'] not in ('NOT_RUN',) for r in rows),
                'complete_documents':sum((r['score']['status']=='COMPLETE' if r['request_kind']=='choice_loglikelihood' else r['score'].get('transport_complete') is True) for r in rows),
                'metrics':{k:{'correct':sum(_flatten(r['score']).get(k) in (True,1) for r in rows),'planned_denominator':len(rows),
                              'observed_denominator':sum(_flatten(r['score']).get(k) is not None for r in rows)} for k in names},
                'latency_http_seconds':_latency([r['score'] for r in rows], 'request_latency_http')}
    paired = []
    by = {(r['task_id'],r['sample_id'],r['arm_id']):r for r in documents}
    for comparison in compiled['semantic']['suite']['comparisons']:
        for source, sample in selected:
            left = by[(source['task_id'],sample['sample_id'],comparison['baseline'])]
            right = by[(source['task_id'],sample['sample_id'],comparison['candidate'])]
            def complete(row): return row['score']['status']=='COMPLETE' if row['request_kind']=='choice_loglikelihood' else row['score'].get('transport_complete') is True
            valid = complete(left) and complete(right); a, b = _flatten(left['score']), _flatten(right['score'])
            metrics = {}
            adapter=get_task(source['task_id'],source['task_version'])
            for metric in sorted(_binary_names(left['score'],adapter) & _binary_names(right['score'],adapter)):
                if a[metric] is not None and b[metric] is not None:
                    metrics[metric] = {'baseline':a[metric],'candidate':b[metric], 'outcome':
                        ('both_correct' if a[metric] and b[metric] else 'loss' if a[metric] else 'gain' if b[metric] else 'both_wrong') if valid else 'UNKNOWN'}
            paired.append({'task_id':source['task_id'],'sample_id':sample['sample_id'],'comparison':comparison,
                'status':'COMPLETE' if valid else 'INCOMPLETE','metrics':metrics,'missing_policy':'No zero imputation or reduced planned denominator'})
    for name, rows in (('document-scores.jsonl',documents),('generations-replayed.jsonl',generations),
                       ('choice-scores.jsonl',choices),('native-tokens.jsonl',native),('paired-outcomes.jsonl',paired)):
        write_jsonl(out/name,rows)
    summary = {'schema_version':1,'analyzer':VERSION,'plan_identity':compiled['plan_identity'],
        'raw_manifest_sha256':sha(root/'RUN_MANIFEST.json'),'tasks':summaries,
        'planned_source_documents':len(selected),'document_score_rows':len(documents),
        'inference_submissions':submitted_count,'inference_responses':len(request_ids),'native_token_rows':len(native),
        'all_required_task_inputs_complete':all((r['score']['status']=='COMPLETE' if r['request_kind']=='choice_loglikelihood' else r['score'].get('transport_complete') is True) for r in documents),
        'new_GPU_measurements_in_replay':0,'new_inference_requests':0,
        'diagnostic_deployment_equivalence':'NOT_ESTABLISHED','combined_task_score':'NOT_DEFINED'}
    write(out/'summary.json',summary); return summary


def analyze_tokens(run, output):
    root, context, sources, compiled, manifest = verify_run(run); out = new_output(output)
    from .tokens import export_tokens, compare_tokens
    if _native_only(context):
        from .suite_native_source import export_native_view
        score_tasks(root, out/'native-source')
        export_native_view(root, context, sources, compiled, manifest, out/'native-source', out/'token-view')
        input_identity=sha(root/'RUN_MANIFEST.json')
        write(out/'INPUT_SOURCES.json',{'schema_version':1,'kind':'native_choice_likelihood_v1',
            'parent_run_manifest_sha256':input_identity,'diagnostic':'NOT_REQUESTED',
            'full_vocabulary':'NOT_COLLECTED','audit_generated_tokens':'EXCLUDED'})
    elif manifest.get('scope')=='suite_e3_task_slice_v1':
        from .tokens import verify_view
        relative=safe_name(context['token_view']);view=root/relative;verify_view(view)
        require(relative+'/TOKEN_VIEW_MANIFEST.json' in manifest['files'],'Slice view not pinned in slice manifest')
        shutil.copytree(view,out/'token-view')
        input_identity=sha(view/'TOKEN_VIEW_MANIFEST.json')
        write(out/'INPUT_SOURCES.json',{'kind':'explicit_portable_token_view_slice','path':relative,'sha256':input_identity,
            'parent_scope':'Selected three-document token slice; no full parent raw manifest bypass'})
    else:
        relative = context.get('diagnostic_run', 'acquisitions/token'); diagnostic = root / safe_name(relative)
        require((relative+'/RAW_MANIFEST.json') in manifest['files'], 'Diagnostic raw manifest not pinned in suite run')
        input_identity=sha(diagnostic/'RAW_MANIFEST.json')
        write(out/'INPUT_SOURCES.json',{'schema_version':1,'kind':'token_input_sources','sources':[
            {'source_id':'suite-token-acquisition','adapter':'token_probe_raw_v1','path':str(diagnostic),
             'raw_manifest_sha256':input_identity}]})
        export_tokens(out/'INPUT_SOURCES.json',out/'token-view')
    comparisons = compiled['semantic']['suite']['comparisons']; require(len(comparisons)==1,'E3 token analysis requires one declared model pair')
    comparison = comparisons[0];mapping={m['arm_id']:m['artifact_ref'] for m in compiled['semantic']['suite']['models']}
    result=compare_tokens(out/'token-view',mapping[comparison['baseline']],mapping[comparison['candidate']],out/'token-metrics')
    write(out/'TOKEN_ANALYSIS_RECEIPT.json',{'status':'RAW_RECOMPUTED','existing_engine':'C1 tokens_export/tokens_compare',
        'source_manifest_sha256':input_identity,'logical_comparison':comparison,
        'logical_to_observed_arm':mapping,'GPU_actions':0,'result':result})
    return result


def _suite_envelope(spec, root, context, sources, compiled, manifest, work):
    from .. import __version__
    from .index_sources import _emit, _token_metrics
    raw_sha=sha(root/'RUN_MANIFEST.json');sid=digest({'adapter':'suite_e3','raw_manifest':raw_sha})
    files=deepcopy(manifest['files']);files['RUN_MANIFEST.json']={'bytes':(root/'RUN_MANIFEST.json').stat().st_size,'sha256':raw_sha}
    out={'source_id':sid,'source_alias':spec['source_id'],'content_identity':raw_sha,
        'locator':{'path':str(root),'kind':'directory','relative_root':'','manifest_path':'RUN_MANIFEST.json','manifest_sha256':raw_sha,'files':files},
        'payload':{'adapter':'suite_e3','scope':'suite_execution_raw_v1','plan_identity':compiled['plan_identity'],
                   'verification':'IMMUTABLE_RAW_RECOMPUTED','new_measurement':'E3 current acquisition; CPU import is not new inference'},
        'acquisitions':[],'analyses':[],'samples':[],'executions':[],'measurements':[],'evidence':[],'relations':[]}
    selected=_selected(sources,compiled);arms=_arms(sources,compiled);sample_maps={};analyses={};executions={};acquisitions={}
    jobs,sample_jobs=_sample_jobs(context,arms,selected)
    prepared=any(s.get('kind')=='prepared_task_source_v1' for s in sources['sources']);native_only=_native_only(context)
    if prepared:
        units=[('serve',group,arm,[(s,r) for s,r in selected if _job_contains(job,s,r)])
               for group,arm,job in jobs]
        if not native_only: units += [('token',arm,arm,selected) for arm in arms]
        out['payload']['new_measurement']='Current prepared-input suite acquisition; CPU import is not new inference'
    else:
        units=[(family,arm,arm,selected) for family in ('serve','token') for arm in arms]
    require(units and all(rows for _,_,_,rows in units),'Empty required suite acquisition coverage')
    for family,unit_id,arm_id,rows in units:
        arm=arms[arm_id]
        identity={'suite_raw_manifest':raw_sha,'family':family,'arm_id':arm_id}
        if prepared and family=='serve':identity['group_id']=unit_id
        aid=digest(identity);acquisitions[(family,unit_id)]=aid
        eid=digest([aid,'execution']);executions[(family,unit_id)]=eid
        revision = _analysis_revision(spec.get('analysis_revision'))
        analysis_key = [aid, VERSION, source_identity()]
        if revision is not None: analysis_key.append({'analysis_revision':revision})
        analysis=digest(analysis_key);analyses[(family,unit_id)]=analysis
        acquisition_payload={'plan_identity':compiled['plan_identity'],'family':family,'arm_id':arm_id,'raw_manifest_sha256':raw_sha,'new_measurement':True}
        execution_payload={'arm_id':arm_id,'source_arm_id':arm['source_arm_id'],'family':family,'artifact_identity':arm['artifact']['artifact_identity'],
            'execution_mode':arm['execution_mode'] if family=='token' else 'NATIVE_ARTIFACT' if arm['source_format']=='native' else 'PACKED_ARTIFACT_RUNTIME'}
        if native_only:
            execution_payload.update(model_id=sources['model_metadata']['pair']['model_id'],
                model_revision=sources['model_metadata']['pair']['revision'],
                quantization_method=arm.get('quantization_method',arm['artifact'].get('method','NOT_EXPOSED')),
                runtime_identity=sources['model_metadata']['runtime_identities']['serving'],
                tokenizer_identity=sources['model_metadata']['tokenizer_identity'])
        if prepared and family=='serve':
            acquisition_payload['group_id']=unit_id;execution_payload['group_id']=unit_id
        out['acquisitions'].append({'id':aid,'payload':acquisition_payload})
        analysis_payload = {'analyzer':VERSION,'source_identity':source_identity(),'package_version':__version__,'new_gpu_measurements':0}
        if revision is not None: analysis_payload['analysis_revision'] = revision
        out['analyses'].append({'id':analysis,'acquisition_id':aid,'payload':analysis_payload})
        out['executions'].append({'id':eid,'acquisition_id':aid,'payload':execution_payload})
        sample_map={};sample_maps[(family,unit_id)]=sample_map
        for source,sample in rows:
            p=sample['diagnostic'];key=digest([aid,source['task_id'],sample['sample_id']])
            require(sample['sample_id'] not in sample_map,'Ambiguous suite acquisition sample lookup')
            sample_map[sample['sample_id']]=key
            payload={'task_id':source['task_id'],'task_version':source['task_version'],'sample_id':sample['sample_id'],'source_id':source['source_id'],
                'prompt_hash':p['prompt_hash'],'reference_hash':p['reference_hash'],'tokenizer_identity':sources['model_metadata']['tokenizer_identity'],
                'token_id_space':sources['model_metadata']['token_id_space'],'prepared':p,'gold':sample['authored_sample']['gold'],
                'reference_text':p['reference_text'],'document':sample['authored_sample'].get('document'),'historical_outcome':False,'family':family}
            if prepared:
                payload.update(dataset_id=sample['raw_record']['dataset_id'],split=sample['raw_record']['split'],usage_scope=sample['raw_record']['usage_scope'])
                if native_only:
                    payload.update(source_usage_scope=sample['raw_record']['usage_scope'],
                        usage_scope=context.get('effective_usage_scope',{}).get(sample['raw_record']['usage_scope'],sample['raw_record']['usage_scope']))
            out['samples'].append({'id':key,'acquisition_id':aid,'task_id':source['task_id'],'sample_id':sample['sample_id'],'payload':payload})
    for name,entry in files.items():
        out['evidence'].append({'id':digest([sid,name,entry['sha256']]),'relative_path':name,'bytes':entry['bytes'],'sha256':entry['sha256'],'payload':{'source_id':sid,'kind':'IMMUTABLE_SUITE_RAW'}})
    score_tasks(root,work/'scores');analyze_tokens(root,work/'tokens')
    for row in jsonl(work/'scores/document-scores.jsonl'):
        arm=row['arm_id'];coords={k:row[k] for k in ('task_id','task_version','sample_id','arm_id')}
        unit=sample_jobs[(arm,row['task_id'],row['sample_id'])][0] if prepared else arm
        require(unit is not None,'Task metric lacks serving acquisition identity')
        for metric,value in _flatten(row['score']).items():
            _emit(out,analyses[('serve',unit)],sample_maps[('serve',unit)],'task.'+metric,value,
                row['score']['status'],'current_task',coords,row,executions[('serve',unit)])
    for row in jsonl(work/'scores/native-tokens.jsonl'):
        coords={k:row[k] for k in ('task_id','task_version','sample_id','arm_id','choice_index','token_position','logit_position','token_id','prefix_hash','reference_scope')}
        unit=sample_jobs[(row['arm_id'],row['task_id'],row['sample_id'])][0] if prepared else row['arm_id']
        require(unit is not None,'Native token metric lacks serving acquisition identity')
        _emit(out,analyses[('serve',unit)],sample_maps[('serve',unit)],'native_token.nll_nat',row['nll_nat'],'MEASURED','native_likelihood',coords,row,executions[('serve',unit)])
    pair=compiled['semantic']['suite']['comparisons'][0];baseline=pair['baseline'];candidate=pair['candidate']
    if native_only:
        native_selected=[(source,sample) for source,sample in selected if source['request_kind']=='choice_loglikelihood']
        require(native_selected,'Serving-only native token analysis lacks required continuation inputs')
        for unit in {sample_jobs[(baseline,source['task_id'],sample['sample_id'])][0] for source,sample in native_selected}:
            require(unit is not None,'Native comparison lacks serving acquisition')
            _token_metrics(out,analyses[('serve',unit)],sample_maps[('serve',unit)],
                work/'tokens/token-metrics/token-comparisons.jsonl',work/'tokens/token-view',
                pass_id='NATIVE_CHOICE_LIKELIHOOD')
    else:
        _token_metrics(out,analyses[('token',baseline)],sample_maps[('token',baseline)],work/'tokens/token-metrics/token-comparisons.jsonl',work/'tokens/token-view',
                       evidence_prefix=context.get('diagnostic_run','acquisitions/token'))
    for source,sample in selected:
        if native_only:
            left_unit=sample_jobs[(baseline,source['task_id'],sample['sample_id'])][0]
            right_unit=sample_jobs[(candidate,source['task_id'],sample['sample_id'])][0]
            left=sample_maps[('serve',left_unit)][sample['sample_id']];right=sample_maps[('serve',right_unit)][sample['sample_id']]
            out['relations'].append({'id':digest([left,right,'WITHIN_RUN_MODEL_PAIR']),'kind':'WITHIN_RUN_MODEL_PAIR',
                'status':'EXISTING_COMPARATOR_CONTRACT_VERIFIED','left_id':left,'right_id':right,
                'payload':{'scope':'native_choice_likelihood' if source['request_kind']=='choice_loglikelihood' else 'task_outcome',
                    'existing_engine':'C1' if source['request_kind']=='choice_loglikelihood' else 'task_scorer',
                    'diagnostic':'NOT_REQUESTED','full_vocabulary':'NOT_COLLECTED','deployment_equivalence':'NOT_ESTABLISHED'}})
            continue
        for arm in arms:
            unit=sample_jobs[(arm,source['task_id'],sample['sample_id'])][0] if prepared else arm
            require(unit is not None,'Related sample lacks serving acquisition')
            left=sample_maps[('serve',unit)][sample['sample_id']];right=sample_maps[('token',arm)][sample['sample_id']]
            out['relations'].append({'id':digest([left,right,'SAME_INPUT_RELATED_ONLY']),'kind':'SAME_INPUT_RELATED_ONLY','status':'PROVEN_RELATED_ONLY',
                'left_id':left,'right_id':right,'payload':{'same_source_sample':True,'same_prompt_reference_provenance':True,'delta_permitted':False,
                    'reason':'Native task response and teacher-forced decoded/instrumented diagnostic are separate acquisitions'}})
        left=sample_maps[('token',baseline)][sample['sample_id']];right=sample_maps[('token',candidate)][sample['sample_id']]
        out['relations'].append({'id':digest([left,right,'WITHIN_RUN_MODEL_PAIR']),'kind':'WITHIN_RUN_MODEL_PAIR','status':'EXISTING_COMPARATOR_CONTRACT_VERIFIED',
            'left_id':left,'right_id':right,'payload':{'scope':'token_pair','existing_engine':'C1','deployment_equivalence':'NOT_ESTABLISHED'}})
    out['verification']={'status':'RAW_RECOMPUTED','raw_members':len(files),'acquisitions':len(out['acquisitions']),
        'measurements':len(out['measurements']),'new_gpu_measurements_in_analyzer':0}
    if native_only:
        out['payload'].update(scope='suite_serving_native_likelihood_v1',diagnostic='NOT_REQUESTED',full_vocabulary='NOT_COLLECTED')
    return out


def load_index_source(spec, base):
    """Explicit E1 adapter; no archive code or arbitrary loader is accepted."""
    require(spec.get('adapter')=='suite_e3' and isinstance(spec.get('source_id'),str) and spec['source_id'],'Unsupported suite index source')
    path=Path(spec['path']);path=path if path.is_absolute() else Path(base)/path
    root,context,sources,compiled,manifest=verify_run(path)
    require(manifest.get('derived_slice_kind') not in ('suite_all_scalar_selected_full_slice_v1','suite_native_task_slice_v1'),
            'SLICE_ONLY: explicit token/task slice cannot register full parent acquisitions')
    if spec.get('raw_manifest_sha256'):require(sha(root/'RUN_MANIFEST.json')==spec['raw_manifest_sha256'],'Suite source manifest pin changed')
    with tempfile.TemporaryDirectory(prefix='inference-kit-suite-index-') as tmp:
        return _suite_envelope(spec,root,context,sources,compiled,manifest,Path(tmp))


def validate_current_index(db, run, *, publication=None):
    """Verify the required current locator now, separately from historical state.

    A different good locator cannot conceal corruption of this analysis's
    published copy. This is also used when re-reading completion evidence.
    """
    from contextlib import closing
    from .result_index import _connect, validate_locator
    from ..core.io import loads
    raw_sha = sha(Path(run) / 'RUN_MANIFEST.json')
    sid = digest({'adapter': 'suite_e3', 'raw_manifest': raw_sha})
    expected = None
    if publication is not None:
        require(publication['manifest_sha256'] == raw_sha and
                publication['canonical_source_id'] == sid,
                'Current publication source identity differs')
        expected = Path(publication['evidence_root']).absolute()
    with closing(_connect(db)) as connection:
        source = connection.execute('SELECT content_identity FROM sources WHERE id=?', (sid,)).fetchone()
        require(source is not None and source[0] == raw_sha, 'MISSING_EVIDENCE: required current source absent')
        locators = [loads(row[0]) for row in connection.execute(
            'SELECT payload FROM locators WHERE source_id=? ORDER BY id', (sid,))]
    if expected is not None:
        locators = [loc for loc in locators if Path(loc['path']).absolute() == expected]
    require(locators, 'MISSING_EVIDENCE: required current publication locator absent')
    checks = [validate_locator(loc) for loc in locators]
    good = next((check for check in checks if check['status'] == 'RAW_HASH_VERIFIED'), None)
    require(good is not None, 'CURRENT_SOURCE_NOT_VERIFIED: ' + str(checks))
    return {'status': 'RAW_HASH_VERIFIED', 'source_id': sid, 'manifest_sha256': raw_sha,
            'required_locator': str(expected) if expected is not None else None,
            'verified_files': good['verified_files'],
            'scope': 'Required current locator re-read; historical source validity is reported separately'}


def validate_index_completion(output, run):
    """Revalidate a new publication on CPU/live completion and no-op reads."""
    output = Path(output)
    receipt = output / 'EVIDENCE_PUBLICATION_RECEIPT.json'
    if receipt.exists():
        return validate_current_index(output / 'index.sqlite', run, publication=read(receipt))
    # Older completion formats and synthetic executor handlers have no
    # publication receipt. Their existing source/file checks remain unchanged.
    return None


def report_packaging_for_run(run):
    """Read policy from immutable Suite intent; legacy runs retain single mode."""
    compiled = read(Path(run) / 'SUITE_PLAN.json')
    return compiled['semantic']['suite']['outputs'].get('report_packaging')


def validate_report_completion(output, report_packaging=None):
    """Validate the exact current collection, including every part and member.

    Legacy executor fixtures have no collection manifest and retain their
    existing per-file completion checks. Explicit collection policy requires
    the collection itself; another valid source locator cannot substitute it.
    """
    from .report_packaging import validate_report_packaging, validate_report_collection
    policy = validate_report_packaging(report_packaging)
    manifest = Path(output) / 'REPORT_COLLECTION_MANIFEST.json'
    if policy['mode'] == 'bounded_collection_v1' or manifest.exists():
        receipt = validate_report_collection(output)
        require(receipt['packaging'] == policy, 'Report completion packaging differs from frozen policy')
        return receipt
    return None


def execute_cpu(kind, run, output, *, stage_outputs=None):
    """Dispatch four fixed CPU operations; no scheduler or arbitrary SQL."""
    stage_outputs=stage_outputs or {};root=Path(run).absolute()
    if kind=='score_task':
        result=score_tasks(root,output)
        return {'status':'COMPLETE' if result['document_score_rows']>0 and result['all_required_task_inputs_complete'] else 'PARTIAL', 'result':result}
    if kind=='analyze_tokens':
        result=analyze_tokens(root,output);coverage=result['coverage']
        return {'status':'COMPLETE' if coverage['requested_pair_coordinates']>0 and coverage['measured']==coverage['requested_pair_coordinates'] else 'PARTIAL','result':result}
    if kind=='index_add':
        from .result_index import _create,add_sources
        from .evidence_publication import publish_suite_evidence
        out=new_output(output);db=out/'index.sqlite';historical=stage_outputs.get('historical_index');history={'status':'NOT_INCLUDED'}
        publication_parent = out.parent / 'published-evidence'
        if publication_parent.resolve().is_relative_to(root.resolve()):
            # Live acquisition roots contain their derived node outputs. Keep
            # the immutable publication beside that root, never recursively
            # inside it. CPU workspaces normally use their own sibling root.
            publication_parent = root.parent / (root.name + '-published-evidence')
        publication = publish_suite_evidence(root, publication_parent)
        write(out/'EVIDENCE_PUBLICATION_RECEIPT.json', publication)
        if historical is not None:
            old=Path(historical).absolute();require(old.is_file() and not old.is_symlink(),'Missing explicit historical index')
            before=sha(old)
            context=read(root/'suite-execution-context.json')
            # A new, explicitly frozen CPU analysis may choose a newer readonly
            # baseline. Legacy live execution retains its original binding pin.
            if 'historical_index_sha256' in stage_outputs:
                expected_history = stage_outputs['historical_index_sha256']
                require(isinstance(expected_history, str) and len(expected_history) == 64 and
                        all(c in '0123456789abcdef' for c in expected_history),
                        'Explicit CPU historical index SHA256 required')
            else:
                expected_history = context.get('historical_index_sha256')
            if expected_history:require(before==expected_history,'Historical index differs from pinned execution binding')
            with sqlite3.connect(old.as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(db) as target:source.backup(target)
            require(sha(old)==before,'Historical index changed during readonly copy')
            history={'status':'READONLY_REGISTERED_INDEX_COPY','sha256':before,'scope':'Prior registered records retained; current raw not re-acquired or relabelled'}
        else:_create(db)
        write(out/'INPUT_SOURCES.json',{'schema_version':1,'sources':[
            index_source_spec(publication['evidence_root'], analysis_revision=stage_outputs.get('analysis_revision'))]})
        result=add_sources(out/'INPUT_SOURCES.json',db,out/'registration');write(out/'HISTORICAL_INDEX_RECEIPT.json',history)
        validation = validate_current_index(db, root, publication=publication)
        write(out/'CURRENT_SOURCE_VALIDATION.json', validation)
        return {'status':'COMPLETE' if result['sources'] and all(r['status']=='REGISTERED' for r in result['sources']) else 'PARTIAL',
                'result':result,'db':str(db),'historical_index':history,
                'publication':publication,'current_source_validation':validation}
    if kind=='report':
        from .result_index import report_index
        prior=Path(stage_outputs.get('index_add',root/'nodes/index_add'));db=prior if prior.suffix=='.sqlite' else prior/'index.sqlite'
        publication_path = db.parent/'EVIDENCE_PUBLICATION_RECEIPT.json'
        publication = read(publication_path) if publication_path.exists() else None
        validate_current_index(db, root, publication=publication)
        packaging = stage_outputs.get('report_packaging', report_packaging_for_run(root))
        result = report_index(db,output,packaging=packaging)
        validation = validate_current_index(db, root, publication=publication)
        collection = validate_report_completion(output, packaging)
        if collection is None:
            write(Path(output)/'CURRENT_SOURCE_VALIDATION.json',validation)
        return {'status':'COMPLETE','result':result,'current_source_validation':validation,
                **({'report_collection_validation':collection} if collection is not None else {})}
    raise ValueError('Unsupported suite CPU operation: '+str(kind))


def replay(run, output, *, historical_index=None, report_packaging=None):
    """Replay sealed immutable raw with installed CPU code, without inference."""
    root,context,sources,compiled,manifest=verify_run(run)
    if manifest.get('scope')=='suite_e3_task_slice_v1':
        require(historical_index is None,'MINI task/token slice does not import historical/full-acquisition index')
        require(report_packaging is None,'MINI slice replay has no cumulative report operation; use explicit CPU postprocess')
        return replay_slice(root,output)
    out=new_output(output);stages={};results={}
    if historical_index is not None:stages['historical_index']=historical_index
    if report_packaging is not None:
        from .report_packaging import validate_report_packaging
        stages['report_packaging']=validate_report_packaging(report_packaging)
    for kind in ('score_task','analyze_tokens','index_add','report'):
        target=out/kind;results[kind]=execute_cpu(kind,run,target,stage_outputs=stages);stages[kind]=target
    receipt={'schema_version':1,'status':'CPU_RAW_REPLAYED','analyzer':VERSION,'source_identity':source_identity(),
        'run_manifest_sha256':sha(Path(run)/'RUN_MANIFEST.json'),'GPU_actions':0,'model_loads':0,'tokenizer_calls':0,
        'inference_requests':0,'historical_index':'Explicit prior registered index' if historical_index else 'NOT_INCLUDED', 'results':results}
    write(out/'CPU_REPLAY_RECEIPT.json',receipt);return receipt


def _portable_metadata(value):
    """Drop locator-only metadata, preserving intent and raw acquisition IDs."""
    if isinstance(value,dict):
        return {k:_portable_metadata(v) for k,v in value.items()
                if k not in {'physical_locator','input_path_as_metadata','source_bundle_path','bindings'}}
    if isinstance(value,list):return [_portable_metadata(v) for v in value]
    return value


def export_mini_slice(run, token_view, output):
    """Create an explicit closed HTTP + three-document TokenView slice.

    HTTP response/journal bytes are copied verbatim. Full numeric arrays retain
    their original bytes and coordinates; the new slice manifest enumerates its
    own members and never purports to satisfy the parent's full manifest.
    """
    root,context,sources,compiled,manifest=verify_run(run)
    from .tokens import verify_view, metric_contract
    view=Path(token_view).absolute();parent_view=verify_view(view);out=new_output(output)
    if _native_only(context):
        records=list(jsonl(view/'token_records.jsonl'))
        require(records and all(r['acquisition_source']=='native_likelihood' and r.get('full_logits') is None for r in records),
                'Native Suite MINI requires nonempty target-only native view')
        require(len({r['record_id'] for r in records})==len(records) and
                all(r['bundle_identity']==sha(root/'RUN_MANIFEST.json') for r in records), 'Native slice parent view differs')
        requests=[]
        for job in context['serving_jobs'].values():
            for key in ('raw_path','journal_path'):
                relative=safe_name(job[key]);destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(_file(root,relative),destination)
                if key=='raw_path':requests.extend(r['request_id'] for r in jsonl(destination))
        require(requests and len(set(requests))==len(requests),'Empty/duplicate native slice requests')
        portable_sources={k:_portable_metadata(v) for k,v in sources.items() if k!='samples'}
        portable_plan=_portable_metadata(compiled)
        write(out/'suite-execution-context.json',{'binding_profile':context['binding_profile'],
            'sources':portable_sources,'suite_plan':portable_plan,'serving_jobs':context['serving_jobs'],
            'evidence_kind':context.get('evidence_kind'),'token_analysis':context['token_analysis'],
            'effective_usage_scope':context.get('effective_usage_scope',{}),
            'scope':'SLICE_ALL_SELECTED_HTTP_AND_NATIVE_CONTINUATION_SCALARS'})
        write(out/'SUITE_PLAN.json',portable_plan)
        parent_sha=sha(root/'RUN_MANIFEST.json')
        receipt={'schema_version':1,'kind':'suite_native_task_slice_v1',
            'parent_raw_manifest_sha256':parent_sha,'parent_plan_identity':compiled['plan_identity'],
            'parent_canonical_source_id':digest({'adapter':'suite_e3','raw_manifest':parent_sha}),
            'parent_acquisition_ids':[digest({'suite_raw_manifest':parent_sha,'family':'serve',
                'arm_id':logical,'group_id':group}) for group,logical,_ in _serving_jobs(context,_arms(sources,compiled))],
            'parent_view_identity':parent_view['view_identity'],'included_request_ids':sorted(requests),
            'included_token_coordinates':[{k:r[k] for k in ('record_id','arm_id','task_id','sample_id','choice_index','token_position','target_token_id','prefix_hash')} for r in records],
            'coverage':{'HTTP_response_rows':len(requests),'native_scalar_rows':len(records),
                'native_measured':sum(r['status']=='MEASURED' for r in records),'full_vectors':0},
            'copied_bytes':'HTTP responses and journals copied verbatim; scalar logprob is re-decoded from original responses',
            'excluded':'Runtime, models, mutable execution checkpoint, ledger and full parent manifest; no diagnostic/full-vocabulary measurements',
            'metadata_derivation':'Only locator-only metadata removed; frozen semantic plan and actual request/coordinate IDs retained',
            'not_supported':'Parent acquisition index registration, GPU/environment/resume recertification, or fresh tokenization'}
        write(out/'SLICE_RECEIPT.json',receipt)
        files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file()}
        write(out/'RUN_MANIFEST.json',{'schema_version':1,'scope':'suite_e3_task_slice_v1',
            'derived_slice_kind':receipt['kind'],'files':files})
        verify_run(out);return receipt
    records=list(jsonl(view/'token_records.jsonl'));tasks=sorted({r['task_id'] for r in records})
    require(len(tasks)==3,'MINI requires three actual diagnostic task panels')
    chosen={task:sorted({r['sample_id'] for r in records if r['task_id']==task},
        key=lambda sid:hashlib.sha256((task+'|'+sid).encode('utf-8')).hexdigest())[0] for task in tasks}
    selected=[deepcopy(r) for r in records if chosen[r['task_id']]==r['sample_id']]
    positions={}
    for task,sid in chosen.items():
        by_arm={}
        for row in selected:
            if row['task_id']==task and row['sample_id']==sid and row.get('full_logits') is not None:
                by_arm.setdefault(row['arm_id'],set()).add(row['token_position'])
        require(len(by_arm)==2,'MINI selected sample lacks two actual full-vocabulary arms')
        common=set.intersection(*by_arm.values());require(common,'MINI selected sample lacks common full coordinate')
        positions[(task,sid)]=min(common)
    target=out/'token-view';target.mkdir();excluded_full=[]
    for row in selected:
        numeric=row.get('full_logits')
        if numeric is None:continue
        if row['token_position']!=positions[(row['task_id'],row['sample_id'])]:
            excluded_full.append({'record_id':row['record_id'],'token_position':row['token_position'],'sha256':numeric['sha256']})
            row['full_logits']=None
        else:
            source=_file(view,numeric['path']);require(sha(source)==numeric['sha256'],'MINI numeric identity changed')
            destination=target/safe_name(numeric['path']);destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copyfile(source,destination)
            require(sha(destination)==numeric['sha256'],'MINI numeric copy changed')
    sequence_ids={r['sequence_id'] for r in selected};sequences=[r for r in jsonl(view/'sequences.jsonl') if r['sequence_id'] in sequence_ids]
    require(len(sequences)==len(sequence_ids),'MINI sequence coverage differs')
    write_jsonl(target/'token_records.jsonl',selected);write_jsonl(target/'sequences.jsonl',sequences);write_jsonl(target/'generation_records.jsonl',[])
    source_panels=_portable_metadata(parent_view['source_panels'])
    for panel in source_panels:
        panel['parent_coverage']=panel.get('coverage');panel['scope']='TOKEN_VIEW_SLICE_ONLY'
        rows=[r for r in selected if r.get('source_id')==panel.get('source_id')]
        panel['coverage']={'target_requested':len(rows),'target_measured':sum(r['status']=='MEASURED' for r in rows),
            'full_vocab_measured':sum(r.get('full_logits') is not None for r in rows),
            'meaning':'This explicit slice only; parent coverage retained separately as provenance'}
    write(target/'SOURCES.json',source_panels);write(target/'METRIC_CONTRACT.json',metric_contract())
    write(target/'INPUT_SOURCES.json',{'kind':'explicit_suite_token_slice','parent_view_identity':parent_view['view_identity'],
        'parent_view_manifest_sha256':sha(view/'TOKEN_VIEW_MANIFEST.json'),'included_samples':chosen})
    coverage={'status':'SLICE_ONLY','records':len(selected),'sequences':len(sequences),'generation_records':0,
        'full_logits_records':sum(r.get('full_logits') is not None for r in selected),'GPU_actions':0,
        'parent_records':len(records),'excluded_records':len(records)-len(selected),'full_position_policy':'one existing common full-logit coordinate per selected document/arm'}
    write(target/'COVERAGE.json',coverage)
    inventory={p.relative_to(target).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in target.rglob('*') if p.is_file()}
    write(target/'TOKEN_VIEW_MANIFEST.json',{'schema_version':1,'kind':'common_token_view_v1','analysis_source_digest':source_identity(),
        'input_manifest_sha256':sha(target/'INPUT_SOURCES.json'),'source_panels':source_panels,'files':inventory,'view_identity':digest(inventory),
        'new_observations':False,'GPU_actions':0})
    for job in context['serving_jobs'].values():
        for key in ('raw_path','journal_path'):
            relative=safe_name(job[key]);destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(_file(root,relative),destination)
    portable_sources={k:_portable_metadata(v) for k,v in sources.items() if k!='samples'}
    portable_plan=_portable_metadata(compiled)
    portable_context={'sources':portable_sources,'suite_plan':portable_plan,'serving_jobs':context['serving_jobs'],
        'evidence_kind':context.get('evidence_kind'),'token_view':'token-view','scope':'SLICE_HTTP_ALL_SELECTED_TASKS_AND_THREE_DIAGNOSTIC_DOCUMENTS'}
    write(out/'suite-execution-context.json',portable_context);write(out/'SUITE_PLAN.json',portable_plan)
    receipt={'schema_version':1,'kind':'suite_e3_mini_slice_v1','parent_raw_manifest_sha256':sha(root/'RUN_MANIFEST.json'),
        'parent_plan_identity':compiled['plan_identity'],'parent_view_identity':parent_view['view_identity'],
        'HTTP':'All actual selected task responses and immutable request journals; copied bytes unchanged',
        'token_samples':chosen,'full_position_by_sample':{task+'|'+sid:position for (task,sid),position in positions.items()},
        'coverage':coverage,'excluded_full_distribution_metadata':excluded_full,
        'metadata_derivation':'Locator-only metadata removed; omitted full_logits references set null in this derived view; original record/observation IDs, scalar values, coordinates and retained numeric bytes preserved',
        'not_supported':'Full parent acquisition replay, resume, historical E1 index, or new model inference'}
    write(out/'SLICE_RECEIPT.json',receipt)
    files={p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file()}
    write(out/'RUN_MANIFEST.json',{'schema_version':1,'scope':'suite_e3_task_slice_v1','files':files})
    verify_run(out);verify_view(target);return receipt


def replay_slice(run, output):
    root,context,sources,compiled,manifest=verify_run(run)
    require(manifest.get('scope')=='suite_e3_task_slice_v1','Not an explicit suite MINI slice')
    receipt=read(_file(root,'SLICE_RECEIPT.json'))
    kind=receipt.get('kind')
    if kind=='suite_native_task_slice_v1':
        require(manifest.get('derived_slice_kind')==kind and _native_only(context),'Native slice manifest/analysis contract differs')
        parent=receipt.get('parent_raw_manifest_sha256')
        require(isinstance(parent,str) and len(parent)==64 and all(c in '0123456789abcdef' for c in parent),'Invalid native slice ancestry')
        requests=[r['request_id'] for j in context['serving_jobs'].values() for r in jsonl(_file(root,j['raw_path']))]
        require(requests and len(set(requests))==len(requests) and sorted(requests)==receipt['included_request_ids'] and
                len(requests)==receipt['coverage']['HTTP_response_rows'],'Native slice request coverage differs')
        out=new_output(output);results={operation:execute_cpu(operation,root,out/operation) for operation in ('score_task','analyze_tokens')}
        records=list(jsonl(out/'analyze_tokens/token-view/token_records.jsonl'))
        coordinates=[{k:r[k] for k in ('record_id','arm_id','task_id','sample_id','choice_index','token_position','target_token_id','prefix_hash')} for r in records]
        # Record IDs use the explicit slice manifest as bundle identity; exact
        # physical coordinates/arm/request values retain the parent meaning.
        axis=lambda row:tuple(row[k] for k in ('arm_id','task_id','sample_id','choice_index','token_position','target_token_id','prefix_hash'))
        require(len(records)==receipt['coverage']['native_scalar_rows'] and
                len({axis(r) for r in coordinates})==len(records) and
                sorted(map(axis,coordinates))==sorted(map(axis,receipt['included_token_coordinates'])) and
                sum(r['status']=='MEASURED' for r in records)==receipt['coverage']['native_measured'],
                'Native slice continuation coverage differs')
        result={'schema_version':1,'status':'CPU_SLICE_REPLAYED','scope':'All selected task HTTP and native continuation scalar rows; no full parent index/import',
            'parent_raw_manifest_sha256':parent,'slice_manifest_sha256':sha(root/'RUN_MANIFEST.json'),
            'native_scalar_rows':len(records),'full_vectors':0,'full_vocabulary':'NOT_COLLECTED',
            'source_identity':source_identity(),'results':results,'GPU_actions':0,'model_loads':0,'tokenizer_calls':0,
            'historical_index':'NOT_INCLUDED','parent_acquisition_index_registration':'NOT_SUPPORTED','resume_certification':'NOT_CLAIMED'}
        write(out/'CPU_REPLAY_RECEIPT.json',result);return result
    require(kind in ('suite_e3_mini_slice_v1','suite_all_scalar_selected_full_slice_v1'),'Missing slice ancestry contract')
    if kind=='suite_all_scalar_selected_full_slice_v1':
        require(manifest.get('derived_slice_kind')==kind,'Slice manifest/ancestry format differs')
    parent=receipt.get('parent_run_manifest_sha256') if kind=='suite_all_scalar_selected_full_slice_v1' else receipt.get('parent_raw_manifest_sha256')
    require(isinstance(parent,str) and len(parent)==64 and all(c in '0123456789abcdef' for c in parent),'Invalid parent slice manifest identity')
    records=list(jsonl(_file(root,safe_name(context['token_view'])+'/token_records.jsonl')))
    require(records and len({r['record_id'] for r in records})==len(records) and
            receipt['coverage']['records']==len(records) and
            receipt['coverage']['full_logits_records']==sum(r.get('full_logits') is not None for r in records),
            'Slice token coverage differs from pinned inventory')
    out=new_output(output);results={kind:execute_cpu(kind,root,out/kind) for kind in ('score_task','analyze_tokens')}
    result={'schema_version':1,'status':'CPU_SLICE_REPLAYED','scope':'Selected HTTP task responses and explicitly inventoried diagnostic token slice only',
        'parent_raw_manifest_sha256':parent,'slice_manifest_sha256':sha(root/'RUN_MANIFEST.json'),
        'diagnostic_documents':len({(r['task_id'],r['task_version'],r['sample_id']) for r in records}),
        'diagnostic_scalar_rows':len(records),'full_vectors':sum(r.get('full_logits') is not None for r in records),
        'source_identity':source_identity(),'results':results,'GPU_actions':0,'model_loads':0,'tokenizer_calls':0,
        'full_acquisition_replay':'NOT_CLAIMED','historical_index':'NOT_INCLUDED','resume_certification':'NOT_CLAIMED'}
    write(out/'CPU_REPLAY_RECEIPT.json',result);return result
