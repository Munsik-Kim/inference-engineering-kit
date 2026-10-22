"""Explicit, data-only E1 adapters for preserved benchmark evidence.

``load_source(spec, manifest_parent)`` returns one normalized envelope.  Its
``source_id`` and acquisition IDs exclude the locator and caller alias.  The
locator is a separate, hash-pinned whitelist; consumers can revalidate it without
trusting an old import.  All metric values come from maintained CPU replayers or
task scorers, never a hand-transcribed report.  No model/runtime is imported.
"""
from contextlib import contextmanager
from pathlib import Path
import tempfile

from ..core.evidence import open_bundle
from ..core.io import digest, read, require, safe_name, sha, write
from .tokens import jsonl, source_identity

VERSION = 'e1_explicit_sources_v1'
ADAPTERS = {
    'structure_d1': ('BENCH05-D1', 'structure_probe_raw_v1'),
    'token_c2': ('BENCH05-C2', 'token_probe_raw_v1'),
    'task_b2_p00': ('BENCH05-B2-CONTRACT', 'bench05b2_contract_raw'),
    'task_b3': ('BENCH05-B3', 'bench05b3_arc_raw'),
    'task_b4': ('BENCH05-B4', 'bench05b4_context_raw'),
}


@contextmanager
def _root(spec, base):
    path = Path(spec['path']).expanduser()
    path = path if path.is_absolute() else Path(base) / path
    path = path.absolute()
    with open_bundle(path, expected_sha256=spec.get('sha256')) as bundle:
        task, scope = ADAPTERS[spec['adapter']]
        candidates = [p.parent for p in bundle.rglob('RAW_MANIFEST.json')
                      if (p.parent / 'PLAN_FROZEN.json').is_file()
                      and read(p.parent / 'PLAN_FROZEN.json').get('task') == task
                      and read(p).get('scope') == scope]
        require(len(candidates) == 1, 'Expected one explicit E1 source run')
        root = candidates[0]
        manifest = read(root / 'RAW_MANIFEST.json')
        if spec.get('raw_manifest_sha256'):
            require(sha(root / 'RAW_MANIFEST.json') == spec['raw_manifest_sha256'], 'E1 raw manifest changed')
        files = dict(manifest['files'])
        files['RAW_MANIFEST.json'] = {'bytes': (root / 'RAW_MANIFEST.json').stat().st_size,
                                      'sha256': sha(root / 'RAW_MANIFEST.json')}
        for name, entry in files.items():
            p = root / safe_name(name)
            require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()), 'Unsafe/missing E1 raw member')
            require(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'E1 raw member changed: ' + name)
        plan = read(root / 'PLAN_FROZEN.json')
        frozen = dict(plan); expected = frozen.pop('plan_identity')
        require(digest(frozen) == expected, 'E1 frozen plan changed')
        locator = {'path': str(path), 'kind': 'directory' if path.is_dir() else 'zip',
                   'relative_root': root.relative_to(bundle).as_posix(),
                   'manifest_path': 'RAW_MANIFEST.json', 'manifest_sha256': sha(root / 'RAW_MANIFEST.json'),
                   'files': files}
        if path.is_file(): locator['archive_sha256'] = sha(path)
        yield root, plan, manifest, locator


def _analysis_identity(adapter, spec):
    from .. import __version__
    selection = {'sample_ids': spec.get('sample_ids')}
    if adapter in ('task_b2_p00', 'task_b4'): selection['splits'] = spec.get('splits', ['EVAL_LOCKED'])
    if adapter == 'task_b3': selection['partitions'] = spec.get('partitions', ['REGRESSION'])
    return {'adapter': adapter, 'adapter_version': VERSION,
            'analyzer_source_identity': source_identity(),
            'package_version': __version__,
            'selection': selection,
            'new_gpu_measurements': 0, 'new_task_quality': 'NOT_EVALUATED'}


def _prepared(plan, adapter):
    if adapter in ('structure_d1', 'token_c2'): return plan['samples']['rows']
    if adapter == 'task_b2_p00':
        return [r for r in plan['samples']['generation_rows'] if r['policy_id'] == 'P00']
    if adapter == 'task_b4': return plan['samples']['generation_rows']
    return plan['samples']['rows']


def _sample_payload(plan, row, prepared, adapter):
    from ..tasks.registry import get_task
    p = prepared or {}
    task_id = row.get('task_id', plan.get('task_id'))
    task_version = row.get('task_version', plan.get('task_version'))
    if adapter in ('structure_d1', 'token_c2'):
        ref = {'text': row.get('reference_text'), 'reference_source': row.get('reference_source')}
    else:
        ref = get_task(task_id, task_version).reference(row)
    reference_text = p.get('reference_text', ref.get('reference_text', ref.get('text')))
    policy = p.get('policy_id', row.get('policy_id'))
    if adapter == 'task_b4': policy = 'FIXED'
    fields = ('task_id', 'task_version', 'sample_id', 'source_id', 'dataset_identity', 'split',
              'policy_id', 'policy_identity', 'prompt', 'reference_text', 'prompt_hash', 'reference_hash',
              'sequence_identity', 'input_ids', 'generation_input_ids', 'prompt_token_boundary',
              'tokenizer_identity', 'token_id_space', 'special_token_mask', 'loss_mask', 'attention_mask',
              'offsets_codepoints', 'offsets_utf8_bytes', 'reference_role', 'reference_source',
              'field_token_coverage', 'reference_evidence_span', 'gold_choice_index', 'sample_identity',
              'source_bundle_identity', 'source_plan_identity', 'source_scorer_identity', 'probe_input_identity',
              'task_prompt_hash', 'document_hash', 'value_spans', 'targets', 'full_vocab_target_positions')
    saved = {k: p[k] for k in fields if k in p}
    if not saved and adapter in ('structure_d1', 'token_c2'): saved = {k: row[k] for k in fields if k in row}
    return {'task_id': task_id, 'task_version': task_version, 'sample_id': row['sample_id'],
            'source_id': p.get('source_id', row.get('source_id')), 'policy_id': policy,
            'split': row.get('split', row.get('usage_partition')),
            'dataset_identity': row.get('dataset_identity', plan.get('data', {}).get('rows_identity')),
            'document_hash': row.get('document_hash', row.get('source_text_hash')),
            'gold_identity': digest(row['gold']) if 'gold' in row else None,
            'prompt_hash': p.get('prompt_hash', row.get('prompt_hash', digest(row['prompt']) if 'prompt' in row else None)),
            'reference_hash': p.get('reference_hash', row.get('reference_hash', digest(reference_text) if reference_text is not None else None)),
            'reference_source': p.get('reference_source', row.get('reference_source', ref.get('reference_source'))),
            'reference_role': p.get('reference_role', row.get('reference_role')),
            'sequence_identity': p.get('sequence_identity', row.get('sequence_identity')),
            'tokenizer_identity': plan['samples']['tokenizer_identity'],
            'token_id_space': plan['samples']['token_id_space'],
            'prepared': saved,
            'document': row.get('document'), 'question': row.get('question'),
            'gold': row.get('gold'), 'reference_text': reference_text,
            'historical_outcome': adapter.startswith('task_'),
            'acquisition_source_identity': digest(plan.get('source_identity', {}))}


def _envelope(spec, root, plan, manifest, locator):
    adapter = spec['adapter']; raw_sha = locator['manifest_sha256']
    acquisition_id = digest({'plan_identity': plan['plan_identity'], 'raw_manifest_sha256': raw_sha})
    source_id = digest({'adapter_family': adapter, 'acquisition_id': acquisition_id})
    analyzer = _analysis_identity(adapter, spec); analysis_id = digest({'acquisition_id': acquisition_id, **analyzer})
    out = {'source_id': source_id, 'source_alias': spec['source_id'], 'content_identity': raw_sha,
           'acquisition_id': acquisition_id, 'locator': locator,
           'payload': {'adapter': adapter, 'scope': manifest['scope'], 'plan_identity': plan['plan_identity'],
                       'verification': 'RAW_MANIFEST_AND_PLAN_VERIFIED', 'new_gpu_measurements': 0},
           'acquisitions': [{'id': acquisition_id, 'payload': {'plan_identity': plan['plan_identity'],
                'raw_manifest_sha256': raw_sha, 'task_profile': plan['task'],
                'source_identity': digest(plan.get('source_identity', {})),
                'evidence_kind': plan.get('evidence_kind'), 'new_measurement': False}}],
           'analyses': [{'id': analysis_id, 'acquisition_id': acquisition_id, 'payload': analyzer}],
           'samples': [], 'executions': [], 'measurements': [], 'evidence': [], 'relations': []}
    original = plan['data']['rows'] if 'rows' in plan.get('data', {}) else plan['samples']['rows']
    prepared = _prepared(plan, adapter); by = {r['sample_id']: r for r in prepared}
    # A selected historical sample slice is explicit and never relabelled full.
    selected = spec.get('sample_ids')
    if selected is not None:
        require(isinstance(selected, list) and selected and len(set(selected)) == len(selected), 'Invalid E1 sample selection')
        require(set(selected) <= {r['sample_id'] for r in original}, 'Unknown E1 source sample')
    sample_map = {}
    for row in original:
        if selected is not None and row['sample_id'] not in selected: continue
        if adapter == 'task_b2_p00' and row['split'] not in spec.get('splits', ['EVAL_LOCKED']): continue
        if adapter == 'task_b4' and row['split'] not in spec.get('splits', ['EVAL_LOCKED']): continue
        if adapter == 'task_b3' and row['usage_partition'] not in spec.get('partitions', ['REGRESSION']): continue
        payload = _sample_payload(plan, row, by.get(row['sample_id']), adapter)
        sid = digest({'acquisition_id': acquisition_id, 'task_id': payload['task_id'], 'sample_id': row['sample_id'], 'policy_id': payload['policy_id']})
        require(row['sample_id'] not in sample_map, 'Duplicate E1 source sample')
        sample_map[row['sample_id']] = sid
        out['samples'].append({'id': sid, 'acquisition_id': acquisition_id, 'task_id': payload['task_id'], 'sample_id': row['sample_id'], 'payload': payload})
    progress = read(root / 'progress.json'); execution_map = {}
    for key, job in progress.get('jobs', {}).items():
        directory = root / safe_name(job['directory']); path = directory / 'execution.json'
        value = read(path) if path.is_file() else {'job_id': key, 'status': job['status']}
        arm_id = value.get('arm_id', key.rsplit('-', 1)[-1]); eid = digest({'acquisition_id': acquisition_id, 'job': key, 'execution': value.get('execution_id')})
        execution_map[key] = eid
        # Select identity/receipt fields only: old runtime/model paths stay private.
        keep = {k: value[k] for k in ('execution_id', 'run_id', 'attempt_id', 'arm_id', 'phase', 'pass_id', 'status', 'evidence_kind', 'execution_mode', 'effective_weight_dtype', 'artifact_identity', 'recipe_identity', 'runtime_identity', 'tokenizer_identity', 'model_id', 'model_revision', 'backend') if k in value}
        out['executions'].append({'id': eid, 'acquisition_id': acquisition_id, 'payload': keep | {'job_id': key, 'arm_id': arm_id, 'job_status': job['status']}})
    for name, entry in locator['files'].items():
        out['evidence'].append({'id': digest({'acquisition_id': acquisition_id, 'path': name, 'sha256': entry['sha256']}),
            'relative_path': name, 'bytes': entry['bytes'], 'sha256': entry['sha256'],
            'payload': {'acquisition_id': acquisition_id, 'source_id': source_id, 'kind': 'RAW_MEMBER'}})
    return out, sample_map, execution_map, analysis_id


def _emit(out, analysis_id, sample_map, metric, value, status, scope, coordinates, payload, execution_id=None, evidence_prefix=None):
    require(value is None or type(value) in (int, float, bool), 'E1 scalar metric required')
    import math
    if type(value) in (int, float): require(math.isfinite(value), 'E1 nonfinite metric')
    sid = coordinates.get('sample_id'); sample_pk = sample_map.get(sid)
    if sid is not None and sample_pk is None: return
    row = {'analysis_id': analysis_id, 'sample_id': sample_pk, 'execution_id': execution_id,
           'task_id': coordinates.get('task_id'), 'metric': metric, 'status': status,
           'value': value, 'scope': scope, 'coordinates': coordinates, 'payload': payload}
    evidence = {r['relative_path']: r['id'] for r in out['evidence']}
    paths = {'PLAN_FROZEN.json'}
    def visit(value, observations_path=None):
        if not isinstance(value, dict): return
        observations_path = value.get('observations_path', value.get('source_observations_path', observations_path))
        if observations_path: paths.add(observations_path)
        for key in ('raw_reference', 'request_reference'):
            if isinstance(value.get(key), str): paths.add(value[key].split('#', 1)[0])
        numeric = value.get('numeric')
        if isinstance(numeric, dict) and observations_path:
            paths.add((Path(observations_path).parent / safe_name(numeric['path'])).as_posix())
        for child in value.values():
            if isinstance(child, dict): visit(child, observations_path)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, dict): visit(item, observations_path)
    visit(payload)
    linked=[]
    for p in sorted(paths):
        resolved=p if p in evidence else safe_name(evidence_prefix)+'/'+safe_name(p) if evidence_prefix else p
        if resolved in evidence and evidence[resolved] not in linked:linked.append(evidence[resolved])
    row['evidence_ids'] = linked
    row['id'] = digest({'analysis_id': analysis_id, 'metric': metric, 'scope': scope, 'coordinates': coordinates})
    out['measurements'].append(row)


def _comparison_rows(out, analysis_id, sample_map, path, scope):
    if not path.is_file(): return
    for row in jsonl(path):
        coordinates = dict(row.get('identity', {}))
        for key in ('sample_id', 'task_id', 'arm_id', 'pass_id', 'kind', 'token_position', 'reference_pass', 'candidate_pass'):
            if key in row: coordinates[key] = row[key]
        coordinates['pass_id'] = coordinates.get('pass_id', 'ON' if scope in ('structure_pair', 'projection_pair', 'same_arm_AV', 'evidence_mass', 'token_pair') else None)
        metrics = row.get('metrics')
        if metrics is None:
            names = ('mass', 'maximum_absolute_difference', 'absolute_error_nat')
            metrics = {k: row[k] for k in names if k in row}
        if not metrics: metrics = {'unavailable': None}
        for name, value in metrics.items():
            if value is None or type(value) in (int, float, bool):
                _emit(out, analysis_id, sample_map, (str(row.get('kind', scope)) + '.' + name), value,
                      row.get('status', 'UNKNOWN'), scope, coordinates, row)


def _tokens(out, analysis_id, sample_map, root, work):
    from .tokens import export_tokens, compare_tokens
    manifest = work / 'sources.json'
    write(manifest, {'schema_version': 1, 'kind': 'token_input_sources', 'sources': [
        {'source_id': 'e1-source', 'adapter': 'token_probe_raw_v1', 'path': str(root)}]})
    export_tokens(manifest, work / 'view')
    arms = [a['arm_id'] for a in read(root / 'PLAN_FROZEN.json')['arms']]
    compare_tokens(work / 'view', arms[0], arms[1], work / 'comparison')
    _token_metrics(out, analysis_id, sample_map, work / 'comparison' / 'token-comparisons.jsonl', work / 'view')


def _token_metrics(out, analysis_id, sample_map, path, view_root=None, evidence_prefix=None, *, pass_id=None):
    records = {r['record_id']: r for r in jsonl(view_root / 'token_records.jsonl')} if view_root is not None else {}
    for row in jsonl(path):
        coordinates = dict(row.get('identity', {})); coordinates.update(pass_id=pass_id or ('ON' if out['payload']['adapter'] == 'structure_d1' else 'TOKEN_ONLY'))
        coordinates.update({k: row[k] for k in ('token_position', 'target_token_id') if k in row})
        # Common C1 owns metric definitions; retain its complete result for drilldown.
        metrics = row.get('metrics') or {'unavailable': None}
        full = metrics.get('full_distribution')
        if isinstance(full, dict):
            full_names = ('kl_ref_candidate_nat', 'js_nat', 'entropy_reference_nat', 'entropy_candidate_nat')
            metrics = dict(metrics, **{'full_distribution.' + k: full.get(k) for k in full_names})
        top = metrics.get('top_k')
        if isinstance(top, dict):
            metrics = dict(metrics, **{'top_' + k + '_overlap_fraction': v.get('overlap_fraction') for k, v in top.items()})
        raw = []
        for key in ('left_record_id', 'right_record_id'):
            value = records.get(row.get(key))
            if value:
                raw.append({k: value[k] for k in ('observation_id', 'execution_id', 'source_observations_path', 'source_observation', 'source_full_observation', 'raw_reference') if k in value})
        payload = dict(row, raw_records=raw)
        for name, value in metrics.items():
            if value is None or type(value) in (int, float, bool):
                status = full.get('status', row['status']) if name.startswith('full_distribution.') and isinstance(full, dict) else row['status']
                metric_payload = payload
                if pass_id == 'NATIVE_CHOICE_LIKELIHOOD' and value is None and (
                        name.startswith(('full_distribution.', 'top_')) or name in ('raw_top1_match',
                        'reference_target_rank', 'candidate_target_rank', 'reference_target_margin', 'candidate_target_margin')):
                    status = 'NOT_COLLECTED'
                    metric_payload = dict(payload, unavailable_reason='FULL_VOCABULARY_NOT_COLLECTED')
                _emit(out, analysis_id, sample_map, 'token.' + name, value, status, 'token_pair', coordinates, metric_payload, evidence_prefix=evidence_prefix)


def _generation(out, spec, root, plan, sample_map, execution_map, analysis_id):
    from .generation_sources import collect_generations
    adapter = 'bench05b2' if spec['adapter'] == 'task_b2_p00' else 'bench05b4'
    source_spec = {'source_id': 'e1-source', 'adapter': adapter, 'path': str(root),
                   'policies': ['P00'] if adapter == 'bench05b2' else ['FIXED'],
                   'splits': spec.get('splits', ['EVAL_LOCKED'])}
    if 'sample_ids' in spec: source_spec['sample_ids'] = spec['sample_ids']
    rows = collect_generations(source_spec, plan, out['source_id'])
    for row in rows:
        coords = {k: row.get(k) for k in ('task_id', 'task_version', 'sample_id', 'policy_id', 'arm_id', 'request_id')}
        score = row['task_score']; metrics = {k: v for k, v in score.items() if v is None or type(v) in (int, float, bool)}
        for family in ('field_correct', 'evidence_correct'):
            metrics.update({family + '.' + k: v for k, v in score.get(family, {}).items()})
        metrics.update(request_latency_http=row['request_latency_http'], request_latency_task=row['request_latency_task'],
                       actual_output_tokens=row['actual_output_token_count'], transport_complete=row['transport_complete'])
        evidence = {k: row.get(k) for k in ('output_text', 'output_token_ids', 'termination', 'finish_reason', 'raw_reference', 'task_score', 'identity', 'reference_source')}
        for metric, value in metrics.items():
            missing = row['record_status'] == 'NOT_RUN'
            _emit(out, analysis_id, sample_map, 'task.' + metric, None if missing else value,
                  'NOT_RUN' if missing else score.get('status', 'UNKNOWN'), 'historical_task', coords,
                  evidence, execution_map.get('serve-' + row['arm_id']))


def _arc(out, spec, root, plan, sample_map, execution_map, analysis_id, work):
    from ..suites.bench05b3_replay import analyze
    work.mkdir(); analyze(root, work)
    by_sample = {r['sample_id']: r for r in plan['data']['rows']}
    progress = read(root / 'progress.json')
    for arm in plan['arms']:
        aid = arm['arm_id']
        response_path = progress['jobs'].get('serve-' + aid, {}).get('directory')
        response_path = response_path + '/http-responses.jsonl' if response_path is not None else None
        for row in jsonl(work / (aid + '-document-scores.jsonl')):
            coords = {'task_id': plan['task_id'], 'task_version': plan['task_version'], 'sample_id': row['sample_id'], 'arm_id': aid}
            for metric in ('acc', 'acc_norm', 'gold_mean_nll_nat', 'gold_margin_raw', 'gold_margin_norm', 'document_choice_http_sum_seconds', 'document_serial_elapsed_seconds'):
                _emit(out, analysis_id, sample_map, 'task.' + metric, row.get(metric), row['status'], 'historical_task', coords,
                      dict(row, raw_reference=response_path + '#sample_id=' + row['sample_id'] if response_path else None), execution_map.get('serve-' + aid))
        for row in jsonl(work / (aid + '-choice-scores.jsonl')):
            source = by_sample[row['sample_id']]; choice = row['choice_index']
            coords = {'task_id': plan['task_id'], 'task_version': plan['task_version'], 'sample_id': row['sample_id'],
                      'arm_id': aid, 'choice_index': choice, 'gold_choice': choice == source['gold'],
                      'reference_scope': 'GOLD_CHOICE' if choice == source['gold'] else 'NON_GOLD_CHOICE', 'request_id': row['request_id']}
            metrics = {'choice_loglikelihood': row.get('choice_loglikelihood'),
                       'continuation_token_count': row.get('continuation_token_count'),
                       'request_latency_http': row.get('request_latency_http')}
            for metric, value in metrics.items():
                _emit(out, analysis_id, sample_map, 'native_choice.' + metric, value, row['status'], 'native_choice', coords,
                      dict(row, raw_reference=response_path + '#request_id=' + row['request_id'] if response_path else None), execution_map.get('serve-' + aid))
        for row in jsonl(work / (aid + '-native-tokens.jsonl')):
            coords = dict(row, task_id=plan['task_id'], task_version=plan['task_version'], arm_id=aid)
            gold = row['choice_index'] == by_sample[row['sample_id']]['gold']
            coords.update(gold_choice=gold, reference_scope='GOLD_CHOICE' if gold else 'NON_GOLD_CHOICE')
            _emit(out, analysis_id, sample_map, 'native_token.nll_nat', row['nll_nat'], 'MEASURED', 'native_likelihood', coords,
                  {'request_reference': response_path + '#request_id=' + row['request_id'] if response_path else None, 'raw_token': row}, execution_map.get('serve-' + aid))


def load_source(spec, manifest_parent):
    """Read and re-score one explicitly selected, versioned local source.

    The returned dictionaries are canonical data.  Temporary replay outputs are
    not locators and are removed after the common scalar values are materialized.
    Original observations and artifacts are never mutated.
    """
    if isinstance(spec, dict) and spec.get('adapter') == 'suite_e3':
        from .suite_results import load_index_source
        return load_index_source(spec, manifest_parent)
    require(isinstance(spec, dict) and spec.get('adapter') in ADAPTERS, 'Unsupported E1 source adapter')
    require(isinstance(spec.get('source_id'), str) and spec['source_id'], 'Explicit E1 source namespace required')
    with _root(spec, manifest_parent) as (root, plan, manifest, locator), tempfile.TemporaryDirectory(prefix='inference-kit-index-') as tmp:
        out, sample_map, execution_map, analysis_id = _envelope(spec, root, plan, manifest, locator)
        integration = spec.get('integration_status', 'RAW_RECOMPUTED')
        require(integration in ('RAW_RECOMPUTED', 'INVALID_ACQUISITION', 'NEEDS_NEW_ACQUISITION', 'EVIDENCE_GAP'), 'Unknown E1 integration gate')
        if integration != 'RAW_RECOMPUTED' or spec.get('quarantine_reason'):
            reason = spec.get('quarantine_reason')
            require(isinstance(reason, str) and reason, 'Quarantined source requires explicit reason')
            out['payload'].update(integration_status=integration if integration != 'RAW_RECOMPUTED' else 'EVIDENCE_GAP', quarantine_reason=reason)
            out['verification'] = {'status': out['payload']['integration_status'], 'reason': reason,
                'raw_members': len(locator['files']), 'selected_samples': len(out['samples']),
                'measurements': 0, 'new_gpu_measurements': 0}
            return out
        work = Path(tmp)
        if spec['adapter'] == 'structure_d1':
            from .structure import replay_structure
            replay_structure(root, work / 'replay')
            for name, scope in (('structure-comparisons.jsonl', 'structure_pair'), ('output-projection-comparisons.jsonl', 'projection_pair'),
                ('reconstructed-native-AV.jsonl', 'same_arm_AV'), ('evidence-content-mass.jsonl', 'evidence_mass'),
                ('observer-controls.jsonl', 'observer_control'), ('scalar-full-consistency.jsonl', 'scalar_full_control')):
                _comparison_rows(out, analysis_id, sample_map, work / 'replay' / name, scope)
            token_file = work / 'replay' / 'token-metrics' / 'token-comparisons.jsonl'
            if token_file.is_file(): _token_metrics(out, analysis_id, sample_map, token_file, work / 'replay' / 'token-view')
        elif spec['adapter'] == 'token_c2': _tokens(out, analysis_id, sample_map, root, work)
        elif spec['adapter'] in ('task_b2_p00', 'task_b4'): _generation(out, spec, root, plan, sample_map, execution_map, analysis_id)
        else: _arc(out, spec, root, plan, sample_map, execution_map, analysis_id, work / 'arc')
        ids = [r['id'] for r in out['measurements']]
        require(len(ids) == len(set(ids)), 'Duplicate normalized E1 measurement')
        out['verification'] = {'status': 'RAW_RECOMPUTED', 'raw_members': len(locator['files']),
            'selected_samples': len(out['samples']), 'measurements': len(ids), 'new_gpu_measurements': 0,
            'new_task_quality': 'NOT_EVALUATED_HISTORICAL_RESCORING_ONLY'}
        return out
