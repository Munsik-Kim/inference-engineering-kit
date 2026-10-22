"""B2 raw SSE scoring and document-paired declared policy contrasts, CPU only."""
from collections import Counter
import json
from pathlib import Path
import statistics
from ..core.io import read, write, sha, digest, safe_name, require
from ..core.evidence import open_bundle
from ..core.state import derived_output, preserve_failure
from ..tasks import tech_record as task
from ..tasks import contract_policy as policy
from .bench05b1_replay import jsonl, generation_from_raw, probe_metrics, memory
from .bench05a_replay import csv_output
from .bench05b2_plan import validate_prepared

METRICS = ('transport_complete', 'json_valid', 'strict_schema_valid', 'status_enum_valid',
 'evidence_required_keys_present', 'evidence_value_type_valid', 'evidence_line_id_exists',
 'field_id', 'field_temperature', 'field_status', 'evidence_temperature', 'evidence_status',
 'core_values_correct', 'strict_task_success', 'strict_task_success_within_10s')


def metric(score, name):
    aliases = {'strict_schema_valid': 'schema_valid', 'strict_task_success': 'task_success',
               'strict_task_success_within_10s': 'task_success_within_10s'}
    if name in aliases:
        return score[aliases[name]]
    if name in ('field_id', 'field_temperature', 'field_status'):
        return score['field_correct'][{'field_id': 'experiment_id', 'field_temperature': 'temperature_c', 'field_status': 'status'}[name]]
    if name.startswith('evidence_') and name in ('evidence_temperature', 'evidence_status'):
        return score['evidence_correct']['temperature_c' if name == 'evidence_temperature' else 'status']
    return score.get(name, score['diagnostics'].get(name))


def aggregate(samples, raw):
    require(len({r['sample_id'] for r in raw}) == len(raw), 'Duplicate B2 document in policy cell')
    expected = {s['sample_id'] for s in samples}; require(all(r['sample_id'] in expected for r in raw), 'Unknown B2 generated document')
    by_id = {r['sample_id']: r for r in raw}
    scores = [policy.diagnostics(s, by_id.get(s['sample_id'])) for s in samples]
    observed = [s for s in scores if s['status'] == 'SCORED']
    result = {'expected': len(samples), 'submitted': len(observed), 'not_run': len(samples) - len(observed),
      'complete': len(samples) == len(observed), 'scope': 'synthetic_locked_contract_pilot',
      'metrics': {m: {'correct': sum(metric(s, m) is True for s in observed), 'planned_denominator': len(samples),
        'observed_denominator': len(observed)} for m in METRICS},
      'error_combinations_overlap': dict(Counter('+'.join(s['error_combination']) or 'NO_ADDITIONAL_DIAGNOSTIC_ERROR' for s in observed)),
      'termination_counts': dict(Counter(s['termination'] for s in observed))}
    for family in ('unsupported_value', 'omission', 'false_abstention'):
        result[family] = {field: sum(s.get(family, {}).get(field) is True for s in observed) for field in ('temperature_c', 'status')}
    return scores, result


def compatible(left, right, mode):
    for key in ('sample_id', 'sample_identity', 'document_hash', 'gold_identity', 'task_id', 'task_version',
                'scorer_identity', 'runtime_identity', 'tokenizer_identity', 'model_id', 'model_revision', 'generation_policy_identity'):
        require(left[key] == right[key], 'B2 incompatible ' + key)
    for value in (left, right):
        require(value['policy_identity'] == policy.policy_record(value['policy_id'])['policy_identity'], 'B2 undeclared policy identity')
    if left['policy_id'] == right['policy_id']:
        require(mode == 'same_prompt_weight_intervention' and left['prompt_hash'] == right['prompt_hash'], 'B2 same-policy prompt mismatch')
    else:
        require(mode == 'declared_prompt_intervention', 'Different prompts require declared intervention')
        require(left['arm_id'] == right['arm_id'] and left['artifact_identity'] == right['artifact_identity'], 'Prompt effect must hold artifact fixed')
    return True


def paired(left, right, mode):
    require(len({r['sample_id'] for r in left}) == len(left) and len({r['sample_id'] for r in right}) == len(right), 'Duplicate B2 comparison sample')
    a, b = {r['sample_id']: r for r in left}, {r['sample_id']: r for r in right}
    require(a.keys() == b.keys(), 'B2 comparison document set differs')
    for sid in a:
        compatible(a[sid]['comparison_identity'], b[sid]['comparison_identity'], mode)
    complete = all(a[s]['status'] == b[s]['status'] == 'SCORED' for s in a)
    result = {'status': 'COMPLETE_DOCUMENT_PAIRED_DESCRIPTIVE' if complete else 'PARTIAL_OR_NOT_RUN',
      'compare_mode': mode, 'planned_documents': len(a),
      'observed_pairs': sum(a[s]['status'] == b[s]['status'] == 'SCORED' for s in a), 'metrics': {}}
    if not complete:
        return result
    for name in METRICS:
        x = [metric(a[s], name) is True for s in a]; y = [metric(b[s], name) is True for s in a]
        delta = sum(y) - sum(x)
        result['metrics'][name] = {'left_correct': sum(x), 'right_correct': sum(y), 'delta_count': delta,
          'delta_fraction': delta / len(a), 'denominator_documents': len(a),
          'gained': sum(not u and v for u, v in zip(x, y)), 'lost': sum(u and not v for u, v in zip(x, y)),
          'correct_retained': sum(u and v for u, v in zip(x, y)), 'incorrect_retained': sum(not u and not v for u, v in zip(x, y))}
    return result


def contrasts(cells):
    result = {'scope': 'same_24_documents_point_descriptive_no_equivalence_claim', 'within_weight': {}, 'within_policy': {}}
    definitions = {'E_S0': ('P00', 'P10'), 'E_S1': ('P01', 'P11'), 'S_E0': ('P00', 'P01'),
                   'S_E1': ('P10', 'P11'), 'combined': ('P00', 'P11')}
    for arm in ('bf16', 'rtn'):
        values = {name: paired(cells[arm, a], cells[arm, b], 'declared_prompt_intervention') for name, (a, b) in definitions.items()}
        complete = all(v['status'] == 'COMPLETE_DOCUMENT_PAIRED_DESCRIPTIVE' for v in values.values())
        interaction = {'status': 'COMPLETE_DOCUMENT_PAIRED_DESCRIPTIVE' if complete else 'PARTIAL_OR_NOT_RUN', 'metrics': {}}
        if complete:
            # All four outcomes use the same source IDs; no independent interval subtraction.
            for m in METRICS:
                delta = values['E_S1']['metrics'][m]['delta_count'] - values['E_S0']['metrics'][m]['delta_count']
                interaction['metrics'][m] = {'delta_count': delta, 'delta_fraction': delta / 24, 'denominator_documents': 24,
                                             'formula': 'P11-P10-P01+P00'}
        result['within_weight'][arm] = values | {'interaction': interaction}
    for name in policy.POLICIES:
        result['within_policy'][name] = paired(cells['bf16', name], cells['rtn', name], 'same_prompt_weight_intervention')
    return result


def verify_raw(root):
    manifest = read(root / 'RAW_MANIFEST.json')
    require(manifest['scope'] == 'bench05b2_contract_raw', 'Wrong B2 raw scope')
    for name, info in manifest['files'].items():
        p = root / safe_name(name)
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()), 'Unsafe/missing B2 raw')
        require(p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], 'B2 raw identity changed: ' + name)
    return len(manifest['files'])


def analyze(root, out):
    plan = read(root / 'PLAN_FROZEN.json'); frozen = dict(plan); identity = frozen.pop('plan_identity')
    require(digest(frozen) == identity, 'B2 plan integrity changed')
    require(plan['scorer_identity'] == sha(Path(task.__file__)), 'B2 unchanged scorer bytes required')
    require(plan['policies'] == {n: policy.policy_record(n) for n in policy.POLICIES}, 'B2 policy meanings changed')
    validate_prepared(plan['samples'], plan['data']); verified = verify_raw(root)
    progress = read(root / 'progress.json'); data = plan['data']['rows']; source = {s['sample_id']: s for s in data}
    prepared = {(r['sample_id'], r['policy_id']): r for r in plan['samples']['generation_rows']}
    cells = {}; arms = {}; flat = []; raw_lookup = {}
    for arm in plan['arms']:
        aid = arm['arm_id']; job = progress['jobs'].get('serve-' + aid); raw = []; directory = None
        if job:
            directory = root / safe_name(job['directory']); path = directory / 'generations.jsonl'
            if path.exists():
                seen = set()
                for record in jsonl(path):
                    key = record['sample_id'], record['policy_id']
                    require(key in prepared and key not in seen, 'B2 unknown/duplicate source-policy request'); seen.add(key)
                    row = prepared[key]
                    require(record['policy_identity'] == row['policy_identity'] and record['prepared_identity'] == digest(row), 'B2 policy/token manifest changed')
                    require(record['scorer_identity'] == plan['scorer_identity'] and record['runtime_identity'] == plan['serving_runtime_identity'], 'B2 scoring/runtime changed')
                    raw.append(generation_from_raw(record, row, arm, plan))
                require([(r['sample_id'], r['policy_id']) for r in raw] == list(prepared)[:len(raw)], 'B2 actual request order differs')
        require(len({r['request_id'] for r in raw}) == len(raw), 'B2 request IDs collide')
        raw_lookup[aid] = {(r['sample_id'], r['policy_id']): r for r in raw}
        values = {}
        for name in policy.POLICIES:
            splits = {}
            for split in ('SMOKE', 'EVAL_LOCKED'):
                samples = [s for s in data if s['split'] == split]
                observed = [r for r in raw if r['split'] == split and r['policy_id'] == name]
                scores, summary = aggregate(samples, observed)
                for sample, value in zip(samples, scores):
                    row = prepared[sample['sample_id'], name]
                    value['comparison_identity'] = {'sample_id': sample['sample_id'], 'sample_identity': sample['sample_identity'],
                      'document_hash': sample['document_hash'], 'gold_identity': digest(sample['gold']), 'task_id': task.TASK,
                      'task_version': task.VERSION, 'scorer_identity': plan['scorer_identity'], 'runtime_identity': plan['serving_runtime_identity'],
                      'tokenizer_identity': plan['samples']['tokenizer_identity'], 'model_id':plan['pair']['model_id'], 'model_revision':plan['pair']['revision'],
                      'generation_policy_identity':digest({'serving':plan['suite']['serving'],'generation':plan['suite']['generation']}), 'policy_id': name, 'policy_identity': row['policy_identity'],
                      'prompt_hash': row['prompt_hash'], 'arm_id': aid, 'artifact_identity': arm['artifact']['artifact_identity']}
                if split == 'EVAL_LOCKED':
                    cells[aid, name] = scores
                with (out / f'{aid}-{name}-{split}-scores.jsonl').open('x', encoding='utf-8') as f:
                    for value in scores:
                        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                splits[split] = summary
            lat = [r['request_latency_http'] for r in raw if r['split'] == 'EVAL_LOCKED' and r['policy_id'] == name and r['request_latency_http'] is not None]
            tokens = [r['usage']['completion_tokens'] for r in raw if r['split'] == 'EVAL_LOCKED' and r['policy_id'] == name and r.get('usage')]
            values[name] = {'splits': splits, 'latency_http_seconds': {'n': len(lat), 'p50': statistics.median(lat) if lat else None, 'max': max(lat) if lat else None},
                            'actual_output_tokens': sum(tokens), 'transport_errors': sum(not r['transport_complete'] for r in raw if r['split'] == 'EVAL_LOCKED' and r['policy_id'] == name)}
            flat.append({'arm': aid, 'policy': name, 'submitted': splits['EVAL_LOCKED']['submitted'],
                         **{m: splits['EVAL_LOCKED']['metrics'][m]['correct'] for m in METRICS},
                         'http_p50_seconds': statistics.median(lat) if lat else None, 'http_max_seconds': max(lat) if lat else None})
            values[name]['breakdown'] = {}
            for family, labels in (('language', ('en', 'ko')), ('case_type', task.CASES)):
                values[name]['breakdown'][family] = {label: aggregate([s for s in data if s['split'] == 'EVAL_LOCKED' and s[family] == label],
                      [r for r in raw if r['split'] == 'EVAL_LOCKED' and r['policy_id'] == name and source[r['sample_id']][family] == label])[1] for label in labels}
        arms[aid] = {'artifact_identity': arm['artifact']['artifact_identity'], 'policies': values,
                     'execution_status': job['status'] if job else 'NOT_RUN', 'memory_shared_across_policies': memory(directory) if directory else {'observed_max': None, 'scope': 'whole_device_mib'},
                     'generation_posts': len(raw), 'actual_output_tokens_all': sum(r['usage']['completion_tokens'] for r in raw if r.get('usage'))}
    comparisons = contrasts(cells); write(out / 'paired-contrasts.json', comparisons); csv_output(out / 'cells.csv', flat)
    probes = {}
    for name in ('P00', 'P11'):
        entry = progress.get('diagnostics', {}).get(name)
        if entry:
            subroot = root / safe_name(entry['directory']); subplan = read(subroot / 'PLAN_FROZEN.json')
            from .bench05b2_run import diagnostic_plan
            require(subplan == diagnostic_plan(plan, name), 'B2 diagnostic plan identity changed')
            # The inherited comparator requires exact prefix/key axes; never cross P00/P11.
            probes[name] = probe_metrics(subroot, subplan, read(subroot / 'progress.json'))
        else:
            probes[name] = {'status': 'NOT_RUN', 'paired_samples': 0, 'records': [], 'executions': {}}
    write(out / 'probe-comparisons.json', probes)
    linked = []
    for sample in [s for s in data if s['split'] == 'EVAL_LOCKED']:
        sid = sample['sample_id']; entries = {}
        for name in policy.POLICIES:
            scores = {a: next(s for s in cells[a, name] if s['sample_id'] == sid) for a in ('bf16', 'rtn')}
            raw_links = {a: {'request_id': raw_lookup[a].get((sid, name), {}).get('request_id'),
                           'raw': progress['jobs'].get('serve-' + a, {}).get('directory', 'NOT_RUN') + '/generations.jsonl'} for a in ('bf16', 'rtn')}
            records = [r for r in probes.get(name, {}).get('records', []) if r['sample_id'] == sid]
            token = [r for r in records if r['kind'] == 'target_logprob']; attention = [r for r in records if r['kind'] == 'attention_row']; av = [r for r in records if r['kind'] == 'attention_output']
            mean = lambda rs, key: statistics.fmean(r[key] for r in rs) if rs else None
            diag = {'status': 'MEASURED_SAME_POLICY_DIAGNOSTIC_ONLY' if token else 'NOT_RUN' if name in ('P00', 'P11') and sid in plan['samples']['probe_selection']['sample_ids'] else 'NOT_SELECTED',
              'target_positions': len(token), 'attention_rows': len(attention), 'reference_nll_nat': mean(token, 'reference_nll_nat'),
              'candidate_nll_nat': mean(token, 'candidate_nll_nat'), 'delta_nll_nat': mean(token, 'delta_nll_nat'),
              'attention_js_nat': mean(attention, 'js_nat'), 'attention_output_nmse': mean(av, 'nmse'), 'attention_output_cosine': mean(av, 'cosine'),
              'raw_prefix': 'diagnostics/' + name + '/', 'comparison': 'probe-comparisons.json',
              'observation_ids': [r['reference_observation'] for r in records] + [r['candidate_observation'] for r in records],
              'execution_ids': {a: v['execution_id'] for a, v in probes.get(name, {}).get('executions', {}).items()}}
            row = prepared[sid, name]
            entries[name] = {'scores': scores, 'raw_links': raw_links, 'teacher_forced_probe': diag,
                             'prompt_hash': row['prompt_hash'], 'reference_hash': row['reference_hash'], 'tokenizer_identity': plan['samples']['tokenizer_identity']}
        linked.append({'sample_id': sid, 'sample_identity': sample['sample_identity'], 'language': sample['language'], 'case_type': sample['case_type'], 'policies': entries})
    with (out / 'sample-linked.jsonl').open('x', encoding='utf-8') as f:
        for value in linked:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    complete = all(v['policies'][n]['splits']['EVAL_LOCKED']['complete'] for v in arms.values() for n in policy.POLICIES)
    summary = {'schema_version': 1, 'task': 'BENCH05-B2-CONTRACT', 'task_id': task.TASK, 'task_version': task.VERSION, 'scorer': task.SCORER,
      'scope': 'synthetic_locked_contract_pilot', 'evidence_kind': plan.get('evidence_kind', 'UNKNOWN'), 'arms': arms,
      'counts': plan['counts'], 'raw_files_verified': verified, 'full_eval_coverage': complete,
      'ENGINEERING_FLOW': 'SYNTHETIC_CONTRACT_ONLY' if plan.get('evidence_kind')=='synthetic_contract' else 'NATURAL_GENERATION_AND_CPU_REPLAY_COMPLETE' if complete else 'PARTIAL_OR_NOT_RUN',
      'PROMPT_CONTRACT_EFFECT': 'OBSERVED_FIXED_POLICY_CONTRASTS_ONLY' if complete else 'INCONCLUSIVE_PARTIAL',
      'QUANTIZATION_EFFECT': 'WITHIN_POLICY_DESCRIPTIVE_NO_QUALITY_EQUIVALENCE',
      'PROBE_LINK': {n: {'status': v['status'], 'paired_samples': v['paired_samples'],
          'token_positions': sum(r['kind'] == 'target_logprob' for r in v['records']),
          'attention_rows': sum(r['kind'] == 'attention_row' for r in v['records'])} for n, v in probes.items()},
      'PERFORMANCE': 'DESCRIPTIVE_C1_SINGLE_SERVER_PER_WEIGHT_ONLY', 'DEPLOYMENT_CUSTOMER': 'NOT_VALIDATED',
      'retained_research': 'HOLD_CLIPPING_VARIANT; prior quality INCONCLUSIVE; worker native exit UNKNOWN',
      'diagnostic_deployment_equivalence': 'NOT_CLAIMED', 'B2_REMOTE_CI': 'NOT_RUN', 'replay_gpu_actions': 0,
      'budget': read(root / 'budget-continuation.json') if (root / 'budget-continuation.json').exists() else None,
      'metric_definition': {'diagnostic_version': 'b2-contract-v1', 'strict_scorer': task.SCORER,
        'denominator': '24 unique documents per cell; 192 POST are repeated treatment outcomes',
        'evidence_line_id_exists': 'Both required values are null or actual row IDs; factual support scored separately',
        'latency_http_seconds': 'response_complete-http_submit; C1 single-server descriptive; 10s demo threshold'}}
    write(out / 'summary.json', summary)
    lines = ['# BENCH05-B2 CPU 재계산', '', '고정된 새 합성24문서의 지시 추가 비교이며 일반 품질·배포 동등성 인증이 아닙니다.', '',
      '| arm | policy | n | schema | enum | evidence keys | core values | strict task |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in flat:
        lines.append(f"| {row['arm']} | {row['policy']} | {row['submitted']}/24 | {row['strict_schema_valid']} | {row['status_enum_valid']} | {row['evidence_required_keys_present']} | {row['core_values_correct']} | {row['strict_task_success']} |")
    (out / 'REPORT_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def replay(bundle, output):
    out = derived_output(bundle, output)
    with preserve_failure(out), open_bundle(bundle) as root:
        if not (root / 'RAW_MANIFEST.json').exists():
            found = [p.parent for p in root.rglob('RAW_MANIFEST.json') if read(p).get('scope') == 'bench05b2_contract_raw']
            require(len(found) == 1, 'Ambiguous/missing B2 raw root'); root = found[0]
        return analyze(root, out)
