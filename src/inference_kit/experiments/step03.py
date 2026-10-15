# Numerical definitions retained; evidence input and derived output separated.
"""Recompute STEP03 cells, two-factor contrasts and source-aware deployment screens."""
import copy
import json
import math
from pathlib import Path
import statistics

from ..core.io import read, require, sha, digest, write
from ..core.assess import assess
from ..core.metrics import summarize
from ..core.quality import score_document


def quantile(values, p):
    values = sorted(values); location = (len(values) - 1) * p
    lo, hi = math.floor(location), math.ceil(location)
    return values[lo] + (values[hi] - values[lo]) * (location - lo)


def immutable(path, value):
    if path.exists():
        require(read(path) == value, 'CPU reanalysis differs from saved output: ' + str(path))
    else:
        write(path, value)


def memory(directory):
    markers = read(directory / 'memory-phases.json')
    samples = [json.loads(line) for line in (directory / 'memory.jsonl').read_text().splitlines()]
    valid = []
    for row in samples:
        fields = [s.strip() for s in row.get('values', '').split(',')]
        if row.get('returncode') == 0 and len(fields) == 10:
            valid.append({'monotonic': row['monotonic'], 'used_mib': float(fields[3]), 'uuid': fields[1], 'hardware': fields[5:]})
    phases = {}
    for start, end in zip(markers, markers[1:]):
        values = [r['used_mib'] for r in valid if start['monotonic'] <= r['monotonic'] < end['monotonic']]
        phases[start['phase']] = {'n_samples': len(values), 'max_mib': max(values) if values else None,
                                  'min_mib': min(values) if values else None, 'median_mib': statistics.median(values) if values else None}
    performance = [v['max_mib'] for k, v in phases.items() if k in ('startup', 'loaded_idle', 'performance_idle_after', 'shutdown') or k.startswith(('warmup_', 'timed_', 'drain_'))]
    performance = [v for v in performance if v is not None]
    before = phases.get('background_before', {}).get('median_mib'); after = phases.get('background_after', {}).get('median_mib')
    return {'scope': 'whole_device_mib', 'phase_scope': 'performance_with_startup', 'max_mib': max(performance) if performance else None,
            'task_max_mib': max(r['used_mib'] for r in valid) if valid else None, 'phases': phases,
            'background_before_mib': before, 'background_after_mib': after,
            'background_drift_mib': abs(after - before) if before is not None and after is not None else None,
            'sampling_interval_median_seconds': statistics.median(b['monotonic']-a['monotonic'] for a,b in zip(valid,valid[1:])) if len(valid)>1 else None,
            'n_samples': len(valid), 'process_memory': 'NOT_AGGREGATED; raw NVML app rows retain availability', 'allocator_memory': 'NOT_MEASURED',
            'limitation': 'Whole-device samples include background and can miss peaks; sanity and post-sanity idle excluded from performance max, shutdown reported separately'}


def cell_summary(cell, directory):
    records = cell['timed']['records']; stored = cell['timed']['summary']
    require(len(records) == 16 and all(r['status'] == 'OK' and r['error'] is None for r in records), 'Timed request coverage damaged')
    out = {'input_length': cell['input_length'], 'output_tokens': 128, 'concurrency': 4, 'n_timed_requests': 16,
           'elapsed_seconds': stored['elapsed_seconds'], 'actual_input_tokens': sum(r['usage']['prompt_tokens'] for r in records),
           'actual_output_tokens': sum(r['usage']['completion_tokens'] for r in records),
           'ttft_gt_one_second_fraction': sum(r['ttft_seconds'] > 1 for r in records) / len(records),
           'request_per_second': 16 / stored['elapsed_seconds'], 'output_tokens_per_second': stored['output_tokens_per_second'],
           'actual_client_overlap': stored['actual_max_client_overlap'], 'warmup_overlap': cell['warmup']['summary']['actual_max_client_overlap'],
           'http_e2e_excludes_client_queue': True, 'task_e2e_includes_client_queue': True}
    require(out['actual_client_overlap'] == out['warmup_overlap'] == 4, 'Warmup/timed overlap contract differs')
    for r in records:
        require(r['usage']['prompt_tokens'] == cell['input_length'] and r['usage']['completion_tokens'] == 128 and r['finish_reason'] == 'length' and r['saw_done'], 'Actual token/finish contract differs')
        require(math.isclose(r['http_complete_seconds'] + r['client_queue_seconds'], r['completed_monotonic'] - r['queued_monotonic'], abs_tol=1e-8), 'Task/client/HTTP time boundaries differ')
    for metric in ('ttft_seconds', 'tpot_stream_estimate_seconds', 'http_complete_seconds', 'client_queue_seconds'):
        out[metric] = {s: quantile([r[metric] for r in records], p) for s, p in (('p50', .5), ('p95', .95))}
        if metric in stored:
            require(out[metric] == stored[metric], 'Raw client statistic readback differs')
    out['task_e2e_seconds'] = {s: quantile([r['completed_monotonic']-r['queued_monotonic'] for r in records], p) for s,p in (('p50',.5),('p95',.95))}
    observations = [json.loads(s) for s in (directory / 'metrics.jsonl').read_text().splitlines()]
    block = cell['timed']
    observations = [block['metrics_before']] + observations + [block['metrics_after']]
    out['scheduler'] = summarize(observations, block['metrics_before']['parsed'], block['metrics_after']['parsed'], block['measurement_begin'], block['measurement_end'])
    out['raw_request_ids'] = [r['request_id'] for r in records]
    out['output_hashes'] = [r['output_text_hash'] for r in records]
    return out


def sanity_pair(left, right):
    if left is None or right is None:
        return {'status': 'NOT_AVAILABLE', 'same_weight_prediction_match': None}
    require(left['status'] == right['status'] == 'VALID_LIVE_SUBSET', 'Sanity request damaged')
    l, r = left['records'], right['records']
    require(len(l) == len(r) == 16 and [x['sample_id'] for x in l] == [x['sample_id'] for x in r], 'Sanity selection differs')
    for record in l + r:
        calculated = score_document(record, record['choice_loglikelihood'])
        require(all(calculated[k] == record[k] for k in ('acc', 'acc_norm', 'prediction_acc', 'prediction_acc_norm', 'choice_normalized_scores')), 'Sanity score readback differs')
    changes = [{'sample_id': a['sample_id'], 'gold': a['gold'], 'left_prediction': a['prediction_acc_norm'], 'right_prediction': b['prediction_acc_norm'],
                'choice_score_changes': [y-x for x,y in zip(a['choice_loglikelihood'],b['choice_loglikelihood'])]} for a,b in zip(l,r)]
    return {'status': 'COMPLETE', 'documents': 16, 'same_weight_prediction_match': all(a['prediction_acc_norm']==b['prediction_acc_norm'] and a['prediction_acc']==b['prediction_acc'] and a['gold']==b['gold'] for a,b in zip(l,r)),
            'max_absolute_choice_score_change': max(abs(v) for row in changes for v in row['choice_score_changes']),
            'left_acc_norm': left['metrics']['acc_norm'], 'right_acc_norm': right['metrics']['acc_norm'], 'per_input': changes,
            'meaning': 'DEV16 cache consistency only; not new quality certification, sample-size increase, or generalization evidence'}


def contrasts(rounds):
    results = []
    for block in sorted({r['block'] for r in rounds}):
        selected = {r['arm']: r for r in rounds if r['block'] == block}
        if set(selected) != set('ABCD'): continue
        for length in (128, 1024):
            def values(arm):
                c = next(c for c in selected[arm]['cells'] if c['input_length'] == length)
                return {'ttft_p95_seconds': c['ttft_seconds']['p95'], 'ttft_gt_one_fraction': c['ttft_gt_one_second_fraction'],
                        'output_tokens_per_second': c['output_tokens_per_second'], 'timed_device_max_mib': c['memory_max_mib'],
                        'waiting_time_weighted_mean': c['scheduler']['waiting']['time_weighted_mean'],
                        'queue_histogram_mean_seconds': c['scheduler']['queue_histogram']['mean_seconds']}
            vectors = {arm: values(arm) for arm in 'ABCD'}
            comparisons = {}
            for name, left, right in (('B-A','B','A'),('D-C','D','C'),('C-A','C','A'),('D-B','D','B')):
                comparisons[name] = {k: vectors[left][k]-vectors[right][k] if vectors[left][k] is not None and vectors[right][k] is not None else None for k in vectors[left]}
            comparisons['interaction'] = {k: comparisons['D-C'][k]-comparisons['B-A'][k] if comparisons['D-C'][k] is not None and comparisons['B-A'][k] is not None else None for k in vectors['A']}
            results.append({'block': block, 'input_length': length, 'vectors': vectors, 'paired_differences': comparisons,
                            'directions': 'TTFT/wait/fraction/memory lower is better; throughput higher is better; differences only, no pseudo-replicate inference'})
    return results


def analyze(spec, root, output):
    root = Path(root).absolute(); output = Path(output).absolute(); frozen = read(root / 'PLAN_FROZEN.json')
    require(frozen['spec'] == spec, 'Analysis spec differs from frozen campaign')
    retained = read(root / 'RETAINED_QUALITY.json'); manifest = dict(retained['evidence_manifest'])
    evidence_root = root.parent
    def evidence(path):
        path = Path(path); require(path.resolve().is_relative_to(evidence_root.resolve()), 'Evidence outside reports root')
        name = str(path.relative_to(evidence_root)); manifest[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}; return name
    for name in ('PLAN_FROZEN.json', 'START.json', 'RETAINED_QUALITY.json', 'data/performance.json', 'data/cache-sanity.json', 'data/SANITY_SELECTION.json', 'CPU_TESTS_RECEIPT.json', 'gpu_budget.json'):
        evidence(root / name)
    rounds = []; files = {}; raw_runs = {}; sanity = {}
    for directory in sorted((root / 'jobs').glob('*')):
        if not (directory / 'performance.json').exists(): continue
        result = read(directory / 'performance.json'); arm = result['arm']['arm']; block = result['block']
        require(read(directory / 'freeze.json')['plan_sha256'] == sha(root / 'PLAN_FROZEN.json'), 'Measured plan identity differs')
        require(result['comparison_group_identity'] == frozen['comparison_group_identity'], 'Measured comparison identity differs')
        require(result['effective_cache']['effective_tokens_logged'] == result['arm']['kv_tokens'], 'Effective cache was not changed')
        require(result['runtime_identity'] == frozen['runtime_identity'] and result['data_sha256'] == frozen['performance_sha256'], 'Runtime or inputs changed')
        require(result['execution_identity'] == frozen['actual_launches'][arm]['execution_identity'], 'Execution artifact/cache/profile identity changed')
        lifecycle = read(directory / 'lifecycle.json') if (directory / 'lifecycle.json').exists() else {}
        ownership = [json.loads(s) for s in (directory / 'ownership.jsonl').read_text().splitlines()]
        clean = not (directory / 'failure.json').exists() and lifecycle.get('owned_process_cleanup') == 'CLEANED' and lifecycle.get('owned_port_cleanup') == 'FREE' and lifecycle.get('parent_returncode') == 0
        observed = bool(ownership) and all(o['returncode'] == 0 and not o['unverified_pids'] for o in ownership)
        m = memory(directory)
        values = [cell_summary(c, directory) for c in result['cells']]
        for cell in values: cell['memory_max_mib'] = m['phases'][f"timed_L{cell['input_length']}_C4"]['max_mib']
        rounds.append({'arm': arm, 'block': block, 'execution_identity': result['execution_identity'], 'effective_cache': result['effective_cache'],
                       'memory': m, 'cells': values, 'clean': clean, 'ownership_observed': observed, 'parent_returncode': lifecycle.get('parent_returncode'),
                       'startup_seconds': result['startup_seconds']})
        raw_runs[(block,arm)] = result
        files.setdefault(arm, []).extend(evidence(p) for p in sorted(directory.rglob('*')) if p.is_file() and 'runtime-cache' not in p.parts)
        if block == 0 and (directory / 'cache-sanity.json').exists(): sanity[arm] = read(directory / 'cache-sanity.json')
    paired_sanity = {'bf16': sanity_pair(sanity.get('A'),sanity.get('B')), 'gptq_w4': sanity_pair(sanity.get('C'),sanity.get('D'))}
    background = [r['memory']['background_before_mib'] for r in rounds if r['memory']['background_before_mib'] is not None]
    span = max(background)-min(background) if background else None
    rows = []
    for arm in frozen['arms']:
        selected = [r for r in rounds if r['arm'] == arm['arm']]; complete = len(selected) == 3 and all(r['clean'] for r in selected)
        isolated = complete and span is not None and span <= 256 and all(r['ownership_observed'] and r['memory']['background_drift_mib'] is not None and r['memory']['background_drift_mib'] <= 256 for r in selected)
        cells = []
        for length in (128,1024):
            selected_cells = [next(c for c in r['cells'] if c['input_length']==length) for r in selected]
            c = {'input_length':length,'output_tokens':128,'concurrency':4,'n_server_rounds':len(selected),'n_timed_requests':len(selected)*16,'rounds':selected_cells}
            for metric in ('ttft_seconds','tpot_stream_estimate_seconds','http_complete_seconds'):
                c[metric] = {s:max(x[metric][s] for x in selected_cells) if selected_cells else None for s in ('p50','p95')}
                all_records=[record for key,run in raw_runs.items() if key[1]==arm['arm'] for raw in run['cells'] if raw['input_length']==length for record in raw['timed']['records']]
                c['pooled_descriptive_'+metric]={s:quantile([r[metric] for r in all_records],p) if all_records else None for s,p in (('p50',.5),('p95',.95))}
            cells.append(c)
        old = retained['rows'][arm['method']]
        rows.append({'schema_version':3,'configuration_id':'qwen3_4b--'+arm['method']+'--kv'+str(arm['kv_tokens']),'arm':arm['arm'],
                     'kind':'live_measurement' if selected else 'blocked_not_measured','measurement_status':'VALID_LIVE_PROFILE' if complete else 'PARTIAL_UNCERTIFIED',
                     'model_id':spec['model']['model_id'],'model_revision':spec['model']['revision'],'treatment_axes':spec['design']['treatment_axes'],
                     'identity':frozen['identity'],'comparison_group_identity':frozen['comparison_group_identity'], 'artifact':frozen['artifacts'][arm['method']],
                     'execution':arm,'execution_identity':selected[0]['execution_identity'] if selected else None,
                     'validity':{'requests_complete':complete,'owned_resources_clean':complete,'server_rounds':len(selected),'parent_returncodes':[r['parent_returncode'] for r in selected],
                                 'comparison_status':'NO_INTERFERENCE_OBSERVED_WITHIN_OBSERVATION_LIMITS' if isolated else 'INTERFERENCE_DRIFT_OR_INCOMPLETE',
                                 'worker_native_exit':'UNKNOWN','kernel_level':'kernel_selected_log'},
                     'memory':{'scope':'whole_device_mib','phase_scope':'performance_with_startup','max_mib':max(r['memory']['max_mib'] for r in selected) if selected and all(r['memory']['max_mib'] is not None for r in selected) else None,
                               'task_max_mib':max(r['memory']['task_max_mib'] for r in selected) if selected and all(r['memory']['task_max_mib'] is not None for r in selected) else None,'server_rounds':[r['memory'] for r in selected],'evidence':files.get(arm['arm'],[])},
                     'performance':{'cells':cells,'evidence':files.get(arm['arm'],[])},'retained_quality':old['quality'],'cache_sanity':paired_sanity[arm['method']],
                     'exact_profile_quality':'NOT_VERIFIED', 'quality_transfer':{'identity_status':'MATCHED_ARTIFACT_RUNTIME_TOKENIZER_SCORING','quality_source_profile':old['quality_source_profile'],
                       'quality_run_id':old['quality_run_id'],'assumptions':['Checkpoint acc_norm reused only if explicitly permitted','Cache/batching numerical paths not fully certified by DEV16 sanity','No extra independent TEST samples']},
                     'evidence_files':files.get(arm['arm'],[])+old['quality']['evidence'],
                     'reexecute_commands':[j['argv'] for j in read(root/'gpu_budget.json')['jobs'] if j['label'].endswith('-'+arm['arm'])],
                     'limitations':['NVML whole-device sampled peak, not process/allocator','Three server blocks; n16/cell; not production p95 or mixed workload','WSL and graphics/brief foreign activity not exhaustively excluded','Retained ARC256 with within-subset interval; exact-profile usability not verified']})
    effects = contrasts(rounds)
    output_comparison = []
    for block in range(spec['design']['blocks']):
        for weight,left,right in (('bf16','A','B'),('gptq_w4','C','D')):
            if (block,left) not in raw_runs or (block,right) not in raw_runs: continue
            for length in (128,1024):
                l=next(c for c in raw_runs[(block,left)]['cells'] if c['input_length']==length)['timed']['records']
                r=next(c for c in raw_runs[(block,right)]['cells'] if c['input_length']==length)['timed']['records']
                require([x['request_id'] for x in l]==[x['request_id'] for x in r],'Performance input pairing differs')
                output_comparison.append({'block':block,'weight':weight,'input_length':length,'matched_output_hashes':sum(a['output_text_hash']==b['output_text_hash'] for a,b in zip(l,r)),'n':16,'meaning':'Output consistency only; not accuracy'})
    index={'schema_version':3,'kind':'verified_result_index','evidence_root':str(evidence_root),'results':rows,'results_identity':digest(rows),'evidence_manifest':manifest,
           'scope':'STEP03 only; two declared factors; retained quality explicitly linked','comparison_group_identity':frozen['comparison_group_identity']}
    immutable(output/'results-index.json',index)
    immutable(output/'comparisons.json',{'rounds':rounds,'paired_effects':effects,'cache_sanity':paired_sanity,'performance_output_consistency':output_comparison,'background_between_round_span_mib':span})
    assessments={}
    for name,evidence_level,allow,quality_policy in (('point','exploratory_point',True,'retained_checkpoint'),('interval','screen_with_interval',True,'retained_checkpoint'),('exact-profile','exploratory_point',True,'exact_profile'),('no-retained','exploratory_point',False,'retained_checkpoint')):
        req={'schema_version':3,'policy':'DEMO','model_id':spec['model']['model_id'],'model_revision':spec['model']['revision'],
             'comparison_group_identity':frozen['comparison_group_identity'],'identity':frozen['identity'],'treatment_axes':spec['design']['treatment_axes'],
             'memory':spec['demo_policy']['memory'],'latency_constraints':spec['demo_policy']['latency_constraints'],'quality':spec['demo_policy']['quality'],
             'allow_retained_quality':allow,'quality_evidence':quality_policy,'evidence_level':evidence_level}
        immutable(output/'requirements'/f'{name}.json',req);result=assess(index,req);immutable(output/'assessments'/f'{name}.json',result);assessments[name]=result
    saved_summary = read(root/'summary.json')
    immutable(output/'summary.json', saved_summary)
    return {'summary':saved_summary,'index':index,'rounds':rounds,'paired_effects':effects,'assessments':assessments}
