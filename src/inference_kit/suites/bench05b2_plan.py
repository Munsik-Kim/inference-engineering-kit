"""CPU-only B2 contract freeze. Intentional prompt axes do not disable identity checks."""
from dataclasses import fields
from pathlib import Path
import random
import shutil
import time
from ..core.io import read, write, sha, digest, require
from ..core.paths import Paths
from ..core.state import new_output
from ..tasks import tech_record as task
from ..tasks import contract_policy as policy
from ..probes.contracts import Sample, attention_coordinates
from ..probes.task_tokens import field_targets, SELECTION
from ..adapters.tensors import check_scheme
from ..models.capabilities import inspect_capabilities
from .bench05a_plan import source_identity, ledger_evidence


def validate_data(data):
    require(data['data_version'] == policy.DATA_VERSION and data['task_id'] == task.TASK and data['task_version'] == task.VERSION and data['scorer'] == task.SCORER, 'B2 data/task/scorer mismatch')
    rows = data['rows']
    require(len(rows) == 28 and len({r['sample_id'] for r in rows}) == 28 and len({r['document_hash'] for r in rows}) == 28, 'B2 needs 28 unique documents')
    require(digest(rows) == data['rows_identity'], 'B2 rows identity changed')
    require(sum(r['split'] == 'SMOKE' for r in rows) == 4 and sum(r['split'] == 'EVAL_LOCKED' for r in rows) == 24, 'B2 split counts')
    for case in task.CASES:
        for language in ('en', 'ko'):
            require(sum(r['split'] == 'EVAL_LOCKED' and r['case_type'] == case and r['language'] == language for r in rows) == 2, 'B2 type/language design changed')
    for row in rows:
        value = dict(row)
        identity = value.pop('sample_identity')
        require(digest(value) == identity, 'B2 sample identity changed')
        policy.verify_gold(row)
        positive = {'task_id': task.TASK, 'task_version': task.VERSION, 'sample_id': row['sample_id'],
          'sample_identity': row['sample_identity'], 'transport_complete': True, 'termination': 'NATURAL_STOP',
          'output_text': task.canonical_reference(row)['text'], 'request_latency_http': 1.0}
        require(task.score(row, positive)['task_success'], 'Canonical gold fails unchanged strict scorer')
    return rows


def validate_prepared(prepared, data):
    require(prepared['kind'] == 'prepared_b2_contract_samples_v1' and prepared['preparation_id'] == policy.PREPARATION, 'Not a B2 640-token preparation')
    require(prepared['bounds'] == {'prompt': 512, 'reference': 128, 'joint': 640}, 'B2 bounds changed')
    source = {r['sample_id']: r for r in validate_data(data)}
    rows = prepared['generation_rows']
    require(len(rows) == 112 and digest(rows) == prepared['generation_rows_identity'], 'B2 prepared rows changed')
    require([{'sample_id': r['sample_id'], 'policy_id': r['policy_id'], 'split': r['split']} for r in rows] == policy.request_order(data['rows']), 'B2 request rotation/order changed')
    require(len({(r['sample_id'], r['policy_id']) for r in rows}) == 112, 'Duplicate B2 request')
    cfg = prepared['model_config']
    for row in rows:
        Sample(**{f.name: row[f.name] for f in fields(Sample)}).validate()
        sample = source[row['sample_id']]
        name = row['policy_id']
        ref = task.canonical_reference(sample)
        content = policy.render(sample, name)
        require(row['sample_identity'] == sample['sample_identity'] and row['document_hash'] == sample['document_hash'], 'B2 source identity mismatch')
        require(row['policy_identity'] == policy.policy_record(name)['policy_identity'], 'B2 policy identity mismatch')
        require(row['task_prompt_hash'] == digest(content) and row['prompt'].count(content) == 1 and ref['text'] not in row['prompt'], 'B2 prompt changed, duplicate chat, or gold leak')
        require(row['reference_text'] == ref['text'] and row['value_spans'] == ref['value_spans'], 'B2 canonical reference/value span changed')
        require(row['reference_source'] == 'task_author_gold_json' and row['tokenization']['policy'] == policy.PREPARATION and row['tokenization']['chat_template_applied'] == 1, 'B2 preparation policy changed')
        require(row['tokenization']['tokenizer_identity'] == prepared['tokenizer_identity'] and row['tokenization']['token_id_space'] == prepared['token_id_space'], 'B2 token space changed')
        ids = row['input_ids']; boundary = row['prompt_token_boundary']
        require(0 < boundary < len(ids) <= 640 and len(row['generation_input_ids']) <= 512 and len(row['reference_separate_ids']) <= 128, 'B2 token bounds exceeded; no truncation')
        require(all(type(v) is int and 0 <= v < prepared['vocabulary_size'] for v in ids + row['generation_input_ids']), 'B2 invalid token ID')
        require(row['generation_input_ids'] == row['prompt_separate_ids'] and row['sequence_identity'] == digest(ids), 'B2 generation/sequence identity changed')
        require(len(row['offsets_codepoints']) == len(row['offsets_utf8_bytes']) == len(ids), 'B2 offset coverage')
        require(row['full_text_hash'] == digest(row['prompt'] + ref['text']) and row['prompt_hash'] == digest(row['prompt']) and row['reference_hash'] == digest(ref['text']), 'B2 text hash mismatch')
        targets, coverage = field_targets(ids, boundary, row['offsets_codepoints'], len(row['prompt']), row['value_spans'], row['special_token_ids'])
        require(row['targets'] == targets and row['field_token_coverage'] == coverage and row['selection_policy'] == SELECTION, 'B2 field target selection changed')
        valid = [i for i in range(boundary, len(ids)) if ids[i] not in row['special_token_ids']]
        require(row['attention_mask'] == [1] * len(ids) and row['loss_mask'] == [int(i in valid) for i in range(len(ids))], 'B2 padding/loss mask mismatch')
        require(row['attention'] == attention_coordinates(cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads'], boundary, [{'token_position': valid[-1]}]), 'B2 must reuse A attention selection')
        require(set(row['full_vocab_target_positions']) <= {t['token_position'] for t in targets} and len(row['full_vocab_target_positions']) <= 4, 'B2 full vocabulary selection')
    selected = policy.select_probes(data['rows'])
    require(prepared['probe_selection']['sample_ids'] == selected, 'B2 probe document selection changed')
    for name in ('P00', 'P11'):
        require(prepared['by_policy'][name] == [next(r for r in rows if r['sample_id'] == sid and r['policy_id'] == name) for sid in selected], 'B2 probe/policy identity mismatch')
    return rows


def plan(suite_file, models_file, local_file, output):
    suite_file, models_file, local_file = [Path(p).absolute() for p in (suite_file, models_file, local_file)]
    suite, pair, local = read(suite_file), read(models_file), read(local_file)
    require(suite['suite_id'] == 'BENCH05-B2-CONTRACT' and suite['schema_version'] == 1 and suite['task_id']==task.TASK and suite['task_version']==task.VERSION, 'Unsupported B2 suite')
    require(suite['treatment_axes'] == ['weight_artifact', 'evidence_instruction', 'status_instruction'], 'B2 axes changed')
    require(suite['generation'] == {'profile': 'natural_generation_v1', 'max_tokens': 192, 'concurrency': 1, 'smoke_samples': 4, 'eval_samples': 24, 'warmup_posts': 0}, 'B2 generation contract changed')
    require(suite['serving'] == {'context': 2048, 'max_num_seqs': 1, 'max_num_batched_tokens': 2048, 'kv_tokens': 4096, 'shutdown_timeout': 10, 'cleanup_grace': 25, 'chunked_prefill': True, 'eager': True, 'prefix_caching': False, 'async_scheduling': False}, 'B2 serving contract changed')
    require(suite['observer'] == {'off_samples': 0, 'off_repeats': 0} and suite['probe_policies'] == ['P00', 'P11'], 'B2 adds no new collector/bridge')
    b = suite['budget']
    require(0 < b['task_gpu_wall_seconds'] <= 600 and b['max_gpu_processes'] <= 6 and b['automatic_retries'] == 0 and b['cleanup_reserve_seconds'] >= 60 and 0 < suite['serving_job_seconds'] <= 240, 'Unbounded B2 execution')
    require(pair['model_id'] == 'HuggingFaceTB/SmolLM2-1.7B-Instruct' and pair['revision'] == '31b70e2e869a7173562077fd711b654946d38674', 'Only existing pinned Smol supported')
    require([a['arm_id'] for a in pair['arms']] == ['bf16', 'rtn'] and [a['execution_mode'] for a in pair['arms']] == ['INSTRUMENTED_PATH', 'DECODED_WEIGHT_DIAGNOSTIC'], 'B2 no precision fallback')
    paths = Paths.load(suite_file, local_file)
    resolved = {k: str(paths.resolve(local[k])) for k in ('authored_data', 'artifact_receipt', 'prepared_samples', 'runtime_metadata', 'runtime_python', 'server_python', 'package_site', 'prior_ledger', 'campaign_root', 'task_clock')}
    data = read(Path(resolved['authored_data'])); prepared = read(Path(resolved['prepared_samples']))
    validate_prepared(prepared, data)
    require(prepared['authored_sha256'] == sha(Path(resolved['authored_data'])), 'B2 authored bytes changed')
    runtime = read(Path(resolved['runtime_metadata'])); refs = read(Path(resolved['artifact_receipt'])); arms = []; available = True
    for declaration in pair['arms']:
        ref = refs['arms'][declaration['receipt_arm']]; p = Path(ref['path']); exists = (p / 'config.json').is_file(); stats = {}
        require(ref['model_id'] == pair['model_id'] and ref['model_revision'] == pair['revision'], 'B2 artifact revision mismatch')
        if exists:
            cfg = read(p / 'config.json')
            require(sha(p / 'config.json') == ref['config_sha256'], 'B2 artifact config changed')
            for k in ('model_type', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'hidden_size', 'vocab_size'):
                require(cfg[k] == prepared['model_config'][k], 'B2 structure mismatch: ' + k)
            if declaration['arm_id'] == 'rtn':
                check_scheme(cfg); require(ref['recipe']['algorithm'] == 'RTN', 'B2 requires original RTN')
            else:
                require(not cfg.get('quantization_config'), 'B2 BF16 artifact is quantized')
            require(all((Path(ref['tokenizer_path']) / n).is_file() and sha(Path(ref['tokenizer_path']) / n) == h for n, h in prepared['tokenizer_files'].items()), 'B2 tokenizer changed')
            stats = {f.name: {'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns} for f in p.glob('*.safetensors')}
            require(stats, 'B2 local weights absent')
        available &= exists
        arms.append(declaration | {'artifact': ref, 'artifact_available': exists, 'weight_file_stat': stats})
    history = ledger_evidence(resolved['prior_ledger'], resolved['campaign_root'])
    origin = local['task_origin_used_seconds']; require(history['used_seconds'] >= origin, 'B2 campaign origin differs')
    allowance = min(b['task_gpu_wall_seconds'] - (history['used_seconds'] - origin), 7200 - history['used_seconds'])
    require(allowance > b['cleanup_reserve_seconds'], 'B2 lacks cleanup-safe budget')
    clock = read(Path(resolved['task_clock'])); require(clock['deadline_monotonic'] == clock['start_monotonic'] + 3600, 'B2 task deadline changed')
    mem = {s.split(':')[0]: int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if s.startswith('MemAvailable:')}
    disk = shutil.disk_usage(suite_file.parent)
    healthy = available and all(Path(resolved[k]).is_file() for k in ('runtime_python', 'server_python')) and mem.get('MemAvailable', 0) >= b['minimum_cpu_available_bytes'] and disk.free >= b['minimum_disk_free_bytes']
    cfg = prepared['model_config']; head = cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])
    maximum = max(len(r['input_ids']) for r in prepared['generation_rows'])
    extra = max(len(r['targets']) for r in prepared['generation_rows']) * cfg['vocab_size'] * 12 + maximum * head * 4 * 12
    require(extra <= b['probe_gpu_buffer_bytes'], 'B2 probe memory cap')
    capability = inspect_capabilities(pair['model_id'], cfg, runtime, available)
    require(not any(v['status'] == 'UNSUPPORTED' for v in capability['features'].values()), 'B2 diagnostic backend unsupported')
    order = ['bf16', 'rtn']; random.Random(suite['seed']).shuffle(order)
    counts = {'unique_eval_documents': 24, 'smoke_documents': 4, 'policies': 4, 'natural_requests_per_arm': 112,
      'natural_requests_total': 224, 'eval_requests_total': 192, 'smoke_posts_per_arm': 16, 'smoke_requests_total': 32,
      'maximum_natural_generated_tokens': 224 * 192, 'warmup_posts': 0, 'serving_jobs': 2, 'optional_diagnostic_jobs': 4, 'max_gpu_jobs': 6}
    result = {'schema_version': 1, 'task': 'BENCH05-B2-CONTRACT', 'task_id': task.TASK, 'task_version': task.VERSION,
      'evidence_kind': 'planned_live_contract_pipeline', 'status': 'CPU_PLANNED_NOT_EXECUTED' if healthy else 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT',
      'gpu_actions': 0, 'model_loads': 0, 'network_actions': 0, 'suite': suite, 'pair': pair, 'data': data, 'samples': prepared,
      'policies': {n: policy.policy_record(n) for n in policy.POLICIES}, 'arms': arms, 'arm_order': order,
      'paths': resolved, 'gpu': {'index': local['gpu'], 'uuid': local['gpu_uuid']}, 'port': local['port'],
      'runtime': runtime, 'runtime_identity': digest(runtime), 'serving_runtime_identity': local['serving_runtime_identity'],
      'capability': capability, 'campaign': history, 'task_origin_used_seconds': origin, 'gpu_allowance_seconds': allowance,
      'clock': {k: clock[k] for k in ('start_utc', 'start_monotonic', 'deadline_monotonic', 'no_new_gpu_after')},
      'counts': counts, 'resource_estimates': {'cpu_available_bytes': mem.get('MemAvailable'), 'disk_free_bytes': disk.free,
        'extra_probe_gpu_buffer_upper_bytes': extra, 'maximum_sequence_tokens': maximum},
      'source_identity': source_identity(), 'scorer': task.SCORER, 'scorer_identity': sha(Path(task.__file__)),
      'input_file_sha256': {k: sha(Path(resolved[k])) for k in ('authored_data', 'prepared_samples', 'artifact_receipt', 'runtime_metadata')},
      'spec_files': {str(p): sha(p) for p in (suite_file, models_file)}, 'scope': 'synthetic_locked_contract_pilot_not_benchmark_test'}
    result['plan_identity'] = digest(result)
    out = new_output(output); write(out / 'PLAN_FROZEN.json', result)
    return result


def check_frozen(plan):
    frozen = dict(plan); identity = frozen.pop('plan_identity')
    require(digest(frozen) == identity and plan['task'] == 'BENCH05-B2-CONTRACT', 'Frozen B2 plan changed')
    require(plan['status'] == 'CPU_PLANNED_NOT_EXECUTED', 'B2 local resources/artifacts unresolved')
    require(plan['source_identity'] == source_identity() and plan['scorer_identity'] == sha(Path(task.__file__)), 'B2 source/scorer changed after freeze')
    require(Path(__file__).absolute().parents[2] == Path(plan['paths']['package_site']).absolute(), 'B2 controller must use explicit installed package_site')
    for k, value in plan['input_file_sha256'].items():
        require(sha(Path(plan['paths'][k])) == value, 'B2 input changed: ' + k)
    for p, value in plan['spec_files'].items():
        require(sha(Path(p)) == value, 'B2 spec changed after freeze')
    require(sha(Path(plan['campaign']['path'])) == plan['campaign']['sha256'], 'Prior B2 ledger changed')
    require(time.monotonic() < plan['clock']['no_new_gpu_after'], 'B2 no new GPU jobs after cutoff')
    require(plan['policies'] == {n: policy.policy_record(n) for n in policy.POLICIES}, 'B2 policy changed')
    validate_prepared(plan['samples'], plan['data'])
