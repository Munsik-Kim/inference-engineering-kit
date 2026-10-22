"""Read preserved B2/B4 free-generation evidence without model dependencies.

This adapter verifies the source raw manifest and frozen plan, then delegates
SSE/wire/timing validation and original task scoring to the maintained recipes.
It never tokenizes text, generates tokens, or treats B3's audit token as output.
"""
from pathlib import Path
from ..core.io import read, require, digest, safe_name, sha, loads
from ..adapters.protocol import Adapter
from ..suites.bench05b1_replay import generation_from_raw, jsonl


def _server_ids(record):
    """Only expose recorded server IDs; text is never re-tokenized."""
    actual = []; saw_ids = False
    for frame in record.get('raw_sse_frames', []):
        if frame.strip() == '[DONE]': continue
        event = Adapter.event(loads(frame)); ids = event['ids']
        if ids is not None:
            require(isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids), 'Invalid raw server output token IDs')
            actual.extend(ids); saw_ids = True
    stored = record.get('output_token_ids')
    if stored is not None:
        require(isinstance(stored, list) and all(type(t) is int and t >= 0 for t in stored), 'Invalid stored output token IDs')
        if saw_ids: require(stored == actual, 'Stored output token IDs differ from raw SSE')
    if saw_ids: return actual, 'RAW_SSE_SERVER_TOKEN_IDS'
    if stored is not None: return stored, 'STORED_SERVER_TOKEN_IDS'
    return None, 'NOT_COLLECTED'


def _plan_check(root, plan):
    source_plan = read(root / 'PLAN_FROZEN.json')
    require(source_plan == plan, 'Generation source/frozen plan mismatch')
    frozen = dict(plan); expected = frozen.pop('plan_identity')
    require(digest(frozen) == expected, 'Generation frozen plan integrity changed')
    return source_plan


def _identity(plan, sample, prepared, policy, arm):
    return {'task_id': sample['task_id'], 'task_version': sample['task_version'],
            'sample_id': sample['sample_id'], 'sample_identity': sample['sample_identity'],
            'split': sample['split'], 'policy_id': policy, 'prompt_hash': prepared['prompt_hash'],
            'tokenizer_identity': plan['samples']['tokenizer_identity'],
            'token_id_space': plan['samples']['token_id_space'],
            'runtime_identity': plan['serving_runtime_identity'], 'scorer_identity': plan['scorer_identity'],
            'document_hash': sample['document_hash'], 'gold_identity': digest(sample['gold']),
            'generation_policy_identity': digest({'serving': plan['suite']['serving'], 'generation': plan['suite']['generation']}),
            'arm_id': arm['arm_id'], 'artifact_identity': arm['artifact']['artifact_identity']}


def _output(spec, plan, source_identity, sample, prepared, arm, policy, record, score, manifest_sha):
    identity = _identity(plan, sample, prepared, policy, arm)
    # Content identity excludes local paths and provenance annotations so a
    # verified relocation has identical bundle and record IDs.
    bundle_identity = digest({'plan_identity': plan['plan_identity'], 'raw_manifest_sha256': manifest_sha})
    rid = record.get('request_id') if record else None
    record_id = digest({'source_id': spec['source_id'], 'bundle_identity': bundle_identity,
                        'task_id': sample['task_id'], 'task_version': sample['task_version'],
                        'arm_id': arm['arm_id'], 'sample_id': sample['sample_id'], 'policy_id': policy,
                        'request_id': rid, 'kind': 'free_generation'})
    ids, provenance = _server_ids(record) if record and not record.get('recovered_from_submission_journal') else (None, 'NOT_COLLECTED')
    if ids is not None:
        require(all(t < plan['samples']['vocabulary_size'] for t in ids), 'Actual server token ID outside declared vocabulary')
    usage = record.get('usage') if record else None
    complete = record.get('transport_complete') is True if record else False
    if ids is not None and complete and isinstance(usage, dict):
        require(usage['completion_tokens'] == len(ids), 'Generation actual token length/usage mismatch')
    out = {'record_id': record_id, 'record_kind': 'free_generation', 'source_id': spec['source_id'],
           'source_identity': source_identity, 'bundle_identity': bundle_identity,
           'source_adapter': spec['adapter'], 'source_plan_identity': plan['plan_identity'],
           'raw_manifest_sha256': manifest_sha, 'raw_manifest_verified': True,
           'task_id': sample['task_id'], 'task_version': sample['task_version'], 'sample_id': sample['sample_id'],
           'sample_identity': sample['sample_identity'], 'policy_id': policy, 'arm_id': arm['arm_id'],
           'identity': identity, **identity,
           'request_id': rid, 'execution_id': record.get('execution_id') if record else None,
           'record_status': 'OBSERVED' if record else 'NOT_RUN',
           'evidence_kind': record.get('evidence_kind') if record else plan.get('evidence_kind'),
           'execution_mode': record.get('execution_mode') if record else None,
           'prompt_sequence_identity': digest(prepared['generation_input_ids']),
           'teacher_forced_sequence_identity': prepared.get('sequence_identity'),
           'generated_sequence_identity': digest({'ids': ids, 'token_id_space': identity['token_id_space']}) if ids is not None else None,
           'output_text': record.get('output_text') if record else None, 'output_token_ids': ids,
           'output_token_provenance': provenance, 'output_ids_available': ids is not None,
           'actual_output_token_count': len(ids) if ids is not None else None,
           'usage_completion_tokens': usage.get('completion_tokens') if isinstance(usage, dict) else None,
           'length_provenance': 'actual saved server token IDs' if ids is not None else 'usage-only; token positions unavailable' if usage else 'NOT_OBSERVED',
           'transport_complete': complete, 'termination': record.get('termination', 'UNKNOWN') if record else 'NOT_RUN',
           'finish_reason': record.get('finish_reason') if record else None,
           'request_latency_http': record.get('request_latency_http') if record else None,
           'request_latency_task': record.get('request_latency_task') if record else None,
           'task_score': score, 'task_success': score.get('task_success'),
           'correctness_source': 'original task pure scorer; never output string/token agreement',
           'reference_source': record.get('reference_source') if record else None,
           'reference_used_for_generation': record.get('reference_used_for_generation') if record else None,
           'censoring_policy': 'LENGTH_CAP/EOF/incomplete are distinct from observed NATURAL_STOP; no EOS inferred',
           'raw_reference': record.get('_c1_raw_reference') if record else None}
    return out


def collect_generations(spec, plan, source_identity):
    """Return source EVAL free generations plus explicit planned NOT_RUN rows.

    ``spec`` names a local verified run with adapter ``bench05b2`` or
    ``bench05b4``, ``path``, ``source_id`` and optional ``policies`` / ``splits``.
    This function verifies manifest ancestry itself and does not trust an
    unverified summary. Paths/archives must already be safely extracted by the
    caller; absolute raw-member paths or traversal are never followed.
    """
    require(isinstance(spec, dict) and spec.get('source_id'), 'Missing generation source ID')
    adapter = spec.get('adapter'); require(adapter in ('bench05b2', 'bench05b4'), 'No free-generation adapter for source; B3 audit is not generation')
    root = Path(spec['path']); _plan_check(root, plan)
    if adapter == 'bench05b2':
        from ..suites.bench05b2_replay import verify_raw
        from ..tasks import tech_record as task
        from ..tasks import contract_policy as policy_module
        verify_raw(root)
        policies = spec.get('policies', ['P00'])
        require(isinstance(policies, list) and policies and len(set(policies)) == len(policies) and all(v in policy_module.POLICIES for v in policies), 'Unknown/duplicate B2 generation policy')
        require(plan['scorer_identity'] == sha(Path(task.__file__)), 'Historical tech-record scorer bytes changed')
        expected_task = (task.TASK, task.VERSION)
    else:
        from ..suites.bench05b1_replay import verify_raw
        from ..suites.bench05b4_replay import decode_generations
        from ..tasks import context_evidence as task
        verify_raw(root, scope='bench05b4_context_raw')
        policies = spec.get('policies', ['FIXED'])
        require(policies == ['FIXED'], 'B4 has one fixed prompt policy')
        require(plan['scorer_identity'] == sha(Path(task.__file__)), 'Historical context-evidence scorer bytes changed')
        expected_task = (task.TASK, task.VERSION)
    require((plan['task_id'], plan['task_version']) == expected_task, 'Generation task source mismatch')
    splits = spec.get('splits', ['EVAL_LOCKED']); require(isinstance(splits, list) and splits and len(set(splits)) == len(splits), 'Invalid generation split selection')
    source_rows = plan['data']['rows']; require(len({s['sample_id'] for s in source_rows}) == len(source_rows), 'Duplicate generation source document')
    sample_ids = spec.get('sample_ids')
    if sample_ids is not None:
        require(isinstance(sample_ids, list) and sample_ids and all(isinstance(sid, str) and sid for sid in sample_ids)
                and len(set(sample_ids)) == len(sample_ids), 'Invalid/duplicate generation sample filter')
        require(set(sample_ids) <= {s['sample_id'] for s in source_rows}, 'Unknown generation sample filter')
    selected_samples = [s for s in source_rows if s['split'] in splits and (sample_ids is None or s['sample_id'] in sample_ids)]
    require(selected_samples and all(v in {s['split'] for s in source_rows} for v in splits), 'Missing selected generation split')
    progress = read(root / 'progress.json'); manifest_sha = sha(root / 'RAW_MANIFEST.json')
    prepared_rows = plan['samples']['generation_rows']; outputs = []
    for arm in plan['arms']:
        aid = arm['arm_id']; job = progress['jobs'].get('serve-' + aid); by_key = {}
        if adapter == 'bench05b2':
            prepared = {(r['sample_id'], r['policy_id']): r for r in prepared_rows}
            require(len(prepared) == len(prepared_rows), 'Duplicate B2 prepared sample-policy')
            if job:
                directory = root / safe_name(job['directory']); path = directory / 'generations.jsonl'
                records = jsonl(path) if path.is_file() else []
                require(len({r['request_id'] for r in records}) == len(records), 'Duplicate B2 raw request ID')
                require(len({(r['sample_id'], r['policy_id']) for r in records}) == len(records), 'Duplicate B2 raw sample-policy')
                require([(r['sample_id'], r['policy_id']) for r in records] == list(prepared)[:len(records)], 'B2 request order changed')
                for record in records:
                    key = (record['sample_id'], record['policy_id']); require(key in prepared, 'Unknown B2 source sample-policy')
                    if key[1] not in policies: continue
                    row = prepared[key]
                    require(record['policy_identity'] == row['policy_identity'] and record['prepared_identity'] == digest(row), 'B2 prepared/policy identity mismatch')
                    require(record['scorer_identity'] == plan['scorer_identity'] and record['runtime_identity'] == plan['serving_runtime_identity'], 'B2 scorer/runtime changed')
                    value = generation_from_raw(record, row, arm, plan)
                    value['_c1_raw_reference'] = str(path.relative_to(root)) + '#request_id=' + record['request_id']
                    by_key[key] = value
        else:
            prepared = {(r['sample_id'], 'FIXED'): r for r in prepared_rows}
            require(len(prepared) == len(prepared_rows), 'Duplicate B4 prepared sample')
            for record in decode_generations(root, job, plan, arm):
                key = (record['sample_id'], 'FIXED'); require(key in prepared and key not in by_key, 'Unknown/duplicate B4 generation sample')
                record['_c1_raw_reference'] = job['directory'] + ('/request-journal.jsonl#request_id=' if record.get('recovered_from_submission_journal') else '/task-generations.jsonl#request_id=') + record['request_id']
                by_key[key] = record
        for sample in selected_samples:
            for policy in policies:
                row = prepared[(sample['sample_id'], policy)]; raw = by_key.get((sample['sample_id'], policy))
                result = task.score(sample, raw)
                outputs.append(_output(spec, plan, source_identity, sample, row, arm, policy, raw, result, manifest_sha))
    require(len({r['record_id'] for r in outputs}) == len(outputs), 'Global generation record ID collision')
    return outputs
