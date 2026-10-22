"""Native choice-likelihood scalars from a verified Suite into the C1 view.

Request decoding, tokenizer boundaries and scoring remain in the existing task
adapter.  This adapter copies target scalars and exact prepared coordinates;
it never reconstructs uncollected vocabulary distributions or loads a runtime.
"""
from copy import deepcopy

from ..core.io import digest, require, sha, write
from .tokens import jsonl, metric_contract, source_identity, write_jsonl
from .token_sources import _sequence, _merge_sequence


KIND = 'native_choice_likelihood_v1'


def _vocabulary_axis(metadata):
    tokenizer = metadata['vocabulary_size']; model = metadata.get('embedding_vocabulary_size', tokenizer)
    require(type(tokenizer) is int and type(model) is int and 1 < tokenizer <= model,
            'Invalid native tokenizer/model vocabulary axes')
    configured = metadata['model_config'].get('vocab_size')
    require(configured is None or configured == model, 'Native probability/model vocabulary axis differs')
    return tokenizer, model


def export_native_view(root, context, sources, compiled, manifest, scores, output):
    """Export all planned continuation positions, including missing evidence.

    ``scores`` must be produced by ``suite_results.score_tasks`` in the same
    analysis call.  The Suite entry point verifies the immutable parent first.
    """
    from .suite_results import _arms, _selected, _sample_jobs
    output.mkdir(); metadata = sources['model_metadata']; arms = _arms(sources, compiled)
    tokenizer_vocabulary, model_vocabulary = _vocabulary_axis(metadata)
    selected = [(s, r) for s, r in _selected(sources, compiled) if s['request_kind'] == 'choice_loglikelihood']
    require(selected, 'INVALID_EMPTY_PLAN: native likelihood token source is empty')
    _, jobs = _sample_jobs(context, arms, selected)
    native = {}; choices = {}
    for row in jsonl(scores / 'native-tokens.jsonl'):
        key = (row['arm_id'], row['task_id'], row['sample_id'], row['choice_index'], row['token_position'])
        require(key not in native, 'Duplicate native continuation observation'); native[key] = row
    for row in jsonl(scores / 'choice-scores.jsonl'):
        key = (row['arm_id'], row['task_id'], row['sample_id'], row['choice_index'])
        require(key not in choices, 'Duplicate native choice result'); choices[key] = row
    records = []; sequences = {}; expected = set(); raw_sha = sha(root / 'RUN_MANIFEST.json')
    panel = 'suite-native-choice-likelihood'
    for source, sample in selected:
        authored = sample['authored_sample']
        for arm_id, arm in arms.items():
            group, job = jobs[(arm_id, source['task_id'], sample['sample_id'])]
            require(group is not None and job is not None, 'Native token source lacks a serving execution unit')
            for request in authored['requests']:
                choice = request['choice_index']; ids = request['input_ids']; start = len(request['context_ids'])
                require(start > 0 and start < len(ids) and ids == request['context_ids'] + request['continuation_ids'],
                        'Native continuation boundary/input mismatch')
                effective_prompt = request['context'].rstrip()
                reference = request['context'][len(effective_prompt):] + request['continuation']
                scope = 'GOLD_CHOICE' if choice == authored['gold'] else 'NON_GOLD_CHOICE'
                sequence_row = {'sample_id': sample['sample_id'], 'input_ids': ids,
                    'prompt_token_boundary': start, 'prompt': effective_prompt, 'reference_text': reference,
                    'full_text': effective_prompt + reference, 'sequence_identity': digest(ids),
                    'reference_source': 'dataset_answer_key_choice' if choice == authored['gold'] else 'dataset_choice_continuation',
                    'reference_role': scope}
                sequence = _sequence(panel, sequence_row, None, choice); _merge_sequence(sequences, sequence)
                choice_result = choices.get((arm_id, source['task_id'], sample['sample_id'], choice))
                for position in range(start, len(ids)):
                    key = (arm_id, source['task_id'], sample['sample_id'], choice, position); expected.add(key)
                    measured = native.get(key); prefix = digest({'token_ids': ids[:position], 'token_id_space': metadata['token_id_space']})
                    if measured is not None:
                        require(measured['token_id'] == ids[position] and measured['logit_position'] == position - 1 and
                                measured['prefix_hash'] == prefix and measured['token_id_space'] == metadata['token_id_space'],
                                'Native scalar/continuation coordinate mismatch')
                    role = 'gold_choice' if choice == authored['gold'] else 'non_gold_choice'
                    record = {'source_id': panel, 'panel': KIND, 'bundle_identity': raw_sha,
                        'plan_identity': compiled['plan_identity'], 'execution_id': group,
                        'arm_id': arm['source_arm_id'], 'logical_arm_id': arm_id,
                        'task_id': source['task_id'], 'task_version': source['task_version'],
                        'dataset_identity': digest({'source_id': source['source_id'], 'dataset_id': sample['raw_record']['dataset_id']}),
                        'split': sample['raw_record']['split'], 'original_split': sample['raw_record']['split'],
                        'sample_id': sample['sample_id'], 'policy_id': None, 'choice_index': choice,
                        'reference_scope': scope, 'reference_source': sequence_row['reference_source'],
                        'reference_role': scope, 'prompt_hash': digest(effective_prompt), 'reference_hash': digest(reference),
                        'sequence_id': sequence['sequence_id'], 'tokenizer_identity': metadata['tokenizer_identity'],
                        'token_id_space': metadata['token_id_space'], 'vocabulary_size': model_vocabulary,
                        'tokenizer_vocabulary_size': tokenizer_vocabulary,
                        'vocabulary_axis_provenance': 'model embedding/lm_head normalization axis; no full distribution collected',
                        'runtime_identity': metadata['runtime_identities']['serving'], 'backend_policy': 'native',
                        'acquisition_source': 'native_likelihood', 'artifact_identity': arm['artifact']['artifact_identity'],
                        'recipe_identity': arm['artifact'].get('recipe_identity'), 'model_id': metadata['pair']['model_id'],
                        'model_revision': metadata['pair']['revision'], 'probability_stage': 'raw_pre_sampling',
                        'metric_contract': 'target_logprob_v1', 'token_position': position, 'logit_position': position - 1,
                        'target_token_id': ids[position], 'prefix_hash': prefix, 'requested': True,
                        'target_logprob': measured['logprob'] if measured else None,
                        'status': 'MEASURED' if measured else 'FAILED' if choice_result else 'NOT_RUN',
                        'evidence_kind': context.get('evidence_kind', 'live_native_likelihood'),
                        'field_tags': ['continuation', role],
                        'field_memberships': [{'field_key': name, 'origins': [{'kind': 'native_continuation_role', 'role': name}]} for name in ('continuation', role)],
                        'field_membership_policy': 'native_continuation_roles_v1',
                        'request_id': measured['request_id'] if measured else choice_result.get('request_id') if choice_result else None,
                        'observation_id': None, 'raw_argmax_token_id': None, 'full_logits': None,
                        'normalization_dtype': 'SERVER_NOT_EXPOSED', 'raw_logits_dtype': 'NOT_EXPOSED',
                        'effective_weight_dtype': arm.get('effective_weight_dtype', arm.get('dtype', 'NOT_EXPOSED')),
                        'execution_mode': 'NATIVE_ARTIFACT' if arm['source_format'] == 'native' else 'PACKED_ARTIFACT_RUNTIME',
                        'native_full_vocabulary': 'NOT_COLLECTED', 'native_argmax_status': 'NOT_COLLECTED',
                        'offset_mapping_status': 'NOT_COLLECTED_NATIVE_SOURCE', 'audit_generated_token_excluded': True,
                        'raw_reference': measured['raw_reference'] if measured else job['raw_path']}
                    record['record_id'] = digest({'raw_manifest': raw_sha, 'execution_unit': group, 'source_id': source['source_id'],
                        'sample_id': sample['sample_id'], 'choice_index': choice, 'token_position': position, 'adapter': KIND})
                    records.append(record)
    require(native.keys() <= expected, 'Unplanned native continuation observation')
    require(records and len({r['record_id'] for r in records}) == len(records), 'Empty/duplicate native token view')
    write_jsonl(output / 'token_records.jsonl', records); write_jsonl(output / 'sequences.jsonl', list(sequences.values()))
    write_jsonl(output / 'generation_records.jsonl', [])
    coverage = {'status': 'EXPORTED_OBSERVED_NOT_NEW_MEASUREMENTS', 'records': len(records),
        'sequences': len(sequences), 'generation_records': 0, 'native_requested': len(records),
        'native_measured': sum(r['status'] == 'MEASURED' for r in records), 'full_logits_records': 0,
        'full_vocabulary': 'NOT_COLLECTED', 'diagnostic': 'NOT_REQUESTED', 'GPU_actions': 0}
    panels = [{'source_id': panel, 'adapter': KIND, 'raw_manifest_sha256': raw_sha,
        'task_ids': sorted({s['task_id'] for s, _ in selected}), 'coverage': deepcopy(coverage),
        'reference_scopes': ['GOLD_CHOICE', 'NON_GOLD_CHOICE'],
        'probability_contract': 'native echoed prompt continuation target logprob; audit generation excluded'}]
    write(output / 'SOURCES.json', panels); write(output / 'METRIC_CONTRACT.json', metric_contract())
    write(output / 'INPUT_SOURCES.json', {'schema_version': 1, 'kind': KIND, 'parent_raw_manifest_sha256': raw_sha})
    write(output / 'COVERAGE.json', coverage)
    inventory = {p.relative_to(output).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(output.rglob('*')) if p.is_file()}
    view = {'schema_version': 1, 'kind': 'common_token_view_v1', 'analysis_source_digest': source_identity(),
        'input_manifest_sha256': sha(output / 'INPUT_SOURCES.json'), 'source_panels': panels,
        'files': inventory, 'view_identity': digest(inventory), 'new_observations': False, 'GPU_actions': 0}
    write(output / 'TOKEN_VIEW_MANIFEST.json', view)
    return {'status': 'TOKEN_VIEW_EXPORTED', 'view_identity': view['view_identity'], 'coverage': coverage}
