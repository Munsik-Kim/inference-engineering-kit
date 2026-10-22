"""Validate explicitly prepared raw-task inputs without loading a tokenizer.

The manifest is an authority for new preparation, not a historical acquisition.
Its independent prompt-only payload and external reference keep distinct roles.
"""
from copy import deepcopy

from ..core.io import digest, fields, require
from ..tasks.registry import get_task
from ..tasks.input_records import render_record, validate_raw
from .suite_inputs import _hash, _ids, validate_diagnostic

KIND = 'prepared_task_source_v1'
SCOPE = 'PREPARED_CONTRACT_VALIDATED_ONLY'


def _rendered(raw, model_profile=None):
    validate_raw({'schema_version': 1, 'kind': 'raw_task_manifest_v1',
                  'dataset_id': raw['dataset_id'], 'usage_scope': raw['usage_scope'], 'rows': [raw]})
    return render_record(raw, model_profile)


def _generation(sample, source, rendered):
    from ..tasks.input_records import render_pinned_chat
    generation = sample['generation']; diagnostic = sample['diagnostic']
    fields(generation, ('prepared', 'wire_payload', 'preparation_proof'))
    row = generation['prepared']; proof = generation['preparation_proof']; body = generation['wire_payload']
    fields(row, ('sample_id', 'prompt', 'prompt_hash', 'generation_input_ids'))
    fields(proof, ('raw_record_hash', 'task_prompt_hash', 'prompt_hash', 'tokenizer_identity',
                   'reference_used_for_generation', 'independent_prompt_encoding',
                   'chat_template_applied', 'rendered_task_prompt'))
    expected_prompt = render_pinned_chat(rendered['content'], source['model_metadata'].get('capability_profile'))
    require(proof == {'raw_record_hash': digest(sample['raw_record']),
        'task_prompt_hash': digest(rendered['content']), 'prompt_hash': digest(expected_prompt),
        'tokenizer_identity': source['model_metadata']['tokenizer_identity'],
        'reference_used_for_generation': False, 'independent_prompt_encoding': True,
        'chat_template_applied': 1, 'rendered_task_prompt': rendered['content']},
        'Independent generation preparation proof changed')
    require(type(proof['chat_template_applied']) is int and
            proof['reference_used_for_generation'] is False and proof['independent_prompt_encoding'] is True,
            'Invalid independent generation role/template policy')
    require(row['sample_id'] == sample['sample_id'] and row['prompt'] == diagnostic['prompt'] == expected_prompt and
            row['prompt_hash'] == diagnostic['prompt_hash'] == digest(expected_prompt),
            'Generation prompt changed or gold/reference suffix included')
    fields(body, ('prompt', 'max_tokens', 'temperature', 'top_p', 'n', 'stream', 'echo', 'ignore_eos', 'add_special_tokens'),
           ('model', 'stream_options', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'return_token_ids'))
    _ids(row['generation_input_ids'], source['model_metadata']['vocabulary_size'], 'generation')
    require(body['prompt'] == row['generation_input_ids'] == diagnostic['generation_input_ids'] and
            body['prompt'] != diagnostic['input_ids'], 'Generation prompt-only token payload changed/gold suffix leakage')
    require(type(body['max_tokens']) is int and body['max_tokens'] ==
            (192 if source['task_id'] == 'tech_record_extract' else 64) and
            body['stream'] is True and body['echo'] is False and body['ignore_eos'] is False and
            type(body['temperature']) in (int, float) and body['temperature'] == 0 and
            type(body['top_p']) in (int, float) and body['top_p'] == 1 and
            type(body['n']) is int and body['n'] == 1 and body['add_special_tokens'] is False,
            'New generation input must retain existing natural-stop policy')
    for key, default in (('repetition_penalty', 1), ('presence_penalty', 0), ('frequency_penalty', 0)):
        require(key not in body or type(body[key]) in (int, float) and body[key] == default,
                'Generation penalty policy changed')
    require(len(body['prompt']) + body['max_tokens'] <= source['preparation']['limits']['context_tokens'],
            'Serving prompt/output context bound exceeded')
    adapter = get_task(source['task_id'], source['task_version'])
    request = adapter.requests(sample['authored_sample'] | {'generation_input_ids': body['prompt']},
                               'prepared-static-validation', source['model_metadata']['tokenizer_identity'])[0]
    require(request.payload == {'token_ids': body['prompt']}, 'Wrong generation task/payload')


def validate_source(value):
    """Strict CPU validation of one new prepared source; no live assertion."""
    fields(value, ('schema_version', 'kind', 'source_id', 'task_id', 'task_version', 'request_kind',
                   'model_metadata', 'samples', 'preparation', 'validation_scope'), ('source_identity',))
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and value['kind'] == KIND,
            'Unsupported new prepared source')
    require(value['validation_scope'] == SCOPE and isinstance(value['source_id'], str) and value['source_id'],
            'New preparation scope/source ID missing')
    if 'source_identity' in value:
        require(value['source_identity'] == digest({k: v for k, v in value.items() if k != 'source_identity'}),
                'Prepared source identity changed')
    adapter = get_task(value['task_id'], value['task_version'])
    require(value['request_kind'] == adapter.request_kind, 'Prepared task request kind changed')
    metadata = value['model_metadata']; preparation = value['preparation']
    require(type(metadata['vocabulary_size']) is int and metadata['vocabulary_size'] > 1, 'Invalid vocabulary')
    for name in ('tokenizer_identity', 'token_id_space'):
        _hash(metadata[name], name)
    require(isinstance(metadata.get('tokenizer_files'), dict) and metadata['tokenizer_files'] and
            metadata['tokenizer_identity'] == digest(metadata['tokenizer_files']),
            'Prepared tokenizer file identity changed')
    fields(preparation, ('schema_version', 'policy_id', 'raw_manifest_identity', 'preparation_spec_identity',
                        'tokenizer_identity', 'token_id_space', 'scorer_version', 'limits', 'selection', 'source_identity'))
    require(type(preparation['schema_version']) is int and preparation['schema_version'] == 1 and
            preparation['policy_id'] == 'existing_tasks_local_tokenizer_v1', 'Unsupported preparation policy')
    for name in ('raw_manifest_identity', 'preparation_spec_identity', 'source_identity'):
        _hash(preparation[name], name)
    require(preparation['scorer_version'] == adapter.scorer_version and
            all(preparation[k] == metadata[k] for k in ('tokenizer_identity', 'token_id_space')),
            'Preparation scorer/tokenizer contract changed')
    fields(preparation['limits'], ('max_prompt_tokens', 'max_reference_tokens', 'max_joint_tokens', 'context_tokens'))
    require(all(type(v) is int and v > 0 for v in preparation['limits'].values()) and
            preparation['limits']['max_prompt_tokens'] <= 512 and
            preparation['limits']['max_reference_tokens'] <= 128 and
            preparation['limits']['max_joint_tokens'] <= 640 and
            preparation['limits']['context_tokens'] == 2048, 'Invalid preparation length bounds')
    selection = preparation['selection']; fields(selection, ('policy_id', 'max_targets', 'max_full'))
    require(selection['policy_id'] == 'field_round_robin_uniform_v1' and
            type(selection['max_targets']) is int and 1 <= selection['max_targets'] <= 32 and
            type(selection['max_full']) is int and 0 <= selection['max_full'] <= min(8, selection['max_targets']),
            'Invalid prepared reference selection')
    samples = value['samples']; require(isinstance(samples, list) and samples, 'INVALID_EMPTY_PLAN: prepared samples empty')
    require(len({s['sample_id'] for s in samples}) == len(samples), 'Duplicate prepared sample')
    for sample in samples:
        branch = 'generation' if adapter.request_kind == 'generate_until' else 'likelihood'
        fields(sample, ('sample_id', 'raw_record', 'authored_sample', 'diagnostic', branch))
        raw = sample['raw_record']; rendered = _rendered(raw, metadata.get('capability_profile')); authored = sample['authored_sample']
        row = validate_diagnostic(sample['diagnostic'], metadata)
        require((row['task_id'], row['task_version'], row['sample_id']) ==
                (value['task_id'], value['task_version'], sample['sample_id']) and
                (raw['task_id'], raw['task_version'], raw['sample_id']) ==
                (value['task_id'], value['task_version'], sample['sample_id']), 'Prepared task/sample namespace mismatch')
        limits = preparation['limits']
        require(len(row['input_ids']) <= limits['max_joint_tokens'] and
                len(row['targets']) <= selection['max_targets'] and
                len(row['full_vocab_target_positions']) <= selection['max_full'], 'Prepared sequence/selection bound exceeded')
        require(row['reference_role'] == rendered['reference_role'] and row['reference_source'] == raw['reference_source'],
                'Prepared reference role changed')
        if adapter.request_kind == 'generate_until':
            require(authored == rendered['authored_sample'], 'Authored source/gold changed')
            require(row['reference_text'] == rendered['reference_text'], 'External gold reference changed')
            require(row.get('document_hash') == authored['document_hash'] and
                    row.get('sample_identity') == authored['sample_identity'], 'Authored document/sample identity changed')
            require(row.get('value_spans') == rendered['value_spans'], 'Reference literal spans changed')
            require(len(row['generation_input_ids']) <= limits['max_prompt_tokens'] and
                    len(row['reference_separate_ids']) <= limits['max_reference_tokens'],
                    'Prepared prompt/reference bound exceeded')
            _generation(sample, value, rendered)
        else:
            expected_authored = deepcopy(rendered['authored_sample']); expected_authored['requests'] = authored['requests']
            if 'request_identity' in authored:
                require(authored['request_identity'] == digest(authored['requests']), 'ARC request identity changed')
                expected_authored['request_identity'] = authored['request_identity']
            require(authored == expected_authored, 'ARC authored document/gold contract changed')
            likelihood = sample['likelihood']; fields(likelihood, ('requests', 'choice_labels', 'gold_index'))
            require(likelihood == {'requests': authored['requests'], 'choice_labels': authored['choice_labels'],
                                   'gold_index': authored['gold']}, 'ARC all-choice order/source changed')
            requests = adapter.requests(authored, 'prepared-static-validation', metadata['tokenizer_identity'], metadata['token_id_space'])
            require(len(requests) == len(authored['choices']) and
                    [r.payload['choice_index'] for r in requests] == list(range(len(requests))), 'Incomplete ARC choices')
            for request in requests:
                _ids(request.payload['input_ids'], metadata['vocabulary_size'], 'ARC choice')
                require(len(request.payload['input_ids']) + 1 <= limits['context_tokens'], 'ARC serving context bound exceeded')
            gold = requests[authored['gold']].payload; prompt = gold['context'].rstrip()
            reference = gold['context'][len(prompt):] + rendered['reference_text']
            require(row['prompt'] == prompt and row['reference_text'] == reference and
                    row['input_ids'] == gold['input_ids'] and row['prompt_token_boundary'] == len(gold['context_ids']) and
                    type(row['gold_choice_index']) is int and row['gold_choice_index'] == authored['gold'],
                    'ARC diagnostic/all-choice boundary changed')
    return value
