"""Explicit raw task records for CPU preparation; no tokenizer or model imports.

Renderers only receive model-visible task fields. Gold validation is separate.
This module supports existing contracts, not arbitrary templates or code loaders.
"""
from copy import deepcopy
import math
import re
from ..core.io import digest, fields, require
from .registry import get_task

KIND = 'raw_task_manifest_v1'
ARC_PROMPT_CONTRACT = 'arc_original_completion_v1'
SUPPORTED = {'tech_record_extract': '0.1', 'context_evidence_retrieval': '0.1', 'arc_challenge': '1.0'}
ROW_KEYS = ('task_id', 'task_version', 'dataset_id', 'source_id', 'sample_id', 'split', 'usage_scope', 'policy_id', 'reference_source', 'sample')
TECH_KEYS = ('task_id', 'task_version', 'sample_id', 'source_id', 'template_family', 'case_type', 'language', 'split', 'experiment_id', 'events', 'document', 'document_hash', 'gold', 'sample_identity')
RETRIEVAL_KEYS = ('task_id', 'task_version', 'data_version', 'sample_id', 'source_id', 'source_family', 'template_family', 'split', 'language', 'case_type', 'entity', 'relation', 'question', 'passages', 'document', 'document_hash', 'gold', 'authored_facts', 'sample_identity')


def _plain(value):
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), 'Raw object keys must be strings')
        for child in value.values(): _plain(child)
    elif isinstance(value, list):
        for child in value: _plain(child)
    else:
        require(value is None or type(value) in (str, int, float, bool), 'Non-JSON raw value')
        require(type(value) is not float or math.isfinite(value), 'Nonfinite raw value')


def _text(value, name):
    require(isinstance(value, str) and bool(value), 'Missing string: ' + name)


QWEN_PROFILE = 'qwen3_4b_gptq_serving_v1'


def render_pinned_chat(content, model_profile=None):
    """Reviewed model template assertion; SDK runs only in prepare worker.

    The legacy default remains the frozen Smol template.  The explicit Qwen
    profile uses its pinned user-only template, without an invented system or
    thinking option.  These are assertions of saved template output, not an
    arbitrary template execution interface.
    """
    _text(content, 'task content')
    if model_profile is not None:
        require(model_profile == QWEN_PROFILE, 'Unsupported preparation model profile')
        return '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n'
    return ('<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n'
            '<|im_start|>user\n' + content + '<|im_end|>\n<|im_start|>assistant\n')


def arc_context(question, model_profile=None):
    """The already-rendered Smol completion prompt preserved by the ARC recipe."""
    _text(question, 'ARC question')
    return render_pinned_chat('Question: ' + question + '\nAnswer:', model_profile)


def sample_key(row):
    return (row['task_id'], row['task_version'], row['dataset_id'], row['sample_id'])


def _validate_sample(row):
    task = row['task_id']; sample = row['sample']
    require(isinstance(sample, dict), 'Raw sample must be an object')
    if task == 'tech_record_extract':
        from . import tech_record, contract_policy
        fields(sample, TECH_KEYS, ('source_family',))
        for k in ('experiment_id', 'document', 'document_hash', 'language', 'case_type'): _text(sample[k], k)
        require(sample['language'] in ('en', 'ko') and sample['case_type'] in tech_record.CASES, 'Unsupported tech language/case')
        fields(sample['gold'], ('experiment_id', 'temperature_c', 'status', 'support'))
        fields(sample['gold']['support'], ('temperature_c', 'status'))
        require(sample['gold']['temperature_c'] is None or type(sample['gold']['temperature_c']) in (int, float), 'Invalid temperature gold type')
        require(sample['gold']['status'] is None or sample['gold']['status'] in ('complete', 'repeat', 'pending'), 'Invalid status gold')
        require(isinstance(sample['events'], list) and sample['events'], 'Empty authored events')
        require(len({e.get('line_id') for e in sample['events'] if isinstance(e, dict)}) == len(sample['events']), 'Duplicate authored line IDs')
        for event in sample['events']:
            require(isinstance(event, dict) and event.get('field') in ('note', 'temperature_c', 'status'), 'Unknown event field')
            require(type(event.get('line_id')) is str and re.fullmatch(r'L[1-9][0-9]*', event['line_id']), 'Invalid authored line ID')
            if event['field'] == 'note': fields(event, ('field', 'text', 'line_id')); _text(event['text'], 'note')
            else:
                fields(event, ('field', 'experiment_id', 'value', 'line_id'), ('final',))
                require('final' not in event or type(event['final']) is bool, 'Invalid final flag')
                require(event['field'] != 'temperature_c' or type(event['value']) in (int, float), 'Invalid temperature event type')
        for support in sample['gold']['support'].values():
            require(isinstance(support, list) and all(type(v) is str for v in support) and len(support) == len(set(support)), 'Invalid support IDs')
        pipe = bool(re.search(r'^\[L[0-9]+\] (?:Record|Final revision|기록|Final) EXP-[0-9]+ \| ', sample['document'], re.MULTILINE))
        event_renderer = contract_policy._render_event if pipe else tech_record._line
        require(sample['document'] == '\n'.join('[' + e['line_id'] + '] ' + event_renderer(e, sample['language']) for e in sample['events']), 'Authored event/document mismatch')
        verifier = contract_policy.verify_gold if pipe else tech_record.verify_gold
        verifier(sample)
    elif task == 'context_evidence_retrieval':
        from . import context_evidence
        fields(sample, RETRIEVAL_KEYS)
        require(sample['language'] in ('en', 'ko') and sample['case_type'] in context_evidence.CASES, 'Unsupported retrieval language/case')
        fields(sample['gold'], ('answer', 'evidence', 'answerable', 'content_span', 'answer_span'))
        require(type(sample['gold']['answerable']) is bool, 'Invalid answerable flag')
        for key in ('content_span', 'answer_span'):
            span = sample['gold'][key]
            require(span is None or isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span), 'Invalid gold span type')
        require(isinstance(sample['passages'], list), 'Missing passages')
        for p in sample['passages']:
            fields(p, ('passage_id', 'content', 'content_span'))
            require(isinstance(p['content_span'], list) and len(p['content_span']) == 2 and all(type(v) is int for v in p['content_span']), 'Invalid passage span')
        context_evidence.verify_gold(sample)
    else:
        fields(sample, ('document', 'prompt_context', 'prompt_contract_id', 'original_index'))
        doc = sample['document']; fields(doc, ('id', 'question', 'choices', 'answerKey'))
        fields(doc['choices'], ('text', 'label'))
        require(type(sample['original_index']) is int and sample['original_index'] >= 0, 'Invalid original ARC index')
        require(sample['prompt_contract_id'] == ARC_PROMPT_CONTRACT, 'Unsupported ARC prompt contract')
        require(sample['prompt_context'] == arc_context(doc['question']), 'ARC original context changed/double template')
        require(doc['id'] == row['sample_id'], 'ARC document/sample mismatch')
        texts = doc['choices']['text']; labels = doc['choices']['label']
        require(isinstance(texts, list) and len(texts) >= 2 and all(type(x) is str and x for x in texts), 'Empty ARC choice')
        require(isinstance(labels, list) and len(labels) == len(texts) and all(type(x) is str and x for x in labels) and len(set(labels)) == len(labels), 'Invalid ARC choice labels')
        require(doc['answerKey'] in labels, 'ARC answerKey absent')
        require(row['usage_scope'] == 'REGRESSION_REUSE', 'Existing ARC must retain REGRESSION_REUSE')
        return
    require((sample['task_id'], sample['task_version'], sample['sample_id'], sample['source_id'], sample['split']) ==
            (row['task_id'], row['task_version'], row['sample_id'], row['source_id'], row['split']), 'Authored/raw task namespace mismatch')
    require(sample['sample_identity'] == digest({k: v for k, v in sample.items() if k != 'sample_identity'}), 'Authored sample identity changed')


def validate_raw(manifest):
    fields(manifest, ('schema_version', 'kind', 'dataset_id', 'usage_scope', 'rows'))
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and manifest['kind'] == KIND, 'Unsupported raw task manifest')
    _plain(manifest); _text(manifest['dataset_id'], 'manifest dataset_id'); _text(manifest['usage_scope'], 'manifest usage_scope')
    rows = manifest['rows']; require(isinstance(rows, list) and rows, 'INVALID_EMPTY_PLAN: raw samples empty')
    seen = set()
    for row in rows:
        fields(row, ROW_KEYS)
        for key in ROW_KEYS[:-1]: _text(row[key], key)
        require(row['task_id'] in SUPPORTED and row['task_version'] == SUPPORTED[row['task_id']], 'Unsupported raw task/version')
        require(sample_key(row) not in seen, 'Duplicate raw sample namespace'); seen.add(sample_key(row))
        require(row['policy_id'] == ('P00' if row['task_id'] == 'tech_record_extract' else 'arc_original_completion_v1' if row['task_id'] == 'arc_challenge' else 'context_evidence_fixed_v1'), 'Unsupported prompt policy')
        expected = {'tech_record_extract': 'task_author_gold_json', 'context_evidence_retrieval': 'task_author_gold_answer_evidence_v1', 'arc_challenge': 'dataset_answer_key_choice'}[row['task_id']]
        require(row['reference_source'] == expected, 'External reference source mismatch')
        _validate_sample(row)
    return manifest


def render_record(row, model_profile=None):
    """Render model content independently; canonical gold is a separate return field.

    Validation belongs to validate_raw. Keeping rendering independent permits the
    explicit hidden-gold-change contract test without making invalid data usable.
    """
    task = row['task_id']; sample = deepcopy(row['sample']); adapter = get_task(task, row['task_version'])
    if task == 'arc_challenge':
        doc = sample['document']; labels = doc['choices']['label']; choices = doc['choices']['text']; gold = labels.index(doc['answerKey'])
        content = arc_context(doc['question'], model_profile)
        authored = {'sample_id': row['sample_id'], 'question': doc['question'], 'choices': deepcopy(choices), 'choice_labels': deepcopy(labels),
                    'document': deepcopy(doc), 'gold': gold, 'original_index': sample['original_index'], 'split': row['split'],
                    'usage_scope': row['usage_scope'], 'prompt': content, 'source_text_hash': digest(doc)}
        reference = ' ' + choices[gold]
        extra = {'arc_context': content, 'prompt_rendering': 'already_rendered', 'reference_role': 'dataset_gold_choice', 'value_spans': {'gold_choice': [0, len(reference)]}}
    else:
        authored = sample; content = adapter.render_prompt(sample); ref = adapter.reference(sample)
        reference = ref.get('text', ref.get('reference_text'))
        extra = {'prompt_rendering': 'task_user_content', 'reference_role': 'author_gold_json' if task == 'tech_record_extract' else 'author_answer_evidence', 'value_spans': ref['value_spans']}
    return {k: deepcopy(row[k]) for k in ROW_KEYS[:-1]} | {
        'authored_sample': authored, 'content': content, 'task_prompt': content, 'reference_text': reference, 'request_kind': adapter.request_kind,
        'scorer_version': adapter.scorer_version, **extra}
