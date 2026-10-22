"""Explicit local tokenizer-only worker for versioned prepared task sources.

Only ``run`` imports a tokenizer SDK.  The pure helpers accept an already supplied
fast tokenizer and never load a model, start inference, or inspect a GPU.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import sys
import unicodedata

from ..core.io import digest, read, require, sha, write
from ..adapters.protocol import Adapter
from ..suites.bench05a_plan import source_identity
from .contracts import Sample
from .prepare_tokens import joint_encoding
from .task_tokens import field_targets
from .arc_tokens import effective_text
from .retrieval_tokens import content_span_records

POLICY = 'existing_tasks_local_tokenizer_v1'
SELECTION = 'field_round_robin_uniform_v1'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'chat_template.jinja', 'vocab.json', 'merges.txt', 'tokenizer.model')


def _uniform(values, maximum):
    count = min(len(values), maximum)
    if not count:
        return []
    return sorted({values[round(i * (len(values) - 1) / max(1, count - 1))]
                   for i in range(count)})


def _offsets(text, ids, offsets, special):
    require(len(ids) == len(offsets) == len(special), 'Offset/mask token coverage differs')
    previous = -1
    result = []
    overlaps = []
    for i, pair in enumerate(offsets):
        require(isinstance(pair, (tuple, list)) and len(pair) == 2 and
                all(type(n) is int for n in pair), 'Invalid tokenizer offset type')
        a, b = pair
        require(0 <= a <= b <= len(text) and a >= previous,
                'UNSUPPORTED_TOKEN_OFFSET: backend offsets are not codepoint intervals')
        if i and a < offsets[i - 1][1] and a < b:
            overlaps.append(i)
        require(a != b or special[i], 'UNSUPPORTED_TOKEN_OFFSET: zero-length ordinary token')
        previous = a
        result.append([a, b])
    byte_offsets = [[len(text[:a].encode('utf-8')), len(text[:b].encode('utf-8'))]
                    for a, b in result]
    return result, byte_offsets, overlaps


def _qwen_normalizer(tokenizer):
    normalizer = getattr(getattr(tokenizer, 'backend_tokenizer', None), 'normalizer', None)
    require(normalizer is not None and str(normalizer) == 'NFC()',
            'UNSUPPORTED_TOKEN_OFFSET: Qwen requires the pinned NFC normalizer')
    return normalizer


def _qwen_prompt_offsets(tokenizer, prompt, reference, offsets):
    """Keep native coordinates; never invent positions for normalized marks."""
    normalizer = _qwen_normalizer(tokenizer)
    text = prompt + reference
    require(normalizer.normalize_str(reference) == reference,
            'UNSUPPORTED_TOKEN_OFFSET: reference normalization changes target span')
    normalized = normalizer.normalize_str(text)
    require(normalized == unicodedata.normalize('NFC', text),
            'UNSUPPORTED_TOKEN_OFFSET: unknown normalization transformation')
    missing = [i for i in range(len(text)) if not any(a <= i < b for a, b in offsets)]
    require(all(i < len(prompt) and unicodedata.combining(text[i]) for i in missing),
            'UNSUPPORTED_TOKEN_OFFSET: native offsets omit non-normalization text')
    changed = normalizer.normalize_str(prompt) != prompt
    require(changed or not missing, 'UNSUPPORTED_TOKEN_OFFSET: unexplained native offset gap')
    return {'status': 'PARTIAL_NATIVE_OFFSETS_NORMALIZATION' if changed else 'NATIVE_OFFSETS_UNCHANGED',
      'backend_normalizer': 'NFC', 'normalizer_changed_prompt': changed,
      'normalizer_changed_reference': False, 'original_text_hash': digest(text),
      'backend_normalized_text_hash': digest(normalized), 'unmapped_codepoint_positions': missing,
      'native_offsets_preserved': True, 'invented_mapping': False}


def tokenizer_offset_probe(tokenizer, model_profile=None):
    """Small actual-backend coordinate check; no Unicode normalization is applied.

    Repeated literals use all known authored spans, rather than selecting a first
    matching answer occurrence. Overlapping byte-level offsets retain whole tokens.
    """
    text = '한글 e\u0301🙂\r\nTOKEN TOKEN \t끝\n'
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        return_special_tokens_mask=True)
    ids = list(encoded['input_ids']); special = list(encoded['special_tokens_mask'])
    cp, byte, overlaps = _offsets(text, ids, encoded['offset_mapping'], special)
    ranges = [('korean', [0, 2]), ('combining_sequence', [3, 5]),
              ('emoji', [5, 6]), ('crlf', [6, 8]),
              ('repeated_literal_first', [8, 13]), ('repeated_literal_second', [14, 19])]
    expected = {'korean': '한글', 'combining_sequence': 'e\u0301', 'emoji': '🙂',
                'crlf': '\r\n', 'repeated_literal_first': 'TOKEN', 'repeated_literal_second': 'TOKEN'}
    spans = []; partial = False
    for name, (a, b) in ranges:
        require(text[a:b] == expected[name], 'Authored Unicode offset probe changed')
        positions = [t for t, (left, right) in enumerate(cp) if left < b and right > a]
        missing = [point for point in range(a, b) if not any(left <= point < right for left, right in cp)]
        if missing:
            from ..tasks.input_records import QWEN_PROFILE
            require(model_profile == QWEN_PROFILE and name == 'combining_sequence' and missing == [4] and
                    _qwen_normalizer(tokenizer).normalize_str(text) == unicodedata.normalize('NFC', text),
                    'UNSUPPORTED_TOKEN_OFFSET: actual backend lost authored codepoints')
            partial = True
        require(positions, 'UNSUPPORTED_TOKEN_OFFSET: authored span has no native token overlap')
        record = {'kind': name, 'text': text[a:b], 'codepoint_span': [a, b],
          'utf8_byte_span': [len(text[:a].encode('utf-8')), len(text[:b].encode('utf-8'))],
          'overlapping_token_positions': positions}
        if partial and missing:
            record.update(status='MAPPING_UNAVAILABLE_NORMALIZATION', unmapped_codepoint_positions=missing,
                          token_mapping='native overlap only; complete authored span mapping unavailable')
        spans.append(record)
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    normalizer = getattr(backend, 'normalizer', None) if backend is not None else None
    return {'schema_version': 1, 'kind': 'actual_tokenizer_offset_contract_v1',
      'status': 'PARTIAL_FIXED_PROBE_NORMALIZATION' if partial else 'VERIFIED_FOR_FIXED_PROBE', 'text': text, 'text_hash': digest(text),
      'input_ids': ids, 'token_strings': list(tokenizer.convert_ids_to_tokens(ids)),
      'offsets_codepoints': cp, 'offsets_utf8_bytes': byte, 'special_token_mask': special,
      'overlapping_offset_token_positions': overlaps, 'authored_spans': spans,
      'backend_offset_unit': ('native_unicode_codepoints_partial_NFC_range_unmapped' if partial else
                              'unicode_codepoints_verified_against_authored_non_ascii_ranges'),
      'preparer_unicode_normalization_applied': False,
      'backend_normalizer_class': type(normalizer).__name__ if normalizer is not None else None,
      'scope': 'fixed Korean, NFD combining character, emoji, CRLF, whitespace and repeated literal; not arbitrary backend certification'}


def _source_document_span(rendered, row):
    """Map an unambiguous complete authored document, never a first answer hit."""
    document = rendered['authored_sample'].get('document')
    if not isinstance(document, str):
        return {'status': 'NOT_APPLICABLE', 'reason': 'source document is not a single model-visible text segment'}
    if row.get('backend_normalization', {}).get('normalizer_changed_prompt'):
        return {'status': 'MAPPING_UNAVAILABLE_NORMALIZATION', 'kind': 'source_document_content',
          'text_hash': digest(document),
          'reason': 'Pinned NFC backend changes the prompt; native offsets are preserved but no complete source-span token mapping is asserted'}
    content = rendered['content']
    if content.count(document) != 1:
        return {'status': 'AMBIGUOUS_SPAN', 'reason': 'whole source document is not a unique renderer segment'}
    tail = '<|im_end|>\n<|im_start|>assistant\n'
    prompt = row['prompt']
    require(prompt.endswith(content + tail), 'Source document chat wrapper differs')
    prefix_length = len(prompt) - len(content) - len(tail)
    a = content.index(document); b = a + len(document); pa, pb = prefix_length + a, prefix_length + b
    require(prompt[pa:pb] == document, 'Whole document/chat text mapping differs')
    positions = [i for i, (left, right) in enumerate(row['offsets_codepoints'])
                 if left < pb and right > pa and right > left]
    require(positions and all(i < row['prompt_token_boundary'] for i in positions),
            'Whole source document has no prompt-only token coverage')
    return {'status': 'AVAILABLE', 'kind': 'source_document_content',
      'text_hash': digest(document), 'task_prompt_codepoint_span': [a, b],
      'prompt_codepoint_span': [pa, pb],
      'prompt_utf8_byte_span': [len(prompt[:pa].encode('utf-8')), len(prompt[:pb].encode('utf-8'))],
      'overlapping_token_positions': positions, 'coordinate_space': 'joint_input_ids',
      'mapping_policy': 'unique complete authored document segment in exact pinned task/chat rendering; whole token overlap'}


def _diagnostic(rendered, prompt, reference, tokenizer, metadata, spec, *, moved=''):
    limits = spec['limits']; maximum = limits['max_joint_tokens']
    max_reference = limits['max_reference_tokens']
    if rendered['task_id'] == 'context_evidence_retrieval':
        maximum = min(maximum, 576); max_reference = min(max_reference, 64)
    require(tokenizer.is_fast, 'Offset-aware fast tokenizer required')
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    ref_ids = list(tokenizer.encode(reference, add_special_tokens=False))
    require(len(prompt_ids) <= limits['max_prompt_tokens'], 'PROMPT_TOO_LONG: explicit prompt bound exceeded')
    require(len(ref_ids) <= max_reference, 'REFERENCE_TOO_LONG: explicit reference bound exceeded')
    # Reuse the original joint offset boundary rule.  Whole crossing tokens remain
    # intact; independently encoded prompt IDs are never obtained by slicing it.
    encoded, ids, offsets, boundary = joint_encoding(tokenizer, prompt, reference, maximum)
    ids = list(ids); special_ids = list(tokenizer.all_special_ids)
    special = list(encoded['special_tokens_mask'])
    require(all(type(i) is int and 0 <= i < metadata['vocabulary_size'] for i in ids),
            'Tokenizer IDs outside declared vocabulary')
    cp, byte, overlaps = _offsets(prompt + reference, ids, offsets, special)
    normalization = None
    from ..tasks.input_records import QWEN_PROFILE
    if metadata.get('capability_profile') == QWEN_PROFILE:
        normalization = _qwen_prompt_offsets(tokenizer, prompt, reference, cp)
    spans = {key: [a + len(moved), b + len(moved)]
             for key, (a, b) in rendered['value_spans'].items()}
    targets, coverage = field_targets(ids, boundary, cp, len(prompt), spans,
                                      set(special_ids), spec['selection']['max_targets'])
    eligible = [t for t in range(boundary, len(ids))
                if t > 0 and not special[t] and ids[t] not in special_ids]
    require(eligible and all(t['token_position'] in eligible for t in targets),
            'INVALID_EMPTY_PLAN: no valid reference targets')
    for target in targets:
        target['prefix_hash'] = digest({'token_ids': ids[:target['token_position']],
                                      'token_id_space': metadata['token_id_space']})
    full = _uniform([t['token_position'] for t in targets], spec['selection']['max_full'])
    natural = rendered['request_kind'] == 'generate_until'
    old_policy = {'tech_record_extract': 'chat_rendered_joint_field_spans_v1',
                  'context_evidence_retrieval': 'retrieval_chat_joint_answer_evidence_spans_v1',
                  'arc_challenge': 'arc_existing_chat_echo_boundary_v1'}[rendered['task_id']]
    row = Sample(rendered['task_id'], rendered['task_version'], rendered['sample_id'],
      rendered['source_id'], prompt, digest(prompt), reference, rendered['reference_source'],
      {'policy': old_policy, 'selection_policy': SELECTION,
       'chat_template_applied': int(natural), 'historical_template_already_applied': not natural,
       'tokenizer_identity': metadata['tokenizer_identity'], 'token_id_space': metadata['token_id_space'],
       'bos_added_by_encoder': False, 'eos_added_by_encoder': False, 'padding_added': False,
       'add_generation_prompt': natural, 'add_special_tokens': False,
       'boundary_rule': 'first joint token with end > prompt codepoint length; whole boundary token'},
      'External gold teacher-forced reference; separate from prompt-only generation').validate()
    authored = rendered['authored_sample']
    row.update({'split': rendered['split'], 'usage_scope': rendered['usage_scope'],
      'dataset_id': rendered['dataset_id'], 'reference_role': rendered['reference_role'],
      'language': authored.get('language', 'en'), 'task_prompt_hash': digest(rendered['content']),
      'full_text': prompt + reference, 'full_text_hash': digest(prompt + reference),
      'input_ids': ids, 'generation_input_ids': prompt_ids if natural else None,
      'prompt_separate_ids': prompt_ids, 'reference_separate_ids': ref_ids,
      'joint_equals_separate_concat': ids == prompt_ids + ref_ids,
      'prompt_token_boundary': boundary, 'boundary_crossing_token': cp[boundary][0] < len(prompt),
      'offsets_codepoints': cp, 'offsets_utf8_bytes': byte, 'offset_mapping_status': 'AVAILABLE',
      'offset_contract': {'unit': 'unicode_codepoints', 'utf8_bytes_derived_from_codepoint_prefix': True,
        'nonempty_overlap_policy': 'keep whole token; overlapping offsets allowed and recorded',
        'overlapping_offset_token_positions': overlaps, 'zero_length': 'special tokens only'},
      'token_strings': list(tokenizer.convert_ids_to_tokens(ids)), 'attention_mask': [1] * len(ids),
      'loss_mask': [int(t >= boundary and ids[t] not in special_ids) for t in range(len(ids))],
      'special_token_mask': special, 'special_token_ids': special_ids,
      'reference_token_count': len(eligible), 'targets': targets, 'value_spans': spans,
      'field_token_coverage': coverage, 'selection_policy': SELECTION,
      'full_vocab_target_positions': full, 'sequence_identity': digest(ids),
      'reference_hash': digest(reference), 'selection_receipt': {
        'eligible_target_count': len(eligible), 'selected_target_count': len(targets),
        'selected_full_count': len(full), 'target_limit': spec['selection']['max_targets'],
        'full_limit': spec['selection']['max_full'],
        'selection_rule': 'first eligible; literal fields round-robin in renderer order; remaining deterministic uniform using Python round ties-to-even; deduplicate token positions',
        'full_rule': 'deterministic uniform subset of selected targets; Python round ties-to-even',
        'full_status': 'NOT_REQUESTED' if spec['selection']['max_full'] == 0 else 'SELECTED'},
      'preparation_policy': POLICY})
    if normalization is not None:
        row['backend_normalization'] = normalization
        if normalization['normalizer_changed_prompt']:
            row['offset_mapping_status'] = 'PARTIAL_NATIVE_OFFSETS_NORMALIZATION'
    for key in ('document_hash', 'sample_identity'):
        if key in authored:
            row[key] = authored[key]
    row['source_document_span'] = _source_document_span(rendered, row)
    if rendered['task_id'] == 'context_evidence_retrieval':
        row.update(content_span_records(authored, row))
    return row


def _arc_requests(rendered, tokenizer, context_tokens, model_profile=None):
    authored = deepcopy(rendered['authored_sample']); requests = []
    for index, choice in enumerate(authored['choices']):
        request = {'choice_index': index, 'context': rendered['content'], 'continuation': ' ' + choice}
        prompt, reference = effective_text(request)
        if model_profile is not None:
            normalizer = _qwen_normalizer(tokenizer)
            require(normalizer.normalize_str(prompt) == prompt and normalizer.normalize_str(reference) == reference,
                    'UNSUPPORTED_TOKEN_OFFSET: ARC context/continuation normalization changes input')
        context_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        continuation_ids = list(tokenizer.encode(reference, add_special_tokens=False))
        joint_ids = list(tokenizer.encode(prompt + reference, add_special_tokens=False))
        require(context_ids and continuation_ids and joint_ids == context_ids + continuation_ids,
                'UNSUPPORTED_TOKEN_BOUNDARY: ARC whitespace/joint IDs differ from native likelihood contract')
        require(len(joint_ids) + 1 <= context_tokens,
                'CHOICE_TOO_LONG: ARC prompt plus audit token exceeds serving context')
        requests.append(request | {'context_ids': context_ids, 'continuation_ids': continuation_ids,
          'input_ids': joint_ids, 'normalization_char_length': len(choice),
          'boundary_policy': 'rstrip context; move original trailing whitespace to continuation',
          'moved_trailing_context_whitespace': request['context'][len(prompt):]})
    authored['requests'] = requests
    authored['request_identity'] = digest(requests)
    authored.setdefault('source_text_hash', digest(authored['document']))
    return authored


def prepare_rows(raw, spec, model_metadata, tokenizer):
    """Tokenize validated raw records into standalone per-task prepared sources.

    ``tokenizer`` is supplied explicitly, so unit tests and base imports need no
    SDK.  Filesystem/runtime loading is confined to ``run`` below.
    """
    from ..tasks.input_records import render_record, validate_raw, render_pinned_chat, QWEN_PROFILE
    require(spec['policy_id'] == POLICY and spec['selection']['policy_id'] == SELECTION,
            'Unsupported preparation/selection policy')
    if isinstance(raw, dict):
        validate_raw(raw)
    records = raw['rows'] if isinstance(raw, dict) else raw
    require(isinstance(records, list) and records, 'INVALID_EMPTY_PLAN: no raw records')
    require(tokenizer.is_fast, 'Offset-aware fast tokenizer required')
    metadata = deepcopy(model_metadata)
    model_profile = metadata.get('capability_profile')
    require(model_profile in (None, QWEN_PROFILE), 'Unsupported preparation model profile')
    if 'token_spec' in metadata:
        metadata['token_spec']['max_sequence_tokens'] = spec['limits']['max_joint_tokens']
        metadata['token_spec']['max_target_positions_per_sample'] = spec['selection']['max_targets']
        metadata['token_spec']['max_full_vocab_positions_per_sample'] = spec['selection']['max_full']
    actual_space = digest(tokenizer.get_vocab())
    require(metadata['token_id_space'] == actual_space and
            metadata['vocabulary_size'] == len(tokenizer.get_vocab()), 'Tokenizer vocabulary pin changed')
    grouped = defaultdict(list); receipts = []
    for raw_record in records:
        rendered = render_record(raw_record, model_profile)
        require(rendered['sample_id'] not in {s['sample_id'] for s in grouped[(rendered['task_id'], rendered['task_version'])]},
                'Duplicate raw task/sample')
        natural = rendered['request_kind'] == 'generate_until'
        if natural:
            # Renderer receives only the authored source schema; the prompt path
            # does not append or serialize external gold/reference metadata.
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': rendered['content']}],
                         tokenize=False, add_generation_prompt=True)
            require(prompt == render_pinned_chat(rendered['content'], model_profile), 'Pinned chat template differs or was applied twice')
            reference = rendered['reference_text']; moved = ''
            authored = deepcopy(rendered['authored_sample'])
        else:
            require(rendered['prompt_rendering'] == 'already_rendered', 'ARC context must preserve its existing template contract')
            if model_profile == QWEN_PROFILE:
                content = 'Question: ' + rendered['authored_sample']['question'] + '\nAnswer:'
                actual_prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                  tokenize=False, add_generation_prompt=True)
                require(actual_prompt == rendered['content'], 'Pinned ARC chat template differs or was applied twice')
            authored = _arc_requests(rendered, tokenizer, spec['limits']['context_tokens'], model_profile)
            request = authored['requests'][authored['gold']]
            prompt, reference = effective_text(request)
            moved = request['context'][len(prompt):]
        row = _diagnostic(rendered, prompt, reference, tokenizer, metadata, spec, moved=moved)
        if model_profile == QWEN_PROFILE:
            row['tokenization'].update({'model_profile': model_profile, 'chat_template_applied': 1,
              'historical_template_already_applied': False, 'add_generation_prompt': True,
              'policy': 'qwen_pinned_chat_joint_reference_v1'})
        sample = {'sample_id': rendered['sample_id'], 'raw_record': deepcopy(raw_record),
                  'authored_sample': authored, 'diagnostic': row}
        if natural:
            maximum = 64 if rendered['task_id'] == 'context_evidence_retrieval' else 192
            require(len(row['generation_input_ids']) + maximum <= spec['limits']['context_tokens'],
                    'GENERATION_TOO_LONG: prompt plus maximum output exceeds context')
            wire = Adapter('vllm').body({'token_ids': row['generation_input_ids']},
                                       'serving-lab', maximum, 'natural_generation_v1')
            sample['generation'] = {
              'prepared': {key: deepcopy(row[key]) for key in ('sample_id', 'prompt', 'prompt_hash', 'generation_input_ids')},
              'wire_payload': wire,
              'preparation_proof': {'raw_record_hash': digest(raw_record), 'task_prompt_hash': digest(rendered['content']),
                'prompt_hash': digest(prompt), 'tokenizer_identity': metadata['tokenizer_identity'],
                'reference_used_for_generation': False, 'independent_prompt_encoding': True,
                'chat_template_applied': 1, 'rendered_task_prompt': rendered['content']}}
        else:
            gold = authored['requests'][authored['gold']]
            require(row['input_ids'] == gold['input_ids'] and row['prompt_token_boundary'] == len(gold['context_ids']),
                    'UNSUPPORTED_TOKEN_BOUNDARY: ARC diagnostic/all-choice boundary mismatch')
            row.update({'gold_choice_index': authored['gold'], 'native_choice_identity': digest(gold),
              'original_context': gold['context'], 'original_continuation': gold['continuation'],
              'moved_trailing_context_whitespace': moved})
            sample['likelihood'] = {'requests': deepcopy(authored['requests']),
              'choice_labels': deepcopy(authored['choice_labels']), 'gold_index': authored['gold']}
        row['probe_input_identity'] = digest(row)
        grouped[(rendered['task_id'], rendered['task_version'])].append(sample)
        receipts.append({'task_id': rendered['task_id'], 'sample_id': rendered['sample_id'],
          'prompt_tokens': len(row['prompt_separate_ids']), 'reference_separate_tokens': len(row['reference_separate_ids']),
          'joint_tokens': len(row['input_ids']), 'boundary': row['prompt_token_boundary'],
          'boundary_crossing_token': row['boundary_crossing_token'],
          'joint_equals_separate_concat': row['joint_equals_separate_concat'],
          'eligible_targets': row['reference_token_count'], 'targets': len(row['targets']),
          'full_positions': len(row['full_vocab_target_positions']),
          'choice_count': 0 if natural else len(authored['choices']),
          'chat_template_applied': int(natural)})
    from ..tasks.registry import get_task
    sources = []
    for (task_id, version), samples in grouped.items():
        adapter = get_task(task_id, version)
        preparation = {'schema_version': 1, 'policy_id': POLICY, 'raw_manifest_identity': digest(raw),
          'preparation_spec_identity': digest(spec), 'tokenizer_identity': metadata['tokenizer_identity'],
          'token_id_space': metadata['token_id_space'], 'scorer_version': adapter.scorer_version,
          'limits': deepcopy(spec['limits']), 'selection': deepcopy(spec['selection']),
          'source_identity': digest(source_identity())}
        sources.append({'schema_version': 1, 'kind': 'prepared_task_source_v1',
          'source_id': 'prepared-' + task_id + '-' + digest({'records': [s['raw_record'] for s in samples], 'spec': spec,
              'tokenizer_identity': metadata['tokenizer_identity']})[:16],
          'task_id': task_id, 'task_version': version, 'request_kind': adapter.request_kind,
          'model_metadata': deepcopy(metadata), 'samples': samples, 'preparation': preparation,
          'validation_scope': 'PREPARED_CONTRACT_VALIDATED_ONLY'})
    return {'sources': sources, 'receipt': {'schema_version': 1, 'policy_id': POLICY,
      'raw_manifest_identity': digest(raw), 'spec_identity': digest(spec), 'rows': receipts,
      'sample_count': len(records), 'task_count': len(sources), 'model_loads': 0,
      'gpu_actions': 0, 'inference_posts': 0, 'tokenizer_loads': 1}}


def _runtime_guard():
    """Deny model weight reads and network use in this explicit CPU subprocess.

    This is a Python audit guard, not an OS sandbox.  GPU initialization entry
    points are blocked separately after the tokenizer SDK's CPU import.
    """
    counts = {'network_attempts': 0, 'model_weight_read_attempts': 0, 'gpu_api_attempts': 0,
              'model_api_attempts': 0}
    def audit(event, args):
        if event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):
            counts['network_attempts'] += 1
            raise RuntimeError('Preparation worker forbids network use')
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            suffix = Path(os.fsdecode(args[0])).suffix.lower()
            if suffix in ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf'):
                counts['model_weight_read_attempts'] += 1
                raise RuntimeError('Preparation worker forbids model weight reads')
    sys.addaudithook(audit)
    return counts


def run(input_file, spec_file, binding_file, output_dir):
    """Execute only in an explicitly selected existing tokenizer runtime."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Tokenizer worker requires hidden GPUs')
    require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'Tokenizer worker requires explicit offline environment')
    from .input_preparation import read_raw, validate_spec, validate_binding
    raw, spec, binding = read_raw(Path(input_file)), validate_spec(read(Path(spec_file))), validate_binding(read(Path(binding_file)))
    output = Path(output_dir)
    require(not output.is_symlink(), 'Unsafe preparation output directory')
    require(not output.exists() or output.is_dir() and not any((output / name).exists() for name in ('SOURCES.json', 'PREPARATION_RECEIPT.json', 'RUNTIME_RECEIPT.json')), 'Refuse preparation output overwrite')
    tokenizer_path = Path(binding['tokenizer_path']); config_path = Path(binding['config_path'])
    require(tokenizer_path.is_dir() and not tokenizer_path.is_symlink() and config_path.is_file(),
            'Missing explicit local tokenizer/config')
    pins = binding.get('tokenizer_files', binding.get('files'))
    require(isinstance(pins, dict) and pins and set(pins) <= set(TOKENIZER_FILES), 'Explicit tokenizer file pins required')
    for name, identity in pins.items():
        path = tokenizer_path / name
        require(path.is_file() and sha(path) == identity, 'Tokenizer file pin changed: ' + name)
    require(binding.get('config_sha256') == sha(config_path), 'Pinned model config bytes changed')
    require(binding.get('toolkit_source_identity') == digest(source_identity()), 'Preparation worker source identity differs')
    model_metadata = deepcopy(binding['model_metadata'])
    require(digest(pins) == model_metadata['tokenizer_identity'], 'Tokenizer file identity differs from model metadata')
    require(read(config_path) == model_metadata['model_config'], 'Pinned model config metadata changed')
    counts = _runtime_guard()
    try:
        from transformers import AutoTokenizer
        import transformers
    except ImportError as error:
        raise ValueError("MISSING_LOCAL_TOKENIZER_RUNTIME: selected interpreter lacks a usable tokenizer SDK") from error
    # The tokenizer can import Torch as an SDK dependency.  It never calls a
    # model class, CUDA query, CUDA initialization, or tensor forward.
    torch = sys.modules.get('torch')
    def forbidden_gpu(*args, **kwargs):
        counts['gpu_api_attempts'] += 1
        raise RuntimeError('Tokenizer worker forbids GPU API calls')
    if torch is not None:
        for name in ('_lazy_init', 'init', 'is_available', 'device_count', 'current_device', 'get_device_properties'):
            if hasattr(torch.cuda, name):
                setattr(torch.cuda, name, forbidden_gpu)
    # Weight-read and network guards are authoritative even if another model
    # class is exposed by this installed SDK; block the common model loaders too.
    def forbidden_model(*args, **kwargs):
        counts['model_api_attempts'] += 1
        raise RuntimeError('Tokenizer worker forbids model loading')
    for name in ('AutoModel', 'AutoModelForCausalLM', 'AutoModelForSeq2SeqLM'):
        cls = getattr(transformers, name)
        cls.from_pretrained = forbidden_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast, 'Offset-aware fast tokenizer required')
    offset_receipt = tokenizer_offset_probe(tokenizer, model_metadata.get('capability_profile'))
    result = prepare_rows(raw, spec, model_metadata, tokenizer)
    # Pin bytes again after tokenization, before publishing any success manifest.
    require(all(sha(tokenizer_path / n) == value for n, value in pins.items()),
            'Tokenizer files changed during preparation')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'kind': 'suite_prepared_sources_v1', 'sources': []}
    for source in result['sources']:
        name = 'source-' + source['task_id'] + '.json'; path = output / name
        write(path, source)
        manifest['sources'].append({'source_id': source['source_id'], 'path': name, 'sha256': sha(path)})
    write(output / 'SOURCES.json', manifest)
    write(output / 'PREPARATION_RECEIPT.json', result['receipt'])
    write(output / 'TOKENIZER_OFFSET_RECEIPT.json', offset_receipt)
    packages = {}
    for name in ('transformers', 'tokenizers', 'torch', 'numpy'):
        try:
            dist = package_metadata.distribution(name)
            packages[name] = {'version': dist.version, 'package_root': str(dist.locate_file(''))}
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    runtime = {'schema_version': 1, 'python': sys.executable, 'python_version': sys.version,
      'package_origin': str(Path(__file__).resolve()), 'source_identity': digest(source_identity()),
      'tokenizer_loader': 'AutoTokenizer.from_pretrained', 'loader_options': {
        'local_files_only': True, 'trust_remote_code': False, 'use_fast': True},
      'tokenizer_class': type(tokenizer).__name__, 'tokenizer_is_fast': tokenizer.is_fast,
      'tokenizer_files': pins, 'tokenizer_file_locators': {name: {'path': str(tokenizer_path / name), 'resolved_path': str((tokenizer_path / name).resolve()), 'symlink': (tokenizer_path / name).is_symlink()} for name in pins}, 'tokenizer_identity': digest(pins),
      'token_id_space': digest(tokenizer.get_vocab()), 'vocabulary_size': len(tokenizer.get_vocab()),
      'tokenizer_revision': binding.get('revision'), 'packages': packages,
      'guard': {'scope': 'in-process Python audit hook and explicit loader/GPU API guards', **counts},
      'gpu_visible_devices': '', 'model_loads': 0, 'model_forwards': 0, 'gpu_queries': 0,
      'inference_posts': 0, 'network_requests': 0, 'tokenizer_loads': 1}
    require(not any(counts.values()), 'Preparation used a forbidden execution path')
    write(output / 'RUNTIME_RECEIPT.json', runtime)
    return result['receipt']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('input', 'spec', 'binding', 'output'):
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.spec, args.binding, args.output), ensure_ascii=True, allow_nan=False))
