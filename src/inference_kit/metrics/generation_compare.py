"""CPU comparisons of observed free generation, never teacher-forced tokens.

String agreement is a diagnostic, not task correctness. After two generated
sequences branch, token positions describe the two observed streams only; they
are not a shared-prefix log-probability or representation comparison.
"""
from ..core.io import require

IDENTITY_KEYS = ('task_id', 'task_version', 'sample_id', 'sample_identity', 'split',
                 'policy_id', 'prompt_hash', 'tokenizer_identity', 'token_id_space',
                 'runtime_identity', 'scorer_identity', 'document_hash',
                 'gold_identity', 'generation_policy_identity')


def _value(row, key):
    return row.get(key, row.get('identity', {}).get(key))


def _tokens(row):
    tokens = row.get('output_token_ids')
    if tokens is None:
        return None
    require(isinstance(tokens, list) and all(type(v) is int and v >= 0 for v in tokens),
            'Invalid actual output token IDs')
    require(row.get('output_token_provenance') in ('RAW_SSE_SERVER_TOKEN_IDS', 'STORED_SERVER_TOKEN_IDS', 'ACTUAL_SERVER_TOKEN_IDS'),
            'Output token IDs require explicit actual-server provenance')
    return tokens


def _complete(row):
    return row.get('transport_complete') is True and row.get('termination') == 'NATURAL_STOP'


def _censor(row):
    termination = row.get('termination', 'UNKNOWN')
    if row.get('record_status') == 'NOT_RUN' or termination == 'NOT_RUN': return 'NOT_RUN'
    if termination == 'LENGTH_CAP': return 'LENGTH_CAP_CENSORED'
    if termination == 'NATURAL_STOP' and row.get('transport_complete') is True: return 'NATURAL_STOP_OBSERVED'
    if termination in ('TRUNCATED_OR_EOF', 'INCOMPLETE_SUBMISSION'): return 'EOF_OR_INCOMPLETE_CENSORED'
    return 'TRANSPORT_OR_TERMINATION_UNKNOWN'


def compare_generation(left, right):
    """Compare same-input BF16/candidate records; declared artifact may differ.

    Missing/incompatible inputs return explicit state rather than false equality.
    Full-sequence claims require two complete natural terminations. The observed
    prefix and explicit truncation states remain available for capped/partial runs.
    """
    result = {'metric_version': 'free_generation_pair_v1', 'status': 'UNKNOWN',
              'left_record_id': left.get('record_id') if isinstance(left, dict) else None,
              'right_record_id': right.get('record_id') if isinstance(right, dict) else None,
              'comparison_scope': 'observed free-generation streams; no teacher-forced comparison after branching',
              'same_output_is_task_accuracy': False, 'incompatible_reasons': [],
              'text_match': None, 'observed_text_match': None, 'full_output_token_match': None,
              'observed_token_sequence_match': None, 'longest_common_prefix_tokens': None,
              'first_different_token_position': None, 'first_different_left_token_id': None,
              'first_different_right_token_id': None, 'first_length_divergence_position': None,
              'left_actual_output_token_length': None, 'right_actual_output_token_length': None,
              'paired_task_outcome': None}
    if not isinstance(left, dict) or not isinstance(right, dict):
        result['status'] = 'MISSING_PAIR'; return result
    result.update(source_id=left.get('source_id') if left.get('source_id') == right.get('source_id') else None,
                  left_source_id=left.get('source_id'), right_source_id=right.get('source_id'),
                  task_id=_value(left, 'task_id'), task_version=_value(left, 'task_version'),
                  sample_id=_value(left, 'sample_id'), policy_id=_value(left, 'policy_id'),
                  left_arm_id=left.get('arm_id'), right_arm_id=right.get('arm_id'),
                  left_bundle_identity=left.get('bundle_identity'), right_bundle_identity=right.get('bundle_identity'))
    for key in IDENTITY_KEYS:
        a, b = _value(left, key), _value(right, key)
        if a is None or b is None: result['incompatible_reasons'].append('MISSING_IDENTITY:' + key)
        elif a != b: result['incompatible_reasons'].append('IDENTITY_MISMATCH:' + key)
    result['left_censoring'] = _censor(left); result['right_censoring'] = _censor(right)
    if result['incompatible_reasons']:
        result['status'] = 'INCOMPATIBLE'; return result
    a, b = _tokens(left), _tokens(right)
    natural_complete = _complete(left) and _complete(right)
    observed_both = left.get('record_status') != 'NOT_RUN' and right.get('record_status') != 'NOT_RUN'
    text_a, text_b = left.get('output_text'), right.get('output_text')
    if observed_both and isinstance(text_a, str) and isinstance(text_b, str):
        result['observed_text_match'] = text_a == text_b
        if natural_complete: result['text_match'] = text_a == text_b
    result['left_text_codepoints'] = len(text_a) if isinstance(text_a, str) and observed_both else None
    result['right_text_codepoints'] = len(text_b) if isinstance(text_b, str) and observed_both else None
    if a is None or b is None:
        result['status'] = 'TEXT_ONLY' if natural_complete else 'TEXT_ONLY_CENSORED'
        result['token_comparison_status'] = 'NOT_COLLECTED_ACTUAL_TOKEN_IDS'
        result['left_actual_output_token_length'] = len(a) if a is not None else None
        result['right_actual_output_token_length'] = len(b) if b is not None else None
    else:
        result['token_comparison_status'] = 'MEASURED_ACTUAL_SERVER_IDS'
        result['left_actual_output_token_length'] = len(a); result['right_actual_output_token_length'] = len(b)
        lcp = 0
        for x, y in zip(a, b):
            if x != y: break
            lcp += 1
        result['longest_common_prefix_tokens'] = lcp
        result['observed_token_sequence_match'] = a == b
        if natural_complete: result['full_output_token_match'] = a == b
        if lcp < min(len(a), len(b)):
            result['first_different_token_position'] = lcp
            result['first_different_left_token_id'] = a[lcp]
            result['first_different_right_token_id'] = b[lcp]
        elif len(a) != len(b):
            # There is no observed token on the exhausted side, so no fake EOS.
            result['first_length_divergence_position'] = lcp
        result['status'] = 'COMPLETE_OBSERVED_PAIR' if natural_complete else 'CENSORED_COMPARISON'
    # Task scores are scorer outputs, not text/token equality. Incomplete transport
    # cannot become an artificial both-wrong pair.
    sx, sy = left.get('task_success'), right.get('task_success')
    if left.get('transport_complete') is True and right.get('transport_complete') is True and type(sx) is bool and type(sy) is bool:
        result['paired_task_outcome'] = 'both_correct' if sx and sy else 'loss' if sx else 'gain' if sy else 'both_wrong'
        result['reference_task_success'] = sx; result['candidate_task_success'] = sy
    return result
