"""CPU-only token comparison over declared, same-prefix observation contracts.

Artifact/recipe are treatments. Task, vocabulary, prefix and acquisition policy
are invariants. Missing observations retain UNKNOWN coverage and never become 0.
Legacy token metrics and historical schemas are not modified by this module.
"""
import math

from ..core.io import digest
from .token import distribution_metrics, log_softmax, nll

MATCHING_AXES = ('source_id', 'panel', 'evidence_kind', 'task_id', 'task_version', 'dataset_identity',
    'split', 'sample_id', 'policy_id', 'choice_index', 'reference_scope',
    'prompt_hash', 'reference_hash', 'tokenizer_identity', 'token_id_space',
    'vocabulary_size', 'runtime_identity', 'backend_policy', 'probability_stage',
    'metric_contract', 'token_position', 'logit_position', 'target_token_id', 'prefix_hash')
REQUIRED_TEXT = ('source_id', 'task_id', 'task_version', 'dataset_identity', 'split',
    'sample_id', 'reference_scope', 'prompt_hash', 'reference_hash',
    'tokenizer_identity', 'token_id_space', 'runtime_identity', 'backend_policy')
GROUP_AXES = ('source_id', 'panel', 'evidence_kind', 'task_id', 'task_version', 'dataset_identity',
    'split', 'policy_id', 'reference_scope', 'backend_policy', 'runtime_identity',
    'tokenizer_identity', 'token_id_space', 'vocabulary_size', 'probability_stage', 'metric_contract')
ACQUISITIONS = {'native_likelihood': 'native', 'instrumented_diagnostic': 'diagnostic',
                'decoded_diagnostic': 'diagnostic'}


def _identity(left, right):
    row = left or right or {}
    result = {key: row.get(key) for key in GROUP_AXES}
    result.update({key: row.get(key) for key in ('sample_id', 'choice_index')})
    result['acquisition_source_left'] = left.get('acquisition_source') if left else None
    result['acquisition_source_right'] = right.get('acquisition_source') if right else None
    return result


def _empty(left, right):
    row = left or right or {}
    return {'schema_version': 1, 'status': 'UNKNOWN', 'reasons': [],
      'left_record_id': left.get('record_id') if left else None,
      'right_record_id': right.get('record_id') if right else None,
      'identity': _identity(left, right), 'requested': bool(row.get('requested', True)),
      'token_position': row.get('token_position'), 'target_token_id': row.get('target_token_id'),
      'field_tags': list(row.get('field_tags', [])),
      'metrics': {'reference_nll_nat': None, 'candidate_nll_nat': None, 'delta_nll_nat': None,
        'raw_top1_match': None, 'reference_target_rank': None, 'candidate_target_rank': None,
        'reference_target_margin': None, 'candidate_target_margin': None,
        'full_distribution': {'status': 'UNKNOWN', 'reason': 'FULL_VOCABULARY_NOT_COLLECTED'},
        'full_consistency': {'reference': {'status': 'UNKNOWN'}, 'candidate': {'status': 'UNKNOWN'}},
        'top_k': {str(k): {'requested_k': k, 'effective_k': None, 'status': 'UNKNOWN',
          'reference_ids': None, 'candidate_ids': None, 'overlap_count': None,
          'overlap_fraction': None} for k in (1, 5, 10)}},
      'coverage': {'target_logprob': False, 'raw_argmax': False, 'full_logits': False},
      'units': {'nll': 'nat', 'log_base': 'e', 'target_margin': 'raw_logit_difference'},
      'scope': 'selected observed token positions; not corpus perplexity or task accuracy'}


def _record_issues(row, sequence, side):
    invalid, missing = [], []
    for key in REQUIRED_TEXT:
        if not isinstance(row.get(key), str) or not row[key]:
            invalid.append(side + ':INVALID_' + key.upper())
    if row.get('probability_stage') != 'raw_pre_sampling':
        invalid.append(side + ':PROBABILITY_STAGE_NOT_RAW')
    if row.get('metric_contract') != 'target_logprob_v1':
        invalid.append(side + ':METRIC_CONTRACT_UNSUPPORTED')
    acquisition = row.get('acquisition_source')
    if acquisition not in ACQUISITIONS or ACQUISITIONS.get(acquisition) != row.get('backend_policy'):
        invalid.append(side + ':ACQUISITION_BACKEND_MISMATCH')
    vocabulary = row.get('vocabulary_size')
    if type(vocabulary) is not int or vocabulary <= 1:
        invalid.append(side + ':INVALID_VOCABULARY_SIZE')
    t, logit, target = (row.get(k) for k in ('token_position', 'logit_position', 'target_token_id'))
    if type(t) is not int or t <= 0 or type(logit) is not int or logit != t - 1:
        invalid.append(side + ':INVALID_NEXT_TOKEN_SHIFT')
    if type(target) is not int or type(vocabulary) is not int or not 0 <= target < vocabulary:
        invalid.append(side + ':INVALID_TARGET_TOKEN')
    tags = row.get('field_tags', [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags) or len(tags) != len(set(tags)):
        invalid.append(side + ':INVALID_FIELD_TAGS')
    if type(row.get('requested')) is not bool:
        invalid.append(side + ':INVALID_REQUESTED_FLAG')
    elif not row['requested']:
        missing.append(side + ':NOT_REQUESTED')
    if row.get('status') != 'MEASURED':
        missing.append(side + ':NOT_MEASURED')
    if not isinstance(sequence, dict):
        missing.append(side + ':MISSING_SEQUENCE')
    else:
        ids = sequence.get('input_ids')
        if not isinstance(ids, list) or not ids or not all(type(i) is int and i >= 0 and
              (type(vocabulary) is int and i < vocabulary) for i in ids):
            invalid.append(side + ':INVALID_INPUT_IDS')
        elif type(t) is int and 0 < t < len(ids):
            if type(target) is int and ids[t] != target:
                invalid.append(side + ':TARGET_SEQUENCE_MISMATCH')
            expected = digest({'token_ids': ids[:t], 'token_id_space': row.get('token_id_space')})
            if row.get('prefix_hash') != expected:
                invalid.append(side + ':PREFIX_HASH_MISMATCH')
        else:
            invalid.append(side + ':TOKEN_POSITION_OUTSIDE_SEQUENCE')
        if not row.get('sequence_id') or sequence.get('sequence_id') != row.get('sequence_id'):
            invalid.append(side + ':SEQUENCE_ID_MISMATCH')
    value = row.get('target_logprob')
    if value is None:
        missing.append(side + ':MISSING_TARGET_LOGPROB')
    else:
        try:
            nll(value)
        except (ValueError, TypeError, OverflowError):
            invalid.append(side + ':INVALID_TARGET_LOGPROB')
    argmax = row.get('raw_argmax_token_id')
    if argmax is not None and (type(argmax) is not int or type(vocabulary) is not int or not 0 <= argmax < vocabulary):
        invalid.append(side + ':INVALID_RAW_ARGMAX')
    return invalid, missing


def _full(row, values, side):
    if values is None:
        return None, [], {'status': 'UNKNOWN', 'reason': 'FULL_VOCABULARY_NOT_COLLECTED'}
    vocabulary = row['vocabulary_size']
    if not isinstance(values, (list, tuple)) or len(values) != vocabulary or not all(
            type(v) in (float, int) and math.isfinite(v) for v in values):
        return None, [side + ':INVALID_FULL_VOCABULARY'], {'status': 'UNKNOWN'}
    try:
        lp = log_softmax(values)
        if not all(math.isfinite(v) for v in lp):
            raise ValueError('Unrepresentable finite-logit normalization')
    except (ValueError, OverflowError):
        return None, [side + ':FULL_LOGITS_UNNORMALIZABLE'], {'status': 'UNKNOWN'}
    order = sorted(range(vocabulary), key=lambda i: (-values[i], i))
    target = row['target_token_id']
    argmax = row.get('raw_argmax_token_id')
    issues = []
    if argmax is not None and argmax != order[0]:
        issues.append(side + ':RAW_ARGMAX_FULL_LOGITS_MISMATCH')
    tolerance = row.get('logit_logprob_tolerance')
    if tolerance is None:
        consistency = {'status': 'UNKNOWN', 'reason': 'ACQUISITION_TOLERANCE_NOT_DECLARED'}
    elif type(tolerance) not in (float, int) or not math.isfinite(tolerance) or tolerance < 0:
        consistency = {'status': 'UNKNOWN', 'reason': 'INVALID_ACQUISITION_TOLERANCE'}
        issues.append(side + ':INVALID_ACQUISITION_TOLERANCE')
    elif row.get('target_logprob') is None:
        consistency = {'status': 'UNKNOWN', 'reason': 'MISSING_TARGET_LOGPROB'}
    else:
        error = abs(lp[target] - row['target_logprob'])
        consistency = {'status': 'MEASURED', 'absolute_error_nat': error,
          'absolute_tolerance_nat': tolerance, 'within_declared_tolerance': error <= tolerance}
        if error > tolerance:
            issues.append(side + ':TARGET_LOGPROB_FULL_LOGITS_MISMATCH')
    return {'values': values, 'logprobs': lp, 'order': order,
            'target_rank': order.index(target) + 1,
            'target_margin': values[target] - max(v for i, v in enumerate(values) if i != target)}, issues, consistency


def compare_token(left, right, left_sequence, right_sequence, left_logits=None, right_logits=None):
    """Compare one declared same-prefix target, without loading files or runtimes.

    Full metrics require two complete raw vocabulary arrays. Consistency with
    stored target logprob is checked only against an acquisition-declared bound.
    Deterministic rank/top-k ties use increasing token ID; rank is one-based.
    """
    result = _empty(left, right)
    if left is None or right is None:
        result['reasons'] = [('MISSING_BASELINE' if left is None else 'MISSING_CANDIDATE')]
        if left is None and right is None:
            result['reasons'].append('MISSING_CANDIDATE')
        return result
    invalid, missing = [], []
    for row, sequence, side in ((left, left_sequence, 'REFERENCE'), (right, right_sequence, 'CANDIDATE')):
        a, b = _record_issues(row, sequence, side)
        invalid.extend(a); missing.extend(b)
    for key in MATCHING_AXES:
        if left.get(key) != right.get(key):
            invalid.append('MISMATCH_' + key.upper())
    if left.get('field_tags', []) != right.get('field_tags', []):
        invalid.append('MISMATCH_FIELD_TAGS')
    if not invalid and isinstance(left_sequence, dict) and isinstance(right_sequence, dict):
        t = left['token_position']
        if left_sequence['input_ids'][:t] != right_sequence['input_ids'][:t]:
            invalid.append('MISMATCH_ACTUAL_PREFIX')
    if invalid:
        result.update(status='INCOMPARABLE', reasons=invalid + missing)
        return result
    details_left, issues_left, consistent_left = _full(left, left_logits, 'REFERENCE')
    details_right, issues_right, consistent_right = _full(right, right_logits, 'CANDIDATE')
    result['metrics']['full_consistency'] = {'reference': consistent_left, 'candidate': consistent_right}
    if issues_left or issues_right:
        result.update(status='INCOMPARABLE', reasons=issues_left + issues_right + missing)
        return result
    metrics = result['metrics']
    if not missing:
        metrics['reference_nll_nat'] = nll(left['target_logprob'])
        metrics['candidate_nll_nat'] = nll(right['target_logprob'])
        metrics['delta_nll_nat'] = metrics['candidate_nll_nat'] - metrics['reference_nll_nat']
        result['coverage']['target_logprob'] = True
        result['status'] = 'MEASURED'
    else:
        result['reasons'] = missing
    if not missing and left.get('raw_argmax_token_id') is not None and right.get('raw_argmax_token_id') is not None:
        metrics['raw_top1_match'] = left['raw_argmax_token_id'] == right['raw_argmax_token_id']
        result['coverage']['raw_argmax'] = True
    for name, detail in (('reference', details_left), ('candidate', details_right)):
        if detail is not None:
            metrics[name + '_target_rank'] = detail['target_rank']
            metrics[name + '_target_margin'] = detail['target_margin']
    if details_left is not None and details_right is not None and not missing:
        distribution = distribution_metrics(details_left['values'], details_right['values'])
        if not all(math.isfinite(v) for v in distribution.values() if type(v) in (int, float)):
            result.update(status='INCOMPARABLE', reasons=['FULL_DISTRIBUTION_NONFINITE'])
            return result
        metrics['full_distribution'] = {'status': 'MEASURED', **distribution}
        result['coverage']['full_logits'] = True
        for k in (1, 5, 10):
            effective = min(k, left['vocabulary_size'])
            a, b = details_left['order'][:effective], details_right['order'][:effective]
            overlap = len(set(a) & set(b))
            metrics['top_k'][str(k)] = {'requested_k': k, 'effective_k': effective,
              'status': 'MEASURED', 'reference_ids': a, 'candidate_ids': b,
              'overlap_count': overlap, 'overlap_fraction': overlap / effective}
    return result


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def aggregate_token_comparisons(rows):
    """Keep task/policy/acquisition/reference/field groups and document units apart.

    Document macro first averages all observed target positions/choices within a
    source sample, then averages documents. Fields may overlap; their denominators
    are explicitly independent and must not be summed as disjoint partitions.
    """
    groups = {}
    for row in rows:
        identity = {key: row['identity'].get(key) for key in GROUP_AXES}
        identity.update({key: row['identity'].get(key) for key in ('acquisition_source_left', 'acquisition_source_right')})
        scopes = [identity['reference_scope']]
        if identity['backend_policy'] == 'native' and identity['reference_scope'] in ('GOLD', 'NON_GOLD', 'GOLD_CHOICE', 'NON_GOLD_CHOICE'):
            scopes = [('GOLD_ONLY' if identity['reference_scope'] in ('GOLD', 'GOLD_CHOICE') else 'NON_GOLD'), 'ALL_CHOICES']
        tags = ['__all__'] + row.get('field_tags', [])
        for scope in scopes:
            scope_identity = identity | {'reference_scope': scope}
            for field in tags:
                key = digest({'identity': scope_identity, 'field_tag': field})
                if key not in groups:
                    groups[key] = {'identity': scope_identity, 'field_tag': field, 'rows': []}
                groups[key]['rows'].append(row)
    result = []
    for key in sorted(groups):
        group = groups[key];values = group.pop('rows');valid = [r for r in values if r['status'] == 'MEASURED']
        documents = {}
        for row in valid:
            doc = row['identity']['sample_id']
            documents.setdefault(doc, []).append(row)
        nlls = {}
        for name in ('reference_nll_nat', 'candidate_nll_nat', 'delta_nll_nat'):
            nlls[name] = {'token_weighted': _mean([r['metrics'][name] for r in valid if r['metrics'][name] is not None]),
              'document_macro': _mean([_mean([r['metrics'][name] for r in rs if r['metrics'][name] is not None]) for rs in documents.values()]),
              'measured_tokens': sum(r['metrics'][name] is not None for r in valid), 'measured_documents': len(documents)}
        argmax = [r['metrics']['raw_top1_match'] for r in valid if r['metrics']['raw_top1_match'] is not None]
        full = [r['metrics']['full_distribution'] for r in valid if r['metrics']['full_distribution']['status'] == 'MEASURED']
        top_k = {}
        for k in (1, 5, 10):
            found = [r['metrics']['top_k'][str(k)] for r in valid if r['metrics']['top_k'][str(k)]['status'] == 'MEASURED']
            top_k[str(k)] = {'requested_k': k, 'measured_tokens': len(found),
              'mean_overlap_fraction': _mean([v['overlap_fraction'] for v in found]),
              'effective_k_values': sorted(set(v['effective_k'] for v in found))}
        result.append(group | {'group_identity': key, 'planned_comparison_tokens': len(values),
          'requested_comparison_tokens': sum(bool(r.get('requested')) for r in values),
          'measured_comparison_tokens': len(valid), 'measured_documents': len(documents),
          'status_counts': {status: sum(r['status'] == status for r in values) for status in ('MEASURED', 'UNKNOWN', 'INCOMPARABLE')},
          'nll': nlls, 'raw_top1': {'matches': sum(argmax), 'measured_tokens': len(argmax), 'match_fraction': _mean(argmax)},
          'full_distribution': {'measured_tokens': len(full), **{name: _mean([v[name] for v in full]) for name in
            ('kl_ref_candidate_nat', 'js_nat', 'entropy_reference_nat', 'entropy_candidate_nat')}},
          'top_k': top_k, 'field_denominator_policy': 'overlap allowed; do not sum field groups'})
    return {'schema_version': 1, 'groups': result, 'no_global_task_average': True,
      'grouping': 'task/dataset/split/policy/acquisition/runtime/reference scope/field kept separate',
      'document_unit': 'source sample_id; choices and token positions are repeated within document',
      'missing_rule': 'UNKNOWN/INCOMPARABLE retained in coverage; never zero-filled'}
