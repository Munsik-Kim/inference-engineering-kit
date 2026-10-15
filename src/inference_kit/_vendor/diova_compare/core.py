# Derived from inference-lab; import/path boundaries changed for internal release01.
"""Versioned records and explicitly scoped paired metrics (Python stdlib)."""
import hashlib
import json
import math
import random
from collections import Counter

REQUIRED = {'schema_version', 'sample_id', 'task', 'task_version', 'prompt_hash',
            'gold', 'gold_definition', 'model_id', 'artifact_id', 'output_type',
            'answer', 'correct', 'evidence_kind'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(rows):
    require(isinstance(rows, list) and bool(rows), 'nonempty record list required')
    seen = set()
    models = set()
    exit_presence = set()
    task_metrics = {}
    for r in rows:
        require(isinstance(r, dict) and REQUIRED <= r.keys(), 'missing record fields')
        require(type(r['schema_version']) is int and r['schema_version'] == 1, 'unsupported schema version')
        for key in ('sample_id', 'task', 'task_version', 'prompt_hash', 'gold_definition',
                    'model_id', 'artifact_id', 'answer', 'evidence_kind'):
            require(isinstance(r[key], str) and bool(r[key]), 'invalid ' + key)
        require(len(r['prompt_hash']) == 64 and all(c in '0123456789abcdef' for c in r['prompt_hash']), 'invalid prompt hash')
        require(r['output_type'] in ('choice', 'extracted_answer'), 'unsupported output type')
        require(r['evidence_kind'] in ('historical', 'measurement', 'synthetic_test'), 'invalid evidence kind')
        require(isinstance(r['correct'], bool), 'correct must be boolean')
        if 'process_exit_code' in r:
            require(type(r['process_exit_code']) is int, 'invalid process exit code')
        exit_presence.add('process_exit_code' in r)
        require(isinstance(r['gold'], list) and len(r['gold']) > 0 and all(isinstance(x, str) for x in r['gold']), 'gold must list accepted answers')
        require(len(set(r['gold'])) == len(r['gold']), 'duplicate gold')
        key = (r['task'], r['task_version'], r['sample_id'])
        require(key not in seen, 'duplicate sample key')
        seen.add(key)
        models.add((r['model_id'], r['artifact_id']))
        if r['output_type'] == 'choice':
            require(r['correct'] == (r['answer'] in r['gold']), 'choice correctness mismatch')
        require(isinstance(r.get('scores', {}), dict), 'scores must be an object')
        for k, x in r.get('scores', {}).items():
            require(k in ('gold_nll', 'brier') and finite_number(x) and x >= 0, 'invalid score')
        d = r.get('distribution')
        if d is not None:
            require(r['output_type'] == 'choice', 'free answers do not have a choice distribution')
            fields = {'kind', 'labels', 'probabilities', 'tokenizer_id', 'prefix_hash', 'dtype', 'normalization'}
            require(isinstance(d, dict) and fields <= d.keys(), 'distribution metadata missing')
            require(d['kind'] in ('choice', 'full_vocabulary'), 'distribution kind')
            labels, p = d['labels'], d['probabilities']
            require(isinstance(labels, list) and len(labels) >= 2 and all(isinstance(x, str) for x in labels), 'labels required')
            require(len(labels) == len(set(labels)), 'duplicate labels')
            require(isinstance(p, list) and len(p) == len(labels), 'distribution length mismatch')
            require(all(finite_number(x) and x >= 0 for x in p), 'invalid probabilities')
            require(abs(math.fsum(p) - 1) <= 1e-8, 'probability sum not one')
            require(d['normalization'] == 'probabilities_sum_to_one', 'normalization contract')
            require(all(isinstance(d[k], str) and d[k] for k in ('tokenizer_id', 'prefix_hash', 'dtype')), 'distribution provenance missing')
            require(d['prefix_hash'] == r['prompt_hash'], 'distribution prefix mismatch')
            require(set(r['gold']) <= set(labels) and r['answer'] in labels, 'gold/answer not in label space')
            if d['kind'] == 'full_vocabulary':
                require(d.get('complete') is True and d.get('vocabulary_size') == len(p), 'full distribution must cover vocabulary')
        metrics = {'delta_' + name for name in probability_scores(r)}
        if d is not None:
            metrics.add('choice_kl_B_to_C' if d['kind'] == 'choice' else 'full_vocabulary_kl_B_to_C')
        check_task_metrics(task_metrics, r, metrics)
    require(len(models) == 1, 'mixed artifacts within one input')
    require(len(exit_presence) == 1, 'mixed process status availability')
    return rows


def load(path):
    def unique_object(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON field: ' + key)
            out[key] = value
        return out
    with open(path, encoding='utf-8') as f:
        return validate([json.loads(line, object_pairs_hook=unique_object) for line in f if line.strip()])


def probability_scores(r):
    d = r.get('distribution')
    if not d or d['kind'] != 'choice':
        return dict(r.get('scores', {}))
    require(len(r['gold']) == 1, 'Brier requires one gold label')
    g = d['labels'].index(r['gold'][0])
    p = d['probabilities']
    require(p[g] > 0, 'zero gold probability has infinite NLL; supply log-space score separately')
    return {'gold_nll': -math.log(p[g]), 'brier': math.fsum((x - (i == g)) ** 2 for i, x in enumerate(p))}


def kl(p, q):
    if any(x > 0 and y == 0 for x, y in zip(p, q)):
        return None  # exact +infinity, represented separately in the pair record
    value = math.fsum(x * (math.log(x) - math.log(y)) for x, y in zip(p, q) if x > 0)
    require(math.isfinite(value), 'nonfinite KL for positive probabilities')
    return value


def check_task_metrics(contracts, record, metrics):
    """Coverage is strict within a task/version, not across unrelated tasks."""
    task = (record['task'], record['task_version'])
    actual = frozenset(metrics)
    expected = contracts.setdefault(task, actual)
    require(expected == actual,
            f"task metric mismatch: task={task[0]!r} version={task[1]!r} "
            f"sample_id={record['sample_id']!r}; expected={sorted(expected)!r} "
            f"actual={sorted(actual)!r}")


def compare(baseline, candidate, intersection=False, replicates=2000, seed=909220):
    validate(baseline); validate(candidate)
    require(isinstance(replicates, int) and replicates > 0, 'positive replicate count')
    key = lambda r: (r['task'], r['task_version'], r['sample_id'])
    b, c = {key(r): r for r in baseline}, {key(r): r for r in candidate}
    require(intersection or b.keys() == c.keys(), 'missing pair members (intersection is opt-in)')
    keys = sorted(b.keys() & c.keys())
    require(bool(keys), 'empty intersection')
    pairs = []
    pair_contracts = {}
    for k in keys:
        x, y = b[k], c[k]
        for field in ('prompt_hash', 'gold', 'gold_definition', 'output_type', 'evidence_kind'):
            require(x[field] == y[field], 'pair mismatch: ' + field)
        dx, dy = x.get('distribution'), y.get('distribution')
        require((dx is None) == (dy is None), 'one-sided distribution')
        bc, cc, changed = x['correct'], y['correct'], x['answer'] != y['answer']
        p = {'task': k[0], 'task_version': k[1], 'sample_id': k[2],
             'baseline_answer': x['answer'], 'candidate_answer': y['answer'],
             'baseline_correct': bc, 'candidate_correct': cc,
             'all_answer_disagreement': changed,
             'correct_to_wrong': bc and not cc, 'wrong_to_correct': not bc and cc,
             'both_correct': bc and cc, 'both_wrong': not bc and not cc,
             'wrong_to_wrong': not bc and not cc and changed,
             'correctness_flip': bc != cc, 'delta_accuracy': int(cc) - int(bc),
             'evidence_kind': x['evidence_kind']}
        require(('process_exit_code' in x) == ('process_exit_code' in y), 'one-sided process status')
        if 'process_exit_code' in x:
            p['baseline_process_exit_code'] = x['process_exit_code']
            p['candidate_process_exit_code'] = y['process_exit_code']
        if dx is not None:
            for f in ('kind', 'labels', 'tokenizer_id', 'prefix_hash', 'dtype', 'normalization'):
                require(dx[f] == dy[f], 'distribution pair mismatch: ' + f)
            metric = 'choice_kl_B_to_C' if dx['kind'] == 'choice' else 'full_vocabulary_kl_B_to_C'
            p[metric] = kl(dx['probabilities'], dy['probabilities'])
            p[metric + '_infinite'] = p[metric] is None
        sx, sy = probability_scores(x), probability_scores(y)
        require(sx.keys() == sy.keys(), 'score availability mismatch')
        for name in sx:
            p['delta_' + name] = sy[name] - sx[name]
        check_task_metrics(pair_contracts, p, {
            name for name in p if name.startswith('delta_') or name.endswith('_kl_B_to_C')})
        pairs.append(p)
    groups = {}
    for task, version in sorted({(p['task'], p['task_version']) for p in pairs}):
        rows = [p for p in pairs if (p['task'], p['task_version']) == (task, version)]
        n = len(rows)
        counts = {name: sum(p[name] for p in rows) for name in ('baseline_correct', 'candidate_correct', 'both_correct', 'both_wrong', 'all_answer_disagreement', 'correct_to_wrong', 'wrong_to_correct', 'wrong_to_wrong', 'correctness_flip')}
        metrics = {}
        # Independent unit is the paired item within a task; never mix benchmark tasks.
        rng = random.Random(seed + int(hashlib.sha256((task + version).encode()).hexdigest()[:8], 16))
        draws = [[rng.randrange(n) for _ in range(n)] for _ in range(replicates)]
        for metric in [k for k in rows[0] if k.startswith('delta_')]:
            v = [r[metric] for r in rows]
            boot = sorted(math.fsum(v[i] for i in ids) / n for ids in draws)
            metrics[metric] = {'mean': math.fsum(v) / n, 'pointwise_95_interval': [boot[int(.025 * (replicates - 1))], boot[int(.975 * (replicates - 1))]], 'unit': 'fraction' if metric == 'delta_accuracy' else ('nats' if 'nll' in metric else 'squared_probability')}
        groups[task + '@' + version] = {'n_pairs': n, 'counts': counts, 'metrics': metrics,
              'regression_denominator': counts['baseline_correct'], 'gain_denominator': n - counts['baseline_correct']}
        if 'baseline_process_exit_code' in rows[0]:
            groups[task + '@' + version]['source_process_exits'] = {
                'baseline': sorted({p['baseline_process_exit_code'] for p in rows}),
                'candidate': sorted({p['candidate_process_exit_code'] for p in rows}),
                'scope': 'Retained scalar calculations; nonzero source exits remain execution failures.'}
    return {'schema_version': 1, 'baseline_artifact': baseline[0]['artifact_id'], 'candidate_artifact': candidate[0]['artifact_id'],
            'n_pairs': len(pairs), 'excluded_baseline': [list(k) for k in sorted(b.keys() - c.keys())],
            'excluded_candidate': [list(k) for k in sorted(c.keys() - b.keys())],
            'bootstrap': {'replicates': replicates, 'seed': seed, 'unit': 'paired item within task; no cross-task aggregate', 'interval': 'pointwise descriptive percentile'},
            'tasks': groups, 'pairs': pairs}
