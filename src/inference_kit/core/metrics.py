# Derived from inference-lab; import/path boundaries changed for internal release01.
"""Prometheus discovery and scope-aware saved metric arithmetic; no GPU imports."""
import json
import math
import re

from .io import require

ROLES = {
    'running': ('vllm:num_requests_running', 'gauge'),
    'waiting': ('vllm:num_requests_waiting', 'gauge'),
    'kv_usage': ('vllm:kv_cache_usage_perc', 'gauge'),
    'preemptions': ('vllm:num_preemptions_total', 'counter'),
    'queue_histogram': ('vllm:request_queue_time_seconds', 'histogram'),
}
LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+[^\s]+)?$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"(?:,|$)')


def parse(text):
    families = {}
    samples = []
    for line in text.splitlines():
        if line.startswith('# HELP '):
            _, _, name, description = line.split(' ', 3)
            families.setdefault(name, {})['help'] = description
        elif line.startswith('# TYPE '):
            _, _, name, kind = line.split(' ', 3)
            families.setdefault(name, {})['type'] = kind
        elif line and not line.startswith('#'):
            match = LINE.fullmatch(line)
            if not match:
                continue
            name, labels, number = match.groups()
            try:
                value = float(number)
                if not math.isfinite(value):
                    continue
                labels = {m[1]: json.loads('"' + m[2] + '"') for m in LABEL.finditer(labels or '')}
                samples.append({'name': name, 'labels': labels, 'value': value})
            except (ValueError, json.JSONDecodeError):
                continue
    discovered = {}
    for role, (name, kind) in ROLES.items():
        family = families.get(name, {})
        present = bool(family.get('help')) and family.get('type') == kind
        names = (name,) if kind != 'histogram' else (name + '_sum', name + '_count', name + '_bucket')
        rows = [r for r in samples if r['name'] in names]
        discovered[role] = {'status': 'AVAILABLE' if present and rows else 'NOT_AVAILABLE', 'name': name,
                            'help': family.get('help'), 'type': family.get('type'), 'samples': rows,
                            'labels': [r['labels'] for r in rows]}
    return {'families': families, 'roles': discovered, 'samples': samples}


def value(snapshot, role, suffix=''):
    metric = snapshot.get('roles', {}).get(role, {})
    if metric.get('status') != 'AVAILABLE':
        return None
    rows = [r for r in metric['samples'] if r['name'] == metric['name'] + suffix]
    # One TP1 engine. Never silently add a second model or duplicate engine series.
    if len(rows) != 1 or rows[0]['labels'].get('model_name', 'serving-lab') != 'serving-lab':
        return None
    return rows[0]['value']


def counter_delta(before, after, role, suffix=''):
    left, right = value(before, role, suffix), value(after, role, suffix)
    if left is None or right is None:
        return {'status': 'NOT_AVAILABLE', 'delta': None}
    def labels(snapshot):
        metric = snapshot['roles'][role]
        return [r['labels'] for r in metric['samples'] if r['name'] == metric['name'] + suffix]
    if labels(before) != labels(after):
        return {'status': 'LABEL_IDENTITY_CHANGED', 'delta': None}
    if right < left:
        return {'status': 'COUNTER_RESET', 'delta': None, 'before': left, 'after': right}
    return {'status': 'OBSERVED', 'delta': right - left, 'before': left, 'after': right}


def gauge_area(observations, role, begin, end):
    """Left-hold integration within observed coverage; missing intervals remain gaps."""
    require(end >= begin, 'Invalid measurement interval')
    rows = sorted(observations, key=lambda r: r['monotonic'])
    area = covered = 0.0
    values = []
    for left, right in zip(rows, rows[1:]):
        lo, hi = max(begin, left['monotonic']), min(end, right['monotonic'])
        if hi <= lo:
            continue
        v = value(left.get('parsed', {}), role)
        if left.get('http_status') != 200 or v is None:
            continue
        area += v * (hi - lo)
        covered += hi - lo
        values.append(v)
    return {'status': 'OBSERVED_PARTIAL_COVERAGE' if covered else 'NOT_AVAILABLE',
            'time_weighted_mean': area / covered if covered else None, 'integral': area if covered else None,
            'integral_unit': 'request-seconds' if role in ('running', 'waiting') else 'fraction-seconds',
            'covered_seconds': covered, 'cell_seconds': end - begin,
            'coverage_fraction': covered / (end - begin) if end > begin else None,
            'max': max(values) if values else None,
            'integration': 'Left hold on polling intervals; no interpolation over unavailable samples'}


def summarize(observations, before, after, begin, end):
    preempt = counter_delta(before, after, 'preemptions')
    queue_sum = counter_delta(before, after, 'queue_histogram', '_sum')
    queue_count = counter_delta(before, after, 'queue_histogram', '_count')
    supported = queue_sum['status'] == queue_count['status'] == 'OBSERVED'
    count = queue_count.get('delta')
    return {'running': gauge_area(observations, 'running', begin, end),
            'waiting': gauge_area(observations, 'waiting', begin, end),
            'kv_usage': gauge_area(observations, 'kv_usage', begin, end),
            'preemption_counter_delta': preempt,
            'queue_histogram': {'status': 'OBSERVED' if supported else 'COUNTER_RESET' if 'COUNTER_RESET' in (queue_sum['status'], queue_count['status']) else 'NOT_AVAILABLE',
                                'sum_seconds_delta': queue_sum.get('delta'), 'count_delta': count,
                                'mean_seconds': queue_sum['delta'] / count if supported and count else None,
                                'meaning': 'Server aggregate histogram; not linked to individual HTTP requests'},
            'failed_polls': sum(r.get('http_status') != 200 for r in observations if begin <= r['monotonic'] <= end)}
