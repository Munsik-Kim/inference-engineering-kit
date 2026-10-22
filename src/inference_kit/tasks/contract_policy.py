"""Frozen evidence/status instruction interventions over unchanged tech_record semantics."""
import random
import re
from decimal import Decimal
from . import tech_record as task
from ..core.io import digest, require

POLICIES = {'P00': (0, 0), 'P10': (1, 0), 'P01': (0, 1), 'P11': (1, 1)}
POLICY_VERSION = 'evidence_enum_emphasis_v1'
DATA_VERSION = 'tech_record_authored_b2_v1'
PREPARATION = 'chat_rendered_contract640_field_spans_v1'
EVIDENCE = {
 'en': 'Required evidence object: include both temperature_c and status keys. Use supporting line-ID strings such as "L2", never temperature numbers or status words. For missing facts use null for the value and its evidence; never omit keys.',
 'ko': 'evidence object 필수: temperature_c/status keys 모두. Values: "L2" 같은 line-ID string/null only, not numbers/status words. Missing fact: value and evidence=null; no omitted keys.'}
STATUS = {
 'en': 'Output status only as complete, repeat, pending, or null: finished=complete, repeat required=repeat, waiting=pending. Missing status=null, not pending. Do not output source synonyms.',
 'ko': 'status only complete/repeat/pending/null. 완료=complete; 반복 필요=repeat; 대기=pending. Missing=null, not pending. No synonyms.'}


def policy_record(policy):
    require(policy in POLICIES, 'Unknown declared prompt policy')
    e, s = POLICIES[policy]
    value = {'policy_id': policy, 'version': POLICY_VERSION, 'evidence_instruction': e,
             'status_instruction': s, 'evidence_block': EVIDENCE if e else None,
             'status_block': STATUS if s else None, 'append_order': ['E', 'S']}
    return value | {'policy_identity': digest(value)}


def render(sample, policy):
    e, s = POLICIES[policy]
    blocks = ([EVIDENCE[sample['language']]] if e else []) + ([STATUS[sample['language']]] if s else [])
    return task.render_prompt(sample) + (('\n\n' + '\n'.join(blocks)) if blocks else '')


def probe_sample(sample, policy):
    return task.build_probe_sample(sample) | {'prompt': render(sample, policy)}


def _render_event(event, language):
    if event['field'] == 'note':
        return event['text']
    if language == 'en':
        marker = 'Final revision' if event.get('final') else 'Record'
        key = 'temperature_c' if event['field'] == 'temperature_c' else 'state'
        value = event['value'] if key == 'temperature_c' else {'complete': 'finished', 'repeat': 'repeat required', 'pending': 'waiting'}[event['value']]
        return f"{marker} {event['experiment_id']} | {key} = {value}" + (' C.' if key == 'temperature_c' else '.')
    marker = 'Final' if event.get('final') else '기록'
    key = '온도' if event['field'] == 'temperature_c' else '상태'
    value = event['value'] if key == '온도' else {'complete': '완료', 'repeat': '반복 필요', 'pending': '대기'}[event['value']]
    return f"{marker} {event['experiment_id']} | {key} = {value}" + (' C.' if key == '온도' else '.')


def verify_gold(sample):
    """Independent rendered-line parser; it does not call the event-to-gold builder."""
    require(digest(sample['document']) == sample['document_hash'], 'B2 document changed')
    pattern = r'^\[(L\d+)\] (Record|Final revision|기록|Final) (EXP-\d+) \| (temperature_c|state|온도|상태) = (.+)\.$'
    parsed = []
    for line in sample['document'].splitlines():
        match = re.fullmatch(pattern, line)
        if not match:
            continue
        lid, marker, eid, key, value = match.groups()
        field = 'temperature_c' if key in ('temperature_c', '온도') else 'status'
        if field == 'temperature_c':
            require(value.endswith(' C'), 'B2 Celsius unit missing')
            value = Decimal(value[:-2])
        else:
            require(value in task.STATUS, 'Unknown source status expression')
            value = task.STATUS[value]
        parsed.append((lid, marker in ('Final revision', 'Final'), eid, field, value))
    require(sample['gold']['experiment_id'] == sample['experiment_id'], 'B2 gold ID differs')
    for field in ('temperature_c', 'status'):
        facts = [r for r in parsed if r[2] == sample['experiment_id'] and r[3] == field]
        finals = [r for r in facts if r[1]]
        chosen = finals or facts
        require(len({r[4] for r in chosen}) <= 1, 'Conflicting final B2 facts')
        want = chosen[0][4] if chosen else None
        gold = sample['gold'][field]
        require(want == (Decimal(str(gold)) if field == 'temperature_c' and gold is not None else gold), 'B2 rendered/gold mismatch')
        require({r[0] for r in facts if r[4] == want} == set(sample['gold']['support'][field]), 'B2 support set mismatch')
    return True


def make_dataset(seed=909241):
    rng = random.Random(seed)
    rows = []
    for split, cases in (('SMOKE', ('basic', 'final_correction')), ('EVAL_LOCKED', task.CASES)):
        for language in ('en', 'ko'):
            for case in cases:
                for variant in range(1 if split == 'SMOKE' else 2):
                    index = len(rows)
                    target = 'EXP-' + str(2203 + index * 13)
                    other = target + '2'
                    temp = round(18.375 + index * 1.125 + (2.25 if language == 'ko' else 0), 3)
                    state = ('pending', 'complete', 'repeat')[(index + variant) % 3]
                    if case == 'numeric':
                        temp = (-7.875 if variant == 0 else 0.0625) + (1.5 if language == 'ko' else 0)
                    events = [{'experiment_id': target, 'field': 'temperature_c', 'value': temp},
                              {'experiment_id': target, 'field': 'status', 'value': state},
                              {'field': 'note', 'text': 'Note: archive.' if language == 'en' else '메모: 보관.'}]
                    if case == 'similar_id':
                        events += [{'experiment_id': other, 'field': 'temperature_c', 'value': temp + 27},
                                   {'experiment_id': other, 'field': 'status', 'value': 'complete' if state != 'complete' else 'pending'}]
                    if case == 'final_correction':
                        events += [{'experiment_id': target, 'field': 'temperature_c', 'value': round(temp - 6.25, 3), 'final': True},
                                   {'experiment_id': target, 'field': 'status', 'value': ('repeat' if state != 'repeat' else 'complete'), 'final': True}]
                    if case == 'missing':
                        missing = 'temperature_c' if variant == 0 else 'status'
                        events = [e for e in events if e['field'] != missing]
                    if case == 'reordered':
                        events += [{'experiment_id': other, 'field': 'temperature_c', 'value': temp + 11}]
                    rng.shuffle(events)
                    for i, event in enumerate(events):
                        event['line_id'] = 'L' + str(i + 1)
                    gold = {'experiment_id': target, 'support': {}}
                    for field in ('temperature_c', 'status'):
                        facts = [e for e in events if e.get('experiment_id') == target and e['field'] == field]
                        chosen = [e for e in facts if e.get('final')] or facts
                        gold[field] = chosen[0]['value'] if chosen else None
                        gold['support'][field] = sorted(e['line_id'] for e in facts if e['value'] == gold[field])
                    document = '\n'.join('[' + e['line_id'] + '] ' + _render_event(e, language) for e in events)
                    row = {'task_id': task.TASK, 'task_version': task.VERSION,
                           'sample_id': f'b2-{split.lower()}-{language}-{case}-{variant + 1}',
                           'source_id': f'b2-author-{index}', 'template_family': f'b2-pipe-record-{case}-v1',
                           'source_family': 'authored_b2_new_facts_wording_seeded_layout', 'case_type': case,
                           'language': language, 'split': split, 'experiment_id': target,
                           'events': events, 'document': document, 'document_hash': digest(document), 'gold': gold}
                    row['sample_identity'] = digest(row)
                    verify_gold(row)
                    rows.append(row)
    return {'schema_version': 1, 'task_id': task.TASK, 'task_version': task.VERSION, 'scorer': task.SCORER,
            'data_version': DATA_VERSION, 'data_seed': seed, 'scope': 'authored_synthetic_locked_pilot_not_benchmark_test',
            'rows': rows, 'rows_identity': digest(rows)}


def select_probes(rows, seed=909243):
    rng = random.Random(seed)
    return [rng.choice([r for r in rows if r['split'] == 'EVAL_LOCKED' and r['case_type'] == case and r['language'] == language])['sample_id']
            for case, language in (('basic', 'en'), ('final_correction', 'ko'), ('missing', 'en'), ('similar_id', 'ko'))]


def request_order(rows, seed=909242):
    rng = random.Random(seed)
    result = []
    for split in ('SMOKE', 'EVAL_LOCKED'):
        ids = [r['sample_id'] for r in rows if r['split'] == split]
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            policies = list(POLICIES)
            policies = policies[i % 4:] + policies[:i % 4]
            result += [{'sample_id': sid, 'policy_id': p, 'split': split} for p in policies]
    return result


def diagnostics(sample, generation):
    """Additional diagnostics never replace or alter task.score's strict result."""
    result = task.score(sample, generation)
    names = ('status_enum_valid', 'evidence_required_keys_present', 'evidence_value_type_valid',
             'evidence_line_id_exists', 'core_values_correct')
    if generation is None:
        return result | {'diagnostic_metric_version': 'b2-contract-v1', 'diagnostics': dict.fromkeys(names)}
    obj = {}
    try:
        value = task.strict_json(generation.get('output_text', ''))
        if isinstance(value, dict):
            obj = value
    except (ValueError, TypeError, ArithmeticError):
        pass
    ev = obj.get('evidence')
    present = isinstance(ev, dict) and {'temperature_c', 'status'} <= ev.keys()
    typed = present and all(ev[k] is None or type(ev[k]) is str for k in ('temperature_c', 'status'))
    lines = set(re.findall(r'^\[(L\d+)\]', sample['document'], re.MULTILINE))
    value = {'status_enum_valid': 'status' in obj and (obj['status'] is None or type(obj['status']) is str and obj['status'] in ('complete', 'repeat', 'pending')),
             'evidence_required_keys_present': present, 'evidence_value_type_valid': typed,
             'evidence_line_id_exists': typed and all(ev[k] is None or ev[k] in lines for k in ('temperature_c', 'status')),
             'core_values_correct': bool(result['transport_complete'] and result['json_valid'] and all(result['field_correct'].values()))}
    return result | {'diagnostic_metric_version': 'b2-contract-v1', 'diagnostics': value,
                     'error_combination': sorted(k for k, v in value.items() if not v)}
