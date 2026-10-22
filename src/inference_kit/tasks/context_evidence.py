"""Closed-context answer/evidence retrieval on authored, controlled documents.

All operations are stdlib CPU functions. The independent source parser reads
only rendered document/question strings; it does not read authored fact/gold data.
"""
import math
import random
import re
from ..core.io import digest, require
from .base import TaskRequest

TASK = 'context_evidence_retrieval'
VERSION = '0.1'
SCORER = 'context_evidence_two_lines_v1'
REFERENCE_SOURCE = 'task_author_gold_answer_evidence_v1'
DATA_VERSION = 'b4-authored-v1'
CASES = ('direct', 'similar_entity', 'relation_distractor', 'not_found')
ASCII_OUTER = ' \t\n\r\v\f'
CODE_PATTERN = r'[A-Z]{4}-[0-9]{2}'
RELATIONS = {'route': ('route code', '경로 코드'), 'service': ('service code', '서비스 코드'),
             'channel': ('channel code', '채널 코드')}


def render_prompt(sample):
    """One fixed instruction policy, without reference/gold metadata."""
    if sample['language'] == 'en':
        instruction = ('Use only the supplied passages. Find the value for the exact asset and relation in the question. '
                       'Return exactly two lines, no explanation or markdown:\n'
                       'ANSWER: <exact code from the passage>\nEVIDENCE: <supporting passage ID, such as P04>\n'
                       'If that asset/relation fact is absent, return ANSWER: NOT_FOUND and EVIDENCE: NONE on two lines.\n')
        return instruction + 'Passages:\n' + sample['document'] + '\nQuestion: ' + sample['question']
    require(sample['language'] == 'ko', 'Unsupported retrieval language')
    instruction = ('문단만 사용. 질문의 자산·관계 값/근거ID를 복사. 두 줄만:\n'
                   'ANSWER: <code>\nEVIDENCE: <Pxx>\n'
                   '없음: ANSWER: NOT_FOUND; EVIDENCE: NONE.\n')
    return instruction + '문단:\n' + sample['document'] + '\n질문: ' + sample['question']


def canonical_reference(sample):
    text = 'ANSWER: ' + sample['gold']['answer'] + '\nEVIDENCE: ' + sample['gold']['evidence']
    start = len('ANSWER: ')
    evidence_start = text.index('\nEVIDENCE: ') + len('\nEVIDENCE: ')
    return {'text': text, 'reference_text': text, 'reference_source': REFERENCE_SOURCE,
            'reference_hash': digest(text),
            'value_spans': {'answer': [start, start + len(sample['gold']['answer'])],
                            'evidence': [evidence_start, len(text)]}}


def build_probe_sample(sample):
    ref = canonical_reference(sample)
    return {'task_id': TASK, 'task_version': VERSION, 'sample_id': sample['sample_id'],
            'source_id': sample['source_id'], 'language': sample['language'],
            'prompt': render_prompt(sample), 'reference_text': ref['text'],
            'reference_source': REFERENCE_SOURCE, 'reference_hash': ref['reference_hash'],
            'value_spans': ref['value_spans']}


def _content(entity, relation, value, language):
    if language == 'en':
        return 'For asset ' + entity + ', the ' + RELATIONS[relation][0] + ' is ' + value + '.'
    return '자산 ' + entity + '의 ' + RELATIONS[relation][1] + '는 ' + value + '이다.'


def _question(entity, relation, language):
    if language == 'en':
        return 'Which ' + RELATIONS[relation][0] + ' is assigned to asset ' + entity + '?'
    return '자산 ' + entity + '에 지정된 ' + RELATIONS[relation][1] + '는 무엇인가?'


def parse_rendered(document, question):
    """Controlled-template baseline; takes ONLY the model-visible source/query.

    This is not a general NLP retrieval system. It independently checks the
    authored template grammar, exact entity/relation and unique support.
    """
    q_en = re.fullmatch(r'Which (route code|service code|channel code) is assigned to asset (UNIT-[0-9]+)\?', question)
    q_ko = re.fullmatch(r'자산 (UNIT-[0-9]+)에 지정된 (경로 코드|서비스 코드|채널 코드)는 무엇인가\?', question)
    require(q_en is not None or q_ko is not None, 'Unknown controlled question grammar')
    if q_en:
        relation_text, entity = q_en.groups(); language = 'en'; relation_index = 0
    else:
        entity, relation_text = q_ko.groups(); language = 'ko'; relation_index = 1
    relation = next(k for k, pair in RELATIONS.items() if pair[relation_index] == relation_text)
    parsed = []; cursor = 0
    for line in document.split('\n'):
        line_match = re.fullmatch(r'\[(P0[1-6])\] (.+)', line)
        require(line_match is not None, 'Unknown controlled passage grammar')
        passage_id, content = line_match.groups()
        pattern = (r'For asset (UNIT-[0-9]+), the (route code|service code|channel code) is (' + CODE_PATTERN + r')\.'
                   if language == 'en' else
                   r'자산 (UNIT-[0-9]+)의 (경로 코드|서비스 코드|채널 코드)는 (' + CODE_PATTERN + r')이다\.')
        fact = re.fullmatch(pattern, content)
        require(fact is not None, 'Unknown controlled source fact')
        fact_entity, fact_relation, value = fact.groups()
        start = cursor + line_match.start(2)
        parsed.append({'passage_id': passage_id, 'entity': fact_entity, 'relation_text': fact_relation,
                       'value': value, 'content': content, 'content_span': [start, start + len(content)],
                       'answer_span': [start + content.index(value), start + content.index(value) + len(value)]})
        cursor += len(line) + 1
    require(len(parsed) == 6 and {p['passage_id'] for p in parsed} == {'P0' + str(i) for i in range(1, 7)},
            'Six unique controlled passage IDs required')
    matches = [p for p in parsed if p['entity'] == entity and p['relation_text'] == relation_text]
    require(len(matches) <= 1, 'Multiple/conflicting answer facts')
    if matches:
        row = matches[0]
        return {'answer': row['value'], 'evidence': row['passage_id'], 'answerable': True,
                'content_span': row['content_span'], 'answer_span': row['answer_span']}
    return {'answer': 'NOT_FOUND', 'evidence': 'NONE', 'answerable': False,
            'content_span': None, 'answer_span': None}


def verify_gold(sample):
    require(sample['task_id'] == TASK and sample['task_version'] == VERSION, 'Wrong retrieval task/version')
    require(digest(sample['document']) == sample['document_hash'], 'Retrieval document hash mismatch')
    require(parse_rendered(sample['document'], sample['question']) == sample['gold'], 'Gold/source contradiction')
    require(len(sample['passages']) == 6, 'Passage metadata missing')
    for passage in sample['passages']:
        start, end = passage['content_span']
        require(sample['document'][start:end] == passage['content'], 'Passage content span mismatch')
        require('[' + passage['passage_id'] + '] ' + passage['content'] in sample['document'], 'Passage ID/content mismatch')
    if sample['gold']['answerable']:
        a, b = sample['gold']['answer_span']; c, d = sample['gold']['content_span']
        require(sample['document'][a:b] == sample['gold']['answer'] and c <= a < b <= d, 'Gold answer span mismatch')
    return True


def make_dataset(seed=909264):
    rng = random.Random(seed); rows = []; used_values = set()
    support_positions = list(range(6)) * 3; rng.shuffle(support_positions)
    answer_index = 0
    for split in ('SMOKE', 'EVAL_LOCKED'):
        specs = ([('en', 'direct', 0), ('en', 'similar_entity', 0), ('ko', 'relation_distractor', 0), ('ko', 'not_found', 0)]
                 if split == 'SMOKE' else [(language, case, variant) for case in CASES for language in ('en', 'ko') for variant in range(3)])
        for language, case, variant in specs:
            index = len(rows)
            entity = 'UNIT-' + str(700 + index * 17 + rng.randrange(7))
            relation = rng.choice(tuple(RELATIONS)); other_relations = [r for r in RELATIONS if r != relation]
            similar = entity[:-1] + str((int(entity[-1]) + 1) % 10)
            unrelated = ['UNIT-' + str(3000 + index * 13 + j) for j in range(6)]
            facts = []
            for j in range(6):
                while True:
                    code = ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(4)) + '-' + str(rng.randrange(10, 100))
                    if code not in used_values: used_values.add(code); break
                facts.append({'entity': unrelated[j], 'relation': rng.choice(tuple(RELATIONS)), 'value': code})
            answerable = case != 'not_found'
            target_position = (rng.randrange(6) if split == 'SMOKE' else support_positions[answer_index]) if answerable else None
            if answerable and split != 'SMOKE': answer_index += 1
            # Strong distractors remain even if the queried combination is absent.
            free = [j for j in range(6) if j != target_position]
            if case in ('relation_distractor', 'not_found'):
                facts[free[0]].update(entity=entity, relation=other_relations[0])
                facts[free[1]].update(entity=unrelated[free[1]], relation=relation)
            if case == 'similar_entity':
                facts[free[0]].update(entity=similar, relation=relation)
            if answerable: facts[target_position].update(entity=entity, relation=relation)
            passages = []; lines = []; cursor = 0
            for j, fact in enumerate(facts):
                pid = 'P0' + str(j + 1); content = _content(fact['entity'], fact['relation'], fact['value'], language)
                line = '[' + pid + '] ' + content; start = cursor + len('[' + pid + '] ')
                passages.append({'passage_id': pid, 'content': content, 'content_span': [start, start + len(content)]})
                lines.append(line); cursor += len(line) + 1
            document = '\n'.join(lines)
            if answerable:
                passage = passages[target_position]; value = facts[target_position]['value']; start = passage['content_span'][0] + passage['content'].index(value)
                gold = {'answer': value, 'evidence': passage['passage_id'], 'answerable': True,
                        'content_span': passage['content_span'], 'answer_span': [start, start + len(value)]}
            else: gold = {'answer': 'NOT_FOUND', 'evidence': 'NONE', 'answerable': False, 'content_span': None, 'answer_span': None}
            row = {'task_id': TASK, 'task_version': VERSION, 'data_version': DATA_VERSION,
                   'sample_id': 'b4-' + split.lower() + '-' + language + '-' + case + '-' + str(variant + 1),
                   'source_id': 'b4-authored-source-' + str(index + 1), 'source_family': 'controlled-single-fact-v1',
                   'template_family': 'closed-context-' + case + '-' + language + '-v1', 'split': split,
                   'language': language, 'case_type': case, 'entity': entity, 'relation': relation,
                   'question': _question(entity, relation, language), 'passages': passages, 'document': document,
                   'document_hash': digest(document), 'gold': gold, 'authored_facts': facts}
            row['sample_identity'] = digest(row); verify_gold(row); rows.append(row)
    require(answer_index == 18, 'Unexpected answerable balance')
    return {'schema_version': 1, 'task_id': TASK, 'task_version': VERSION, 'data_version': DATA_VERSION,
            'scorer': SCORER, 'scope': 'SYNTHETIC_DEV_LOCKED', 'data_seed': seed,
            'rows': rows, 'rows_identity': digest(rows), 'limitations': 'Controlled grammar; shared source family; not a general retrieval benchmark or translation pairs.'}


authored_data = make_dataset
generate_data = make_dataset


def select_probes(rows, seed=909265):
    rng = random.Random(seed); result = []
    for case, language in (('direct', 'en'), ('similar_entity', 'ko'), ('relation_distractor', 'en'), ('not_found', 'ko')):
        options = sorted((r for r in rows if r['split'] == 'EVAL_LOCKED' and r['case_type'] == case and r['language'] == language), key=lambda r: r['sample_id'])
        require(options, 'Missing prespecified retrieval diagnostic stratum'); result.append(rng.choice(options)['sample_id'])
    return result


def parse_output(text):
    require(isinstance(text, str), 'Output text must be a string')
    normalized = text.replace('\r\n', '\n').replace('\r', '\n').strip(ASCII_OUTER)
    lines = normalized.split('\n') if normalized else []
    fields = {}
    for name in ('ANSWER', 'EVIDENCE'):
        matches = []
        for line in lines:
            match = re.fullmatch(r'[ \t]*' + name + r':[ \t]*(.*?)[ \t]*', line)
            if match: matches.append(match.group(1))
        fields[name.lower()] = {'present': bool(matches), 'count': len(matches), 'ambiguous': len(matches) > 1,
                                'value': matches[0] if len(matches) == 1 else None}
    answer = fields['answer']; evidence = fields['evidence']
    answer['format_valid'] = bool(answer['value'] is not None and re.fullmatch(CODE_PATTERN + '|NOT_FOUND', answer['value']))
    evidence['format_valid'] = bool(evidence['value'] is not None and re.fullmatch(r'P[0-9]{2}|NONE', evidence['value']))
    ordered = len(lines) == 2 and re.fullmatch(r'[ \t]*ANSWER:[ \t]*(.*?)[ \t]*', lines[0]) is not None and re.fullmatch(r'[ \t]*EVIDENCE:[ \t]*(.*?)[ \t]*', lines[1]) is not None
    return {'normalized_text': normalized, 'normalization_version': 'crlf_ascii_outer_v1', 'fields': fields,
            'format_valid': bool(ordered and answer['format_valid'] and evidence['format_valid'])}


def score(sample, generation):
    verify_gold(sample)
    if generation is not None:
        require(generation.get('task_id') == TASK and generation.get('task_version') == VERSION and generation.get('sample_id') == sample['sample_id'], 'Generation/task/sample identity mismatch')
        require(generation.get('sample_identity') == sample['sample_identity'], 'Generation sample content changed')
        require(generation.get('request_kind', 'generate_until') == 'generate_until', 'Wrong collector for retrieval generation')
        for key, want in (('prompt_identity', digest(render_prompt(sample))), ('scorer_version', SCORER)):
            if key in generation: require(generation[key] == want, 'Retrieval generation identity mismatch: ' + key)
    result = {'task_id': TASK, 'task_version': VERSION, 'scorer': SCORER, 'sample_id': sample['sample_id'],
              'sample_identity': sample['sample_identity'], 'status': 'SCORED' if generation is not None else 'NOT_RUN',
              'request_id': generation.get('request_id') if generation else None, 'answerable': sample['gold']['answerable'],
              'language': sample['language'], 'case_type': sample['case_type'], 'termination': 'NOT_RUN',
              'transport_complete': None, 'natural_stop': None, 'answer_field_present': None, 'answer_format_valid': None,
              'evidence_field_present': None, 'evidence_format_valid': None, 'answer_exact': None, 'evidence_exact': None,
              'cited_id_exists': None, 'cited_id_status': 'NOT_OBSERVED', 'format_valid': None,
              'task_success': None, 'task_success_within_10s': None, 'false_abstention': None, 'false_answer': None}
    if generation is None: return result
    parsed = parse_output(generation.get('output_text', '')); answer = parsed['fields']['answer']; evidence = parsed['fields']['evidence']
    termination = generation.get('termination', 'UNKNOWN'); transport = bool(generation.get('transport_complete'))
    result.update(transport_complete=transport, termination=termination, natural_stop=termination == 'NATURAL_STOP',
                  length_cap=termination == 'LENGTH_CAP', empty=termination == 'EMPTY_OUTPUT',
                  answer_field_present=answer['present'], answer_format_valid=answer['format_valid'],
                  evidence_field_present=evidence['present'], evidence_format_valid=evidence['format_valid'],
                  answer_exact=answer['format_valid'] and answer['value'] == sample['gold']['answer'],
                  evidence_exact=evidence['format_valid'] and evidence['value'] == sample['gold']['evidence'],
                  format_valid=parsed['format_valid'], parsed=parsed, request_latency_http=generation.get('request_latency_http'),
                  request_latency_task=generation.get('request_latency_task'))
    if evidence['value'] == 'NONE' and evidence['format_valid']: result['cited_id_status'] = 'NOT_APPLICABLE'
    elif evidence['value'] is not None:
        result['cited_id_exists'] = evidence['value'] in {p['passage_id'] for p in sample['passages']}
        result['cited_id_status'] = 'OBSERVED'
    result['false_abstention'] = bool(sample['gold']['answerable'] and answer['value'] == 'NOT_FOUND')
    result['false_answer'] = bool(not sample['gold']['answerable'] and answer['value'] is not None and answer['value'] != 'NOT_FOUND')
    result['answer_correct_evidence_wrong'] = result['answer_exact'] and not result['evidence_exact']
    result['evidence_correct_answer_wrong'] = result['evidence_exact'] and not result['answer_exact']
    result['values_correct_format_wrong'] = result['answer_exact'] and result['evidence_exact'] and not result['format_valid']
    result['task_success'] = bool(transport and result['natural_stop'] and result['format_valid'] and result['answer_exact'] and result['evidence_exact'])
    latency = generation.get('request_latency_http')
    result['task_success_within_10s'] = bool(result['task_success'] and type(latency) in (int, float) and math.isfinite(latency) and 0 <= latency <= 10)
    return result


def aggregate(samples, generations):
    require(len({g['sample_id'] for g in generations}) == len(generations), 'Duplicate generation sample')
    expected = {s['sample_id'] for s in samples}; require(all(g['sample_id'] in expected for g in generations), 'Unknown generation sample')
    by_id = {g['sample_id']: g for g in generations}; scores = [score(s, by_id.get(s['sample_id'])) for s in samples]
    observed = [s for s in scores if s['status'] == 'SCORED']
    names = ('transport_complete', 'natural_stop', 'answer_field_present', 'answer_format_valid', 'evidence_field_present',
             'evidence_format_valid', 'answer_exact', 'evidence_exact', 'format_valid', 'task_success', 'task_success_within_10s')
    def counts(subset):
        seen = [s for s in subset if s['status'] == 'SCORED']
        return {'planned': len(subset), 'observed': len(seen), 'not_run': len(subset) - len(seen),
                'metrics': {name: {'correct': sum(s[name] is True for s in seen), 'planned_denominator': len(subset),
                                   'observed_denominator': len(seen)} for name in names},
                'false_abstention': sum(s['false_abstention'] is True for s in seen),
                'false_answer': sum(s['false_answer'] is True for s in seen)}
    summary = dict(counts(scores), expected=len(samples), submitted=len(observed), complete=len(observed) == len(samples), scope='SYNTHETIC_DEV_LOCKED')
    summary['groups'] = {'language': {k: counts([s for s in scores if s['language'] == k]) for k in ('en', 'ko')},
                         'case_type': {k: counts([s for s in scores if s['case_type'] == k]) for k in CASES},
                         'answerable': {str(k).lower(): counts([s for s in scores if s['answerable'] == k]) for k in (True, False)}}
    return scores, summary


def baseline_generations(rows, mode='rule_based_controlled_template'):
    require(mode in ('rule_based_controlled_template', 'always_not_found'), 'Unknown controlled baseline')
    result = []
    for row in rows:
        fact = parse_rendered(row['document'], row['question']) if mode == 'rule_based_controlled_template' else {'answer': 'NOT_FOUND', 'evidence': 'NONE'}
        result.append({'sample_id': row['sample_id'], 'task_id': TASK, 'task_version': VERSION, 'sample_identity': row['sample_identity'],
                       'request_id': mode + '|' + row['sample_id'], 'request_kind': 'generate_until',
                       'output_text': 'ANSWER: ' + fact['answer'] + '\nEVIDENCE: ' + fact['evidence'],
                       'transport_complete': True, 'termination': 'NATURAL_STOP', 'request_latency_http': None,
                       'evidence_kind': 'CPU_CONTROLLED_TEMPLATE_BASELINE'})
    return result


class ContextEvidenceAdapter:
    task_id = TASK; task_version = VERSION; request_kind = 'generate_until'; scorer_version = SCORER
    def render_prompt(self, sample): return render_prompt(sample)
    def requests(self, sample, execution, tokenizer_identity, token_id_space=None, runtime_identity=None):
        verify_gold(sample)
        require(sample.get('scorer_version', SCORER) == SCORER, 'Wrong retrieval scorer version')
        prompt_identity = digest(render_prompt(sample))
        if 'prompt_identity' in sample: require(sample['prompt_identity'] == prompt_identity, 'Prepared retrieval prompt changed')
        if 'tokenizer_identity' in sample: require(sample['tokenizer_identity'] == tokenizer_identity, 'Prepared tokenizer identity mismatch')
        identity = {'task_id': TASK, 'task_version': VERSION, 'scorer_version': SCORER, 'split': sample['split'],
                    'sample_id': sample['sample_id'], 'sample_identity': sample['sample_identity'],
                    'source_identity': sample['document_hash'], 'prompt_identity': prompt_identity,
                    'tokenizer_identity': tokenizer_identity, 'token_id_space': token_id_space,
                    'runtime_identity': runtime_identity}
        return [TaskRequest(execution + '|' + TASK + '|' + sample['sample_id'], TASK, VERSION, self.request_kind,
                            sample['sample_id'], {'token_ids': sample['generation_input_ids']}, identity).validate()]
    def reference(self, sample): return canonical_reference(sample)
    def score(self, sample, records): return score(sample, records)
    def metric_specs(self):
        return [{'metric_id': 'task_success', 'version': SCORER, 'unit': 'document fraction',
                 'formula': 'transport complete AND natural stop AND exact two-line format AND answer exact AND evidence exact'},
                {'metric_id': 'answer_exact', 'version': SCORER, 'unit': 'document fraction', 'formula': 'unique valid ANSWER literal equals external authored gold'},
                {'metric_id': 'evidence_exact', 'version': SCORER, 'unit': 'document fraction', 'formula': 'unique valid EVIDENCE literal equals support ID, or NONE if absent'},
                {'metric_id': 'task_success_within_10s', 'version': 'b4-http-deadline-v1', 'unit': 'document fraction',
                 'formula': 'task_success AND 0 <= response_complete-http_submit <= 10 seconds; DEMO only'}]
