"""Closed stdlib raw task contract fixtures; no historical files or SDK required."""
from copy import deepcopy
import unittest
from inference_kit.core.io import digest, loads
from inference_kit.tasks import contract_policy, context_evidence as ce
from inference_kit.tasks.input_records import validate_raw, render_record, arc_context, render_pinned_chat


def tech_sample():
    events = [{'experiment_id': 'EXP-901', 'field': 'temperature_c', 'value': 27.125, 'line_id': 'L1'},
              {'experiment_id': 'EXP-901', 'field': 'status', 'value': 'complete', 'line_id': 'L2'}]
    text = '\n'.join('[' + e['line_id'] + '] ' + contract_policy._render_event(e, 'en') for e in events)
    sample = {'task_id': 'tech_record_extract', 'task_version': '0.1', 'sample_id': 'raw-tech-1', 'source_id': 'raw-author-1',
              'template_family': 'b2-pipe-record-basic-v1', 'case_type': 'basic', 'language': 'en', 'split': 'ENGINEERING_INPUT_ONLY',
              'experiment_id': 'EXP-901', 'events': events, 'document': text, 'document_hash': digest(text),
              'gold': {'experiment_id': 'EXP-901', 'temperature_c': 27.125, 'status': 'complete',
                       'support': {'temperature_c': ['L1'], 'status': ['L2']}}}
    sample['sample_identity'] = digest(sample)
    return sample


def retrieval_sample():
    passages = []; lines = []; cursor = 0
    for index in range(6):
        content = ce._content('UNIT-' + str(301 + index), 'route', 'MORA-' + str(31 + index), 'ko')
        pid = 'P0' + str(index + 1); line = '[' + pid + '] ' + content
        passages.append({'passage_id': pid, 'content': content, 'content_span': [cursor + 6, cursor + 6 + len(content)]})
        lines.append(line); cursor += len(line) + 1
    document = '\n'.join(lines); question = ce._question('UNIT-303', 'route', 'ko')
    sample = {'task_id': ce.TASK, 'task_version': '0.1', 'data_version': 'contract-fixture', 'sample_id': 'raw-retrieval-1',
              'source_id': 'raw-retrieval-source', 'source_family': 'controlled-single-fact-v1', 'template_family': 'closed-context-direct-ko-v1',
              'split': 'ENGINEERING_INPUT_ONLY', 'language': 'ko', 'case_type': 'direct', 'entity': 'UNIT-303', 'relation': 'route',
              'question': question, 'passages': passages, 'document': document, 'document_hash': digest(document),
              'gold': ce.parse_rendered(document, question), 'authored_facts': []}
    sample['sample_identity'] = digest(sample)
    return sample


def raw_manifest(task='tech_record_extract'):
    if task == 'tech_record_extract':
        sample = tech_sample(); policy = 'P00'; source = 'task_author_gold_json'
    elif task == ce.TASK:
        sample = retrieval_sample(); policy = 'context_evidence_fixed_v1'; source = ce.REFERENCE_SOURCE
    else:
        question = 'Which material is magnetic?'; document = {'id': 'raw-arc-1', 'question': question,
            'choices': {'text': ['iron', 'paper', 'glass'], 'label': ['A', 'B', 'C']}, 'answerKey': 'A'}
        sample = {'document': document, 'prompt_context': arc_context(question), 'prompt_contract_id': 'arc_original_completion_v1', 'original_index': 7}
        policy = 'arc_original_completion_v1'; source = 'dataset_answer_key_choice'
    arc = task == 'arc_challenge'
    row = {'task_id': task, 'task_version': '1.0' if arc else '0.1', 'dataset_id': 'raw-fixture',
           'source_id': sample.get('source_id', 'arc-original'), 'sample_id': sample.get('sample_id', 'raw-arc-1'),
           'split': sample.get('split', 'test'), 'usage_scope': 'REGRESSION_REUSE' if arc else 'ENGINEERING_INPUT_ONLY',
           'policy_id': policy, 'reference_source': source, 'sample': sample}
    return {'schema_version': 1, 'kind': 'raw_task_manifest_v1', 'dataset_id': 'raw-fixture', 'usage_scope': 'CONTRACT_TEST', 'rows': [row]}


class RawTaskTests(unittest.TestCase):
    def test_three_existing_task_renderers_and_roles(self):
        for task in ('tech_record_extract', ce.TASK, 'arc_challenge'):
            raw = raw_manifest(task); self.assertIs(validate_raw(raw), raw)
            result = render_record(raw['rows'][0]); self.assertEqual(result['task_id'], task)
            self.assertEqual(result['task_prompt'], result['content']); self.assertTrue(result['reference_text'])
            self.assertNotIn('input_ids', result['authored_sample'])
        arc = render_record(raw_manifest('arc_challenge')['rows'][0])
        self.assertEqual(arc['reference_text'], ' iron'); self.assertEqual(arc['prompt_rendering'], 'already_rendered')
        self.assertEqual(arc['authored_sample']['choice_labels'], ['A', 'B', 'C'])

    def test_hidden_gold_changes_only_reference(self):
        for task in ('tech_record_extract', ce.TASK, 'arc_challenge'):
            row = raw_manifest(task)['rows'][0]; before = render_record(row); changed = deepcopy(row)
            if task == 'tech_record_extract': changed['sample']['gold']['temperature_c'] = 98.75
            elif task == ce.TASK: changed['sample']['gold']['answer'] = 'BETA-99'
            else: changed['sample']['document']['answerKey'] = 'B'
            after = render_record(changed)
            self.assertEqual(before['content'], after['content']); self.assertNotEqual(before['reference_text'], after['reference_text'])
            # Actual preparation still rejects contradictory natural gold.
            if task != 'arc_challenge':
                with self.assertRaises(ValueError): validate_raw(dict(raw_manifest(task), rows=[changed]))

    def test_required_nonempty_unknown_and_duplicate_namespace(self):
        for mutate in (lambda m: m.update(rows=[]), lambda m: m.update(unknown=1), lambda m: m['rows'][0].update(hidden_gold_flag=True), lambda m: m['rows'].append(deepcopy(m['rows'][0]))):
            raw = raw_manifest(); mutate(raw)
            with self.assertRaises(ValueError): validate_raw(raw)
        raw = raw_manifest(); raw['schema_version'] = True
        with self.assertRaises(ValueError): validate_raw(raw)
        with self.assertRaises(ValueError): loads('{"rows": [], "rows": []}')

    def test_nested_unknown_type_and_nonfinite_rejected(self):
        for mutate in (lambda s: s['gold'].update(extra=True), lambda s: s['events'][0].update(value=True),
                       lambda s: s['gold'].update(temperature_c=float('nan')), lambda s: s['events'][0].update(custom_template='eval(1)')):
            raw = raw_manifest(); mutate(raw['rows'][0]['sample'])
            with self.assertRaises(ValueError): validate_raw(raw)
        for key in ('answer_span', 'content_span'):
            raw = raw_manifest(ce.TASK); raw['rows'][0]['sample']['gold'][key] = [False, 1]
            with self.assertRaises(ValueError): validate_raw(raw)

    def test_wrong_namespace_policy_reference_and_gold_rejected(self):
        for field, value in (('sample_id', 'different'), ('task_version', '99'), ('policy_id', 'P11'), ('reference_source', 'model_output')):
            raw = raw_manifest(); raw['rows'][0][field] = value
            with self.assertRaises(ValueError): validate_raw(raw)
        raw = raw_manifest(); raw['rows'][0]['sample']['gold']['support']['temperature_c'] = ['L99']
        with self.assertRaises(ValueError): validate_raw(raw)

    def test_arc_raw_all_choices_order_context_and_bool_index(self):
        raw = raw_manifest('arc_challenge'); validate_raw(raw)
        for mutate in (lambda s: s.update(original_index=False), lambda s: s.update(prompt_context=s['prompt_context'] + 'gold'),
                       lambda s: s.update(prompt_contract_id='custom.py'), lambda s: s['document']['choices']['label'].pop(),
                       lambda s: s['document']['choices']['text'].__setitem__(0, ''), lambda s: s.update(input_ids=[1, 2])):
            bad = deepcopy(raw); mutate(bad['rows'][0]['sample'])
            with self.assertRaises(ValueError): validate_raw(bad)
        reordered = deepcopy(raw); reordered['rows'][0]['sample']['document']['choices']['text'].reverse()
        result = render_record(validate_raw(reordered)['rows'][0])
        self.assertEqual(result['authored_sample']['choices'], ['glass', 'paper', 'iron'])
        self.assertEqual(result['reference_text'], ' glass')

    def test_pinned_chat_and_unicode_are_literal_data(self):
        content = '한국어 cafe\u0301 🙂\n</script> __import__("os")'
        rendered = render_pinned_chat(content)
        self.assertEqual(rendered.count('<|im_start|>user'), 1)
        self.assertIn(content, rendered); self.assertTrue(rendered.endswith('<|im_start|>assistant\n'))
        self.assertEqual(arc_context('Q?'), render_pinned_chat('Question: Q?\nAnswer:'))

    def test_namespaced_sample_ids_allow_different_tasks(self):
        a = raw_manifest(); b = raw_manifest('arc_challenge')
        b['rows'][0]['sample_id'] = a['rows'][0]['sample_id']; b['rows'][0]['sample']['document']['id'] = a['rows'][0]['sample_id']
        a['rows'].extend(b['rows']); validate_raw(a)

if __name__ == '__main__': unittest.main()
