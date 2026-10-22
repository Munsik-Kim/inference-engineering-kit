"""Closed synthetic boundary tests: no historical checkout/helper dependence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference_kit.core.io import digest, sha, write
from inference_kit.suites.suite_inputs import load_sources, validate_slice, validate_diagnostic
from inference_kit.tasks.tech_record import canonical_reference


def synthetic_slice():
    authored = {'sample_id': '합성-contract', 'task_id': 'tech_record_extract', 'task_version': '0.1',
                'experiment_id': 'EXP-1', 'document': 'No result.', 'gold': {'experiment_id': 'EXP-1', 'temperature_c': None, 'status': None, 'support': {'temperature_c': [], 'status': []}}}
    authored['document_hash'] = digest(authored['document'])
    authored['sample_identity'] = digest(authored)
    text = canonical_reference(authored)['text']; prompt = 'p'; ids = [3] * (len(text) + 1)
    row = {'task_id': 'tech_record_extract', 'task_version': '0.1', 'sample_id': authored['sample_id'],
           'source_bundle_identity': '3'*64, 'source_plan_identity': '5'*64, 'source_scorer_identity': '8'*64, 'dataset_identity': '7'*64,
           'reference_role': 'author_gold_json', 'reference_source': 'task_author_gold_json',
           'document_hash': authored['document_hash'], 'sample_identity': authored['sample_identity'],
           'input_ids': ids, 'generation_input_ids': [4], 'prompt': prompt, 'reference_text': text, 'full_text': prompt + text,
           'prompt_hash': digest(prompt), 'reference_hash': digest(text), 'full_text_hash': digest(prompt + text),
           'sequence_identity': digest(ids), 'prompt_token_boundary': 1, 'attention_mask': [1] * len(ids),
           'special_token_ids': [0], 'special_token_mask': [0] * len(ids), 'loss_mask': [0] + [1] * len(text),
           'offsets_codepoints': [[i, i + 1] for i in range(len(ids))], 'offsets_utf8_bytes': [[i, i + 1] for i in range(len(ids))],
           'tokenization': {'tokenizer_identity': '1' * 64, 'token_id_space': '2' * 64},
           'targets': [{'token_position': 1, 'logit_position': 0, 'target_token_id': 3, 'prefix_hash': digest({'token_ids': [3], 'token_id_space': '2' * 64})}],
           'full_vocab_target_positions': [], 'value_spans': {}}
    row['probe_input_identity'] = digest(row)
    body = {'prompt': [4], 'stream': True, 'echo': False, 'ignore_eos': False, 'max_tokens': 192, 'temperature': 0, 'top_p': 1, 'n': 1, 'add_special_tokens': False}
    record = {'request_body': body, 'sample_id': authored['sample_id'], 'task_id': 'tech_record_extract', 'task_version': '0.1', 'tokenizer_identity': '1' * 64, 'reference_used_for_generation': False}
    line = json.dumps(record, ensure_ascii=False) + '\n'
    return {'schema_version': 1, 'kind': 'prepared_suite_slice_v1', 'source_id': 'synthetic', 'task_id': 'tech_record_extract', 'task_version': '0.1', 'request_kind': 'generate_until',
            'validation_scope': 'SLICE_CONTRACT_VALIDATED_ONLY',
            'ancestry': {'parent_raw_manifest_sha256': '3' * 64, 'parent_plan_sha256': '4' * 64, 'parent_plan_identity': '5' * 64, 'parent_acquisition_source_identity': '6' * 64, 'parent_dataset_identity': '7' * 64, 'scorer_identity': '8' * 64, 'scorer_version': 'tech_record_strict_json_v1', 'probe_inputs_sha256': '9' * 64, 'probe_selection_identity': 'a' * 64, 'included_sample_ids': [authored['sample_id']], 'excluded_sample_count': 0, 'scope': 'SYNTHETIC_CONTRACT_TEST'},
            'model_metadata': {'pair': {'model_id': 'synthetic', 'revision': 'synthetic'}, 'model_config': {'hidden_size': 2}, 'tokenizer_identity': '1' * 64, 'token_id_space': '2' * 64, 'vocabulary_size': 128},
            'samples': [{'sample_id': authored['sample_id'], 'authored_sample': authored, 'diagnostic': row,
                         'generation': {'prepared': deepcopy(row), 'wire_payload': body, 'proof': {'parent_file': 'attempts/raw.jsonl', 'parent_file_sha256': 'b' * 64, 'line_number': 1, 'line_sha256': hashlib.sha256(line.encode('utf-8')).hexdigest(), 'raw_line': line}}}]}


def repin_row(row):
    row['probe_input_identity'] = digest({k: v for k, v in row.items() if k != 'probe_input_identity'})


def synthetic_arc_slice():
    value = synthetic_slice(); value.update(task_id='arc_challenge', task_version='1.0', request_kind='choice_loglikelihood')
    value['ancestry']['scorer_version'] = 'arc_choice_sum_character_norm_v1'
    sid = 'arc-contract'; choices = ['one', 'two', 'six']; labels = ['1','2','3']; prompt = 'Q:'
    reqs = [{'choice_index': i, 'context': prompt, 'continuation': ' '+choice, 'context_ids': [3,3],
             'continuation_ids': [4]*(len(choice)+1), 'input_ids': [3,3]+[4]*(len(choice)+1)} for i,choice in enumerate(choices)]
    authored = {'sample_id': sid, 'choices': choices, 'choice_labels': labels, 'gold': 1, 'prompt': prompt, 'question': 'Q?',
                'document': {'id': sid, 'question': 'Q?', 'answerKey': '2', 'choices': {'label': labels, 'text': choices}},
                'requests': reqs, 'split': 'test', 'source_text_hash': 'f'*64}
    reference=' two'; ids=reqs[1]['input_ids']; text=prompt+reference
    row=value['samples'][0]['diagnostic'];row.update(task_id='arc_challenge', task_version='1.0', sample_id=sid,
        reference_role='dataset_gold_choice', reference_source='dataset_answer_key_choice', input_ids=ids,
        prompt=prompt, reference_text=reference, full_text=text, prompt_hash=digest(prompt), reference_hash=digest(reference),
        full_text_hash=digest(text), sequence_identity=digest(ids), prompt_token_boundary=2, attention_mask=[1]*len(ids),
        special_token_mask=[0]*len(ids), loss_mask=[0,0]+[1]*len(reference), offsets_codepoints=[[i,i+1] for i in range(len(ids))],
        offsets_utf8_bytes=[[i,i+1] for i in range(len(ids))], gold_choice_index=1,
        targets=[{'token_position':2,'logit_position':1,'target_token_id':4,'prefix_hash':digest({'token_ids':[3,3],'token_id_space':'2'*64})}])
    repin_row(row)
    value['ancestry']['included_sample_ids']=[sid]
    value['samples']=[{'sample_id':sid,'authored_sample':authored,'diagnostic':row,
                      'likelihood':{'requests':deepcopy(reqs),'choice_labels':labels[:],'gold_index':1}}]
    return value


class SuitePreparedInputsTests(unittest.TestCase):
    def setUp(self):
        self.value = synthetic_slice()

    def rejects(self, mutate):
        value = deepcopy(self.value); mutate(value)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            validate_slice(value)

    def test_positive_control_no_full_distribution_is_valid(self):
        self.assertEqual(validate_slice(self.value)['samples'][0]['diagnostic']['full_vocab_target_positions'], [])

    def test_unknown_slice_field_rejected(self):
        self.rejects(lambda v: v.update(unknown=True))

    def test_empty_samples_rejected(self):
        self.rejects(lambda v: v.update(samples=[]))

    def test_duplicate_sample_rejected(self):
        self.rejects(lambda v: v['samples'].append(deepcopy(v['samples'][0])))

    def test_wrong_task_version_rejected(self):
        self.rejects(lambda v: v.update(task_version='2'))

    def test_wrong_scorer_rejected(self):
        self.rejects(lambda v: v['ancestry'].update(scorer_version='loosened'))

    def test_gold_suffix_generation_rejected(self):
        self.rejects(lambda v: v['samples'][0]['generation']['wire_payload'].update(prompt=v['samples'][0]['diagnostic']['input_ids']))

    def test_raw_wire_hash_tamper_rejected(self):
        self.rejects(lambda v: v['samples'][0]['generation']['proof'].update(raw_line='{}'))

    def test_wrong_raw_task_rejected(self):
        def mutate(v):
            proof=v['samples'][0]['generation']['proof']; record=json.loads(proof['raw_line']);record['task_id']='arc_challenge'
            proof['raw_line']=json.dumps(record);proof['line_sha256']=hashlib.sha256(proof['raw_line'].encode()).hexdigest()
        self.rejects(mutate)

    def test_reference_change_rejected(self):
        self.rejects(lambda v: v['samples'][0]['authored_sample']['gold'].update(status='complete'))

    def test_empty_targets_rejected(self):
        def mutate(v):
            row=v['samples'][0]['diagnostic']; row['targets']=[]; repin_row(row)
        self.rejects(mutate)

    def test_bool_target_index_rejected(self):
        def mutate(v):
            row=v['samples'][0]['diagnostic']; row['targets'][0]['token_position']=True; repin_row(row)
        self.rejects(mutate)

    def test_duplicate_target_rejected(self):
        def mutate(v):
            row=v['samples'][0]['diagnostic'];row['targets'].append(deepcopy(row['targets'][0]));repin_row(row)
        self.rejects(mutate)

    def test_wrong_prefix_rejected(self):
        def mutate(v):
            row=v['samples'][0]['diagnostic'];row['targets'][0]['prefix_hash']='c'*64;repin_row(row)
        self.rejects(mutate)

    def test_bool_input_id_rejected(self):
        def mutate(v):
            row=v['samples'][0]['diagnostic'];row['input_ids'][0]=True;row['sequence_identity']=digest(row['input_ids']);repin_row(row)
        self.rejects(mutate)

    def test_byte_offset_mismatch_rejected(self):
        def mutate(v):
            row=v['samples'][0]['diagnostic'];row['offsets_utf8_bytes'][2]=[0,1];repin_row(row)
        self.rejects(mutate)

    def test_tokenizer_mismatch_rejected(self):
        self.rejects(lambda v: v['model_metadata'].update(tokenizer_identity='d'*64))

    def test_load_hash_and_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for name in ('old','moved'):
                root=Path(tmp)/name;root.mkdir();write(root/'slice.json',self.value)
                write(root/'sources.json',{'schema_version':1,'kind':'suite_prepared_sources_v1','sources':[{'source_id':'synthetic','path':'slice.json','sha256':sha(root/'slice.json')}]})
                paths.append(load_sources(root/'sources.json'))
            self.assertEqual(paths[0]['data_identity'],paths[1]['data_identity'])
            self.assertNotEqual(paths[0]['receipts'][0]['physical_locator'],paths[1]['receipts'][0]['physical_locator'])

    def test_missing_or_tampered_slice_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'slice.json',self.value)
            manifest={'schema_version':1,'kind':'suite_prepared_sources_v1','sources':[{'source_id':'synthetic','path':'slice.json','sha256':'e'*64}]}
            with self.assertRaisesRegex(ValueError,'hash changed'):load_sources(manifest,base=root)

    def test_source_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unsafe artifact path'):
            load_sources({'schema_version':1,'kind':'suite_prepared_sources_v1','sources':[{'source_id':'synthetic','path':'../slice.json','sha256':'e'*64}]})

    def test_symlink_slice_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'actual.json',self.value);(root/'link.json').symlink_to(root/'actual.json')
            with self.assertRaises(ValueError):load_sources({'schema_version':1,'kind':'suite_prepared_sources_v1','sources':[{'source_id':'synthetic','path':'link.json','sha256':sha(root/'actual.json')}]},base=root)

    def test_no_implicit_source_discovery(self):
        with self.assertRaisesRegex(ValueError,'INVALID_EMPTY_PLAN'):
            load_sources({'schema_version':1,'kind':'suite_prepared_sources_v1','sources':[]})


    def test_hidden_generation_metadata_rejected(self):
        def mutate(v):
            s=v['samples'][0];s['generation']['wire_payload']['gold']='hidden'
            proof=s['generation']['proof'];record=json.loads(proof['raw_line']);record['request_body']=s['generation']['wire_payload']
            proof['raw_line']=json.dumps(record);proof['line_sha256']=hashlib.sha256(proof['raw_line'].encode()).hexdigest()
        self.rejects(mutate)

    def test_boolean_generation_limit_rejected(self):
        def mutate(v):
            s=v['samples'][0];s['generation']['wire_payload']['max_tokens']=True
            proof=s['generation']['proof'];record=json.loads(proof['raw_line']);record['request_body']=s['generation']['wire_payload']
            proof['raw_line']=json.dumps(record);proof['line_sha256']=hashlib.sha256(proof['raw_line'].encode()).hexdigest()
        self.rejects(mutate)


class SuiteArcInputsTests(unittest.TestCase):
    def test_all_choices_order_positive_control(self):
        self.assertEqual(len(validate_slice(synthetic_arc_slice())['samples'][0]['likelihood']['requests']),3)

    def test_gold_only_evaluation_rejected(self):
        value=synthetic_arc_slice(); value['samples'][0]['likelihood']['requests']=value['samples'][0]['likelihood']['requests'][1:2]
        with self.assertRaisesRegex(ValueError,'all-choice'): validate_slice(value)

    def test_reordered_choices_rejected(self):
        value=synthetic_arc_slice(); value['samples'][0]['likelihood']['requests'].reverse()
        with self.assertRaisesRegex(ValueError,'all-choice'): validate_slice(value)

    def test_wrong_gold_diagnostic_rejected(self):
        value=synthetic_arc_slice(); row=value['samples'][0]['diagnostic'];row['gold_choice_index']=0;repin_row(row)
        with self.assertRaisesRegex(ValueError,'gold diagnostic'): validate_slice(value)

    def test_changed_ancestry_rejected(self):
        value=synthetic_arc_slice();value['ancestry']['parent_plan_identity']='b'*64
        with self.assertRaisesRegex(ValueError,'lineage'):validate_slice(value)
