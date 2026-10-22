"""New prepared authority and plan-only contracts; all token IDs are synthetic."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from inference_kit.core.io import digest, sha, write
from inference_kit.tasks.input_records import render_record, render_pinned_chat, arc_context
from inference_kit.tasks.tech_record import make_dataset
from inference_kit.tasks.context_evidence import make_dataset as retrieval_dataset
from inference_kit.suites.prepared_inputs import validate_source
from inference_kit.suites.suite_inputs import load_sources
from inference_kit.suites.suite_plan import compile_suite
from bench05e2.test_plan import model_metadata, suite_fixture, local_fixture


def _raw(task, index):
    if task == 'arc_challenge':
        sid='new-arc-'+str(index);doc={'id':sid,'question':'Which code?', 'choices':{'text':['one','two','six','ten'],'label':['A','B','C','D']},'answerKey':'B'}
        sample={'document':doc,'prompt_context':arc_context(doc['question']),'prompt_contract_id':'arc_original_completion_v1','original_index':index}
        return {'task_id':task,'task_version':'1.0','dataset_id':'new-arc','source_id':'new-arc-source','sample_id':sid,'split':'test','usage_scope':'REGRESSION_REUSE','policy_id':'arc_original_completion_v1','reference_source':'dataset_answer_key_choice','sample':sample}
    sample=deepcopy((make_dataset() if task=='tech_record_extract' else retrieval_dataset())['rows'][index])
    # A distinct source identity does not claim a new real benchmark.
    sample['sample_id']='new-'+task+'-'+str(index);sample['sample_identity']=digest({k:v for k,v in sample.items() if k!='sample_identity'})
    return {'task_id':task,'task_version':'0.1','dataset_id':'synthetic-contract','source_id':sample['source_id'],'sample_id':sample['sample_id'],'split':sample['split'],'usage_scope':'ENGINEERING_INPUT_ONLY','policy_id':'P00' if task=='tech_record_extract' else 'context_evidence_fixed_v1','reference_source':'task_author_gold_json' if task=='tech_record_extract' else 'task_author_gold_answer_evidence_v1','sample':sample}


def source_fixture(task='tech_record_extract', count=1):
    metadata=model_metadata();metadata['tokenizer_files']={'tokenizer.json':'a'*64};metadata['tokenizer_identity']=digest(metadata['tokenizer_files'])
    metadata['token_spec']['max_target_positions_per_sample']=16;metadata['token_spec']['max_full_vocab_positions_per_sample']=4
    samples=[]
    for index in range(count):
        raw=_raw(task,index);rendered=render_record(raw);authored=rendered['authored_sample'];content=rendered['content']
        prompt=render_pinned_chat(content) if task!='arc_challenge' else content.rstrip();reference=rendered['reference_text']
        if task=='arc_challenge':
            shift=len(content)-len(prompt);reference=content[len(prompt):]+reference
            rendered['value_spans']={k:[a+shift,b+shift] for k,(a,b) in rendered['value_spans'].items()}
        # This deliberately tiny contract fixture is not tokenizer execution.
        ids=[3]+[4]*len(reference);offsets=[[0,len(prompt)]]+[[len(prompt)+i,len(prompt)+i+1] for i in range(len(reference))]
        targets=[{'token_position':i,'logit_position':i-1,'target_token_id':ids[i],'prefix_hash':digest({'token_ids':ids[:i],'token_id_space':metadata['token_id_space']})} for i in range(1,min(len(ids),17))]
        coverage={k:{'overlapping_positions':[i+1 for i in range(a,b)],'selected_positions':[i+1 for i in range(a,b) if i+1<=16]} for k,(a,b) in rendered['value_spans'].items()}
        row={'task_id':task,'task_version':raw['task_version'],'sample_id':raw['sample_id'],'reference_role':rendered['reference_role'],'reference_source':raw['reference_source'],
             'input_ids':ids,'generation_input_ids':[3],'reference_separate_ids':[4]*len(reference),'prompt':prompt,'reference_text':reference,'full_text':prompt+reference,
             'prompt_hash':digest(prompt),'reference_hash':digest(reference),'full_text_hash':digest(prompt+reference),'sequence_identity':digest(ids),'prompt_token_boundary':1,
             'attention_mask':[1]*len(ids),'special_token_ids':[0],'special_token_mask':[0]*len(ids),'loss_mask':[0]+[1]*len(reference),
             'offsets_codepoints':offsets,'offsets_utf8_bytes':[[len((prompt+reference)[:a].encode()),len((prompt+reference)[:b].encode())] for a,b in offsets],
             'tokenization':{'tokenizer_identity':metadata['tokenizer_identity'],'token_id_space':metadata['token_id_space']},'targets':targets,'full_vocab_target_positions':[t['token_position'] for t in targets[:4]],
             'value_spans':rendered['value_spans'],'field_token_coverage':coverage,'task_prompt_hash':digest(content)}
        sample={'sample_id':raw['sample_id'],'raw_record':raw,'authored_sample':authored,'diagnostic':row}
        if task=='arc_challenge':
            requests=[{'choice_index':i,'context':content,'continuation':' '+choice,'context_ids':[3],'continuation_ids':[4]*(len(choice)+1+shift),'input_ids':[3]+[4]*(len(choice)+1+shift)} for i,choice in enumerate(authored['choices'])]
            authored['requests']=requests;row['gold_choice_index']=authored['gold']
            sample['likelihood']={'requests':deepcopy(requests),'choice_labels':authored['choice_labels'][:],'gold_index':authored['gold']}
        else:
            row['document_hash']=authored['document_hash'];row['sample_identity']=authored['sample_identity']
            sample['generation']={'prepared':{'sample_id':raw['sample_id'],'prompt':prompt,'prompt_hash':digest(prompt),'generation_input_ids':[3]},
                'wire_payload':{'prompt':[3],'max_tokens':192 if task=='tech_record_extract' else 64,'temperature':0,'top_p':1,'n':1,'stream':True,'echo':False,'ignore_eos':False,'add_special_tokens':False},
                'preparation_proof':{'raw_record_hash':digest(raw),'task_prompt_hash':digest(content),'prompt_hash':digest(prompt),'tokenizer_identity':metadata['tokenizer_identity'],
                    'reference_used_for_generation':False,'independent_prompt_encoding':True,'chat_template_applied':1,'rendered_task_prompt':content}}
        row['probe_input_identity']=digest(row);samples.append(sample)
    return {'schema_version':1,'kind':'prepared_task_source_v1','source_id':'prepared-'+task,'task_id':task,'task_version':'1.0' if task=='arc_challenge' else '0.1','request_kind':'choice_loglikelihood' if task=='arc_challenge' else 'generate_until',
        'validation_scope':'PREPARED_CONTRACT_VALIDATED_ONLY','model_metadata':metadata,'samples':samples,
        'preparation':{'schema_version':1,'policy_id':'existing_tasks_local_tokenizer_v1','raw_manifest_identity':'b'*64,'preparation_spec_identity':'c'*64,'tokenizer_identity':metadata['tokenizer_identity'],
            'token_id_space':metadata['token_id_space'],'scorer_version':'arc_choice_sum_character_norm_v1' if task=='arc_challenge' else 'tech_record_strict_json_v1' if task=='tech_record_extract' else 'context_evidence_two_lines_v1',
            'limits':{'max_prompt_tokens':512,'max_reference_tokens':128,'max_joint_tokens':640,'context_tokens':2048},'selection':{'policy_id':'field_round_robin_uniform_v1','max_targets':16,'max_full':4},'source_identity':'d'*64}}


def loaded_fixture(root, count=1):
    sources=[source_fixture(task,count) for task in ('tech_record_extract','arc_challenge','context_evidence_retrieval')];entries=[]
    for source in sources:
        validate_source(source);path=root/(source['source_id']+'.json');write(path,source);entries.append({'source_id':source['source_id'],'path':path.name,'sha256':sha(path)})
    write(root/'SOURCES.json',{'schema_version':1,'kind':'suite_prepared_sources_v1','sources':entries})
    return load_sources(root/'SOURCES.json')


def prepared_suite(sources, mode='serve', *, full=True, structure=False):
    spec=suite_fixture(mode,full=full,structure=structure)
    for task,source in zip(spec['tasks'],sources['sources']):
        task.update(task_id=source['task_id'],task_version=source['task_version'],source_id=source['source_id'],sample_ids=[s['sample_id'] for s in source['samples']])
    spec['observations']['token']['selection']='prepared_reference_positions_v1'
    return spec


class NewPreparedSuiteTests(unittest.TestCase):
    def test_three_tasks_new_authority_without_historical_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=loaded_fixture(Path(tmp));self.assertEqual(len(sources['samples']),3)
            self.assertEqual(sources['validation_scope'],'PREPARED_CONTRACT_VALIDATED_ONLY')
            self.assertTrue(all('ancestry' not in s for s in sources['sources']))
            plan=compile_suite(prepared_suite(sources),sources,local_fixture())
            self.assertEqual(plan['status'],'VALIDATED_NOT_EXECUTED');self.assertEqual(plan['semantic']['counts']['inference_POSTs'],12)
            self.assertEqual(plan['semantic']['input_execution_scope']['live_binding'],'LIVE_BINDING_NOT_VALIDATED')

    def test_nine_documents_counts_are_data_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=loaded_fixture(Path(tmp),3);plan=compile_suite(prepared_suite(sources),sources,local_fixture())
            self.assertEqual(plan['semantic']['counts']['documents'],9);self.assertEqual(plan['semantic']['counts']['inference_POSTs'],36)
            token=compile_suite(prepared_suite(sources,'diagnostic'),sources,local_fixture())
            self.assertEqual(token['status'],'VALIDATED_NOT_EXECUTED');self.assertEqual(token['semantic']['counts']['forward_calls'],18)
            self.assertNotEqual(token['semantic']['counts']['main_target_observations'],220)
            nodes=[n for n in token['semantic']['graph']['nodes'] if n['operation_kind']=='collect_token_diagnostic']
            self.assertEqual(nodes[0]['output_contract']['resolved_probe_inputs']['kind'],'prepared_reference_probe_inputs_v1')

    def test_full_off_remains_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=loaded_fixture(Path(tmp));plan=compile_suite(prepared_suite(sources,'diagnostic',full=False),sources,local_fixture())
            self.assertEqual(plan['status'],'VALIDATED_NOT_EXECUTED');self.assertEqual(plan['semantic']['counts']['main_full_observations'],0)
            self.assertEqual(plan['semantic']['optional_features']['full_vocabulary'],'NOT_REQUESTED')

    def test_new_structure_and_legacy_selection_blocked_without_relaxation(self):
        for mode in ('structure','legacy'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                sources=loaded_fixture(Path(tmp));spec=prepared_suite(sources,'diagnostic',structure=mode=='structure')
                if mode=='legacy':spec['observations']['token']['selection']='preserved_c2_main_v1'
                plan=compile_suite(spec,sources,local_fixture());self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')
                self.assertEqual(plan['execution_count'],0)

    def test_new_manifest_hash_and_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'한글';moved=Path(tmp)/'moved';old.mkdir();moved.mkdir()
            a=loaded_fixture(old);b=loaded_fixture(moved)
            self.assertEqual(a['data_identity'],b['data_identity'])
            value=read_json=deepcopy(a['sources'][0]);value['samples'][0]['generation']['wire_payload']['prompt'].append(7)
            path=old/(value['source_id']+'.json');path.write_text(__import__('json').dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash changed'):load_sources(old/'SOURCES.json')

    def test_generation_gold_suffix_rejected(self):
        source=source_fixture();s=source['samples'][0];s['generation']['wire_payload']['prompt']=s['diagnostic']['input_ids']
        with self.assertRaisesRegex(ValueError,'prompt-only|leakage'):validate_source(source)

    def test_changed_rendered_prompt_or_gold_rejected(self):
        source=source_fixture();source['samples'][0]['generation']['preparation_proof']['rendered_task_prompt']+='gold'
        with self.assertRaisesRegex(ValueError,'proof'):validate_source(source)
        source=source_fixture();source['samples'][0]['authored_sample']['gold']['status']='pending'
        with self.assertRaises(ValueError):validate_source(source)

    def test_wrong_tokenizer_pin_and_empty_targets_rejected(self):
        source=source_fixture();source['model_metadata']['tokenizer_files']['tokenizer.json']='f'*64
        with self.assertRaisesRegex(ValueError,'file identity'):validate_source(source)
        source=source_fixture();row=source['samples'][0]['diagnostic'];row['targets']=[];row['probe_input_identity']=digest({k:v for k,v in row.items() if k!='probe_input_identity'})
        with self.assertRaisesRegex(ValueError,'no diagnostic targets'):validate_source(source)

    def test_arc_all_choices_and_external_gold_preserved(self):
        source=source_fixture('arc_challenge');s=source['samples'][0];s['likelihood']['requests']=s['likelihood']['requests'][:1]
        with self.assertRaisesRegex(ValueError,'all-choice'):validate_source(source)

    def test_bool_selection_limit_and_target_shift_rejected(self):
        source=source_fixture();source['preparation']['selection']['max_targets']=True
        with self.assertRaisesRegex(ValueError,'selection'):validate_source(source)
        source=source_fixture();r=source['samples'][0]['diagnostic'];r['targets'][0]['logit_position']=True;r['probe_input_identity']=digest({k:v for k,v in r.items() if k!='probe_input_identity'})
        with self.assertRaisesRegex(ValueError,'target shift'):validate_source(source)
        source=source_fixture();source['preparation']['limits']['context_tokens']=4096
        with self.assertRaisesRegex(ValueError,'length bounds'):validate_source(source)

    def test_output_locator_not_semantic_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir();left=loaded_fixture(a);right=loaded_fixture(b)
            self.assertEqual(compile_suite(prepared_suite(left),left,local_fixture())['plan_identity'],compile_suite(prepared_suite(right),right,local_fixture())['plan_identity'])
