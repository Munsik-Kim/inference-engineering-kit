"""Closed E2 compiler controls. Synthetic metadata never certifies live support."""
from copy import deepcopy
import hashlib
import json
import tempfile
import unittest
import builtins
import shutil
import socket
import subprocess
from pathlib import Path
from unittest.mock import patch

from inference_kit.core.io import digest, read, sha, write
from inference_kit.probes.structure_inputs import OBSERVER_POLICY
from inference_kit.suites import suite_plan
from inference_kit.suites.suite_inputs import load_sources, validate_slice
from inference_kit.tasks.context_evidence import make_dataset, canonical_reference
from bench05e2.test_inputs import synthetic_slice, synthetic_arc_slice, repin_row


def token_spec():
    return {'schema_version':1,'selection_policy':'preserved_field_round_robin_uniform_v1',
      'selection_version':'1','reference_role_filter':['author_gold_json','dataset_gold_choice','author_answer_evidence'],
      'max_sequence_tokens':640,'max_target_positions_per_sample':32,'max_full_vocab_positions_per_sample':8,
      'observe_attention':False,'normalization':{'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'},
      'consistency_policy':{'id':'same_pass_saved_f32_v1','absolute_tolerance_nat':2e-5,
        'scope':'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'},
      'budget':{'task_gpu_wall_seconds':600,'cleanup_reserve_seconds':120,'max_gpu_processes':3,'automatic_retries':0,
        'probe_gpu_buffer_bytes':256*1024**2,'raw_bytes_total':256*1024**2,
        'minimum_cpu_available_bytes':10*1024**3,'minimum_disk_free_bytes':4*1024**3},'seed':1}


def model_metadata():
    spec=token_spec()
    return {'pair':{'model_id':'HuggingFaceTB/SmolLM2-1.7B-Instruct','revision':'31b70e2e869a7173562077fd711b654946d38674'},
      'model_config':{'model_type':'llama','num_hidden_layers':4,'num_attention_heads':2,
        'num_key_value_heads':2,'hidden_size':4,'attention_dropout':0},
      'tokenizer_identity':'1'*64,'token_id_space':'2'*64,'vocabulary_size':128,
      'arms':[{'arm_id':'bf16','artifact':{'artifact_identity':'0c668d5b8e51322d63898007d3d4a4ea488f21168208c3f3fc0d0f9643ac371b'},'source_format':'native','execution_mode':'INSTRUMENTED_PATH'},
        {'arm_id':'rtn','artifact':{'artifact_identity':'c8debb3188403ab9d332149969e6e0f9b7b689dc06e77e882fde07a9cb8da761'},'source_format':'compressed-tensors/pack-quantized','execution_mode':'DECODED_WEIGHT_DIAGNOSTIC'}],
      # Contract-only synthetic inputs reference the preserved receipt IDs;
      # this is not execution evidence for the synthetic fixture/model shape.
      'runtime_identities':{'serving':'cc88384a71f03de82c3f130c62cfa134b73aa301e8218f17d8c20cf25a54445b',
        'diagnostic':'33c7cd42d0c4fd2bec331491685445fb8abff9c9536d9a9f6c496f32d037c595'},
      'serving_policy':{'scope':'SYNTHETIC_CONTRACT_TEST','max_num_seqs':1},'token_spec':spec,
      'structure_spec':{'schema_version':1,'kind':'structure_probe_spec_v1','profile':'MAIN','max_sequence_tokens':640,
        'observe_attention':True,'backend':'native_transformers_sdpa_smol_mha_v1','numeric':{'dtype':'float32','endianness':'little'},
        'layers':'first_middle_last','heads':'first_middle','query_roles':['last_prompt','before_last_reference'],
        'observer_consistency_policy':deepcopy(OBSERVER_POLICY),'budget':{**spec['budget'],'max_gpu_processes':4}}}


def _proof(sample, task, version):
    body=sample['generation']['wire_payload']
    record={'request_body':body,'sample_id':sample['sample_id'],'task_id':task,'task_version':version,
      'tokenizer_identity':'1'*64,'reference_used_for_generation':False}
    line=json.dumps(record,ensure_ascii=False)+'\n'
    sample['generation']['proof'].update(raw_line=line,line_sha256=hashlib.sha256(line.encode('utf-8')).hexdigest())


def _expand(value, source_id):
    value=deepcopy(value);value['source_id']=source_id;original=value['samples'][0];value['samples']=[]
    for index in range(2):
        sample=deepcopy(original);sid=source_id+'-'+str(index);sample['sample_id']=sid
        authored=sample['authored_sample'];authored['sample_id']=sid
        if value['request_kind']=='choice_loglikelihood':authored['document']['id']=sid
        else:authored['sample_identity']=digest({k:v for k,v in authored.items() if k!='sample_identity'})
        row=sample['diagnostic'];row['sample_id']=sid
        if 'sample_identity' in authored:row['sample_identity']=authored['sample_identity']
        row['original_targets']=deepcopy(row['targets']);row['full_vocab_target_positions']=[row['targets'][0]['token_position']]
        row['original_full_vocab_target_positions']=row['full_vocab_target_positions'][:]
        row['original_field_token_coverage']={};row['field_token_coverage']={}
        repin_row(row)
        if value['request_kind']=='generate_until':
            sample['generation']['prepared']=deepcopy(row);_proof(sample,value['task_id'],value['task_version'])
        value['samples'].append(sample)
    value['ancestry']['included_sample_ids']=[s['sample_id'] for s in value['samples']]
    value['model_metadata']=model_metadata()
    return value


def retrieval_slice():
    value=synthetic_slice();authored=make_dataset(123)['rows'][0];text=canonical_reference(authored)['text']
    row=value['samples'][0]['diagnostic'];ids=[3]*(len(text)+1)
    row.update(task_id='context_evidence_retrieval',task_version='0.1',sample_id=authored['sample_id'],
      document_hash=authored['document_hash'],sample_identity=authored['sample_identity'],reference_role='author_answer_evidence',
      reference_source='task_author_gold_answer_evidence_v1',input_ids=ids,reference_text=text,reference_hash=digest(text),
      full_text='p'+text,full_text_hash=digest('p'+text),sequence_identity=digest(ids),attention_mask=[1]*len(ids),
      loss_mask=[0]+[1]*len(text),special_token_mask=[0]*len(ids),offsets_codepoints=[[i,i+1] for i in range(len(ids))],
      offsets_utf8_bytes=[[i,i+1] for i in range(len(ids))])
    repin_row(row)
    value.update(task_id='context_evidence_retrieval',task_version='0.1')
    value['ancestry']['scorer_version']='context_evidence_two_lines_v1';value['samples'][0]['authored_sample']=authored
    value['samples'][0]['sample_id']=authored['sample_id'];value['samples'][0]['generation']['wire_payload']['max_tokens']=64
    return _expand(value,'retrieval')


def sources_fixture(root):
    slices=[_expand(synthetic_slice(),'tech'),_expand(synthetic_arc_slice(),'arc'),retrieval_slice()]
    entries=[]
    for value in slices:
        validate_slice(value);path=root/(value['source_id']+'.json');write(path,value)
        entries.append({'source_id':value['source_id'],'path':path.name,'sha256':sha(path)})
    write(root/'sources.json',{'schema_version':1,'kind':'suite_prepared_sources_v1','sources':entries})
    return load_sources(root/'sources.json')


def suite_fixture(mode='serve', *, structure=False, full=True):
    return {'schema_version':1,'kind':'suite_spec_v1','suite_id':'closed-e2','version':'1','purpose':'diagnostic' if mode!='serve' else 'smoke',
      'tasks':[{'task_id':task,'task_version':version,'source_id':source,'sample_ids':[source+'-0',source+'-1'],
        'input_mode':'prepared_manifest','execution_kind':mode,'required':True}
        for task,version,source in [('tech_record_extract','0.1','tech'),('arc_challenge','1.0','arc'),('context_evidence_retrieval','0.1','retrieval')]],
      'models':[{'arm_id':'baseline','role':'baseline','artifact_ref':'bf16','profile':'preserved_smol_v1'},
        {'arm_id':'candidate','role':'candidate','artifact_ref':'rtn','profile':'preserved_smol_v1'}],
      'observations':{'token':{'enabled':mode!='serve','required':mode!='serve','full_vocabulary':full,'selection':'preserved_c2_main_v1'},
        'structure':{'enabled':structure,'required':structure,'selection':'preserved_d1_main_v1','provides_token_on':structure}},
      'metrics':['task_default_v1'] if mode=='serve' else ['common_tokens_c1_v1','selected_structure_d1_v1'] if structure else ['common_tokens_c1_v1'],
      'comparisons':[{'baseline':'baseline','candidate':'candidate','mode':'within_recipe'}],
      'budget':{'max_inference_posts':64,'max_output_tokens':20000,'max_numeric_bytes':256*1024**2,'max_probe_buffer_bytes':256*1024**2,
        'max_gpu_jobs':8,'gpu_wall_seconds':600,'cleanup_reserve_seconds':120},
      'permissions':{'gpu_execution':'NONE','network':'NONE','tokenizer':'NONE'},'seed':123,
      'outputs':{'index':True,'report':True,'historical_source_ids':[]}}


def local_fixture():
    return {'schema_version':1,'artifact_check':'not_requested','artifact_roots':{},'runtime_check':'not_run','authorization':'NONE'}


class SuiteCompilerTests(unittest.TestCase):
    def compile(self, root, suite=None):
        return suite_plan.compile_suite(suite or suite_fixture(),sources_fixture(root),local_fixture())

    def test_three_task_serve_count_prompt_only_and_all_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.compile(Path(tmp));counts=plan['semantic']['counts']
            self.assertEqual(counts['inference_POSTs'],20)  # four generation docs + six ARC choices per arm
            self.assertEqual(counts['natural_generation_token_upper'],2*(2*192+2*64))
            self.assertEqual(counts['audit_token_upper'],12)
            self.assertEqual(counts['GPU_jobs'],6)
            requests=[r for n in plan['semantic']['graph']['nodes'] for r in n['output_contract'].get('requests',[])]
            self.assertEqual(len({r['request_id'] for r in requests}),20)
            self.assertTrue(all(not r['reference_used_for_generation'] for r in requests if r['request_kind']=='generate_until'))
            self.assertEqual(plan['execution_count'],0);self.assertEqual(plan['acquisition_count'],0)
            self.assertEqual(plan['preflight']['authorization'],'NONE')

    def test_token_only_and_structure_share_on_without_duplicate_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=sources_fixture(root)
            token=suite_plan.compile_suite(suite_fixture('diagnostic'),sources,local_fixture())
            structure=suite_plan.compile_suite(suite_fixture('diagnostic',structure=True),sources,local_fixture())
            tc=token['semantic']['counts'];sc=structure['semantic']['counts']
            self.assertEqual(tc['GPU_jobs'],2);self.assertEqual(sc['GPU_jobs'],2)
            self.assertEqual(tc['forward_calls'],12);self.assertEqual(sc['forward_calls'],36)
            self.assertEqual(tc['main_target_observations'],12);self.assertEqual(sc['main_target_observations'],12)
            self.assertEqual(sc['all_pass_target_observations'],36)
            self.assertEqual(sc['attention_rows'],144);self.assertEqual(sc['output_projection_vectors'],72)
            self.assertEqual(sc['main_full_observations'],12)
            self.assertGreater(sc['numeric_bytes'],tc['numeric_bytes'])
            self.assertNotIn('collect_token_diagnostic',[n['operation_kind'] for n in structure['semantic']['graph']['nodes']])
            self.assertEqual(structure['status'],'VALIDATED_NOT_EXECUTED')

    def test_full_off_is_not_requested_while_token_plan_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.compile(Path(tmp),suite_fixture('diagnostic',full=False))
            self.assertEqual(plan['status'],'VALIDATED_NOT_EXECUTED')
            self.assertEqual(plan['semantic']['counts']['main_full_observations'],0)
            self.assertEqual(plan['semantic']['optional_features']['full_vocabulary'],'NOT_REQUESTED')
            self.assertEqual(plan['semantic']['optional_features']['structure'],'NOT_REQUESTED')

    def test_raw_text_and_unsupported_model_produce_blocked_not_ready(self):
        for failure in ('raw','model'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                suite=suite_fixture()
                if failure=='raw':
                    for task in suite['tasks']:task['input_mode']='raw_text'
                else:
                    for model in suite['models']:model['profile']='unknown_structure'
                plan=self.compile(Path(tmp),suite)
                self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')
                self.assertTrue(plan['semantic']['graph']['blocked_required_node_ids'])
                self.assertEqual(plan['execution_count'],0)

    def test_unknown_diagnostic_runtime_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp))
            sources['model_metadata']['runtime_identities']['diagnostic']='unverified-new-runtime'
            plan=suite_plan.compile_suite(suite_fixture('diagnostic',structure=True),sources,local_fixture())
            collectors=[n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='collect_structure_diagnostic']
            self.assertEqual(len(collectors),2)
            self.assertTrue(all(n['state']=='UNSUPPORTED_CAPABILITY' for n in collectors))
            self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')
            self.assertEqual(plan['acquisition_count'],0)

    def test_unknown_artifact_with_preserved_model_name_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp))
            sources['model_metadata']['arms'][1]['artifact']['artifact_identity']='f'*64
            plan=suite_plan.compile_suite(suite_fixture('diagnostic'),sources,local_fixture())
            candidate=[n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='collect_token_diagnostic' and n['arm_id']=='candidate']
            self.assertEqual(len(candidate),1)
            self.assertEqual(candidate[0]['state'],'UNSUPPORTED_CAPABILITY')
            self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')

    def test_unknown_serving_runtime_is_blocked_per_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp))
            sources['sources'][1]['model_metadata']['runtime_identities']['serving']='unverified-new-runtime'
            plan=suite_plan.compile_suite(suite_fixture(),sources,local_fixture())
            serving=[n for n in plan['semantic']['graph']['nodes'] if n['operation_kind'].startswith('serve_')]
            arc=[n for n in serving if n['task_id']=='arc_challenge']
            other=[n for n in serving if n['task_id']!='arc_challenge']
            self.assertEqual(len(arc),2)
            self.assertTrue(all(n['state']=='UNSUPPORTED_CAPABILITY' for n in arc))
            self.assertTrue(all(n['state']=='NOT_EXECUTED' for n in other))
            self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')
            self.assertEqual(plan['execution_count'],0)

    def test_empty_required_sets_bool_indices_and_nonfinite_budget_rejected(self):
        for field,bad in (('tasks',[]),('models',[]),('budget',{'max_inference_posts':float('inf')})):
            suite=suite_fixture();suite[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):suite_plan.validate_spec(suite)
        for mutate in (lambda v:v['tasks'][0].update(sample_ids=[]),
                       lambda v:[t.update(required=False) for t in v['tasks']],
                       lambda v:v['tasks'][0].update(sample_ids=['tech-0','tech-0']),
                       lambda v:v['budget'].update(max_gpu_jobs=True)):
            suite=suite_fixture();mutate(suite)
            with self.assertRaises(ValueError):suite_plan.validate_spec(suite)

    def test_plan_identity_repeat_semantic_change_and_no_time_or_locator_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=sources_fixture(root);spec=suite_fixture()
            first=suite_plan.compile_suite(spec,sources,local_fixture());second=suite_plan.compile_suite(spec,sources,local_fixture())
            self.assertEqual(first['plan_identity'],second['plan_identity'])
            relocated=deepcopy(sources);relocated['receipts'][0]['physical_locator']='/relocated/approved/slice.json'
            self.assertEqual(first['plan_identity'],suite_plan.compile_suite(spec,relocated,local_fixture())['plan_identity'])
            changed=deepcopy(spec);changed['seed']+=1
            self.assertNotEqual(first['plan_identity'],suite_plan.compile_suite(changed,sources,local_fixture())['plan_identity'])

    def test_frozen_mutation_or_compiler_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.compile(Path(tmp));suite_plan.check_frozen(plan)
            changed=deepcopy(plan);changed['semantic']['counts']['inference_POSTs']+=1
            with self.assertRaisesRegex(ValueError,'identity changed'):suite_plan.check_frozen(changed)
            with patch.object(suite_plan,'source_identity',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'source changed'):suite_plan.check_frozen(plan)

    def test_count_budget_and_permission_enforced_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp));spec=suite_fixture();spec['budget']['max_inference_posts']=19
            with self.assertRaisesRegex(ValueError,'BUDGET_EXCEEDED'):suite_plan.compile_suite(spec,sources,local_fixture())
            spec=suite_fixture();spec['permissions']['gpu_execution']='ALLOW'
            with self.assertRaisesRegex(ValueError,'permission'):suite_plan.compile_suite(spec,sources,local_fixture())

    def test_file_api_validate_describe_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources_fixture(root);write(root/'suite.json',suite_fixture());write(root/'local.json',local_fixture())
            plan=suite_plan.plan(root/'suite.json',root/'sources.json',root/'local.json',root/'plan')
            receipt=suite_plan.validate(root/'plan/FROZEN_PLAN.json',root/'sources.json',root/'validate')
            described=suite_plan.describe(root/'plan/FROZEN_PLAN.json',root/'describe')
            self.assertEqual(receipt['plan_identity'],plan['plan_identity']);self.assertEqual(described['plan_identity'],plan['plan_identity'])
            self.assertTrue((root/'describe/DESCRIPTION_KO.md').read_text(encoding='utf-8').startswith('# Suite'))
            with self.assertRaises(ValueError):suite_plan.plan(root/'suite.json',root/'sources.json',root/'local.json',root/'plan')

    def test_stale_sources_rejected_and_missing_subset_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=sources_fixture(root);spec=suite_fixture()
            spec['tasks'][0]['sample_ids']=['unknown']
            with self.assertRaisesRegex(ValueError,'Unknown requested sample'):suite_plan.compile_suite(spec,sources,local_fixture())
            write(root/'suite.json',suite_fixture());write(root/'local.json',local_fixture())
            suite_plan.plan(root/'suite.json',root/'sources.json',root/'local.json',root/'plan')
            (root/'tech.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash changed'):suite_plan.validate(root/'plan/FROZEN_PLAN.json',root/'sources.json',root/'recheck')

    def test_relocated_file_sources_revalidate_same_semantic_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'original';old.mkdir();sources_fixture(old)
            write(old/'suite.json',suite_fixture());write(old/'local.json',local_fixture())
            frozen=suite_plan.plan(old/'suite.json',old/'sources.json',old/'local.json',old/'plan')
            moved=root/'relocated';moved.mkdir()
            for name in ('tech.json','arc.json','retrieval.json','sources.json'):
                shutil.copyfile(old/name,moved/name)
            checked=suite_plan.validate(old/'plan/FROZEN_PLAN.json',moved/'sources.json',moved/'validated')
            self.assertEqual(checked['plan_identity'],frozen['plan_identity'])
            self.assertEqual(checked['physical_relocation'],'CONTENTS_VERIFIED_NO_NEW_ACQUISITION')
            self.assertEqual(checked['acquisition_count'],0)

    def test_provides_must_be_explicit_to_satisfy_token_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp));spec=suite_fixture('diagnostic',structure=True)
            spec['observations']['structure']['provides_token_on']=False
            plan=suite_plan.compile_suite(spec,sources,local_fixture())
            count=plan['semantic']['counts'];nodes=plan['semantic']['graph']['nodes']
            self.assertEqual(count['GPU_jobs'],4);self.assertEqual(count['forward_calls'],48)
            self.assertEqual(count['main_target_observations'],12)
            self.assertEqual(count['all_pass_target_observations'],48)
            self.assertEqual(sum(n['operation_kind']=='collect_token_diagnostic' for n in nodes),2)
            self.assertEqual(sum(n['operation_kind']=='collect_structure_diagnostic' for n in nodes),2)
            self.assertTrue(all(n['output_contract'].get('controls_in_main_token_aggregate') is False
                                for n in nodes if n['operation_kind']=='analyze_tokens'))

    def test_compile_import_launch_network_write_boundaries(self):
        forbidden={'torch','transformers','vllm','numpy','aiohttp','compressed_tensors','tokenizers'}
        old_import=builtins.__import__
        def guarded_import(name,*args,**kwargs):
            if name.split('.')[0] in forbidden:raise AssertionError('Unexpected SDK import: '+name)
            return old_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=sources_fixture(root);write(root/'ledger.json',{'used_seconds':4722.306879})
            before=sha(root/'ledger.json');local=local_fixture();local['ledger']=str(root/'ledger.json')
            with patch.object(builtins,'__import__',guarded_import),\
                 patch.object(subprocess,'Popen',side_effect=AssertionError('Unexpected child launch')),\
                 patch.object(socket,'socket',side_effect=AssertionError('Unexpected network')),\
                 patch.object(Path,'write_text',side_effect=AssertionError('Unexpected write')):
                plan=suite_plan.compile_suite(suite_fixture('diagnostic',structure=True),sources,local)
            self.assertEqual(plan['ledger_changes'],0);self.assertEqual(sha(root/'ledger.json'),before)
            self.assertEqual(plan['GPU_queries'],plan['model_loads'],plan['tokenizer_calls'])
            self.assertEqual(plan['network_requests'],0)

    def test_metric_contract_without_producer_cannot_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp));spec=suite_fixture();spec['metrics']=['common_tokens_c1_v1']
            try:
                plan=suite_plan.compile_suite(spec,sources,local_fixture())
            except ValueError:
                return
            self.assertNotEqual(plan['status'],'VALIDATED_NOT_EXECUTED')

    def test_paired_metrics_depend_on_both_arm_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources=sources_fixture(Path(tmp))
            diagnostic=suite_plan.compile_suite(suite_fixture('diagnostic',structure=True),sources,local_fixture())
            analyses=[n for n in diagnostic['semantic']['graph']['nodes']
                      if n['operation_kind'] in ('analyze_tokens','analyze_structure')]
            self.assertEqual(len(analyses),2)
            for node in analyses:
                self.assertIsNone(node['arm_id'])
                self.assertEqual(len(node['dependencies']),2)
                self.assertEqual({r['arm_id'] for r in node['inputs']},{'baseline','candidate'})
                self.assertEqual(node['output_contract']['comparison'],
                                 {'baseline':'baseline','candidate':'candidate','mode':'within_recipe'})
                self.assertEqual(node['output_contract']['pass_filter'],'ON')
            serve=suite_plan.compile_suite(suite_fixture(),sources,local_fixture())
            paired=[n for n in serve['semantic']['graph']['nodes']
                    if n['operation_kind']=='score_task' and n['output_contract']['scoring_mode']=='paired_saved_document_scores']
            self.assertEqual(len(paired),3)
            for node in paired:
                self.assertEqual(len(node['dependencies']),2)
                self.assertEqual({r['arm_id'] for r in node['inputs']},{'baseline','candidate'})
                self.assertEqual(len({r['task_id'] for r in node['inputs']}),1)
                self.assertEqual(node['output_contract']['missing_pair'],'UNKNOWN; no zero imputation')


if __name__=='__main__':unittest.main()
