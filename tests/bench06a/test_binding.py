"""Serving-only units are exact typed coverage, without invoking a runtime."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import sys

from inference_kit.core.io import digest, read, sha, write
from inference_kit.suites import prepared_live as binding
from inference_kit.suites.suite_execution import execution_units
from bench06a.test_results import serving_fixture


def layout(root, count=1):
    run=serving_fixture(root/'run',count);context=read(run/'suite-execution-context.json')
    compiled=context['suite_plan'];loaded=context['sources']
    samples=binding.selected_samples(compiled,loaded,execution_kind='serve')
    groups=binding.serving_groups(compiled,samples)
    context.update(serving_groups=groups,execution_unit_ids=[g['group_id'] for g in groups],
        expected_gpu_jobs=len(groups),runtime_plan={})
    return compiled,loaded,samples,context


class ServingOnlyBindingTests(unittest.TestCase):
    def test_toy_vocabulary_cannot_claim_pinned_qwen_preparation_identity(self):
        from inference_kit.probes.input_preparation_worker import prepare_rows
        from inference_kit.models.capabilities import QWEN_SERVING_PROFILE
        from bench05e2.test_plan import model_metadata
        from bench06a.test_plan import qwen_metadata
        from bench06a.test_preparation import qwen_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,spec,_,tokenizer=qwen_fixture();metadata=qwen_metadata(model_metadata())
            metadata['capability_profile']=QWEN_SERVING_PROFILE
            # Receipt hashes are metadata only; toy encoding is explicitly a
            # synthetic contract test, never a local tokenizer/runtime claim.
            metadata['tokenizer_files']={
                'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
                'tokenizer_config.json':'a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3',
                'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
                'merges.txt':'599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'}
            self.assertNotIn('token_spec',metadata);self.assertNotIn('structure_spec',metadata)
            with self.assertRaisesRegex(ValueError,'Tokenizer vocabulary pin changed'):
                prepare_rows(raw,spec,metadata,tokenizer)

    def test_qwen_shared_launcher_builds_real_native_and_gptq_contracts_without_start(self):
        from inference_kit.suites.bench05b1_run import server_launch
        from bench05e2.test_plan import model_metadata
        from bench06a.test_plan import qwen_metadata
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata=qwen_metadata(model_metadata());tokenizer=root/'tokenizer';tokenizer.mkdir()
            write(tokenizer/'tokenizer.json',{});write(tokenizer/'tokenizer_config.json',{})
            pins={p.name:sha(p) for p in tokenizer.iterdir()}
            plan={'binding_profile':'prepared_qwen_serve_v1','pair':metadata['pair'],'samples':{'tokenizer_files':pins},
                'suite':{'seed':1,'serving':{'context':2048,'max_num_seqs':1,'max_num_batched_tokens':2048,
                    'kv_tokens':4096,'chunked_prefill':False,'shutdown_timeout':1}},
                'paths':{'server_python':sys.executable},'port':18195,'gpu':{'index':0,'uuid':'GPU-SYNTHETIC-NO-QUERY'}}
            for original in metadata['arms']:
                arm=deepcopy(original);arm['arm_id']='qwen_'+arm['arm_id'];artifact=root/arm['arm_id'];artifact.mkdir()
                config=deepcopy(metadata['model_config'])
                if arm['source_format']!='native':config['quantization_config']=arm['quantization_config']
                write(artifact/'config.json',config)
                arm['artifact'].update(path=str(artifact),tokenizer_path=str(tokenizer),config_sha256=sha(artifact/'config.json'))
                out=root/('launch-'+arm['arm_id']);out.mkdir();result=server_launch(plan,arm,out)
                method={'name':'bf16' if arm['source_format']=='native' else 'gptq_w4','format':arm['source_format'],
                    'activation_dtype':'bfloat16','kv_dtype':'bfloat16'}
                self.assertEqual(read(out/'identity-receipt.json')['recipe_identity'],digest(method))
                self.assertEqual(result['artifact_identity'],arm['artifact']['artifact_identity'])
                self.assertIn('--host',result['argv']);self.assertIn('127.0.0.1',result['argv'])
                self.assertEqual(result['kv_bytes_per_token'],2*36*8*128*2)
                self.assertEqual(result['kv_bytes'],4096*result['kv_bytes_per_token'])
                self.assertEqual(result['environment']['HF_HUB_OFFLINE'],'1')

    def test_qwen_shared_launcher_rejects_wrong_architecture_and_quantization(self):
        from inference_kit.suites.bench05b1_run import server_launch
        from bench05e2.test_plan import model_metadata
        from bench06a.test_plan import qwen_metadata
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);metadata=qwen_metadata(model_metadata())
            plan={'binding_profile':'prepared_qwen_serve_v1','pair':metadata['pair']}
            for case in ('architecture','packing'):
                arm=deepcopy(metadata['arms'][1]);artifact=root/case;artifact.mkdir();config=deepcopy(metadata['model_config'])
                config['quantization_config']=deepcopy(arm['quantization_config'])
                if case=='architecture':config['model_type']='llama'
                else:config['quantization_config']['format']='unknown'
                write(artifact/'config.json',config);arm['artifact']['path']=str(artifact)
                # The capability checks both the pinned arm recipe and the
                # actual config; mismatched on-disk packing remains refused.
                if case=='packing':arm['quantization_config']=config['quantization_config']
                out=root/('launch-'+case);out.mkdir()
                with self.subTest(case=case),self.assertRaisesRegex(ValueError,'UNSUPPORTED_CAPABILITY'):
                    server_launch(plan,arm,out)
                self.assertFalse((out/'identity-receipt.json').exists())

    def test_selected_manifest_and_counts_vary_without_smol_probe_constants(self):
        results=[]
        for size in (1,3):
            with self.subTest(size=size),tempfile.TemporaryDirectory() as tmp:
                plan,loaded,samples,context=layout(Path(tmp),size)
                before=deepcopy(samples)
                metadata=binding.serving_inputs(samples,loaded['model_metadata'],'f'*64)
                self.assertEqual(metadata['rows'],[s['diagnostic'] for s in before])
                self.assertEqual(samples,before);self.assertNotIn('spec',metadata)
                self.assertEqual(metadata['kind'],'prepared_serving_inputs_v1')
                units=execution_units(plan,context);gpu=[u for u in units if u['operation_kind']=='serve_group']
                self.assertEqual(len(gpu),6)
                self.assertFalse(any(u['operation_kind']=='token_group' for u in units))
                results.append((len(samples),sum(u['counts']['inference_POSTs'] for u in gpu)))
        self.assertEqual(results,[(3,12),(9,36)])

    def test_each_typed_operation_is_provided_once_and_cpu_analysis_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,_,_,context=layout(Path(tmp));units=execution_units(plan,context)
            actual=[p for u in units for p in u['provides']]
            expected={n['node_id'] for n in plan['semantic']['graph']['nodes']}
            self.assertEqual(len(actual),len(set(actual)));self.assertEqual(set(actual),expected)
            self.assertEqual(len(units),11)
            self.assertTrue(all(u['required'] for u in units))
            for kind in ('score_task','analyze_tokens','index_add','report'):
                self.assertEqual(sum(u['operation_kind']==kind for u in units),1)
            native=next(u for u in units if u['operation_kind']=='analyze_tokens')
            self.assertEqual(set(native['dependencies']),set(context['execution_unit_ids']))

    def test_each_arm_task_has_distinct_paths_job_ids_and_exact_request_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,_,samples,context=layout(Path(tmp),3);groups=context['serving_groups']
            for field in ('group_id','job_id','raw_root'):
                self.assertEqual(len({g[field] for g in groups}),6)
            for arm in ('bf16','rtn'):
                own=[g for g in groups if g['arm_id']==arm]
                self.assertEqual(len(own),3);self.assertEqual(len({g['raw_root'] for g in own}),3)
            self.assertTrue(all('qualified_sample_id' in r for g in groups for r in g['requests']))
            self.assertEqual(sum(len(g['requests']) for g in groups),plan['semantic']['counts']['inference_POSTs'])
            self.assertEqual(len(binding.qualified_samples(samples)),9)

    def test_missing_duplicate_or_unknown_expected_unit_refuses_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,_,_,context=layout(Path(tmp))
            for case in ('missing','duplicate','unknown','count'):
                changed=deepcopy(context)
                if case=='missing':changed['execution_unit_ids'].pop()
                elif case=='duplicate':changed['serving_groups'].append(deepcopy(changed['serving_groups'][0]))
                elif case=='unknown':changed['execution_unit_ids'][0]='unplanned-serving-job'
                else:changed['expected_gpu_jobs']=4
                with self.subTest(case=case),self.assertRaisesRegex(ValueError,'execution unit coverage'):
                    execution_units(plan,changed)

    def test_serving_selection_requires_exact_frozen_source_and_disabled_observation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,loaded,_,_=layout(Path(tmp));wrong=deepcopy(loaded);wrong['data_identity']='f'*64
            with self.assertRaisesRegex(ValueError,'sources differ'):binding.selected_samples(plan,wrong,execution_kind='serve')
            with self.assertRaisesRegex(ValueError,'operations differ'):binding.selected_samples(plan,loaded)
            changed=deepcopy(plan);changed['semantic']['suite']['tasks'][0]['sample_ids']=[]
            with self.assertRaisesRegex(ValueError,'Frozen plan identity'):binding.selected_samples(changed,loaded,execution_kind='serve')

    def test_generation_and_all_choice_payloads_are_exact_before_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,_,samples,_=layout(Path(tmp))
            for kind in ('serve_generate_until','serve_choice_loglikelihood'):
                changed=deepcopy(plan);node=next(n for n in changed['semantic']['graph']['nodes'] if n['operation_kind']==kind)
                node['output_contract']['requests'][0]['wire_body']['prompt']=[99]
                with self.subTest(kind=kind),self.assertRaises(ValueError):binding.serving_groups(changed,samples)



class VersionedBindingContractTests(unittest.TestCase):
    def test_qwen_contract_names_the_actual_serving_only_capability(self):
        value=binding.contract(binding.SERVING_PROFILE)
        self.assertEqual(value['profile'],binding.SERVING_PROFILE)
        self.assertIn('Qwen3-4B',value['models'])
        self.assertNotIn('collect_token_diagnostic',value['operations'])
        self.assertEqual(value['diagnostic'],'NOT_REQUESTED')
        self.assertEqual(binding.contract()['profile'],binding.PROFILE)
        self.assertIn('collect_token_diagnostic',binding.contract()['operations'])
        with self.assertRaises(ValueError):binding.contract('unknown')
