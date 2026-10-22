"""Serving-only Qwen contract controls using explicitly synthetic metadata."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference_kit.models.capabilities import (
    QWEN_ARTIFACTS, QWEN_SERVING_PROFILE, QWEN_SERVING_RUNTIME,
    QWEN_TOKENIZER_IDENTITY,QWEN_TOKEN_ID_SPACE,qwen_serving_capability,qwen_input_capability,
)
from inference_kit.core.io import digest,write,sha
from inference_kit.suites.suite_plan import compile_suite, NATIVE_TOKEN_METRIC, validate_spec
from inference_kit.suites.suite_inputs import load_sources,validate_slice
from bench05e2.test_plan import sources_fixture, suite_fixture, local_fixture


def qwen_metadata(original):
    """The IDs name preserved receipts, never synthetic live measurements."""
    value=deepcopy(original)
    value['pair']={'model_id':'Qwen/Qwen3-4B-Instruct-2507',
                   'revision':'cdbee75f17c01a7cc42f958dc650907174af0554'}
    value['model_config']={'architectures':['Qwen3ForCausalLM'],'model_type':'qwen3',
        'num_hidden_layers':36,'num_attention_heads':32,'num_key_value_heads':8,
        'head_dim':128,'hidden_size':2560,'vocab_size':151936,'torch_dtype':'bfloat16'}
    value['vocabulary_size']=151669
    value['embedding_vocabulary_size']=151936
    value['tokenizer_identity']=QWEN_TOKENIZER_IDENTITY
    value['token_id_space']=QWEN_TOKEN_ID_SPACE
    quant={'quant_method':'compressed-tensors','format':'pack-quantized',
        'quantization_status':'compressed','ignore':['lm_head'],'kv_cache_scheme':None,
        'config_groups':{'group_0':{'format':'pack-quantized','targets':['Linear'],
            'input_activations':None,'output_activations':None,'weights':{
                'num_bits':4,'group_size':128,'symmetric':True,'dynamic':False,
                'strategy':'group','type':'int','actorder':None,'block_structure':None,
                'scale_dtype':None,'zp_dtype':None}}}}
    value['arms']=[{'arm_id':'bf16','source_format':'native',
        'execution_mode':'NATIVE_SERVING','quantization_method':'bf16',
        'artifact':{'artifact_identity':QWEN_ARTIFACTS['bf16']}},
      {'arm_id':'gptq_w4','source_format':'compressed-tensors/pack-quantized',
        'execution_mode':'NATIVE_SERVING','quantization_method':'gptq_w4',
        'quantization_config':quant,'recipe':{'method':'GPTQ','num_bits':4,
          'group_size':128,'symmetric':True,'actorder':None},
        'artifact':{'artifact_identity':QWEN_ARTIFACTS['gptq_w4']}}]
    value['runtime_identities']={'serving':QWEN_SERVING_RUNTIME}
    value.pop('token_spec');value.pop('structure_spec')
    return value


def qwen_fixture(root):
    sources=sources_fixture(root)
    metadata=qwen_metadata(sources['model_metadata'])
    entries=[];prepared_root=root/'qwen-contract'
    for source in sources['sources']:
        source['model_metadata']=deepcopy(metadata)
        for sample in source['samples']:
            row=sample['diagnostic']
            row['tokenization'].update(tokenizer_identity=metadata['tokenizer_identity'],token_id_space=metadata['token_id_space'])
            for target in row['targets']:
                target['prefix_hash']=digest({'token_ids':row['input_ids'][:target['token_position']],
                                              'token_id_space':metadata['token_id_space']})
            row['probe_input_identity']=digest({k:v for k,v in row.items() if k!='probe_input_identity'})
            if 'generation' in sample:
                generation=sample['generation'];generation['prepared']=deepcopy(row)
                record=json.loads(generation['proof']['raw_line']);record['tokenizer_identity']=metadata['tokenizer_identity']
                line=json.dumps(record,ensure_ascii=False)+'\n'
                generation['proof'].update(raw_line=line,line_sha256=hashlib.sha256(line.encode('utf-8')).hexdigest())
        validate_slice(source)
        path=prepared_root/(source['source_id']+'.json');write(path,source)
        entries.append({'source_id':source['source_id'],'path':path.name,'sha256':sha(path)})
    write(prepared_root/'sources.json',{'schema_version':1,'kind':'suite_prepared_sources_v1','sources':entries})
    sources=load_sources(prepared_root/'sources.json')
    spec=suite_fixture('serve',full=False)
    spec['suite_id']='synthetic-qwen-serving-contract'
    spec['models'][0].update(profile=QWEN_SERVING_PROFILE)
    spec['models'][1].update(artifact_ref='gptq_w4',profile=QWEN_SERVING_PROFILE)
    spec['metrics'].append(NATIVE_TOKEN_METRIC)
    return spec,sources


class QwenServingPlanTests(unittest.TestCase):
    def test_exact_config_receipt_and_packing_not_model_name_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,sources=qwen_fixture(Path(tmp));metadata=sources['model_metadata']
            for arm in metadata['arms']:
                self.assertTrue(qwen_serving_capability(metadata['pair'],metadata['model_config'],arm,QWEN_SERVING_RUNTIME))
            mutations=[lambda m:m['pair'].update(revision='unknown'),
                lambda m:m['model_config'].update(architectures=['LlamaForCausalLM']),
                lambda m:m['model_config'].update(num_key_value_heads=32),
                lambda m:m['arms'][1]['artifact'].update(artifact_identity='f'*64),
                lambda m:m['arms'][1].update(source_format='gptq_pack'),
                lambda m:m['arms'][1].update(quantization_method='rtn'),
                lambda m:m['arms'][1]['recipe'].update(method='RTN'),
                lambda m:m['arms'][1]['quantization_config'].update(format='unknown'),
                lambda m:m['arms'][1]['quantization_config']['config_groups']['group_0']['weights'].update(group_size=64),
                lambda m:m['arms'][1]['quantization_config']['config_groups']['group_0']['weights'].update(symmetric=1),
                lambda m:m['arms'][1]['quantization_config']['config_groups']['group_0']['weights'].update(zp_dtype='int8'),
                lambda m:m['arms'][1]['quantization_config']['config_groups']['group_0']['weights'].update(actorder='dynamic')]
            for mutate in mutations:
                changed=deepcopy(metadata);mutate(changed)
                with self.subTest(mutate=mutations.index(mutate)):
                    self.assertFalse(qwen_serving_capability(changed['pair'],changed['model_config'],changed['arms'][1],QWEN_SERVING_RUNTIME))
            self.assertFalse(qwen_serving_capability(metadata['pair'],metadata['model_config'],metadata['arms'][1],'new-runtime'))

    def test_serving_only_native_token_graph_has_no_diagnostic_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp));plan=compile_suite(spec,sources,local_fixture())
            self.assertEqual(plan['status'],'VALIDATED_NOT_EXECUTED')
            nodes=plan['semantic']['graph']['nodes'];counts=plan['semantic']['counts']
            self.assertEqual(counts['GPU_jobs'],6)
            self.assertEqual(counts['inference_POSTs'],20)
            self.assertEqual(counts['forward_calls'],0)
            self.assertEqual(counts['main_target_observations'],0)
            self.assertEqual(counts['main_full_observations'],0)
            self.assertFalse(any(n['operation_kind'].startswith('collect_') for n in nodes))
            native=[n for n in nodes if n['operation_kind']=='analyze_tokens']
            self.assertEqual(len(native),1)
            node=native[0]
            self.assertTrue(node['sample_ids'])
            self.assertGreater(node['counts']['paired_native_continuation_targets'],0)
            self.assertEqual(counts['native_continuation_target_observations'],2*node['counts']['paired_native_continuation_targets'])
            self.assertEqual(node['output_contract']['full_distribution'],'NOT_COLLECTED')
            self.assertEqual(node['output_contract']['pass_filter'],'NATIVE_CHOICE_LIKELIHOOD')
            self.assertEqual(node['output_contract']['diagnostic_worker'],'NOT_REQUESTED')
            self.assertEqual(len(node['dependencies']),2)
            serving={n['node_id'] for n in nodes if n['operation_kind']=='serve_choice_loglikelihood'}
            self.assertEqual(set(node['dependencies']),serving)
            self.assertEqual(plan['semantic']['optional_features']['token_diagnostic'],'NOT_REQUESTED')
            self.assertEqual(plan['semantic']['optional_features']['structure'],'NOT_REQUESTED')

    def test_qwen_diagnostic_remains_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp))
            for task in spec['tasks']:task['execution_kind']='both'
            spec['observations']['token'].update(enabled=True,required=True)
            spec['metrics'].append('common_tokens_c1_v1')
            plan=compile_suite(spec,sources,local_fixture())
            collectors=[n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='collect_token_diagnostic']
            self.assertEqual(len(collectors),2)
            self.assertTrue(all(n['state']=='UNSUPPORTED_CAPABILITY' for n in collectors))
            self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')

    def test_unsupported_packing_blocks_dependencies_without_live_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp))
            sources['model_metadata']['arms'][1]['quantization_config']['format']='unknown'
            plan=compile_suite(spec,sources,local_fixture())
            self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')
            native=[n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='analyze_tokens']
            self.assertEqual(native[0]['state'],'BLOCKED_DEPENDENCY')
            self.assertEqual(plan['execution_count'],0)
            self.assertEqual(plan['model_loads'],0)
            self.assertEqual(plan['tokenizer_calls'],0)

    def test_smol_tokenizer_or_mismatched_qwen_input_is_not_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp));metadata=sources['model_metadata']
            self.assertTrue(qwen_input_capability(metadata))
            for key,value in (('tokenizer_identity','1'*64),('token_id_space','2'*64),
                              ('vocabulary_size',151936),('embedding_vocabulary_size',151669)):
                changed=deepcopy(sources);changed['model_metadata'][key]=value
                with self.subTest(key=key):
                    plan=compile_suite(spec,changed,local_fixture())
                    self.assertEqual(plan['status'],'BLOCKED_REQUIRED_OPERATIONS')
                    self.assertTrue(all(n['state']=='UNSUPPORTED_CAPABILITY'
                        for n in plan['semantic']['graph']['nodes'] if n['operation_kind'].startswith('serve_')))

    def test_native_arc_metric_requires_serving_all_choices_not_gold_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp))
            changed=deepcopy(spec);changed['tasks']=[t for t in changed['tasks'] if t['task_id']!='arc_challenge']
            with self.assertRaisesRegex(ValueError,'requires complete ARC'):
                validate_spec(changed)
            # Existing TaskAdapter rejects a partial/reordered choice set before
            # either task accuracy or native target metrics are planned.
            for change in ('missing','order'):
                invalid=deepcopy(sources)
                arc=next(s for s in invalid['sources'] if s['task_id']=='arc_challenge')
                requests=arc['samples'][0]['authored_sample']['requests']
                if change=='missing':requests.pop()
                else:requests.reverse()
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Missing/duplicate/unknown choice'):
                    compile_suite(spec,invalid,local_fixture())

    def test_optional_native_metric_absent_preserves_task_only_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp));spec['metrics'].remove(NATIVE_TOKEN_METRIC)
            plan=compile_suite(spec,sources,local_fixture())
            self.assertFalse(any(n['operation_kind']=='analyze_tokens' for n in plan['semantic']['graph']['nodes']))
            self.assertNotIn('native_continuation_target_observations',plan['semantic']['counts'])
            self.assertEqual(plan['semantic']['counts']['GPU_jobs'],6)

    def test_native_plan_identity_changes_with_choice_tokens_not_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp))
            first=compile_suite(spec,sources,local_fixture());again=compile_suite(spec,sources,local_fixture())
            self.assertEqual(first['plan_identity'],again['plan_identity'])
            changed=deepcopy(sources)
            arc=next(s for s in changed['sources'] if s['task_id']=='arc_challenge')
            request=arc['samples'][0]['authored_sample']['requests'][0]
            request['input_ids'][-1]=(request['input_ids'][-1]+1)%151936
            # Existing boundary identity is exact: a mutated payload does not
            # become a semantically equivalent new plan.
            with self.assertRaises(ValueError):compile_suite(spec,changed,local_fixture())


if __name__=='__main__':unittest.main()
