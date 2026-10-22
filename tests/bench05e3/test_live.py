"""CPU controls for mixed wire contracts and the existing owned launch boundary."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from inference_kit.core.io import read, write, digest, sha
# Keep the module present before patch.dict(sys.modules) snapshots on Python 3.10.
from inference_kit.adapters import task_executor as _stable_task_executor
from inference_kit.suites.bench05b1_run import normalize_collector_contract, collector_metadata, validate_collector_records
from inference_kit.suites import suite_live


class MixedCollectorTests(unittest.TestCase):
    def setUp(self):self.contract=suite_live.mixed_contract()
    def test_explicit_typed_mixed_contract_normalizes(self):
        value=normalize_collector_contract({},object(),self.contract)
        self.assertEqual(value,self.contract);self.assertEqual(collector_metadata(value)['stream'],'per_request')
    def test_legacy_default_natural192_unchanged(self):
        self.assertEqual(normalize_collector_contract({})['max_new_tokens'],192)
        self.assertEqual(normalize_collector_contract({},object())['request_kind'],'choice_loglikelihood')
    def test_three_wire_profiles_validate(self):
        records=[{'request_kind':v['request_kind'],'request_body':{'max_tokens':v['max_new_tokens'],'stream':v['stream'],'ignore_eos':v['ignore_eos']}} for v in self.contract['contracts']]
        validate_collector_records(records,self.contract)
    def test_mixed_requires_typed_driver(self):
        with self.assertRaisesRegex(ValueError,'explicit typed driver'):normalize_collector_contract({},None,self.contract)
    def test_mixed_duplicate_profiles_rejected(self):
        self.contract['contracts'].append(deepcopy(self.contract['contracts'][0]))
        with self.assertRaises(ValueError):normalize_collector_contract({},object(),self.contract)
    def test_nested_mixed_rejected(self):
        self.contract['contracts'].append(deepcopy(self.contract))
        with self.assertRaisesRegex(ValueError,'nonnested'):normalize_collector_contract({},object(),self.contract)
    def test_mixed_scalar_flag_rejected(self):
        self.contract['ignore_eos']=True
        with self.assertRaisesRegex(ValueError,'per_request'):normalize_collector_contract({},object(),self.contract)
    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(ValueError,'undeclared'):
            validate_collector_records([{'request_kind':'generate_until','request_body':{'max_tokens':123,'stream':True,'ignore_eos':False}}],self.contract)
    def test_wrong_kind_and_wire_flags_rejected(self):
        with self.assertRaises(ValueError):validate_collector_records([{'request_kind':'choice_loglikelihood','request_body':{'max_tokens':1,'stream':True,'ignore_eos':True}}],self.contract)
    def test_empty_mixed_response_set_never_verified(self):
        with self.assertRaisesRegex(ValueError,'no request records'):validate_collector_records([],self.contract)
    def test_boolean_max_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError,'token bound'):
            validate_collector_records([{'request_kind':'choice_loglikelihood','request_body':{'max_tokens':True,'stream':False,'ignore_eos':True}}],self.contract)


class LivePreflightTests(unittest.TestCase):
    def test_low_resource_prevents_identity_gpu_and_ledger(self):
        context={'runtime_plan':{'suite':{'budget':{}}}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite_live,'resources',return_value={'ready':False}),patch.object(suite_live,'verify_live_context') as verify,patch('inference_kit.suites.bench05a_run.check_gpu') as gpu:
            result=suite_live.preflight(context,tmp)
            self.assertEqual(result['status'],'BLOCKED_CPU_OR_DISK_RESOURCE');self.assertEqual(result['ledger_changes'],0)
            verify.assert_not_called();gpu.assert_not_called();self.assertEqual(list(Path(tmp).iterdir()),[])
    def test_identity_failure_prevents_gpu_and_ledger(self):
        context={'runtime_plan':{'suite':{'budget':{}}}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite_live,'resources',return_value={'ready':True}),patch.object(suite_live,'verify_live_context',side_effect=ValueError('changed')),patch('inference_kit.suites.bench05a_run.check_gpu') as gpu:
            result=suite_live.preflight(context,tmp)
            self.assertEqual(result['status'],'BLOCKED_LOCAL_IDENTITY');gpu.assert_not_called()
    def test_cpu_only_preflight_never_queries_gpu(self):
        context={'runtime_plan':{'suite':{'budget':{}}}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite_live,'resources',return_value={'ready':True}),patch.object(suite_live,'verify_live_context',return_value=True),patch('inference_kit.suites.bench05a_run.check_gpu') as gpu:
            result=suite_live.preflight(context,tmp,query_gpu=False)
            self.assertEqual(result['gpu_queries'],0);gpu.assert_not_called()
    def test_shared_gpu_status_preserved(self):
        context={'runtime_plan':{'suite':{'budget':{}}}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite_live,'resources',return_value={'ready':True}),patch.object(suite_live,'verify_live_context',return_value=True),patch('inference_kit.suites.bench05a_run.check_gpu',return_value={'status':'BLOCKED_SHARED_GPU'}):
            self.assertEqual(suite_live.preflight(context,tmp)['status'],'BLOCKED_SHARED_GPU')
    def test_new_ledger_retains_predecessor_identity_and_usage(self):
        context={'runtime_plan':{'gpu_allowance_seconds':600},'campaign':{'used_seconds':4722.3,'path':'original-ledger','sha256':'a'*64}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new.json';record=suite_live.initialize_ledger(context,path)
            self.assertEqual(record['used_seconds'],4722.3);self.assertEqual(record['cap_seconds'],5322.3)
            self.assertEqual(record['inherited_from']['sha256'],'a'*64);self.assertEqual(record['jobs'],[])
            with self.assertRaises(ValueError):suite_live.initialize_ledger(context,path)


class _Session:
    async def __aenter__(self):return self
    async def __aexit__(self,*args):return False


class MixedDriverTests(unittest.IsolatedAsyncioTestCase):
    def plan(self):
        body={'model':'serving-lab','prompt':[1,2,3],'max_tokens':64,'stream':True,'ignore_eos':False}
        sample={'task_id':'tech_record_extract','task_version':'0.1','sample_id':'sample','authored_sample':{'sample_id':'sample','task_id':'tech_record_extract','split':'SYNTHETIC_CONTRACT_TEST','document_hash':'d'*64,'sample_identity':'s'*64},
                'generation':{'wire_payload':body,'prepared':{'prompt_hash':'p'*64,'task_prompt_hash':'t'*64}},
                'diagnostic':{'source_scorer_identity':'c'*64,'reference_source':'task_author_gold_json'}}
        return {'e3_sources':{'samples':[sample]},'e3_group':{'requests':[{'sample_id':'sample','task_id':'tech_record_extract','wire_body':body}]},'parent_suite_plan_identity':'plan','e3_run_name':'test-run',
                'samples':{'tokenizer_identity':'tok','token_id_space':'vocab'},'serving_runtime_identity':'runtime','clock':{'deadline_monotonic':time.monotonic()+1000},'suite':{'budget':{'cleanup_reserve_seconds':120}}}
    async def test_raw_and_submission_journal_saved_before_later_scoring(self):
        plan=self.plan();plan['e3_group']['logical_arm_id']='baseline';arm={'arm_id':'bf16','artifact':{'artifact_identity':'a'*64}}
        async def fake_execute(request,**kwargs):
            kwargs['event_sink']({'event':'http_submit','request_id':request.request_id})
            return {'request_id':request.request_id,'request_body':plan['e3_group']['requests'][0]['wire_body'],'transport_complete':True,'output_text':'wrong answer','raw_sse_frames':['data: original'],'termination':'NATURAL_STOP'}
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'aiohttp':SimpleNamespace(ClientSession=_Session)}),patch('inference_kit.adapters.task_executor.execute',side_effect=fake_execute):
            records=await suite_live.mixed_request_driver(plan,arm,Path(tmp),'http://mock',SimpleNamespace(poll=lambda:None),asyncio.Event(),lambda _:None)
            self.assertEqual(len(records),1);self.assertEqual(records[0]['output_text'],'wrong answer')
            self.assertEqual(json.loads((Path(tmp)/'task-responses.jsonl').read_text(encoding='utf-8'))['request_body']['prompt'],[1,2,3])
            self.assertEqual(len((Path(tmp)/'request-journal.jsonl').read_text(encoding='utf-8').splitlines()),1)
            self.assertFalse(records[0]['reference_used_for_generation'])
            self.assertEqual(records[0]['arm_id'],'baseline');self.assertEqual(records[0]['source_arm_id'],'bf16')
            self.assertEqual(json.loads((Path(tmp)/'request-journal.jsonl').read_text(encoding='utf-8'))['arm_id'],'baseline')
    async def test_transport_failure_retains_raw_and_has_no_retry(self):
        plan=self.plan();arm={'arm_id':'bf16','artifact':{'artifact_identity':'a'*64}}
        async def fake_execute(request,**kwargs):
            kwargs['event_sink']({'event':'http_submit','request_id':request.request_id})
            return {'request_id':request.request_id,'request_body':plan['e3_group']['requests'][0]['wire_body'],'transport_complete':False,'output_text':'','termination':'HTTP_ERROR'}
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'aiohttp':SimpleNamespace(ClientSession=_Session)}),patch('inference_kit.adapters.task_executor.execute',side_effect=fake_execute) as execute:
            with self.assertRaisesRegex(ValueError,'transport failed'):
                await suite_live.mixed_request_driver(plan,arm,Path(tmp),'http://mock',SimpleNamespace(poll=lambda:None),asyncio.Event(),lambda _:None)
            self.assertEqual(execute.call_count,1);self.assertTrue((Path(tmp)/'task-responses.jsonl').stat().st_size>0)
    async def test_frozen_wire_mismatch_is_rejected(self):
        plan=self.plan();arm={'arm_id':'bf16','artifact':{'artifact_identity':'a'*64}}
        async def fake_execute(request,**kwargs):return {'request_body':{'prompt':[99]},'transport_complete':True}
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'aiohttp':SimpleNamespace(ClientSession=_Session)}),patch('inference_kit.adapters.task_executor.execute',side_effect=fake_execute):
            with self.assertRaisesRegex(ValueError,'frozen per-request'):
                await suite_live.mixed_request_driver(plan,arm,Path(tmp),'http://mock',SimpleNamespace(poll=lambda:None),asyncio.Event(),lambda _:None)
            self.assertFalse(json.loads((Path(tmp)/'task-responses.jsonl').read_text(encoding='utf-8'))['frozen_wire_contract_match'])


class SavedTokenCompletionTests(unittest.TestCase):
    def fixture(self,root):
        import math
        from inference_kit.probes.contracts import Observation
        from inference_kit.probes.blobs import save
        from inference_kit.probes.token_worker import CONSISTENCY_POLICY,NORMALIZATION
        row={'sample_id':'coordinate','input_ids':[3,1],'sequence_identity':digest([3,1]),
             'targets':[{'token_position':1,'logit_position':0,'target_token_id':1}],'full_vocab_target_positions':[1]}
        arm={'arm_id':'bf16','artifact':{'artifact_identity':'a'*64,'recipe_identity':'b'*64},'execution_mode':'INSTRUMENTED_PATH'}
        plan={'samples':{'rows':[row],'vocabulary_size':4,'tokenizer_identity':'tok','token_id_space':'space'},
              'runtime_identity':'runtime','probe_spec':{'normalization':NORMALIZATION}}
        execution={'status':'MEASURED','phase':'main','arm_id':'bf16','artifact_identity':'a'*64,'recipe_identity':'b'*64,
                   'tokenizer_identity':'tok','runtime_identity':'runtime','execution_mode':'INSTRUMENTED_PATH','execution_id':'exe'}
        write(root/'execution.json',execution)
        coords={'token_position':1,'logit_position':0,'target_token_id':1,'sample_id':'coordinate','sequence_identity':row['sequence_identity'],
                'prefix_hash':digest({'token_ids':[3],'token_id_space':'space'}),'tokenizer_identity':'tok','token_id_space':'space','pass_id':'ON'}
        observations=[]
        for identifier,kind,values in [('scalar','target_logprob',[-math.log(4)]),('full','full_logits',[0,0,0,0])]:
            observations.append(Observation(identifier,kind,'exe','coordinate','live_model_diagnostic','INSTRUMENTED_PATH','MEASURED',coords,
                save(root,identifier,values,[len(values)]),{'normalization':NORMALIZATION,'consistency_policy':CONSISTENCY_POLICY}).validate())
        with (root/'observations.jsonl').open('x',encoding='utf-8') as stream:
            for row in observations:stream.write(json.dumps(row)+'\n')
        write(root/'result.json',{'status':'MEASURED','samples':1,'main_target_positions':1,'full_vocab_positions':1,
            'main_attention_rows':0,'attention_hooks_installed':0,'no_attention_observations_created':True,'no_full_T_by_V_logits_created':True,
            'same_pass_consistency':{'checked':1,'violations':0}})
        return plan,arm,observations
    def test_nonempty_actual_numeric_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,arm,_=self.fixture(root)
            self.assertEqual(suite_live.validate_token_attempt(root,plan,arm),{'inference_POSTs':0,'main_target_observations':1,'main_full_observations':1,'forward_calls':1})
    def test_missing_saved_coordinate_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,arm,obs=self.fixture(root)
            (root/'observations.jsonl').write_text(json.dumps(obs[0])+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'incomplete'):suite_live.validate_token_attempt(root,plan,arm)
    def test_duplicate_saved_coordinate_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,arm,obs=self.fixture(root)
            with (root/'observations.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(obs[0])+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):suite_live.validate_token_attempt(root,plan,arm)
    def test_corrupted_numeric_blob_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,arm,obs=self.fixture(root)
            numeric=root/obs[1]['numeric']['path'];numeric.write_bytes(b'changed')
            with self.assertRaises(ValueError):suite_live.validate_token_attempt(root,plan,arm)
    def test_boolean_coordinate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,arm,obs=self.fixture(root);obs[0]['coordinates']['token_position']=True
            with (root/'observations.jsonl').open('w',encoding='utf-8') as stream:
                for row in obs:stream.write(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):suite_live.validate_token_attempt(root,plan,arm)
