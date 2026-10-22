import asyncio
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import read,write,sha
from inference_kit.suites.bench05b1_run import (
    collector_metadata,normalize_collector_contract,server_launch,
    serving_job,validate_collector_records,
)


B4={'request_kind':'generate_until','stream':True,'max_new_tokens':64,
    'ignore_eos':False,'audit_generation_tokens':0}


class CollectorContractTests(unittest.TestCase):
    def test_legacy_b1_default_natural192_preserved(self):
        contract=normalize_collector_contract({})
        self.assertEqual(contract,B4|{'max_new_tokens':192})
        metadata=collector_metadata(contract)
        self.assertEqual(metadata['natural_max_tokens'],192)
        self.assertFalse(metadata['ignore_eos'])
        self.assertNotIn('audit_generation_tokens',metadata)

    def test_legacy_b3_injected_likelihood_audit1_preserved(self):
        contract=normalize_collector_contract({'request_kind':'choice_loglikelihood','collector_output_tokens':1},object())
        self.assertEqual(contract,{'request_kind':'choice_loglikelihood','stream':False,'max_new_tokens':1,'ignore_eos':True,'audit_generation_tokens':1})
        self.assertEqual(collector_metadata(contract)['audit_generation_tokens'],1)
        self.assertNotIn('natural_max_tokens',collector_metadata(contract))

    def test_explicit_b4_driver_is_natural64_not_likelihood(self):
        contract=normalize_collector_contract({'request_kind':'generate_until','collector_output_tokens':64},object(),B4)
        self.assertEqual(contract,B4)
        self.assertEqual(collector_metadata(contract)['natural_max_tokens'],64)
        self.assertNotIn('audit_generation_tokens',collector_metadata(contract))

    def test_invalid_contract_rejected(self):
        changes=({'request_kind':'other'},{'stream':False},{'ignore_eos':True},
                 {'max_new_tokens':0},{'max_new_tokens':True},{'stream':1},
                 {'ignore_eos':0},{'audit_generation_tokens':1},{'audit_generation_tokens':False})
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                normalize_collector_contract({},object(),B4|change)
        for value in ({k:v for k,v in B4.items() if k!='stream'},B4|{'unexpected':1}):
            with self.assertRaises(ValueError):normalize_collector_contract({},object(),value)

    def test_default_b1_does_not_silently_accept_b4_contract(self):
        with self.assertRaises(ValueError):normalize_collector_contract({},None,B4)

    def test_plan_kind_and_token_bound_mismatch_rejected(self):
        for plan in ({'request_kind':'choice_loglikelihood'},{'collector_output_tokens':192}):
            with self.assertRaises(ValueError):normalize_collector_contract(plan,object(),B4)

    def test_actual_wire_bound_stream_eos_and_kind_checked(self):
        row={'request_kind':'generate_until','request_body':{'max_tokens':64,'stream':True,'ignore_eos':False}}
        validate_collector_records([row],B4)
        for key,value in (('max_tokens',192),('max_tokens',True),('stream',False),('ignore_eos',True)):
            changed=copy.deepcopy(row);changed['request_body'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_collector_records([changed],B4)
        with self.assertRaises(ValueError):validate_collector_records([row|{'request_kind':'choice_loglikelihood'}],B4)

    def test_launch_profile_uses_explicit64_not_default192(self):
        plan={'pair':{'model_id':'synthetic/model','revision':'revision'},'samples':{'tokenizer_files':{'tokenizer.json':'hash'}},
              'suite':{'serving':{'context':2048,'max_num_batched_tokens':2048,'kv_tokens':4096},'seed':9},
              'paths':{'server_python':'/explicit/python'},'port':18188,'gpu':{'index':0,'uuid':'synthetic'}}
        arm={'arm_id':'bf16','source_format':'native','artifact':{'artifact_identity':'identity','config_sha256':'hash','path':'/artifact','tokenizer_path':'/tokenizer'}}
        with tempfile.TemporaryDirectory() as t,patch('inference_kit.suites.bench05b1_run.launch',return_value={}) as launch:
            server_launch(plan,arm,Path(t),B4)
        profile=launch.call_args.args[0]['profile']
        self.assertEqual(profile['output_tokens'],64)
        self.assertEqual((profile['context'],profile['max_num_batched_tokens'],profile['kv_tokens']),(2048,2048,4096))

    def test_shared_gpu_blocks_without_start_or_budget_change(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ledger=root/'ledger.json';write(ledger,{'jobs':[],'used_seconds':12,'cap_seconds':612})
            before=sha(ledger);plan={'request_kind':'generate_until','collector_output_tokens':64,'suite':{'budget':{'max_gpu_processes':6}}};progress={'jobs':{}}
            with patch('inference_kit.suites.bench05b1_run.check_gpu',return_value={'status':'BLOCKED_SHARED_GPU'}),patch('inference_kit.suites.bench05b1_run.owned.start',side_effect=AssertionError('must not launch')):
                result=asyncio.run(serving_job(plan,root,ledger,progress,{'arm_id':'bf16'},request_driver=object(),collector_contract=B4))
            self.assertFalse(result);self.assertEqual(sha(ledger),before)
            self.assertEqual(progress['jobs']['serve-bf16']['status'],'BLOCKED_SHARED_GPU')

    def test_bad_contract_rejected_before_gpu_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ledger=root/'ledger.json';write(ledger,{'jobs':[]});before=sha(ledger)
            with patch('inference_kit.suites.bench05b1_run.check_gpu',side_effect=AssertionError('must not query GPU')):
                with self.assertRaises(ValueError):asyncio.run(serving_job({},root,ledger,{'jobs':{}},{'arm_id':'bf16'},request_driver=object(),collector_contract=B4|{'ignore_eos':True}))
            self.assertEqual(sha(ledger),before);self.assertFalse((root/'attempts').exists())


if __name__=='__main__':unittest.main()
