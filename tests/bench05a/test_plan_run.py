import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from inference_kit import bench_plan,bench_run
from inference_kit.core.io import read,write,sha
from .helpers import prepared


def local_fixture(root):
    p=prepared();kit=Path(__file__).parents[2];suite=read(kit/'configs/bench05a/suite.json');pair=read(kit/'configs/bench05a/pair.json')
    # Tiny metadata fixtures do not load a model and must not require its 10GiB RAM.
    # Keep the live suite's resource limits unchanged; test its gate separately.
    suite['budget'].update(minimum_cpu_available_bytes=1,minimum_disk_free_bytes=1)
    write(root/'suite.json',suite);write(root/'pair.json',pair)
    authored=[{k:r[k] for k in ('sample_id','source_id','prompt','reference_text','language')} for r in p['rows']];write(root/'samples.json',authored)
    p['authored_sha256']=sha(root/'samples.json');write(root/'prepared.json',p)
    write(root/'runtime.json',{'packages':{'transformers':{'version':'5.14.1'},'compressed-tensors':{'version':'0.18.0'}},'evidence_kind':'synthetic_contract'})
    refs={}
    for key,arm in (('BF16_REF','bf16'),('R00','rtn')):
        d=root/arm;d.mkdir();cfg=copy.deepcopy(p['model_config'])
        if arm=='rtn':cfg['quantization_config']={'quant_method':'compressed-tensors','format':'pack-quantized','config_groups':{'g':{'input_activations':None,'output_activations':None,'targets':['Linear'],'weights':{'num_bits':4,'type':'int','strategy':'group','group_size':128,'symmetric':True,'actorder':None}}}}
        write(d/'config.json',cfg);(d/'tiny.safetensors').write_bytes(b'metadata-only-test-not-a-weight')
        write(d/'artifact.json',{'synthetic_contract':True})
        refs[key]={'model_id':pair['model_id'],'model_revision':pair['revision'],'artifact_identity':'synthetic-'+arm,'recipe_identity':'toy-'+arm,'path':str(d),'tokenizer_path':str(d),
             'config_sha256':sha(d/'config.json'),'manifest_path':str(d/'artifact.json'),'manifest_sha256':sha(d/'artifact.json'),'recipe':{'algorithm':'RTN','ignore':['lm_head']}}
    write(root/'receipts.json',{'arms':refs});write(root/'prior.json',{'used_seconds':10,'running':None,'jobs':[{'wall_seconds':10}]})
    local={'artifact_receipt':str(root/'receipts.json'),'prepared_samples':str(root/'prepared.json'),'runtime_metadata':str(root/'runtime.json'),
      'runtime_python':sys.executable,'server_python':sys.executable,'package_site':str(root/'installed-site'),'prior_ledger':str(root/'prior.json'),'campaign_root':str(root),
      'gpu':0,'gpu_uuid':'synthetic-only','port':18186};write(root/'local.json',local)
    return root/'suite.json',root/'pair.json',root/'local.json'


class PlanTests(unittest.TestCase):
    def test_plan_no_subprocess_model_or_network(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root)
            with patch('subprocess.run',side_effect=AssertionError('plan executed runtime')):result=bench_plan(*args,root/'plan')
            self.assertEqual((result['gpu_actions'],result['model_loads'],result['network_actions']),(0,0,0));self.assertEqual(result['gpu_allowance_seconds'],1200)
            self.assertTrue(all(x['status']=='PLANNED_NOT_VERIFIED' for x in result['capability']['features'].values()))
    def test_shared_gpu_blocks_before_owned_process_or_budget_charge(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root);bench_plan(*args,root/'plan')
            with patch('inference_kit.suites.bench05a_run.check_gpu',return_value={'status':'BLOCKED_SHARED_GPU','unverified_pid':9999}),patch('inference_kit.adapters.owned.start',side_effect=AssertionError('foreign GPU launch')):
                result=bench_run(root/'plan/PLAN_FROZEN.json',root/'run')
            self.assertEqual(result['status'],'BLOCKED_SHARED_GPU');self.assertEqual(result['new_gpu_job_wall_seconds'],0)
            self.assertEqual(read(root/'run/budget-continuation.json')['used_seconds'],10);self.assertEqual(read(root/'prior.json')['used_seconds'],10)
    def test_runtime_signature_mismatch_blocks_without_model_loading(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root);(root/'runtime.json').write_text(json.dumps({'packages':{'transformers':{'version':'unknown'}}}))
            with self.assertRaises(ValueError):bench_plan(*args,root/'plan')
    def test_resource_shortfall_blocks_before_gpu_query_or_launch(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root);suite=read(args[0])
            suite['budget']['minimum_cpu_available_bytes']=sys.maxsize;args[0].write_text(json.dumps(suite),encoding='utf-8')
            result=bench_plan(*args,root/'plan')
            self.assertEqual(result['status'],'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT')
            with patch('inference_kit.suites.bench05a_run.check_gpu',side_effect=AssertionError('resource-blocked GPU query')),patch('inference_kit.adapters.owned.start',side_effect=AssertionError('resource-blocked launch')):
                with self.assertRaisesRegex(ValueError,'unresolved local artifacts/resources'):
                    bench_run(root/'plan/PLAN_FROZEN.json',root/'run')
            self.assertFalse((root/'run').exists());self.assertEqual(read(root/'prior.json')['used_seconds'],10)
