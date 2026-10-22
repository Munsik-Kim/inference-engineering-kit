"""CPU-only configured planning/resource/launch gates; no model or GPU query."""
import asyncio,copy,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import unittest
import inference_kit
from unittest.mock import patch,AsyncMock
from inference_kit.core.io import read,write,sha,digest
from inference_kit.suites import token_probe as capture
from inference_kit.suites.bench05a_run import diagnostic_launch
from bench05c2.test_inputs import source_fixture,spec_fixture


def fixture(root):
    manifest=source_fixture(root)
    write(root/'sources.json',manifest);write(root/'spec.json',spec_fixture());write(root/'pair.json',{'schema_version':1,'model_id':'HuggingFaceTB/SmolLM2-1.7B-Instruct','revision':'31b70e2e869a7173562077fd711b654946d38674','arms':[{'arm_id':'bf16','receipt_arm':'BF16_REF','execution_mode':'INSTRUMENTED_PATH'},{'arm_id':'rtn','receipt_arm':'R00','execution_mode':'DECODED_WEIGHT_DIAGNOSTIC'}]})
    write(root/'artifact.json',{});write(root/'runtime.json',{})
    write(root/'ledger.json',{'running':None,'used_seconds':1000.,'cap_seconds':7200,'jobs':[{'wall_seconds':1000.}]})
    now=time.monotonic();write(root/'clock.json',{'start_utc':'synthetic-only','start_monotonic':now,'deadline_monotonic':now+3600,'no_new_gpu_after':now+3000})
    local={'artifact_receipt':str(root/'artifact.json'),'runtime_metadata':str(root/'runtime.json'),'runtime_python':sys.executable,'package_site':str(root/'package-site'),
      'prior_ledger':str(root/'ledger.json'),'campaign_root':str(root),'task_clock':str(root/'clock.json'),'task_origin_used_seconds':1000.,'gpu':0,'gpu_uuid':'synthetic-never-query'}
    write(root/'local.json',local)
    ids={'bf16':'0c668d5b8e51322d63898007d3d4a4ea488f21168208c3f3fc0d0f9643ac371b','rtn':'c8debb3188403ab9d332149969e6e0f9b7b689dc06e77e882fde07a9cb8da761'}
    arms=[{'arm_id':name,'execution_mode':mode,'artifact_available':True,'artifact':{'artifact_identity':ids[name],'recipe_identity':'synthetic','path':str(root/'no-weights')},'weight_file_stat':{}}
      for name,mode in [('bf16','INSTRUMENTED_PATH'),('rtn','DECODED_WEIGHT_DIAGNOSTIC')]]
    return arms


class ConfiguredPlanTests(unittest.TestCase):
    def plan(self,root,arms,healthy=True):
        with patch.object(capture,'inspect_artifacts',return_value=arms),patch.object(capture,'resources',return_value={'ready':healthy,'cpu_available_bytes':123,'disk_free_bytes':456}),patch.object(capture,'inspect_capabilities',return_value={'features':{'target_logprobs':{'status':'PLANNED_NOT_VERIFIED'},'selected_full_vocab':{'status':'PLANNED_NOT_VERIFIED'},'attention_output':{'status':'PLANNED_NOT_VERIFIED'}}}),patch.object(capture,'check_gpu') as gpu:
            result=capture.plan(root/'spec.json',root/'sources.json',root/'pair.json',root/'local.json',root/'plan');gpu.assert_not_called();return result
    def test_plan_never_queries_gpu_or_mutates_prior_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arms=fixture(root);prior=sha(root/'ledger.json');plan=self.plan(root,arms)
            self.assertEqual(plan['status'],'CPU_PLANNED_NOT_EXECUTED');self.assertEqual(plan['counts']['planned_gpu_jobs'],2)
            self.assertEqual(plan['counts']['forward_calls_total'],12);self.assertEqual(sha(root/'ledger.json'),prior);self.assertEqual(plan['attention_observations'],0)
    def test_resource_failure_blocks_before_gpu_budget_and_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arms=fixture(root);plan=self.plan(root,arms,False);prior=sha(root/'ledger.json')
            with patch.object(capture,'check_gpu') as gpu,patch.object(capture,'launch_job',new_callable=AsyncMock) as launch:
                with self.assertRaises(ValueError):capture.run(root/'plan/PLAN_FROZEN.json',root/'run')
                gpu.assert_not_called();launch.assert_not_awaited()
            self.assertFalse((root/'run').exists());self.assertEqual(sha(root/'ledger.json'),prior)
    def test_recheck_resource_failure_does_not_query_gpu_or_create_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arms=fixture(root);self.plan(root,arms);prior=sha(root/'ledger.json')
            with patch.object(capture,'resources',return_value={'ready':False}),patch.object(capture,'check_gpu') as gpu:
                result=capture.run(root/'plan/PLAN_FROZEN.json',root/'run');gpu.assert_not_called()
            self.assertEqual(result['status'],'BLOCKED_CPU_OR_DISK_RESOURCE');self.assertFalse((root/'run/budget-continuation.json').exists());self.assertEqual(sha(root/'ledger.json'),prior)
    def test_shared_gpu_blocks_without_launch_or_budget_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arms=fixture(root);self.plan(root,arms);prior=sha(root/'ledger.json')
            with patch.object(capture,'resources',return_value={'ready':True}),patch.object(capture,'check_gpu',return_value={'status':'BLOCKED_SHARED_GPU'}),patch.object(capture,'launch_job',new_callable=AsyncMock) as launch:
                result=capture.run(root/'plan/PLAN_FROZEN.json',root/'run');launch.assert_not_awaited()
            self.assertEqual(result['status'],'BLOCKED_SHARED_GPU');self.assertFalse((root/'run/budget-continuation.json').exists());self.assertEqual(sha(root/'ledger.json'),prior)
    def test_frozen_spec_and_source_ancestry_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arms=fixture(root);p=self.plan(root,arms);capture.check_frozen(p)
            modified=copy.deepcopy(p);modified['probe_spec']['max_target_positions_per_sample']=31
            with self.assertRaises(ValueError):capture.check_frozen(modified)
            (root/'sources.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):capture.check_frozen(p)
    def test_legacy_and_token_only_worker_argv_explicit(self):
        root=Path('/synthetic');plan={'paths':{'runtime_python':'/explicit/venv/bin/python','package_site':'/own/site'},'gpu':{'index':0}}
        legacy=diagnostic_launch(plan,root,'main',{'arm_id':'candidate'},root/'job')
        self.assertIn('inference_kit.probes.local_transformer',legacy['argv']);self.assertIn('--stage',legacy['argv'])
        token=diagnostic_launch(plan,root,'main',{'arm_id':'candidate'},root/'job',capture.WORKER)
        self.assertIn('inference_kit.probes.token_worker',token['argv']);self.assertNotIn('--stage',token['argv'])
        self.assertEqual(token['argv'][0],'/explicit/venv/bin/python')
        with self.assertRaises(ValueError):diagnostic_launch(plan,root,'main',{'arm_id':'candidate'},root/'job',{'module':'arbitrary.code','token_only':True})
    def test_cli_run_requires_explicit_gpu_before_reading_plan(self):
        from inference_kit.cli import main
        from contextlib import redirect_stderr
        from io import StringIO
        with tempfile.TemporaryDirectory() as d,patch('inference_kit.api.tokens_run') as run:
            error=StringIO()
            with redirect_stderr(error):code=main(['bench','tokens','run','--plan',str(Path(d)/'없는계획.json'),'--output',str(Path(d)/'out')])
            self.assertEqual(code,2);run.assert_not_called();self.assertEqual(json.loads(error.getvalue())['error_type'],'ValueError')
    def test_non_utf8_cli_error_and_probe_input_files_remain_utf8(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=source_fixture(root);write(root/'sources.json',manifest)
            package=str(Path(inference_kit.__file__).absolute().parent.parent)
            code='import sys,locale;sys.path.insert(0,'+repr(package)+');assert locale.getpreferredencoding(False).lower() not in ("utf-8","utf8");from pathlib import Path;from inference_kit.probes.token_inputs import prepare_inputs;from inference_kit.core.io import write;write(Path("inputs.json"),prepare_inputs("sources.json"));from inference_kit.cli import main;raise SystemExit(main(["bench","tokens","plan","--spec",chr(0xd55c)+".json","--inputs","sources.json","--models","x","--local","x","--output","x"]))'
            child=subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',code],cwd=root,env=dict(os.environ,LC_ALL='C',LANG='C'),capture_output=True,encoding='utf-8')
            # The deliberately ASCII filesystem codec cannot encode this path;
            # the error JSON must still be valid ASCII, while file CONTENT is UTF-8.
            self.assertEqual(child.returncode,2,child.stderr);self.assertEqual(json.loads(child.stderr)['error_type'],'UnicodeEncodeError')
            self.assertEqual(len(read(root/'inputs.json')['rows']),6)


if __name__=='__main__':unittest.main()
