"""D1 CPU plans keep C2 inputs, resource gates and token-only defaults intact."""
import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,AsyncMock
import inference_kit
from inference_kit.core.io import read,write,digest,sha
from inference_kit.probes.structure_inputs import select,indices,OBSERVER_POLICY
from inference_kit.probes.token_inputs import prepare_inputs,select_inputs
from inference_kit.suites import structure_probe,token_probe
from inference_kit.suites.bench05a_run import diagnostic_launch
from bench05c2.test_inputs import source_fixture,spec_fixture,refresh
from bench05c2.test_plan import fixture as c2_fixture


def spec(profile='MAIN'):
    return {'schema_version':1,'kind':'structure_probe_spec_v1','profile':profile,'max_sequence_tokens':640,
      'observe_attention':True,'backend':'native_transformers_sdpa_smol_mha_v1','numeric':{'dtype':'float32','endianness':'little'},
      'layers':'first_middle_last' if profile=='MAIN' else 'middle','heads':'first_middle' if profile=='MAIN' else 'first',
      'query_roles':['last_prompt','before_last_reference'] if profile=='MAIN' else ['last_prompt'],
      'observer_consistency_policy':copy.deepcopy(OBSERVER_POLICY),'budget':spec_fixture()['budget']}


def inputs(root):return select_inputs(prepare_inputs(source_fixture(root)),spec_fixture())


class StructurePlanTests(unittest.TestCase):
    def test_small_main_differ_without_reselecting_token_targets(self):
        with tempfile.TemporaryDirectory() as d:
            value=inputs(Path(d));before=copy.deepcopy(value);small=select(value,spec('SMALL'));main=select(value,spec())
            self.assertEqual(value,before);self.assertLess(small['counts']['attention_rows_per_arm'],main['counts']['attention_rows_per_arm'])
            self.assertLess(small['resource_estimates']['raw_numeric_bytes_both_arms'],main['resource_estimates']['raw_numeric_bytes_both_arms'])
            self.assertEqual(small['counts']['target_positions_per_arm'],main['counts']['target_positions_per_arm'])
            self.assertEqual(main['counts']['output_projection_vectors_per_arm']*len(main['heads']),main['counts']['attention_rows_per_arm'])

    def test_indices_are_zero_based_duplicate_and_outside_rejected(self):
        self.assertEqual(indices('first_middle_last',24,'layer'),[0,12,23])
        self.assertEqual(indices('first_middle',32,'head'),[0,16])
        self.assertEqual(indices('first_middle_last',1,'layer'),[0])
        for value in ([],[0,0],[-1],[24],[True],[1.]):
            with self.subTest(value=value),self.assertRaises(ValueError):indices(value,24,'layer')

    def test_query_alias_stores_one_physical_coordinate(self):
        with tempfile.TemporaryDirectory() as d:
            value=inputs(Path(d))
            # Alias semantics are independently validated by worker on a tiny row.
            from inference_kit.probes.structure_worker import validate_selection
            toy={'input_ids':[3,4],'prompt_token_boundary':1,'loss_mask':[0,1],'special_token_mask':[0,0],'attention_mask':[1,1],'targets':[{'token_position':1}]}
            result=validate_selection(toy,{'layers':[0],'heads':[0],'queries':[0],'query_roles':{'0':['last_prompt','before_last_reference']}},value['model_config'])
            self.assertEqual(result['queries'],[0])

    def test_invalid_mask_backend_query_roles_and_buffer_fail_before_model(self):
        with tempfile.TemporaryDirectory() as d:
            value=inputs(Path(d))
            for change in ({'observe_attention':False},{'query_roles':['last_prompt','last_prompt']},{'max_sequence_tokens':128},{'backend':'unknown'}):
                with self.subTest(change=change),self.assertRaises(ValueError):select(value,spec()|change)
            altered=copy.deepcopy(value);altered['model_config']['num_key_value_heads']=1
            altered['selection_identity']=digest({k:v for k,v in altered.items() if k!='selection_identity'})
            with self.assertRaisesRegex(ValueError,'MHA'):select(altered,spec())
            limited=spec();limited['budget']['probe_gpu_buffer_bytes']=1
            with self.assertRaisesRegex(ValueError,'buffer'):select(value,limited)

    def setup_plan(self,root,healthy=True):
        arms=c2_fixture(root);common=prepare_inputs(read(root/'sources.json'));selected=select_inputs(common,read(root/'spec.json'))
        write(root/'inputs.json',selected);write(root/'structure.json',spec());prior=root/'c2-run';prior.mkdir()
        old={'task':'BENCH05-C2','samples':selected,'source_identity':{'fixture.py':'synthetic'}};old['plan_identity']=digest(old);write(prior/'PLAN_FROZEN.json',old)
        write(prior/'RAW_MANIFEST.json',{'scope':'token_probe_raw_v1','files':{'PLAN_FROZEN.json':{'bytes':(prior/'PLAN_FROZEN.json').stat().st_size,'sha256':sha(prior/'PLAN_FROZEN.json')}}})
        local=read(root/'local.json');local.update(prior_input_run=str(prior),prior_input_manifest_sha256=sha(prior/'RAW_MANIFEST.json'))
        (root/'local.json').write_text(json.dumps(local),encoding='utf-8')
        with patch.object(token_probe,'inspect_artifacts',return_value=arms),patch.object(token_probe,'resources',return_value={'ready':healthy,'cpu_available_bytes':123,'disk_free_bytes':456}),patch.object(token_probe,'inspect_capabilities',return_value={'features':{'target_logprobs':{'status':'PLANNED_NOT_VERIFIED'},'selected_full_vocab':{'status':'PLANNED_NOT_VERIFIED'}}}),patch.object(token_probe,'check_gpu') as gpu:
            value=structure_probe.plan(root/'structure.json',root/'inputs.json',root/'pair.json',root/'local.json',root/'plan');gpu.assert_not_called()
        return value

    def test_cpu_plan_source_inputs_and_prior_ledger_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.setup_plan(root);old=sha(root/'ledger.json')
            self.assertEqual(plan['task'],'BENCH05-D1');self.assertEqual(plan['samples'],read(root/'inputs.json'));self.assertEqual(plan['counts']['forward_calls_total'],36)
            self.assertEqual(plan['GPU_actions'],0);self.assertEqual(sha(root/'ledger.json'),old);structure_probe.validate_frozen(plan)
            changed=copy.deepcopy(plan);changed['structure_selection']['layers']=[0]
            with self.assertRaises(ValueError):structure_probe.validate_frozen(changed)

    def test_resource_and_shared_gpu_gates_have_no_budget_or_launch_side_effect(self):
        for resource,status in ((False,'BLOCKED_CPU_OR_DISK_RESOURCE'),(True,'BLOCKED_SHARED_GPU')):
            with self.subTest(resource=resource),tempfile.TemporaryDirectory() as d:
                root=Path(d);plan=self.setup_plan(root);old=sha(root/'ledger.json')
                with patch.object(token_probe,'resources',return_value={'ready':resource}),patch.object(token_probe,'check_gpu',return_value={'status':'BLOCKED_SHARED_GPU'}) as gpu,patch.object(token_probe,'launch_job',new_callable=AsyncMock) as launch:
                    result=structure_probe.run(root/'plan/PLAN_FROZEN.json',root/'run');launch.assert_not_awaited()
                    if not resource:gpu.assert_not_called()
                self.assertEqual(result['status'],status);self.assertEqual(sha(root/'ledger.json'),old);self.assertFalse((root/'run/budget-continuation.json').exists())

    def test_explicit_worker_keeps_legacy_and_c2_defaults(self):
        p={'paths':{'runtime_python':'/venv/bin/python','package_site':'/own/site'},'gpu':{'index':0}};r=Path('/own')
        old=diagnostic_launch(p,r,'main',{'arm_id':'x'},r/'job');self.assertIn('--stage',old['argv'])
        token=diagnostic_launch(p,r,'main',{'arm_id':'x'},r/'job',token_probe.WORKER);self.assertIn('inference_kit.probes.token_worker',token['argv'])
        new=diagnostic_launch(p,r,'main',{'arm_id':'x'},r/'job',structure_probe.WORKER);self.assertIn('inference_kit.probes.structure_worker',new['argv']);self.assertNotIn('--stage',new['argv'])

    def test_nonutf8_locale_api_import_and_ascii_cli_error(self):
        package=str(Path(inference_kit.__file__).parent.parent)
        code=r"""import sys,json,locale
sys.path.insert(0,sys.argv[1])
assert locale.getpreferredencoding(False).lower() not in ('utf-8','utf8')
def guard(event,args):
 if event=='import' and args and str(args[0]).split('.')[0] in {'torch','numpy','transformers','aiohttp','vllm'}:raise AssertionError('CPU SDK import')
sys.addaudithook(guard)
from inference_kit.cli import main
assert main(['bench','structure','run','--plan','\ud55c\uae00.json','--output','unused'])==2
"""
        env=dict(os.environ,LC_ALL='C',LANG='C');child=subprocess.run([sys.executable,'-I','-X','utf8=0','-c',code,package],env=env,capture_output=True,timeout=30)
        self.assertEqual(child.returncode,0,child.stderr.decode('ascii'));self.assertIn('allow-gpu',json.loads(child.stderr.decode('ascii'))['error'])
