"""E2 locale and public contract controls; no model or network."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import inference_kit
from inference_kit.core.io import sha,write
from inference_kit.suites import suite_plan
from bench05e2.test_plan import sources_fixture,suite_fixture,local_fixture

class PortableContractTests(unittest.TestCase):
    def test_scorer_declares_gold_input_and_preserves_pair_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=suite_plan.compile_suite(suite_fixture(),sources_fixture(Path(tmp)),local_fixture())
            rows=[n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='score_task' and n['arm_id']]
            self.assertEqual(len(rows),6)
            for node in rows:
                self.assertEqual({r['schema'] for r in node['inputs']},{'native_task_raw_v1','validated_prepared_task_v1'})
                self.assertEqual(len({tuple(r['sample_ids']) for r in node['inputs']}),1)

    def test_historical_labels_never_replace_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=sources_fixture(Path(tmp));spec=suite_fixture()
            spec['outputs']['historical_source_ids']=['tech','unknown-stale-source']
            result=suite_plan.compile_suite(spec,source,local_fixture())
            links=result['semantic']['historical_links']
            self.assertEqual([r['status'] for r in links],['PROVENANCE_RELATED_ONLY','UNRESOLVED_RELATED_ONLY'])
            self.assertTrue(all(not r['cache_eligible'] and not r['replaces_acquisition'] for r in links))
            self.assertEqual(result['semantic']['counts']['inference_POSTs'],20)
            self.assertEqual(result['acquisition_count'],0)

    def test_optional_all_blocked_is_not_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec=suite_fixture('diagnostic');spec['observations']['token']['required']=False
            spec['outputs'].update(index=False,report=False)
            for model in spec['models']:model['profile']='unsupported'
            result=suite_plan.compile_suite(spec,sources_fixture(Path(tmp)),local_fixture())
            self.assertEqual(result['status'],'BLOCKED_REQUIRED_OPERATIONS')
            self.assertTrue(result['semantic']['graph']['all_acquisition_operations_blocked'])

    def test_frozen_reporting_state_cannot_claim_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=suite_plan.compile_suite(suite_fixture(),sources_fixture(Path(tmp)),local_fixture())
            for key,value in [('execution_count',1),('acquisition_count',True),('status','VERIFIED')]:
                changed=copy.deepcopy(original);changed[key]=value
                with self.assertRaises(ValueError):suite_plan.check_frozen(changed)

    def test_actual_ascii_locale_api_files_and_cli_success_error(self):
        package_root=str(Path(inference_kit.__file__).resolve().parent.parent)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources_fixture(root);spec=suite_fixture();spec['suite_id']='한글-실험'
            write(root/'suite.json',spec);write(root/'local.json',local_fixture())
            code='import sys,locale,json;sys.path.insert(0,'+repr(package_root)+');\n'
            code+="assert locale.getpreferredencoding(False).lower() not in ('utf-8','utf8')\n"
            code+='from inference_kit import suite_plan,suite_validate,suite_describe\n'
            code+='from inference_kit.cli import main\n'
            code+='p=suite_plan('+repr(str(root/'suite.json'))+','+repr(str(root/'sources.json'))+','+repr(str(root/'local.json'))+','+repr(str(root/'api-plan'))+')\n'
            code+='suite_validate('+repr(str(root/'api-plan/FROZEN_PLAN.json'))+','+repr(str(root/'sources.json'))+','+repr(str(root/'api-validate'))+')\n'
            code+='main(["bench","suite","describe","--plan",'+repr(str(root/'api-plan/FROZEN_PLAN.json'))+',"--output",'+repr(str(root/'cli-description'))+'])\n'
            env=dict(os.environ,LC_ALL='C',LANG='C',PYTHONCOERCECLOCALE='0');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            child=subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',code],env=env,capture_output=True)
            self.assertEqual(child.returncode,0,child.stderr.decode('ascii',errors='backslashreplace'))
            value=json.loads(child.stdout.decode('ascii'));self.assertIn('한글-실험',json.dumps(value,ensure_ascii=False))
            self.assertIn('한글-실험',(root/'cli-description/PLAN_DESCRIPTION.json').read_text(encoding='utf-8'))
            invalid=copy.deepcopy(spec);invalid['없는-필드']=True;write(root/'invalid.json',invalid)
            bad=code.split('p=suite_plan(')[0]+'raise SystemExit(main(["bench","suite","plan","--suite",'+repr(str(root/'invalid.json'))+',"--sources",'+repr(str(root/'sources.json'))+',"--local",'+repr(str(root/'local.json'))+',"--output",'+repr(str(root/'error'))+']))\n'
            error=subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',bad],env=env,capture_output=True)
            self.assertNotEqual(error.returncode,0)
            parsed=json.loads(error.stderr.decode('ascii'));self.assertIn('없는',json.dumps(parsed,ensure_ascii=False))

if __name__=='__main__':unittest.main()
