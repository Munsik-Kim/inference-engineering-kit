import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch,AsyncMock
from contextlib import redirect_stdout,redirect_stderr
import inference_kit
from inference_kit import bench_b1_plan,bench_b1_run,bench_b1_replay
from inference_kit.core.io import read,write,digest,sha
from inference_kit.tasks.tech_record import make_dataset,canonical_reference,score,aggregate,verify_gold,render_prompt
from inference_kit.suites.bench05b1_plan import validate_data,validate_prepared
from inference_kit.probes.task_tokens import field_targets
from inference_kit.cli import main
from .helpers import prepared_fixture,local_fixture,synthetic_run


class ScorerTests(unittest.TestCase):
    def test_authored_task_resource_present_in_distribution(self):
        from importlib.resources import files
        packaged=json.loads(files('inference_kit.resources').joinpath('bench05b1-data.json').read_text(encoding='utf-8'))
        self.assertEqual(packaged,make_dataset())
    def setUp(self):self.sample=make_dataset()['rows'][0]
    def generation(self,value=None):
        s=self.sample
        return {'task_id':s['task_id'],'task_version':s['task_version'],'sample_id':s['sample_id'],'sample_identity':s['sample_identity'],
                'output_text':canonical_reference(s)['text'] if value is None else value,'transport_complete':True,'termination':'NATURAL_STOP','request_latency_http':.1,'request_latency_task':21.}
    def answer(self):return json.loads(canonical_reference(self.sample)['text'])
    def test_all_authored_gold_and_independent_source_checks(self):
        data=make_dataset();self.assertEqual(len(validate_data(data)),28)
        for s in data['rows']:
            self.sample=s;self.assertTrue(score(s,self.generation())['task_success'])
    def test_wrong_number(self):
        x=self.answer();x['temperature_c']+=1;r=score(self.sample,self.generation(json.dumps(x)));self.assertFalse(r['field_correct']['temperature_c']);self.assertTrue(r['unsupported_value']['temperature_c'])
    def test_wrong_id(self):
        x=self.answer();x['experiment_id']='EXP-other';self.assertFalse(score(self.sample,self.generation(json.dumps(x)))['task_success'])
    def test_wrong_status_and_evidence(self):
        x=self.answer();x['status']='repeat';x['evidence']['temperature_c']='L99';r=score(self.sample,self.generation(json.dumps(x)));self.assertFalse(r['field_correct']['status']);self.assertFalse(r['evidence_correct']['temperature_c'])
    def test_missing_key_is_not_gold_null(self):
        self.sample=next(s for s in make_dataset()['rows'] if s['gold']['temperature_c'] is None)
        x=self.answer();del x['temperature_c'];r=score(self.sample,self.generation(json.dumps(x)));self.assertFalse(r['schema_valid']);self.assertFalse(r['field_correct']['temperature_c'])
    def test_null_nonnull_false_abstention(self):
        x=self.answer();x['temperature_c']=None;r=score(self.sample,self.generation(json.dumps(x)));self.assertTrue(r['false_abstention']['temperature_c']);self.assertFalse(r['task_success'])
    def test_omission_nonnull(self):
        x=self.answer();del x['status'];r=score(self.sample,self.generation(json.dumps(x)));self.assertTrue(r['omission']['status'])
    def test_duplicate_nan_infinity_fence_trailing_rejected(self):
        good=self.generation()['output_text']
        for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','```json\n'+good+'\n```',good+' explanation'):
            with self.subTest(text=text):self.assertFalse(score(self.sample,self.generation(text))['json_valid'])
    def test_boolean_and_numeric_string_not_number(self):
        for value in (True,'31'):
            x=self.answer();x['temperature_c']=value;self.assertFalse(score(self.sample,self.generation(json.dumps(x)))['schema_valid'])
    def test_extra_key_and_evidence_key_rejected(self):
        for field in ('top','evidence'):
            x=self.answer()
            if field=='top':x['extra']=1
            else:x['evidence']['extra']='L1'
            self.assertFalse(score(self.sample,self.generation(json.dumps(x)))['schema_valid'])
    def test_decimal_absolute_tolerance(self):
        for delta,correct in ((.0000005,True),(.000002,False)):
            x=self.answer();x['temperature_c']+=delta;self.assertEqual(score(self.sample,self.generation(json.dumps(x)))['field_correct']['temperature_c'],correct)
    def test_length_cap_can_be_correct_json(self):
        g=self.generation();g['termination']='LENGTH_CAP';self.assertTrue(score(self.sample,g)['task_success'])
    def test_transport_failure_cannot_be_task_success(self):
        g=self.generation();g['transport_complete']=False;g['termination']='TIMEOUT';r=score(self.sample,g);self.assertTrue(r['schema_valid']);self.assertFalse(r['task_success'])
    def test_http_deadline_excludes_client_queue(self):
        g=self.generation();self.assertTrue(score(self.sample,g)['task_success_within_10s']);g['request_latency_http']=10.1;self.assertFalse(score(self.sample,g)['task_success_within_10s'])
    def test_wrong_target_or_initial_source_value_classified(self):
        self.sample=next(s for s in make_dataset()['rows'] if s['case_type']=='final_correction')
        x=self.answer();x['temperature_c']=self.sample['events'][0]['value'];r=score(self.sample,self.generation(json.dumps(x)));self.assertEqual(r['value_error_type']['temperature_c'],'OTHER_TARGET_OR_INITIAL_SOURCE_VALUE')
    def test_gold_source_corruption_rejected(self):
        sample=copy.deepcopy(self.sample);sample['gold']['temperature_c']+=1
        with self.assertRaises(ValueError):verify_gold(sample)
    def test_pairing_version_and_sample_identity_rejected(self):
        for key,value in (('sample_id','wrong'),('task_version','2'),('sample_identity','bad')):
            g=self.generation();g[key]=value
            with self.assertRaises(ValueError):score(self.sample,g)
    def test_errors_and_not_run_stay_in_planned_denominator(self):
        samples=make_dataset()['rows'][4:];self.sample=samples[0];g=self.generation('bad json');scores,summary=aggregate(samples,[g])
        self.assertEqual((summary['expected'],summary['submitted'],summary['not_run']),(24,1,23));self.assertEqual(summary['metrics']['task_success']['correct'],0);self.assertIsNone(scores[1]['task_success'])
    def test_duplicate_generated_sample_rejected(self):
        g=self.generation()
        with self.assertRaises(ValueError):aggregate([self.sample],[g,g])
    def test_prompt_does_not_include_gold_json_or_hidden_metadata(self):
        prompt=render_prompt(self.sample);self.assertNotIn(canonical_reference(self.sample)['text'],prompt);self.assertNotIn('sample_identity',prompt);self.assertNotIn('gold',prompt)


class PlanReplayTests(unittest.TestCase):
    def test_value_key_spans_and_serving_prompt_mismatch_rejected(self):
        for kind in ('span','prompt'):
            data,p=prepared_fixture()
            if kind=='span':p['generation_rows'][0]['value_spans']['status']=[1,5]
            else:p['generation_rows'][0]['generation_input_ids']=[1,2]
            p['generation_rows_identity']=digest(p['generation_rows'])
            with self.assertRaises(ValueError):validate_prepared(p,data)
    def test_A_wrapper_refuses_B1_task_policy(self):
        from inference_kit.suites.bench05a_plan import validate_samples
        _,prepared=prepared_fixture();prepared['kind']='prepared_diagnostic_samples'
        with self.assertRaisesRegex(ValueError,'A wrapper'):validate_samples(prepared)
    def test_orchestration_only_two_servers_and_two_diagnostic_jobs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root);bench_b1_plan(*args,root/'plan')
            server=AsyncMock(return_value=True);diagnostic=AsyncMock(return_value=True)
            with patch('inference_kit.suites.bench05b1_run.serving_job',server),patch('inference_kit.suites.bench05b1_run.launch_job',diagnostic):
                result=bench_b1_run(root/'plan/PLAN_FROZEN.json',root/'run')
            self.assertEqual(result['status'],'COMPLETE');self.assertEqual(server.await_count,2);self.assertEqual(diagnostic.await_count,2)
            self.assertTrue(all(call.args[4]=='main' for call in diagnostic.await_args_list));self.assertEqual(read(root/'run/budget-continuation.json')['used_seconds'],10)
    def test_prepared_field_selection_has_next_token_shift(self):
        data,p=prepared_fixture();validate_prepared(p,data)
        for row in p['rows']:
            self.assertLessEqual(len(row['targets']),16)
            for t in row['targets']:self.assertEqual(t['logit_position'],t['token_position']-1)
            self.assertTrue(all(v['selected_positions'] for v in row['field_token_coverage'].values()))
    def test_plan_no_gpu_runtime_or_network(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root)
            with patch('subprocess.run',side_effect=AssertionError('plan runtime')):r=bench_b1_plan(*args,root/'plan')
            self.assertEqual((r['gpu_actions'],r['network_actions'],r['model_loads']),(0,0,0));self.assertEqual(r['counts']['natural_requests_total'],56)
    def test_real_resource_limits_not_lowered(self):
        kit=Path(__file__).parents[2];suite=read(kit/'configs/bench05b1/suite.json');self.assertEqual(suite['budget']['minimum_cpu_available_bytes'],10*2**30);self.assertEqual(suite['budget']['minimum_disk_free_bytes'],4*2**30)
    def test_cpu_ram_and_disk_shortfall_block_before_gpu_query_and_budget(self):
        for resource in ('minimum_cpu_available_bytes','minimum_disk_free_bytes'):
            with self.subTest(resource=resource),tempfile.TemporaryDirectory() as t:
                root=Path(t);args=local_fixture(root);suite=read(args[0]);suite['budget'][resource]=sys.maxsize;args[0].write_text(json.dumps(suite),encoding='utf-8');bench_b1_plan(*args,root/'plan')
                before=sha(root/'prior.json')
                with patch('inference_kit.suites.bench05b1_run.check_gpu',side_effect=AssertionError('resource GPU query')),patch('inference_kit.adapters.owned.start',side_effect=AssertionError('resource launch')):
                    with self.assertRaises(ValueError):bench_b1_run(root/'plan/PLAN_FROZEN.json',root/'run')
                self.assertEqual(sha(root/'prior.json'),before);self.assertFalse((root/'run').exists())
    def test_changed_input_after_freeze_rejected_without_launch(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root);bench_b1_plan(*args,root/'plan');data=read(root/'data.json');data['scorer']='changed';(root/'data.json').write_text(json.dumps(data),encoding='utf-8')
            with patch('inference_kit.suites.bench05b1_run.check_gpu',side_effect=AssertionError('changed input query')):
                with self.assertRaises(ValueError):bench_b1_run(root/'plan/PLAN_FROZEN.json',root/'run')
    def test_changed_source_after_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);args=local_fixture(root);bench_b1_plan(*args,root/'plan')
            with patch('inference_kit.suites.bench05b1_plan.source_identity',return_value={}):
                with self.assertRaises(ValueError):bench_b1_run(root/'plan/PLAN_FROZEN.json',root/'run')
    def test_replay_raw_generations_and_probe_links_synthetic_not_live(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw');r=bench_b1_replay(raw,root/'replay')
            self.assertEqual(r['arms']['bf16']['splits']['DEV_LOCKED']['metrics']['task_success']['correct'],24);self.assertEqual(r['probe']['paired_samples'],4);self.assertEqual(r['probe']['status'],'SYNTHETIC_CONTRACT_ONLY')
            linked=[json.loads(s) for s in (root/'replay/sample-linked.jsonl').read_text(encoding='utf-8').splitlines()];self.assertEqual(len(linked),24);self.assertEqual(sum(s['teacher_forced_probe']['target_positions']>0 for s in linked),4)
            with self.assertRaises(ValueError):bench_b1_replay(raw,root/'replay')
    def test_replay_raw_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_run(root/'raw');path=next(raw.rglob('generations.jsonl'));path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):bench_b1_replay(raw,root/'replay')
    def test_cli_run_requires_opt_in(self):
        with redirect_stderr(io.StringIO()),patch('subprocess.run',side_effect=AssertionError('run without opt-in')):
            self.assertEqual(main(['bench','b1','run','--plan','없음','--output','없음']),2)
    def test_teacher_forced_reference_is_independent_of_free_generation(self):
        data,p=prepared_fixture();sid=p['rows'][0]['sample_id'];sample=next(s for s in data['rows'] if s['sample_id']==sid)
        before=digest(p['rows'][0]);self.assertEqual(canonical_reference(sample)['text'],p['rows'][0]['reference_text']);score(sample,{'task_id':sample['task_id'],'task_version':sample['task_version'],'sample_id':sid,'sample_identity':sample['sample_identity'],'output_text':'wrong answer','transport_complete':True})
        self.assertEqual(before,digest(p['rows'][0]))


class LocaleTests(unittest.TestCase):
    def child(self,code,directory):
        package_root=str(Path(inference_kit.__file__).parent.parent.absolute())
        body='import sys,locale;sys.path.insert(0,'+repr(package_root)+');assert locale.getpreferredencoding(False).lower() not in ("utf-8","utf8");'+code
        return subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',body],cwd=directory,env=dict(os.environ,LC_ALL='C',LANG='C'),capture_output=True,encoding='utf-8')
    def test_korean_raw_score_csv_markdown_and_sample_links_ascii_locale(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);synthetic_run(root/'raw');r=self.child('from inference_kit import bench_b1_replay;r=bench_b1_replay("raw","replay");assert r["probe"]["paired_samples"]==4',root)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('한글',(root/'replay/sample-linked.jsonl').read_text(encoding='utf-8'));self.assertIn('한글',(root/'replay/sample-linked.csv').read_text(encoding='utf-8'));self.assertIn('합성',(root/'replay/REPORT_KO.md').read_text(encoding='utf-8'))
    def test_cli_success_json_matches_utf8_score_file_and_digest(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);sample=make_dataset()['rows'][-1];sample['sample_id']='한글-ID';sample.pop('sample_identity');sample['sample_identity']=digest(sample)
            g={'task_id':sample['task_id'],'task_version':sample['task_version'],'sample_id':sample['sample_id'],'sample_identity':sample['sample_identity'],'output_text':canonical_reference(sample)['text'],'transport_complete':True}
            write(root/'sample.json',sample);write(root/'generation.json',g);before=digest(sample)
            r=self.child('from inference_kit.cli import main;raise SystemExit(main(["bench","b1","score","--sample","sample.json","--generation","generation.json","--output","out"]))',root)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),read(root/'out/score.json'));self.assertEqual(before,digest(read(root/'sample.json')))
    def test_cli_error_json_ascii_locale_with_korean_error(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'sample.json').write_text('{"한글오류":1,"한글오류":2}',encoding='utf-8')
            r=self.child('from inference_kit.cli import main;raise SystemExit(main(["bench","b1","score","--sample","sample.json","--generation","missing","--output","out"]))',t)
            self.assertEqual(r.returncode,2);self.assertIn('한글오류',json.loads(r.stderr)['error'])
