import asyncio
import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import inference_kit
from inference_kit.core.io import digest,read,write,sha
from inference_kit.tasks.arc_challenge import ArcChallengeAdapter,validate_sample,select_pool,decode_record,body
from inference_kit.tasks.registry import get_task
from inference_kit.tasks.base import compatible
from inference_kit.probes.arc_tokens import effective_text
from inference_kit.suites.bench05b3_plan import plan,check_frozen
from inference_kit.suites.bench05b3_replay import replay
from .helpers import sample,raw,local_fixture,synthetic_run

class ArcTests(unittest.TestCase):
    def setUp(self):
        self.sample=sample();self.adapter=ArcChallengeAdapter();self.requests=self.adapter.requests(self.sample,'run|attempt|bf16','tok','space','runtime')
    def scored(self,values):
        return [{'status':'COMPLETE','sample_id':self.sample['sample_id'],'choice_index':i,'choice_loglikelihood':v,'continuation_token_count':2} for i,v in enumerate(values)]
    def test_raw_and_character_normalized_different(self):
        r=self.adapter.score(self.sample,self.scored([-2,-3,-5,-6]));self.assertEqual(r['prediction_acc'],0);self.assertEqual(r['prediction_acc_norm'],1)
    def test_three_four_five_label_mapping(self):
        for n in (3,4,5):
            s=sample(n);r=self.adapter.score(s,self.scored([-5.]*(n-1)+[-.1]));self.assertEqual(r['acc'],1);self.assertEqual(r['acc_norm'],1)
    def test_first_index_tie(self):self.assertEqual(self.adapter.score(self.sample,self.scored([-1]*4))['prediction_acc'],0)
    def test_external_gold_not_reference_prediction(self):self.assertEqual(self.adapter.reference(self.sample)['choice_index'],3)
    def test_empty_choice_rejected(self):
        s=copy.deepcopy(self.sample);s['choices'][0]=''
        with self.assertRaises(ValueError):validate_sample(s)
    def test_duplicate_label_rejected(self):
        s=copy.deepcopy(self.sample);s['choice_labels'][0]=s['choice_labels'][1]
        with self.assertRaises(ValueError):validate_sample(s)
    def test_unknown_answer_key_rejected(self):
        s=copy.deepcopy(self.sample);s['document']['answerKey']='X'
        with self.assertRaises(ValueError):validate_sample(s)
    def test_missing_choice_not_argmax(self):self.assertEqual(self.adapter.score(self.sample,self.scored([-1]*3))['status'],'INCOMPLETE')
    def test_missing_all_not_run(self):self.assertEqual(self.adapter.score(self.sample,[])['status'],'NOT_RUN')
    def test_duplicate_and_unknown_choice_rejected(self):
        for choices in ([{'choice_index':0}]*2,[{'choice_index':7}]):
            with self.assertRaises(ValueError):self.adapter.score(self.sample,choices)
    def test_nonfinite_choice_rejected(self):
        with self.assertRaises(ValueError):self.adapter.score(self.sample,self.scored([float('nan')]*4))
    def test_prompt_none_allowed_audit_excluded_shift(self):
        r=decode_record(self.requests[0],raw(self.requests[0]));self.assertEqual(r['choice_loglikelihood'],-3);self.assertEqual([t['logit_position'] for t in r['tokens']],[1,2]);self.assertEqual([t['nll_nat'] for t in r['tokens']],[1,2])
    def test_continuation_none_and_inf_rejected(self):
        for bad in (None,float('inf')):
            r=raw(self.requests[0]);r['response']['choices'][0]['logprobs']['token_logprobs'][2]=bad;r['raw_response_text']=json.dumps(r['response'])
            if bad is None:self.assertEqual(decode_record(self.requests[0],r)['status'],'INCOMPLETE')
            else:
                with self.assertRaises(ValueError):decode_record(self.requests[0],r)
    def test_echo_token_mismatch_rejected(self):
        r=raw(self.requests[0]);r['response']['choices'][0]['logprobs']['tokens'][2]='token_id:999';r['raw_response_text']=json.dumps(r['response'])
        self.assertEqual(decode_record(self.requests[0],r)['status'],'INCOMPLETE')
    def test_request_body_mismatch_rejected(self):
        r=raw(self.requests[0]);r['request_body']['echo']=False
        with self.assertRaises(ValueError):decode_record(self.requests[0],r)
    def test_raw_response_text_mismatch_rejected(self):
        r=raw(self.requests[0]);r['raw_response_text']='{}'
        with self.assertRaises(ValueError):decode_record(self.requests[0],r)
    def test_transport_failure_incomplete(self):
        r=raw(self.requests[0]);r['transport_complete']=False;self.assertEqual(decode_record(self.requests[0],r)['status'],'INCOMPLETE')
    def test_generate_cannot_enter_likelihood_collector(self):
        req=dataclasses.replace(self.requests[0],request_kind='generate_until')
        with self.assertRaises(ValueError):body(req)
    def test_generation_score_cannot_accept_likelihood(self):
        from inference_kit.tasks.tech_record import make_dataset
        with self.assertRaises(ValueError):get_task('tech_record_extract').score(make_dataset()['rows'][0],{'request_kind':'choice_loglikelihood'})
    def test_pairing_rejects_each_axis(self):
        left=self.requests[0].identity
        for key in ('task_id','task_version','split','sample_id','prompt_identity','tokenizer_identity','scorer_version','source_identity','runtime_identity'):
            with self.assertRaises(ValueError):compatible(left,left|{key:'different'})
    def test_request_ids_unique_per_arm_choice(self):
        right=self.adapter.requests(self.sample,'run|attempt|rtn','tok');self.assertEqual(len({r.request_id for r in self.requests+right}),8)
    def test_trailing_whitespace_is_continuation(self):
        p,r=effective_text({'context':'Answer:\n','continuation':' yes'});self.assertEqual(p,'Answer:');self.assertEqual(r,'\n yes')
    def test_original_delimiter_not_changed(self):
        s=copy.deepcopy(self.sample);s['requests'][0]['continuation']='x'
        with self.assertRaises(ValueError):validate_sample(s)
    def test_registry_two_kinds_no_dynamic_loader(self):
        self.assertEqual({get_task(t).request_kind for t in ('tech_record_extract','arc_challenge')},{'generate_until','choice_loglikelihood'})
        with self.assertRaises(ValueError):get_task('unknown')
    def test_registry_fake_collect_and_score_both_tasks(self):
        from inference_kit.adapters.task_executor import execute
        from inference_kit.tasks.tech_record import make_dataset,canonical_reference
        s=make_dataset()['rows'][0];a=get_task('tech_record_extract');req=a.requests(s|{'generation_input_ids':[1,2]},'toy','tok')[0]
        async def natural(r):return {'task_id':a.task_id,'task_version':a.task_version,'sample_id':s['sample_id'],'sample_identity':s['sample_identity'],'transport_complete':True,'termination':'NATURAL_STOP','output_text':canonical_reference(s)['text'],'request_latency_http':1.}
        result=asyncio.run(execute(req,natural_collector=natural));self.assertTrue(a.score(s,result)['task_success'])
        choices=[decode_record(req,raw(req)) for req in self.requests];self.assertEqual(self.adapter.score(self.sample,choices)['status'],'COMPLETE')
    def test_pool_deterministic_nonoverlap(self):
        pool=[sample(sid='s'+str(i)) for i in range(68)];a=select_pool(pool);b=select_pool(list(reversed(pool)));self.assertEqual(a,b)
    def test_plan_no_gpu_or_network(self):
        with tempfile.TemporaryDirectory() as d,patch('inference_kit.suites.bench05a_run.gpu_query',side_effect=AssertionError('GPU')):
            args=local_fixture(Path(d));p=plan(*args,Path(d)/'planned');self.assertEqual(p['counts']['total_posts'],544);self.assertEqual(p['gpu_actions'],0)
    def test_resource_gate_before_gpu_start_budget(self):
        from inference_kit.suites.bench05b3_run import run
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);args=local_fixture(d);s=read(args[0]);s['budget']['minimum_cpu_available_bytes']=10**30;args[0].write_text(json.dumps(s),encoding='utf-8');p=plan(*args,d/'planned');ledger=Path(p['campaign']['path']);before=sha(ledger)
            with patch('inference_kit.suites.bench05a_run.gpu_query',side_effect=AssertionError('GPU')),patch('inference_kit.adapters.owned.start',side_effect=AssertionError('start')):
                with self.assertRaises(ValueError):run(d/'planned/PLAN_FROZEN.json',d/'run')
            self.assertFalse((d/'run').exists());self.assertEqual(sha(ledger),before)
    def test_frozen_plan_source_and_input_changes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);args=local_fixture(d);p=plan(*args,d/'planned');p['source_identity']={};p.pop('plan_identity');p['plan_identity']=digest(p)
            with self.assertRaisesRegex(ValueError,'source/scorer'):check_frozen(p)
    def test_replay_partial_preserves_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);r=replay(synthetic_run(d/'raw',partial=True),d/'out');self.assertEqual(r['arms']['bf16']['REGRESSION']['planned_documents'],64);self.assertEqual(r['paired']['status'],'PARTIAL')
    def test_replay_complete_synthetic_not_live(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);r=replay(synthetic_run(d/'raw'),d/'out');self.assertEqual(r['arms']['bf16']['REGRESSION']['complete_documents'],64);self.assertEqual(r['evidence_kind'],'synthetic_contract');self.assertEqual(r['live_status'],'NOT_LIVE_SYNTHETIC_CONTRACT')
            with self.assertRaises(ValueError):replay(d/'raw',d/'out')
    def test_locale_cpu_replay_no_sdk(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);root=synthetic_run(d/'raw');site=str(Path(inference_kit.__file__).absolute().parent.parent)
            code='import sys,locale;sys.path.insert(0,'+repr(site)+');assert locale.getpreferredencoding(False).lower() not in ("utf-8","utf8");from inference_kit.api import bench_b3_replay;from inference_kit.cli import main;r=bench_b3_replay('+repr(str(root))+','+repr(str(d/'replay'))+');assert not any(x in sys.modules for x in ("torch","vllm","transformers","numpy","aiohttp"));raise SystemExit(main(["bench","b3","replay","--run",'+repr(str(root))+',"--output",'+repr(str(d/'cli'))+']))'
            p=subprocess.run([sys.executable,'-I','-X','utf8=0','-B','-c',code],cwd=d,env=dict(os.environ,LC_ALL='C',LANG='C'),capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr.decode('ascii'));self.assertEqual(json.loads(p.stdout)['evidence_kind'],'synthetic_contract');self.assertIn('文-',(d/'replay/bf16-document-scores.jsonl').read_text(encoding='utf-8'))

    def test_score_sample_mismatch_rejected(self):
        rows=self.scored([-1]*4);rows[0]['sample_id']='other'
        with self.assertRaises(ValueError):self.adapter.score(self.sample,rows)
    def test_missing_native_logprobs_incomplete_not_zero(self):
        r=raw(self.requests[0]);r['response']['choices'][0]['logprobs']['token_logprobs']=[];r['raw_response_text']=json.dumps(r['response'])
        value=decode_record(self.requests[0],r);self.assertEqual(value['status'],'INCOMPLETE');self.assertNotIn('choice_loglikelihood',value)
    def test_boundary_input_ids_rejected(self):
        req=self.requests[0];payload=copy.deepcopy(req.payload);payload['continuation_ids']=[99]
        with self.assertRaises(ValueError):dataclasses.replace(req,payload=payload).validate()
    def test_non_stream_latency_not_ttft(self):
        value=decode_record(self.requests[0],raw(self.requests[0]));self.assertAlmostEqual(value['request_latency_http'],.3);self.assertAlmostEqual(value['request_latency_task'],.5);self.assertNotIn('ttft',value)
    def test_replay_damaged_archive_member_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);root=synthetic_run(d/'raw');p=root/'progress.json';p.write_text('{}',encoding='utf-8')
            with self.assertRaises(ValueError):replay(root,d/'replay')
    def test_historical_cannot_be_promoted_to_native(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);root=synthetic_run(d/'raw');p=read(root/'PLAN_FROZEN.json');p.pop('plan_identity');p['evidence_kind']='planned_live_arc_regression';p['plan_identity']=digest(p)
            (root/'PLAN_FROZEN.json').write_text(json.dumps(p),encoding='utf-8');(root/'RAW_MANIFEST.json').unlink()
            from inference_kit.suites.bench05b1_run import seal
            seal(root,scope='bench05b3_arc_raw')
            with self.assertRaises(ValueError):replay(root,d/'replay')

    def test_missing_timing_unknown_not_zero(self):
        r=self.adapter.score(self.sample,self.scored([-1]*4));self.assertIsNone(r['document_serial_elapsed_seconds']);self.assertIsNone(r['document_choice_http_sum_seconds'])
    def test_shared_gpu_guard_preserves_budget_and_not_live(self):
        from inference_kit.suites.bench05b3_run import run
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);args=local_fixture(d);p=plan(*args,d/'planned');before=sha(Path(p['campaign']['path']))
            with patch('inference_kit.suites.bench05b1_run.check_gpu',return_value={'status':'BLOCKED_SHARED_GPU'}),patch('inference_kit.adapters.owned.start',side_effect=AssertionError('owned start forbidden')):
                status=run(d/'planned/PLAN_FROZEN.json',d/'run')
            r=replay(d/'run',d/'replay');self.assertEqual(r['live_status'],'NOT_RUN_OR_BLOCKED_NO_NATIVE_REQUESTS');self.assertIsNone(r['arms']['bf16']['REGRESSION']['observed_complete_acc_norm']);self.assertEqual(r['budget']['jobs'],[]);self.assertEqual(sha(Path(p['campaign']['path'])),before)
            linked=[json.loads(line) for line in (d/'replay/sample-linked.jsonl').read_text(encoding='utf-8').splitlines()];self.assertTrue(all(v is None for row in linked for v in row['native_raw_links'].values()))
    def test_typed_executor_collects_likelihood_without_scorer(self):
        from inference_kit.adapters.task_executor import execute
        request=self.requests[0];wire=raw(request)
        class Response:
            status=200
            async def __aenter__(self):return self
            async def __aexit__(self,*args):return False
            async def text(self,encoding):
                assert encoding=='utf-8'
                return wire['raw_response_text']
        class Session:
            def post(self,url,json,timeout):
                assert url=='http://127.0.0.1:18188/v1/completions' and json==body(request)
                return Response()
        value=asyncio.run(execute(request,session=Session(),url='http://127.0.0.1:18188'))
        decoded=decode_record(request,value);self.assertEqual(decoded['choice_loglikelihood'],-3);self.assertEqual(value['request_kind'],'choice_loglikelihood')

    def test_artifact_stat_change_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);args=local_fixture(d);p=plan(*args,d/'planned');weight=Path(p['arms'][0]['artifact']['path'])/'tiny.safetensors';weight.write_bytes(b'changed-synthetic')
            with self.assertRaisesRegex(ValueError,'artifact file stat'):check_frozen(p)

    def test_cancelled_request_submission_journal_survives(self):
        from inference_kit.adapters.task_executor import execute
        events=[]
        class Response:
            status=200
            async def __aenter__(self):return self
            async def __aexit__(self,*args):return False
            async def text(self,encoding):await asyncio.Future()
        class Session:
            def post(self,*args,**kwargs):return Response()
        async def exercise():
            task=asyncio.create_task(execute(self.requests[0],session=Session(),url='http://127.0.0.1',event_sink=events.append))
            await asyncio.sleep(0);task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        asyncio.run(exercise());self.assertEqual([e['event'] for e in events],['http_submit','response_end']);self.assertFalse(events[-1]['transport_complete']);self.assertEqual(events[-1]['error']['type'],'CancelledError')
    def test_journal_pending_choice_counted_incomplete(self):
        from inference_kit.suites.bench05b1_run import seal
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);root=synthetic_run(d/'raw',partial=True);directory=root/'attempts/serve-bf16-attempt1';path=directory/'http-responses.jsonl';rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()];events=[]
            for row in rows:
                events.append({k:v for k,v in row.items() if k in ('request_id','sample_id','choice_index','identity','request_body','request_body_identity','queued','semaphore_acquired','http_submit','arm_id','artifact_identity')}|{'event':'http_submit'})
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows[:-1]),encoding='utf-8');(directory/'request-journal.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in events),encoding='utf-8');(root/'RAW_MANIFEST.json').unlink();seal(root,scope='bench05b3_arc_raw')
            result=replay(root,d/'replay');q=result['arms']['bf16']['SMOKE'];self.assertEqual(q['submitted_posts'],4);self.assertEqual(q['complete_documents'],0);self.assertEqual(q['submitted_documents'],1);self.assertEqual(q['not_run_documents'],3)
