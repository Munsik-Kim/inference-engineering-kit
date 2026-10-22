"""C1 free-generation diagnostics on saved streams, no model/tokenizer runtime."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

from inference_kit.core.io import digest
from inference_kit.metrics.generation_compare import compare_generation
from inference_kit.analysis.generation_sources import _server_ids, collect_generations


def row(tokens=None, text='same', **kwargs):
    values={'record_id':'left','record_status':'OBSERVED','task_id':'test','task_version':'1','sample_id':'s1','sample_identity':'sample',
      'split':'EVAL_LOCKED','policy_id':'FIXED','prompt_hash':'prompt','tokenizer_identity':'tokenizer','token_id_space':'vocab',
      'runtime_identity':'runtime','scorer_identity':'scorer','document_hash':'document','gold_identity':'gold','generation_policy_identity':'policy',
      'output_token_ids':tokens,'output_token_provenance':'RAW_SSE_SERVER_TOKEN_IDS' if tokens is not None else 'NOT_COLLECTED',
      'output_text':text,'transport_complete':True,'termination':'NATURAL_STOP','task_success':False,'artifact_identity':'bf16'}
    values.update(kwargs);return values


class GenerationMetricTests(unittest.TestCase):
    def test_identity_match_and_zero_difference(self):
        a=row([1,2,3]);b=row([1,2,3],record_id='right',artifact_identity='rtn')
        v=compare_generation(a,b)
        self.assertEqual(v['status'],'COMPLETE_OBSERVED_PAIR');self.assertTrue(v['full_output_token_match'])
        self.assertEqual(v['longest_common_prefix_tokens'],3);self.assertIsNone(v['first_different_token_position'])
        self.assertEqual(v['paired_task_outcome'],'both_wrong');self.assertFalse(v['same_output_is_task_accuracy'])

    def test_actual_first_token_divergence(self):
        v=compare_generation(row([1,2,3]),row([1,8,9],record_id='right'))
        self.assertEqual(v['longest_common_prefix_tokens'],1)
        self.assertEqual((v['first_different_token_position'],v['first_different_left_token_id'],v['first_different_right_token_id']),(1,2,8))
        self.assertFalse(v['full_output_token_match'])
        self.assertIn('no teacher-forced',v['comparison_scope'])

    def test_prefix_exhaustion_does_not_fabricate_token_or_eos(self):
        v=compare_generation(row([1,2]),row([1,2,9]))
        self.assertEqual(v['longest_common_prefix_tokens'],2);self.assertEqual(v['first_length_divergence_position'],2)
        self.assertIsNone(v['first_different_token_position']);self.assertIsNone(v['first_different_left_token_id'])
        self.assertFalse(v['full_output_token_match'])

    def test_text_only_never_retokenizes(self):
        v=compare_generation(row(None,text='한글'),row(None,text='한글'))
        self.assertEqual(v['status'],'TEXT_ONLY');self.assertTrue(v['text_match'])
        self.assertIsNone(v['full_output_token_match']);self.assertIsNone(v['longest_common_prefix_tokens'])
        self.assertEqual(v['left_text_codepoints'],2)

    def test_one_missing_token_axis_text_only(self):
        v=compare_generation(row([1,2]),row(None))
        self.assertEqual(v['status'],'TEXT_ONLY');self.assertEqual(v['left_actual_output_token_length'],2)
        self.assertIsNone(v['right_actual_output_token_length'])

    def test_cap_full_match_unknown_observed_prefix_retained(self):
        a=row([1,2],termination='LENGTH_CAP');b=row([1,2])
        v=compare_generation(a,b)
        self.assertEqual(v['status'],'CENSORED_COMPARISON');self.assertTrue(v['observed_token_sequence_match'])
        self.assertIsNone(v['full_output_token_match']);self.assertIsNone(v['text_match'])
        self.assertEqual(v['left_censoring'],'LENGTH_CAP_CENSORED')

    def test_eof_censored_not_eos_or_both_wrong(self):
        a=row([1,2],termination='TRUNCATED_OR_EOF',transport_complete=False)
        v=compare_generation(a,row([1,2]))
        self.assertEqual(v['status'],'CENSORED_COMPARISON');self.assertIsNone(v['paired_task_outcome'])
        self.assertEqual(v['left_censoring'],'EOF_OR_INCOMPLETE_CENSORED')

    def test_not_run_and_missing_pair_unknown(self):
        v=compare_generation(row(None,record_status='NOT_RUN',termination='NOT_RUN',transport_complete=False,task_success=None,output_text=None),row([1]))
        self.assertIsNone(v['paired_task_outcome']);self.assertIsNone(v['observed_text_match'])
        self.assertEqual(compare_generation(None,row([1]))['status'],'MISSING_PAIR')

    def test_each_comparison_axis_required(self):
        for key in ('task_id','task_version','sample_id','sample_identity','split','policy_id','prompt_hash','tokenizer_identity','token_id_space','runtime_identity','scorer_identity','document_hash','gold_identity','generation_policy_identity'):
            with self.subTest(key=key):
                a=row([1]);b=row([1],**{key:'different'})
                v=compare_generation(a,b);self.assertEqual(v['status'],'INCOMPATIBLE');self.assertIn('IDENTITY_MISMATCH:'+key,v['incompatible_reasons'])
                self.assertIsNone(v['full_output_token_match'])

    def test_missing_identity_not_equal_unknowns(self):
        a=row([1]);b=row([1]);a.pop('token_id_space');b.pop('token_id_space')
        v=compare_generation(a,b);self.assertEqual(v['status'],'INCOMPATIBLE');self.assertIn('MISSING_IDENTITY:token_id_space',v['incompatible_reasons'])

    def test_retokenized_or_negative_tokens_rejected(self):
        for a in (row([1],output_token_provenance='RETOKENIZED'),row([-1]),row([True])):
            with self.assertRaises(ValueError):compare_generation(a,row([1]))

    def test_task_gain_loss_from_scorer_not_text(self):
        self.assertEqual(compare_generation(row([1],task_success=False),row([1],task_success=True))['paired_task_outcome'],'gain')
        self.assertEqual(compare_generation(row([1],task_success=True),row([2],task_success=False))['paired_task_outcome'],'loss')
        self.assertEqual(compare_generation(row([1],task_success=True),row([2],task_success=True))['paired_task_outcome'],'both_correct')
        self.assertTrue(compare_generation(row([1],task_success=False),row([1],task_success=False))['full_output_token_match'])

    def test_partial_text_only(self):
        v=compare_generation(row(None,termination='HTTP_ERROR',transport_complete=False),row(None))
        self.assertEqual(v['status'],'TEXT_ONLY_CENSORED');self.assertIsNone(v['text_match']);self.assertIsNone(v['paired_task_outcome'])

    def test_empty_actual_generation_is_length_zero_not_missing(self):
        v=compare_generation(row([],text=''),row([],text=''))
        self.assertEqual(v['left_actual_output_token_length'],0);self.assertEqual(v['longest_common_prefix_tokens'],0)
        self.assertTrue(v['full_output_token_match'])


class GenerationSourceTests(unittest.TestCase):
    def test_saved_ids_match_raw_server_ids(self):
        record={'raw_sse_frames':[json.dumps({'choices':[{'text':'x','token_ids':[4,5]}]}),'[DONE]'],'output_token_ids':[4,5]}
        self.assertEqual(_server_ids(record),([4,5],'RAW_SSE_SERVER_TOKEN_IDS'))
        record['output_token_ids']=[4,6]
        with self.assertRaisesRegex(ValueError,'differ'):_server_ids(record)

    def test_text_frames_without_ids_remain_text_only(self):
        self.assertEqual(_server_ids({'raw_sse_frames':[json.dumps({'choices':[{'text':'x'}]}),'[DONE]']}),(None,'NOT_COLLECTED'))

    def test_stored_server_ids_without_frames_explicit_provenance(self):
        self.assertEqual(_server_ids({'output_token_ids':[1,2]}),([1,2],'STORED_SERVER_TOKEN_IDS'))

    def test_invalid_id_types_rejected(self):
        for ids in ([True],[-1],['1']):
            with self.assertRaises(ValueError):_server_ids({'output_token_ids':ids})

    def test_b3_audit_rejected_as_free_generation(self):
        with self.assertRaisesRegex(ValueError,'audit'):
            collect_generations({'adapter':'bench05b3','source_id':'b3','path':'unused'}, {}, 'id')

    def test_source_comparison_trace(self):
        a=row([1],source_id='b4',arm_id='bf16',bundle_identity='bundle')
        b=row([2],source_id='b4',arm_id='rtn',bundle_identity='bundle')
        result=compare_generation(a,b)
        self.assertEqual(result['source_id'],'b4');self.assertEqual(result['sample_id'],'s1')
        self.assertEqual(result['policy_id'],'FIXED');self.assertEqual(result['left_arm_id'],'bf16')
        self.assertEqual(result['right_arm_id'],'rtn')

    def test_missing_source_id_rejected(self):
        with self.assertRaisesRegex(ValueError,'source ID'):
            collect_generations({'adapter':'bench05b4','path':'unused'}, {}, 'id')


class GenerationSavedSourceTests(unittest.TestCase):
    def test_b4_source_preserves_actual_tokens_score_scope_and_unique_ids(self):
        from bench05b4.test_replay import synthetic_run
        from inference_kit.core.io import read
        with tempfile.TemporaryDirectory() as tmp:
            root=synthetic_run(Path(tmp)/'raw');plan=read(root/'PLAN_FROZEN.json')
            spec={'adapter':'bench05b4','path':str(root),'source_id':'source-한글'}
            records=collect_generations(spec,plan,{'plan':plan['plan_identity']})
            self.assertEqual(len(records),48);self.assertEqual(len({r['record_id'] for r in records}),48)
            self.assertTrue(all(r['evidence_kind']=='synthetic_contract' for r in records))
            self.assertTrue(all(r['task_success'] is True for r in records))
            self.assertTrue(all(r['output_token_ids']==[11,12] for r in records))
            self.assertTrue(all(r['output_token_provenance']=='RAW_SSE_SERVER_TOKEN_IDS' for r in records))
            self.assertTrue(all(r['raw_manifest_verified'] for r in records))
            other=collect_generations(dict(spec,source_id='another'),plan,{'plan':plan['plan_identity']})
            self.assertFalse({r['record_id'] for r in records}&{r['record_id'] for r in other})
            relocated=collect_generations(spec,plan,{'plan':plan['plan_identity'],'input_path_as_metadata':'/different/location'})
            self.assertEqual([r['record_id'] for r in records],[r['record_id'] for r in relocated])
            self.assertEqual([r['bundle_identity'] for r in records],[r['bundle_identity'] for r in relocated])
            sid=next(r['sample_id'] for r in plan['data']['rows'] if r['split']=='EVAL_LOCKED')
            selected=collect_generations(dict(spec,sample_ids=[sid]),plan,'identity')
            self.assertEqual(len(selected),2);self.assertEqual({r['sample_id'] for r in selected},{sid})
            for invalid in ([sid,sid],[],['unknown']):
                with self.assertRaises(ValueError):collect_generations(dict(spec,sample_ids=invalid),plan,'identity')

    def test_not_run_and_journal_incomplete_never_become_both_wrong(self):
        from bench05b4.test_replay import synthetic_run
        from inference_kit.core.io import read
        with tempfile.TemporaryDirectory() as tmp:
            root=synthetic_run(Path(tmp)/'raw',partial=True);plan=read(root/'PLAN_FROZEN.json')
            spec={'adapter':'bench05b4','path':str(root),'source_id':'partial','splits':['SMOKE']}
            records=collect_generations(spec,plan,'source-identity')
            self.assertEqual(len(records),8)
            submitted=[r for r in records if r['termination']=='INCOMPLETE_SUBMISSION']
            self.assertEqual(len(submitted),2)
            self.assertTrue(all('request-journal' in r['raw_reference'] for r in submitted))
            self.assertTrue(all(r['output_token_ids'] is None for r in submitted))
            self.assertTrue(all(r['request_latency_http'] is None for r in submitted))
            a,b=submitted
            self.assertIsNone(compare_generation(a,b)['paired_task_outcome'])
            missing=[r for r in records if r['record_status']=='NOT_RUN']
            self.assertEqual(len(missing),6);self.assertTrue(all(r['task_success'] is None for r in missing))

    def test_actual_token_ids_outside_saved_vocabulary_rejected(self):
        from bench05b4.test_replay import synthetic_run, reseal
        from inference_kit.core.io import read
        with tempfile.TemporaryDirectory() as tmp:
            root=synthetic_run(Path(tmp)/'raw');plan=read(root/'PLAN_FROZEN.json')
            path=next(root.rglob('task-generations.jsonl'))
            rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
            value=next(r for r in rows if r['split']=='EVAL_LOCKED')
            frame=json.loads(value['raw_sse_frames'][0]);frame['choices'][0]['token_ids']=[plan['samples']['vocabulary_size'],12]
            value['raw_sse_frames'][0]=json.dumps(frame,ensure_ascii=False)
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');reseal(root)
            with self.assertRaisesRegex(ValueError,'outside declared vocabulary'):
                collect_generations({'adapter':'bench05b4','path':str(root),'source_id':'vocab'},plan,'source')

    def test_raw_corruption_and_plan_ancestry_mismatch_fail_closed(self):
        from bench05b4.test_replay import synthetic_run
        from inference_kit.core.io import read
        with tempfile.TemporaryDirectory() as tmp:
            root=synthetic_run(Path(tmp)/'raw');plan=read(root/'PLAN_FROZEN.json')
            spec={'adapter':'bench05b4','path':str(root),'source_id':'source'}
            wrong=copy.deepcopy(plan);wrong['plan_identity']='wrong'
            with self.assertRaisesRegex(ValueError,'plan mismatch'):
                collect_generations(spec,wrong,'id')
            path=next(root.rglob('task-generations.jsonl'))
            path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'raw identity'):
                collect_generations(spec,plan,'id')


if __name__=='__main__':unittest.main()
