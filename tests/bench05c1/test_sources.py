"""CPU-only source adapters; tiny historical-shaped raw, never live models."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.analysis.token_sources import collect_source
from inference_kit.core.io import read,sha,digest
from inference_kit.suites.bench05b1_run import seal
from bench05b2.helpers import synthetic_run as b2_run
from bench05b3.helpers import synthetic_run as b3_run
from bench05b4.test_replay import synthetic_run as b4_run


def reseal(root,scope):
    (root/'RAW_MANIFEST.json').unlink();seal(root,scope=scope)


def replace_lines(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


class TokenSourceTests(unittest.TestCase):
    def sink(self):
        self.transferred=[]
        def copy(directory,metadata):
            self.transferred.append((directory,metadata));return dict(metadata,path='copied/'+metadata['sha256']+'.f32')
        return copy

    def spec(self,root,adapter,**extra):
        return {'source_id':'panel-'+adapter,'adapter':adapter,'path':str(root),**extra}

    def test_b4_targets_fields_full_blob_sink_and_identity(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw',probes=True)
            result=collect_source(self.spec(root,'bench05b4'),self.sink())
            self.assertEqual(len(result['sequences']),4)
            self.assertTrue(result['tokens']);self.assertTrue(self.transferred)
            self.assertTrue(any('answer' in row['field_tags'] for row in result['tokens']))
            self.assertTrue(any('evidence' in row['field_tags'] for row in result['tokens']))
            by_id={s['sequence_id']:s for s in result['sequences']}
            for token in result['tokens']:
                sequence=by_id[token['sequence_id']];position=token['token_position']
                self.assertEqual(token['logit_position'],position-1)
                self.assertEqual(token['target_token_id'],sequence['input_ids'][position])
                self.assertEqual(token['prefix_hash'],digest({'token_ids':sequence['input_ids'][:position],'token_id_space':token['token_id_space']}))
                self.assertEqual(token['evidence_kind'],'synthetic_contract')
            self.assertEqual(result['source']['raw_manifest_sha256'],sha(root/'RAW_MANIFEST.json'))

    def test_target_source_does_not_decode_attention_arrays(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw',probes=True)
            from inference_kit.analysis import token_sources
            original=token_sources.load;shapes=[]
            def scalar_only(directory,metadata):
                shapes.append(metadata['shape']);self.assertEqual(metadata['shape'],[1]);return original(directory,metadata)
            with patch.object(token_sources,'load',side_effect=scalar_only):
                collect_source(self.spec(root,'bench05b4'),self.sink())
            self.assertTrue(shapes)

    def test_filter_after_full_manifest_verification(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw',probes=True);p=read(root/'PLAN_FROZEN.json');sid=p['samples']['rows'][0]['sample_id']
            result=collect_source(self.spec(root,'bench05b4',sample_ids=[sid],splits=['EVAL_LOCKED']),self.sink())
            self.assertEqual({r['sample_id'] for r in result['tokens']},{sid})
            unselected=next(root.glob('attempts/serve-*/task-generations.jsonl'))
            unselected.write_text(unselected.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b4',sample_ids=[sid]),self.sink())

    def test_manifest_pin_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw')
            with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b4',raw_manifest_sha256='0'*64),self.sink())

    def test_plan_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw');plan=read(root/'PLAN_FROZEN.json');plan['task_version']='wrong'
            (root/'PLAN_FROZEN.json').write_text(json.dumps(plan),encoding='utf-8');reseal(root,'bench05b4_context_raw')
            with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b4'),self.sink())

    def test_wrong_adapter_and_policy_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw')
            for spec in (self.spec(root,'bench05b3'),self.spec(root,'bench05b4',policies=['P00']),self.spec(root,'unknown')):
                with self.subTest(adapter=spec['adapter']),self.assertRaises(ValueError):collect_source(spec,self.sink())

    def test_missing_diagnostic_remains_requested_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw');result=collect_source(self.spec(root,'bench05b4'),self.sink())
            self.assertGreater(result['source']['coverage']['diagnostic_requested'],0)
            self.assertEqual(result['source']['coverage']['diagnostic_measured'],0)
            self.assertTrue(all(r['target_logprob'] is None and r['status']=='NOT_RUN' for r in result['tokens']))

    def test_prefix_shift_space_and_duplicate_observations_rejected(self):
        for change in ('prefix','shift','space','duplicate'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                root=b4_run(Path(t)/'raw',probes=True);path=next(root.glob('attempts/main-*/observations.jsonl'))
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                target=next(r for r in rows if r['kind']=='target_logprob')
                if change=='prefix':target['coordinates']['prefix_hash']='invalid'
                if change=='shift':target['coordinates']['logit_position']+=1
                if change=='space':target['coordinates']['token_id_space']='different'
                if change=='duplicate':rows.append(target)
                replace_lines(path,rows);reseal(root,'bench05b4_context_raw')
                with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b4'),self.sink())

    def test_full_vocab_axis_and_claim_required(self):
        for key,value in (('shape',[255]),('full_vocabulary',False)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                root=b4_run(Path(t)/'raw',probes=True);path=next(root.glob('attempts/main-*/observations.jsonl'))
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()];full=next(r for r in rows if r['kind']=='full_logits')
                full['numeric' if key=='shape' else 'details'][key]=value
                replace_lines(path,rows);reseal(root,'bench05b4_context_raw')
                with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b4'),self.sink())

    def test_native_arc_choices_gold_scope_and_no_full_distribution(self):
        with tempfile.TemporaryDirectory() as t:
            root=b3_run(Path(t)/'raw');plan=read(root/'PLAN_FROZEN.json');sid=plan['data']['rows'][0]['sample_id']
            result=collect_source(self.spec(root,'bench05b3',sample_ids=[sid]),self.sink())
            native=[r for r in result['tokens'] if r['acquisition_source']=='native_likelihood']
            self.assertEqual(len(native),16);self.assertEqual({r['choice_index'] for r in native},{0,1,2,3})
            self.assertEqual({r['reference_scope'] for r in native},{'GOLD','NON_GOLD'})
            self.assertEqual({r['split'] for r in native},{'SMOKE'})
            self.assertTrue(all(r['target_logprob'] in (-1.,-2.) for r in native))
            self.assertTrue(all(r['full_logits'] is None and r['audit_generated_token_excluded'] for r in native))
            self.assertFalse(self.transferred)

    def test_native_disabled_and_split_filter(self):
        with tempfile.TemporaryDirectory() as t:
            root=b3_run(Path(t)/'raw');result=collect_source(self.spec(root,'bench05b3',include_native=False),self.sink())
            self.assertFalse(result['tokens'])
            result=collect_source(self.spec(root,'bench05b3',splits=['REGRESSION']),self.sink())
            self.assertTrue(result['tokens']);self.assertEqual({r['split'] for r in result['tokens']},{'REGRESSION'})

    def test_native_raw_identity_and_echo_mismatch_rejected_or_incomplete(self):
        with tempfile.TemporaryDirectory() as t:
            root=b3_run(Path(t)/'raw');path=next(root.glob('attempts/serve-*/http-responses.jsonl'))
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            rows[0]['identity']['tokenizer_identity']='different';replace_lines(path,rows);reseal(root,'bench05b3_arc_raw')
            with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b3'),self.sink())

    def test_native_partial_does_not_manufacture_missing_logprobs(self):
        with tempfile.TemporaryDirectory() as t:
            root=b3_run(Path(t)/'raw',partial=True);result=collect_source(self.spec(root,'bench05b3'),self.sink())
            coverage=result['source']['coverage'];self.assertGreater(coverage['native_requested'],coverage['native_measured'])
            missing=[r for r in result['tokens'] if r['status']=='NOT_RUN']
            self.assertTrue(missing);self.assertTrue(all(r['target_logprob'] is None for r in missing))

    def test_b2_missing_probe_keeps_both_policies_distinct(self):
        with tempfile.TemporaryDirectory() as t:
            root=b2_run(Path(t)/'raw');result=collect_source(self.spec(root,'bench05b2',policies=['P00','P11']),self.sink())
            self.assertEqual({r['policy_id'] for r in result['tokens']},{'P00','P11'})
            self.assertEqual(len(result['sequences']),8)
            self.assertTrue(all(r['status']=='NOT_RUN' for r in result['tokens']))
            with self.assertRaises(ValueError):collect_source(self.spec(root,'bench05b2',policies=['P10']),self.sink())

    def test_global_record_ids_bind_bundle_and_survive_relocation(self):
        for adapter,make,scope in (('bench05b4',lambda p:b4_run(p,probes=True),'bench05b4_context_raw'),
                                    ('bench05b3',b3_run,'bench05b3_arc_raw')):
            with self.subTest(adapter=adapter),tempfile.TemporaryDirectory() as t:
                base=Path(t);root=make(base/'raw');spec=self.spec(root,adapter)
                first=collect_source(spec,self.sink())
                moved=base/'relocated';shutil.copytree(root,moved)
                relocated=collect_source(dict(spec,path=str(moved)),self.sink())
                self.assertEqual([r['record_id'] for r in first['tokens']],[r['record_id'] for r in relocated['tokens']])
                progress=read(moved/'progress.json');progress['raw_copy_metadata']='different-raw-bundle-same-local-ids'
                (moved/'progress.json').write_text(json.dumps(progress),encoding='utf-8');reseal(moved,scope)
                changed=collect_source(dict(spec,path=str(moved)),self.sink())
                self.assertEqual(first['source']['plan_identity'],changed['source']['plan_identity'])
                self.assertEqual([r['observation_id'] for r in first['tokens']],[r['observation_id'] for r in changed['tokens']])
                self.assertTrue(set(r['record_id'] for r in first['tokens']).isdisjoint(r['record_id'] for r in changed['tokens']))

    def test_reference_roles_offsets_and_requested_full_coverage_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw',probes=True);p=read(root/'PLAN_FROZEN.json')
            result=collect_source(self.spec(root,'bench05b4'),self.sink())
            self.assertEqual(result['source']['coverage']['full_vocab_requested'],sum(len(r['full_vocab_target_positions']) for r in p['samples']['rows'])*2)
            for row in result['sequences']:
                self.assertEqual(row['reference_source'],'task_author_gold_answer_evidence_v1')
                self.assertEqual(row['reference_role'],'AUTHOR_GOLD_REFERENCE')
                self.assertEqual(row['offset_mapping_status'],'AVAILABLE');self.assertTrue(row['field_spans'])
                self.assertEqual(row['field_spans_coordinate_space'],'reference_codepoints')
        with tempfile.TemporaryDirectory() as t:
            root=b3_run(Path(t)/'raw');result=collect_source(self.spec(root,'bench05b3'),self.sink())
            self.assertEqual({r['reference_role'] for r in result['sequences']},{'GOLD','NON_GOLD'})
            self.assertTrue(all(r['offset_mapping_status']=='NOT_COLLECTED' and r['offsets_codepoints'] is None for r in result['sequences']))
            self.assertTrue(all(r['native_argmax_status']=='NOT_COLLECTED_ACTUAL_ARGMAX' for r in result['tokens']))

    def test_unsupported_full_observation_is_missing_not_zero_or_failure(self):
        with tempfile.TemporaryDirectory() as t:
            root=b4_run(Path(t)/'raw',probes=True)
            first_arm=read(root/'PLAN_FROZEN.json')['arms'][0]['arm_id']
            self.assertEqual(first_arm,'bf16')
            path=root/read(root/'progress.json')['jobs']['main-'+first_arm]['directory']/'observations.jsonl'
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            full=next(r for r in rows if r['kind']=='full_logits');full.update(status='UNSUPPORTED',numeric=None,details={})
            replace_lines(path,rows);reseal(root,'bench05b4_context_raw')
            result=collect_source(self.spec(root,'bench05b4'),self.sink())
            coverage=result['source']['coverage'];self.assertEqual(coverage['full_vocab_requested'],coverage['full_vocab_measured']+1)
            record=next(r for r in result['tokens'] if r['observation_id'] and r['sample_id']==full['sample_id'] and r['token_position']==full['coordinates']['token_position'] and r['arm_id']=='bf16')
            self.assertIsNone(record['full_logits']);self.assertEqual(record['status'],'MEASURED')


if __name__=='__main__':unittest.main()
