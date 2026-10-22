"""Serving-only native scalar analysis; all requests are CPU fixture bytes."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from inference_kit.analysis import suite_results
from inference_kit.analysis.tokens import compare_tokens, jsonl, write_jsonl
from inference_kit.core.io import digest, read, sha, write
from inference_kit.suites.suite_plan import compile_suite, NATIVE_TOKEN_METRIC
from bench05e2.test_plan import local_fixture
from bench05e3.test_results import seal
from bench05e4a.test_suite_prepared import prepared_suite
from bench05e4b.test_results import fixture as prepared_fixture


def serving_fixture(root, count=1):
    """Reuse typed request production; remove the fixture's diagnostic branch."""
    prepared_fixture(root, count)
    context=read(root/'suite-execution-context.json');sources=context['sources']
    spec=prepared_suite(sources,'serve');spec['metrics'].append(NATIVE_TOKEN_METRIC)
    compiled=compile_suite(spec,sources,local_fixture())
    for group,job in context['serving_jobs'].items():
        source=next(s for s in sources['sources'] if s['source_id']==job['source_id'])
        rows={s['sample_id']:s for s in source['samples']}
        prefix=compiled['plan_identity']+'|synthetic-native|attempt1|'+job['source_arm_id']+'|'+group
        for key in ('raw_path','journal_path'):
            path=root/job[key];records=list(jsonl(path))
            for record in records:
                requests=suite_results._requests(source,rows[record['sample_id']],prefix,sources['model_metadata'])
                request=requests[record.get('choice_index',0)]
                record['request_id']=request.request_id;record['execution_prefix']=prefix
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    context.pop('diagnostic_run');context['suite_plan']=compiled
    context['binding_profile']='prepared_qwen_serve_v1'
    context['token_analysis']={'source':'native_choice_likelihood_v1','diagnostic':'NOT_REQUESTED','full_vocabulary':'NOT_COLLECTED'}
    (root/'suite-execution-context.json').write_text(json.dumps(context),encoding='utf-8')
    (root/'SUITE_PLAN.json').write_text(json.dumps(compiled),encoding='utf-8')
    shutil.rmtree(root/'acquisitions/token')
    seal(root);return root


class NativeSuiteResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.run=serving_fixture(self.root/'run')

    def test_serving_only_exports_all_choice_scalars_into_existing_c1(self):
        result=suite_results.analyze_tokens(self.run,self.root/'tokens')
        rows=list(jsonl(self.root/'tokens/token-view/token_records.jsonl'))
        context=read(self.run/'suite-execution-context.json')
        arc=next(s for s in context['sources']['sources'] if s['request_kind']=='choice_loglikelihood')
        expected=sum(len(r['continuation_ids']) for s in arc['samples'] for r in s['authored_sample']['requests'])
        self.assertEqual(result['coverage']['requested_pair_coordinates'],expected)
        self.assertEqual(result['coverage']['measured'],expected);self.assertEqual(len(rows),expected*2)
        self.assertTrue(all(r['acquisition_source']=='native_likelihood' and r['full_logits'] is None for r in rows))
        self.assertTrue(all(r['audit_generated_token_excluded'] for r in rows))
        comparisons=list(jsonl(self.root/'tokens/token-metrics/token-comparisons.jsonl'))
        self.assertTrue(all(r['metrics']['delta_nll_nat']==0 for r in comparisons))
        self.assertTrue(all(r['metrics']['full_distribution']['reason']=='FULL_VOCABULARY_NOT_COLLECTED' for r in comparisons))
        groups=read(self.root/'tokens/token-metrics/aggregates.json')['groups']
        self.assertEqual({g['identity']['reference_scope'] for g in groups},{'GOLD_ONLY','NON_GOLD','ALL_CHOICES'})

    def test_native_serving_arm_does_not_claim_unsupported_diagnostic_path(self):
        from bench06a.test_plan import qwen_fixture
        with tempfile.TemporaryDirectory() as tmp:
            spec,sources=qwen_fixture(Path(tmp));compiled=compile_suite(spec,sources,local_fixture())
            arms=suite_results._arms(sources,compiled)
            self.assertTrue(all(a['execution_mode']=='NATIVE_SERVING' for a in arms.values()))
            self.assertTrue(all(m['diagnostic_path']=='UNSUPPORTED_CAPABILITY' for m in compiled['semantic']['models']))
            changed=deepcopy(sources);changed['model_metadata']['arms'][0]['execution_mode']='INSTRUMENTED_PATH'
            with self.assertRaisesRegex(ValueError,'frozen native path'):suite_results._arms(changed,compiled)

    def test_native_probability_axis_is_model_vocabulary_not_tokenizer_length(self):
        from inference_kit.analysis.suite_native_source import _vocabulary_axis
        self.assertEqual(_vocabulary_axis({'vocabulary_size':151669,'embedding_vocabulary_size':151936,
            'model_config':{'vocab_size':151936}}),(151669,151936))
        with self.assertRaisesRegex(ValueError,'model vocabulary axis differs'):
            _vocabulary_axis({'vocabulary_size':151669,'embedding_vocabulary_size':151669,'model_config':{'vocab_size':151936}})
        with self.assertRaisesRegex(ValueError,'vocabulary axes'):
            _vocabulary_axis({'vocabulary_size':151936,'embedding_vocabulary_size':151669,'model_config':{}})

    def test_nonempty_native_analysis_and_no_diagnostic_acquisitions_in_index(self):
        from inference_kit.analysis.result_index import _create,add_sources,canonical_export
        db=self.root/'index.sqlite';_create(db);source=self.root/'sources.json'
        write(source,{'schema_version':1,'sources':[suite_results.index_source_spec(self.run)]})
        first=add_sources(source,db,self.root/'registered');second=add_sources(source,db,self.root/'registered-again')
        self.assertEqual(first['after']['acquisitions'],6);self.assertEqual(second['before'],second['after'])
        snapshot=canonical_export(db)
        self.assertTrue(all(a['payload']['family']=='serve' for a in snapshot['acquisitions']))
        metrics=snapshot['measurements'];paired=[r for r in metrics if r['metric']=='token.delta_nll_nat']
        self.assertTrue(paired);self.assertTrue(all(r['coordinates']['pass_id']=='NATIVE_CHOICE_LIKELIHOOD' for r in paired))
        self.assertTrue(all(r['evidence_ids'] for r in paired))
        ranks=[r for r in metrics if r['metric']=='token.reference_target_rank']
        self.assertTrue(ranks);self.assertTrue(all(r['value'] is None and r['status']=='NOT_COLLECTED' and
            r['payload']['unavailable_reason']=='FULL_VOCABULARY_NOT_COLLECTED' for r in ranks))
        self.assertTrue(all(r['status']=='MEASURED' for r in paired))
        rebuilt=self.root/'rebuilt.sqlite';_create(rebuilt);add_sources(source,rebuilt,self.root/'rebuilt')
        self.assertEqual(canonical_export(db)['logical_identity'],canonical_export(rebuilt)['logical_identity'])

    def test_missing_choice_preserves_unknown_tokens_and_original_task_denominator(self):
        context=read(self.run/'suite-execution-context.json')
        job=next(j for j in context['serving_jobs'].values() if j['logical_arm_id']=='candidate' and j['task_id']=='arc_challenge')
        path=self.run/job['raw_path'];rows=list(jsonl(path));missing=rows.pop(0);path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');seal(self.run)
        result=suite_results.analyze_tokens(self.run,self.root/'tokens')
        self.assertGreater(result['coverage']['unknown'],0);self.assertEqual(result['coverage']['incomparable'],0)
        scores=list(jsonl(self.root/'tokens/native-source/document-scores.jsonl'))
        arc=next(r['score'] for r in scores if r['arm_id']=='candidate' and r['task_id']=='arc_challenge')
        self.assertEqual(arc['status'],'INCOMPLETE');self.assertIsNone(arc['acc']);self.assertEqual(arc['planned_choices'],4)
        native=[r for r in jsonl(self.root/'tokens/token-view/token_records.jsonl') if r['arm_id']=='rtn' and r['choice_index']==missing['choice_index']]
        self.assertTrue(native);self.assertTrue(all(r['target_logprob'] is None for r in native))

    def test_serving_only_contract_cannot_hide_a_diagnostic_graph_or_run(self):
        context=read(self.run/'suite-execution-context.json');context['diagnostic_run']='omitted'
        (self.run/'suite-execution-context.json').write_text(json.dumps(context),encoding='utf-8');seal(self.run)
        with self.assertRaisesRegex(ValueError,'cannot carry a diagnostic'):suite_results.analyze_tokens(self.run,self.root/'bad')

    def test_full_pipeline_is_product_replay_with_exact_serving_unit_count(self):
        result=suite_results.replay(self.run,self.root/'replay')
        self.assertEqual(result['status'],'CPU_RAW_REPLAYED')
        self.assertTrue(all(r['status']=='COMPLETE' for r in result['results'].values()))
        self.assertEqual(result['results']['index_add']['result']['after']['acquisitions'],6)
        self.assertTrue((self.root/'replay/report/index.html').is_file())
        self.assertEqual(result['inference_requests'],0)

    def test_native_mini_is_lossless_declared_slice_and_cannot_register_parent_acquisition(self):
        suite_results.analyze_tokens(self.run,self.root/'tokens');target=self.root/'mini'
        receipt=suite_results.export_mini_slice(self.run,self.root/'tokens/token-view',target)
        self.assertEqual(receipt['kind'],'suite_native_task_slice_v1');self.assertEqual(receipt['coverage']['full_vectors'],0)
        context=read(self.run/'suite-execution-context.json')
        for job in context['serving_jobs'].values():
            for key in ('raw_path','journal_path'):self.assertEqual((self.run/job[key]).read_bytes(),(target/job[key]).read_bytes())
        portable=read(target/'suite-execution-context.json');self.assertNotIn('runtime_plan',portable)
        self.assertNotIn('bindings',portable['suite_plan']);self.assertNotIn(str(self.root),json.dumps(portable))
        replay=suite_results.replay(target,self.root/'mini-replay')
        self.assertEqual(replay['status'],'CPU_SLICE_REPLAYED');self.assertEqual(replay['full_vectors'],0)
        self.assertEqual(replay['native_scalar_rows'],receipt['coverage']['native_scalar_rows'])
        before=list(jsonl(self.root/'tokens/token-metrics/token-comparisons.jsonl'))
        after=list(jsonl(self.root/'mini-replay/analyze_tokens/token-metrics/token-comparisons.jsonl'))
        self.assertEqual([r['metrics'] for r in before],[r['metrics'] for r in after])
        self.assertFalse((self.root/'mini-replay/index_add').exists())
        with self.assertRaisesRegex(ValueError,'SLICE_ONLY'):
            suite_results.load_index_source(suite_results.index_source_spec(target),self.root)

    def test_native_mini_missing_or_changed_members_and_ancestry_are_refused(self):
        suite_results.analyze_tokens(self.run,self.root/'tokens');target=self.root/'mini'
        suite_results.export_mini_slice(self.run,self.root/'tokens/token-view',target)
        path=target/'SLICE_RECEIPT.json';receipt=read(path);receipt['included_request_ids'].pop()
        path.write_text(json.dumps(receipt),encoding='utf-8');seal(target)
        # Resealing a test-only inventory does not authorize a different request
        # set: the explicit ancestry/request contract remains exact.
        manifest=read(target/'RUN_MANIFEST.json');manifest['scope']='suite_e3_task_slice_v1';manifest['derived_slice_kind']='suite_native_task_slice_v1'
        (target/'RUN_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'request coverage'):suite_results.replay(target,self.root/'bad-mini')

    def test_native_pair_does_not_make_cross_tokenizer_or_prefix_comparisons(self):
        suite_results.analyze_tokens(self.run,self.root/'tokens');view=self.root/'tokens/token-view'
        rows=list(jsonl(view/'token_records.jsonl'));row=next(r for r in rows if r['arm_id']=='rtn')
        row['tokenizer_identity']='f'*64
        path=view/'token_records.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        manifest=read(view/'TOKEN_VIEW_MANIFEST.json');entry=manifest['files']['token_records.jsonl'];entry.update(bytes=path.stat().st_size,sha256=sha(path));manifest['view_identity']=digest(manifest['files'])
        (view/'TOKEN_VIEW_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
        result=compare_tokens(view,'bf16','rtn',self.root/'cross-tokenizer')
        self.assertEqual(result['coverage']['incomparable'],1)
        self.assertIn('MISMATCH_TOKENIZER_IDENTITY',result['compatibility']['reasons'])

    def test_task_request_and_native_coordinate_content_remain_strict(self):
        context=read(self.run/'suite-execution-context.json');job=next(j for j in context['serving_jobs'].values() if j['task_id']=='arc_challenge')
        path=self.run/job['raw_path'];rows=list(jsonl(path));rows[0]['identity']['token_id_space']='f'*64
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');seal(self.run)
        with self.assertRaisesRegex(ValueError,'typed request'):suite_results.analyze_tokens(self.run,self.root/'bad')

    def test_native_counts_scale_from_actual_choices_not_smol_diagnostic_selection(self):
        other=self.root/'other';other.mkdir();run=serving_fixture(other/'run',3)
        a=suite_results.analyze_tokens(self.run,self.root/'one');b=suite_results.analyze_tokens(run,self.root/'three')
        self.assertEqual(b['coverage']['measured'],3*a['coverage']['measured'])
        self.assertNotEqual(b['coverage']['measured'],123)
