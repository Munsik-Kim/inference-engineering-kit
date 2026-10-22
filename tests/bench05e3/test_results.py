"""Closed E3 HTTP/raw replay controls; no model, GPU, tokenizer or server."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.adapters.http import sha as wire_sha
from inference_kit.adapters.likelihood import request_body
from inference_kit.adapters.protocol import Adapter
from inference_kit.analysis import suite_results
from inference_kit.analysis.tokens import jsonl, write_jsonl
from inference_kit.core.io import digest, read, sha, write
from inference_kit.suites import suite_plan
from inference_kit.tasks.registry import get_task
from inference_kit.metrics.token import log_softmax
from inference_kit.probes.blobs import save as save_blob
from bench05e2.test_inputs import repin_row
from bench05e2.test_plan import sources_fixture, suite_fixture, local_fixture, _proof


def seal(root):
    path=root/'RUN_MANIFEST.json'
    if path.exists():path.unlink()  # Only this test's own mutable construction fixture.
    files={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in root.rglob('*') if p.is_file() and 'analysis' not in p.relative_to(root).parts}
    write(path,{'schema_version':1,'scope':'suite_execution_raw_v1','files':files})


def fixture(root):
    root.mkdir();inputs=root.parent/'input-fixture';inputs.mkdir();sources=sources_fixture(inputs)
    for source in sources['sources']:
        if source['request_kind']!='generate_until':
            for index,sample in enumerate(source['samples']):
                sample['authored_sample'].update(original_index=index,request_identity=digest(sample['authored_sample']['requests']))
            continue
        for sample in source['samples']:
            authored=sample['authored_sample']
            if source['task_id']=='tech_record_extract':
                authored.update(events=[],split='DEV_LOCKED',language='en',case_type='missing')
                authored['sample_identity']=digest({k:v for k,v in authored.items() if k!='sample_identity'})
                sample['diagnostic']['sample_identity']=authored['sample_identity'];repin_row(sample['diagnostic'])
            row=sample['diagnostic'];sample['generation']['prepared']=deepcopy(row)
            sample['generation']['wire_payload']=Adapter('vllm').body({'token_ids':row['generation_input_ids']},'serving-lab',
                192 if source['task_id']=='tech_record_extract' else 64,'natural_generation_v1')
            _proof(sample,source['task_id'],source['task_version'])
    compiled=suite_plan.compile_suite(suite_fixture(),sources,local_fixture());jobs={};metadata=sources['model_metadata']
    for model in compiled['semantic']['suite']['models']:
        logical=model['arm_id'];observed=model['artifact_ref'];prefix=compiled['plan_identity']+'|fixture|attempt1|'+observed
        directory=root/'acquisitions/serve/attempts'/('serve-'+observed+'-attempt1');directory.mkdir(parents=True)
        records=[];journal=[]
        for source in sources['sources']:
            adapter=get_task(source['task_id'],source['task_version'])
            for sample in source['samples']:
                requests=suite_results._requests(source,sample,prefix,metadata)
                for req in requests:
                    common={'request_id':req.request_id,'request_kind':req.request_kind,'identity':req.identity,
                        'task_id':source['task_id'],'task_version':source['task_version'],'sample_id':sample['sample_id'],
                        'sample_identity':sample['authored_sample'].get('sample_identity'),'arm_id':logical,'source_arm_id':observed,
                        'artifact_identity':next(a['artifact']['artifact_identity'] for a in metadata['arms'] if a['arm_id']==observed),'execution_prefix':prefix,
                        'tokenizer_identity':metadata['tokenizer_identity'],'transport_complete':True,'evidence_kind':'synthetic_contract'}
                    if req.request_kind=='generate_until':
                        text=adapter.reference(sample['authored_sample'])['text'];ids=[5]*len(text);body=sample['generation']['wire_payload']
                        usage={'prompt_tokens':len(body['prompt']),'completion_tokens':len(ids),'total_tokens':len(body['prompt'])+len(ids)}
                        frame={'choices':[{'index':0,'text':text,'token_ids':ids,'prompt_token_ids':body['prompt'],'finish_reason':'stop'}],'usage':usage}
                        record=common|{'request_body':body,'request_identity':wire_sha(body),'prompt_hash':sample['diagnostic']['prompt_hash'],
                            'reference_used_for_generation':False,'raw_sse_frames':[json.dumps(frame),'[DONE]'],'output_text':text,
                            'output_token_ids':ids,'usage':usage,'finish_reason':'stop','termination':'NATURAL_STOP',
                            'queued_monotonic':1.,'http_submit_monotonic':1.1,'completed_monotonic':1.5,'first_content_monotonic':1.2,
                            'request_latency_http':1.5-1.1,'request_latency_task':.5,'ttft_http':1.2-1.1}
                    else:
                        ids=req.payload['input_ids'];start=len(req.payload['context_ids']);body=request_body(req)
                        response={'usage':{'prompt_tokens':len(ids),'completion_tokens':1},'choices':[{'logprobs':{
                            'tokens':['token_id:'+str(t) for t in ids]+['token_id:7'],
                            'token_logprobs':[None]*start+[-1.]*(len(ids)-start)+[-999.]}}]}
                        record=common|{'request_body':body,'request_body_identity':digest(body),'response':response,
                            'raw_response_text':json.dumps(response),'http_status':200,'choice_index':req.payload['choice_index'],
                            'queued':1.,'semaphore_acquired':1.05,'http_submit':1.1,'response_complete':1.5}
                    records.append(record)
                    # The natural collector journal lacks request_kind/identity;
                    # its declared task and prefix reconstruct typed identity.
                    event={k:v for k,v in common.items() if k not in ('identity','request_kind','sample_identity','transport_complete','evidence_kind','tokenizer_identity')}
                    if 'choice_index' in record:event['choice_index']=record['choice_index']
                    journal.append(event|{'event':'http_submit','request_body':body})
                    journal.append(event|{'event':'response_end'})
        write_jsonl(directory/'task-responses.jsonl',records);write_jsonl(directory/'request-journal.jsonl',journal)
        jobs[logical]={'raw_path':str((directory/'task-responses.jsonl').relative_to(root)),
                      'journal_path':str((directory/'request-journal.jsonl').relative_to(root))}
    context={'sources':sources,'suite_plan':compiled,'serving_jobs':jobs,'diagnostic_run':'acquisitions/token','evidence_kind':'synthetic_contract'}
    write(root/'SUITE_PLAN.json',compiled);write(root/'suite-execution-context.json',context);seal(root)
    return root


def replace_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def view_fixture(root, context):
    root.mkdir();records=[];sequences=[];logits=[0.,1.,2.,-1.]
    for source in context['sources']['sources']:
        for sample in source['samples']:
            sid=sample['sample_id'];seqid=digest({'sample':sid,'ids':[1,2,3]})
            sequences.append({'sequence_id':seqid,'input_ids':[1,2,3]})
            for arm,acquisition in (('bf16','instrumented_diagnostic'),('rtn','decoded_diagnostic')):
                for position,target in ((1,2),(2,3)):
                    ids=[1,2,3];record_id=digest([sid,arm,position]);numeric=save_blob(root,record_id,logits,[4])
                    records.append({'record_id':record_id,'source_id':'source','panel':'same-runtime','evidence_kind':'synthetic_contract',
                        'task_id':source['task_id'],'task_version':source['task_version'],'dataset_identity':'data',
                        'split':'REGRESSION_REUSE','sample_id':sid,'policy_id':'fixed','choice_index':None,
                        'reference_scope':'author_gold','prompt_hash':'prompt','reference_hash':'reference','sequence_id':seqid,
                        'tokenizer_identity':'tokenizer','token_id_space':'vocabulary','vocabulary_size':4,
                        'runtime_identity':'runtime','backend_policy':'diagnostic','acquisition_source':acquisition,'arm_id':arm,
                        'probability_stage':'raw_pre_sampling','metric_contract':'target_logprob_v1','token_position':position,
                        'logit_position':position-1,'target_token_id':target,'prefix_hash':digest({'token_ids':ids[:position],'token_id_space':'vocabulary'}),
                        'requested':True,'status':'MEASURED','field_tags':[],'target_logprob':log_softmax(logits)[target],
                        'raw_argmax_token_id':2,'logit_logprob_tolerance':2e-5,'full_logits':numeric})
    write_jsonl(root/'token_records.jsonl',records);write_jsonl(root/'sequences.jsonl',sequences);write_jsonl(root/'generation_records.jsonl',[])
    write(root/'COVERAGE.json',{'records':len(records)});write(root/'SOURCES.json',[]);write(root/'INPUT_SOURCES.json',{'kind':'synthetic_contract'})
    write(root/'METRIC_CONTRACT.json',{})
    files={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in root.rglob('*') if p.is_file()}
    write(root/'TOKEN_VIEW_MANIFEST.json',{'schema_version':1,'kind':'common_token_view_v1','analysis_source_digest':'synthetic',
        'input_manifest_sha256':sha(root/'INPUT_SOURCES.json'),'source_panels':[{'source_id':'source','coverage':{'records':len(records)}}],
        'files':files,'view_identity':digest(files),'new_observations':False,'GPU_actions':0})
    return root


class SuiteResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.run=fixture(self.root/'run')
    def tearDown(self):self.tmp.cleanup()
    def score(self):return suite_results.score_tasks(self.run,self.root/'analysis')
    def mutate_raw(self,arm,mutate):
        context=read(self.run/'suite-execution-context.json');path=self.run/context['serving_jobs'][arm]['raw_path']
        records=list(jsonl(path));mutate(records);path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8');seal(self.run)
    def test_gold_positive_controls_all_three_tasks_and_logical_arm_mapping(self):
        result=self.score();self.assertEqual(result['document_score_rows'],12);self.assertEqual(result['inference_responses'],20)
        self.assertEqual(result['inference_submissions'],20);self.assertTrue(result['all_required_task_inputs_complete'])
        self.assertEqual(set(result['tasks']),{'tech_record_extract','arc_challenge','context_evidence_retrieval'})
        self.assertEqual(set(result['tasks']['arc_challenge']),{'baseline','candidate'})
        self.assertEqual(result['tasks']['tech_record_extract']['baseline']['metrics']['task_success']['correct'],2)
        self.assertEqual(result['tasks']['arc_challenge']['baseline']['metrics']['acc']['planned_denominator'],2)
        self.assertEqual(result['tasks']['context_evidence_retrieval']['candidate']['metrics']['task_success']['correct'],2)
        paired=list(jsonl(self.root/'analysis/paired-outcomes.jsonl'));self.assertEqual(len(paired),6)
        self.assertIn('acc',next(r for r in paired if r['task_id']=='arc_challenge')['metrics'])
    def test_registry_missing_generation_is_not_run(self):
        sources=read(self.run/'suite-execution-context.json')['sources'];sample=sources['sources'][0]['samples'][0]['authored_sample']
        self.assertEqual(get_task('tech_record_extract','0.1').score(sample,None)['status'],'NOT_RUN')
    def test_tampered_unrepinned_raw_rejected(self):
        context=read(self.run/'suite-execution-context.json');path=self.run/context['serving_jobs']['baseline']['raw_path']
        path.write_text('{}\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'content changed'):self.score()
    def test_missing_response_recovered_from_submission_not_fabricated_zero(self):
        self.mutate_raw('candidate',lambda rows:rows.pop(0));result=self.score()
        self.assertFalse(result['all_required_task_inputs_complete']);self.assertEqual(result['inference_submissions'],20)
        self.assertEqual(result['inference_responses'],19)
        rows=list(jsonl(self.root/'analysis/document-scores.jsonl'))
        score=next(r['score'] for r in rows if r['arm_id']=='candidate' and r['task_id']=='tech_record_extract')
        self.assertEqual(score['termination'],'INCOMPLETE_SUBMISSION');self.assertIsNone(score['request_latency_http'])
        pair=next(r for r in jsonl(self.root/'analysis/paired-outcomes.jsonl') if r['status']=='INCOMPLETE')
        self.assertTrue(all(v['outcome']=='UNKNOWN' for v in pair['metrics'].values()))
    def test_duplicate_request_or_wrong_task_rejected(self):
        self.mutate_raw('baseline',lambda rows:rows.append(deepcopy(rows[0])))
        with self.assertRaisesRegex(ValueError,'Duplicate global'):self.score()
    def test_raw_artifact_treatment_mismatch_rejected(self):
        self.mutate_raw('baseline',lambda rows:rows[0].update(artifact_identity='f'*64))
        with self.assertRaisesRegex(ValueError,'artifact/arm'):self.score()
    def test_raw_sse_text_tamper_even_after_manifest_repin_rejected(self):
        self.mutate_raw('baseline',lambda rows:rows[0].update(output_text='changed'))
        with self.assertRaisesRegex(ValueError,'SSE/text'):self.score()
    def test_partial_arc_choices_do_not_form_argmax(self):
        self.mutate_raw('candidate',lambda rows:rows.pop(next(i for i,r in enumerate(rows) if r['request_kind']=='choice_loglikelihood')))
        self.score();rows=list(jsonl(self.root/'analysis/document-scores.jsonl'))
        score=next(r['score'] for r in rows if r['arm_id']=='candidate' and r['task_id']=='arc_challenge')
        self.assertEqual(score['status'],'INCOMPLETE');self.assertIsNone(score['acc']);self.assertIsNone(score['acc_norm'])
    def test_raw_echo_token_identity_mismatch_rejected_as_incomplete(self):
        def mutate(rows):
            raw=next(r for r in rows if r['request_kind']=='choice_loglikelihood')
            raw['response']['choices'][0]['logprobs']['tokens'][2]='token_id:127'
            raw['raw_response_text']=json.dumps(raw['response'])
        self.mutate_raw('candidate',mutate);self.score()
        self.assertTrue(any(r['status']=='INCOMPLETE' for r in jsonl(self.root/'analysis/choice-scores.jsonl')))
    def test_inline_gold_change_cannot_be_hidden_by_repinning_context_only(self):
        path=self.run/'suite-execution-context.json';value=read(path)
        value['sources']['sources'][0]['samples'][0]['authored_sample']['events'].append({'field':'status','value':'repeat'})
        replace_json(path,value);seal(self.run)
        with self.assertRaises(ValueError):self.score()
    def test_cpu_stage_status_is_complete_or_partial_from_raw_coverage(self):
        result=suite_results.execute_cpu('score_task',self.run,self.root/'analysis')
        self.assertEqual(result['status'],'COMPLETE')
        self.mutate_raw('candidate',lambda rows:rows.pop(0))
        result=suite_results.execute_cpu('score_task',self.run,self.root/'partial')
        self.assertEqual(result['status'],'PARTIAL')
    def test_no_output_overwrite_and_unknown_cpu_kind(self):
        self.score()
        with self.assertRaises(ValueError):self.score()
        with self.assertRaisesRegex(ValueError,'Unsupported suite CPU'):
            suite_results.execute_cpu('shell',self.run,self.root/'bad')
    def test_journal_missing_or_body_mismatch_rejected(self):
        context=read(self.run/'suite-execution-context.json');path=self.run/context['serving_jobs']['baseline']['journal_path']
        rows=list(jsonl(path));rows[0]['request_body']['prompt']=[9];replace_json(path,rows[0]);seal(self.run)
        with self.assertRaisesRegex(ValueError,'Journal wire'):self.score()
    def test_adapter_new_index_dispatch_and_explicit_historical_copy(self):
        from inference_kit.analysis import index_sources
        with patch.object(suite_results,'load_index_source',return_value={'test':'envelope'}) as reader:
            self.assertEqual(index_sources.load_source({'adapter':'suite_e3','source_id':'new','path':'run'},self.root),{'test':'envelope'})
            reader.assert_called_once()
    def test_explicit_mini_slice_replays_http_and_three_document_token_scope(self):
        context=read(self.run/'suite-execution-context.json');view=view_fixture(self.root/'view',context)
        receipt=suite_results.export_mini_slice(self.run,view,self.root/'mini')
        self.assertEqual(receipt['coverage']['full_logits_records'],6)
        self.assertEqual(len(receipt['token_samples']),3)
        replay=suite_results.replay(self.root/'mini',self.root/'mini-replay')
        self.assertEqual(replay['status'],'CPU_SLICE_REPLAYED')
        self.assertEqual(replay['results']['score_task']['result']['inference_responses'],20)
        coverage=replay['results']['analyze_tokens']['result']['coverage']
        self.assertEqual(coverage['requested_pair_coordinates'],6);self.assertEqual(coverage['measured'],6)
        full=sum(row['coverage']['full_logits'] for row in jsonl(self.root/'mini-replay/analyze_tokens/token-metrics/token-comparisons.jsonl'))
        self.assertEqual(full,3)
        self.assertFalse((self.root/'mini-replay/index_add').exists())
        mini_context=read(self.root/'mini/suite-execution-context.json')
        self.assertNotIn('bindings',mini_context['suite_plan'])
    def test_mini_does_not_bypass_missing_blob_or_parent_manifest(self):
        context=read(self.run/'suite-execution-context.json');view=view_fixture(self.root/'view',context)
        suite_results.export_mini_slice(self.run,view,self.root/'mini')
        blob=next((self.root/'mini/token-view/blobs').iterdir());blob.unlink()
        with self.assertRaisesRegex(ValueError,'unsafe suite raw member'):
            suite_results.replay(self.root/'mini',self.root/'reject')
    def test_nested_token_evidence_prefix_links_observation_and_blob(self):
        from inference_kit.analysis.index_sources import _emit
        names={'acquisitions/token/PLAN_FROZEN.json':'plan',
            'acquisitions/token/attempts/a/observations.jsonl':'observation',
            'acquisitions/token/attempts/a/blobs/value.f32':'numeric'}
        envelope={'evidence':[{'relative_path':p,'id':i} for p,i in names.items()],'measurements':[]}
        payload={'source_observations_path':'attempts/a/observations.jsonl','numeric':{'path':'blobs/value.f32'}}
        _emit(envelope,'analysis',{'s':'sample'},'token.nll',1.,'MEASURED','token_pair',
              {'sample_id':'s','task_id':'task'},payload,evidence_prefix='acquisitions/token')
        self.assertEqual(set(envelope['measurements'][0]['evidence_ids']),{'plan','observation','numeric'})
    def test_historical_index_must_match_pinned_readonly_binding(self):
        from inference_kit.analysis.result_index import _create
        old=self.root/'historical.sqlite';_create(old);path=self.run/'suite-execution-context.json';context=read(path)
        context['historical_index_sha256']=sha(old);replace_json(path,context);seal(self.run)
        with old.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Historical index differs'):
            suite_results.execute_cpu('index_add',self.run,self.root/'rejected',stage_outputs={'historical_index':old})
