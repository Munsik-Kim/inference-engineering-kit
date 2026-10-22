"""Synthetic prepared-input replay; fake bytes never assert live collection."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from inference_kit.adapters.http import sha as wire_sha
from inference_kit.adapters.likelihood import request_body
from inference_kit.adapters.protocol import Adapter
from inference_kit.analysis import suite_results
from inference_kit.analysis.tokens import jsonl, write_jsonl
from inference_kit.core.io import digest, read, sha, write
from inference_kit.metrics.token import log_softmax
from inference_kit.probes.blobs import save
from inference_kit.probes.contracts import Execution, Observation
from inference_kit.suites.bench05b1_run import seal as seal_token
from inference_kit.suites.suite_plan import compile_suite
from inference_kit.tasks.registry import get_task
from bench05e2.test_plan import local_fixture
from bench05e3.test_results import seal
from bench05e4a.test_suite_prepared import loaded_fixture, prepared_suite


def synthetic_tokens(root, sources):
    root.mkdir(parents=True);metadata=sources['model_metadata'];rows=[]
    for source in sources['sources']:
        for sample in source['samples']:
            row=deepcopy(sample['diagnostic']);row.update(dataset_identity=source['source_id'],split='SYNTHETIC',
                source_id=source['source_id'],policy_id='synthetic-prepared',policy_identity='synthetic-policy')
            rows.append(row)
    policy={'id':'same_pass_saved_f32_v1','absolute_tolerance_nat':2e-5,
            'scope':'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'}
    arms=[dict(a,execution_mode='REIMPLEMENTED_OR_SYNTHETIC',artifact=a['artifact']|{'recipe_identity':'synthetic-recipe'}) for a in metadata['arms']]
    plan={'task':'BENCH05-C2','pair':metadata['pair'],'arms':arms,'runtime_identity':'synthetic-runtime',
        'source_identity':{'fixture':'SYNTHETIC_EXECUTOR_TEST'},'evidence_kind':'synthetic_contract',
        'samples':{'rows':rows,'tokenizer_identity':metadata['tokenizer_identity'],'token_id_space':metadata['token_id_space'],
                   'vocabulary_size':metadata['vocabulary_size']},
        'probe_spec':{'normalization':{'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'},'consistency_policy':policy}}
    plan['plan_identity']=digest(plan);write(root/'PLAN_FROZEN.json',plan);jobs={}
    for arm in arms:
        name=arm['arm_id'];directory=root/'attempts'/('main-'+name+'-attempt1');directory.mkdir(parents=True)
        execution=Execution('synthetic-prepared',directory.name,name,plan['pair']['model_id'],plan['pair']['revision'],
            arm['artifact']['artifact_identity'],'synthetic-recipe',metadata['tokenizer_identity'],'synthetic-runtime',
            'REIMPLEMENTED_OR_SYNTHETIC','float32','toy','synthetic_contract','MEASURED','observations.jsonl').validate()
        execution['phase']='main';execution['execution_id']=digest(execution);write(directory/'execution.json',execution);observations=[]
        logits=[i*.01 for i in range(metadata['vocabulary_size'])];lp=log_softmax(logits)
        for row in rows:
            for target in row['targets']:
                coordinate={'sample_id':row['sample_id'],'pass_id':'ON','sequence_identity':row['sequence_identity'],
                    'token_id_space':metadata['token_id_space'],'tokenizer_identity':metadata['tokenizer_identity'],**target}
                details={'normalization_dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e',
                    'consistency_policy':policy,'raw_argmax_token_id':len(logits)-1}
                values=[('target_logprob',[lp[target['target_token_id']]])]
                if target['token_position'] in row['full_vocab_target_positions']:values.append(('full_logits',logits))
                for kind,value in values:
                    oid=digest([name,row['sample_id'],kind,coordinate]);numeric=save(directory,oid,value,[len(value)])
                    observations.append(Observation(oid,kind,execution['execution_id'],row['sample_id'],'synthetic_contract',
                        'REIMPLEMENTED_OR_SYNTHETIC','MEASURED',coordinate,numeric,
                        details|({'full_vocabulary':True} if kind=='full_logits' else {})).validate())
        write_jsonl(directory/'observations.jsonl',observations);write(directory/'result.json',{'status':'MEASURED'})
        jobs['main-'+name]={'status':'COMPLETE','directory':directory.relative_to(root).as_posix()}
    write(root/'progress.json',{'status':'SYNTHETIC_CONTRACT_ONLY','jobs':jobs});seal_token(root,scope='token_probe_raw_v1')


def fixture(root, count=1):
    root.mkdir();input_root=root.parent/'inputs';input_root.mkdir();sources=loaded_fixture(input_root,count)
    for source in sources['sources']:
        for sample in source['samples']:
            if source['request_kind']=='choice_loglikelihood':
                sample['authored_sample']['request_identity']=digest(sample['authored_sample']['requests'])
            if source['request_kind']=='generate_until':
                sample['generation']['wire_payload']=Adapter('vllm').body(
                    {'token_ids':sample['generation']['prepared']['generation_input_ids']},'serving-lab',
                    192 if source['task_id']=='tech_record_extract' else 64,'natural_generation_v1')
    spec=prepared_suite(sources,'both');spec['metrics'].insert(0,'task_default_v1')
    compiled=compile_suite(spec,sources,local_fixture());jobs={};metadata=sources['model_metadata']
    for model in compiled['semantic']['suite']['models']:
        logical=model['arm_id'];observed=model['artifact_ref']
        for source in sources['sources']:
            group='serve-'+observed+'-'+source['task_id'];prefix=compiled['plan_identity']+'|fixture|'+'attempt1|'+observed+'|'+group
            directory=root/'acquisitions/serve'/group/'attempts'/(group+'-attempt1');directory.mkdir(parents=True)
            records=[];journal=[];adapter=get_task(source['task_id'],source['task_version'])
            for sample in source['samples']:
                for req in suite_results._requests(source,sample,prefix,metadata):
                    namespace={k:sample['raw_record'][k] for k in ('task_id','task_version','dataset_id','split','sample_id')}
                    common={'source_sample_key':namespace,'qualified_sample_identity':digest(namespace),'qualified_sample_id':digest(namespace),'request_id':req.request_id,'request_kind':req.request_kind,'identity':req.identity,
                        'task_id':source['task_id'],'task_version':source['task_version'],'sample_id':sample['sample_id'],
                        'sample_identity':sample['authored_sample'].get('sample_identity'),'arm_id':logical,'source_arm_id':observed,
                        'artifact_identity':next(a['artifact']['artifact_identity'] for a in metadata['arms'] if a['arm_id']==observed),
                        'execution_prefix':prefix,'execution_unit_id':group,'tokenizer_identity':metadata['tokenizer_identity'],
                        'transport_complete':True,'evidence_kind':'synthetic_contract'}
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
                    event={k:v for k,v in common.items() if k not in ('identity','request_kind','sample_identity','transport_complete','evidence_kind','tokenizer_identity')}
                    if 'choice_index' in record:event['choice_index']=record['choice_index']
                    journal.extend([event|{'event':'http_submit','request_body':body},event|{'event':'response_end'}])
            write_jsonl(directory/'task-responses.jsonl',records);write_jsonl(directory/'request-journal.jsonl',journal)
            jobs[group]={'group_id':group,'logical_arm_id':logical,'source_arm_id':observed,'task_id':source['task_id'],'source_id':source['source_id'],
                'raw_path':(directory/'task-responses.jsonl').relative_to(root).as_posix(),
                'journal_path':(directory/'request-journal.jsonl').relative_to(root).as_posix()}
    context={'profile':'prepared_smol_live_v1','sources':sources,'suite_plan':compiled,'serving_jobs':jobs,
             'diagnostic_run':'acquisitions/token','evidence_kind':'synthetic_contract'}
    write(root/'SUITE_PLAN.json',compiled);write(root/'suite-execution-context.json',context)
    synthetic_tokens(root/'acquisitions/token',sources);seal(root)
    return root


class PreparedSuiteResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.run=fixture(self.root/'run')

    def test_multiple_groups_same_arm_score_exact_requests_and_raw_links(self):
        result=suite_results.score_tasks(self.run,self.root/'scores')
        self.assertEqual(result['planned_source_documents'],3);self.assertEqual(result['document_score_rows'],6)
        self.assertEqual(result['inference_submissions'],12);self.assertEqual(result['inference_responses'],12)
        self.assertTrue(result['all_required_task_inputs_complete'])
        rows=list(jsonl(self.root/'scores/document-scores.jsonl'))
        self.assertEqual(len({r['serving_group_id'] for r in rows}),6)
        for row in rows:
            self.assertTrue(all('/'+row['serving_group_id']+'/' in p for p in row['raw_references']))
            self.assertEqual(row['scope'],row['usage_scope'])

    def test_three_and_nine_inputs_vary_without_historical_sample_contract(self):
        other=self.root/'other';other.mkdir();run=fixture(other/'run',3)
        a=suite_results.score_tasks(self.run,self.root/'a');b=suite_results.score_tasks(run,self.root/'b')
        self.assertEqual((a['inference_responses'],b['inference_responses']),(12,36))
        self.assertEqual((a['document_score_rows'],b['document_score_rows']),(6,18))
        self.assertTrue(all('ancestry' not in s for s in read(run/'suite-execution-context.json')['sources']['sources']))

    def test_index_acquisition_count_comes_from_groups_and_reregister_is_idempotent(self):
        from inference_kit.analysis.result_index import _create, add_sources, canonical_export
        db=self.root/'index.sqlite';_create(db);manifest=self.root/'index-sources.json'
        write(manifest,{'schema_version':1,'sources':[{'source_id':'synthetic-prepared','adapter':'suite_e3','path':str(self.run)}]})
        first=add_sources(manifest,db,self.root/'first');second=add_sources(manifest,db,self.root/'second')
        self.assertEqual(first['after']['acquisitions'],8);self.assertEqual(second['before'],second['after'])
        self.assertGreater(first['after']['measurements'],0)
        rebuilt=self.root/'rebuilt.sqlite';_create(rebuilt);add_sources(manifest,rebuilt,self.root/'rebuilt')
        self.assertEqual(canonical_export(db)['logical_identity'],canonical_export(rebuilt)['logical_identity'])

    def test_cross_task_same_sample_id_is_explicitly_rejected_before_index_lookup(self):
        context=read(self.run/'suite-execution-context.json');sources=deepcopy(context['sources']);compiled=deepcopy(context['suite_plan'])
        first,second=sources['sources'][:2];old=second['samples'][0]['sample_id'];same=first['samples'][0]['sample_id']
        second['samples'][0]['sample_id']=same
        next(t for t in compiled['semantic']['suite']['tasks'] if t['source_id']==second['source_id'])['sample_ids']=[same]
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_AMBIGUOUS_SAMPLE_ID'):suite_results._selected(sources,compiled)
        self.assertNotEqual(old,same)

    def test_group_scope_and_evidence_path_collision_are_rejected(self):
        path=self.run/'suite-execution-context.json';context=read(path);groups=list(context['serving_jobs'])
        context['serving_jobs'][groups[0]]['task_id']='wrong-task';path.write_text(json.dumps(context),encoding='utf-8');seal(self.run)
        with self.assertRaisesRegex(ValueError,'another serving execution unit'):suite_results.score_tasks(self.run,self.root/'bad')
        context['serving_jobs'][groups[0]]['raw_path']=context['serving_jobs'][groups[1]]['raw_path']
        with self.assertRaisesRegex(ValueError,'share an evidence path'):suite_results._serving_jobs(context,suite_results._arms(context['sources'],context['suite_plan']))

    def test_record_unit_and_qualified_namespace_are_not_inferred_from_sample_id(self):
        context=read(self.run/'suite-execution-context.json');job=next(iter(context['serving_jobs'].values()))
        path=self.run/job['raw_path'];rows=list(jsonl(path));rows[0]['source_sample_key']['dataset_id']='wrong-dataset'
        write_jsonl(path.with_name('mutated.jsonl'),rows)
        path.write_bytes(path.with_name('mutated.jsonl').read_bytes());seal(self.run)
        with self.assertRaisesRegex(ValueError,'qualified sample namespace'):suite_results.score_tasks(self.run,self.root/'namespace-bad')

    def test_token_analysis_reuses_existing_c1_and_separates_from_serving(self):
        result=suite_results.analyze_tokens(self.run,self.root/'tokens');self.assertGreater(result['coverage']['measured'],0)
        self.assertEqual(result['coverage']['incomparable'],0)
        self.assertEqual(read(self.root/'tokens/TOKEN_ANALYSIS_RECEIPT.json')['existing_engine'],'C1 tokens_export/tokens_compare')

    def test_missing_live_manifest_cannot_be_presented_as_empty_success(self):
        (self.run/'RUN_MANIFEST.json').unlink()
        with self.assertRaisesRegex(ValueError,'Missing or unsafe'):suite_results.score_tasks(self.run,self.root/'absent')
