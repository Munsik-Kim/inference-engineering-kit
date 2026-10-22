"""Pure CPU generic-source contracts with explicitly synthetic tiny arrays."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from inference_kit.analysis.token_probe_source import collect_source
from inference_kit.analysis.tokens import export_tokens,compare_tokens
from inference_kit.core.io import read,write,digest,sha
from inference_kit.metrics.token import log_softmax
from inference_kit.probes.blobs import save
from inference_kit.probes.contracts import Execution,Observation
from inference_kit.suites.bench05b1_run import seal


def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def replace_lines(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def reseal(root):
    (root/'RAW_MANIFEST.json').unlink();seal(root,scope='token_probe_raw_v1')


def synthetic_raw(root,*,failed=None):
    root.mkdir();rows=[]
    policy={'id':'same_pass_saved_f32_v1','absolute_tolerance_nat':2e-5,
            'scope':'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'}
    for index,task in enumerate(('generic_alpha','generic_beta','future_unregistered_task')):
        ids=[0,1,2,3]
        rows.append({'task_id':task,'task_version':'9.1','dataset_identity':'synthetic-'+task,'sample_id':'sample-'+str(index),
            'split':'SYNTHETIC','source_id':'synthetic-source','policy_id':'policy-'+str(index),'policy_identity':'frozen-'+str(index),
            'reference_source':'independent_authored_reference','reference_role':'dataset_gold_choice' if index==1 else 'author_gold_reference',
            'choice_index':2 if index==1 else None,'prompt':'AB','prompt_hash':digest('AB'),'reference_text':'CD','reference_hash':digest('CD'),'full_text':'ABCD',
            'input_ids':ids,'sequence_identity':digest(ids),'prompt_token_boundary':2,'offsets_codepoints':[[i,i+1] for i in range(4)],
            'offsets_utf8_bytes':[[i,i+1] for i in range(4)],'loss_mask':[0,0,1,1],'attention_mask':[1]*4,'special_token_mask':[0]*4,
            'targets':[{'token_position':i,'logit_position':i-1,'target_token_id':ids[i]} for i in (2,3)],'full_vocab_target_positions':[2,3],
            'field_token_coverage':{'value':{'overlapping_positions':[2,3],'selected_positions':[2,3]}},'value_spans':{'value':[0,2]}})
    arms=[{'arm_id':name,'execution_mode':'REIMPLEMENTED_OR_SYNTHETIC','artifact':{'artifact_identity':'synthetic-'+name,'recipe_identity':'recipe-'+name}} for name in ('bf16','rtn')]
    plan={'task':'BENCH05-C2','pair':{'model_id':'synthetic/model','revision':'pinned-fixture'},'arms':arms,
          'runtime_identity':'synthetic-runtime','source_identity':{'fixture':'synthetic-not-model-code'},'evidence_kind':'synthetic_contract',
          'samples':{'rows':rows,'tokenizer_identity':'synthetic-tokenizer','token_id_space':'synthetic-space','vocabulary_size':8},
          'probe_spec':{'normalization':{'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'},'consistency_policy':policy}}
    plan['plan_identity']=digest(plan);write(root/'PLAN_FROZEN.json',plan);jobs={}
    for arm in arms:
        name=arm['arm_id'];directory=root/'attempts'/('main-'+name+'-attempt1');directory.mkdir(parents=True)
        execution=Execution('synthetic-c2',directory.name,name,plan['pair']['model_id'],plan['pair']['revision'],arm['artifact']['artifact_identity'],
            arm['artifact']['recipe_identity'],'synthetic-tokenizer','synthetic-runtime','REIMPLEMENTED_OR_SYNTHETIC','float32','toy',
            'synthetic_contract','MEASURED','observations.jsonl').validate()
        execution['phase']='main';execution['execution_id']=digest(execution);write(directory/'execution.json',execution)
        observations=[]
        for row in rows:
            logits=[i*.25 for i in range(8)];lp=log_softmax(logits)
            for target in row['targets']:
                coordinate={'sample_id':row['sample_id'],'pass_id':'ON','sequence_identity':row['sequence_identity'],
                    'token_id_space':'synthetic-space','tokenizer_identity':'synthetic-tokenizer',**target,
                    'prefix_hash':digest({'token_ids':row['input_ids'][:target['token_position']],'token_id_space':'synthetic-space'})}
                details={'normalization_dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e','consistency_policy':policy,'raw_argmax_token_id':7}
                for kind,values in (('target_logprob',[lp[target['target_token_id']]]),('full_logits',logits)):
                    identifier=digest({'arm':name,'kind':kind,'coordinates':coordinate})[:28]
                    numeric=save(directory,identifier,values,[len(values)])
                    observations.append(Observation(identifier,kind,execution['execution_id'],row['sample_id'],'synthetic_contract','REIMPLEMENTED_OR_SYNTHETIC',
                        'MEASURED',coordinate,numeric,details|({'full_vocabulary':True} if kind=='full_logits' else {})).validate())
        replace_lines(directory/'observations.jsonl',observations);write(directory/'result.json',{'status':'MEASURED'})
        jobs['main-'+name]={'status':'FAILED' if failed==name else 'COMPLETE','directory':str(directory.relative_to(root))}
    write(root/'progress.json',{'status':'SYNTHETIC_CONTRACT_ONLY','jobs':jobs});seal(root,scope='token_probe_raw_v1')
    return root


class GenericTokenSourceTests(unittest.TestCase):
    def spec(self,root,**extra):return {'source_id':'new-configured-source','adapter':'token_probe_raw_v1','path':str(root),**extra}

    def test_three_unregistered_tasks_use_generic_source_without_registry(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw');result=collect_source(self.spec(raw),lambda directory,metadata:metadata)
            self.assertEqual(len(result['source']['task_ids']),3);self.assertEqual(len(result['tokens']),12)
            self.assertEqual(len({r['panel'] for r in result['tokens']}),3)
            self.assertTrue(all(r['evidence_kind']=='synthetic_contract' and r['status']=='MEASURED' for r in result['tokens']))
            self.assertEqual(result['source']['coverage']['same_pass_consistency_verified'],12)
            self.assertTrue(all(r['logit_logprob_tolerance']==2e-5 for r in result['tokens']))

    def test_canonical_dataset_gold_role_not_mislabeled_non_gold(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw');result=collect_source(self.spec(raw),lambda directory,metadata:metadata)
            choice=[r for r in result['tokens'] if r['choice_index'] is not None]
            self.assertTrue(choice);self.assertTrue(all('gold_choice' in r['field_tags'] and 'non_gold_choice' not in r['field_tags'] for r in choice))

    def test_new_dispatcher_never_calls_generation_adapter(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_raw(root/'raw');write(root/'sources.json',{'schema_version':1,'kind':'token_input_sources','sources':[self.spec(raw)]})
            with patch('inference_kit.analysis.generation_sources.collect_generations',side_effect=AssertionError('new token raw is not generation')):
                result=export_tokens(root/'sources.json',root/'view')
            self.assertEqual(result['coverage']['records'],12);self.assertEqual(result['coverage']['generation_records'],0)
            compared=compare_tokens(root/'view','bf16','rtn',root/'analysis')
            self.assertEqual(compared['coverage']['measured'],6);self.assertEqual(compared['coverage']['incomparable'],0)

    def test_nested_zip_raw_root_is_discovered_without_executing_code(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_raw(root/'raw');archive=root/'raw.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for path in raw.rglob('*'):
                    if path.is_file():z.write(path,'nested/run/'+path.relative_to(raw).as_posix())
                z.writestr('trap.py','raise RuntimeError("archive Python must not execute")')
            write(root/'sources.json',{'schema_version':1,'kind':'token_input_sources','sources':[self.spec(archive)]})
            result=export_tokens(root/'sources.json',root/'view');self.assertEqual(result['coverage']['records'],12)

    def test_task_filter_runs_after_full_raw_verification(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw');result=collect_source(self.spec(raw,task_ids=['future_unregistered_task']),lambda directory,metadata:metadata)
            self.assertEqual({r['task_id'] for r in result['tokens']},{'future_unregistered_task'})
            path=raw/'progress.json';path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):collect_source(self.spec(raw,task_ids=['generic_alpha']),lambda directory,metadata:metadata)

    def test_plan_manifest_and_pinned_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw')
            with self.assertRaises(ValueError):collect_source(self.spec(raw,raw_manifest_sha256='0'*64),lambda directory,metadata:metadata)
            plan=read(raw/'PLAN_FROZEN.json');plan['task']='wrong';(raw/'PLAN_FROZEN.json').write_text(json.dumps(plan),encoding='utf-8');reseal(raw)
            with self.assertRaises(ValueError):collect_source(self.spec(raw),lambda directory,metadata:metadata)

    def test_source_execution_model_artifact_tokenizer_and_runtime_match(self):
        for key in ('artifact_identity','tokenizer_identity','runtime_identity','model_revision'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                raw=synthetic_raw(Path(t)/'raw');path=next(raw.glob('attempts/main-*/execution.json'));e=read(path);e[key]='mismatch'
                path.write_text(json.dumps(e),encoding='utf-8');reseal(raw)
                with self.assertRaises(ValueError):collect_source(self.spec(raw),lambda directory,metadata:metadata)

    def test_observation_probability_stage_consistency_and_prefix_match(self):
        for change in ('probability_stage','consistency_policy','prefix_hash','target_shift','boolean_logit'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                raw=synthetic_raw(Path(t)/'raw');path=next(raw.glob('attempts/main-*/observations.jsonl'));rows=lines(path)
                if change=='probability_stage':rows[0]['details']['probability_stage']='processed_after_sampling'
                if change=='consistency_policy':rows[0]['details']['consistency_policy']['absolute_tolerance_nat']=.1
                if change=='prefix_hash':rows[0]['coordinates']['prefix_hash']='bad'
                if change=='target_shift':rows[0]['coordinates']['logit_position']+=1
                if change=='boolean_logit':rows[0]['coordinates']['logit_position']=True
                replace_lines(path,rows);reseal(raw)
                with self.assertRaises(ValueError):collect_source(self.spec(raw),lambda directory,metadata:metadata)

    def test_saved_full_target_inconsistency_rejected_before_sink(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw')
            first_arm=read(raw/'PLAN_FROZEN.json')['arms'][0]['arm_id']
            path=raw/read(raw/'progress.json')['jobs']['main-'+first_arm]['directory']/'observations.jsonl';rows=lines(path)
            target=next(r for r in rows if r['kind']=='target_logprob');directory=path.parent
            target['numeric']=save(directory,'corrupted-scalar',[-.1],[1]);replace_lines(path,rows);reseal(raw);calls=[]
            with self.assertRaisesRegex(ValueError,'same-pass'):
                collect_source(self.spec(raw),lambda directory,metadata:calls.append(metadata))
            self.assertEqual(calls,[])

    def test_saved_full_vocabulary_shape_and_argmax_are_checked(self):
        for change in ('shape','argmax'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                raw=synthetic_raw(Path(t)/'raw');path=next(raw.glob('attempts/main-*/observations.jsonl'));rows=lines(path)
                if change=='shape':next(r for r in rows if r['kind']=='full_logits')['numeric']['shape']=[7]
                else:next(r for r in rows if r['kind']=='target_logprob')['details']['raw_argmax_token_id']=0
                replace_lines(path,rows);reseal(raw)
                with self.assertRaises(ValueError):collect_source(self.spec(raw),lambda directory,metadata:metadata)

    def test_failed_job_emits_requested_none_not_zero(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw',failed='rtn');result=collect_source(self.spec(raw),lambda directory,metadata:metadata)
            failed=[r for r in result['tokens'] if r['arm_id']=='rtn']
            self.assertEqual(len(failed),6);self.assertTrue(all(r['status']=='FAILED' and r['target_logprob'] is None and r['full_logits'] is None for r in failed))

    def test_missing_coordinate_remains_unknown_and_unsupported_full_is_null(self):
        with tempfile.TemporaryDirectory() as t:
            raw=synthetic_raw(Path(t)/'raw');path=next(raw.glob('attempts/main-*/observations.jsonl'));rows=lines(path)
            scalar=next(r for r in rows if r['kind']=='target_logprob');rows.remove(scalar)
            full=next(r for r in rows if r['kind']=='full_logits' and r['coordinates']['token_position']!=scalar['coordinates']['token_position'])
            full.update(status='UNSUPPORTED',numeric=None,details={});replace_lines(path,rows);reseal(raw)
            result=collect_source(self.spec(raw),lambda directory,metadata:metadata)
            self.assertTrue(any(r['status']=='NOT_RUN' and r['target_logprob'] is None for r in result['tokens']))
            self.assertTrue(any(r['full_logits_status']=='UNSUPPORTED' and r['full_logits'] is None for r in result['tokens']))

    def test_global_ids_stable_on_relocation_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=synthetic_raw(root/'raw');first=collect_source(self.spec(raw),lambda directory,metadata:metadata)
            moved=root/'relocated';shutil.copytree(raw,moved);copy=collect_source(self.spec(moved),lambda directory,metadata:metadata)
            self.assertEqual([r['record_id'] for r in first['tokens']],[r['record_id'] for r in copy['tokens']])
            progress=read(moved/'progress.json');progress['copy_metadata']='different';(moved/'progress.json').write_text(json.dumps(progress),encoding='utf-8');reseal(moved)
            changed=collect_source(self.spec(moved),lambda directory,metadata:metadata)
            self.assertTrue(set(r['record_id'] for r in first['tokens']).isdisjoint(r['record_id'] for r in changed['tokens']))


if __name__=='__main__':unittest.main()
