"""Closed, stdlib-only structure replay contracts with synthetic tiny arrays."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from inference_kit.analysis.structure import replay_structure, aggregate, _evidence_positions, _compatible
from inference_kit.core.io import digest, read, write
from inference_kit.metrics.token import log_softmax
from inference_kit.probes.blobs import save
from inference_kit.probes.contracts import Observation, Execution
from inference_kit.suites.bench05b1_run import seal


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def replace_lines(path, values):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in values), encoding='utf-8')


def reseal(root):
    (root/'RAW_MANIFEST.json').unlink(); seal(root, scope='structure_probe_raw_v1')


def raw_fixture(root, *, on_shift=0., missing=None, zero=False):
    root.mkdir(); policy = {'id':'same_pass_saved_f32_v1', 'absolute_tolerance_nat':2e-5,
        'scope':'same-pass saved float32 full logits CPU log_softmax[target] vs saved float32 scalar'}
    rows=[]
    for i in range(2):
        ids=[0,1,2,3]
        row={'task_id':'future_task_'+str(i), 'task_version':'7', 'sample_id':'sample-'+str(i),
            'dataset_identity':'synthetic-data-'+str(i), 'source_id':'authored-source', 'split':'SYNTHETIC',
            'reference_role':'author_gold_reference', 'reference_source':'author_gold', 'policy_id':'unchanged',
            'prompt':'AB', 'prompt_hash':digest('AB'), 'reference_text':'CD', 'reference_hash':digest('CD'), 'full_text':'ABCD',
            'input_ids':ids, 'sequence_identity':digest(ids), 'attention_mask':[1,1,1,1],
            'loss_mask':[0,0,1,1], 'special_token_mask':[0]*4, 'prompt_token_boundary':2,
            'offsets_codepoints':[[i,i+1] for i in range(4)], 'offsets_utf8_bytes':[[i,i+1] for i in range(4)],
            'targets':[{'token_position':t, 'logit_position':t-1, 'target_token_id':ids[t]} for t in (2,3)],
            'full_vocab_target_positions':[2,3]}
        if i==0:
            row['reference_evidence_span']={'status':'AVAILABLE', 'coordinate_space':'joint_input_ids',
                'content':'B', 'content_hash':digest('B'), 'prompt_codepoint_span':[1,2], 'prompt_utf8_byte_span':[1,2],
                'overlapping_token_positions':[1], 'passage_id':'P02', 'overlap_rule':'nonempty_token_interval_intersects_content; include_whole_boundary_token'}
        else: row['reference_evidence_span']={'status':'NOT_APPLICABLE'}
        rows.append(row)
    arms=[{'arm_id':name, 'execution_mode':'REIMPLEMENTED_OR_SYNTHETIC',
           'artifact':{'artifact_identity':'artifact-'+name, 'recipe_identity':'recipe-'+name}} for name in ('reference','candidate')]
    plan={'task':'BENCH05-D1', 'evidence_kind':'synthetic_contract', 'source_identity':{'fixture':'synthetic'},
        'runtime_identity':'synthetic-runtime', 'pair':{'model_id':'synthetic/model', 'revision':'fixed'}, 'arms':arms,
        'samples':{'rows':rows, 'tokenizer_identity':'synthetic-tokenizer', 'token_id_space':'synthetic-space', 'vocabulary_size':8,
            'model_config':{'num_hidden_layers':4, 'num_attention_heads':4, 'num_key_value_heads':2, 'hidden_size':8}},
        'probe_spec':{'normalization':{'dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e'}, 'consistency_policy':policy},
        'structure_spec':{'observer_consistency_policy':{'id':'d1_off_repeat_and_on_v1','absolute_tolerance':1e-5,'passes':['OFF1','OFF2','ON']}},
        'structure_selection':{'layers':[0,2], 'heads':[0,2], 'rows':{r['sample_id']:{'queries':[1,2],
            'query_roles':{'1':['last_prompt'], '2':['before_last_reference']}} for r in rows}}}
    plan['structure_selection']['selection_identity']=digest(plan['structure_selection'])
    plan['plan_identity']=digest(plan); write(root/'PLAN_FROZEN.json',plan); jobs={}
    for arm in arms:
        name=arm['arm_id']; directory=root/'attempts'/('main-'+name+'-attempt1'); directory.mkdir(parents=True)
        execution=Execution('synthetic-d1','attempt1',name,'synthetic/model','fixed',arm['artifact']['artifact_identity'],
            arm['artifact']['recipe_identity'],'synthetic-tokenizer','synthetic-runtime','REIMPLEMENTED_OR_SYNTHETIC','float32','toy',
            'synthetic_contract','MEASURED','observations.jsonl').validate()
        execution['phase']='main'; execution['execution_id']=digest(execution); write(directory/'execution.json',execution)
        observations=[]
        def emit(kind,row,c,values,details):
            identifier=digest({'arm':name,'kind':kind,'coords':c})[:28]
            numeric=save(directory,identifier,values,[len(values)])
            observations.append(Observation(identifier,kind,execution['execution_id'],row['sample_id'],'synthetic_contract',
                'REIMPLEMENTED_OR_SYNTHETIC','MEASURED',c,numeric,details).validate())
        for row in rows:
            base={k:row[k] for k in ('sample_id','task_id','task_version','source_id','split','reference_role','sequence_identity')}
            base.update(tokenizer_identity='synthetic-tokenizer',token_id_space='synthetic-space')
            for pass_id in ('OFF1','OFF2','ON'):
                logits=[i*.25+(on_shift if pass_id=='ON' else 0.) for i in range(8)]; lp=log_softmax(logits)
                for target in row['targets']:
                    c=base|target|{'pass_id':pass_id,'prefix_hash':digest({'token_ids':row['input_ids'][:target['token_position']], 'token_id_space':'synthetic-space'})}
                    details={'normalization_dtype':'float32','probability_stage':'raw_pre_sampling','log_base':'e','consistency_policy':policy,'raw_argmax_token_id':7}
                    emit('target_logprob',row,c,[lp[target['target_token_id']]],details)
                    emit('full_logits',row,c,logits,details|{'full_vocabulary':True})
            for layer in plan['structure_selection']['layers']:
                for query in (1,2):
                    c=base|{'layer':layer,'query_position':query,'query_roles':plan['structure_selection']['rows'][row['sample_id']]['query_roles'][str(query)],
                        'pass_id':'ON','prefix_hash':digest({'token_ids':row['input_ids'][:query+1],'token_id_space':'synthetic-space'})}
                    for head in plan['structure_selection']['heads']:
                        coordinate=c|{'head':head,'kv_head':head//2,'module_path':'model.layers.'+str(layer)+'.self_attn',
                            'key_positions':list(range(4)),'allowed_keys':[i<=query for i in range(4)]}
                        for kind in ('attention_row','attention_output','reconstructed_AV'):
                            if missing==(name,row['sample_id'],kind,layer,head,query): continue
                            values=[1/(query+1) if i<=query else 0. for i in range(4)] if kind=='attention_row' else [0.,0.] if zero else [1.,2.]
                            paths={'attention_row':'reconstructed_from_native_qk','attention_output':'native_returned_SDPA_AV','reconstructed_AV':'reconstructed_AV_from_FP32_row_and_native_V'}
                            emit(kind,row,coordinate,values,{'measurement_path':paths[kind]})
                    emit('output_projection',row,c|{'module_path':'model.layers.'+str(layer)+'.self_attn.o_proj'},[0.]*8 if zero else [1.]*8,{'measurement_path':'native_o_proj_forward_hook_before_residual'})
        replace_lines(directory/'observations.jsonl',observations); write(directory/'result.json',{'status':'MEASURED'})
        jobs['main-'+name]={'status':'COMPLETE','directory':str(directory.relative_to(root))}
    write(root/'progress.json',{'status':'SYNTHETIC_ONLY','jobs':jobs}); seal(root,scope='structure_probe_raw_v1')
    return root


class StructureAnalysisTests(unittest.TestCase):
    def replay(self,raw,output):
        # The new raw source dispatcher is separately contract-tested by C2/D1.
        # Keep these structural fixtures independent of historical helper trees.
        with patch('inference_kit.analysis.tokens.export_tokens',return_value={'status':'EXPORTED'}), \
             patch('inference_kit.analysis.tokens.compare_tokens',return_value={'coverage':{'measured':4}}):
            return replay_structure(raw,output)

    def mutate(self,raw,callback):
        path=next(raw.glob('attempts/main-reference-*/observations.jsonl')); records=lines(path); callback(records)
        replace_lines(path,records); reseal(raw)

    def test_task_neutral_self_pair_and_projection_has_no_head_duplication(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); summary=self.replay(raw_fixture(p/'raw'),p/'analysis')
            self.assertEqual(summary['OBSERVER_INTEGRITY'],'VERIFIED')
            self.assertEqual(summary['coverage']['per_kind']['attention_row']['requested_pairs'],16)
            self.assertEqual(summary['coverage']['per_kind']['output_projection']['requested_pairs'],8)
            self.assertEqual({g['task_id'] for g in summary['aggregates']},{'future_task_0','future_task_1'})
            for r in lines(p/'analysis'/'structure-comparisons.jsonl'):
                self.assertEqual(r['status'],'MEASURED')
                if r['kind']=='attention_row': self.assertEqual(r['metrics']['l1'],0.); self.assertEqual(r['metrics']['js_nat'],0.)
                else: self.assertEqual(r['metrics']['nmse'],0.); self.assertAlmostEqual(r['metrics']['cosine'],1.)
            self.assertTrue(all(r['identity']['head'] is None for r in lines(p/'analysis'/'output-projection-comparisons.jsonl')))
            self.assertEqual(summary['coverage']['same_arm_reconstruction_pairs'],32)

    def test_scalar_full_and_control_receipts_are_new_raw_recalculations(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); self.replay(raw_fixture(p/'raw'),p/'out')
            controls=read(p/'out'/'OBSERVER_CONTROL_RECEIPT.json'); same=read(p/'out'/'SCALAR_FULL_CONSISTENCY_RECEIPT.json')
            self.assertEqual(controls['status_counts'],{'MEASURED':32}); self.assertEqual(controls['maximum_absolute_difference'],0.)
            self.assertEqual(same['status_counts'],{'MEASURED':24}); self.assertLess(same['maximum_absolute_error_nat'],2e-5)

    def test_real_c1_dispatch_exports_only_ON_and_uses_existing_token_metrics(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); summary=replay_structure(raw_fixture(p/'raw'),p/'out')
            self.assertEqual(summary['ON_token_coverage']['requested_pair_coordinates'],4)
            self.assertEqual(summary['ON_token_coverage']['measured'],4)
            self.assertEqual(read(p/'out'/'token-view'/'COVERAGE.json')['records'],8)
            self.assertEqual(read(p/'out'/'token-view'/'COVERAGE.json')['full_logits_records'],8)
            self.assertTrue(all(r['pass_id']=='ON' for r in lines(p/'out'/'token-view'/'token_records.jsonl')))
            self.assertEqual(read(p/'out'/'TOKEN_COMPARISON_STATUS.json')['status'],'MEASURED')

    def test_on_logit_shift_preserves_scalar_but_marks_observer_integrity_unverified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); summary=self.replay(raw_fixture(p/'raw',on_shift=.1),p/'out')
            self.assertEqual(summary['OBSERVER_INTEGRITY'],'INTEGRITY_UNVERIFIED')
            self.assertTrue(all(r['status']=='INTEGRITY_UNVERIFIED' and r['metrics'] is None for r in lines(p/'out'/'structure-comparisons.jsonl')))
            self.assertIn('TOLERANCE_VIOLATION',read(p/'out'/'OBSERVER_CONTROL_RECEIPT.json')['status_counts'])
            self.assertEqual(read(p/'out'/'SCALAR_FULL_CONSISTENCY_RECEIPT.json')['status_counts'],{'MEASURED':24})

    def test_missing_coordinate_does_not_shrink_requested_denominator_or_become_zero(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); missing=('candidate','sample-0','attention_row',0,0,1)
            summary=self.replay(raw_fixture(p/'raw',missing=missing),p/'out')
            coverage=summary['coverage']['per_kind']['attention_row']
            self.assertEqual(coverage,{'requested_pairs':16,'status_counts':{'UNKNOWN':1,'MEASURED':15}})
            missing=[r for r in lines(p/'out'/'structure-comparisons.jsonl') if r['status']=='UNKNOWN']
            self.assertEqual(len(missing),1); self.assertIsNone(missing[0]['metrics'])

    def test_zero_energy_vectors_keep_null_nmse_and_cosine(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); self.replay(raw_fixture(p/'raw',zero=True),p/'out')
            rows=lines(p/'out'/'output-projection-comparisons.jsonl')
            self.assertTrue(all(r['metrics']['nmse'] is None and r['metrics']['cosine'] is None for r in rows))

    def test_missing_control_pass_remains_unknown_and_not_normal_model_pair(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); raw=raw_fixture(p/'raw')
            self.mutate(raw,lambda rows:rows.__setitem__(slice(None),[r for r in rows if r['coordinates']['pass_id']!='OFF2']))
            summary=self.replay(raw,p/'out'); receipt=read(p/'out'/'OBSERVER_CONTROL_RECEIPT.json')
            self.assertEqual(summary['OBSERVER_INTEGRITY'],'INTEGRITY_UNVERIFIED')
            self.assertEqual(receipt['requested_comparisons'],32); self.assertEqual(receipt['status_counts']['UNKNOWN'],8)
            self.assertEqual(read(p/'out'/'TOKEN_COMPARISON_STATUS.json')['status'],'INTEGRITY_UNVERIFIED')

    def test_scalar_full_violation_preserves_coordinate_and_blocks_normal_token_comparison(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); raw=raw_fixture(p/'raw'); path=next(raw.glob('attempts/main-reference-*/observations.jsonl')); records=lines(path)
            target=next(r for r in records if r['kind']=='target_logprob' and r['coordinates']['pass_id']=='ON')
            target['numeric']=save(path.parent,'known-invalid-scalar',[-.1],[1]); replace_lines(path,records); reseal(raw)
            with patch('inference_kit.analysis.tokens.export_tokens',side_effect=AssertionError('invalid same-pass input cannot be exported as normal')):
                summary=replay_structure(raw,p/'out')
            receipt=read(p/'out'/'SCALAR_FULL_CONSISTENCY_RECEIPT.json')
            self.assertEqual(receipt['status_counts']['TOLERANCE_VIOLATION'],1)
            self.assertEqual(summary['OBSERVER_INTEGRITY'],'INTEGRITY_UNVERIFIED')
            invalid=[r for r in lines(p/'out'/'scalar-full-consistency.jsonl') if r['status']=='TOLERANCE_VIOLATION']
            self.assertEqual(invalid[0]['pass_id'],'ON'); self.assertEqual(invalid[0]['sample_id'],'sample-0')

    def test_changed_predeclared_observer_tolerance_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); raw=raw_fixture(p/'raw'); plan=read(raw/'PLAN_FROZEN.json'); plan.pop('plan_identity')
            plan['structure_spec']['observer_consistency_policy']['absolute_tolerance']=1.
            plan['plan_identity']=digest(plan); (raw/'PLAN_FROZEN.json').write_text(json.dumps(plan),encoding='utf-8'); reseal(raw)
            with self.assertRaisesRegex(ValueError,'Observer control policy'): self.replay(raw,p/'out')

    def test_evidence_content_mass_uses_validated_span_and_no_answer_is_not_zero(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); self.replay(raw_fixture(p/'raw'),p/'out')
            rows=lines(p/'out'/'evidence-content-mass.jsonl'); measured=[r for r in rows if r['status']=='MEASURED']
            self.assertTrue(measured); self.assertTrue(all(r['evidence_token_count']==1 and r['passage_id']=='P02' for r in measured))
            self.assertEqual({round(r['mass'],5) for r in measured},{.5,round(1/3,5)})
            absent=[r for r in rows if r['status']=='NOT_APPLICABLE']; self.assertTrue(absent); self.assertTrue(all(r['mass'] is None for r in absent))

    def test_wrong_prefix_query_head_mask_or_full_axis_rejected(self):
        mutations={'prefix_hash':'different','query_position':0,'head':1,'kv_head':1,'key_positions':[1,2,3,4],'allowed_keys':[True]*4}
        for key,value in mutations.items():
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                p=Path(t); raw=raw_fixture(p/'raw')
                def change(rows): next(r for r in rows if r['kind']=='attention_row')['coordinates'][key]=value
                self.mutate(raw,change)
                with self.assertRaises(ValueError): self.replay(raw,p/'out')

    def test_corrupted_blob_and_duplicate_observation_are_rejected(self):
        for mode in ('blob','duplicate'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                p=Path(t); raw=raw_fixture(p/'raw')
                if mode=='duplicate': self.mutate(raw,lambda rows:rows.append(copy.deepcopy(rows[-1])))
                else:
                    blob=next(raw.glob('attempts/*/blobs/*.f32')); blob.write_bytes(b'corrupt'); reseal(raw)
                with self.assertRaises(ValueError): self.replay(raw,p/'out')

    def test_partial_job_cannot_promote_saved_values_to_valid_model_comparison(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); raw=raw_fixture(p/'raw'); progress=read(raw/'progress.json'); progress['jobs']['main-candidate']['status']='FAILED'
            (raw/'progress.json').write_text(json.dumps(progress),encoding='utf-8'); reseal(raw)
            summary=self.replay(raw,p/'out'); self.assertEqual(summary['OBSERVER_INTEGRITY'],'INTEGRITY_UNVERIFIED')
            self.assertTrue(all(r['status']=='INTEGRITY_UNVERIFIED' for r in lines(p/'out'/'structure-comparisons.jsonl')))

    def test_cross_coordinate_comparison_reasons_are_not_ignored(self):
        a={'kind':'attention_row','coordinates':{'prefix_hash':'a','head':0},'details':{'measurement_path':'reconstructed_from_native_qk'}}
        b=copy.deepcopy(a); b['coordinates']['prefix_hash']='b'; b['coordinates']['head']=1
        self.assertEqual(set(_compatible(a,b,{})),{'prefix_hash','head'})

    def test_document_macro_is_not_row_weighted(self):
        values=[{'identity':{'task_id':'task','task_version':'1','layer':0,'sample_id':s},'kind':'attention_row','status':'MEASURED','metrics':{'js_nat':v}} for s,v in [('a',0.),('b',1.),('b',1.),('b',1.)]]
        group=aggregate(values)[0]['metrics']['js_nat']
        self.assertEqual(group['row_mean'],.75); self.assertEqual(group['document_macro_mean'],.5)

    def test_evidence_hash_utf8_offset_and_overlap_mismatch_rejected(self):
        row={'prompt':'한글','prompt_token_boundary':2,'offsets_codepoints':[[0,1],[1,2]],'reference_evidence_span':{'status':'AVAILABLE','coordinate_space':'joint_input_ids',
            'content':'글','content_hash':digest('글'),'prompt_codepoint_span':[1,2],'prompt_utf8_byte_span':[3,6],'overlapping_token_positions':[1]}}
        self.assertEqual(_evidence_positions(row)['positions'],[1])
        for field,value in (('content_hash','bad'),('prompt_utf8_byte_span',[1,2]),('overlapping_token_positions',[0])):
            with self.subTest(field=field):
                bad=copy.deepcopy(row); bad['reference_evidence_span'][field]=value
                with self.assertRaises((ValueError,UnicodeDecodeError)): _evidence_positions(bad)

    def test_nested_zip_replay_does_not_execute_python_and_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); raw=raw_fixture(p/'raw'); archive=p/'evidence.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for source in raw.rglob('*'):
                    if source.is_file(): z.write(source,'nested/run/'+source.relative_to(raw).as_posix())
                z.writestr('trap.py','raise RuntimeError("never execute archive code")')
            self.assertEqual(self.replay(archive,p/'out')['CPU_REPLAY'],'RAW_RECALCULATED')
            with self.assertRaises(ValueError): self.replay(archive,p/'out')

    def test_safe_archive_rejects_traversal_before_reading_raw(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); archive=p/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape','bad')
            with self.assertRaises(ValueError): self.replay(archive,p/'out')

    def test_raw_plan_mismatch_and_unmanifested_numeric_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); raw=raw_fixture(p/'raw'); plan=read(raw/'PLAN_FROZEN.json'); plan['samples']['token_id_space']='other'
            (raw/'PLAN_FROZEN.json').write_text(json.dumps(plan),encoding='utf-8'); reseal(raw)
            with self.assertRaisesRegex(ValueError,'plan identity'):self.replay(raw,p/'out')


if __name__=='__main__': unittest.main()
