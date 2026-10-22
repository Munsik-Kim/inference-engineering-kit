"""Synthetic source contracts; actual MAIN9 evidence is replayed separately."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from inference_kit.analysis.token_probe_source import collect_source, _field_membership
from inference_kit.analysis.tokens import export_tokens, compare_tokens
from inference_kit.core.io import digest, read, write
from inference_kit.metrics.token_diagnostics import aggregate_token_comparisons, compare_token
from bench05c2.test_source import synthetic_raw, lines, replace_lines, reseal


def change_plan(root, change):
    path=root/'PLAN_FROZEN.json';plan=read(path);change(plan['samples']['rows'])
    del plan['plan_identity'];plan['plan_identity']=digest(plan)
    path.write_text(json.dumps(plan,ensure_ascii=False),encoding='utf-8');reseal(root)


def source(root):
    return {'source_id':'membership-fixture','adapter':'token_probe_raw_v1','path':str(root)}


class FieldMembershipTests(unittest.TestCase):
    def test_choice_literal_and_role_have_one_membership_two_origins(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=synthetic_raw(root/'raw')
            change_plan(raw,lambda rows: rows[1]['field_token_coverage'].update(
                gold_choice={'overlapping_positions':[2,3],'selected_positions':[2,3]}))
            result=collect_source(source(raw),lambda directory,metadata:metadata)
            choice=[row for row in result['tokens'] if row['choice_index'] is not None]
            self.assertEqual(len(choice),4);self.assertEqual(len(result['tokens']),12)
            self.assertTrue(all(row['field_tags'].count('gold_choice')==1 for row in choice))
            for row in choice:
                membership=next(m for m in row['field_memberships'] if m['field_key']=='gold_choice')
                self.assertEqual([o['kind'] for o in membership['origins']],['field_token_coverage','choice_reference_role'])
            coverage=result['source']['coverage']
            self.assertEqual(coverage['field_membership_repeated_entries'],4)
            self.assertEqual(coverage['field_membership_reconciled_records'],4)
            write(root/'sources.json',{'schema_version':1,'kind':'token_input_sources','sources':[source(raw)]})
            export_tokens(root/'sources.json',root/'view');analysis=compare_tokens(root/'view','bf16','rtn',root/'analysis')
            self.assertEqual(analysis['coverage']['measured'],6)
            groups=[g for g in analysis['aggregates']['groups'] if g['identity']['task_id']=='generic_beta']
            self.assertEqual(next(g for g in groups if g['field_tag']=='__all__')['measured_comparison_tokens'],2)
            self.assertEqual(next(g for g in groups if g['field_tag']=='gold_choice')['measured_comparison_tokens'],2)

    def test_distinct_overlap_and_explicit_field_key_namespaces_are_preserved(self):
        row={'reference_role':'author_gold','input_ids':[0,1,2,3],
             'field_token_coverage':{'reference:answer':{'overlapping_positions':[2]},
                                     'content:answer':{'overlapping_positions':[2,2]},
                                     'evidence':{'overlapping_positions':[2]}},'value_spans':{}}
        tags,memberships,origins=_field_membership(row,2,None)
        self.assertEqual(tags,['continuation','content:answer','evidence','reference:answer'])
        self.assertEqual(origins,5);self.assertEqual(len(memberships),4)
        self.assertEqual(len(next(m for m in memberships if m['field_key']=='content:answer')['origins']),2)
        reordered=deepcopy(row);reordered['field_token_coverage']=dict(reversed(list(row['field_token_coverage'].items())))
        self.assertEqual(_field_membership(reordered,2,None),(tags,memberships,origins))

    def test_structured_same_label_different_namespace_is_explicitly_unsupported(self):
        for key in ('namespace','field_namespace','role','reference_role','field_key','label'):
            row={'reference_role':'gold','input_ids':[0,1],
                 'field_token_coverage':{'answer':{'overlapping_positions':[1],key:'content'}}}
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'namespace/role unsupported'):
                _field_membership(row,1,None)

    def test_invalid_flat_membership_metadata_is_rejected(self):
        for field,info in (('',{'overlapping_positions':[1]}),('__all__',{'overlapping_positions':[1]}),
              ('continuation',{'overlapping_positions':[1]}),(1,{'overlapping_positions':[1]}),
              ('answer',None),('answer',{'overlapping_positions':[True]}),
              ('answer',{'overlapping_positions':[0]}),('answer',{'overlapping_positions':[2]}),
              ('answer',{'overlapping_positions':'1'}),('answer',{})):
            row={'reference_role':'gold','input_ids':[0,1],'field_token_coverage':{field:info}}
            with self.subTest(field=field,info=info),self.assertRaises(ValueError):_field_membership(row,1,None)

    def test_valid_metadata_keeps_original_observation_ids_values_and_tags(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=synthetic_raw(Path(temp)/'raw');result=collect_source(source(raw),lambda directory,metadata:metadata)
            for row in result['tokens']:
                expected=['continuation','value']+(['gold_choice'] if row['choice_index'] is not None else [])
                self.assertEqual(row['field_tags'],expected)
                original=row['source_observation'];self.assertEqual(row['observation_id'],original['observation_id'])
                self.assertEqual(row['source_numeric_sha256'],original['numeric']['sha256'])
            self.assertEqual(result['source']['coverage']['field_membership_reconciled_records'],0)
            self.assertEqual(result['source']['field_membership_reconciliation'],[])

    def test_conflicting_observation_coordinate_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=synthetic_raw(Path(temp)/'raw');path=next(raw.glob('attempts/main-*/observations.jsonl'))
            observations=lines(path);conflict=deepcopy(observations[0]);conflict['observation_id']='other-observation'
            observations.append(conflict);replace_lines(path,observations);reseal(raw)
            with self.assertRaisesRegex(ValueError,'Duplicate configured observation coordinate'):
                collect_source(source(raw),lambda directory,metadata:metadata)

    def test_validator_stays_strict_and_failed_sources_keep_null_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=synthetic_raw(Path(temp)/'raw',failed='rtn');result=collect_source(source(raw),lambda directory,metadata:metadata)
            seqs={seq['sequence_id']:seq for seq in result['sequences']}
            a=next(row for row in result['tokens'] if row['arm_id']=='bf16')
            b=next(row for row in result['tokens'] if row['arm_id']=='rtn' and row['sample_id']==a['sample_id'] and row['token_position']==a['token_position'])
            self.assertIsNone(b['target_logprob']);paired=compare_token(a,b,seqs[a['sequence_id']],seqs[b['sequence_id']])
            # The absent execution also lacks live/synthetic evidence identity;
            # the existing comparator correctly refuses that pairing.
            self.assertEqual(paired['status'],'INCOMPARABLE')
            self.assertIn('CANDIDATE:NOT_MEASURED',paired['reasons'])
            self.assertIsNone(paired['metrics']['candidate_nll_nat'])
            c=deepcopy(a);c['field_tags'].append(c['field_tags'][0])
            self.assertEqual(compare_token(a,c,seqs[a['sequence_id']],seqs[a['sequence_id']])['status'],'INCOMPARABLE')

    def test_field_overlap_does_not_change_unique_token_or_document_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            raw=synthetic_raw(Path(temp)/'raw');result=collect_source(source(raw),lambda directory,metadata:metadata)
            seqs={seq['sequence_id']:seq for seq in result['sequences']};base=result['tokens'][0];comparisons=[]
            for sample,position,nll,tags in (('doc-a',2,1.,['continuation','left','right']),
                  ('doc-a',3,3.,['continuation','left']),('doc-b',2,10.,['continuation','right'])):
                row=deepcopy(base);seq=deepcopy(seqs[base['sequence_id']]);row.update(sample_id=sample,
                    token_position=position,logit_position=position-1,target_token_id=seq['input_ids'][position],
                    target_logprob=-nll,field_tags=tags,
                    prefix_hash=digest({'token_ids':seq['input_ids'][:position],'token_id_space':row['token_id_space']}))
                comparisons.append(compare_token(row,row,seq,seq))
            groups=aggregate_token_comparisons(comparisons)['groups'];all_group=next(g for g in groups if g['field_tag']=='__all__')
            self.assertEqual(all_group['measured_comparison_tokens'],3)
            self.assertEqual(all_group['measured_documents'],2)
            self.assertAlmostEqual(all_group['nll']['reference_nll_nat']['token_weighted'],14/3)
            self.assertEqual(all_group['nll']['reference_nll_nat']['document_macro'],6.)
            self.assertEqual(sum(g['measured_comparison_tokens'] for g in groups if g['field_tag'] in ('left','right')),4)


if __name__=='__main__':unittest.main()
