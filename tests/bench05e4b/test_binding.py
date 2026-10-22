"""Prepared live recipe tests use synthetic token IDs and no model resources."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import digest
from inference_kit.suites import prepared_live as binding
from inference_kit.suites.suite_plan import compile_suite
from bench05e4a.test_suite_prepared import loaded_fixture, prepared_suite
from bench05e2.test_plan import local_fixture


def fixture(root, count=1):
    sources = loaded_fixture(root, count)
    suite = prepared_suite(sources, 'both')
    suite['metrics'] = ['task_default_v1', 'common_tokens_c1_v1']
    return compile_suite(suite, sources, local_fixture()), sources


class PreparedLiveBindingTests(unittest.TestCase):
    def test_main_and_subset_counts_derive_from_input(self):
        counts = []
        for size in (1, 3):
            with tempfile.TemporaryDirectory() as tmp:
                plan, sources = fixture(Path(tmp), size)
                selected = binding.selected_samples(plan, sources)
                groups = binding.serving_groups(plan, selected)
                probe = binding.probe_inputs(plan, selected, sources['model_metadata'], 'f'*64)
                counts.append((len(selected),sum(len(g['requests']) for g in groups),
                               probe['counts']['target_positions_per_arm'],probe['counts']['full_vocab_positions_per_arm']))
                self.assertEqual(len(groups), 6)
                self.assertEqual(sum(len(g['requests']) for g in groups), size*12)
                self.assertEqual(probe['counts']['samples'], size*3)
                self.assertNotEqual(probe['counts']['target_positions_per_arm'], 110)
                self.assertEqual(probe['counts']['attention_rows_per_arm'], 0)
        self.assertNotEqual(counts[0],counts[1])

    def test_group_paths_jobs_and_nodes_cannot_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources = fixture(Path(tmp),3)
            groups=binding.serving_groups(plan,binding.selected_samples(plan,sources))
            for name in ('group_id','job_id','raw_root'):
                self.assertEqual(len({g[name] for g in groups}),len(groups))
            self.assertEqual(len({g['node_ids'][0] for g in groups}),len(groups))
            self.assertEqual({g['logical_arm_id'] for g in groups},{'baseline','candidate'})
            self.assertTrue(all('qualified_sample_id' in r for g in groups for r in g['requests']))

    def test_preparation_bytes_and_all_token_coordinates_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp));before=deepcopy(sources)
            selected=binding.selected_samples(plan,sources);rows=binding.probe_inputs(plan,selected,sources['model_metadata'],'f'*64)['rows']
            self.assertEqual(sources,before)
            coordinates=('input_ids','prompt','reference_text','prompt_token_boundary','offsets_codepoints',
                         'offsets_utf8_bytes','attention_mask','loss_mask','special_token_mask','targets','full_vocab_target_positions')
            for original,runtime in zip(selected,rows):
                self.assertEqual({k:original['diagnostic'][k] for k in coordinates},{k:runtime[k] for k in coordinates})
                self.assertEqual(runtime['prepared_probe_input_identity'],original['diagnostic']['probe_input_identity'])
                self.assertNotEqual(runtime['probe_input_identity'],runtime['prepared_probe_input_identity'])
                self.assertEqual(runtime['probe_input_identity'],digest({k:v for k,v in runtime.items() if k!='probe_input_identity'}))
                self.assertNotIn('source_bundle_identity',runtime)
                self.assertNotIn('historical_plan',runtime)

    def test_cross_namespace_same_id_is_explicitly_unsupported(self):
        samples=[{'sample_id':'same','source_id':'one','raw_record':{'task_id':'one','task_version':'0.1','dataset_id':'d','split':'x','sample_id':'same'}},
                 {'sample_id':'same','source_id':'two','raw_record':{'task_id':'two','task_version':'0.1','dataset_id':'d','split':'x','sample_id':'same'}}]
        with self.assertRaisesRegex(ValueError,'UNSUPPORTED_SAMPLE_ID_COLLISION'):
            binding.qualified_samples(samples)
        with self.assertRaisesRegex(ValueError,'Duplicate qualified'):
            binding.qualified_samples([samples[0],samples[0]])

    def test_exact_arc_choices_and_order_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp));selected=binding.selected_samples(plan,sources)
            node=next(n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='serve_choice_loglikelihood')
            for mutate in ('missing','duplicate','reordered_body'):
                broken=deepcopy(plan);n=next(n for n in broken['semantic']['graph']['nodes'] if n['node_id']==node['node_id'])
                if mutate=='missing':n['output_contract']['requests'].pop()
                elif mutate=='duplicate':n['output_contract']['requests'][1]=deepcopy(n['output_contract']['requests'][0])
                else:n['output_contract']['requests'][0]['wire_body']['prompt']=[7]
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):binding.serving_groups(broken,selected)

    def test_generation_reference_or_wire_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp));selected=binding.selected_samples(plan,sources)
            node=next(n for n in plan['semantic']['graph']['nodes'] if n['operation_kind']=='serve_generate_until')
            node['output_contract']['requests'][0]['wire_body']['prompt']=[99]
            with self.assertRaisesRegex(ValueError,'prompt-only wire'):binding.serving_groups(plan,selected)

    def test_missing_full_or_empty_target_cannot_be_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp));selected=binding.selected_samples(plan,sources)
            selected[0]['diagnostic']['targets']=[]
            with self.assertRaises(ValueError):binding.probe_inputs(plan,selected,sources['model_metadata'],'f'*64)

    def test_missing_scope_source_or_wrong_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp))
            for change in ('scope','source','identity'):
                value=deepcopy(sources)
                if change=='scope':value['sources'][0]['validation_scope']='REPORT_ONLY'
                elif change=='source':value['sources'][0]['kind']='prepared_suite_slice_v1'
                else:value['data_identity']='a'*64
                with self.subTest(change=change),self.assertRaises(ValueError):binding.selected_samples(plan,value)

    def test_empty_selection_and_wrong_operation_are_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp));plan['semantic']['suite']['tasks'][0]['execution_kind']='serve'
            # Recompiled plans also refuse the missing required diagnostic role;
            # a changed frozen plan fails even before source resolution.
            with self.assertRaises(ValueError):binding.selected_samples(plan,sources)
        with self.assertRaisesRegex(ValueError,'INVALID_EMPTY_PLAN'):binding.qualified_samples([])


    def test_portable_layout_has_exact_provides_and_no_live_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,sources=fixture(Path(tmp),3);layout=binding.binding_layout(plan,sources)
            self.assertEqual(layout['status'],'CONTRACT_VALIDATED_ONLY')
            self.assertEqual(layout['artifact_presence'],'ARTIFACT_PRESENCE_NOT_CHECKED')
            self.assertEqual(layout['counts']['execution_units'],8)
            self.assertEqual(layout['counts']['inference_POSTs'],36)
            self.assertEqual(layout['inference_POSTs'],0)
            self.assertEqual(len({u['node_id'] for u in layout['execution_units']}),8)
            provided=[n for u in layout['execution_units'] for n in u['provides']]
            expected={n['node_id'] for n in plan['semantic']['graph']['nodes'] if n['operation_kind'] in binding.KINDS|{'collect_token_diagnostic'}}
            self.assertEqual(set(provided),expected);self.assertEqual(len(provided),len(expected))

    def test_mismatched_wall_or_cleanup_policy_refused_before_launch(self):
        for field, value in (('gpu_wall_seconds',300),('cleanup_reserve_seconds',121)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                plan,sources=fixture(Path(tmp));spec=deepcopy(plan['semantic']['suite'])
                spec['budget'][field]=value
                changed=compile_suite(spec,sources,local_fixture())
                with self.assertRaisesRegex(ValueError,'UNSUPPORTED_BUDGET_POLICY'):
                    binding.binding_layout(changed,sources)
                # Matching a new policy on both sides does not authorize a
                # lifecycle whose controller still uses the retained policy.
                if field == 'gpu_wall_seconds':
                    metadata=deepcopy(sources['model_metadata'])
                    metadata['token_spec']['budget']['task_gpu_wall_seconds']=value
                    with self.assertRaisesRegex(ValueError,'UNSUPPORTED_BUDGET_POLICY'):
                        binding.validate_budget_compatibility(changed,metadata)

    def test_full_binding_inventory_pin_fails_before_resource_or_artifact_access(self):
        from inference_kit.core.io import write
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,sources=fixture(root);dummy=root/'pin.json';write(dummy,{'kind':'SYNTHETIC_CONTRACT_TEST'})
            local={k:str(dummy) for k in ('prepared_manifest','artifact_receipt','runtime_metadata','runtime_python','server_python','prior_ledger')}
            local.update(schema_version=1,kind='prepared_suite_live_binding_v1',profile=binding.PROFILE,
                prepared_manifest_sha256='a'*64,package_site=str(root),campaign_root=str(root),gpu=0,gpu_uuid='GPU-synthetic',port=19000)
            with patch.object(binding,'resources',side_effect=AssertionError('resource gate must follow input pin')), \
                 patch.object(binding,'inspect_artifacts',side_effect=AssertionError('artifact access must follow input pin')), \
                 self.assertRaisesRegex(ValueError,'inventory SHA'):
                binding.bind_live_plan(plan,root/'SOURCES.json',local,{})
            local['unexpected_loader']='arbitrary.module'
            with self.assertRaisesRegex(ValueError,'Unknown fields'):
                binding.bind_live_plan(plan,root/'SOURCES.json',local,{})

    def test_cpu_helpers_do_not_query_gpu_or_connect_network(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','transformers','vllm','numpy','aiohttp'):
                raise AssertionError('ML import in CPU binding')
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch('builtins.__import__',side_effect=guarded),patch('socket.socket',side_effect=AssertionError('network')):
            plan,sources=fixture(Path(tmp));selected=binding.selected_samples(plan,sources)
            self.assertTrue(binding.serving_groups(plan,selected))
            self.assertTrue(binding.probe_inputs(plan,selected,sources['model_metadata'],'f'*64)['rows'])
