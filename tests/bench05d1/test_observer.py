"""Multi-layer interface/hook contracts; tiny CPU tensors do not claim live GQA."""
import contextlib
import unittest
from unittest.mock import patch

from inference_kit.probes.structure_observer import StructureObserver, selected_temporary_upper_bytes


class Tensor:
    def __init__(self, shape, values=None):
        self.shape = shape; self.ndim = len(shape); self.dtype = 'torch.bfloat16'; self._version = 0
        self.values = values
    def __getitem__(self, key):
        return Tensor((self.shape[-1],))
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.shape, self.values)
    def tolist(self): return self.values


class Hook:
    def __init__(self, owner, callback): self.owner = owner; self.callback = callback
    def remove(self): self.owner.hooks.remove(self)


class Projection:
    def __init__(self, fail=False): self.hooks = []; self.fail = fail
    def register_forward_hook(self, callback):
        if self.fail: raise RuntimeError('hook registration failed')
        hook = Hook(self, callback); self.hooks.append(hook); return hook
    def emit(self, output):
        for hook in list(self.hooks): hook.callback(self, (), output)


class Module:
    is_causal = True
    def __init__(self, fail=False): self.o_proj = Projection(fail)


def fake_row(module, q, k, v, mask, head, query, scaling, causal=None):
    allowed = [index <= query for index in range(k.shape[2])]
    return Tensor((k.shape[2],)), Tensor((q.shape[-1],)), Tensor((k.shape[2],), allowed), head // (q.shape[1] // k.shape[1])


class LocalInterface:
    def __init__(self, previous): self.previous = previous; self._local_mapping = {}
    def __getitem__(self, key): return self._local_mapping.get(key, self.previous)
    def __setitem__(self, key, value): self._local_mapping[key] = value
    def __delitem__(self, key): del self._local_mapping[key]


class ObserverContractTests(unittest.TestCase):
    def setUp(self):
        self.modules = {0: Module(), 2: Module(), 4: Module()}; self.events = []; self.native_calls = []
        self.selection = {'layers': [0, 2, 4], 'heads': [0, 2], 'queries': [1, 3],
                          'query_roles': {'1': ['last_prompt'], '3': ['before_last_reference', 'alias']}}
        def native(module, q, k, v, mask, **kwargs):
            self.native_calls.append(module); result = (Tensor((1, q.shape[2], q.shape[1], q.shape[3])), None)
            self.last_result = result; return result
        self.previous = native; self.interface = {'sdpa': native}
        self.q = Tensor((1, 4, 5, 3)); self.k = self.v = Tensor((1, 2, 5, 3))
        self.patches = contextlib.ExitStack()
        self.patches.enter_context(patch('inference_kit.probes.structure_observer.selected_torch_row', side_effect=fake_row))
        self.patches.enter_context(patch('inference_kit.probes.structure_observer._row_statistics', return_value={'raw_row_sum': 1., 'future_probability_mass': 0., 'masked_probability_mass': 0.}))
        self.addCleanup(self.patches.close)
    def observer(self, consume=None, **kwargs):
        return StructureObserver(self.interface, self.modules, self.selection, consume or (lambda *event: self.events.append(event)), **kwargs)
    def call(self, module=None):
        return self.interface['sdpa'](module or self.modules[0], self.q, self.k, self.v, None, scaling=0.5)

    def test_one_dispatch_three_layers_native_called_once_per_layer_unchanged_return(self):
        with self.observer() as observer:
            dispatch = self.interface['sdpa']
            for module in self.modules.values():
                result = self.call(module); self.assertIs(result, self.last_result)
                self.assertIs(self.interface['sdpa'], dispatch)
            self.assertEqual(observer.calls_by_layer, {0: 1, 2: 1, 4: 1})
        self.assertEqual(len(self.native_calls), 3)
        self.assertEqual(sum(event[0] == 'attention_row' for event in self.events), 12)
        self.assertTrue(observer.restored and observer.hooks_removed and observer.no_inplace_changes)
        self.assertIs(self.interface['sdpa'], self.previous)

    def test_unselected_module_native_call_has_no_observations(self):
        with self.observer() as observer: result = self.call(Module())
        self.assertIs(result, self.last_result); self.assertFalse(self.events); self.assertEqual(observer.calls, 0)

    def test_projection_headless_count_and_query_aliases(self):
        with self.observer() as observer:
            for module in self.modules.values(): module.o_proj.emit(Tensor((1, 5, 12)))
        self.assertEqual(len(self.events), 6)
        for kind, coordinate, tensor, details in self.events:
            self.assertEqual(kind, 'output_projection'); self.assertNotIn('head', coordinate)
            self.assertNotIn('kv_head', coordinate); self.assertIn('before residual', details['observation_point'])
            self.assertEqual(coordinate['query_roles'], self.selection['query_roles'][str(coordinate['query_position'])])

    def test_consumer_receives_distinct_detached_cpu_copy(self):
        native = Tensor((1, 5, 12)); copies = []
        with self.observer(lambda *event: copies.append(event[2])):
            self.modules[0].o_proj.emit(native)
        self.assertEqual(len(copies), 2); self.assertTrue(all(copy is not native for copy in copies))
        self.assertEqual(native._version, 0)

    def test_registration_failure_removes_already_installed_hooks(self):
        self.modules[2] = Module(fail=True)
        observer = self.observer()
        with self.assertRaisesRegex(RuntimeError, 'registration'):
            with observer: self.fail('must not enter')
        self.assertFalse(self.modules[0].o_proj.hooks); self.assertTrue(observer.restored and observer.hooks_removed)
        self.assertIs(self.interface['sdpa'], self.previous)

    def test_forward_exception_restores_hooks_and_interface(self):
        with self.assertRaisesRegex(RuntimeError, 'forward'):
            with self.observer() as observer: raise RuntimeError('forward fixture')
        self.assertTrue(observer.restored); self.assertTrue(all(not module.o_proj.hooks for module in self.modules.values()))

    def test_callback_exception_restores_before_escaping_dispatch(self):
        def fail(*args): raise RuntimeError('callback fixture')
        with self.observer(fail) as observer:
            with self.assertRaisesRegex(RuntimeError, 'callback'): self.call()
            self.assertIs(self.interface['sdpa'], self.previous)
            self.assertTrue(observer.restored and observer.hooks_removed)
        self.assertTrue(all(not module.o_proj.hooks for module in self.modules.values()))

    def test_projection_callback_exception_restores_all_hooks(self):
        def fail(*args): raise RuntimeError('projection callback')
        with self.observer(fail) as observer:
            with self.assertRaisesRegex(RuntimeError, 'projection'): self.modules[0].o_proj.emit(Tensor((1, 5, 12)))
            self.assertIs(self.interface['sdpa'], self.previous)
        self.assertTrue(observer.hooks_removed)

    def test_global_fallback_mapping_is_deleted_instead_of_shadowed(self):
        self.interface = LocalInterface(self.previous)
        with self.observer() as observer: self.call()
        self.assertEqual(self.interface._local_mapping, {}); self.assertTrue(observer.restored)

    def test_invalid_selection_duplicate_module_and_missing_layer_refused(self):
        variants = [dict(self.selection, layers=[]), dict(self.selection, heads=[0, 0]), dict(self.selection, queries=[True]),
                    dict(self.selection, queries=[-1]), dict(self.selection, layers=[8])]
        for selection in variants:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                StructureObserver(self.interface, self.modules, selection, lambda *args: None)
        self.modules[2] = self.modules[0]
        with self.assertRaisesRegex(ValueError, 'module identity'): self.observer()

    def test_head_query_native_axes_scaling_and_buffer_checked_before_native_call(self):
        for selection, q, cap, scaling in ((dict(self.selection, heads=[4]), self.q, 99999, .5),
                    (dict(self.selection, queries=[5]), self.q, 99999, .5),
                    (self.selection, Tensor((2, 4, 5, 3)), 99999, .5),
                    (self.selection, self.q, 1, .5), (self.selection, self.q, 99999, float('nan'))):
            with self.subTest(selection=selection, shape=q.shape, cap=cap):
                self.q = q
                with StructureObserver(self.interface, self.modules, selection, lambda *args: None, buffer_cap=cap) as observer:
                    with self.assertRaises(ValueError):
                        self.interface['sdpa'](self.modules[0], q, self.k, self.v, None, scaling=scaling)
                    self.assertTrue(observer.restored)
        self.assertEqual(self.native_calls, [])

    def test_all_masked_row_is_explicit_and_native_av_still_separate(self):
        def masked(*args): return None, None, Tensor((5,), [False] * 5), 0
        with patch('inference_kit.probes.structure_observer.selected_torch_row', side_effect=masked):
            with self.observer(): self.call()
        self.assertEqual(len(self.events), 12)
        for kind, coord, tensor, detail in self.events:
            if kind in ('attention_row', 'reconstructed_AV'):
                self.assertIsNone(tensor); self.assertEqual(detail['status'], 'NOT_APPLICABLE')
                self.assertEqual(detail['reason'], 'ALL_MASKED_ROW')
            else: self.assertTrue(detail['all_masked_row']); self.assertEqual(detail['output_source'], 'native_returned_SDPA_AV')

    def test_qkv_inplace_native_mutation_rejected_and_context_restored(self):
        def mutate(module, q, k, v, mask, **kwargs):
            q._version += 1
            return Tensor((1, q.shape[2], q.shape[1], q.shape[3])), None
        self.interface['sdpa'] = mutate
        with self.observer() as observer:
            with self.assertRaisesRegex(ValueError, 'in-place'): self.call()
        self.assertFalse(observer.no_inplace_changes); self.assertIs(self.interface['sdpa'], mutate)

    def test_raw_row_coordinates_native_gqa_mapping_and_metadata(self):
        with self.observer() as observer: self.call()
        rows = [event for event in self.events if event[0] == 'attention_row']
        self.assertEqual({row[1]['kv_head'] for row in rows}, {0, 1})
        self.assertEqual(rows[0][1]['key_positions'], list(range(5)))
        self.assertEqual(rows[0][1]['allowed_keys'], [True, True, False, False, False])
        self.assertEqual(rows[0][3]['probability_source'], 'reconstructed_from_native_qk')
        self.assertEqual(observer.max_selected_temporary_bytes, selected_temporary_upper_bytes(5, 3))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional existing Torch CPU runtime required; no installation or GPU')
class ObserverTorchOracleTests(unittest.TestCase):
    def exercise(self, heads=4, kv_heads=2, all_masked=False):
        import torch.nn.functional as functional
        torch.manual_seed(37)
        class Attention:
            is_causal = True
            def __init__(self): self.o_proj = torch.nn.Linear(heads * 3, heads * 3, bias=False)
        modules = {0: Attention(), 2: Attention(), 4: Attention()}
        q = torch.randn(1, heads, 5, 3); k = torch.randn(1, kv_heads, 5, 3); v = torch.randn(1, kv_heads, 5, 3)
        mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril(); mask[:, :, :, 0] = False
        if all_masked: mask.zero_()
        events = []; native_calls = []
        def native(module, q, k, v, mask, **kwargs):
            native_calls.append(module)
            output = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.,
                                                             scale=kwargs['scaling'], enable_gqa=heads != kv_heads)
            return output.transpose(1, 2), None
        mapping = {'sdpa': native}
        selection = {'layers': [0, 2, 4], 'heads': [0, heads // 2], 'queries': [2, 4],
                     'query_roles': {'2': ['last_prompt'], '4': ['before_last_reference']}}
        with torch.no_grad(), StructureObserver(mapping, modules, selection, lambda *event: events.append(event)) as observer:
            for module in modules.values():
                expected = native(module, q, k, v, mask, scaling=.6)
                before_count = len(native_calls)
                output = mapping['sdpa'](module, q, k, v, mask, scaling=.6)
                self.assertEqual(len(native_calls), before_count + 1)
                self.assertTrue(torch.equal(output[0], expected[0]))
                module.o_proj(output[0].reshape(1, 5, heads * 3))
        self.assertTrue(observer.restored and observer.hooks_removed and observer.no_inplace_changes)
        self.assertTrue(all(not module.o_proj._forward_hooks for module in modules.values()))
        self.assertFalse(torch.cuda.is_initialized())
        return events

    def test_tiny_mha_and_gqa_selected_rows_equal_full_cpu_oracle(self):
        for heads, kv_heads in ((2, 2), (4, 2)):
            events = self.exercise(heads, kv_heads)
            for kind, coordinate, numeric, detail in events:
                if kind == 'attention_row':
                    self.assertAlmostEqual(float(numeric.sum()), 1., places=6)
                    self.assertEqual(float(numeric[coordinate['query_position'] + 1:].sum()), 0.)
                    self.assertEqual(float(numeric[~torch.tensor(coordinate['allowed_keys'])].sum()), 0.)
                if kind == 'reconstructed_AV':
                    native = next(event[2] for event in events if event[0] == 'attention_output' and event[1] == coordinate)
                    self.assertTrue(torch.allclose(numeric, native, atol=2e-6, rtol=1e-6))
            self.assertEqual(sum(event[0] == 'output_projection' for event in events), 6)

    def test_global_query_causal_index_without_explicit_mask(self):
        from inference_kit.probes.attention import selected_torch_row
        q = k = v = torch.ones(1, 1, 5, 2)
        row, av, allowed, kv = selected_torch_row(type('M', (), {'is_causal': True})(), q, k, v, None, 0, 3, 1.)
        self.assertEqual(allowed.tolist(), [True, True, True, True, False]); self.assertEqual(row.tolist(), [.25, .25, .25, .25, 0.])
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_masked_tiny_cpu_is_not_a_fake_zero_probability_row(self):
        events = self.exercise(all_masked=True)
        rows = [event for event in events if event[0] == 'attention_row']
        self.assertTrue(all(event[2] is None and event[3]['status'] == 'NOT_APPLICABLE' for event in rows))
