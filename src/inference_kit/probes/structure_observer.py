"""One temporary SDPA dispatch for selected diagnostic layers, heads and queries.

The original interface is called once and its exact return object is preserved.
Only selected detached CPU copies reach consumers; this observer retains no GPU
activation tensor. This is the reviewed Smol SDPA path, not a new attention kernel.
"""
import math

from ..core.io import require
from .attention import selected_torch_row


def selected_temporary_upper_bytes(length, head_dim):
    """Conservative selected-row FP32 K/V conversion, scores/mask and AV bound.

    Native Q/K/V and native SDPA workspace already exist in the model and are
    excluded. K/V casts are included although their lifetimes need not overlap.
    """
    require(type(length) is int and length > 0 and type(head_dim) is int and head_dim > 0,
            'Invalid selected observer tensor dimensions')
    return 2 * length * head_dim * 4 + length * 21 + head_dim * 12


def _cpu_copy(value):
    return value.detach().cpu().clone()


def _row_statistics(row, allowed, query):
    import torch
    require(bool(torch.isfinite(row).all()), 'Non-finite reconstructed attention row')
    require(not bool((row[~allowed] != 0).any()), 'Masked attention probability is nonzero')
    require(not bool((row[query + 1:] != 0).any()), 'Future attention probability is nonzero')
    return {'raw_row_sum': float(row.sum()), 'future_probability_mass': float(row[query + 1:].sum()),
            'masked_probability_mass': float(row[~allowed].sum()), 'allowed_key_count': int(allowed.sum())}


class StructureObserver:
    """Install one SDPA dispatch plus one projection hook per selected layer.

    ``selection`` has layers/heads/queries and query_roles keyed by decimal query
    index. The consume callback receives (kind, coordinates, detached CPU copy,
    details); output_projection coordinates intentionally have no head axis.
    """
    def __init__(self, interface, modules_by_layer, selection, consume, *,
                 buffer_limit_bytes=256 * 1024 ** 2, buffer_cap=None):
        if buffer_cap is not None:
            require(buffer_limit_bytes == 256 * 1024 ** 2, 'Duplicate observer buffer limit')
            buffer_limit_bytes = buffer_cap
        require(type(buffer_limit_bytes) is int and buffer_limit_bytes > 0, 'Invalid observer buffer cap')
        self.interface, self.modules, self.selection, self.consume = interface, dict(modules_by_layer), dict(selection), consume
        self.buffer_limit_bytes = buffer_limit_bytes
        self.layers = tuple(selection.get('layers', ()))
        self.heads = tuple(selection.get('heads', ()))
        self.queries = tuple(selection.get('queries', ()))
        for label, values in (('layers', self.layers), ('heads', self.heads), ('queries', self.queries)):
            require(values and all(type(x) is int and x >= 0 for x in values) and len(set(values)) == len(values),
                    'Invalid/duplicate selected ' + label)
        require(set(self.layers) <= self.modules.keys(), 'Selected layer module is missing')
        require(len({id(self.modules[layer]) for layer in self.layers}) == len(self.layers), 'Duplicate selected module identity')
        self.module_layers = {id(self.modules[layer]): layer for layer in self.layers}
        self.roles = {}
        for query in self.queries:
            roles = selection.get('query_roles', {}).get(str(query), [])
            require(isinstance(roles, (list, tuple)) and all(isinstance(role, str) and role for role in roles)
                    and len(set(roles)) == len(roles), 'Invalid query role aliases')
            self.roles[query] = list(roles)
        self.calls_by_layer = {layer: 0 for layer in self.layers}
        self.projection_calls_by_layer = {layer: 0 for layer in self.layers}
        self.restored = False
        self.hooks_removed = False
        self.no_inplace_changes = True
        self.max_selected_temporary_bytes = 0
        self._hooks = []
        self._entered = False
        self._installed = False
        self._closed = False

    @property
    def calls(self):
        return sum(self.calls_by_layer.values())

    def _path(self, layer):
        return self.selection.get('module_paths', {}).get(str(layer), 'model.layers.' + str(layer) + '.self_attn')

    def _check_inputs(self, module, q, k, v, mask, kwargs):
        require(q.ndim == k.ndim == v.ndim == 4 and q.shape[0] == k.shape[0] == v.shape[0] == 1,
                'Structure observer requires batch1 B,H,T,D')
        require(k.shape == v.shape and q.shape[2] == k.shape[2] and q.shape[3] == k.shape[3]
                and k.shape[1] > 0 and q.shape[1] % k.shape[1] == 0, 'Unreviewed Q/K/V length or head mapping')
        require(all(head < q.shape[1] for head in self.heads) and all(query < q.shape[2] for query in self.queries),
                'Selected head/query is outside native tensor')
        if mask is not None:
            require(mask.ndim == 4 and tuple(mask.shape[-2:]) == (q.shape[2], k.shape[2])
                    and mask.shape[0] == 1 and mask.shape[1] in (1, q.shape[1]), 'Unreviewed native mask axes')
        scaling = kwargs.get('scaling')
        require(isinstance(scaling, (int, float)) and not isinstance(scaling, bool)
                and math.isfinite(scaling) and scaling > 0, 'Missing/non-finite native attention scaling')
        require(kwargs.get('softcap') in (None, 0) and kwargs.get('sliding_window') in (None, 0),
                'Softcap/sliding-window structure observation unsupported')
        require(getattr(module, 'q_norm', None) is None and getattr(module, 'k_norm', None) is None,
                'QK-normalized structure observation unsupported')
        needed = selected_temporary_upper_bytes(k.shape[2], k.shape[3])
        require(needed <= self.buffer_limit_bytes, 'Selected observer temporary buffer cap')
        self.max_selected_temporary_bytes = max(self.max_selected_temporary_bytes, needed)

    def _emit(self, kind, coordinates, tensor, details):
        # Never expose a view of native storage to consumer code.
        self.consume(kind, coordinates, None if tensor is None else _cpu_copy(tensor), details)

    def _projection(self, layer):
        def projection(module, args, output):
            try:
                require(output.ndim == 3 and output.shape[0] == 1
                        and all(query < output.shape[1] for query in self.queries), 'Native o_proj output axes differ')
                version = getattr(output, '_version', None)
                self.projection_calls_by_layer[layer] += 1
                for query in self.queries:
                    coord = {'layer': layer, 'module_path': self._path(layer) + '.o_proj',
                             'query_position': query, 'query_roles': self.roles[query]}
                    self._emit('output_projection', coord, output[0, query],
                               {'output_source': 'native_o_proj_forward_hook', 'measurement_path': 'native_o_proj_forward_hook', 'observation_point': 'after all-head output projection, before residual addition',
                                'raw_dtype': str(output.dtype), 'head_axis': 'NOT_APPLICABLE', 'status': 'MEASURED'})
                require(version == getattr(output, '_version', None), 'Observer modified native projection output')
            except BaseException:
                self.no_inplace_changes = self.no_inplace_changes and getattr(output, '_version', None) == locals().get('version', getattr(output, '_version', None))
                self._restore()
                raise
        return projection

    def __enter__(self):
        require(not self._entered, 'Structure observer context cannot be entered twice')
        self._entered = True
        self.previous = self.interface['sdpa']
        local = getattr(self.interface, '_local_mapping', None)
        self.had_local = True if local is None else 'sdpa' in local

        def observed(module, q, k, v, mask, **kwargs):
            if id(module) not in self.module_layers:
                return self.previous(module, q, k, v, mask, **kwargs)
            layer = self.module_layers[id(module)]
            originals = (q, k, v, mask)
            versions = [getattr(value, '_version', None) for value in originals]
            try:
                self._check_inputs(module, q, k, v, mask, kwargs)
                result = self.previous(module, q, k, v, mask, **kwargs)
                require(isinstance(result, (tuple, list)) and result, 'Native SDPA result contract differs')
                native = result[0]
                require(tuple(native.shape) == (1, q.shape[2], q.shape[1], q.shape[3]), 'Installed native SDPA output axes differ')
                native_version = getattr(native, '_version', None)
                self.calls_by_layer[layer] += 1
                for head in self.heads:
                    for query in self.queries:
                        row, reconstructed, allowed, kv = selected_torch_row(module, q, k, v, mask, head, query, kwargs['scaling'], kwargs.get('is_causal'))
                        coord = {'layer': layer, 'module_path': self._path(layer), 'head': head, 'kv_head': kv,
                                 'query_position': query, 'query_roles': self.roles[query],
                                 'key_positions': list(range(k.shape[2])), 'allowed_keys': allowed.cpu().tolist()}
                        details = {'qk_point': 'native_post_RoPE; verified Smol has no QK norm/softcap',
                                   'scaling': kwargs['scaling'], 'probability_source': 'reconstructed_from_native_qk', 'measurement_path': 'reconstructed_from_native_qk',
                                   'normalization_dtype': 'float32', 'native_qkv_dtype': str(q.dtype),
                                   'mask_source': 'native_sdpa_mask' if mask is not None else 'native_sdpa_is_causal',
                                   'causal_index_policy': 'global sequence query index',
                                   'key_scope': 'all native keys, masked/future keys retained as zeros',
                                   'selected_temporary_upper_bytes': self.max_selected_temporary_bytes}
                        if row is None:
                            self._emit('attention_row', coord, None, details | {'status': 'NOT_APPLICABLE', 'reason': 'ALL_MASKED_ROW', 'raw_row_sum': None})
                            self._emit('reconstructed_AV', coord, None, details | {'status': 'NOT_APPLICABLE', 'reason': 'ALL_MASKED_ROW', 'measurement_path': 'reconstructed_AV_from_FP32_row_and_native_V'})
                        else:
                            details.update(_row_statistics(row, allowed, query))
                            self._emit('attention_row', coord, row, details | {'status': 'MEASURED'})
                            self._emit('reconstructed_AV', coord, reconstructed, details | {'status': 'MEASURED', 'output_source': 'reconstructed_AV_from_FP32_row_and_native_V', 'measurement_path': 'reconstructed_AV_from_FP32_row_and_native_V'})
                        self._emit('attention_output', coord, native[0, query, head], details | {'status': 'MEASURED',
                                   'output_source': 'native_returned_SDPA_AV', 'measurement_path': 'native_returned_SDPA_AV', 'raw_dtype': str(native.dtype), 'all_masked_row': row is None})
                        del row, reconstructed, allowed
                unchanged = versions == [getattr(value, '_version', None) for value in originals] and native_version == getattr(native, '_version', None)
                self.no_inplace_changes = self.no_inplace_changes and unchanged
                require(unchanged, 'Observer changed native Q/K/V/mask/return in-place')
                return result
            except BaseException:
                unchanged = versions == [getattr(value, '_version', None) for value in originals]
                self.no_inplace_changes = self.no_inplace_changes and unchanged
                self._restore()
                raise

        try:
            for layer in self.layers:
                self._hooks.append(self.modules[layer].o_proj.register_forward_hook(self._projection(layer)))
            # One dispatch for all selected modules; wrappers are never nested.
            self.interface['sdpa'] = observed
            self._installed = True
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        if self._closed:
            return
        errors = []
        for hook in reversed(self._hooks):
            try:
                hook.remove()
            except BaseException as error:
                errors.append(error)
        self._hooks.clear()
        self.hooks_removed = not errors
        try:
            if self._installed:
                if self.had_local:
                    self.interface['sdpa'] = self.previous
                else:
                    del self.interface['sdpa']
            self.restored = self.interface['sdpa'] is self.previous
        except BaseException as error:
            errors.append(error)
        self._closed = True
        require(not errors and self.restored and self.hooks_removed, 'Structure observer restoration failed')

    def __exit__(self, *_):
        self._restore()


MultiLayerAttentionObserver = StructureObserver
