#!/usr/bin/env python3
"""Explicit CPU example over a closed *actual* D1 SLICE; never a full run replay.

This is kit orchestration, not a new product source adapter. Existing product
Observation/blob validation and attention metrics perform all numeric work.
"""
import argparse
import collections
import copy
import json
import math
from pathlib import Path

from inference_kit.core.io import digest, read, require, safe_name, sha, write
from inference_kit.metrics.attention import row_metrics, vector_metrics
from inference_kit.probes.blobs import load
from inference_kit.probes.contracts import Observation


def _closed(root):
    manifest = read(root / 'SLICE_MANIFEST.json')
    require(manifest['schema_version'] == 1 and manifest['kind'] == 'closed_actual_structure_metric_slice_v1', 'Wrong slice schema')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'SLICE_MANIFEST.json'}, 'Missing or unmanifested slice member')
    for name, pin in manifest['files'].items():
        p = root / safe_name(name)
        require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) and p.is_file(), 'Unsafe slice member')
        require(p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'Slice member bytes/hash differ: ' + name)
    lineage = read(root / 'SLICE_LINEAGE.json')
    require(lineage['scope'] == 'SLICE_NOT_FULL_D1_RUN' and lineage['selected_observation_ids'], 'Empty or wrong slice scope')
    require(sha(root / 'PARENT_RAW_MANIFEST.json') == lineage['parent_manifest_sha256'], 'Parent manifest hash differs')
    parent = read(root / 'PARENT_RAW_MANIFEST.json')
    for name in lineage['included_original_members']:
        pin = parent['files'][safe_name(name)]; p = root / name
        require(p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'Original member differs from parent pin')
    return manifest, lineage


def _compatible(a, b):
    require(a['kind'] == b['kind'], 'Different observation kind')
    require(a['sample_id'] == b['sample_id'], 'Different sample')
    require(a['coordinates'] == b['coordinates'], 'Different layer/head/query/prefix/mask/tokenizer coordinates')
    require(a['status'] == b['status'] == 'MEASURED', 'Missing actual observation')
    require(a['numeric']['shape'] == b['numeric']['shape'], 'Different numeric shape')


def replay(fixture, output):
    root = Path(fixture).resolve(); output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(root), 'New output outside immutable fixture required')
    manifest, lineage = _closed(root)
    selected = set(lineage['selected_observation_ids']); observations = {}; directories = {}
    for arm in ('bf16', 'rtn'):
        directory = root / ('attempts/main-' + arm + '-attempt1')
        execution = read(directory / 'execution.json')
        require(execution['arm_id'] == arm and execution['status'] == 'MEASURED', 'Execution identity/status differs')
        require(execution['evidence_kind'] == 'live_model_diagnostic', 'Actual/synthetic identity differs')
        directories[arm] = directory
        for line in (directory / 'observations.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            obs = json.loads(line)
            if obs['observation_id'] not in selected: continue
            Observation(**obs).validate()
            require(obs['execution_id'] == execution['execution_id'] and obs['label'] == execution['execution_mode'], 'Observation/execution identity differs')
            require(obs['coordinates']['pass_id'] == 'ON', 'Control included as main observation')
            require(obs['observation_id'] not in observations, 'Duplicate observation ID')
            observations[obs['observation_id']] = (arm, obs)
    require(set(observations) == selected, 'Selected observation coverage differs')
    rows = []; reconstruction = []; selfchecks = 0
    def actual(ref):
        arm, obs = observations[ref['observation_id']]
        require(obs['numeric'] == ref['numeric'] and obs['execution_id'] == ref['execution_id'], 'Expected reference identity differs')
        return obs, load(directories[arm], obs['numeric'])
    def check_metric(got, expected):
        for key, value in got.items():
            want = expected[key]
            if type(value) is float:
                require(type(want) in (float, int) and math.isclose(value, want, abs_tol=1e-10, rel_tol=1e-9), 'Historical metric differs: ' + key)
            else: require(type(value) is type(want) and value == want, 'Historical metric status/count differs: ' + key)
    for line in (root / 'EXPECTED_COMPARISONS.jsonl').read_text(encoding='utf-8').splitlines():
        expected = json.loads(line); a, av = actual(expected['reference']); b, bv = actual(expected['candidate_observation'])
        _compatible(a, b); c = a['coordinates']
        if a['kind'] == 'attention_row':
            metrics = row_metrics(av, bv, c['allowed_keys'])
            own = row_metrics(av, av, c['allowed_keys']); require(own['l1'] == own['js_nat'] == 0., 'Attention self comparison differs')
        else:
            metrics = vector_metrics(av, bv); own = vector_metrics(av, av)
            require(own['nmse'] == 0. and math.isclose(own['cosine'], 1., abs_tol=1e-12), 'Vector self comparison differs')
        if a['kind'] == 'output_projection': require(c.get('head') is None and c.get('kv_head') is None, 'Projection carries head axis')
        check_metric(metrics, expected['metrics']); selfchecks += 1
        rows.append({'identity': expected['identity'], 'kind': a['kind'], 'status': 'MEASURED', 'metrics': metrics,
                     'reference_observation_id': a['observation_id'], 'candidate_observation_id': b['observation_id']})
    for line in (root / 'EXPECTED_RECONSTRUCTION.jsonl').read_text(encoding='utf-8').splitlines():
        expected = json.loads(line); a, av = actual(expected['native_reference']); b, bv = actual(expected['reconstructed_candidate'])
        require(a['kind'] == 'attention_output' and b['kind'] == 'reconstructed_AV' and a['coordinates'] == b['coordinates'], 'Reconstruction/native coordinates differ')
        metrics = vector_metrics(av, bv); check_metric(metrics, expected['metrics'])
        reconstruction.append({'identity':expected['identity'], 'arm_id':expected['arm_id'], 'metrics':metrics, 'status':'MEASURED'})
    require(rows and reconstruction, 'No actual comparison coverage')
    # A malformed coordinate must be rejected before calling numeric metrics.
    first = next(v[1] for v in observations.values() if v[1]['kind'] == 'attention_row')
    altered = copy.deepcopy(first); altered['coordinates']['head'] += 1
    rejected = False
    try: _compatible(first, altered)
    except ValueError: rejected = True
    require(rejected, 'Coordinate mismatch was accepted')
    output.mkdir(parents=True)
    write(output / 'STRUCTURE_COMPARISONS.json', rows)
    write(output / 'SAME_ARM_AV.json', reconstruction)
    result = {'status':'ACTUAL_SELECTED_STRUCTURE_SLICE_REPLAYED', 'scope':'SLICE_NOT_FULL_D1_RUN',
              'source_parent_manifest_sha256':lineage['parent_manifest_sha256'], 'slice_manifest_sha256':sha(root/'SLICE_MANIFEST.json'),
              'samples':len(lineage['selected_coordinates']), 'physical_observations':len(observations),
              'cross_arm_rows':len(rows), 'kind_counts':dict(collections.Counter(r['kind'] for r in rows)),
              'same_arm_reconstruction_rows':len(reconstruction), 'self_comparisons':selfchecks,
              'coordinate_mismatch_rejected':rejected, 'historical_metric_parity':'abs1e-10/rel1e-9 for every field returned by product metric functions',
              'stored_extra_derived_fields_not_recomputed':['entropy_delta_nat'],
              'full_structure_controls':'NOT_REPLAYED_FROM_SLICE', 'new_acquisitions':0, 'model_loads':0, 'GPU_actions':0,
              'tokenizer_loads':0, 'implementation':'Existing product Observation/blob validators and row_metrics/vector_metrics; no copied metric formulas'}
    write(output / 'REPLAY_RECEIPT.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    print(json.dumps(replay(args.fixture, args.output), ensure_ascii=True, allow_nan=False))

if __name__ == '__main__':main()
