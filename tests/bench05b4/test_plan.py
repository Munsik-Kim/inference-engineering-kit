"""Small synthetic preflight fixtures; no real weights, GPU, or SDK dependency."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import inference_kit
from inference_kit.core.io import digest, read, sha, write
from inference_kit.probes.retrieval_tokens import BOUNDS, POLICY, SPAN_POLICY, prepare_retrieval_row
from inference_kit.tasks import context_evidence as task
from .test_tokens import TinyTokenizer


def prepared_fixture():
    data = task.make_dataset()
    cfg = {'model_type': 'llama', 'num_hidden_layers': 2, 'num_attention_heads': 2,
           'num_key_value_heads': 2, 'hidden_size': 128, 'head_dim': 64,
           'vocab_size': 256, 'attention_dropout': 0., 'torch_dtype': 'bfloat16'}
    rows = [prepare_retrieval_row(r, TinyTokenizer(), cfg, 'tiny-tokenizer', 'tiny-space')
            for r in data['rows']]
    by_id = {r['sample_id']: r for r in rows}
    selected = task.select_probes(data['rows'])
    probes = [by_id[sid] for sid in selected]
    return data, {'schema_version': 1, 'kind': 'prepared_retrieval_samples_v1',
      'task_id': task.TASK, 'task_version': task.VERSION, 'preparation_id': POLICY,
      'task_rows_identity': data['rows_identity'], 'generation_rows': rows,
      'generation_rows_identity': digest(rows), 'rows': probes, 'rows_identity': digest(probes),
      'model_config': cfg, 'tokenizer_files': {}, 'tokenizer_identity': 'tiny-tokenizer',
      'token_id_space': 'tiny-space', 'vocabulary_size': 256, 'bounds': BOUNDS,
      'source_span_policy': SPAN_POLICY, 'probe_selection': {'seed': 909265, 'sample_ids': selected},
      'model_loaded': False, 'gpu_used': False}


def local_fixture(root):
    data, prepared = prepared_fixture()
    checkout = Path(__file__).parents[2]
    suite = read(checkout / 'configs/bench05b4/suite.json')
    suite['budget'].update(minimum_cpu_available_bytes=1, minimum_disk_free_bytes=1)
    pair = read(checkout / 'configs/bench05a/pair.json')
    write(root / 'suite.json', suite)
    write(root / 'pair.json', pair)
    write(root / 'data.json', data)
    prepared['authored_sha256'] = sha(root / 'data.json')
    write(root / 'prepared.json', prepared)
    write(root / 'runtime.json', {'packages': {'transformers': {'version': '5.14.1'},
          'compressed-tensors': {'version': '0.18.0'}}, 'synthetic_contract': True})
    refs = {}
    for key, name in (('BF16_REF', 'bf16'), ('R00', 'rtn')):
        artifact = root / name
        artifact.mkdir()
        cfg = copy.deepcopy(prepared['model_config'])
        if name == 'rtn':
            cfg['quantization_config'] = {'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
              'config_groups': {'g': {'input_activations': None, 'output_activations': None,
                'targets': ['Linear'], 'weights': {'num_bits': 4, 'type': 'int', 'strategy': 'group',
                  'group_size': 128, 'symmetric': True, 'actorder': None}}}}
        write(artifact / 'config.json', cfg)
        (artifact / 'tiny.safetensors').write_bytes(b'synthetic-metadata-only-not-model-weights')
        refs[key] = {'model_id': pair['model_id'], 'model_revision': pair['revision'],
          'path': str(artifact), 'tokenizer_path': str(artifact), 'artifact_identity': 'synthetic-' + name,
          'recipe_identity': 'tiny-' + name, 'config_sha256': sha(artifact / 'config.json'),
          'recipe': {'algorithm': 'RTN'}}
    write(root / 'receipt.json', {'arms': refs})
    write(root / 'prior.json', {'used_seconds': 10., 'running': None, 'jobs': [{'wall_seconds': 10.}]})
    now = time.monotonic()
    write(root / 'clock.json', {'start_utc': 'synthetic-test', 'start_monotonic': now,
      'deadline_monotonic': now + 3600, 'no_new_gpu_after': now + 3000})
    local = {'authored_data': str(root / 'data.json'), 'prepared_samples': str(root / 'prepared.json'),
      'artifact_receipt': str(root / 'receipt.json'), 'runtime_metadata': str(root / 'runtime.json'),
      'runtime_python': sys.executable, 'server_python': sys.executable,
      'package_site': str(Path(inference_kit.__file__).absolute().parent.parent),
      'prior_ledger': str(root / 'prior.json'), 'campaign_root': str(root),
      'task_clock': str(root / 'clock.json'), 'task_origin_used_seconds': 10.,
      'gpu': 0, 'gpu_uuid': 'synthetic-not-a-gpu', 'port': 18188,
      'serving_runtime_identity': 'synthetic-serving-runtime'}
    write(root / 'local.json', local)
    return root / 'suite.json', root / 'pair.json', root / 'local.json'


class RetrievalPlanTests(unittest.TestCase):
    def test_validate_all_data_and_prepared_rows(self):
        from inference_kit.suites.bench05b4_plan import validate_data
        data, prepared = prepared_fixture()
        self.assertEqual(len(validate_data(data, prepared)), 28)

    def test_prepared_task_reference_and_bounds_mismatch_rejected(self):
        from inference_kit.suites.bench05b4_plan import validate_data
        for key, value in (('task_id', 'tech_record_extract'), ('reference_source', 'task_author_gold_json'),
                           ('generation_input_ids', [1] * 513), ('reference_separate_ids', [1] * 65),
                           ('input_ids', [1] * 577)):
            data, prepared = prepared_fixture()
            prepared['generation_rows'][0][key] = value
            prepared['generation_rows_identity'] = digest(prepared['generation_rows'])
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_data(data, prepared)

    def test_tampered_gold_or_evidence_coordinates_rejected(self):
        from inference_kit.suites.bench05b4_plan import validate_data
        data, prepared = prepared_fixture()
        prepared['generation_rows'][0]['reference_evidence_span']['status'] = 'MEASURED'
        prepared['generation_rows_identity'] = digest(prepared['generation_rows'])
        with self.assertRaises(ValueError):
            validate_data(data, prepared)
        data, prepared = prepared_fixture()
        data['rows'][0]['gold']['answer'] = 'FAKE-99'
        data['rows_identity'] = digest(data['rows'])
        with self.assertRaises(ValueError):
            validate_data(data, prepared)

    def test_plan_uses_no_subprocess_gpu_or_network_and_counts_requests(self):
        from inference_kit.suites.bench05b4_plan import plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = local_fixture(root)
            with patch('subprocess.run', side_effect=AssertionError('CPU plan started subprocess')):
                result = plan(*args, root / 'plan')
            self.assertEqual((result['gpu_actions'], result['model_loads'], result['network_actions']), (0, 0, 0))
            self.assertEqual(result['counts']['total_posts'], 56)
            self.assertEqual(result['counts']['max_output_tokens'], 3584)
            self.assertEqual(result['campaign']['used_seconds'], 10.)
            self.assertEqual(read(root / 'prior.json')['used_seconds'], 10.)

    def test_resource_limits_block_before_gpu_or_prior_ledger_mutation(self):
        from inference_kit.suites.bench05b4_plan import plan, check_frozen
        from inference_kit.suites.bench05b4_run import run
        for resource in ('minimum_cpu_available_bytes', 'minimum_disk_free_bytes'):
            with self.subTest(resource=resource), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = local_fixture(root)
                suite = read(args[0])
                suite['budget'][resource] = sys.maxsize
                args[0].write_text(json.dumps(suite), encoding='utf-8')
                frozen = plan(*args, root / 'plan')
                original = sha(root / 'prior.json')
                self.assertEqual(frozen['status'], 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT')
                with patch('subprocess.run', side_effect=AssertionError('GPU/resource launch attempted')):
                    with self.assertRaises(ValueError):
                        check_frozen(frozen)
                with patch('inference_kit.suites.bench05b1_run.check_gpu', side_effect=AssertionError('GPU query before resource gate')), \
                     patch('inference_kit.adapters.owned.start', side_effect=AssertionError('Owned launch before resource gate')):
                    with self.assertRaises(ValueError):
                        run(root / 'plan/PLAN_FROZEN.json', root / 'run')
                self.assertEqual(sha(root / 'prior.json'), original)
                self.assertFalse((root / 'budget-continuation.json').exists())
                self.assertFalse((root / 'run').exists())

    def test_source_and_authored_data_change_reject_frozen_execution(self):
        from inference_kit.suites.bench05b4_plan import plan, check_frozen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = plan(*local_fixture(root), root / 'plan')
            with patch('inference_kit.suites.bench05b4_plan.source_identity', return_value={}):
                with self.assertRaises(ValueError):
                    check_frozen(frozen)
            data = read(root / 'data.json')
            data['scorer'] = 'changed-after-freeze'
            (root / 'data.json').write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                check_frozen(frozen)
            self.assertEqual(read(root / 'prior.json')['used_seconds'], 10.)

    def test_actual_resources_are_not_relaxed_for_synthetic_tests(self):
        suite = read(Path(__file__).parents[2] / 'configs/bench05b4/suite.json')
        self.assertEqual(suite['budget']['minimum_cpu_available_bytes'], 10 * 2 ** 30)
        self.assertEqual(suite['budget']['minimum_disk_free_bytes'], 4 * 2 ** 30)
        self.assertEqual(suite['budget']['cleanup_reserve_seconds'], 120)


if __name__ == '__main__':
    unittest.main()
