import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from inference_kit.core.io import read, digest, sha
from inference_kit.tasks import tech_record as task
from inference_kit.tasks import contract_policy as policy
from inference_kit.suites.bench05b2_plan import validate_data, validate_prepared, plan, check_frozen
from inference_kit.suites.bench05b2_replay import aggregate, compatible, contrasts, replay
from .helpers import local_fixture, synthetic_run, prepared_fixture
from bench05b1.helpers import generation
import test_release_cpu as release_tests


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.data = policy.make_dataset(); self.sample = self.data['rows'][0]

    def test_p00_exact_b1_instruction(self):
        self.assertEqual(policy.render(self.sample, 'P00'), task.render_prompt(self.sample))

    def test_e_only_and_s_only_exact(self):
        for lang in ('en', 'ko'):
            s = next(r for r in self.data['rows'] if r['language'] == lang)
            base = task.render_prompt(s)
            self.assertEqual(policy.render(s, 'P10'), base + '\n\n' + policy.EVIDENCE[lang])
            self.assertEqual(policy.render(s, 'P01'), base + '\n\n' + policy.STATUS[lang])

    def test_p11_same_blocks_e_then_s(self):
        for sample in self.data['rows']:
            lang = sample['language']
            self.assertEqual(policy.render(sample, 'P11'), task.render_prompt(sample) + '\n\n' + policy.EVIDENCE[lang] + '\n' + policy.STATUS[lang])

    def test_original_contract_not_missing_instruction(self):
        original = task.render_prompt(self.sample)
        for s in ('line-ID strings or null', 'complete/repeat/pending', 'Missing facts and their evidence must be null'):
            self.assertIn(s, original)

    def test_data_new_documents_and_values(self):
        old = task.make_dataset()['rows']; rows = validate_data(self.data)
        self.assertFalse({s['document_hash'] for s in old} & {s['document_hash'] for s in rows})
        self.assertFalse({s['source_id'] for s in old} & {s['source_id'] for s in rows})
        self.assertTrue(all('|' in r['document'] for r in rows))

    def test_gold_controls_and_independent_line_parser(self):
        for sample in self.data['rows']:
            self.assertTrue(policy.verify_gold(sample))
            gold = task.canonical_reference(sample)['text']
            g = {'sample_id': sample['sample_id'], 'sample_identity': sample['sample_identity'], 'task_id': task.TASK,
                 'task_version': task.VERSION, 'output_text': gold, 'transport_complete': True, 'request_latency_http': 1.0}
            self.assertTrue(task.score(sample, g)['task_success'])

    def test_gold_corruption_rejected(self):
        bad = copy.deepcopy(self.sample); bad['gold']['temperature_c'] += 1
        with self.assertRaises(ValueError):
            policy.verify_gold(bad)

    def test_evidence_rows_have_seeded_variety(self):
        self.assertGreater(len({x for r in self.data['rows'] for x in r['gold']['support']['temperature_c']}), 2)

    def test_prompt_no_gold_or_flags(self):
        for sample in self.data['rows']:
            for name in policy.POLICIES:
                text = policy.render(sample, name)
                self.assertNotIn(task.canonical_reference(sample)['text'], text)
                for forbidden in ('sample_identity', '"support"', 'source_family', '"gold"'):
                    self.assertNotIn(forbidden, text)

    def test_balanced_positions_identical_arm_schedule(self):
        order = policy.request_order(self.data['rows']); self.assertEqual(len(order), 112)
        dev = [r for r in order if r['split'] == 'EVAL_LOCKED']
        self.assertEqual(len(dev), 96)
        for i in range(4):
            self.assertEqual({p: sum(r['policy_id'] == p for r in dev[i::4]) for p in policy.POLICIES}, dict.fromkeys(policy.POLICIES, 6))

    def test_probe_choice_fixed_and_reference_not_generation(self):
        ids = policy.select_probes(self.data['rows']); self.assertEqual(len(ids), 4)
        for sid in ids:
            sample = next(r for r in self.data['rows'] if r['sample_id'] == sid)
            a = policy.probe_sample(sample, 'P00'); b = policy.probe_sample(sample, 'P11')
            self.assertEqual(a['reference_text'], b['reference_text'])
            self.assertEqual(a['reference_source'], 'task_author_gold_json')
            self.assertNotEqual(a['prompt'], b['prompt'])


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.sample = policy.make_dataset()['rows'][0]
        self.obj = json.loads(task.canonical_reference(self.sample)['text'])

    def evaluate(self, value):
        g = {'task_id': task.TASK, 'task_version': task.VERSION, 'sample_id': self.sample['sample_id'],
             'sample_identity': self.sample['sample_identity'], 'output_text': json.dumps(value), 'transport_complete': True, 'request_latency_http': .5}
        before = task.score(self.sample, g); after = policy.diagnostics(self.sample, g)
        self.assertEqual({k: after[k] for k in before}, before)
        return after

    def test_canonical_strict_pass(self):
        r = self.evaluate(self.obj); self.assertTrue(r['task_success']); self.assertTrue(all(r['diagnostics'].values()))

    def test_values_in_evidence_not_row_ids(self):
        self.obj['evidence']['temperature_c'] = self.obj['temperature_c']
        r = self.evaluate(self.obj)
        self.assertTrue(r['diagnostics']['core_values_correct']); self.assertFalse(r['diagnostics']['evidence_value_type_valid']); self.assertFalse(r['task_success'])

    def test_evidence_omission_separate(self):
        self.obj.pop('evidence'); r = self.evaluate(self.obj)
        self.assertFalse(r['diagnostics']['evidence_required_keys_present']); self.assertTrue(r['diagnostics']['status_enum_valid'])

    def test_existing_wrong_row_not_support(self):
        self.obj['evidence']['temperature_c'] = next(e['line_id'] for e in self.sample['events'] if e['field'] == 'note')
        r = self.evaluate(self.obj)
        self.assertTrue(r['diagnostics']['evidence_line_id_exists']); self.assertFalse(r['evidence_correct']['temperature_c'])

    def test_nonexistent_row_id(self):
        self.obj['evidence']['status'] = 'L99'; r = self.evaluate(self.obj)
        self.assertTrue(r['diagnostics']['evidence_value_type_valid']); self.assertFalse(r['diagnostics']['evidence_line_id_exists'])

    def test_enum_valid_not_content_correct(self):
        self.obj['status'] = 'complete' if self.obj['status'] != 'complete' else 'pending'; r = self.evaluate(self.obj)
        self.assertTrue(r['diagnostics']['status_enum_valid']); self.assertFalse(r['field_correct']['status'])

    def test_source_synonym_is_not_enum(self):
        self.obj['status'] = 'waiting'; r = self.evaluate(self.obj)
        self.assertFalse(r['diagnostics']['status_enum_valid']); self.assertFalse(r['schema_valid'])

    def test_missing_not_run_preserves_denominator(self):
        scores, summary = aggregate([self.sample], [])
        self.assertIsNone(scores[0]['task_success']); self.assertEqual(summary['not_run'], 1)
        self.assertEqual(summary['metrics']['strict_task_success']['planned_denominator'], 1)


class PlanReplayTests(unittest.TestCase):
    def fixture(self, root):
        return local_fixture(root)

    def test_plan_cpu_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args = self.fixture(root)
            with patch('inference_kit.suites.bench05a_run.gpu_query', side_effect=AssertionError('GPU forbidden')):
                p = plan(*args, root / 'plan')
            self.assertEqual(p['counts']['natural_requests_total'], 224)
            self.assertEqual(p['counts']['maximum_natural_generated_tokens'], 43008)
            self.assertEqual(p['gpu_actions'], 0)
            self.assertEqual(p['status'], 'CPU_PLANNED_NOT_EXECUTED')

    def test_resource_block_before_gpu_or_budget(self):
        from inference_kit.suites.bench05b2_run import run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args = self.fixture(root); suite = read(args[0]); suite['budget']['minimum_cpu_available_bytes'] = 10**30
            args[0].write_text(json.dumps(suite), encoding='utf-8'); before = sha(root / 'prior.json')
            p = plan(*args, root / 'plan'); self.assertTrue(p['status'].startswith('BLOCKED'))
            with patch('inference_kit.suites.bench05a_run.gpu_query', side_effect=AssertionError('GPU forbidden')):
                with self.assertRaises(ValueError):
                    run(root / 'plan/PLAN_FROZEN.json', root / 'run')
            self.assertFalse((root / 'run').exists()); self.assertEqual(sha(root / 'prior.json'), before)

    def test_frozen_source_changed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = plan(*self.fixture(root), root / 'plan')
            with patch('inference_kit.suites.bench05b2_plan.source_identity', return_value='different'):
                with self.assertRaises(ValueError):
                    check_frozen(p)

    def test_frozen_input_changed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = plan(*self.fixture(root), root / 'plan'); (root / 'data.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                check_frozen(p)

    def test_raw_replay_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = synthetic_run(root / 'raw'); p = replay(raw, root / 'derived')
            self.assertTrue(p['full_eval_coverage']); self.assertEqual(p['arms']['rtn']['policies']['P11']['splits']['EVAL_LOCKED']['metrics']['strict_task_success']['correct'], 24)
            self.assertEqual(p['replay_gpu_actions'], 0)
            with self.assertRaises(ValueError):
                replay(raw, root / 'derived')

    def test_all_zero_not_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = synthetic_run(root / 'raw', invalid=True); replay(raw, root / 'derived')
            c = read(root / 'derived/paired-contrasts.json')
            self.assertEqual(c['within_weight']['rtn']['combined']['metrics']['strict_task_success']['delta_count'], 0)
            self.assertIn('no_equivalence', c['scope'])

    def test_partial_not_full_contrast(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = synthetic_run(root / 'raw', partial=True); p = replay(raw, root / 'derived')
            self.assertFalse(p['full_eval_coverage'])
            self.assertEqual(read(root / 'derived/paired-contrasts.json')['within_policy']['P11']['status'], 'PARTIAL_OR_NOT_RUN')

    def test_missing_blob_or_raw_hash_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); raw = synthetic_run(root / 'raw')
            (raw / 'progress.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                replay(raw, root / 'derived')

    def test_same_prefix_required_for_direct_observation_comparison(self):
        from inference_kit.suites.bench05a_replay import matching
        a = {'record': {'kind': 'attention_row', 'sample_id': 'same', 'evidence_kind': 'synthetic_contract', 'coordinates': {'prefix_hash': 'P00'}, 'details': {}}}
        b = copy.deepcopy(a); b['record']['coordinates']['prefix_hash'] = 'P11'
        with self.assertRaises(ValueError):
            matching(a, b)

    def test_undeclared_prompt_mismatch_rejected(self):
        identity = dict(sample_id='a', sample_identity='a', document_hash='d', gold_identity='g', task_id=task.TASK,
            task_version=task.VERSION, scorer_identity='s', runtime_identity='r', tokenizer_identity='t',model_id='same',model_revision='r',generation_policy_identity='g',
            policy_id='P00', policy_identity=policy.policy_record('P00')['policy_identity'], prompt_hash='p', arm_id='bf16', artifact_identity='a')
        other = identity | {'policy_id': 'P11', 'policy_identity': policy.policy_record('P11')['policy_identity'], 'prompt_hash': 'q'}
        with self.assertRaises(ValueError):
            compatible(identity, other, 'same_prompt_weight_intervention')
        self.assertTrue(compatible(identity, other, 'declared_prompt_intervention'))
        other['scorer_identity'] = 'changed'
        with self.assertRaises(ValueError):
            compatible(identity, other, 'declared_prompt_intervention')

    def test_ascii_locale_korean_replay_files_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); synthetic_run(root / 'raw')
            result = release_tests.EncodingTests().child('from inference_kit.cli import main;raise SystemExit(main(["bench","b2","replay","--run","raw","--output","derived"]))', root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['task'], 'BENCH05-B2-CONTRACT')
            self.assertIn('재계산', (root / 'derived/REPORT_KO.md').read_text(encoding='utf-8'))
            self.assertIn('ko-', (root / 'derived/sample-linked.jsonl').read_text(encoding='utf-8'))
            self.assertTrue((root / 'derived/cells.csv').read_text(encoding='utf-8').splitlines())

    def test_cpu_import_no_ml_sdks(self):
        package = str(Path(__import__('inference_kit').__file__).absolute().parent.parent)
        code = 'import sys;sys.path.insert(0,' + repr(package) + ');import inference_kit;from inference_kit.suites import bench05b2_plan,bench05b2_replay;assert not any(n.split(".")[0] in ("torch","vllm","transformers","numpy","aiohttp") for n in sys.modules)'
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)


class ContrastTests(unittest.TestCase):
    def cells(self):
        rows = [s for s in policy.make_dataset()['rows'] if s['split'] == 'EVAL_LOCKED']
        result = {}
        for arm in ('bf16', 'rtn'):
            for name, correct in (('P00', 0), ('P10', 6), ('P01', 8), ('P11', 12)):
                cell = []
                for i, source in enumerate(rows):
                    value = i < correct
                    c = dict(sample_id=source['sample_id'], sample_identity=source['sample_identity'], document_hash=source['document_hash'],
                        gold_identity=digest(source['gold']), task_id=task.TASK, task_version=task.VERSION,
                        scorer_identity='same-scorer', runtime_identity='same-runtime', tokenizer_identity='same-tokenizer',model_id='same',model_revision='r',generation_policy_identity='g',
                        policy_id=name, policy_identity=policy.policy_record(name)['policy_identity'], prompt_hash=name+'-prompt',
                        arm_id=arm, artifact_identity=arm)
                    cell.append({'sample_id': source['sample_id'], 'status': 'SCORED', 'comparison_identity': c,
                      'transport_complete': True, 'json_valid': True, 'schema_valid': value, 'task_success': value,
                      'task_success_within_10s': value, 'field_correct': dict.fromkeys(task.FIELDS, value),
                      'evidence_correct': dict.fromkeys(('temperature_c', 'status'), value),
                      'diagnostics': dict.fromkeys(('status_enum_valid', 'evidence_required_keys_present', 'evidence_value_type_valid',
                         'evidence_line_id_exists', 'core_values_correct'), value)})
                result[arm, name] = cell
        return result

    def test_contrast_sign_and_interaction(self):
        result = contrasts(self.cells())['within_weight']['bf16']
        for name, want in (('E_S0', 6), ('E_S1', 4), ('S_E0', 8), ('S_E1', 6), ('combined', 12), ('interaction', -2)):
            self.assertEqual(result[name]['metrics']['strict_task_success']['delta_count'], want)

    def test_weight_comparison_holds_policy(self):
        result = contrasts(self.cells())['within_policy']
        for value in result.values():
            self.assertEqual(value['metrics']['strict_task_success']['delta_count'], 0)

    def test_wrong_runtime_and_document_rejected(self):
        cells = self.cells()
        for key in ('runtime_identity', 'document_hash', 'scorer_identity', 'tokenizer_identity'):
            broken = copy.deepcopy(cells)
            broken['rtn', 'P00'][0]['comparison_identity'][key] = 'different'
            with self.assertRaises(ValueError):
                contrasts(broken)

    def test_same_policy_prompt_mismatch_rejected(self):
        cells = self.cells(); cells['rtn', 'P00'][0]['comparison_identity']['prompt_hash'] = 'different'
        with self.assertRaises(ValueError):
            contrasts(cells)

    def test_policy_intervention_does_not_change_artifact(self):
        cells = self.cells(); cells['bf16', 'P10'][0]['comparison_identity']['artifact_identity'] = 'changed'
        with self.assertRaises(ValueError):
            contrasts(cells)

    def test_duplicate_or_missing_pair_rejected(self):
        cells = self.cells(); cells['rtn', 'P11'] = cells['rtn', 'P11'][:-1]
        with self.assertRaises(ValueError):
            contrasts(cells)
