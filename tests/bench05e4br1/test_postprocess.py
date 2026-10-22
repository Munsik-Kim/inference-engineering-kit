"""Closed CPU revision tests. All acquired inputs here are synthetic evidence.

Real compiler, raw hash validation, scorers, C1, SQLite and report paths run.
Fault injections affect only explicit CPU operation/write failures.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit import api
from inference_kit.analysis import suite_postprocess as cpu, suite_results
from inference_kit.core.io import read, write, sha, digest
from inference_kit.core.state import atomic
from inference_kit.suites.suite_execution import _lock
from bench05e3.test_results import fixture, view_fixture


class CPUPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        parent = fixture(self.root/'parent')
        view = view_fixture(self.root/'view', read(parent/'suite-execution-context.json'))
        self.raw = self.root/'raw'
        suite_results.export_mini_slice(parent, view, self.raw)
        self.analysis = self.root/'한글-analysis'

    def tearDown(self):
        self.temp.cleanup()

    def run_cpu(self, **kwargs):
        return api.suite_postprocess(self.raw, self.analysis, **kwargs)

    def test_real_cpu_stages_pause_resume_and_completed_noop_preserve_outputs(self):
        first = self.run_cpu(stop_after='score_task')
        self.assertEqual(first['status'], 'CPU_PAUSED_BOUNDARY')
        sealed = sha(self.analysis/'completions/score_task.json')
        last = api.suite_postprocess_resume(self.analysis)
        self.assertEqual(last['status'], 'CPU_POSTPROCESS_COMPLETE')
        checkpoint = (self.analysis/'checkpoint.json').read_bytes()
        before = cpu._inventory(self.analysis, self.analysis/'stages')
        with patch.object(cpu, 'execute_cpu', side_effect=AssertionError('No-op cannot execute CPU stage')):
            noop = api.suite_postprocess_resume(self.analysis)
        self.assertTrue(noop['noop']); self.assertFalse(noop['checkpoint_reconciled'])
        self.assertEqual(noop['controller_sessions'], 2)
        self.assertEqual((self.analysis/'checkpoint.json').read_bytes(), checkpoint)
        self.assertEqual(cpu._inventory(self.analysis, self.analysis/'stages'), before)
        self.assertEqual(sha(self.analysis/'completions/score_task.json'), sealed)
        self.assertEqual(len(api.suite_postprocess_status(self.analysis)['verified_completed']), 4)

    def test_terminal_commit_gap_is_saved_once_without_dispatch(self):
        self.run_cpu(); state = read(self.analysis/'checkpoint.json')
        terminal = cpu.OPERATIONS[-1]; state['status'] = 'CPU_RUNNING'
        state['nodes'][terminal] = {'status': 'CPU_RUNNING', 'attempts': state['nodes'][terminal]['attempts']}
        atomic(self.analysis/'checkpoint.json', state)
        with patch.object(cpu, 'execute_cpu', side_effect=AssertionError('No replay')):
            first = api.suite_postprocess_resume(self.analysis)
            checkpoint = sha(self.analysis/'checkpoint.json')
            second = api.suite_postprocess_resume(self.analysis)
        self.assertTrue(first['checkpoint_reconciled']); self.assertFalse(second['checkpoint_reconciled'])
        self.assertEqual(sha(self.analysis/'checkpoint.json'), checkpoint)
        self.assertEqual(read(self.analysis/'checkpoint.json')['status'], 'CPU_POSTPROCESS_COMPLETE')
        self.assertIn('output', read(self.analysis/'checkpoint.json')['nodes'][terminal])

    def test_committed_stage_gap_continues_remaining_cpu_work(self):
        self.run_cpu(stop_after='score_task'); original = sha(self.analysis/'completions/score_task.json')
        state = read(self.analysis/'checkpoint.json'); state['nodes']['score_task'] = {'status': 'CPU_RUNNING', 'attempts': 1}
        atomic(self.analysis/'checkpoint.json', state)
        result = api.suite_postprocess_resume(self.analysis)
        self.assertEqual(result['status'], 'CPU_POSTPROCESS_COMPLETE')
        self.assertEqual(sha(self.analysis/'completions/score_task.json'), original)
        self.assertTrue(all(v['attempts'] == 1 for v in result['nodes'].values()))

    def test_cpu_failure_retains_evidence_and_resumes_new_cpu_attempt(self):
        original = cpu.execute_cpu
        def fail_report(kind, *args, **kwargs):
            if kind == 'report':
                raise RuntimeError('SYNTHETIC_CPU_REPORT_FAILURE')
            return original(kind, *args, **kwargs)
        with patch.object(cpu, 'execute_cpu', side_effect=fail_report), self.assertRaisesRegex(RuntimeError, 'SYNTHETIC_CPU'):
            self.run_cpu()
        self.assertFalse((self.analysis/'completions/report.json').exists())
        self.assertEqual(read(self.analysis/'checkpoint.json')['status'], 'CPU_POSTPROCESS_FAILED')
        marker = sha(self.analysis/'completions/index_add.json')
        result = api.suite_postprocess_resume(self.analysis)
        self.assertEqual(result['status'], 'CPU_POSTPROCESS_COMPLETE')
        self.assertEqual(result['nodes']['report']['attempts'], 2)
        self.assertEqual(sha(self.analysis/'completions/index_add.json'), marker)

    def test_empty_operation_result_cannot_be_complete(self):
        self.run_cpu(stop_after='score_task')
        with patch.object(cpu, 'execute_cpu', return_value={'status': 'COMPLETE'}), self.assertRaises(ValueError):
            api.suite_postprocess_resume(self.analysis)
        self.assertFalse((self.analysis/'completions/analyze_tokens.json').exists())
        self.assertEqual(read(self.analysis/'checkpoint.json')['status'], 'CPU_POSTPROCESS_FAILED')

    def test_changed_source_plan_and_raw_are_rejected_before_execution(self):
        self.run_cpu(stop_after='score_task'); plan = read(self.analysis/'ANALYSIS_PLAN.json')
        bad = deepcopy(plan); bad['semantic']['analyzer_source_digest'] = 'different'
        bad['analysis_plan_identity'] = digest(bad['semantic']); bad.pop('record_identity'); bad['record_identity'] = digest(bad)
        atomic(self.analysis/'ANALYSIS_PLAN.json', bad)
        with patch.object(cpu, 'execute_cpu') as dispatch, self.assertRaisesRegex(ValueError, 'analyzer source changed'):
            api.suite_postprocess_resume(self.analysis)
        dispatch.assert_not_called(); atomic(self.analysis/'ANALYSIS_PLAN.json', plan)
        raw_file = next(self.raw.glob('acquisitions/**/*.jsonl')); raw_file.write_text('{}\n', encoding='utf-8')
        with patch.object(cpu, 'execute_cpu') as dispatch, self.assertRaisesRegex(ValueError, 'raw member content changed'):
            api.suite_postprocess_resume(self.analysis)
        dispatch.assert_not_called()

    def test_output_or_completion_tamper_cannot_return_completed(self):
        self.run_cpu(); output = self.analysis/'stages/index_add/attempt1/INPUT_SOURCES.json'
        output.write_text('{}', encoding='utf-8')
        with patch.object(cpu, 'execute_cpu') as dispatch, self.assertRaisesRegex(ValueError, 'STALE_EVIDENCE'):
            api.suite_postprocess_resume(self.analysis)
        dispatch.assert_not_called()
        with self.assertRaises(ValueError):
            api.suite_postprocess_status(self.analysis)

    def test_no_gpu_operation_empty_coverage_or_invalid_revision_plan(self):
        for revision in ('', ' x ', False, 'x'*129):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, 'revision'):
                self.run_cpu(analysis_revision=revision)
        self.run_cpu(stop_after='score_task'); record = read(self.analysis/'ANALYSIS_PLAN.json')
        for operations in ([], ['serve_group']):
            modified = deepcopy(record); modified['semantic']['operations'] = operations
            modified['analysis_plan_identity'] = digest(modified['semantic']); modified.pop('record_identity')
            modified['record_identity'] = digest(modified); atomic(self.analysis/'ANALYSIS_PLAN.json', modified)
            with self.assertRaisesRegex(ValueError, 'fixed CPU operations'):
                api.suite_postprocess_resume(self.analysis)

    def test_lock_and_immutable_output_boundaries(self):
        with self.assertRaisesRegex(ValueError, 'outside immutable'):
            api.suite_postprocess(self.raw, self.raw/'bad-analysis')
        self.run_cpu(stop_after='score_task')
        with _lock(self.analysis), self.assertRaisesRegex(ValueError, 'RUN_LOCKED'):
            api.suite_postprocess_resume(self.analysis)
        self.assertEqual(api.suite_postprocess_resume(self.analysis)['status'], 'CPU_POSTPROCESS_COMPLETE')

    def test_atomic_terminal_write_failure_does_not_return_success(self):
        self.run_cpu(); state = read(self.analysis/'checkpoint.json'); state['status'] = 'CPU_RUNNING'
        atomic(self.analysis/'checkpoint.json', state); original = cpu.atomic; count = []
        def fail_once(path, value):
            if Path(path).name == 'checkpoint.json' and not count:
                count.append(1); raise OSError('SYNTHETIC_REPLACE_FAILURE')
            return original(path, value)
        with patch.object(cpu, 'atomic', side_effect=fail_once), self.assertRaisesRegex(OSError, 'SYNTHETIC_REPLACE'):
            api.suite_postprocess_resume(self.analysis)
        self.assertEqual(read(self.analysis/'checkpoint.json')['status'], 'CPU_RUNNING')
        self.assertTrue(api.suite_postprocess_resume(self.analysis)['checkpoint_reconciled'])

    def test_status_rejects_boolean_or_changed_attempt_and_missing_marker(self):
        self.run_cpu(); baseline = read(self.analysis/'checkpoint.json')
        for attempt in (True, 9):
            state = deepcopy(baseline); state['nodes']['score_task']['attempts'] = attempt
            atomic(self.analysis/'checkpoint.json', state)
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                api.suite_postprocess_status(self.analysis)
        atomic(self.analysis/'checkpoint.json', baseline)
        (self.analysis/'completions/report.json').unlink()
        with self.assertRaises(ValueError):
            api.suite_postprocess_resume(self.analysis)

    def test_explicit_selected_full_slice_has_dynamic_coverage_and_no_parent_index(self):
        receipt = read(self.raw/'SLICE_RECEIPT.json')
        receipt.update(kind='suite_all_scalar_selected_full_slice_v1',
                       parent_run_manifest_sha256=receipt['parent_raw_manifest_sha256'])
        atomic(self.raw/'SLICE_RECEIPT.json', receipt)
        manifest = read(self.raw/'RUN_MANIFEST.json'); manifest['derived_slice_kind'] = receipt['kind']
        manifest['files']['SLICE_RECEIPT.json'] = {'bytes':(self.raw/'SLICE_RECEIPT.json').stat().st_size,
                                                'sha256':sha(self.raw/'SLICE_RECEIPT.json')}
        atomic(self.raw/'RUN_MANIFEST.json', manifest)
        result = api.suite_replay(self.raw, self.root/'slice-replay')
        self.assertEqual(result['diagnostic_scalar_rows'], receipt['coverage']['records'])
        self.assertEqual(result['full_vectors'], receipt['coverage']['full_logits_records'])
        with self.assertRaisesRegex(ValueError, 'SLICE_ONLY'):
            suite_results.load_index_source(suite_results.index_source_spec(self.raw), self.root)


if __name__ == '__main__':
    unittest.main()
