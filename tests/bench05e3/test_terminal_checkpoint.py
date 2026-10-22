"""Terminal commit/checkpoint gaps, using real package identity and CPU fake work.

Synthetic dispatch is permitted only to prepare evidence or finish the one
explicit middle-gap CPU case. Terminal resume is guarded against every dispatch.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import builtins
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_kit import api
from inference_kit.core.io import digest, read, sha, write
from inference_kit.core.state import atomic
from inference_kit.suites import suite_execution as execute
from inference_kit.suites import suite_live
from bench05e3.test_execution import fixture, fake_dispatch


@contextmanager
def no_work_guard():
    """Observable Python-call boundary, not an OS sandbox or child inheritance."""
    counts = {key: 0 for key in ('dispatch', 'GPU_preflight', 'acquisition', 'subprocess', 'socket', 'ML_import')}

    def reject(key):
        def fail(*args, **kwargs):
            counts[key] += 1
            raise AssertionError('Terminal resume attempted ' + key)
        return fail

    original_import = builtins.__import__

    def cpu_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'vllm', 'numpy', 'aiohttp', 'compressed_tensors'}:
            return reject('ML_import')()
        return original_import(name, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(execute, '_dispatch', side_effect=reject('dispatch')))
        stack.enter_context(patch.object(suite_live, 'preflight', side_effect=reject('GPU_preflight')))
        stack.enter_context(patch.object(suite_live, 'execute_acquisition', side_effect=reject('acquisition')))
        stack.enter_context(patch.object(subprocess, 'Popen', side_effect=reject('subprocess')))
        stack.enter_context(patch.object(socket, 'socket', side_effect=reject('socket')))
        stack.enter_context(patch.object(builtins, '__import__', side_effect=cpu_import))
        yield counts


def protected_hashes(root):
    """Exclude only mutable checkpoint/event/lock and derived status outputs."""
    paths = [root / 'EXECUTION_PLAN.json', root / 'suite-execution-context.json',
             root / 'prepared/input.json', root / 'parent-ledger.json', root / 'budget-continuation.json']
    paths += sorted((root / 'nodes').rglob('*'))
    return {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}


def terminal_gap(root, *, global_only=False):
    plan = read(root / 'EXECUTION_PLAN.json')
    terminal = plan['units'][-1]['node_id']
    state = read(root / 'checkpoint.json')
    state['status'] = 'RUNNING'
    if not global_only:
        entry = state['nodes'][terminal]
        entry['status'] = 'RUNNING'
        entry.pop('completion', None)
        entry.pop('output', None)
    atomic(root / 'checkpoint.json', state)
    return terminal, deepcopy(state)


class TerminalCheckpointTests(unittest.TestCase):
    def make(self, name='한글-run'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = fixture(Path(temporary.name) / name)
        initial = read(root / 'checkpoint.json')
        initial.update(started_utc='2026-01-01T00:00:00+00:00', started_monotonic=123.5)
        atomic(root / 'checkpoint.json', initial)
        self.assertEqual(read(root / 'EXECUTION_PLAN.json')['executor_source'], execute.source_identity())
        calls = []
        with patch.object(execute, '_dispatch', side_effect=fake_dispatch(calls)):
            result = execute._drive(root, root / 'local.json', True)
        self.assertEqual(result['status'], 'COMPLETE_VALIDATED')
        self.assertEqual(len(calls), len(read(root / 'EXECUTION_PLAN.json')['units']))
        return root

    def assert_no_work(self, counts):
        self.assertEqual(counts, {key: 0 for key in counts})

    def assert_disk_complete(self, root):
        disk = read(root / 'checkpoint.json')
        self.assertEqual(disk['status'], 'COMPLETE_VALIDATED')
        self.assertTrue(disk['nodes'])
        self.assertEqual({entry['status'] for entry in disk['nodes'].values()}, {'COMPLETE_VALIDATED'})
        self.assertEqual(api.suite_status(root)['status'], 'COMPLETE_VALIDATED')
        for node in read(root / 'EXECUTION_PLAN.json')['units']:
            entry = disk['nodes'][node['node_id']]
            completion = read(root / entry['completion'])
            self.assertEqual(entry['output'], completion['evidence_directory'])
        return disk

    def test_terminal_commit_gap_persists_exact_evidence_and_repeat_is_noop(self):
        root = self.make()
        terminal, before = terminal_gap(root)
        pins = protected_hashes(root)
        with no_work_guard() as calls:
            result = api.suite_resume(root, allow_gpu=False)
            self.assert_disk_complete(root)
        self.assert_no_work(calls)
        self.assertTrue(result['noop'])
        self.assertTrue(result['checkpoint_reconciled'])
        self.assertEqual(result['reconciled_nodes'], [terminal])
        after = read(root / 'checkpoint.json')
        for key in ('controller_sessions', 'active_controller_seconds', 'paused_seconds', 'pause_started_monotonic',
                    'started_utc', 'started_monotonic', 'run_id', 'execution_plan_identity'):
            self.assertEqual(after[key], before[key])
        self.assertEqual({k: v['attempts'] for k, v in after['nodes'].items()},
                         {k: v['attempts'] for k, v in before['nodes'].items()})
        self.assertEqual(protected_hashes(root), pins)
        stable = sha(root / 'checkpoint.json')
        stable_stat = (root / 'checkpoint.json').stat()
        with no_work_guard() as calls:
            second = api.suite_resume(root, allow_gpu=False)
        self.assert_no_work(calls)
        self.assertFalse(second['checkpoint_reconciled'])
        self.assertEqual(second['reconciled_nodes'], [])
        self.assertEqual(sha(root / 'checkpoint.json'), stable)
        self.assertEqual((root / 'checkpoint.json').stat().st_mtime_ns, stable_stat.st_mtime_ns)
        self.assertEqual((root / 'checkpoint.json').stat().st_ino, stable_stat.st_ino)
        self.assertEqual(protected_hashes(root), pins)

    def test_global_only_gap_preserves_node_entries_and_accounting(self):
        root = self.make()
        _, before = terminal_gap(root, global_only=True)
        pins = protected_hashes(root)
        with no_work_guard() as calls:
            result = api.suite_resume(root)
            after = self.assert_disk_complete(root)
        self.assert_no_work(calls)
        self.assertTrue(result['checkpoint_reconciled'])
        self.assertEqual(result['reconciled_nodes'], [])
        self.assertEqual(after['nodes'], before['nodes'])
        self.assertEqual({k: v for k, v in after.items() if k != 'status'},
                         {k: v for k, v in before.items() if k != 'status'})
        self.assertEqual(protected_hashes(root), pins)

    def test_normal_completed_api_and_fresh_cli_noop_do_not_rewrite_checkpoint(self):
        # POSIX non-UTF8 argv decoding is separate from UTF8 JSON/Korean-path checks.
        root = self.make(name='run')
        pins = protected_hashes(root)
        stable = sha(root / 'checkpoint.json')
        stable_stat = (root / 'checkpoint.json').stat()
        with no_work_guard() as calls:
            result = api.suite_resume(root)
        self.assert_no_work(calls)
        self.assertFalse(result['checkpoint_reconciled'])
        self.assertEqual(result['reconciled_nodes'], [])
        env = dict(os.environ)
        env.update(LC_ALL='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0')
        # These are independent CPU CLI processes; the parent guard is not inherited.
        resumed = subprocess.run([sys.executable, '-B', '-m', 'inference_kit.cli', 'bench', 'suite',
                                  'resume', '--run', str(root)], check=True, capture_output=True, env=env)
        self.assertTrue(resumed.stdout.isascii())
        self.assertEqual(json.loads(resumed.stdout)['status'], 'COMPLETE_VALIDATED')
        status_output = root.parent / 'CLI-status'
        checked = subprocess.run([sys.executable, '-B', '-m', 'inference_kit.cli', 'bench', 'suite',
                                  'status', '--run', str(root), '--output', str(status_output)],
                                 check=True, capture_output=True, env=env)
        self.assertTrue(checked.stdout.isascii())
        self.assertEqual(json.loads(checked.stdout)['status'], 'COMPLETE_VALIDATED')
        self.assertEqual(read(status_output / 'STATUS.json')['status'], 'COMPLETE_VALIDATED')
        self.assertEqual(sha(root / 'checkpoint.json'), stable)
        self.assertEqual((root / 'checkpoint.json').stat().st_mtime_ns, stable_stat.st_mtime_ns)
        self.assertEqual((root / 'checkpoint.json').stat().st_ino, stable_stat.st_ino)
        self.assertEqual(protected_hashes(root), pins)

    def test_recovered_terminal_status_is_visible_to_fresh_cli_process(self):
        root = self.make()
        terminal_gap(root)
        with no_work_guard() as calls:
            api.suite_resume(root)
        self.assert_no_work(calls)
        stable = sha(root / 'checkpoint.json')
        completed = subprocess.run([sys.executable, '-B', '-m', 'inference_kit.cli', 'bench', 'suite',
                                    'status', '--run', str(root), '--output', str(root.parent / 'fresh-status')],
                                   check=True, capture_output=True)
        self.assertEqual(json.loads(completed.stdout)['status'], 'COMPLETE_VALIDATED')
        self.assertEqual(sha(root / 'checkpoint.json'), stable)

    def test_terminal_completion_evidence_input_ledger_and_journal_guards(self):
        for alteration in ('missing_completion', 'completion_hash', 'dependency_identity', 'missing_raw',
                           'raw_bytes', 'immutable_input', 'ledger_usage', 'parent_ledger', 'event_chain'):
            with self.subTest(alteration=alteration):
                root = self.make(name=alteration)
                terminal, _ = terminal_gap(root)
                completion = root / 'nodes' / terminal / 'COMPLETION.json'
                evidence = root / read(completion)['evidence_directory'] / 'EVIDENCE.json'
                if alteration == 'missing_completion':
                    # A COMPLETE node still requires its durable receipt; never rerun it.
                    state = read(root / 'checkpoint.json')
                    state['nodes'][terminal]['status'] = 'COMPLETE_VALIDATED'
                    atomic(root / 'checkpoint.json', state)
                    completion.unlink()
                elif alteration == 'completion_hash':
                    changed = read(completion); changed['completed_utc'] = 'changed'; atomic(completion, changed)
                elif alteration == 'dependency_identity':
                    changed = read(completion); changed.pop('completion_identity'); changed['input_identity'] = 'changed'
                    changed['completion_identity'] = digest(changed); atomic(completion, changed)
                elif alteration == 'missing_raw': evidence.unlink()
                elif alteration == 'raw_bytes': evidence.write_text('{}', encoding='utf-8')
                elif alteration == 'immutable_input': (root / 'prepared/input.json').write_text('{}', encoding='utf-8')
                elif alteration == 'ledger_usage':
                    changed = read(root / 'budget-continuation.json'); changed['used_seconds'] += 1
                    atomic(root / 'budget-continuation.json', changed)
                elif alteration == 'parent_ledger': (root / 'parent-ledger.json').write_text('{}', encoding='utf-8')
                else:
                    events = (root / 'events.jsonl').read_text(encoding='utf-8').splitlines()
                    changed = json.loads(events[-1]); changed['previous_hash'] = 'changed'
                    events[-1] = json.dumps(changed); (root / 'events.jsonl').write_text('\n'.join(events) + '\n', encoding='utf-8')
                checkpoint = sha(root / 'checkpoint.json')
                with no_work_guard() as calls, self.assertRaises(ValueError):
                    api.suite_resume(root)
                self.assert_no_work(calls)
                self.assertEqual(sha(root / 'checkpoint.json'), checkpoint)
                self.assertEqual(read(root / 'checkpoint.json')['status'], 'RUNNING')

    def test_missing_running_or_failed_gpu_commit_is_never_resubmitted(self):
        for missing_status in ('RUNNING', 'FAILED'):
            with self.subTest(status=missing_status):
                temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
                root = fixture(Path(temporary.name) / 'run')
                with patch.object(execute, '_dispatch', side_effect=fake_dispatch([])):
                    execute._drive(root, root / 'local.json', True, stop_after_node='serve-bf16')
                state = read(root / 'checkpoint.json')
                state['status'] = 'RUNNING'
                state['nodes']['serve-bf16'] = {'status': missing_status, 'attempts': 1}
                atomic(root / 'checkpoint.json', state)
                (root / 'nodes/serve-bf16/COMPLETION.json').unlink()
                pins = protected_hashes(root); checkpoint = sha(root / 'checkpoint.json')
                with no_work_guard() as calls, self.assertRaisesRegex(ValueError, 'automatic retry forbidden'):
                    api.suite_resume(root, local=root / 'local.json', allow_gpu=True)
                self.assert_no_work(calls)
                self.assertEqual(sha(root / 'checkpoint.json'), checkpoint)
                self.assertEqual(protected_hashes(root), pins)

    def test_middle_commit_gap_still_runs_only_remaining_cpu_stage(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = fixture(Path(temporary.name) / 'run')
        calls = []
        with patch.object(execute, '_dispatch', side_effect=fake_dispatch(calls)):
            execute._drive(root, root / 'local.json', True, stop_after_node='serve-rtn')
        state = read(root / 'checkpoint.json')
        state['nodes']['serve-rtn'] = {'status': 'RUNNING', 'attempts': 1}
        atomic(root / 'checkpoint.json', state)
        ledger = sha(root / 'budget-continuation.json')
        def cpu_only(*args, **kwargs):
            self.assertNotIn(args[2]['operation_kind'], execute.GPU_KINDS)
            return fake_dispatch(calls)(*args, **kwargs)
        with patch.object(execute, '_dispatch', side_effect=cpu_only):
            result = api.suite_resume(root)
        self.assertEqual(result['status'], 'COMPLETE_VALIDATED')
        self.assertEqual(calls, [node['node_id'] for node in read(root / 'EXECUTION_PLAN.json')['units']])
        self.assertEqual(sha(root / 'budget-continuation.json'), ledger)
        self.assert_disk_complete(root)

    def test_empty_required_coverage_and_current_source_mismatch_are_rejected(self):
        for alteration in ('empty_units', 'no_required', 'zero_count', 'source_identity'):
            with self.subTest(alteration=alteration):
                root = self.make(name=alteration)
                terminal_gap(root)
                plan = read(root / 'EXECUTION_PLAN.json'); plan.pop('execution_plan_identity')
                if alteration == 'empty_units': plan['units'] = []
                elif alteration == 'no_required':
                    for node in plan['units']: node['required'] = False
                elif alteration == 'zero_count': plan['units'][-1]['counts'] = {'documents': 0}
                else: plan['executor_source'] = 'explicit-source-mismatch'
                plan['execution_plan_identity'] = digest(plan); atomic(root / 'EXECUTION_PLAN.json', plan)
                checkpoint = sha(root / 'checkpoint.json')
                with no_work_guard() as calls, self.assertRaises(ValueError):
                    api.suite_resume(root)
                self.assert_no_work(calls)
                self.assertEqual(sha(root / 'checkpoint.json'), checkpoint)

    def test_missing_plan_node_or_checkpoint_coverage_is_not_completed(self):
        for alteration in ('missing_terminal', 'extra_complete', 'extra_pending'):
            with self.subTest(alteration=alteration):
                root = self.make(name=alteration)
                terminal, _ = terminal_gap(root)
                state = read(root / 'checkpoint.json')
                if alteration == 'missing_terminal':
                    state['nodes'].pop(terminal)
                else:
                    state['nodes']['unplanned-node'] = {
                        'status': 'COMPLETE_VALIDATED' if alteration == 'extra_complete' else 'PENDING',
                        'attempts': 0,
                    }
                atomic(root / 'checkpoint.json', state)
                checkpoint = sha(root / 'checkpoint.json')
                with no_work_guard() as calls, self.assertRaises((ValueError, KeyError)):
                    api.suite_resume(root)
                self.assert_no_work(calls)
                self.assertEqual(sha(root / 'checkpoint.json'), checkpoint)

    def test_terminal_reconciliation_respects_run_lock(self):
        root = self.make()
        terminal_gap(root)
        checkpoint = sha(root / 'checkpoint.json')
        with execute._lock(root), no_work_guard() as calls, self.assertRaisesRegex(ValueError, 'RUN_LOCKED'):
            api.suite_resume(root)
        self.assert_no_work(calls)
        self.assertEqual(sha(root / 'checkpoint.json'), checkpoint)
        with no_work_guard() as calls:
            result = api.suite_resume(root)
        self.assertTrue(result['checkpoint_reconciled'])
        self.assert_no_work(calls)

    def test_checkpoint_replace_failure_is_not_success_and_retry_recovers(self):
        root = self.make()
        terminal_gap(root)
        checkpoint = sha(root / 'checkpoint.json'); pins = protected_hashes(root)
        original_replace = os.replace
        failures = []
        def fail_before_replace(source, target):
            if Path(target) == root / 'checkpoint.json' and not failures:
                failures.append(str(target))
                raise OSError('synthetic checkpoint replace failure')
            return original_replace(source, target)
        with no_work_guard() as calls, patch('inference_kit.core.state.os.replace', side_effect=fail_before_replace):
            with self.assertRaisesRegex(OSError, 'checkpoint replace failure'):
                api.suite_resume(root)
        self.assert_no_work(calls)
        self.assertEqual(len(failures), 1)
        self.assertEqual(sha(root / 'checkpoint.json'), checkpoint)
        self.assertEqual(read(root / 'checkpoint.json')['status'], 'RUNNING')
        self.assertEqual(list(root.glob('checkpoint.json.*.tmp')), [])
        self.assertEqual(protected_hashes(root), pins)
        with no_work_guard() as calls:
            result = api.suite_resume(root)
            self.assert_disk_complete(root)
        self.assert_no_work(calls)
        self.assertTrue(result['checkpoint_reconciled'])
        self.assertEqual(protected_hashes(root), pins)


if __name__ == '__main__':
    unittest.main()
