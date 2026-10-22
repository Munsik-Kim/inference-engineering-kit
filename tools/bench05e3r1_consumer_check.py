"""Explicit E3-R1 installed-wheel terminal recovery acceptance.

Run with ``python -I`` and an explicitly copied, reviewed helper directory.
Only fixture preparation uses a synthetic dispatcher. Every terminal resume
uses the real installed executor with a dispatcher that raises if called.
The separate guarded CLI child invokes the installed public ``cli.main``;
the installed console executable is checked independently on completed data.
No archive code discovery, source-identity stub, ML import or live acquisition.
"""
import argparse
import copy
import json
import locale
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def inside(path, root):
    return path == root or root in path.parents


def guards(denied):
    counts = {'checkout': 0, 'SDK': 0, 'live_path_import': 0,
              'GPU_library': 0, 'network': 0, 'subprocess': 0, 'dispatch': 0}
    forbidden = {'torch', 'numpy', 'transformers', 'vllm', 'aiohttp',
                 'compressed_tensors', 'tokenizers', 'safetensors', 'pynvml'}
    def audit(event, values):
        if event in ('open', 'os.scandir', 'os.listdir') and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(values[0])))
            if any(inside(path, root) for root in denied):
                counts['checkout'] += 1
                raise PermissionError('Original checkout access blocked')
        if event == 'import' and values:
            module = str(values[0])
            if module.split('.')[0] in forbidden:
                counts['SDK'] += 1
                raise ImportError('ML/runtime SDK blocked')
            if module in ('inference_kit.suites.suite_live', 'inference_kit.adapters.nvml',
                          'inference_kit.probes.token_worker', 'inference_kit.probes.local_transformer'):
                counts['live_path_import'] += 1
                raise ImportError('Live acquisition path blocked')
        if event in ('socket.__new__', 'socket.connect', 'socket.bind', 'socket.getaddrinfo'):
            counts['network'] += 1
            raise PermissionError('Network blocked')
        if event == 'ctypes.dlopen' and values and any(name in str(values[0]).lower() for name in ('cuda', 'nvidia', 'nvml', 'cublas', 'torch')):
            counts['GPU_library'] += 1
            raise PermissionError('GPU library loading blocked')
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
            counts['subprocess'] += 1
            raise PermissionError('Child process blocked inside guarded phase')
    def dispatch(*args, **kwargs):
        counts['dispatch'] += 1
        raise AssertionError('Terminal resume must not dispatch a node')
    return counts, audit, dispatch, forbidden


def identity(expected_origin, expected_version):
    import inference_kit
    from inference_kit.suites import suite_execution
    origin = Path(inference_kit.__file__).absolute()
    assert inside(origin, expected_origin.absolute()) and 'site-packages' in origin.parts, origin
    assert inference_kit.__version__ == expected_version
    return {'version': inference_kit.__version__, 'package_import_origin': str(origin),
            'interpreter': sys.executable, 'package_source_identity': suite_execution.source_identity()}


def child_cli(args):
    """Explicit separate-process public CLI with in-process guards of its own."""
    assert sys.flags.isolated
    assert locale.getpreferredencoding(False).lower() not in ('utf8', 'utf-8')
    assert sys.stdout.encoding.lower() in ('ascii', 'us-ascii')
    os.chdir(args.output.absolute())
    counts, audit, sentinel, forbidden = guards([p.absolute() for p in args.deny_root])
    sys.addaudithook(audit)
    installed = identity(args.expected_origin, args.expected_version)
    from inference_kit.cli import main as cli_main
    from inference_kit.suites import suite_execution
    argv = ['bench', 'suite', args.child_action, '--run', str(args.child_run)]
    if args.child_action == 'status':
        argv += ['--output', str(args.child_status_output)]
    with patch.object(suite_execution, '_dispatch', side_effect=sentinel):
        result = cli_main(argv)
    assert result in (None, 0), result
    assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)
    assert not any(counts.values()), counts
    receipt = {'status': 'PASS', 'guard': counts, **installed,
               'preferred_encoding': locale.getpreferredencoding(False),
               'stdout_encoding': sys.stdout.encoding,
               'scope': 'This CLI child only; no OS sandbox or inherited parent audit hook'}
    args.child_receipt.write_text(json.dumps(receipt, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'expected-origin'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-version', default='0.1.1.dev14')
    parser.add_argument('--deny-root', action='append', default=[], type=Path)
    parser.add_argument('--helpers', type=Path)
    parser.add_argument('--child-action', choices=('resume', 'status'))
    parser.add_argument('--child-run', type=Path)
    parser.add_argument('--child-status-output', type=Path)
    parser.add_argument('--child-receipt', type=Path)
    args = parser.parse_args()
    if args.child_action:
        child_cli(args)
        return
    assert sys.flags.isolated, 'Use Python -I'
    assert args.helpers is not None
    output = args.output.absolute()
    denied = [p.absolute() for p in args.deny_root]
    helpers = args.helpers.absolute()
    assert not output.exists() and not output.is_symlink()
    assert not any(inside(output, root) or inside(helpers, root) for root in denied)
    assert (helpers / 'tests/bench05e3/test_execution.py').is_file()
    assert not (helpers / 'inference_kit').exists() and not (helpers / 'src/inference_kit').exists()
    output.mkdir(parents=True)
    os.chdir(output)
    installed = identity(args.expected_origin, args.expected_version)
    # This reviewed folder contains fixture helpers, never the inference package.
    sys.path.insert(0, str(helpers))
    from tests.bench05e3.test_execution import fixture, fake_dispatch
    from inference_kit import suite_resume, suite_status
    from inference_kit.core.io import digest, read, sha
    from inference_kit.core.state import atomic
    from inference_kit.suites import suite_execution as execute
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.update(LC_ALL='C', LANG='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0', PYTHONIOENCODING='ascii')
    child_receipts = []
    console_records = []
    api_records = []
    refusal_records = []

    def snapshot(root):
        selected = ['EXECUTION_PLAN.json', 'suite-execution-context.json', 'prepared/input.json',
                    'budget-continuation.json', 'parent-ledger.json']
        selected += [str(p.relative_to(root)) for p in root.glob('nodes/*/COMPLETION.json')]
        selected += [str(p.relative_to(root)) for p in root.glob('nodes/*/attempt*/EVIDENCE.json')]
        return {name: {'bytes': (root / name).stat().st_size, 'sha256': sha(root / name)} for name in sorted(selected)}

    def prepared_case(label, kind):
        root = fixture(output / label)
        calls = []
        with patch.object(execute, '_dispatch', side_effect=fake_dispatch(calls)):
            result = execute._drive(root, root / 'local.json', True)
        assert result['status'] == 'COMPLETE_VALIDATED'
        plan = read(root / 'EXECUTION_PLAN.json')
        assert plan['executor_source'] == execute.source_identity(), 'Fixture uses actual installed source'
        terminal = plan['units'][-1]['node_id']
        state = read(root / 'checkpoint.json')
        expected = copy.deepcopy(state)
        if kind == 'terminal_gap':
            state['status'] = 'RUNNING'
            state['nodes'][terminal]['status'] = 'RUNNING'
            state['nodes'][terminal].pop('completion', None)
            state['nodes'][terminal].pop('output', None)
            atomic(root / 'checkpoint.json', state)
        elif kind == 'global_gap':
            state['status'] = 'RUNNING'
            atomic(root / 'checkpoint.json', state)
        assert calls == [node['node_id'] for node in plan['units']]
        return root, terminal, expected, calls

    def stable_fields(expected, actual):
        # Reconciliation may add only its own explicit metadata. Historical
        # attempts, execution/source IDs and time accounting remain exact.
        for key in ('run_id', 'execution_plan_identity', 'controller_sessions',
                    'active_controller_seconds', 'paused_seconds', 'pause_started_monotonic'):
            assert actual[key] == expected[key], (key, actual[key], expected[key])
        for key in ('started_utc', 'started_monotonic'):
            if key in expected:
                assert actual[key] == expected[key]
        assert set(actual['nodes']) == set(expected['nodes'])
        for name, entry in expected['nodes'].items():
            assert actual['nodes'][name]['attempts'] == entry['attempts']
            assert actual['nodes'][name]['status'] == 'COMPLETE_VALIDATED'
            assert actual['nodes'][name]['completion'] == entry['completion']
            assert actual['nodes'][name]['output'] == entry['output']

    def cli(root, action, label, expect_ok=True):
        receipt = output / (label + '-guard.json')
        command = [sys.executable, '-I', '-X', 'utf8=0', str(Path(__file__).absolute()),
                   '--output', str(output), '--expected-origin', str(args.expected_origin),
                   '--expected-version', args.expected_version,
                   '--child-action', action, '--child-run', str(root), '--child-receipt', str(receipt)]
        if action == 'status':
            command += ['--child-status-output', str(output / (label + '-status'))]
        for root_deny in denied:
            command += ['--deny-root', str(root_deny)]
        response = subprocess.run(command, cwd=output, env=env, capture_output=True, timeout=90)
        (output / (label + '.stdout')).write_bytes(response.stdout)
        (output / (label + '.stderr')).write_bytes(response.stderr)
        response.stdout.decode('ascii')
        if expect_ok:
            assert response.returncode == 0, response.stderr.decode('ascii', errors='backslashreplace')
            value = json.loads(response.stdout.decode('ascii'))
            child_receipts.append(read(receipt))
            return value
        assert response.returncode != 0, 'Invalid fixture accepted by CLI'
        return {'returncode': response.returncode,
                'stderr': response.stderr.decode('ascii', errors='backslashreplace')}

    # Actual separate CLI processes are run before this parent's API audit hook.
    # Non-UTF8 POSIX argv/path handling is checked with ASCII paths. Korean
    # filesystem paths are separately exercised by the UTF8 API cases below.
    for label, kind in (('CLI-terminal-node', 'terminal_gap'), ('CLI-global', 'global_gap'), ('CLI-normal', 'normal')):
        root, terminal, expected, initial_calls = prepared_case(label, kind)
        pinned = snapshot(root)
        before_checkpoint = sha(root / 'checkpoint.json')
        first = cli(root, 'resume', label + '-first')
        disk = read(root / 'checkpoint.json')
        assert first['status'] == disk['status'] == 'COMPLETE_VALIDATED'
        assert first['checkpoint_reconciled'] is (kind != 'normal')
        stable_fields(expected, disk)
        assert disk['nodes'][terminal]['output'] == read(root / disk['nodes'][terminal]['completion'])['evidence_directory']
        after_checkpoint = sha(root / 'checkpoint.json')
        assert (before_checkpoint != after_checkpoint) == (kind != 'normal')
        status = cli(root, 'status', label + '-status')
        assert status['status'] == 'COMPLETE_VALIDATED'
        second = cli(root, 'resume', label + '-second')
        assert second['status'] == 'COMPLETE_VALIDATED' and second['noop'] is True
        assert second['checkpoint_reconciled'] is False and second['reconciled_nodes'] == []
        assert sha(root / 'checkpoint.json') == after_checkpoint and snapshot(root) == pinned
        console_records.append({'case': kind, 'terminal_node': terminal, 'resume': first,
            'second_resume': second, 'fresh_process_status': status['status'],
            'checkpoint_before': before_checkpoint, 'checkpoint_after': after_checkpoint,
            'pinned_evidence': pinned, 'pinned_unchanged': True,
            'synthetic_fixture_preparation_dispatches': initial_calls,
            'terminal_dispatches': 0})
    # Independently exercise the installed console entry point, not only cli.main.
    direct_root = output / 'CLI-normal'
    pinned = snapshot(direct_root)
    checkpoint = sha(direct_root / 'checkpoint.json')
    direct_command = [str(Path(sys.executable).parent / 'inference-kit'), 'bench', 'suite',
                      'resume', '--run', str(direct_root)]
    direct = subprocess.run(direct_command, cwd=output, env=env, capture_output=True, timeout=90)
    assert direct.returncode == 0, direct.stderr.decode('ascii', errors='backslashreplace')
    direct_value = json.loads(direct.stdout.decode('ascii'))
    assert direct_value['status'] == 'COMPLETE_VALIDATED' and direct_value['noop'] is True
    assert snapshot(direct_root) == pinned and sha(direct_root / 'checkpoint.json') == checkpoint
    console_records.append({'case': 'installed_console_normal_noop', 'command': direct_command,
                            'result': direct_value, 'pinned_unchanged': True,
                            'guard_scope': 'Direct console subprocess separately checked; no parent audit-hook inheritance claim'})

    # Source mismatch is a validly re-hashed synthetic plan, not a source stub.
    wrong, terminal, expected, initial_calls = prepared_case('CLI-source-refusal', 'terminal_gap')
    plan = read(wrong / 'EXECUTION_PLAN.json')
    plan.pop('execution_plan_identity')
    plan['executor_source'] = 'intentional-source-mismatch'
    plan['execution_plan_identity'] = digest(plan)
    atomic(wrong / 'EXECUTION_PLAN.json', plan)
    before = sha(wrong / 'checkpoint.json')
    rejected = cli(wrong, 'resume', 'CLI-source-rejection', expect_ok=False)
    assert 'Executor source changed' in rejected['stderr']
    assert sha(wrong / 'checkpoint.json') == before
    refusal_records.append({'case': 'CLI_source_mismatch', 'status': 'REJECTED', **rejected})

    counts, audit, sentinel, forbidden = guards(denied)
    sys.addaudithook(audit)
    for label, kind in (('API-마지막-node', 'terminal_gap'), ('API-global', 'global_gap'), ('API-normal', 'normal')):
        root, terminal, expected, initial_calls = prepared_case(label, kind)
        pinned = snapshot(root)
        before_checkpoint = sha(root / 'checkpoint.json')
        with patch.object(execute, '_dispatch', side_effect=sentinel):
            first = suite_resume(root, allow_gpu=False)
            status = suite_status(root)
        disk = read(root / 'checkpoint.json')
        assert first['status'] == status['status'] == disk['status'] == 'COMPLETE_VALIDATED'
        assert first['checkpoint_reconciled'] is (kind != 'normal')
        stable_fields(expected, disk)
        after_checkpoint = sha(root / 'checkpoint.json')
        assert (before_checkpoint != after_checkpoint) == (kind != 'normal')
        with patch.object(execute, '_dispatch', side_effect=sentinel):
            second = suite_resume(root, allow_gpu=False)
        assert second['status'] == 'COMPLETE_VALIDATED' and second['noop'] is True
        assert second['checkpoint_reconciled'] is False and second['reconciled_nodes'] == []
        assert sha(root / 'checkpoint.json') == after_checkpoint and snapshot(root) == pinned
        api_records.append({'case': kind, 'terminal_node': terminal, 'resume': first,
            'status': status['status'], 'disk_status': disk['status'], 'second_resume': second,
            'checkpoint_before': before_checkpoint, 'checkpoint_after': after_checkpoint,
            'output_recovered': disk['nodes'][terminal]['output'], 'pinned_evidence': pinned,
            'pinned_unchanged': True, 'synthetic_fixture_preparation_dispatches': initial_calls,
            'terminal_dispatches': counts['dispatch']})

    for alteration in ('terminal_completion_missing', 'terminal_evidence_changed',
                       'immutable_input_changed', 'ledger_changed', 'source_changed'):
        root, terminal, expected, initial_calls = prepared_case('API-refusal-' + alteration, 'global_gap')
        if alteration == 'terminal_completion_missing':
            (root / 'nodes' / terminal / 'COMPLETION.json').unlink()
        elif alteration == 'terminal_evidence_changed':
            receipt = read(root / 'nodes' / terminal / 'COMPLETION.json')
            file_name = next(iter(receipt['files']))
            (root / file_name).write_text('{}', encoding='utf-8')
        elif alteration == 'immutable_input_changed':
            (root / 'prepared/input.json').write_text('{}', encoding='utf-8')
        elif alteration == 'ledger_changed':
            ledger = read(root / 'budget-continuation.json')
            ledger['used_seconds'] += 1
            atomic(root / 'budget-continuation.json', ledger)
        else:
            plan = read(root / 'EXECUTION_PLAN.json')
            plan.pop('execution_plan_identity')
            plan['executor_source'] = 'intentional-source-mismatch'
            plan['execution_plan_identity'] = digest(plan)
            atomic(root / 'EXECUTION_PLAN.json', plan)
        checkpoint = sha(root / 'checkpoint.json')
        try:
            with patch.object(execute, '_dispatch', side_effect=sentinel):
                suite_resume(root, allow_gpu=False)
        except (ValueError, FileNotFoundError) as exc:
            message = str(exc)
        else:
            raise AssertionError('Invalid fixture accepted: ' + alteration)
        assert sha(root / 'checkpoint.json') == checkpoint
        assert read(root / 'checkpoint.json')['status'] == 'RUNNING'
        refusal_records.append({'case': alteration, 'status': 'REJECTED',
                                'error': message, 'checkpoint_unchanged': True})

    assert not any(counts.values()), counts
    assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)
    receipt = {'status': 'PASS', 'evidence_kind': 'SYNTHETIC_EXECUTOR_TEST', **installed,
               'API_cases': api_records, 'CLI_cases': console_records,
               'refusals': refusal_records, 'API_guard': counts,
               'CLI_child_guards': child_receipts,
               'guard_scope': 'API process and explicit CLI child guards only; direct installed console checked separately; no OS sandbox/inherited-hook claim',
               'fixtures': 'Fresh synthetic plans use the installed package source_identity; no source_identity/compiler/completion/ledger validator stub',
               'nonUTF8_console_environment': {name: env[name] for name in ('LC_ALL', 'LANG', 'PYTHONUTF8', 'PYTHONCOERCECLOCALE', 'PYTHONIOENCODING')},
               'GPU_queries': 0, 'model_loads': 0, 'inference_POSTs': 0,
               'tokenizer_calls': 0, 'campaign_debit_seconds': 0,
               'new_live_acquisitions': 0,
               'historical_live': 'Not resumed, migrated, modified or independently reacquired'}
    (output / 'WHEEL_CONSUMER_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', **installed, 'API_cases': len(api_records),
                      'CLI_cases': len(console_records), 'refusals': len(refusal_records),
                      'API_guard': counts, 'output': str(output)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
