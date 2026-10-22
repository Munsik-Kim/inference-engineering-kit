"""Installed-wheel, metadata-only E2 suite compiler consumer.

Inputs are explicitly copied before this command. Module/console subprocesses
run before an in-process checkout/SDK/network/subprocess audit guard. The guard
is not an OS sandbox and does not propagate to child processes. No archive code
is executed and no artifact presence or live readiness is claimed.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

SDK = frozenset(('torch', 'tf', 'tensorflow', 'transformers', 'vllm', 'numpy', 'aiohttp',
                 'compressed_tensors', 'safetensors', 'accelerate', 'tokenizers'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant: ' + value)))


def read(path):
    return strict_loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')


def inside(path, root):
    return path == root or root in path.parents


def explicit_file(value, anchor, forbidden):
    path = Path(value)
    path = path if path.is_absolute() else anchor / path
    require(path.is_file() and not path.is_symlink(), 'Missing or unsafe copied metadata input')
    resolved = path.resolve()
    require(inside(resolved, anchor), 'Copied metadata escapes explicit input root')
    require(not any(inside(resolved, root) for root in forbidden), 'Input remains in original checkout')
    return resolved


def load_cases(path, forbidden):
    path = Path(path).absolute()
    anchor = path.parent.resolve()
    path = explicit_file(path, anchor, forbidden)
    value = read(path)
    require(set(value) <= {'schema_version', 'cases'} and value.get('schema_version') == 1,
            'Unknown consumer cases schema/field')
    require(isinstance(value.get('cases'), list) and value['cases'], 'Nonempty explicit cases required')
    result = []; names = set()
    for case in value['cases']:
        require(isinstance(case, dict) and set(case) <= {'id', 'suite_path', 'expected_status', 'expect_error'}, 'Unknown consumer case field')
        name = case.get('id')
        require(isinstance(name, str) and name and name.replace('-', '').replace('_', '').isalnum()
                and name not in names, 'Invalid or duplicate consumer case ID')
        names.add(name)
        require(type(case.get('expect_error', False)) is bool, 'expect_error must be boolean')
        result.append(dict(case, suite_path=explicit_file(case['suite_path'], anchor, forbidden)))
    return anchor, result


def clean_environment():
    result = dict(os.environ)
    result.pop('PYTHONPATH', None); result.pop('PYTHONHOME', None)
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    return result


def invoke(argv, cwd, environment, expect_error=False):
    child = subprocess.run(argv, cwd=cwd, env=environment, capture_output=True, timeout=180)
    require((child.returncode != 0) if expect_error else child.returncode == 0,
            'Unexpected consumer command exit: ' + child.stderr.decode('ascii', errors='backslashreplace'))
    # The application uses ASCII-safe JSON on both success and error channels.
    channel = child.stderr if expect_error else child.stdout
    value = strict_loads(channel.decode('ascii'))
    return value, {'argv': [str(a) for a in argv], 'returncode': child.returncode,
                   'stdout_bytes': len(child.stdout), 'stderr_bytes': len(child.stderr),
                   'json_ascii': True, 'audit_hook_inherited': False}


def validate_plan(plan, expected_status=None):
    require(isinstance(plan, dict) and isinstance(plan.get('plan_identity'), str), 'Missing frozen plan identity')
    require(plan.get('execution_count') == 0 and type(plan['execution_count']) is int
            and plan.get('acquisition_count') == 0 and type(plan['acquisition_count']) is int,
            'A CPU compilation was promoted to a new acquisition or execution')
    if expected_status is not None:
        allowed = [expected_status] if isinstance(expected_status, str) else expected_status
        require(plan.get('status') in allowed, 'Unexpected suite plan status')
    return {'plan_identity': plan['plan_identity'], 'status': plan['status'],
            'execution_count': 0, 'acquisition_count': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--local', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deny-root', action='append', required=True, type=Path)
    parser.add_argument('--expected-origin', required=True, type=Path)
    parser.add_argument('--expected-version', default='0.1.1.dev12')
    args = parser.parse_args(argv)
    require(sys.flags.isolated, 'Consumer must run with the installed interpreter and -I')
    forbidden = [path.absolute().resolve() for path in args.deny_root]
    anchor, cases = load_cases(args.cases, forbidden)
    sources = explicit_file(args.sources.absolute(), anchor, forbidden)
    local = explicit_file(args.local.absolute(), anchor, forbidden)
    local_value = read(local)
    require(local_value.get('artifact_check') == 'not_requested'
            and local_value.get('runtime_check') == 'not_run'
            and local_value.get('authorization') == 'NONE', 'Consumer may only validate metadata contracts')
    output = args.output.absolute()
    require(not output.exists() and not any(inside(output.resolve(), p) for p in forbidden), 'Consumer output must be a new directory outside denied roots')
    output.mkdir(parents=True, exist_ok=False)
    environment = clean_environment(); module = [sys.executable, '-I', '-B', '-m', 'inference_kit.cli']
    console = Path(sys.executable).parent / 'inference-kit'
    require(console.is_file(), 'Ordinary installed console entrypoint missing')
    child_commands = []; child_plans = {}
    for case in cases:
        name = case['id']; expected_error = case.get('expect_error', False)
        directory = output / ('module-' + name)
        _, command = invoke(module + ['bench', 'suite', 'plan', '--suite', str(case['suite_path']),
            '--sources', str(sources), '--local', str(local), '--output', str(directory)], output, environment, expected_error)
        child_commands.append(command)
        if expected_error:
            continue
        plan_file = directory / 'FROZEN_PLAN.json'
        child_plans[name] = validate_plan(read(plan_file), case.get('expected_status'))
        for action in ('validate', 'describe'):
            cmd = module + ['bench', 'suite', action, '--plan', str(plan_file), '--output', str(output / ('module-' + action + '-' + name))]
            if action == 'validate':
                cmd += ['--sources', str(sources), '--local', str(local)]
            _, command = invoke(cmd, output, environment); child_commands.append(command)
        console_dir = output / ('console-' + name)
        _, command = invoke([str(console), 'bench', 'suite', 'plan', '--suite', str(case['suite_path']),
            '--sources', str(sources), '--local', str(local), '--output', str(console_dir)], output, environment)
        child_commands.append(command)
        require(validate_plan(read(console_dir / 'FROZEN_PLAN.json'), case.get('expected_status')) == child_plans[name],
                'Console and module compiler semantics differ')
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        os.environ.pop(key, None)
    os.chdir(output)
    sys.path[:] = [p for p in sys.path if p and not any(inside(Path(p).absolute(), base) for base in forbidden)]
    audit_counts = {'checkout': 0, 'SDK': 0, 'network': 0, 'subprocess': 0}

    def guard(event, values):
        if event in ('open', 'os.listdir', 'os.scandir') and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(values[0])))
            if any(inside(path, base) for base in forbidden):
                audit_counts['checkout'] += 1
                raise PermissionError('Original checkout access blocked by in-process guard')
        if event == 'import' and values and str(values[0]).split('.')[0] in SDK:
            audit_counts['SDK'] += 1
            raise ImportError('ML SDK import blocked by in-process guard')
        if event in ('socket.__new__', 'socket.bind', 'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'):
            audit_counts['network'] += 1
            raise PermissionError('Network access blocked by in-process guard')
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
            audit_counts['subprocess'] += 1
            raise PermissionError('Subprocess blocked by in-process guard')

    sys.addaudithook(guard)
    import inference_kit
    from inference_kit.cli import main as cli
    origin = Path(inference_kit.__file__).absolute()
    require('site-packages' in origin.parts and inside(origin, args.expected_origin.absolute())
            and inside(origin, Path(sys.prefix).absolute()), 'Package origin is not the expected installed wheel')
    require(inference_kit.__version__ == args.expected_version, 'Installed wheel version mismatch')
    require(not SDK.intersection(n.split('.')[0] for n in sys.modules), 'ML SDK imported before guarded compilation')
    results = []
    for case in cases:
        name = case['id']; directory = output / ('api-' + name)
        if case.get('expect_error', False):
            try:
                inference_kit.suite_plan(case['suite_path'], sources, local, directory)
            except ValueError as exc:
                results.append({'id': name, 'status': 'REFUSED_AS_EXPECTED', 'reason': str(exc)})
            else:
                raise ValueError('Invalid case was accepted by installed API')
            continue
        inference_kit.suite_plan(case['suite_path'], sources, local, directory)
        plan_file = directory / 'FROZEN_PLAN.json'
        meaning = validate_plan(read(plan_file), case.get('expected_status'))
        require(meaning == child_plans[name], 'Installed API and module/console compiler semantics differ')
        inference_kit.suite_validate(plan_file, sources, output / ('api-validate-' + name), local=local)
        inference_kit.suite_describe(plan_file, output / ('api-describe-' + name))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = cli(['bench', 'suite', 'describe', '--plan', str(plan_file), '--output', str(output / ('in-process-cli-describe-' + name))])
        require(code == 0, 'Guarded in-process CLI describe failed')
        strict_loads(stream.getvalue().encode('ascii').decode('ascii'))
        results.append({'id': name, **meaning, 'API_module_console_semantic_parity': True})
    require(not SDK.intersection(n.split('.')[0] for n in sys.modules), 'CPU compilation imported an ML SDK')
    receipt = {'schema_version': 1, 'status': 'PASS', 'package_version': inference_kit.__version__,
        'package_import_origin': str(origin), 'interpreter': sys.executable, 'python_version': sys.version, 'isolated': bool(sys.flags.isolated),
        'cases': results, 'subprocesses': child_commands, 'audit_counts': audit_counts,
        'audit_scope': 'In-process API and CLI only; not an OS sandbox; module/console children checked separately before guard',
        'input_scope': 'Explicit copied lightweight prepared metadata and contract fixtures only',
        'artifact_presence': 'ARTIFACT_PRESENCE_NOT_CHECKED', 'runtime_preflight': 'CHECK_NOT_RUN',
        'GPU_queries': 0, 'GPU_actions': 0, 'model_loads': 0, 'tokenizer_calls': 0, 'network_actions': 0,
        'execution_count': 0, 'acquisition_count': 0, 'remote_ci': 'NOT_RUN'}
    write(output / 'CONSUMER_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
