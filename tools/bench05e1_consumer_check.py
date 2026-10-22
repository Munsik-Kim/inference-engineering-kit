"""Installed-wheel E1 consumer using only parent-copied explicit raw evidence.

Run with a new base CPU venv's ``python -I -B``. Console children run before the
in-process audit hook and are verified separately. The hook is not inherited by
children and does not claim an OS sandbox. This tool never copies raw inputs or
executes code stored in an archive.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

SDK = frozenset(('torch', 'tf', 'tensorflow', 'transformers', 'vllm', 'numpy',
                 'aiohttp', 'compressed_tensors', 'safetensors', 'accelerate', 'tokenizers'))


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
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Invalid JSON constant: ' + value)))


def read(path):
    return strict_loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(path, root):
    return path == root or root in path.parents


def validate_inputs(sources, forbidden):
    """Inputs must already be explicitly copied outside every denied checkout."""
    sources = Path(sources).absolute()
    root = sources.parent.resolve()
    require(sources.is_file() and not sources.is_symlink(), 'Missing/unsafe explicit source manifest')
    require(not any(inside(sources.resolve(), base) for base in forbidden), 'Source manifest remains under denied checkout')
    manifest = read(sources)
    require(manifest.get('schema_version') == 1 and isinstance(manifest.get('sources'), list)
            and manifest['sources'], 'Nonempty explicit source manifest required')
    inventory = []
    for source in manifest['sources']:
        path = Path(source['path'])
        path = path if path.is_absolute() else root / path
        require(path.exists() and not path.is_symlink(), 'Missing/unsafe explicitly copied source')
        resolved = path.resolve()
        require(inside(resolved, root), 'Consumer source must be under the explicitly copied manifest root')
        require(not any(inside(resolved, base) for base in forbidden), 'Source remains under denied checkout')
        inventory.append({'source_id': source['source_id'], 'adapter': source['adapter'],
                          'relative_locator': resolved.relative_to(root).as_posix(),
                          'raw_manifest_sha256': source.get('raw_manifest_sha256')})
    return inventory


def clean_environment():
    result = dict(os.environ)
    result.pop('PYTHONPATH', None)
    result.pop('PYTHONHOME', None)
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    return result


def invoke(argv, cwd, environment):
    child = subprocess.run(argv, cwd=cwd, env=environment, capture_output=True, timeout=300)
    require(child.returncode == 0, 'Consumer command failed: ' + child.stderr.decode('ascii', errors='backslashreplace'))
    # CLI stdout must be ASCII JSON even when values contain Korean.
    value = strict_loads(child.stdout.decode('ascii'))
    return value, {'argv': [str(a) for a in argv], 'returncode': child.returncode,
                   'stdout_json_ascii': True, 'stdout_bytes': len(child.stdout),
                   'stderr_bytes': len(child.stderr), 'audit_hook_inherited': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deny-root', action='append', required=True, type=Path)
    parser.add_argument('--expected-origin', required=True, type=Path,
                        help='Actual installed site-packages root, or its inference_kit package directory')
    parser.add_argument('--expected-version', default='0.1.1.dev11')
    args = parser.parse_args(argv)
    require(sys.flags.isolated, 'Use the new installed CPU venv interpreter with -I')
    forbidden = [p.absolute().resolve() for p in args.deny_root]
    sources = args.sources.absolute(); output = args.output.absolute()
    require(not any(inside(output.resolve(), base) for base in forbidden), 'Consumer output must be outside denied checkout')
    require(not output.exists(), 'Consumer output must be new')
    copied = validate_inputs(sources, forbidden)
    output.mkdir(parents=True, exist_ok=False)
    environment = clean_environment()
    # Children are deliberately run before installing the in-process guard.
    # Their API/CLI output and canonical index meaning are checked below.
    module = [sys.executable, '-I', '-B', '-m', 'inference_kit.cli']
    child_commands = []
    _, command = invoke(module + ['bench', 'index', 'build', '--sources', str(sources),
                        '--db', str(output / 'console.sqlite'), '--output', str(output / 'console-build')], output, environment)
    child_commands.append(command)
    _, command = invoke(module + ['bench', 'index', 'query', '--db', str(output / 'console.sqlite'),
                        '--output', str(output / 'console-query')], output, environment)
    child_commands.append(command)
    _, command = invoke(module + ['bench', 'index', 'report', '--db', str(output / 'console.sqlite'),
                        '--output', str(output / 'console-report')], output, environment)
    child_commands.append(command)
    console = Path(sys.executable).parent / 'inference-kit'
    require(console.is_file(), 'Ordinary wheel console entrypoint is missing')
    _, command = invoke([str(console), 'bench', 'index', 'query', '--db', str(output / 'console.sqlite'),
                        '--output', str(output / 'console-script-query')], output, environment)
    child_commands.append(command)
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        os.environ.pop(name, None)
    os.chdir(output)
    sys.path[:] = [p for p in sys.path if p and not any(inside(Path(p).absolute(), base) for base in forbidden)]
    audit_counts = {'denied_checkout': 0, 'denied_SDK': 0, 'denied_network': 0, 'denied_subprocess': 0}

    def guard(event, values):
        if event in ('open', 'os.listdir', 'os.scandir') and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(values[0])))
            if any(inside(path, base) for base in forbidden):
                audit_counts['denied_checkout'] += 1
                raise PermissionError('Original checkout read blocked by in-process consumer guard')
        if event == 'import' and values and str(values[0]).split('.')[0] in SDK:
            audit_counts['denied_SDK'] += 1
            raise ImportError('ML/HTTP SDK import blocked by in-process consumer guard')
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'):
            audit_counts['denied_network'] += 1
            raise PermissionError('Network blocked by in-process consumer guard')
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
            audit_counts['denied_subprocess'] += 1
            raise PermissionError('Subprocess blocked by in-process consumer guard')

    sys.addaudithook(guard)
    import inference_kit
    from inference_kit.cli import main as cli
    from inference_kit.core.parity import equal
    origin = Path(inference_kit.__file__).absolute()
    require('site-packages' in origin.parts and inside(origin, args.expected_origin.absolute())
            and inside(origin, Path(sys.prefix).absolute()), 'Package origin is not the expected ordinary installed wheel')
    require(inference_kit.__version__ == args.expected_version, 'Installed package version differs')
    require(not SDK.intersection(name.split('.')[0] for name in sys.modules), 'ML/HTTP SDK was imported before guarded API')
    build = inference_kit.index_build(sources, output / 'api.sqlite', output / 'api-build')
    query = inference_kit.index_query(output / 'api.sqlite', output / 'api-query')
    report = inference_kit.index_report(output / 'api.sqlite', output / 'api-report')
    initial = read(output / 'api-query/canonical-export.json')
    require(all(s['validation']['status'] == 'RAW_HASH_VERIFIED' for s in initial['sources']), 'Consumer contains stale/missing evidence')
    equal(initial, read(output / 'console-query/canonical-export.json'))
    equal(initial, read(output / 'console-script-query/canonical-export.json'))
    added = inference_kit.index_add(sources, output / 'api.sqlite', output / 'api-add-identical')
    equal(added['before'], added['after'])
    inference_kit.index_query(output / 'api.sqlite', output / 'api-after-add')
    equal(initial, read(output / 'api-after-add/canonical-export.json'))
    rebuild = inference_kit.index_build(sources, output / 'rebuilt.sqlite', output / 'rebuilt-build')
    inference_kit.index_query(output / 'rebuilt.sqlite', output / 'rebuilt-query')
    equal(initial, read(output / 'rebuilt-query/canonical-export.json'))
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = cli(['bench', 'index', 'query', '--db', str(output / 'api.sqlite'), '--output', str(output / 'inprocess-cli-query')])
    require(code == 0 and not stderr.getvalue(), 'In-process installed CLI query failed')
    stdout.getvalue().encode('ascii')
    equal(strict_loads(stdout.getvalue()), json.loads(json.dumps(query)))
    equal(initial, read(output / 'inprocess-cli-query/canonical-export.json'))
    equal(report['table_counts'], read(output / 'console-report/REPORT_MANIFEST.json')['table_counts'])
    require(not SDK.intersection(name.split('.')[0] for name in sys.modules), 'ML/HTTP SDK imported by CPU index')
    receipt = {'schema_version': 1, 'status': 'E1_INSTALLED_CPU_INDEX_CONSUMER_PASS',
               'package_version': inference_kit.__version__, 'package_import_origin': str(origin),
               'interpreter': sys.executable, 'python_version': sys.version, 'isolated_mode': bool(sys.flags.isolated),
               'sys_path': list(sys.path), 'input_manifest_sha256': sha(sources), 'explicit_copied_sources': copied,
               'counts': query['counts'], 'logical_identity': query['logical_identity'],
               'dedup_count_unchanged': added['before'] == added['after'],
               'rebuilt_logical_export_equal': True, 'canonical_export_normalizations': [],
               'API_inprocess_CLI_module_console_parity': 'EXACT_LOGICAL_EXPORT',
               'report_table_counts_equal': True, 'console_commands': child_commands,
               'inprocess_guard': {'deny_roots': [str(p) for p in forbidden], 'counts': audit_counts,
                                   'scope': 'Current Python process only; not inherited by children; no OS sandbox claim'},
               'separate_console_scope': 'Children ran before in-process guard, with clean PYTHONPATH/PYTHONHOME and external cwd; module CLI uses -I',
               'raw_copy': 'Parent explicitly copied inputs before this tool; no hidden source fallback or raw copying',
               'GPU_actions': 0, 'model_loads': 0, 'tokenizer_loads': 0, 'network_requests': 0,
               'downloads': 0, 'ledger_changes': 0, 'archive_code_executed': False,
               'browser_visual_validation': 'NOT_RUN', 'remote_CI': 'NOT_RUN'}
    write(output / 'CONSUMER_RECEIPT.json', receipt)
    print(json.dumps({'status': receipt['status'], 'version': inference_kit.__version__, 'counts': query['counts'],
                      'logical_identity': query['logical_identity']}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'FAILED', 'error': str(error)}, ensure_ascii=True), file=sys.stderr)
        raise SystemExit(2)
