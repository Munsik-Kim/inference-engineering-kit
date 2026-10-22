"""Explicit installed-wheel prepared binding consumer; no live acquisition.

Saved prepared contracts, Suite plans and metadata-only execution layouts are
checked through API, module CLI and a separate installed console. Subprocesses
run before the in-process audit hook; this is not an OS sandbox or inherited
child guard. A closed synthetic unit subset may be run explicitly and is
reported separately from actual source metadata and absent new live raw.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

SDK = frozenset(('torch', 'tensorflow', 'transformers', 'tokenizers', 'numpy',
                 'vllm', 'aiohttp', 'compressed_tensors', 'safetensors', 'accelerate'))
METADATA_LOCAL = {'schema_version': 1, 'artifact_check': 'not_requested', 'artifact_roots': {},
                  'runtime_check': 'not_run', 'authorization': 'NONE'}


def require(value, message):
    if not value:
        raise ValueError(message)


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def read(path):
    return loads(Path(path).read_text(encoding='utf-8'))


def inside(path, root):
    return path == root or root in path.parents


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(name):
    require(isinstance(name, str), 'Relative input path must be text')
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and '..' not in path.parts and
            '\\' not in name and ':' not in name and path.as_posix() == name, 'Unsafe relative input path')
    return path


def copy_prepared(manifest, output):
    """Copy exactly a pinned bundle, or a standalone explicitly declared slice."""
    manifest = manifest.absolute(); root = manifest.parent.resolve(); value = read(manifest)
    require(manifest.is_file() and not manifest.is_symlink(), 'Explicit prepared manifest required')
    if value.get('kind') == 'prepared_input_bundle_v1':
        entries = value['files']; source_manifest = value['source_manifest']
    else:
        require(value.get('kind') == 'suite_prepared_sources_v1', 'Unsupported consumer prepared input')
        entries = [{'path': s['path'], 'sha256': s['sha256']} for s in value['sources']]
        source_manifest = manifest.name
    names = [item['path'] for item in entries]
    require(names and len(names) == len(set(names)) and manifest.name not in names, 'Empty/duplicate inventory')
    output.mkdir(parents=True, exist_ok=False)
    for item in entries:
        relative = safe(item['path']); origin = root / relative
        require(origin.is_file() and not origin.is_symlink() and inside(origin.resolve(), root), 'Prepared member escapes input root')
        cursor = root
        for component in relative.parts:
            cursor = cursor / component
            require(not cursor.is_symlink(), 'Prepared symlink member or ancestor refused')
        require(file_hash(origin) == item['sha256'], 'Prepared input pin changed')
        require('bytes' not in item or type(item['bytes']) is int and origin.stat().st_size == item['bytes'], 'Prepared input byte count changed')
        destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
    shutil.copyfile(manifest, output / manifest.name)
    return output / manifest.name, output / safe(source_manifest)


def validate_layout(layout, plan, digest):
    require(layout['status'] == 'CONTRACT_VALIDATED_ONLY' and layout['live_status'] == 'LIVE_BINDING_NOT_VALIDATED', 'Metadata binding overclaims live execution')
    require(layout['artifact_presence'] == 'ARTIFACT_PRESENCE_NOT_CHECKED' and layout['authorization'] == 'NONE', 'Consumer unexpectedly resolved/authorized artifacts')
    for key in ('GPU_queries', 'model_loads', 'inference_POSTs', 'tokenizer_calls', 'ledger_changes'):
        require(type(layout[key]) is int and layout[key] == 0, 'Metadata binding performed work: ' + key)
    require(layout['plan_identity'] == plan['plan_identity'], 'Binding and compiler plan identity differ')
    units = layout['execution_units']; groups = layout['serving_groups']; probe = layout['probe_inputs']; counts = plan['semantic']['counts']
    require(units and len({u['node_id'] for u in units}) == len(units), 'Empty/duplicate execution layout')
    require(len({g['raw_root'] for g in groups}) == len(groups) and len({g['job_id'] for g in groups}) == len(groups), 'Serving evidence or ledger job collision')
    require(sum(len(g['requests']) for g in groups) == counts['inference_POSTs'], 'Grouped request count changed')
    collectors = {n['node_id'] for n in plan['semantic']['graph']['nodes'] if n['operation_kind'] in
                  ('serve_generate_until', 'serve_choice_loglikelihood', 'collect_token_diagnostic')}
    provided = [name for unit in units for name in unit['provides']]
    require(provided and len(provided) == len(set(provided)) and set(provided) == collectors, 'Collector exact coverage differs')
    models = plan['semantic']['suite']['models']; rows = probe['rows']
    require(rows and len(rows) * len(models) == counts['forward_calls'], 'Diagnostic forward count changed')
    require(len(models) * sum(len(r['targets']) for r in rows) == counts['main_target_observations'] and
            len(models) * sum(len(r['full_vocab_target_positions']) for r in rows) == counts['main_full_observations'], 'Diagnostic coordinate count changed')
    require(probe['counts']['attention_rows_per_arm'] == 0, 'Token-only consumer acquired attention')
    qualified = layout['qualified_samples']; require(len(qualified) == len(rows), 'Qualified sample coverage differs')
    for sample in qualified:
        namespace = {name: sample[name] for name in ('task_id', 'task_version', 'dataset_id', 'split', 'sample_id')}
        require(sample['qualified_sample_id'] == digest(namespace), 'Qualified namespace identity changed')
    return {'documents': len(rows), 'serving_units': len(groups),
            'token_units': sum(u['operation_kind'] == 'token_group' for u in units),
            'execution_units': len(units), 'inference_POSTs_planned': counts['inference_POSTs'],
            'max_output_tokens_planned': counts['output_token_upper'],
            'forward_calls_planned': counts['forward_calls'],
            'target_positions_per_arm': probe['counts']['target_positions_per_arm'],
            'full_positions_per_arm': probe['counts']['full_vocab_positions_per_arm']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'suite', 'local', 'output', 'expected-origin'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--expected-version', default='0.1.1.dev16')
    parser.add_argument('--expected-source-digest')
    parser.add_argument('--deny-root', action='append', required=True, type=Path)
    parser.add_argument('--synthetic-tests-root', type=Path)
    args = parser.parse_args(argv)
    require(sys.flags.isolated, 'Run this explicit consumer with python -I')
    os.environ.pop('PYTHONPATH', None); os.environ.pop('PYTHONHOME', None)
    denied = [root.absolute().resolve() for root in args.deny_root]
    output = args.output.absolute()
    require(not output.exists() and not any(inside(output.resolve(), root) for root in denied), 'Consumer output must be new and outside checkout')
    output.mkdir(parents=True, exist_ok=False)
    prepared, sources = copy_prepared(args.prepared, output / 'prepared')
    suite = output / 'SUITE.json'; local = output / 'LOCAL_METADATA_ONLY.json'
    shutil.copyfile(args.suite, suite); shutil.copyfile(args.local, local)
    require(read(local) == METADATA_LOCAL, 'Only metadata local policy is accepted')
    environment = dict(os.environ); environment.pop('PYTHONPATH', None); environment.pop('PYTHONHOME', None)
    environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    child_receipts = []

    def invoke(arguments, console=False):
        prefix = [str(Path(sys.executable).parent / 'inference-kit')] if console else [sys.executable, '-I', '-B', '-m', 'inference_kit']
        child = subprocess.run(prefix + arguments, cwd=output, env=environment, capture_output=True, timeout=180)
        require(child.returncode == 0, child.stderr.decode('ascii', errors='backslashreplace'))
        result = loads(child.stdout.decode('ascii'))
        child_receipts.append({'entry': 'installed_console' if console else 'module', 'arguments': arguments,
                               'returncode': child.returncode, 'stdout_bytes': len(child.stdout), 'audit_hook_inherited': False})
        return result

    console_path = Path(sys.executable).parent / 'inference-kit'
    require(console_path.is_file(), 'Ordinary installed console entry is missing')
    cli_inputs = invoke(['bench', 'inputs', 'validate', '--prepared', str(prepared), '--output', str(output/'module-inputs')])
    cli_plan = invoke(['bench', 'suite', 'plan', '--suite', str(suite), '--sources', str(sources), '--local', str(local), '--output', str(output/'module-plan')])
    module_plan = output/'module-plan/FROZEN_PLAN.json'
    cli_validate = invoke(['bench', 'suite', 'validate', '--plan', str(module_plan), '--sources', str(sources), '--output', str(output/'module-validate')])
    cli_describe = invoke(['bench', 'suite', 'describe', '--plan', str(module_plan), '--output', str(output/'module-describe')])
    cli_bind = invoke(['bench', 'suite', 'bind', '--plan', str(module_plan), '--local', str(local), '--metadata-only', '--output', str(output/'module-bind')])
    console_bind = invoke(['bench', 'suite', 'bind', '--plan', str(module_plan), '--local', str(local), '--metadata-only', '--output', str(output/'console-bind')], console=True)
    synthetic = {'status': 'NOT_RUN', 'scope': 'SYNTHETIC_CONTRACT_ONLY; not new live raw replay'}
    if args.synthetic_tests_root is not None:
        test_root = args.synthetic_tests_root.absolute().resolve()
        require(test_root.is_dir() and not any(inside(test_root, root) for root in denied), 'Closed synthetic tests must be outside checkout')
        code = ('import json,sys,unittest;sys.path.insert(0,' + repr(str(test_root)) + ');'
                'suite=unittest.defaultTestLoader.loadTestsFromName("bench05e4b.test_results");'
                'result=unittest.TextTestRunner(verbosity=2).run(suite);'
                'print(json.dumps({"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped)}));'
                'raise SystemExit(0 if result.testsRun>0 and result.wasSuccessful() else 1)')
        child = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=output, env=environment, capture_output=True, timeout=180)
        (output/'SYNTHETIC_TESTS.log').write_bytes(child.stderr)
        require(child.returncode == 0, child.stderr.decode('ascii', errors='backslashreplace'))
        synthetic = loads(child.stdout.decode('ascii')) | {'status': 'PASS', 'scope': 'SYNTHETIC_CONTRACT_ONLY; not new live raw replay', 'audit_hook_inherited': False}
    guards = {'checkout': 0, 'SDK': 0, 'network': 0, 'launch': 0}; blocked = []

    def guard(event, values):
        if event in ('open', 'os.scandir', 'os.listdir') and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(values[0])))
            if any(inside(path, root) for root in denied):
                guards['checkout'] += 1; blocked.append({'event': event, 'path': str(path)}); raise PermissionError('Checkout access blocked')
        if event == 'import' and values and str(values[0]).split('.')[0] in SDK:
            guards['SDK'] += 1; raise ImportError('ML SDK import blocked')
        if event in ('socket.__new__', 'socket.connect', 'socket.bind', 'socket.getaddrinfo'):
            guards['network'] += 1; raise PermissionError('Network/GPU query bridge blocked')
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
            guards['launch'] += 1; raise PermissionError('Runtime launch blocked')

    os.chdir(output); sys.addaudithook(guard)
    import inference_kit
    from inference_kit import api
    from inference_kit.core.io import digest
    from inference_kit.core.parity import equal
    from inference_kit.analysis.tokens import source_identity
    origin = Path(inference_kit.__file__).absolute()
    require(inside(origin, args.expected_origin.absolute()) and 'site-packages' in origin.parts, 'Package did not come from explicit installed site-packages')
    require(inference_kit.__version__ == args.expected_version, 'Installed package version differs')
    actual_source = source_identity()
    require(args.expected_source_digest is None or actual_source == args.expected_source_digest, 'Installed package source digest differs')
    require(not SDK.intersection(name.split('.')[0] for name in sys.modules), 'Consumer already imported an ML SDK')
    api_inputs = api.inputs_validate(prepared, output/'api-inputs')
    api_plan = api.suite_plan(suite, sources, local, output/'api-plan')
    api_plan_path = output/'api-plan/FROZEN_PLAN.json'
    api_validate = api.suite_validate(api_plan_path, sources, output/'api-validate')
    api_describe = api.suite_describe(api_plan_path, output/'api-describe')
    api_bind = api.suite_bind(api_plan_path, local, output/'api-bind', metadata_only=True)
    equal(api_inputs, cli_inputs, path='inputs_validate')
    equal(api_plan['plan_identity'], cli_plan['plan_identity'], path='plan_identity')
    equal(api_plan['semantic']['counts'], cli_plan['counts'], path='plan_counts')
    equal(api_validate, cli_validate, path='suite_validate')
    equal(api_describe, cli_describe, path='suite_describe')
    equal(api_bind, cli_bind, path='binding_layout')
    equal(api_bind, console_bind, path='console_binding_layout')
    require(api_plan['status'] == 'VALIDATED_NOT_EXECUTED' and api_plan['execution_count'] == api_plan['acquisition_count'] == 0,
            'CPU plan is not a validated unexecuted plan')
    counts = validate_layout(api_bind, api_plan, digest)
    require(not any(guards.values()), 'CPU path attempted a forbidden action: ' + repr(guards))
    receipt = {'schema_version': 1, 'status': 'PASS', 'package_version': inference_kit.__version__,
        'package_import_origin': str(origin), 'package_source_digest': actual_source, 'interpreter': sys.executable,
        'isolated': bool(sys.flags.isolated), 'PYTHONPATH_present': 'PYTHONPATH' in os.environ, 'PYTHONHOME_present': 'PYTHONHOME' in os.environ,
        'prepared_status': api_inputs['status'], 'plan_identity': api_plan['plan_identity'], 'counts': counts,
        'API_module_console_semantic_parity': 'PASS', 'artifact_presence': 'ARTIFACT_PRESENCE_NOT_CHECKED',
        'runtime_preflight': 'CHECK_NOT_RUN', 'live_binding': 'LIVE_BINDING_NOT_VALIDATED',
        'new_raw_replay': 'NOT_RUN_NO_NEW_LIVE_RAW', 'synthetic_tests': synthetic,
        'GPU_queries': 0, 'model_loads': 0, 'inference_POSTs': 0, 'tokenizer_loads': 0, 'ledger_writes': 0,
        'guards': guards, 'blocked_attempts': blocked, 'subprocess_receipts': child_receipts,
        'audit_scope': 'In-process API guard only; separate module/console/tests checked before hook. No child hook inheritance or OS sandbox claim.'}
    (output/'CONSUMER_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
