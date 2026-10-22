"""Explicit, CPU-only MINI reproduction. Read this file before invoking it.

This verifies stored prepared metadata, not tokenizer/model execution or the
blocked live suite. Only the two named consumer invocations and the closed
synthetic regression runner are executed. No archive code discovery occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deny-root', type=Path, action='append', required=True)
    args = parser.parse_args()
    require(sys.flags.isolated, 'Use the explicitly installed wheel with python -I')
    bundle = args.bundle.absolute().resolve()
    output = args.output.absolute()
    require(not output.exists(), 'Refusing existing output')
    manifest = json.loads((bundle / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    require(manifest['kind'] == 'e4b_review_mini_files_v1' and manifest['files'], 'Empty/unsupported MINI inventory')
    names = set()
    for item in manifest['files']:
        name = item['path']; relative = PurePosixPath(name)
        require(name and not relative.is_absolute() and '..' not in relative.parts and
                '\\' not in name and ':' not in name and relative.as_posix() == name and name not in names,
                'Unsafe/duplicate MINI path')
        names.add(name)
        path = bundle / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(bundle), 'Missing/unsafe MINI file')
        current = bundle
        for component in relative.parts:
            current = current / component
            require(not current.is_symlink(), 'Symlink MINI ancestor refused')
        require(path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
                'MINI bytes changed: ' + name)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    require(actual == names | {'FILE_MANIFEST.json'}, 'Unexpected/unpinned MINI files')
    require(not any(bundle.rglob('src/inference_kit')), 'Checkout source fallback is not allowed')
    import inference_kit
    from inference_kit.analysis.tokens import source_identity
    origin = Path(inference_kit.__file__).absolute()
    require('site-packages' in origin.parts and inference_kit.__version__ == manifest['package_version'], 'Expected ordinary wheel installation')
    require(source_identity() == manifest['source_digest'], 'Installed package source differs')
    output.mkdir(parents=True)
    environment = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        environment.pop(key, None)
    environment.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    invocations = []

    def run(label, arguments):
        child = subprocess.run([sys.executable, '-I', '-B'] + arguments, cwd=bundle,
                               env=environment, capture_output=True, timeout=300)
        (output / (label + '.stdout')).write_bytes(child.stdout)
        (output / (label + '.stderr')).write_bytes(child.stderr)
        invocations.append({'label': label, 'returncode': child.returncode, 'audit_hook_inherited': False})
        require(child.returncode == 0, label + ' failed; see explicit output logs')

    run('closed-regressions', [str(bundle/'tools/test_bench05e4b.py'), '--tests-root', str(bundle/'tests'),
      '--installed', '--expected-origin', str(origin.parent.parent),
      '--expected-source-digest', manifest['source_digest'], '--output', str(output/'TARGETED.json')])
    for label in ('main9', 'subset3'):
        run(label, [str(bundle/'tools/bench05e4b_consumer_check.py'),
          '--prepared', str(bundle/'inputs'/label/'SOURCES.json'),
          '--suite', str(bundle/'plans'/label/'SUITE.json'),
          '--local', str(bundle/'LOCAL_METADATA_ONLY.json'), '--output', str(output/label),
          '--expected-origin', str(origin.parent.parent), '--expected-version', manifest['package_version'],
          '--expected-source-digest', manifest['source_digest'],
          *[argument for root in args.deny_root for argument in ('--deny-root', str(root.absolute()))]])
        expected = json.loads((bundle/'plans'/label/'EXPECTED_SEMANTICS.json').read_text(encoding='utf-8'))
        consumer = json.loads((output/label/'CONSUMER_RECEIPT.json').read_text(encoding='utf-8'))
        require(consumer['plan_identity'] == expected['plan_identity'], 'Portable plan differs from local frozen plan')
        mapping = {'documents': 'documents', 'execution_units': 'execution_units',
          'inference_POSTs_planned': 'inference_POSTs', 'max_output_tokens_planned': 'output_token_upper',
          'forward_calls_planned': 'forward_calls'}
        require(all(type(consumer['counts'][key]) is int and consumer['counts'][key] == expected['counts'][source]
                    for key, source in mapping.items()), 'Portable plan counts differ from frozen expectations')
        require(consumer['counts']['target_positions_per_arm'] * expected['counts']['arms'] == expected['counts']['main_target_observations'] and
                consumer['counts']['full_positions_per_arm'] * expected['counts']['arms'] == expected['counts']['main_full_observations'],
                'Portable selected token coverage differs from local frozen plan')
    receipt = {'status': 'PASS', 'package_version': inference_kit.__version__, 'package_origin': str(origin),
      'source_digest': source_identity(), 'invocations': invocations,
      'scope': 'Stored actual MAIN9/subset3 prepared metadata and CPU binding; closed synthetic executor/results tests',
      'full_prepared_bundle_validation': 'NOT_RUN; explicit inventory file slice; SOURCES authority verified',
      'artifact_presence': 'ARTIFACT_PRESENCE_NOT_CHECKED',
      'new_live_raw_replay': 'NOT_RUN_NO_NEW_LIVE_RAW', 'actual_pause_resume': 'NOT_RUN_BLOCKED_LOOPBACK',
      'retokenization': 'NOT_RUN; local tokenizer/runtime not included',
      'GPU_actions': 0, 'model_loads': 0, 'inference_POSTs': 0, 'real_campaign_debit': 0,
      'guard_scope': 'Consumer API guard is in-process only; child subprocesses separately checked, not OS sandbox'}
    (output/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'output': str(output), 'scope': receipt['scope']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
