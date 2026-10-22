"""Record release package bytes without importing a checkout library or ML SDK.

The GitHub job records its exact head separately from archive metadata.  Package
Python/resources must equal the clean checkout and the sdist rebuild byte for
byte; wheel container timestamps/RECORD are not claimed to be reproducible.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(',', ':')).encode())


def package(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel members')
        return {name: archive.read(name) for name in names
                if name.startswith('inference_kit/') and not name.endswith('/')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--rebuilt-wheel', type=Path, required=True)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing receipt overwrite')
    repo = args.repo.resolve()
    direct, rebuilt = package(args.wheel), package(args.rebuilt_wheel)
    if not direct or direct != rebuilt:
        raise ValueError('Wheel/sdist package or resource bytes differ')
    source = {str(p.relative_to(repo/'src')).replace(os.sep, '/'): p.read_bytes()
              for p in sorted((repo/'src/inference_kit').rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts
              and p.suffix not in {'.pyc', '.pyo'}}
    if direct != source:
        raise ValueError('Clean checkout package/resource closure differs from wheel')
    members = {name: sha(value) for name, value in direct.items()}
    python = {name.removeprefix('inference_kit/'): value
              for name, value in members.items() if name.endswith('.py')}
    resources = {name.removeprefix('inference_kit/'): value
                 for name, value in members.items() if not name.endswith('.py')}
    identity = json.loads((args.kit/'RELEASE_IDENTITY.json').read_text(encoding='utf-8'))
    if digest(python) != identity['python_source_digest']:
        raise ValueError('Frozen release Python identity differs')
    if 'resources_identity' in identity and digest(resources) != identity['resources_identity']:
        raise ValueError('Frozen release resource identity differs')
    version = (args.kit/'VERSION').read_text(encoding='utf-8').strip()
    if version != identity['version']:
        raise ValueError('Kit version differs from frozen release identity')
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    result = {'schema_version': 1, 'status': 'RELEASE_PACKAGE_BYTES_MATCH',
              'head_sha': head, 'github_sha': os.environ.get('GITHUB_SHA'),
              'github_workflow_ref': os.environ.get('GITHUB_WORKFLOW_REF'),
              'github_workflow_sha': os.environ.get('GITHUB_WORKFLOW_SHA'),
              'version': version, 'python_source_digest': digest(python),
              'resources_identity': digest(resources), 'python_members': len(python),
              'resource_members': len(resources), 'package_members': members,
              'direct_wheel': {'name': args.wheel.name,
                               'bytes': args.wheel.stat().st_size,
                               'sha256': sha(args.wheel.read_bytes())},
              'sdist_rebuilt_wheel': {'name': args.rebuilt_wheel.name,
                                     'bytes': args.rebuilt_wheel.stat().st_size,
                                     'sha256': sha(args.rebuilt_wheel.read_bytes())},
              'archive_container_identity': 'SEPARATE_FROM_PACKAGE_BYTES',
              'model_or_tokenizer_execution': 0}
    if os.environ.get('GITHUB_ACTIONS') == 'true' and head != os.environ.get('GITHUB_SHA'):
        raise ValueError('CI head differs from dispatched GitHub SHA')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2,
                                      ensure_ascii=True, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'head_sha', 'version',
                                           'python_source_digest', 'resources_identity')}))


if __name__ == '__main__':
    main()
