"""Installed D1 CPU consumer over explicitly copied manifested numeric raw.

Invoke a base CPU venv with -I -B. The in-process read/import/network audit hook
is not inherited by the separately checked console child and is not an OS sandbox.
No Python or pickle from the evidence is executed.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


SEMANTIC_FILES = (
    'summary.json', 'COVERAGE.json', 'METRIC_CONTRACT.json', 'OBSERVER_CONTROL_RECEIPT.json',
    'SCALAR_FULL_CONSISTENCY_RECEIPT.json', 'TOKEN_COMPARISON_STATUS.json', 'task-layer-aggregates.json',
    'structure-comparisons.jsonl', 'output-projection-comparisons.jsonl', 'reconstructed-native-AV.jsonl',
    'evidence-content-mass.jsonl', 'observer-controls.jsonl', 'scalar-full-consistency.jsonl',
    'token-metrics/summary.json', 'token-metrics/aggregates.json', 'token-metrics/per-document.json',
    'token-metrics/token-comparisons.jsonl', 'token-metrics/COMPATIBILITY.json', 'token-metrics/COVERAGE.json')
RELOCATION_FIELDS = (
    'token-metrics/summary.json:view_identity (derived manifest fingerprint includes source-path metadata)',
    'token-metrics/summary.json:acquisition_source_panels[*].source_bundle_path',
    'token-metrics/summary.json:acquisition_source_panels[*].input_path_as_metadata')


def strict_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Invalid JSON constant: ' + value)))


def strict_read(path): return strict_loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def copy_raw(source, destination):
    source = Path(source).absolute()
    if not source.is_dir() or source.is_symlink(): raise ValueError('Explicit non-symlink D1 raw directory required')
    manifest_path = source / 'RAW_MANIFEST.json'; manifest = strict_read(manifest_path)
    if manifest.get('scope') != 'structure_probe_raw_v1' or not isinstance(manifest.get('files'), dict):
        raise ValueError('D1 structure_probe_raw_v1 manifest required')
    files = list(manifest['files'].items()) + [('RAW_MANIFEST.json', {'bytes': manifest_path.stat().st_size, 'sha256': file_sha(manifest_path)})]
    if len(files) > 10000: raise ValueError('Consumer raw file count limit')
    destination.mkdir(parents=True, exist_ok=False); seen = set(); total = 0
    for name, entry in files:
        path = Path(name)
        if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or path.as_posix() != name:
            raise ValueError('Unsafe raw member')
        if name.casefold() in seen: raise ValueError('Duplicate raw member')
        seen.add(name.casefold()); member = source / path
        if not member.is_file() or member.is_symlink() or not member.resolve().is_relative_to(source.resolve()):
            raise ValueError('Missing/unsafe raw member')
        size = entry['bytes']
        if type(size) is not int or not 0 <= size <= 64 * 1024 ** 2: raise ValueError('Consumer member size limit')
        if member.stat().st_size != size or file_sha(member) != entry['sha256']: raise ValueError('Original raw identity mismatch')
        total += size
        if total > 512 * 1024 ** 2: raise ValueError('Consumer raw total size limit')
        target = destination / path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(member, target)
        if target.stat().st_size != size or file_sha(target) != entry['sha256']: raise ValueError('Copied raw identity mismatch')
    return {'manifest_sha256': file_sha(destination / 'RAW_MANIFEST.json'), 'members': len(seen), 'bytes': total,
            'scope': 'All explicitly manifested D1 raw members; no model/cache/environment directory copied'}


def semantic_snapshot(root):
    result = {}; hashes = {}
    for name in SEMANTIC_FILES:
        path = Path(root) / name
        if not path.is_file(): continue
        text = path.read_text(encoding='utf-8')
        value = [strict_loads(line) for line in text.splitlines() if line] if name.endswith('.jsonl') else strict_loads(text)
        if name == 'token-metrics/summary.json':
            # Only relocation-derived identifiers/paths vary. Numerical values,
            # statuses, IDs, null/types and order retain the exact parity contract.
            value['view_identity'] = '<PATH_DEPENDENT_VIEW_FINGERPRINT>'
            for panel in value['acquisition_source_panels']:
                for field in ('source_bundle_path', 'input_path_as_metadata'):
                    if field in panel: panel[field] = '<RELOCATED_RAW_ROOT>'
        result[name] = value; hashes[name] = file_sha(path)
    if 'summary.json' not in result: raise ValueError('Expected explicit structure replay summary')
    return result, hashes


def tree_inventory(root):
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', '--workspace', dest='output', required=True, type=Path)
    parser.add_argument('--forbidden-root', action='append', required=True)
    parser.add_argument('--expected-version', default='0.1.1.dev10')
    parser.add_argument('--expected-analysis', type=Path,
                        help='Optional prior replay: only the enumerated raw-derived metric files are compared')
    args = parser.parse_args()
    if not sys.flags.isolated: raise ValueError('Invoke installed CPU interpreter with -I')
    root = args.output.absolute(); forbidden = [os.path.abspath(p) for p in args.forbidden_root]
    if any(str(root) == p or str(root).startswith(p + os.sep) for p in forbidden):
        raise ValueError('Consumer output must be outside original checkouts')
    root.mkdir(parents=True, exist_ok=False)
    recorded = root / 'recorded'; copied = copy_raw(args.run, recorded)
    expected, expected_hashes = semantic_snapshot(args.expected_analysis) if args.expected_analysis else (None, None)
    for name in ('PYTHONPATH', 'PYTHONHOME'): os.environ.pop(name, None)
    os.chdir(root)
    sys.path[:] = [p for p in sys.path if p and not any(os.path.abspath(p) == f or os.path.abspath(p).startswith(f + os.sep) for f in forbidden)]
    sdk = ('torch', 'vllm', 'transformers', 'tensorflow', 'numpy', 'aiohttp', 'compressed_tensors', 'safetensors', 'accelerate')
    allowed_child = {'console': False}; audit_counts = {'checkout': 0, 'ml_sdk': 0, 'network': 0, 'unexpected_subprocess': 0}
    def audit(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(values[0]))
            if any(path == f or path.startswith(f + os.sep) for f in forbidden):
                audit_counts['checkout'] += 1; raise AssertionError('Original checkout read forbidden in CPU consumer')
        if event == 'import' and values and str(values[0]).split('.')[0] in sdk:
            audit_counts['ml_sdk'] += 1; raise AssertionError('ML SDK import forbidden in CPU consumer')
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname'):
            audit_counts['network'] += 1; raise AssertionError('Network forbidden in CPU consumer')
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn') and not allowed_child['console']:
            audit_counts['unexpected_subprocess'] += 1; raise AssertionError('Unexpected consumer subprocess forbidden')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import structure_replay
    from inference_kit.cli import main as cli
    from inference_kit.core.io import read, write
    from inference_kit.core.parity import equal
    origin = Path(inference_kit.__file__).absolute()
    assert 'site-packages' in origin.parts and origin.is_relative_to(Path(sys.prefix)), str(origin)
    assert inference_kit.__version__ == args.expected_version
    calls = []
    def json_shape(value): return strict_loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    def command(argv, expected_code=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try: code = cli(argv)
            except SystemExit as exc: code = exc.code
        assert code == expected_code, (code, stderr.getvalue())
        calls.append({'args': argv, 'code': code})
        if expected_code: strict_loads(stderr.getvalue())
        return stdout.getvalue()
    summary = structure_replay(recorded, root / 'api-replay')
    plan = read(recorded / 'PLAN_FROZEN.json'); selection = plan['structure_selection']
    token_targets = sum(len(row['targets']) for row in plan['samples']['rows'])
    token_full = sum(len(row['full_vocab_target_positions']) for row in plan['samples']['rows'])
    row_count = sum(len(selection['layers']) * len(selection['heads']) * len(row['queries']) for row in selection['rows'].values())
    projection_count = sum(len(selection['layers']) * len(row['queries']) for row in selection['rows'].values())
    assert summary['coverage']['per_kind']['attention_row']['requested_pairs'] == row_count
    assert summary['coverage']['per_kind']['attention_output']['requested_pairs'] == row_count
    assert summary['coverage']['per_kind']['reconstructed_AV']['requested_pairs'] == row_count
    assert summary['coverage']['per_kind']['output_projection']['requested_pairs'] == projection_count
    assert summary['coverage']['observer_comparisons'] == 2 * 2 * (token_targets + token_full)
    assert summary['coverage']['same_pass_scalar_full'] == 2 * 3 * token_full
    if summary['OBSERVER_INTEGRITY'] == 'VERIFIED':
        assert summary['ON_token_coverage']['measured'] == token_targets
        assert summary['ON_token_coverage']['free_generation_pairs'] == 0
    if expected is not None:
        actual, _ = semantic_snapshot(root / 'api-replay'); equal(actual, expected)
    command(['bench', 'structure', '--help'])
    text = command(['bench', 'structure', 'replay', '--run', 'recorded', '--output', 'cli-replay'])
    equal(strict_loads(text), json_shape(summary))
    equal(tree_inventory(root / 'cli-replay'), tree_inventory(root / 'api-replay'))
    command(['bench', 'structure', 'replay', '--run', 'recorded', '--output', 'cli-replay'], 2)
    argv = [str(Path(sys.executable).parent / 'inference-kit'), 'bench', 'structure', 'replay',
            '--run', 'recorded', '--output', 'console-replay']
    allowed_child['console'] = True
    try: child = subprocess.run(argv, capture_output=True, encoding='utf-8', timeout=240)
    finally: allowed_child['console'] = False
    assert child.returncode == 0, (child.returncode, child.stderr)
    equal(strict_loads(child.stdout), json_shape(summary))
    equal(tree_inventory(root / 'console-replay'), tree_inventory(root / 'api-replay'))
    assert not any(name.split('.')[0] in sdk for name in sys.modules)
    receipt = {'status': 'D1_INSTALLED_CPU_RAW_REPLAY_CONSUMER_PASS', 'package_version': inference_kit.__version__,
        'package_import_origin': str(origin), 'interpreter': sys.executable, 'isolated_mode': True,
        'raw_copy_before_guard': copied, 'raw_manifest_sha256': copied['manifest_sha256'],
        'documents': summary['coverage']['documents'], 'layers': selection['layers'], 'heads': selection['heads'],
        'requested_attention_pairs': row_count, 'requested_native_AV_pairs': row_count,
        'requested_output_projection_pairs': projection_count, 'ON_target_pairs': token_targets, 'ON_full_pairs': token_full,
        'OBSERVER_INTEGRITY': summary['OBSERVER_INTEGRITY'], 'commands': calls, 'console_command': argv,
        'same_location_parity': 'API/CLI/console complete derived-file inventory bytes identical after one explicit raw relocation',
        'prior_location_semantic_parity': 'PASS' if expected is not None else 'NOT_REQUESTED',
        'prior_semantic_file_hashes': expected_hashes,
        'relocation_normalization_exact_fields': list(RELOCATION_FIELDS) if expected is not None else [],
        'float_tolerance': {'absolute': 1e-10, 'relative': 1e-9}, 'exact': 'IDs/counts/status/null/types/order; tuple/JSON-array representation only normalized',
        'audit_counts': audit_counts, 'checkout': 'DENIED_IN_PROCESS_AFTER_EXPLICIT_COPY', 'ML_SDK_imports': [],
        'network': 'DENIED_IN_PROCESS', 'subprocess_scope': 'One explicit console child; audit hook not inherited; no OS sandbox claim',
        'GPU_actions': 0, 'model_loads': 0, 'tokenizer_loads': 0, 'downloads': 0, 'ledger_changes': 0,
        'archive_code_executed': False, 'unit_suite_scope': 'Consumer raw replay only; does not claim standalone execution of all historical unit tests'}
    write(root / 'CONSUMER_RECEIPT.json', receipt)
    print(json.dumps({'status': receipt['status'], 'version': inference_kit.__version__, 'package_import_origin': str(origin),
        'documents': receipt['documents'], 'attention_pairs': row_count, 'projection_pairs': projection_count,
        'ON_token_pairs': token_targets, 'ON_full_pairs': token_full, 'OBSERVER_INTEGRITY': receipt['OBSERVER_INTEGRITY']}, ensure_ascii=True))


if __name__ == '__main__': main()
