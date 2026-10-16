"""Ordinary wheel consumer: deny both checkouts and engine SDK imports.

The explicit recorded bundle is copied before the read gate. The new CPU plan
uses rebound historical metadata and deliberately absent artifacts: it must
report BLOCKED, never pretend capability/model execution was verified.
"""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--forbidden-root', action='append', required=True)
    args = parser.parse_args()
    root = args.workspace.absolute()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.fixture, root / 'recorded')
    forbidden = [os.path.abspath(p) for p in args.forbidden_root]
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        os.environ.pop(name, None)
    os.chdir(root)
    sys.path[:] = [p for p in sys.path if p and not any(os.path.abspath(p) == f or os.path.abspath(p).startswith(f + os.sep) for f in forbidden)]

    def audit(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(values[0]))
            if any(path == f or path.startswith(f + os.sep) for f in forbidden):
                raise AssertionError('Consumer read a forbidden checkout: ' + path)
        if event == 'import' and values and str(values[0]).split('.')[0] in ('torch', 'vllm', 'transformers', 'numpy', 'compressed_tensors'):
            raise AssertionError('CPU consumer imported a numerical/engine SDK')

    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import bench_plan, bench_replay
    from inference_kit.cli import main as cli
    from inference_kit.core.io import read, write, sha
    from inference_kit.metrics.token import distribution_metrics

    origin = Path(inference_kit.__file__).absolute()
    assert 'site-packages' in origin.parts
    assert not any(str(origin).startswith(f + os.sep) for f in forbidden)
    commands = []

    def command(argv, expected=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = cli(argv)
            except SystemExit as exc:
                code = exc.code
        assert code == expected, (argv, code, stderr.getvalue())
        commands.append({'argv': ['inference-kit', *argv], 'returncode': code})

    command(['bench', '--help'])
    command(['bench', 'replay', '--run', 'recorded', '--output', 'cli-replay'])
    result = bench_replay('recorded', 'api-replay')
    assert result['counts']['paired_token_positions'] == 23
    assert result['counts']['paired_attention_rows'] == 16
    assert result['counts']['paired_full_vocab_positions'] == 0
    assert result['metrics']['mean_selected_full_kl_nat'] is None
    assert read(root / 'cli-replay/summary.json') == result
    assert all((root / 'api-replay' / name).is_file() for name in ('token.csv', 'attention.csv', 'comparisons.json', 'REPORT_KO.md'))
    command(['bench', 'replay', '--run', 'recorded', '--output', 'cli-replay'], 2)
    assert distribution_metrics([1, 2, 3], [1, 2, 3])['js_nat'] == 0

    # A new non-GPU plan, with explicit runtime path and local root rebinding.
    frozen = read(root / 'recorded/PLAN_FROZEN.json')
    prepared = copy.deepcopy(frozen['samples'])
    authored = [{k: row[k] for k in ('sample_id', 'source_id', 'prompt', 'reference_text', 'language')} for row in prepared['rows']]
    write(root / 'samples.json', authored)
    prepared['authored_sha256'] = sha(root / 'samples.json')
    write(root / 'prepared.json', prepared)
    suite = copy.deepcopy(frozen['suite']); suite['samples'] = 'samples.json'
    write(root / 'suite.json', suite); write(root / 'pair.json', frozen['pair'])
    write(root / 'runtime.json', frozen['runtime'])
    refs = {}
    for arm in frozen['arms']:
        reference = copy.deepcopy(arm['artifact'])
        reference['path'] = str(root / 'absent-artifacts' / arm['arm_id'])
        reference['tokenizer_path'] = str(root / 'absent-tokenizer')
        refs[arm['receipt_arm']] = reference
    write(root / 'receipts.json', {'arms': refs})
    records = frozen['campaign']['chain']
    rebound = {}

    def clone_ledger(name):
        if name in rebound:
            return rebound[name]
        value = copy.deepcopy(records[name]['record'])
        if value.get('inherited_from'):
            previous = clone_ledger(value['inherited_from']['path'])
            value['inherited_from'].update(path=str(previous), sha256=sha(previous))
        destination = root / 'rebound-history' / (str(len(rebound)) + '.json')
        write(destination, value); rebound[name] = destination
        return destination

    prior = clone_ledger(frozen['campaign']['path'])
    write(root / 'local.json', {
        'artifact_receipt': str(root / 'receipts.json'), 'prepared_samples': str(root / 'prepared.json'),
        'runtime_metadata': str(root / 'runtime.json'), 'runtime_python': sys.executable,
        'server_python': sys.executable, 'package_site': str(origin.parent.parent),
        'prior_ledger': str(prior), 'campaign_root': str(root), 'gpu': 0,
        'gpu_uuid': 'NOT_QUERIED_CPU_PLAN', 'port': 18186,
    })
    planned = bench_plan(root / 'suite.json', root / 'pair.json', root / 'local.json', root / 'plan')
    assert planned['status'] == 'BLOCKED_LOCAL_RESOURCE_OR_ARTIFACT'
    assert (planned['gpu_actions'], planned['model_loads'], planned['network_actions']) == (0, 0, 0)
    command(['bench', 'plan', '--suite', 'suite.json', '--models', 'pair.json', '--local', 'local.json', '--output', 'cli-plan'])
    imports = [n for n in sys.modules if n.split('.')[0] in ('torch', 'vllm', 'transformers', 'numpy', 'compressed_tensors')]
    assert not imports
    receipt = {
        'status': 'BENCH05A_ORDINARY_WHEEL_CONSUMER_PASS', 'package_version': inference_kit.__version__,
        'import_origin': str(origin), 'editable': False, 'commands': commands,
        'original_checkout_reads': 'DENIED_BY_AUDIT_HOOK', 'forbidden_roots': forbidden,
        'engine_numerical_sdk_imports': imports, 'plan_status': planned['status'],
        'historical_ledger_rebound_for_cpu_plan_only': True,
        'real_recorded_raw_replay_counts': result['counts'],
        'missing_distributions': 'NOT_RUN; full KL/JS not fabricated', 'gpu_jobs': 0, 'downloads': 0,
    }
    write(root / 'CONSUMER_RECEIPT.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
