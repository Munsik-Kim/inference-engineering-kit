"""Explicit CPU acceptance for stored prepared Suite plans; no tokenization/live.

Run only this named, reviewed script. API mode uses installed public functions;
module/console modes launch the actual installed entry points in fresh children.
No checkout path is added to sys.path. The import guard is in this process only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CASES = (('smol-task', 'smol'), ('smol-token', 'smol'), ('qwen-task', 'qwen'))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('api', 'module', 'console'))
    args = parser.parse_args()
    kit, output = args.kit.absolute(), args.output.absolute()
    if output.exists():
        raise ValueError('Refusing existing acceptance output')
    output.mkdir(parents=True)
    forbidden = ('torch', 'transformers', 'tokenizers', 'vllm', 'numpy', 'aiohttp')
    denied = []
    def audit(event, values):
        if event == 'import' and any(str(values[0]) == name or str(values[0]).startswith(name + '.')
                                     for name in forbidden):
            denied.append(str(values[0])); raise RuntimeError('ML/HTTP SDK import blocked')
        if event.startswith('socket.'):
            denied.append(event); raise RuntimeError('Socket use blocked')
    sys.addaudithook(audit)
    import inference_kit as api
    from inference_kit.analysis.tokens import source_identity
    library = Path(api.__file__).absolute()
    if 'site-packages' not in library.parts:
        raise ValueError('Acceptance requires ordinary installed site-packages origin')
    expected = {r['name']: r['counts'] for r in read(kit/'examples/EXPECTED_PLAN_COUNTS.json')['plans']}
    fixture_files = sorted(p for p in (kit/'fixtures/prepared').rglob('*') if p.is_file())
    before = {str(p.relative_to(kit)): sha(p) for p in fixture_files}
    commands, plans, refusals = [], [], []
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    console = Path(sys.executable).with_name('inference-kit')
    def invoke(action, values, name, expect_success=True):
        if args.mode == 'api':
            try:
                if action == 'plan':
                    result = api.suite_plan(values['suite'], values['sources'], values['local'], values['output'])
                elif action == 'validate':
                    result = api.suite_validate(values['plan'], values['sources'], values['output'])
                else:
                    result = api.suite_describe(values['plan'], values['output'])
            except (ValueError, KeyError, TypeError, OSError) as error:
                if expect_success: raise
                return {'exception': type(error).__name__, 'error': str(error)}
            if not expect_success: raise AssertionError('Negative control unexpectedly succeeded')
            return result
        prefix = [sys.executable, '-I', '-B', '-m', 'inference_kit'] if args.mode == 'module' else [str(console)]
        command = prefix + ['bench', 'suite', action]
        for key, value in values.items(): command += ['--'+key, str(value)]
        run = subprocess.run(command, cwd=kit, env=env, capture_output=True, text=True, encoding='utf-8')
        commands.append({'name': name, 'command': command, 'returncode': run.returncode,
                         'guard_scope': 'Independent child; parent audit hook is not inherited or an OS sandbox'})
        (output/(name+'.stdout')).write_text(run.stdout, encoding='utf-8')
        (output/(name+'.stderr')).write_text(run.stderr, encoding='utf-8')
        if (run.returncode == 0) != expect_success:
            raise AssertionError('Unexpected CLI result: '+name+': '+run.stderr)
        return {'returncode': run.returncode, 'stdout': run.stdout}
    for name, model in CASES:
        sources = kit/'fixtures/prepared'/model/'SOURCES.json'; plan_dir = output/(name+'-plan')
        invoke('plan', {'suite':kit/'examples'/(name+'.suite.json'), 'sources':sources,
                        'local':kit/'examples/CPU_LOCAL.json', 'output':plan_dir}, name+'-plan')
        frozen = read(plan_dir/'FROZEN_PLAN.json')
        assert frozen['status'] == 'VALIDATED_NOT_EXECUTED'
        assert frozen['semantic']['counts'] == expected[name]
        assert all(frozen[k] == 0 for k in ('execution_count','acquisition_count','GPU_queries',
                                           'model_loads','tokenizer_calls','network_requests','ledger_changes'))
        invoke('validate', {'plan':plan_dir/'FROZEN_PLAN.json','sources':sources,
                            'output':output/(name+'-validate')}, name+'-validate')
        invoke('describe', {'plan':plan_dir/'FROZEN_PLAN.json','output':output/(name+'-describe')}, name+'-describe')
        description = read(output/(name+'-describe')/'PLAN_DESCRIPTION.json')
        validation = read(output/(name+'-validate')/'VALIDATION_RECEIPT.json')
        assert description['plan_identity'] == validation['plan_identity'] == frozen['plan_identity']
        assert description['counts'] == expected[name] and validation['content_revalidated'] is True
        plans.append({'name':name, 'status':frozen['status'], 'plan_identity':frozen['plan_identity'],
                      'data_identity':frozen['semantic']['inputs']['data_identity'], 'counts':expected[name],
                      'artifact_presence':'ARTIFACT_PRESENCE_NOT_CHECKED',
                      'execution_count':0,'acquisition_count':0})
    # Required empty selection must fail, not become a successful empty plan.
    empty = deepcopy(read(kit/'examples/smol-task.suite.json')); empty['tasks'][0]['sample_ids'] = []
    write(output/'empty-required.suite.json', empty)
    result = invoke('plan', {'suite':output/'empty-required.suite.json',
        'sources':kit/'fixtures/prepared/smol/SOURCES.json', 'local':kit/'examples/CPU_LOCAL.json',
        'output':output/'empty-required-plan'}, 'empty-required', expect_success=False)
    failed = read(output/'empty-required-plan'/'VALIDATION_RECEIPT.json')
    assert failed['status'] == 'INVALID_EMPTY_PLAN'
    refusals.append({'case':'empty_required_selection','status':failed['status'],
                     'error':failed['error'],'GPU_queries':0})
    # A known Qwen profile over Smol prepared identity remains syntactically valid
    # but unsupported. It must produce a blocked static plan, never READY/live.
    wrong = deepcopy(read(kit/'examples/smol-task.suite.json'))
    for model in wrong['models']: model['profile'] = 'qwen3_4b_gptq_serving_v1'
    write(output/'wrong-model-profile.suite.json', wrong)
    invoke('plan', {'suite':output/'wrong-model-profile.suite.json',
        'sources':kit/'fixtures/prepared/smol/SOURCES.json', 'local':kit/'examples/CPU_LOCAL.json',
        'output':output/'wrong-model-profile-plan'}, 'wrong-model-profile')
    blocked = read(output/'wrong-model-profile-plan'/'FROZEN_PLAN.json')
    assert blocked['status'] == 'BLOCKED_REQUIRED_OPERATIONS'
    nodes = [n for n in blocked['semantic']['graph']['nodes'] if n['operation_kind'].startswith('serve_')]
    assert nodes and all(n['state'] == 'UNSUPPORTED_CAPABILITY' for n in nodes)
    refusals.append({'case':'Qwen_profile_with_Smol_tokenizer_and_artifact_metadata',
                     'status':blocked['status'],'serve_states':sorted({n['state'] for n in nodes}),
                     'execution_count':blocked['execution_count']})
    # Tamper only a newly copied fixture; keep its original source inventory pins.
    tampered = output/'tampered-prepared';shutil.copytree(kit/'fixtures/prepared/smol', tampered)
    member = tampered/read(tampered/'SOURCES.json')['sources'][0]['path']
    member.write_bytes(member.read_bytes()+b'\n')
    invoke('plan', {'suite':kit/'examples/smol-task.suite.json','sources':tampered/'SOURCES.json',
        'local':kit/'examples/CPU_LOCAL.json','output':output/'tampered-plan'}, 'tampered-input', expect_success=False)
    invalid = read(output/'tampered-plan'/'VALIDATION_RECEIPT.json')
    assert invalid['status'] == 'INVALID_INPUT_OR_CONTRACT' and 'hash changed' in invalid['error']
    refusals.append({'case':'tampered_prepared_member','status':invalid['status'],'error':invalid['error']})
    assert before == {str(p.relative_to(kit)):sha(p) for p in fixture_files}
    receipt = {'status':'PASS','mode':args.mode,'package_version':api.__version__,
       'package_origin':str(library),'source_digest':source_identity(), 'plans':plans,'refusals':refusals,
       'fixture_pins_unchanged':True,'forbidden_import_or_socket_attempts':denied,'commands':commands,
       'guard_scope':'This process import/socket audit only. Literal module/console children tested independently; no hook inheritance or OS sandbox claim.',
       'new_acquisitions':0,'GPU_queries':0,'model_loads':0,'inference_POSTs':0,'tokenizer_calls':0,
       'fresh_retokenization':'NOT_RUN','runtime_preflight':'CHECK_NOT_RUN'}
    assert not denied
    write(output/'PLAN_ACCEPTANCE_RECEIPT.json',receipt)
    print(json.dumps({'status':'PASS','mode':args.mode,'plans':len(plans),'refusals':len(refusals),
                      'source_digest':receipt['source_digest']},ensure_ascii=True))

if __name__ == '__main__':
    main()
