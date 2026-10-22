"""Explicit local-tokenizer preparation; validation and descriptions are stdlib-only.

Prepared manifests are data authorities for CPU planning, not acquisition receipts.
Only prepare() launches the separately installed tokenizer-only worker.
"""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from ..core.io import digest, fields, loads, read, require, safe_name, sha, write
from ..suites.bench05a_plan import source_identity

POLICY = 'existing_tasks_local_tokenizer_v1'
SELECTION = 'field_round_robin_uniform_v1'
TOKENIZER_FILES = {'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'chat_template.jinja', 'vocab.json', 'merges.txt', 'tokenizer.model'}


def contract():
    return {'schema_version': 1, 'kind': 'input_preparation_contract_v1',
      'raw': 'raw_task_manifest_v1', 'spec': 'input_preparation_spec_v1',
      'prepared': 'prepared_input_bundle_v1', 'suite_source': 'prepared_task_source_v1',
      'policy_id': POLICY, 'selection_policy': SELECTION,
      'tokenizer': 'Explicit installed worker; pinned local fast tokenizer; offline; no remote code',
      'roles': ['generation_prompt_only', 'all_choice_likelihood', 'teacher_forced_external_gold'],
      'limits': {'max_joint_tokens': 640, 'max_targets': 32, 'max_full': 8},
      'live_binding': 'LIVE_BINDING_NOT_VALIDATED', 'model_execution': 'NOT_REQUESTED'}


def validate_spec(value):
    fields(value, ('schema_version', 'kind', 'policy_id', 'limits', 'selection', 'seed'))
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and
            value['kind'] == 'input_preparation_spec_v1' and value['policy_id'] == POLICY,
            'Unsupported preparation spec')
    fields(value['limits'], ('max_prompt_tokens', 'max_reference_tokens', 'max_joint_tokens', 'context_tokens'))
    limits = value['limits']
    require(all(type(v) is int and v > 0 for v in limits.values()) and
            limits['max_prompt_tokens'] <= 512 and limits['max_reference_tokens'] <= 128 and
            limits['max_joint_tokens'] <= 640 and limits['context_tokens'] == 2048,
            'Invalid preparation length bounds')
    fields(value['selection'], ('policy_id', 'max_targets', 'max_full'))
    selection = value['selection']
    require(selection['policy_id'] == SELECTION and type(selection['max_targets']) is int and
            1 <= selection['max_targets'] <= 32 and type(selection['max_full']) is int and
            0 <= selection['max_full'] <= min(8, selection['max_targets']), 'Invalid selection limits')
    require(type(value['seed']) is int and 0 <= value['seed'] < 2**32, 'Invalid preparation seed')
    return value


def _hash(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'Invalid SHA256 pin')


def validate_binding(value, verify_files=True, verify_package=True):
    fields(value, ('schema_version', 'kind', 'runtime_python', 'package_site', 'tokenizer_path',
                  'config_path', 'revision', 'model_id', 'tokenizer_files', 'config_sha256',
                  'toolkit_source_identity', 'model_metadata'), ('timeout_seconds',))
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and
            value['kind'] == 'local_tokenizer_binding_v1', 'Unsupported local tokenizer binding')
    from ..tasks.input_records import QWEN_PROFILE
    model = (value['model_id'], value['revision'])
    smol = ('HuggingFaceTB/SmolLM2-1.7B-Instruct', '31b70e2e869a7173562077fd711b654946d38674')
    qwen = ('Qwen/Qwen3-4B-Instruct-2507', 'cdbee75f17c01a7cc42f958dc650907174af0554')
    require(model in (smol, qwen), 'Unsupported tokenizer revision/model')
    if model == qwen:
        config = value['model_metadata']['model_config']
        require(value['model_metadata'].get('capability_profile') == QWEN_PROFILE and
                config.get('architectures') == ['Qwen3ForCausalLM'] and config.get('model_type') == 'qwen3' and
                config.get('vocab_size') == 151936, 'Qwen tokenizer/model profile differs')
    else:
        require(value['model_metadata'].get('capability_profile') != QWEN_PROFILE,
                'Tokenizer/model profile differs')
    pins = value['tokenizer_files']
    require(isinstance(pins, dict) and {'tokenizer.json', 'tokenizer_config.json'} <= pins.keys() and
            pins.keys() <= TOKENIZER_FILES, 'Explicit tokenizer file pins required')
    for identity in [*pins.values(), value['config_sha256'], value['toolkit_source_identity']]:
        _hash(identity)
    require(value['model_metadata']['tokenizer_identity'] == digest(pins) and
            value['model_metadata']['tokenizer_files'] == pins, 'Tokenizer/model metadata pin differs')
    require(type(value.get('timeout_seconds', 180)) is int and 1 <= value.get('timeout_seconds', 180) <= 300,
            'Invalid tokenizer worker timeout')
    for key in ('runtime_python', 'package_site', 'tokenizer_path', 'config_path'):
        require(isinstance(value[key], str) and Path(value[key]).is_absolute(), 'Explicit absolute local locator required')
    if verify_files:
        runtime, tokenizer, config = map(Path, (value['runtime_python'], value['tokenizer_path'], value['config_path']))
        require(runtime.is_file() and os.access(runtime, os.X_OK), 'MISSING_LOCAL_TOKENIZER_RUNTIME')
        require(tokenizer.is_dir() and not tokenizer.is_symlink() and config.is_file(),
                'Missing/unsafe tokenizer/config locator')
        require({n for n in TOKENIZER_FILES if (tokenizer / n).exists()} == set(pins),
                'Unpinned tokenizer/template file set changed')
        for name, identity in pins.items():
            candidate = tokenizer / name
            require(candidate.is_file() and sha(candidate) == identity,
                    'Tokenizer file pin changed: ' + name)
        require(sha(config) == value['config_sha256'] and read(config) == value['model_metadata']['model_config'],
                'Pinned config bytes/metadata changed')
    if verify_package:
        package = Path(value['package_site']) / 'inference_kit'
        require(package.is_dir() and not package.is_symlink(), 'Missing ordinary installed worker package')
        actual = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}
        require(actual and digest(actual) == value['toolkit_source_identity'] == digest(source_identity()),
                'Installed worker/controller source identity differs')
    return value


def read_raw(path):
    from ..tasks.input_records import validate_raw
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe raw input')
    if path.suffix == '.jsonl':
        rows = [loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        require(rows, 'INVALID_EMPTY_PLAN: empty raw JSONL')
        fields(rows[0], ('schema_version', 'kind', 'dataset_id', 'usage_scope'))
        value = rows[0] | {'rows': rows[1:]}
    else:
        value = read(path)
    return validate_raw(value)


def _load(prepared):
    from ..suites.suite_inputs import load_sources
    path = Path(prepared).absolute()
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe prepared manifest')
    value = read(path); root = path.parent.resolve()
    if value.get('kind') != 'prepared_input_bundle_v1':
        return load_sources(path), None
    fields(value, ('schema_version', 'kind', 'source_manifest', 'raw_input', 'spec', 'files',
                   'semantic_identity', 'raw_identity', 'spec_identity', 'data_identity', 'validation_scope'))
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and
            value['validation_scope'] == 'PREPARED_CONTRACT_VALIDATED_ONLY', 'Invalid prepared bundle version/scope')
    files = value['files']; require(isinstance(files, list) and files, 'Empty prepared inventory')
    names = [f['path'] for f in files]
    require(len(names) == len(set(names)), 'Duplicate prepared inventory path')
    for item in files:
        fields(item, ('path', 'bytes', 'sha256'))
        candidate = root / safe_name(item['path'])
        require(candidate.is_file() and not candidate.is_symlink() and candidate.resolve().is_relative_to(root),
                'Unsafe/missing prepared file')
        require(type(item['bytes']) is int and item['bytes'] == candidate.stat().st_size and
                item['sha256'] == sha(candidate), 'Prepared inventory content changed')
    for name in ('source_manifest', 'raw_input', 'spec'):
        require(safe_name(value[name]) in names, 'Unpinned prepared authority')
    loaded = load_sources(root / value['source_manifest'])
    raw = read_raw(root / value['raw_input']); spec = validate_spec(read(root / value['spec']))
    require(digest(raw) == value['raw_identity'] and digest(spec) == value['spec_identity'] and
            loaded['data_identity'] == value['data_identity'], 'Prepared authority identity changed')
    from ..tasks.input_records import sample_key
    expected = {sample_key(r): r for r in raw['rows']}
    actual = {sample_key(s['raw_record']): s['raw_record'] for s in loaded['samples']}
    require(actual == expected and len(loaded['samples']) == len(expected), 'Prepared/raw membership differs')
    for source in loaded['sources']:
        require(source['preparation']['raw_manifest_identity'] == digest(raw) and
                source['preparation']['preparation_spec_identity'] == digest(spec), 'Prepared ancestry changed')
    semantic = digest({'data_identity': loaded['data_identity'], 'raw_identity': digest(raw), 'spec_identity': digest(spec)})
    require(semantic == value['semantic_identity'], 'Prepared semantic identity changed')
    return loaded, value


def _summary(loaded, manifest=None):
    rows = loaded['samples']
    return {'schema_version': 1, 'status': 'PREPARED_CONTRACT_VALIDATED_ONLY',
      'sample_count': len(rows), 'task_count': len(loaded['sources']),
      'tasks': [{'task_id': s['task_id'], 'sample_count': len(s['samples']),
                 'selected_targets': sum(len(r['diagnostic']['targets']) for r in s['samples']),
                 'full_positions': sum(len(r['diagnostic']['full_vocab_target_positions']) for r in s['samples'])}
                for s in loaded['sources']],
      'data_identity': loaded['data_identity'], 'semantic_identity': manifest['semantic_identity'] if manifest else None,
      'tokenizer_assets': 'ARTIFACT_PRESENCE_NOT_CHECKED', 'retokenization': 'NOT_RUN',
      'live_binding': 'LIVE_BINDING_NOT_VALIDATED', 'gpu_actions': 0, 'model_loads': 0,
      'inference_posts': 0, 'tokenizer_loads': 0}


def validate(prepared, output, tokenizer_local=None):
    loaded, manifest = _load(prepared); result = _summary(loaded, manifest)
    if tokenizer_local is not None:
        binding = validate_binding(read(Path(tokenizer_local)), verify_package=False)
        require(all(s['model_metadata']['tokenizer_identity'] == digest(binding['tokenizer_files'])
                    for s in loaded['sources']), 'Prepared/local tokenizer pin differs')
        result['tokenizer_assets'] = 'PINNED_LOCAL_FILES_VERIFIED_NO_TOKENIZER_LOAD'
    out = Path(output); require(not out.exists(), 'Refusing existing validation output')
    write(out / 'VALIDATION_RECEIPT.json', result)
    return result


def describe(prepared, output):
    loaded, manifest = _load(prepared); result = _summary(loaded, manifest)
    out = Path(output); require(not out.exists(), 'Refusing existing description output')
    write(out / 'DESCRIPTION.json', result)
    text = 'Prepared inputs: ' + str(result['sample_count']) + ' samples / ' + str(result['task_count']) + ' tasks\n\n'
    text += '\n'.join(s['task_id'] + ': ' + str(len(s['samples'])) + ' documents' for s in loaded['sources'])
    text += '\n\nGeneration prompts and diagnostic external references have separate roles.\nLive binding: NOT_VALIDATED. Model execution/tokenizer loading: NOT_RUN.\n'
    (out / 'DESCRIPTION.txt').write_text(text, encoding='utf-8')
    return result


def prepare(spec, input, tokenizer_local, output):
    """Publish a new bundle atomically after worker and stdlib contract validation."""
    spec, input, binding_file, output = map(lambda p: Path(p).absolute(), (spec, input, tokenizer_local, output))
    raw = read_raw(input); configuration = validate_spec(read(spec)); binding = validate_binding(read(binding_file))
    raw_sha, spec_sha, binding_sha = sha(input), sha(spec), sha(binding_file)
    require(not output.exists() and not output.is_symlink(), 'Refusing existing preparation output')
    output.parent.mkdir(parents=True, exist_ok=True)
    lockpath = output.parent / ('.' + output.name + '.prepare.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), 'Refusing existing preparation output')
        stage = output.parent / ('.' + output.name + '.preparing-' + uuid.uuid4().hex)
        stage.mkdir()
        try:
            raw_name = 'RAW_INPUT' + input.suffix
            shutil.copyfile(input, stage / raw_name); shutil.copyfile(spec, stage / 'PREPARATION_SPEC.json')
            env = dict(os.environ); env.pop('PYTHONHOME', None)
            env.update({'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
              'USE_TORCH': '0', 'USE_TF': '0', 'USE_FLAX': '0', 'TOKENIZERS_PARALLELISM': 'false',
              'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': binding['package_site']})
            args = [binding['runtime_python'], '-B', '-m', 'inference_kit.probes.input_preparation_worker',
                    '--input', str(stage / raw_name), '--spec', str(stage / 'PREPARATION_SPEC.json'),
                    '--binding', str(binding_file), '--output', str(stage)]
            result = subprocess.run(args, cwd=stage, env=env, capture_output=True, timeout=binding.get('timeout_seconds', 180))
            (stage / 'WORKER_STDOUT.log').write_bytes(result.stdout); (stage / 'WORKER_STDERR.log').write_bytes(result.stderr)
            failure = 'MISSING_LOCAL_TOKENIZER_RUNTIME' if b'MISSING_LOCAL_TOKENIZER_RUNTIME' in result.stderr else 'LOCAL_TOKENIZER_PREPARATION_FAILED'
            require(result.returncode == 0, failure + ': see ' + str(stage / 'WORKER_STDERR.log'))
            require(sha(input) == sha(stage / raw_name) == raw_sha and sha(spec) == sha(stage / 'PREPARATION_SPEC.json') == spec_sha and
                    sha(binding_file) == binding_sha and read_raw(stage / raw_name) == raw and read(stage / 'PREPARATION_SPEC.json') == configuration,
                    'Inputs changed during preparation')
            from ..suites.suite_inputs import load_sources
            loaded = load_sources(stage / 'SOURCES.json')
            require(len(loaded['samples']) == len(raw['rows']), 'Prepared sample coverage differs')
            validate_binding(binding)
            manifest = {'schema_version': 1, 'kind': 'prepared_input_bundle_v1', 'source_manifest': 'SOURCES.json',
              'raw_input': raw_name, 'spec': 'PREPARATION_SPEC.json', 'raw_identity': digest(raw),
              'spec_identity': digest(configuration), 'data_identity': loaded['data_identity'],
              'semantic_identity': digest({'data_identity': loaded['data_identity'], 'raw_identity': digest(raw), 'spec_identity': digest(configuration)}),
              'validation_scope': 'PREPARED_CONTRACT_VALIDATED_ONLY',
              'files': [{'path': str(p.relative_to(stage)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                        for p in sorted(stage.rglob('*')) if p.is_file()]}
            write(stage / 'PREPARED_MANIFEST.json', manifest)
            _load(stage / 'PREPARED_MANIFEST.json')
            require(not output.exists(), 'Preparation destination appeared during publication')
            os.rename(stage, output)
            return _summary(loaded, manifest) | {'tokenizer_loads': 1, 'retokenization': 'COMPLETED_LOCAL_ONLY',
              'tokenizer_assets': 'PINNED_LOCAL_FILES_VERIFIED', 'output_manifest': str(output / 'PREPARED_MANIFEST.json')}
        except Exception as error:
            for name in ('PREPARED_MANIFEST.json', 'SOURCES.json'):
                success = stage / name
                if success.exists():
                    success.rename(stage / ('UNPUBLISHED_' + name))
            write(stage / 'FAILURE.json', {'schema_version': 1, 'status': 'PREPARATION_FAILED',
                  'error_type': type(error).__name__, 'reason': str(error), 'success_manifest_published': False})
            raise
