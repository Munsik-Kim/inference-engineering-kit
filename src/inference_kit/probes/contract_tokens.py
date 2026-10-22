"""B2 local tokenizer preparation. No model loading; explicit 512/128/640 profile."""
import argparse
import sys
from pathlib import Path
from importlib import metadata
from ..core.io import read, write, sha, digest, require
from ..tasks import contract_policy as policy
from ..tasks import tech_record as task
from .task_tokens import prepare_row


def prepare(data_file, tokenizer_path, config_path, output, runtime_output):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    data, cfg = read(Path(data_file)), read(Path(config_path))
    require(data['rows_identity'] == digest(data['rows']), 'B2 data changed')
    identities = {p.name: sha(p) for p in Path(tokenizer_path).iterdir() if p.name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'vocab.json', 'merges.txt')}
    tokenizer_id = digest(identities)
    space = digest(tokenizer.get_vocab())
    selected = policy.select_probes(data['rows'])
    by_id = {r['sample_id']: r for r in data['rows']}
    cache = {}
    for sample in data['rows']:
        policy.verify_gold(sample)
        for name in policy.POLICIES:
            row = prepare_row(sample, policy.probe_sample(sample, name), policy.render(sample, name),
                              tokenizer, cfg, tokenizer_id, space, policy.PREPARATION, 512, 640)
            row.update(policy_id=name, policy_identity=policy.policy_record(name)['policy_identity'])
            cache[sample['sample_id'], name] = row
    rows = [cache[r['sample_id'], r['policy_id']] for r in policy.request_order(data['rows'])]
    probes = {name: [cache[sid, name] for sid in selected] for name in ('P00', 'P11')}
    prepared = {'schema_version': 1, 'kind': 'prepared_b2_contract_samples_v1', 'task_id': task.TASK,
      'task_version': task.VERSION, 'preparation_id': policy.PREPARATION, 'authored_sha256': sha(Path(data_file)),
      'task_rows_identity': data['rows_identity'], 'model_config': cfg, 'model_config_sha256': sha(Path(config_path)),
      'tokenizer_files': identities, 'tokenizer_identity': tokenizer_id, 'token_id_space': space,
      'vocabulary_size': len(tokenizer.get_vocab()), 'generation_rows': rows, 'generation_rows_identity': digest(rows),
      'by_policy': probes, 'probe_selection': {'seed': 909243, 'sample_ids': selected,
        'rule': 'basic en / correction ko / missing en / similar-ID ko, selected before responses'},
      'bounds': {'prompt': 512, 'reference': 128, 'joint': 640}, 'model_loaded': False, 'gpu_used': False}
    write(Path(output), prepared)
    packages = {name: {'version': metadata.version(name), 'package_root': str(metadata.distribution(name).locate_file(''))}
                for name in ('torch', 'transformers', 'compressed-tensors', 'safetensors', 'accelerate', 'numpy')}
    import torch
    require(not torch.cuda.is_initialized(), 'Tokenizer preparation initialized CUDA')
    write(Path(runtime_output), {'python': sys.executable, 'packages': packages, 'model_loads': 0,
                                'gpu_used': False, 'cuda_initialized': False, 'preparation_id': policy.PREPARATION})
    return prepared


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('data', 'tokenizer', 'config', 'output', 'runtime-output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    result = prepare(a.data, a.tokenizer, a.config, a.output, a.runtime_output)
    print(__import__('json').dumps({'rows': len(result['generation_rows']), 'gpu_used': False,
          'max_prompt': max(len(r['generation_input_ids']) for r in result['generation_rows']),
          'max_joint': max(len(r['input_ids']) for r in result['generation_rows'])}, ensure_ascii=True))
