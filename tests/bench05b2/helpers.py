"""Explicit synthetic B2 contracts, no real tokenizer/model or ML dependency."""
import copy
import json
from pathlib import Path
from inference_kit.core.io import read, write, digest, sha
from inference_kit.tasks import contract_policy as policy
from inference_kit.probes.task_tokens import prepare_row
from inference_kit.suites.bench05b1_run import seal
from bench05b1.helpers import local_fixture as b1_local, generation


class Tokenizer:
    is_fast = True
    all_special_ids = []
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt and len(messages) == 1
        return 'SYNTHETIC CHAT\n' + messages[0]['content'] + '\nASSISTANT\n'
    def __call__(self, text, **kwargs):
        offsets = [(i, min(i + 4, len(text))) for i in range(0, len(text), 4)]
        return {'input_ids': [1 + i % 254 for i in range(len(offsets))], 'offset_mapping': offsets,
                'special_tokens_mask': [0] * len(offsets)}
    def encode(self, text, **kwargs):
        return self(text)['input_ids']
    def convert_ids_to_tokens(self, ids):
        return ['toy-' + str(i) for i in ids]


def prepared_fixture(cfg):
    data = policy.make_dataset(); cache = {}; tokenizer = Tokenizer()
    for sample in data['rows']:
        for name in policy.POLICIES:
            row = prepare_row(sample, policy.probe_sample(sample, name), policy.render(sample, name),
                              tokenizer, cfg, 'synthetic-tokenizer', 'synthetic-space', policy.PREPARATION, 512, 640)
            row.update(policy_id=name, policy_identity=policy.policy_record(name)['policy_identity'])
            cache[sample['sample_id'], name] = row
    rows = [cache[r['sample_id'], r['policy_id']] for r in policy.request_order(data['rows'])]
    return data, {'schema_version': 1, 'kind': 'prepared_b2_contract_samples_v1', 'task_id': 'tech_record_extract',
      'task_version': '0.1', 'preparation_id': policy.PREPARATION, 'generation_rows': rows,
      'generation_rows_identity': digest(rows), 'model_config': cfg, 'tokenizer_files': {},
      'tokenizer_identity': 'synthetic-tokenizer', 'token_id_space': 'synthetic-space', 'vocabulary_size': 256,
      'by_policy': {n: [cache[sid, n] for sid in policy.select_probes(data['rows'])] for n in ('P00', 'P11')},
      'probe_selection': {'seed': 909243, 'sample_ids': policy.select_probes(data['rows'])},
      'bounds': {'prompt': 512, 'reference': 128, 'joint': 640}}


def local_fixture(root):
    suite_file, pair_file, local_file = b1_local(root)
    cfg = read(root / 'prepared.json')['model_config']; data, prepared = prepared_fixture(cfg)
    suite = read(Path(__file__).parents[2] / 'configs/bench05b2/suite.json')
    suite['budget'].update(minimum_cpu_available_bytes=1, minimum_disk_free_bytes=1)
    # These are mutable synthetic test inputs before any freeze, not historical evidence.
    for p, value in ((root / 'data.json', data), (suite_file, suite)):
        p.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    prepared['authored_sha256'] = sha(root / 'data.json')
    (root / 'prepared.json').write_text(json.dumps(prepared, ensure_ascii=False), encoding='utf-8')
    local = read(local_file); local['serving_runtime_identity'] = 'synthetic-serving-runtime'
    local_file.write_text(json.dumps(local), encoding='utf-8')
    return suite_file, pair_file, local_file


def synthetic_run(root, partial=False, invalid=False):
    from inference_kit.suites.bench05b2_plan import plan
    root = Path(root); root.mkdir(); inputs = root.parent / (root.name + '-inputs'); inputs.mkdir()
    p = plan(*local_fixture(inputs), root / 'plan'); p.pop('plan_identity')
    p['evidence_kind'] = 'synthetic_contract'; p['plan_identity'] = digest(p)
    write(root / 'PLAN_FROZEN.json', p); jobs = {}
    source = {s['sample_id']: s for s in p['data']['rows']}
    for arm in p['arms']:
        d = root / 'attempts' / ('serve-' + arm['arm_id'] + '-attempt1'); d.mkdir(parents=True)
        rows = []
        for prepared in p['samples']['generation_rows']:
            sample = source[prepared['sample_id']]
            g = generation(sample, prepared, arm, output='{}' if invalid else None)
            g.update(policy_id=prepared['policy_id'], policy_identity=prepared['policy_identity'],
                     prepared_identity=digest(prepared), scorer_identity=p['scorer_identity'],
                     runtime_identity=p['serving_runtime_identity'], request_id=arm['arm_id'] + '-' + prepared['policy_id'] + '-' + sample['sample_id'])
            rows.append(g)
        if partial:
            rows = rows[:16]
        (d / 'generations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        jobs['serve-' + arm['arm_id']] = {'status': 'SYNTHETIC_CONTRACT_ONLY', 'directory': str(d.relative_to(root))}
    write(root / 'progress.json', {'status': 'SYNTHETIC_CONTRACT_ONLY', 'jobs': jobs, 'diagnostics': {}})
    seal(root, scope='bench05b2_contract_raw')
    return root
