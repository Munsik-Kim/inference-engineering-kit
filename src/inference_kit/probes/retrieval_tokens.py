"""Explicit local tokenizer preparation for closed-context B4; never loads weights.

CPU plan/replay use the stored offsets and do not import a tokenizer or ML SDK.
The shared B1 wrapper keeps its original defaults; this task declares its own bounds.
"""
import argparse
import sys
from importlib import metadata
from pathlib import Path

from ..core.io import digest, read, require, sha, write
from .task_tokens import prepare_row

POLICY = 'retrieval_chat_joint_answer_evidence_spans_v1'
REFERENCE_SOURCE = 'task_author_gold_answer_evidence_v1'
SPAN_POLICY = 'prompt_passage_content_codepoint_utf8_any_token_overlap_v1'
BOUNDS = {'prompt': 512, 'reference': 64, 'joint': 576}


def content_span_records(sample, row):
    """Map exact passage CONTENT, excluding its [Pxx] label, into chat coordinates.

    A token counts when its nonempty source interval intersects content. A token
    crossing the boundary is retained whole; no key axis is cropped/renormalized.
    The same helper is safe to rerun against saved tokenizer offsets on CPU.
    """
    document = sample['document']
    prompt = row['prompt']
    require(prompt.count(document) == 1, 'Document must occur exactly once in rendered prompt')
    start = prompt.index(document)
    offsets = row['offsets_codepoints']
    require(len(offsets) == len(row['input_ids']), 'Passage token offset coverage differs')
    records = []
    for passage in sample['passages']:
        a, b = passage['content_span']
        content = passage['content']
        require(type(a) is int and type(b) is int and 0 <= a < b <= len(document), 'Invalid passage content span')
        require(document[a:b] == content, 'Passage content/source text differs')
        pa, pb = start + a, start + b
        require(prompt[pa:pb] == content, 'Chat-rendered passage text differs')
        positions = [i for i, (left, right) in enumerate(offsets)
                     if left < pb and right > pa and right > left]
        require(positions and max(positions) < row['prompt_token_boundary'],
                'Passage content must have prompt token coverage only')
        records.append({'passage_id': passage['passage_id'], 'content': content,
          'content_hash': digest(content), 'document_codepoint_span': [a, b],
          'prompt_codepoint_span': [pa, pb],
          'prompt_utf8_byte_span': [len(prompt[:pa].encode('utf-8')), len(prompt[:pb].encode('utf-8'))],
          'overlapping_token_positions': positions, 'coordinate_space': 'joint_input_ids',
          'overlap_rule': 'nonempty_token_interval_intersects_content; include_whole_boundary_token'})
    require(len(records) == 6 and len({r['passage_id'] for r in records}) == 6,
            'Six unique source passages required')
    gold = sample['gold']
    if not gold['answerable']:
        require(gold['evidence'] == 'NONE' and gold['content_span'] is None,
                'Absent answer must have no evidence content span')
        evidence = {'status': 'NOT_APPLICABLE', 'passage_id': None, 'content': None,
                    'document_codepoint_span': None, 'prompt_codepoint_span': None,
                    'prompt_utf8_byte_span': None, 'overlapping_token_positions': None}
    else:
        matched = [r for r in records if r['passage_id'] == gold['evidence']]
        require(len(matched) == 1 and matched[0]['document_codepoint_span'] == gold['content_span'],
                'Gold evidence passage identity/span differs')
        evidence = {'status': 'AVAILABLE', **matched[0]}
    return {'source_span_policy': SPAN_POLICY, 'passage_content_spans': records,
            'reference_evidence_span': evidence}


def validate_content_spans(sample, row):
    expected = content_span_records(sample, row)
    require(all(row.get(k) == value for k, value in expected.items()),
            'Stored passage content coordinate identity differs')
    return expected


def prepare_retrieval_row(sample, tokenizer, cfg, tokenizer_identity, token_space):
    from ..tasks import context_evidence as task
    task.verify_gold(sample)
    row = prepare_row(sample, task.build_probe_sample(sample), task.render_prompt(sample),
      tokenizer, cfg, tokenizer_identity, token_space, POLICY, BOUNDS['prompt'], BOUNDS['joint'],
      task_id=task.TASK, task_version=task.VERSION, reference_source=REFERENCE_SOURCE,
      max_reference=BOUNDS['reference'])
    row.update(content_span_records(sample, row))
    return row


def prepare(data_file, tokenizer_path, config_path, output, runtime_output):
    """Opt-in tokenizer-only worker using an already installed, local runtime."""
    from ..tasks import context_evidence as task
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True,
                                             trust_remote_code=False)
    data, cfg = read(Path(data_file)), read(Path(config_path))
    require(data['task_id'] == task.TASK and data['task_version'] == task.VERSION,
            'B4 tokenizer task identity differs')
    require(data['rows_identity'] == digest(data['rows']), 'B4 authored data changed')
    files = {p.name: sha(p) for p in Path(tokenizer_path).iterdir()
             if p.name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                           'chat_template.jinja', 'vocab.json', 'merges.txt')}
    tid, space = digest(files), digest(tokenizer.get_vocab())
    rows = [prepare_retrieval_row(s, tokenizer, cfg, tid, space) for s in data['rows']]
    selected = task.select_probes(data['rows'])
    by_id = {row['sample_id']: row for row in rows}
    probes = [by_id[sid] for sid in selected]
    require(len(rows) == 28 and len(probes) == 4, 'B4 must prepare SMOKE4/EVAL24 and four probes')
    prepared = {'schema_version': 1, 'kind': 'prepared_retrieval_samples_v1',
      'task_id': task.TASK, 'task_version': task.VERSION, 'preparation_id': POLICY,
      'authored_sha256': sha(Path(data_file)), 'task_rows_identity': data['rows_identity'],
      'model_config': cfg, 'model_config_sha256': sha(Path(config_path)),
      'tokenizer_files': files, 'tokenizer_identity': tid, 'token_id_space': space,
      'vocabulary_size': len(tokenizer.get_vocab()), 'generation_rows': rows,
      'generation_rows_identity': digest(rows), 'rows': probes, 'rows_identity': digest(probes),
      'probe_selection': {'seed': 909265, 'sample_ids': selected,
        'rule': 'one per case, English2/Korean2 including answerable/absent, before responses'},
      'bounds': BOUNDS, 'source_span_policy': SPAN_POLICY, 'model_loaded': False, 'gpu_used': False}
    write(Path(output), prepared)
    packages = {n: {'version': metadata.version(n),
                    'package_root': str(metadata.distribution(n).locate_file(''))}
                for n in ('torch', 'transformers', 'compressed-tensors', 'safetensors', 'accelerate', 'numpy')}
    import torch
    require(not torch.cuda.is_initialized(), 'Tokenizer preparation initialized CUDA')
    write(Path(runtime_output), {'python': sys.executable, 'packages': packages, 'model_loads': 0,
      'gpu_used': False, 'cuda_initialized': False, 'preparation_id': POLICY})
    return prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'tokenizer', 'config', 'output', 'runtime-output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.data, args.tokenizer, args.config, args.output, args.runtime_output)
    print(__import__('json').dumps({'generation_rows': len(result['generation_rows']),
      'probe_rows': len(result['rows']), 'max_prompt': max(len(r['generation_input_ids']) for r in result['generation_rows']),
      'max_reference': max(len(r['reference_separate_ids']) for r in result['generation_rows']),
      'max_joint': max(len(r['input_ids']) for r in result['generation_rows']), 'gpu_used': False}, ensure_ascii=True))
