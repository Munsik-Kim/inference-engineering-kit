"""Qwen preparation controls; toy encoding is SYNTHETIC_CONTRACT_TEST only."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import unicodedata
from types import SimpleNamespace

from inference_kit.core.io import digest, sha, write
from inference_kit.probes.input_preparation import validate_binding
from inference_kit.probes.input_preparation_worker import prepare_rows, tokenizer_offset_probe, _qwen_prompt_offsets
from inference_kit.suites.prepared_inputs import validate_source
from inference_kit.tasks.input_records import QWEN_PROFILE, arc_context, render_pinned_chat, render_record
from bench05e4a.test_token_preparation import ToyFastTokenizer, worker_fixture


class ToyNFC:
    def __str__(self): return 'NFC()'
    def normalize_str(self, value): return unicodedata.normalize('NFC', value)


class QwenToyTokenizer(ToyFastTokenizer):
    backend_tokenizer = SimpleNamespace(normalizer=ToyNFC())
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return render_pinned_chat(messages[0]['content'], QWEN_PROFILE)


def qwen_fixture():
    raw, spec, metadata, _ = worker_fixture()
    metadata.update(capability_profile=QWEN_PROFILE,
                    tokenizer_files={'tokenizer.json': '1' * 64, 'tokenizer_config.json': '2' * 64})
    metadata['tokenizer_identity'] = digest(metadata['tokenizer_files'])
    return raw, spec, metadata, QwenToyTokenizer()


class QwenPreparationTests(unittest.TestCase):
    def test_old_smol_template_and_raw_arc_parent_are_unchanged(self):
        raw, spec, metadata, tokenizer = qwen_fixture(); before = deepcopy(raw)
        sources = prepare_rows(raw, spec, metadata, tokenizer)['sources']
        self.assertEqual(raw, before)
        arc = sources[2]['samples'][0]
        self.assertEqual(raw['rows'][2]['sample']['prompt_context'], arc_context(arc['authored_sample']['question']))
        self.assertNotEqual(arc['authored_sample']['prompt'], raw['rows'][2]['sample']['prompt_context'])
        self.assertNotIn('SmolLM', arc['authored_sample']['prompt'])
        self.assertEqual(arc['authored_sample']['prompt'], arc_context(arc['authored_sample']['question'], QWEN_PROFILE))

    def test_qwen_three_task_prepared_sources_validate(self):
        raw, spec, metadata, tokenizer = qwen_fixture()
        sources = prepare_rows(raw, spec, metadata, tokenizer)['sources']
        self.assertEqual(len(sources), 3)
        for source in sources:
            self.assertIs(validate_source(source), source)
            row = source['samples'][0]['diagnostic']
            self.assertEqual(row['tokenization']['model_profile'], QWEN_PROFILE)
            self.assertEqual(row['tokenization']['chat_template_applied'], 1)
        arc = sources[2]['samples'][0]
        self.assertEqual(len(arc['likelihood']['requests']), 3)
        for choice, request in zip(arc['authored_sample']['choices'], arc['likelihood']['requests']):
            self.assertEqual(request['normalization_char_length'], len(choice))
            self.assertEqual(request['input_ids'], request['context_ids'] + request['continuation_ids'])
            self.assertEqual(request['moved_trailing_context_whitespace'], '\n')

    def test_smol_prepared_cannot_be_relabelled_qwen(self):
        raw, spec, metadata, tokenizer = worker_fixture()
        metadata['tokenizer_files'] = {'tokenizer.json': '1' * 64, 'tokenizer_config.json': '2' * 64}
        metadata['tokenizer_identity'] = digest(metadata['tokenizer_files'])
        source = prepare_rows(raw, spec, metadata, tokenizer)['sources'][0]
        source['model_metadata']['capability_profile'] = QWEN_PROFILE
        with self.assertRaisesRegex(ValueError, 'proof changed|prompt changed'):
            validate_source(source)

    def test_template_mismatch_and_unknown_profile_rejected(self):
        raw, spec, metadata, tokenizer = qwen_fixture()
        with self.assertRaisesRegex(ValueError, 'chat template'):
            prepare_rows(raw, spec, metadata, ToyFastTokenizer())
        metadata['capability_profile'] = 'unknown-model-profile'
        with self.assertRaisesRegex(ValueError, 'Unsupported preparation model profile'):
            prepare_rows(raw, spec, metadata, tokenizer)

    def test_hidden_gold_changes_reference_but_not_qwen_generation(self):
        raw, _, _, _ = qwen_fixture(); original = raw['rows'][0]; changed = deepcopy(original)
        changed['sample']['gold']['temperature_c'] = 123456
        left, right = render_record(original, QWEN_PROFILE), render_record(changed, QWEN_PROFILE)
        self.assertEqual(render_pinned_chat(left['content'], QWEN_PROFILE), render_pinned_chat(right['content'], QWEN_PROFILE))
        self.assertNotEqual(left['reference_text'], right['reference_text'])

    def test_qwen_pinned_binding_requires_matching_model_profile_and_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); tokenizer = root / 'tokenizer'; tokenizer.mkdir()
            for name in ('tokenizer.json', 'tokenizer_config.json'):
                (tokenizer / name).write_text('{}', encoding='utf-8')
            config = {'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3', 'vocab_size': 151936}
            write(root / 'config.json', config); pins = {p.name: sha(p) for p in tokenizer.iterdir()}
            value = {'schema_version': 1, 'kind': 'local_tokenizer_binding_v1', 'runtime_python': sys.executable,
              'package_site': str(root / 'site'), 'tokenizer_path': str(tokenizer), 'config_path': str(root / 'config.json'),
              'revision': 'cdbee75f17c01a7cc42f958dc650907174af0554', 'model_id': 'Qwen/Qwen3-4B-Instruct-2507',
              'tokenizer_files': pins, 'config_sha256': sha(root / 'config.json'), 'toolkit_source_identity': 'a' * 64,
              'model_metadata': {'capability_profile': QWEN_PROFILE, 'tokenizer_files': pins,
                                 'tokenizer_identity': digest(pins), 'model_config': config}}
            validate_binding(value, verify_package=False)
            for change in ('profile', 'revision', 'config'):
                altered = deepcopy(value)
                if change == 'profile': altered['model_metadata'].pop('capability_profile')
                elif change == 'revision': altered['revision'] = '31b70e2e869a7173562077fd711b654946d38674'
                else: altered['model_metadata']['model_config']['model_type'] = 'llama'
                with self.assertRaises(ValueError): validate_binding(altered, verify_package=False)

    def test_nfc_probe_reports_native_gap_without_inventing_offsets(self):
        class NativeGapProbe(QwenToyTokenizer):
            def __call__(self, text, **kwargs):
                positions = [i for i in range(len(text)) if i != 4]
                return {'input_ids': [1] * len(positions),
                        'offset_mapping': [(i, i + 1) for i in positions],
                        'special_tokens_mask': [0] * len(positions)}
        tokenizer = NativeGapProbe()
        with self.assertRaisesRegex(ValueError, 'lost authored codepoints'):
            tokenizer_offset_probe(tokenizer)
        result = tokenizer_offset_probe(tokenizer, QWEN_PROFILE)
        self.assertEqual(result['status'], 'PARTIAL_FIXED_PROBE_NORMALIZATION')
        affected = [s for s in result['authored_spans'] if s.get('status')]
        self.assertEqual(len(affected), 1)
        self.assertEqual(affected[0]['unmapped_codepoint_positions'], [4])
        self.assertNotIn([4, 5], result['offsets_codepoints'])
        receipt = _qwen_prompt_offsets(tokenizer, 'e\u0301', 'A', [[0, 1], [2, 3]])
        self.assertTrue(receipt['normalizer_changed_prompt'])
        self.assertEqual(receipt['unmapped_codepoint_positions'], [1])
        with self.assertRaisesRegex(ValueError, 'reference normalization'):
            _qwen_prompt_offsets(tokenizer, 'A', 'e\u0301', [[0, 3]])
        with self.assertRaisesRegex(ValueError, 'omit non-normalization'):
            _qwen_prompt_offsets(tokenizer, 'AB', 'C', [[0, 1], [2, 3]])


if __name__ == '__main__': unittest.main()
