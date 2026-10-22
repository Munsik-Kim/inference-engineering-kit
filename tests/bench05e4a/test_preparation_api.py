"""Preparation orchestration and portable bundle tests; synthetic tokenizer only."""
from copy import deepcopy
from contextlib import redirect_stdout
import builtins
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import digest, read, sha, write
from inference_kit.probes import input_preparation as preparation
from inference_kit.probes.input_preparation_worker import prepare_rows
from bench05e4a.test_token_preparation import worker_fixture
from bench05e2.test_plan import model_metadata


def bundle_fixture(root, *, publish_manifest=True):
    """A closed synthetic bundle, never a local tokenizer or model measurement."""
    raw, spec, observed, tokenizer = worker_fixture()
    metadata = model_metadata(); metadata.update(observed)
    metadata['tokenizer_files'] = {'tokenizer.json': '1' * 64, 'tokenizer_config.json': '2' * 64}
    metadata['tokenizer_identity'] = digest(metadata['tokenizer_files'])
    metadata['token_spec'].update(max_target_positions_per_sample=16, max_full_vocab_positions_per_sample=4)
    result = prepare_rows(raw, spec, metadata, tokenizer)
    write(root / 'RAW_INPUT.json', raw); write(root / 'PREPARATION_SPEC.json', spec)
    entries = []
    for index, source in enumerate(result['sources']):
        name = 'source-' + str(index) + '.json'; write(root / name, source)
        entries.append({'source_id': source['source_id'], 'path': name, 'sha256': sha(root / name)})
    write(root / 'SOURCES.json', {'schema_version': 1, 'kind': 'suite_prepared_sources_v1', 'sources': entries})
    from inference_kit.suites.suite_inputs import load_sources
    loaded = load_sources(root / 'SOURCES.json')
    manifest = {'schema_version': 1, 'kind': 'prepared_input_bundle_v1', 'source_manifest': 'SOURCES.json',
                'raw_input': 'RAW_INPUT.json', 'spec': 'PREPARATION_SPEC.json',
                'raw_identity': digest(raw), 'spec_identity': digest(spec), 'data_identity': loaded['data_identity'],
                'semantic_identity': digest({'data_identity': loaded['data_identity'], 'raw_identity': digest(raw), 'spec_identity': digest(spec)}),
                'validation_scope': 'PREPARED_CONTRACT_VALIDATED_ONLY',
                'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(root.iterdir()) if p.is_file()]}
    if publish_manifest: write(root / 'PREPARED_MANIFEST.json', manifest)
    return raw, spec, metadata, manifest


def repin_file(root, manifest, name):
    """Change a synthetic fixture's explicit authority to test deeper ancestry."""
    for item in manifest['files']:
        if item['path'] == name: item.update(bytes=(root / name).stat().st_size, sha256=sha(root / name))
    (root / 'PREPARED_MANIFEST.json').write_text(json.dumps(manifest), encoding='utf-8')


class PreparationApiTests(unittest.TestCase):
    def test_strict_spec_rejects_bool_nonfinite_and_unknown(self):
        _, original, _, _ = worker_fixture()
        for mutate in (
            lambda v: v.update(unrecognized=True), lambda v: v.update(seed=True),
            lambda v: v['limits'].update(max_joint_tokens=True),
            lambda v: v['selection'].update(max_targets=True),
            lambda v: v['selection'].update(max_full=float('nan')),
            lambda v: v['limits'].update(context_tokens=4096),
            lambda v: v['selection'].update(max_targets=0),
        ):
            value=deepcopy(original); mutate(value)
            with self.assertRaises(ValueError): preparation.validate_spec(value)
        self.assertEqual(preparation.validate_spec(original), original)

    def test_jsonl_header_and_duplicate_keys_are_strict(self):
        raw, _, _, _ = worker_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'한글.jsonl'
            header={k:v for k,v in raw.items() if k!='rows'}
            path.write_text('\n'.join(json.dumps(v,ensure_ascii=False) for v in [header,*raw['rows']])+'\n', encoding='utf-8')
            self.assertEqual(preparation.read_raw(path),raw)
            path.write_text(json.dumps(raw['rows'][0])+'\n', encoding='utf-8')
            with self.assertRaises(ValueError): preparation.read_raw(path)
            path.write_text('{"schema_version":1,"schema_version":1}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Duplicate JSON key'): preparation.read_raw(path)

    def test_bundle_validate_describe_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'자료'; root.mkdir(); bundle_fixture(root)
            manifest=root/'PREPARED_MANIFEST.json'; before={p.name:sha(p) for p in root.iterdir()}
            first=preparation.validate(manifest,Path(tmp)/'validation')
            second=preparation.describe(manifest,Path(tmp)/'description')
            self.assertEqual(first['sample_count'],3); self.assertEqual(first,second)
            self.assertEqual(first['live_binding'],'LIVE_BINDING_NOT_VALIDATED')
            self.assertEqual(first['tokenizer_loads'],0);self.assertEqual(before,{p.name:sha(p) for p in root.iterdir()})
            with self.assertRaisesRegex(ValueError,'existing'):preparation.validate(manifest,Path(tmp)/'validation')
            with self.assertRaisesRegex(ValueError,'existing'):preparation.describe(manifest,Path(tmp)/'description')

    def test_bundle_file_tamper_missing_and_duplicate_inventory_rejected(self):
        for corruption in ('bytes','missing','duplicate'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); _,_,_,manifest=bundle_fixture(root)
                if corruption=='bytes':(root/'RAW_INPUT.json').write_bytes((root/'RAW_INPUT.json').read_bytes()+b' ')
                elif corruption=='missing':(root/'SOURCES.json').unlink()
                else:
                    manifest['files'].append(deepcopy(manifest['files'][0]));(root/'PREPARED_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
                with self.assertRaises(ValueError):preparation._load(root/'PREPARED_MANIFEST.json')

    def test_bundle_scope_cannot_be_silently_relabelled_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,manifest=bundle_fixture(root)
            manifest['validation_scope']='INVALID'
            (root/'PREPARED_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaises(ValueError):preparation._load(root/'PREPARED_MANIFEST.json')

    def test_bundle_ancestry_rejects_changed_parent_even_with_new_inventory_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,manifest=bundle_fixture(root)
            entry=read(root/'SOURCES.json')['sources'][0]; path=root/entry['path']; source=read(path)
            source['preparation']['raw_manifest_identity']='f'*64;path.write_text(json.dumps(source),encoding='utf-8')
            sources=read(root/'SOURCES.json');sources['sources'][0]['sha256']=sha(path)
            (root/'SOURCES.json').write_text(json.dumps(sources),encoding='utf-8')
            from inference_kit.suites.suite_inputs import load_sources
            manifest['data_identity']=load_sources(root/'SOURCES.json')['data_identity']
            manifest['semantic_identity']=digest({k:manifest[k] for k in ('data_identity','raw_identity','spec_identity')})
            repin_file(root,manifest,entry['path']);repin_file(root,manifest,'SOURCES.json')
            with self.assertRaisesRegex(ValueError,'ancestry'):preparation._load(root/'PREPARED_MANIFEST.json')

    def test_bundle_inventory_path_and_symlink_escape_rejected(self):
        for corruption in ('parent','symlink'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'bundle';root.mkdir();_,_,_,manifest=bundle_fixture(root)
                if corruption=='parent':manifest['files'][0]['path']='../outside.json'
                else:
                    outside=Path(tmp)/'outside.json';outside.write_text('{}',encoding='utf-8');(root/'escape.json').symlink_to(outside)
                    manifest['files'].append({'path':'escape.json','bytes':2,'sha256':sha(outside)})
                (root/'PREPARED_MANIFEST.json').write_text(json.dumps(manifest),encoding='utf-8')
                with self.assertRaises(ValueError):preparation._load(root/'PREPARED_MANIFEST.json')

    def test_validate_describe_and_cli_help_do_not_import_ml_or_launch(self):
        blocked=[]; original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in {'torch','transformers','tokenizers','vllm','numpy','aiohttp','compressed_tensors'}:
                blocked.append(name);raise AssertionError('Unexpected SDK import: '+name)
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';root.mkdir();bundle_fixture(root)
            with patch('builtins.__import__',guarded),patch('subprocess.run',side_effect=AssertionError('No subprocess')),patch('socket.socket',side_effect=AssertionError('No network')):
                preparation.validate(root/'PREPARED_MANIFEST.json',Path(tmp)/'v')
                preparation.describe(root/'PREPARED_MANIFEST.json',Path(tmp)/'d')
                from inference_kit.cli import main
                with redirect_stdout(io.StringIO()) as stream,self.assertRaises(SystemExit) as stopped:main(['bench','inputs','--help'])
                self.assertEqual(stopped.exception.code,0);self.assertIn('prepare',stream.getvalue())
            self.assertEqual(blocked,[])

    def test_explicit_tokenizer_file_verification_detects_mutation_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tokenizer=root/'tokenizer';tokenizer.mkdir()
            for name in ('tokenizer.json','tokenizer_config.json'):(tokenizer/name).write_text('{}',encoding='utf-8')
            write(root/'config.json',{'model_type':'llama'})
            pins={p.name:sha(p) for p in tokenizer.iterdir()}
            value={'schema_version':1,'kind':'local_tokenizer_binding_v1','runtime_python':sys.executable,'package_site':str(root/'site'),
                   'tokenizer_path':str(tokenizer),'config_path':str(root/'config.json'),'revision':'31b70e2e869a7173562077fd711b654946d38674','model_id':'HuggingFaceTB/SmolLM2-1.7B-Instruct',
                   'tokenizer_files':pins,'config_sha256':sha(root/'config.json'),'toolkit_source_identity':'a'*64,
                   'model_metadata':{'tokenizer_identity':digest(pins),'tokenizer_files':pins,'model_config':{'model_type':'llama'}}}
            preparation.validate_binding(value,verify_package=False)
            (tokenizer/'tokenizer.json').write_text('{"changed":true}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'pin changed'):preparation.validate_binding(value,verify_package=False)

    def test_prepare_worker_failure_preserves_unpublished_failure_only(self):
        raw,spec,_,_=worker_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'raw.json',raw);write(root/'spec.json',spec);write(root/'binding.json',{'fixture':'synthetic'})
            binding={'runtime_python':sys.executable,'package_site':str(root/'site')}
            with patch.object(preparation,'validate_binding',return_value=binding),patch.object(preparation.subprocess,'run',return_value=subprocess.CompletedProcess([],2,b'',b'SYNTHETIC_FAILURE')) as dispatch:
                with self.assertRaisesRegex(ValueError,'PREPARATION_FAILED'):preparation.prepare(root/'spec.json',root/'raw.json',root/'binding.json',root/'prepared')
            self.assertEqual(dispatch.call_count,1);self.assertFalse((root/'prepared').exists())
            stages=list(root.glob('.prepared.preparing-*'));self.assertEqual(len(stages),1)
            self.assertFalse((stages[0]/'PREPARED_MANIFEST.json').exists())
            self.assertFalse(read(stages[0]/'FAILURE.json')['success_manifest_published'])
            self.assertEqual((stages[0]/'WORKER_STDERR.log').read_bytes(),b'SYNTHETIC_FAILURE')

    def test_missing_sdk_is_explicit_and_unpublished(self):
        raw,spec,_,_=worker_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'raw.json',raw);write(root/'spec.json',spec);write(root/'binding.json',{})
            binding={'runtime_python':sys.executable,'package_site':str(root/'site')}
            failure=subprocess.CompletedProcess([],1,b'',b'MISSING_LOCAL_TOKENIZER_RUNTIME: tokenizer SDK unavailable')
            with patch.object(preparation,'validate_binding',return_value=binding),patch.object(preparation.subprocess,'run',return_value=failure):
                with self.assertRaisesRegex(ValueError,'MISSING_LOCAL_TOKENIZER_RUNTIME'):
                    preparation.prepare(root/'spec.json',root/'raw.json',root/'binding.json',root/'prepared')
            self.assertFalse((root/'prepared').exists())

    def test_preparation_output_refuses_existing_directory_before_subprocess(self):
        raw,spec,_,_=worker_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'raw.json',raw);write(root/'spec.json',spec);write(root/'binding.json',{});(root/'existing').mkdir()
            with patch.object(preparation,'validate_binding',return_value={}),patch.object(preparation.subprocess,'run',side_effect=AssertionError('Unexpected subprocess')):
                with self.assertRaisesRegex(ValueError,'existing'):preparation.prepare(root/'spec.json',root/'raw.json',root/'binding.json',root/'existing')

    def test_parent_input_bytes_cannot_change_during_mock_worker(self):
        raw,spec,_,_=worker_fixture()
        for changed in ('raw.json','spec.json','binding.json'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);write(root/'raw.json',raw);write(root/'spec.json',spec);write(root/'binding.json',{})
                binding={'runtime_python':sys.executable,'package_site':str(root/'site')}
                def mock_worker(*args,**kwargs):
                    # Same JSON semantics but changed immutable input bytes.
                    path=root/changed;path.write_bytes(path.read_bytes()+b' ')
                    write(Path(kwargs['cwd'])/'SOURCES.json',{'synthetic_unpublished_candidate':True})
                    return subprocess.CompletedProcess([],0,b'',b'')
                with patch.object(preparation,'validate_binding',return_value=binding),patch.object(preparation.subprocess,'run',side_effect=mock_worker) as dispatch:
                    with self.assertRaisesRegex(ValueError,'Inputs changed'):preparation.prepare(root/'spec.json',root/'raw.json',root/'binding.json',root/'prepared')
                self.assertEqual(dispatch.call_count,1);self.assertFalse((root/'prepared').exists())
                stage=next(root.glob('.prepared.preparing-*'))
                self.assertFalse((stage/'PREPARED_MANIFEST.json').exists());self.assertFalse((stage/'SOURCES.json').exists())
                self.assertTrue((stage/'UNPUBLISHED_SOURCES.json').is_file())


if __name__=='__main__':unittest.main()
