"""Real locale failures and runner evidence gates, without a model or SDK."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import inference_kit


class EncodingTests(unittest.TestCase):
    def child(self, body, root):
        # Test exactly the package already selected by the parent (wheel or src).
        package_root = str(Path(inference_kit.__file__).absolute().parent.parent)
        code = ('import sys,locale;sys.path.insert(0,' + repr(package_root) + ');'
                'assert locale.getpreferredencoding(False).lower() not in ("utf-8","utf8");' + body)
        env = dict(os.environ, LC_ALL='C', LANG='C', PYTHONDONTWRITEBYTECODE='1')
        return subprocess.run([sys.executable, '-I', '-X', 'utf8=0', '-B', '-c', code],
                              cwd=root, env=env, capture_output=True, encoding='utf-8')

    def test_utf8_json_read_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_bytes(json.dumps({'text': '\ud55c\uae00'}, ensure_ascii=False).encode('utf-8'))
            result = self.child('from pathlib import Path;from inference_kit.core.io import read;'
                'assert read(Path("input.json"))["text"]==chr(0xd55c)+chr(0xae00)', directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_utf8_json_write_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.child('from pathlib import Path;from inference_kit.core.io import write;'
                'write(Path("output.json"),{"text":chr(0xd55c)+chr(0xae00)})', directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(directory) / 'output.json').read_bytes()), {'text': '\ud55c\uae00'})

    def test_public_jsonl_compare_under_ascii_locale(self):
        from importlib.resources import files
        fixture = json.loads(files('inference_kit.resources').joinpath('demo.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'records.jsonl'
            rows = fixture['baseline']
            for index, row in enumerate(rows):
                row['sample_id'] = '\ubb38\ud56d-' + str(index)
            path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows), encoding='utf-8')
            result = self.child('from inference_kit import compare;'
                'assert isinstance(compare("records.jsonl","records.jsonl",replicates=20),dict)', directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_json_output_under_ascii_locale(self):
        from importlib.resources import files
        fixture = json.loads(files('inference_kit.resources').joinpath('demo.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            for index, row in enumerate(fixture['baseline']):
                row['sample_id'] = '\ubb38\ud56d-' + str(index)
            (Path(directory) / 'records.json').write_text(
                json.dumps(fixture['baseline'], ensure_ascii=False), encoding='utf-8')
            result = self.child('from inference_kit.cli import main;'
                'raise SystemExit(main(["compare","--baseline","records.json",'
                '"--candidate","records.json","--replicates","20","--output","comparison"]))', directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsInstance(json.loads(result.stdout), dict)
            self.assertTrue((Path(directory) / 'comparison').is_dir())

    def test_quality_and_recipe_jsonl_readers_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'raw.jsonl').write_bytes(
                json.dumps({'sample_id': '\ud55c\uae00'}, ensure_ascii=False).encode('utf-8') + b'\n')
            result = self.child('from inference_kit.core.quality import jsonl;'
                'from inference_kit.experiments.step04 import jsonl as recipe_jsonl;'
                'assert jsonl("raw.jsonl")==recipe_jsonl("raw.jsonl");'
                'assert jsonl("raw.jsonl")[0]["sample_id"]==chr(0xd55c)+chr(0xae00)', directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_bench_raw_replay_and_korean_report_under_ascii_locale(self):
        from bench05a.helpers import synthetic_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_run(root / 'raw')
            result = self.child('from inference_kit import bench_replay;'
                'r=bench_replay("raw","replay");'
                'assert r["counts"]["paired_token_positions"]==8', directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (root / 'replay/REPORT_KO.md').read_text(encoding='utf-8')
            self.assertIn('\uc7ac\uacc4\uc0b0', report)


class RunnerEvidenceTests(unittest.TestCase):
    def test_empty_discovery_fails_and_records_actual_ci_context(self):
        tools_root = Path(os.environ.get('RELEASE_TOOLS_ROOT', Path(__file__).parents[1]))
        runner = tools_root / 'tools/test_cpu.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'tests').mkdir()
            output = root / 'receipt.json'
            env = dict(os.environ, GITHUB_ACTIONS='true', GITHUB_RUN_ID='synthetic-context-only', PYTHONDONTWRITEBYTECODE='1')
            package_root = str(Path(inference_kit.__file__).absolute().parent.parent)
            code = ('import sys,runpy;sys.path.insert(0,' + repr(package_root) + ');'
                    'sys.argv=' + repr([str(runner), '--output', str(output)]) + ';'
                    'runpy.run_path(' + repr(str(runner)) + ',run_name="__main__")')
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(receipt['test_invocations'], 0)
            self.assertFalse(receipt['successful'])
            self.assertEqual(receipt['collection_status'], 'NO_TESTS_COLLECTED')
            self.assertEqual(receipt['remote_ci'], 'GITHUB_ACTIONS_CPU_RUN')
            self.assertEqual(receipt['github_run_id'], 'synthetic-context-only')
            self.assertTrue(receipt['package_import_origin'].endswith('inference_kit/__init__.py'))
