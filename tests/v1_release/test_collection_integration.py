"""Frozen policy wiring over real CPU scorers, index and report operations.

All input observations in this fixture are SYNTHETIC_CONTRACT_TEST. No model,
tokenizer, inference client or GPU inspection is involved.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_kit import api
from inference_kit.analysis import suite_postprocess as cpu, suite_results
from inference_kit.core.io import digest, read, sha
from inference_kit.suites import suite_plan
from bench05e2.test_plan import sources_fixture, suite_fixture, local_fixture
from bench05e3.test_results import fixture, view_fixture


POLICY = {'schema_version': 1, 'mode': 'bounded_collection_v1'}


class ReportCollectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def raw(self):
        parent = fixture(self.root/'parent')
        view = view_fixture(self.root/'view', read(parent/'suite-execution-context.json'))
        raw = self.root/'raw'
        suite_results.export_mini_slice(parent, view, raw)
        return raw

    def test_suite_policy_is_frozen_without_changing_task_or_observation_counts(self):
        sources = sources_fixture(self.root)
        before = suite_plan.compile_suite(suite_fixture(), sources, local_fixture())
        spec = suite_fixture(); spec['outputs']['report_packaging'] = deepcopy(POLICY)
        after = suite_plan.compile_suite(spec, sources, local_fixture())
        self.assertEqual(before['semantic']['counts'], after['semantic']['counts'])
        self.assertNotEqual(before['plan_identity'], after['plan_identity'])
        policy = after['semantic']['suite']['outputs']['report_packaging']
        report = next(n for n in after['semantic']['graph']['nodes'] if n['operation_kind'] == 'report')
        self.assertEqual(report['output_contract']['report_packaging'], policy)
        self.assertEqual(policy['mode'], 'bounded_collection_v1')
        self.assertEqual(policy['max_volumes'], 2)
        self.assertEqual(spec['outputs']['report_packaging'], POLICY)
        suite_plan.check_frozen(after)

    def test_unbounded_policy_or_policy_without_report_is_rejected(self):
        for change in ({'max_volumes': 3}, {'volume_payload_bytes': True}, {'unexpected': 1}):
            with self.subTest(change=change):
                spec = suite_fixture(); spec['outputs']['report_packaging'] = POLICY | change
                with self.assertRaises(ValueError):
                    suite_plan.validate_spec(spec)
        spec = suite_fixture(); spec['outputs'].update(report=False, report_packaging=POLICY)
        with self.assertRaisesRegex(ValueError, 'requires report'):
            suite_plan.validate_spec(spec)

    def test_cpu_boundary_resume_noop_validates_collection_and_public_cli_after_relocation(self):
        raw = self.raw(); workspace = self.root/'analysis'
        first = api.suite_postprocess(raw, workspace, report_packaging=POLICY, stop_after='index_add')
        self.assertEqual(first['status'], 'CPU_PAUSED_BOUNDARY')
        semantic = read(workspace/'ANALYSIS_PLAN.json')['semantic']
        self.assertEqual(semantic['report_packaging']['mode'], 'bounded_collection_v1')
        self.assertEqual(read(workspace/'ANALYSIS_PLAN.json')['analysis_plan_identity'], digest(semantic))
        done = api.suite_postprocess_resume(workspace)
        self.assertEqual(done['status'], 'CPU_POSTPROCESS_COMPLETE')
        report = workspace/done['nodes']['report']['output']
        checked = api.index_report_validate(report)
        self.assertEqual(checked['status'], 'REPORT_COLLECTION_VERIFIED')
        self.assertEqual(checked['packaging'], semantic['report_packaging'])
        before = cpu._inventory(workspace, workspace/'stages')
        checkpoint = sha(workspace/'checkpoint.json')
        with patch.object(cpu, 'execute_cpu', side_effect=AssertionError('Completed CPU revision must not dispatch')):
            noop = api.suite_postprocess_resume(workspace)
            self.assertTrue(noop['noop'])
            self.assertEqual(api.suite_postprocess_status(workspace)['status'], 'CPU_POSTPROCESS_COMPLETE')
        self.assertEqual(sha(workspace/'checkpoint.json'), checkpoint)
        self.assertEqual(cpu._inventory(workspace, workspace/'stages'), before)
        relocated = self.root/'이동 경로'/'report collection'
        shutil.copytree(report, relocated)
        self.assertEqual(api.index_report_validate(relocated), checked)
        process = subprocess.run([sys.executable, '-B', '-m', 'inference_kit', 'bench', 'index',
            'validate-report', '--report', str(relocated), '--output', str(self.root/'cli-check')],
            text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(read(self.root/'cli-check/REPORT_COLLECTION_VALIDATION.json')['status'],
                         'REPORT_COLLECTION_VERIFIED')
        member = next(p for p in report.rglob('*') if p.is_file() and 'evidence' in p.relative_to(report).parts
                      and p.name != 'PART_MANIFEST.json')
        member.write_bytes(member.read_bytes() + b' ')
        with patch.object(cpu, 'execute_cpu') as dispatch:
            with self.assertRaises(ValueError):
                api.suite_postprocess_resume(workspace)
            dispatch.assert_not_called()
        self.assertEqual(sha(workspace/'checkpoint.json'), checkpoint)
        with self.assertRaises(ValueError):
            api.index_report_validate(report)


if __name__ == '__main__':
    unittest.main()
