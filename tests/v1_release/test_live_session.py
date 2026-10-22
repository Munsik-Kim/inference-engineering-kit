"""Socket-free tests of the shipped explicit session helper, not live execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import read, write

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/release_live_session.py'
SPEC = importlib.util.spec_from_file_location('reviewed_release_live_session', SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class SessionHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = self.root/'session.json'
        self.clock = self.root/'clock.json'

    def start(self):
        helper.start_session(self.session)
        helper.make_clock(self.session, self.clock)

    def test_clock_native_bounds_and_no_overwrite(self):
        self.start(); value = helper.check_clock(self.session, self.clock)
        self.assertEqual(value['deadline_monotonic'], value['start_monotonic']+3600)
        self.assertEqual(value['no_new_gpu_after'], value['start_monotonic']+3000)
        before = self.clock.read_bytes()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            helper.make_clock(self.session, self.clock)
        self.assertEqual(self.clock.read_bytes(), before)

    def test_earlier_utc_origin_preserves_outer_deadline(self):
        import datetime
        origin = datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=120)
        value = helper.start_session(self.session, origin.isoformat())
        self.assertEqual(value['start_utc'], origin.isoformat())
        self.assertEqual(value['monotonic_origin'], 'UTC_DELTA_MAPPING_AT_RECEIPT_CREATION')
        self.assertAlmostEqual(value['observed_monotonic']-value['start_monotonic'], 120, delta=1)
        helper.make_clock(self.session, self.clock)
        self.assertEqual(read(self.clock)['enclosing_deadline_monotonic'], value['deadline_monotonic'])

    def test_suite_is_derived_from_all_prepared_records(self):
        repo = SCRIPT.parents[1]
        for family, name in [('qwen','qwen-task.suite.json'),('smol','smol-live-combined.suite.template.json')]:
            with self.subTest(family=family):
                result = helper.make_suite(repo/'release-kit/examples'/name,
                    repo/'release-kit/fixtures/prepared'/family/'SOURCES.json', self.root/(family+'-suite.json'))
                self.assertEqual(result['status'], 'SUITE_SPEC_DERIVED_NOT_EXECUTED')
                value = read(self.root/(family+'-suite.json'))
                self.assertEqual(sum(len(t['sample_ids']) for t in value['tasks']), result['counts']['documents'])
                self.assertGreater(result['counts']['inference_POSTs'], 0)
                self.assertEqual(value['budget']['max_output_tokens'], result['counts']['output_token_upper'])
                expected_policy = read(repo/'release-kit/examples/REPORT_PACKAGING.json')
                self.assertEqual(value['outputs']['report_packaging'], expected_policy)
                self.assertEqual(result['report_packaging'], expected_policy)

    def test_changed_identity_is_rejected(self):
        self.start(); value = read(self.clock); value['start_monotonic'] += 1
        self.clock.unlink(); write(self.clock, value)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            helper.check_clock(self.session, self.clock)

    def test_foreign_host_or_boot_is_rejected(self):
        self.start()
        with patch.object(helper, '_host_boot', return_value={'host_identity':'other', 'boot_identity':'other'}):
            with self.assertRaisesRegex(ValueError, 'another host/boot'):
                helper.check_clock(self.session, self.clock)

    def test_outer_session_expiry_cannot_be_extended(self):
        self.start(); value = read(self.session)
        with patch.object(helper.time, 'monotonic', return_value=value['no_new_gpu_after']):
            with self.assertRaisesRegex(ValueError, 'SESSION_EXPIRED'):
                helper.make_clock(self.session, self.root/'late-clock.json')
        self.assertFalse((self.root/'late-clock.json').exists())

    def test_source_change_is_rejected(self):
        self.start()
        with patch.object(helper, 'source_identity', return_value='other'):
            with self.assertRaisesRegex(ValueError, 'package source changed'):
                helper.check_clock(self.session, self.clock)

    def test_cpu_local_and_placeholder_are_rejected_without_output(self):
        self.start(); local = self.root/'local.json'; prepared = self.root/'PREPARED_MANIFEST.json'
        write(prepared, {'not': 'full inventory; product binder must validate'})
        write(local, {'artifact_presence': 'NOT_REQUESTED'})
        with self.assertRaisesRegex(ValueError, 'CPU_LOCAL'):
            helper.resolve_local(local, prepared, self.session, self.clock, self.root/'resolved.json')
        self.assertFalse((self.root/'resolved.json').exists())
        local.unlink(); write(local, {'profile':'<PLACEHOLDER>'})
        with self.assertRaisesRegex(ValueError, 'UNRESOLVED'):
            helper.resolve_local(local, prepared, self.session, self.clock, self.root/'resolved.json')

    def test_first_serving_unit_comes_from_actual_order(self):
        binding = self.root/'binding.json'
        write(binding, {'status':'CPU_BOUND_NOT_EXECUTED','execution_units':[
            {'node_id':'validate_inputs','operation_kind':'validate_prepared_inputs'},
            {'node_id':'unit-not-a-historical-name','operation_kind':'serve_group','required':True,'counts':{'inference_POSTs':7}},
            {'node_id':'another','operation_kind':'serve_group','required':True,'counts':{'inference_POSTs':2}}]})
        self.assertEqual(helper.first_serving_unit(binding), 'unit-not-a-historical-name')

    def test_metadata_only_empty_duplicate_units_are_rejected(self):
        for index, value in enumerate([
            {'status':'CONTRACT_VALIDATED_ONLY','execution_units':[]},
            {'status':'CPU_BOUND_NOT_EXECUTED','execution_units':[]},
            {'status':'CPU_BOUND_NOT_EXECUTED','execution_units':[
                {'node_id':'same','operation_kind':'serve_group','required':True,'counts':{'inference_POSTs':1}},
                {'node_id':'same','operation_kind':'serve_group','required':True,'counts':{'inference_POSTs':1}}]},
            {'status':'CPU_BOUND_NOT_EXECUTED','execution_units':[
                {'node_id':'empty','operation_kind':'serve_group','required':True,'counts':{'inference_POSTs':0}}]}
        ]):
            path=self.root/('bad-'+str(index)+'.json'); write(path, value)
            with self.subTest(case=index), self.assertRaises(ValueError): helper.first_serving_unit(path)


if __name__ == '__main__': unittest.main()
