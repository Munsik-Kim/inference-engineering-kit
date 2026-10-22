"""Closed CPU executor state tests; fake acquisition is never live evidence."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import digest, read, sha, write
from inference_kit.core.state import atomic
from inference_kit.suites import suite_execution as execute


def fixture(root):
    """Tiny immutable executor input plus a valid synthetic campaign ledger."""
    root.mkdir(); (root/'prepared').mkdir()
    write(root/'prepared/input.json', {'evidence_kind':'SYNTHETIC_EXECUTOR_TEST','sample_id':'sample','token_ids':[1,2]})
    write(root/'parent-ledger.json', {'used_seconds':100.,'campaign_cap_seconds':7200})
    write(root/'local.json', {'evidence_kind':'SYNTHETIC_EXECUTOR_TEST','authorization':'test-only'})
    campaign={'path':str(root/'parent-ledger.json'),'sha256':sha(root/'parent-ledger.json'),'used_seconds':100.}
    context={'evidence_kind':'SYNTHETIC_EXECUTOR_TEST','campaign':campaign,'local_binding_sha256':sha(root/'local.json')}
    write(root/'suite-execution-context.json',context)
    ledger={'schema_version':1,'task_id':'BENCH05-E3','campaign_cap_seconds':7200,'task_origin_used_seconds':100.,
            'cap_seconds':700.,'used_seconds':100.,'running':None,'jobs':[],
            'inherited_from':campaign}
    write(root/'budget-continuation.json',ledger)
    units=[{'node_id':'validate','operation_kind':'validate_prepared_inputs','dependencies':[],'required':True,'counts':{'documents':1}},
           {'node_id':'serve-bf16','operation_kind':'serve_group','dependencies':['validate'],'required':True,'counts':{'inference_POSTs':1}},
           {'node_id':'serve-rtn','operation_kind':'serve_group','dependencies':['serve-bf16'],'required':True,'counts':{'inference_POSTs':1}},
           {'node_id':'score_task','operation_kind':'score_task','dependencies':['serve-rtn'],'required':True,'counts':{'documents':1}}]
    immutable={name:{'sha256':sha(root/name),'bytes':(root/name).stat().st_size}for name in ('prepared/input.json','suite-execution-context.json')}
    plan={'schema_version':1,'kind':'suite_execution_plan_v1','executor_source':execute.source_identity(),
          'units':units,'immutable_inputs':immutable,'evidence_kind':'SYNTHETIC_EXECUTOR_TEST'}
    plan['execution_plan_identity']=digest(plan);write(root/'EXECUTION_PLAN.json',plan)
    state={'schema_version':1,'run_id':'synthetic-executor','execution_plan_identity':plan['execution_plan_identity'],'status':'PENDING',
           'active_controller_seconds':0.,'paused_seconds':0.,'pause_started_monotonic':None,'controller_sessions':0,
           'nodes':{u['node_id']:{'status':'PENDING','attempts':0}for u in units}}
    write(root/'checkpoint.json',state)
    return root


def fake_dispatch(calls, fail_cpu_once=False, invalid_cleanup=False):
    def dispatch(root,context,node,state,allow_gpu,output):
        calls.append(node['node_id'])
        if node['node_id']=='score_task' and fail_cpu_once and calls.count('score_task')==1:
            raise RuntimeError('synthetic CPU analysis interruption')
        output.mkdir(parents=True,exist_ok=True)
        write(output/'EVIDENCE.json',{'evidence_kind':'SYNTHETIC_EXECUTOR_TEST','node_id':node['node_id'],'attempt':state['nodes'][node['node_id']]['attempts']})
        if node['operation_kind']in execute.GPU_KINDS:
            ledger=read(root/'budget-continuation.json')
            ledger['jobs'].append({'id':node['node_id']+'-attempt1','wall_seconds':1.})
            ledger['used_seconds']+=1.;atomic(root/'budget-continuation.json',ledger)
            return {'status':'COMPLETE','owned_cleanup_verified':not invalid_cleanup,'counts':deepcopy(node['counts']),'evidence_kind':'SYNTHETIC_EXECUTOR_TEST'},output
        return {'status':'PASS','evidence_kind':'SYNTHETIC_EXECUTOR_TEST'},output
    return dispatch


class SuiteExecutionTests(unittest.TestCase):
    def make(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        return fixture(Path(temp.name)/'run')

    def drive(self,root,calls,**kwargs):
        with patch.object(execute,'_dispatch',side_effect=fake_dispatch(calls)):
            return execute._drive(root,root/'local.json',True,**kwargs)

    def test_pause_resume_and_completed_noop_never_resend(self):
        root=self.make();calls=[]
        first=self.drive(root,calls,stop_after_node='serve-bf16')
        self.assertEqual(first['status'],'PAUSED_BOUNDARY');self.assertEqual(calls,['validate','serve-bf16'])
        self.assertEqual(self.drive(root,calls)['status'],'COMPLETE_VALIDATED')
        before=sha(root/'budget-continuation.json')
        with patch.object(execute,'_dispatch',side_effect=AssertionError('no-op dispatch')):
            noop=execute.resume(root,allow_gpu=False)
        self.assertTrue(noop['noop']);self.assertEqual(noop['new_GPU_jobs'],0)
        self.assertEqual(sha(root/'budget-continuation.json'),before)
        self.assertEqual(calls,['validate','serve-bf16','serve-rtn','score_task'])

    def test_unauthorized_pending_never_dispatches(self):
        root=self.make();before=sha(root/'budget-continuation.json')
        with patch.object(execute,'_dispatch')as dispatch,self.assertRaisesRegex(ValueError,'allow-gpu'):
            execute.resume(root,allow_gpu=False)
        dispatch.assert_not_called();self.assertEqual(sha(root/'budget-continuation.json'),before)

    def test_missing_completion_or_tampered_raw_cannot_be_complete(self):
        for alteration in ('missing_completion','missing_raw','changed_raw'):
            with self.subTest(alteration=alteration):
                root=self.make();self.drive(root,[])
                if alteration=='missing_completion':(root/'nodes/serve-bf16/COMPLETION.json').unlink()
                elif alteration=='missing_raw':(root/'nodes/serve-bf16/attempt1/EVIDENCE.json').unlink()
                else:(root/'nodes/serve-bf16/attempt1/EVIDENCE.json').write_text('{}',encoding='utf-8')
                with patch.object(execute,'_dispatch')as dispatch,self.assertRaisesRegex(ValueError,'MISSING_EVIDENCE|STALE_EVIDENCE'):
                    execute.resume(root,allow_gpu=False)
                dispatch.assert_not_called()

    def test_commit_checkpoint_gap_reconciles_evidence_directory_once(self):
        root=self.make();calls=[];self.drive(root,calls,stop_after_node='serve-bf16')
        state=read(root/'checkpoint.json');state['nodes']['serve-bf16']={'status':'RUNNING','attempts':1};atomic(root/'checkpoint.json',state)
        self.assertEqual(self.drive(root,calls)['status'],'COMPLETE_VALIDATED')
        self.assertEqual(calls.count('serve-bf16'),1)
        self.assertEqual(read(root/'checkpoint.json')['nodes']['serve-bf16']['output'],'nodes/serve-bf16/attempt1')

    def test_cpu_failure_derives_new_attempt_without_gpu_retry(self):
        root=self.make();calls=[]
        with patch.object(execute,'_dispatch',side_effect=fake_dispatch(calls,fail_cpu_once=True)),self.assertRaisesRegex(RuntimeError,'CPU analysis'):
            execute._drive(root,root/'local.json',True)
        before=sha(root/'budget-continuation.json')
        with patch.object(execute,'_dispatch',side_effect=fake_dispatch(calls)):
            result=execute.resume(root,allow_gpu=False)
        self.assertEqual(result['status'],'COMPLETE_VALIDATED');self.assertEqual(calls.count('score_task'),2)
        self.assertEqual(calls.count('serve-bf16'),1);self.assertEqual(calls.count('serve-rtn'),1)
        self.assertEqual(sha(root/'budget-continuation.json'),before)

    def test_running_or_failed_gpu_without_commit_never_resubmits(self):
        for status in ('RUNNING','FAILED'):
            with self.subTest(status=status):
                root=self.make();state=read(root/'checkpoint.json');state['nodes']['serve-bf16'].update(status=status,attempts=1);atomic(root/'checkpoint.json',state)
                with patch.object(execute,'_dispatch')as dispatch,self.assertRaisesRegex(ValueError,'automatic retry forbidden'):
                    execute._drive(root,root/'local.json',True)
                dispatch.assert_not_called()

    def test_lock_rejects_concurrent_owner_and_releases_on_exception(self):
        root=self.make()
        with self.assertRaisesRegex(RuntimeError,'test exit'):
            with execute._lock(root):
                with self.assertRaisesRegex(ValueError,'RUN_LOCKED'):
                    with execute._lock(root):pass
                raise RuntimeError('test exit')
        with execute._lock(root):pass

    def test_ledger_tamper_rejected_before_dispatch_and_noop(self):
        changes=(lambda v:v.update(cap_seconds=701.),lambda v:v.update(task_origin_used_seconds=99.),
                 lambda v:v['inherited_from'].update(sha256='bad'),lambda v:v.update(used_seconds=999.),
                 lambda v:v['jobs'][0].update(wall_seconds=float('inf')))
        for alter in changes:
            with self.subTest(alter=alter):
                root=self.make();self.drive(root,[]);ledger=read(root/'budget-continuation.json');alter(ledger)
                # core.io forbids nonfinite JSON: write the explicit invalid wire for this negative case.
                if any(not isinstance(j.get('wall_seconds'),(int,float)) or j['wall_seconds']==float('inf')for j in ledger['jobs']):
                    import json
                    (root/'budget-continuation.json').write_text(json.dumps(ledger),encoding='utf-8')
                else:atomic(root/'budget-continuation.json',ledger)
                with patch.object(execute,'_dispatch')as dispatch,self.assertRaises(ValueError):execute.resume(root,allow_gpu=False)
                dispatch.assert_not_called()
        root=self.make();(root/'parent-ledger.json').write_text('{}',encoding='utf-8')
        with patch.object(execute,'_dispatch')as dispatch,self.assertRaisesRegex(ValueError,'Prior campaign ledger changed'):
            execute._drive(root,root/'local.json',True)
        dispatch.assert_not_called()

    def test_source_plan_and_immutable_input_tamper_rejected(self):
        root=self.make()
        with patch.object(execute,'source_identity',return_value='changed'),self.assertRaisesRegex(ValueError,'Executor source changed'):
            execute._frozen(root)
        plan=read(root/'EXECUTION_PLAN.json');plan['units'][0]['counts']['documents']=2;atomic(root/'EXECUTION_PLAN.json',plan)
        with self.assertRaisesRegex(ValueError,'Execution plan changed'):execute._frozen(root)
        root=self.make();(root/'prepared/input.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'STALE_EVIDENCE'):execute._frozen(root)

    def test_empty_units_or_zero_required_coverage_never_completes(self):
        for mutate in (lambda v:v.update(units=[]),lambda v:v['units'][0].update(counts={'documents':0})):
            with self.subTest(mutate=mutate):
                root=self.make();plan=read(root/'EXECUTION_PLAN.json');plan.pop('execution_plan_identity');mutate(plan);plan['execution_plan_identity']=digest(plan);atomic(root/'EXECUTION_PLAN.json',plan)
                with patch.object(execute,'_dispatch')as dispatch,self.assertRaisesRegex(ValueError,'INVALID_EMPTY_PLAN|EMPTY_REQUIRED_OUTPUT'):
                    execute._drive(root,root/'local.json',True)
                dispatch.assert_not_called()

    def test_unverified_owned_cleanup_blocks_commit_and_automatic_retry(self):
        root=self.make();calls=[]
        with patch.object(execute,'_dispatch',side_effect=fake_dispatch(calls,invalid_cleanup=True)),self.assertRaisesRegex(ValueError,'cleanup not verified'):
            execute._drive(root,root/'local.json',True)
        with patch.object(execute,'_dispatch')as dispatch,self.assertRaisesRegex(ValueError,'automatic retry forbidden'):
            execute._drive(root,root/'local.json',True)
        dispatch.assert_not_called();self.assertEqual(calls.count('serve-bf16'),1)

    def test_base_executor_import_needs_no_ml_sdk(self):
        code="import sys;from inference_kit.suites import suite_execution;assert not any(x in sys.modules for x in ('torch','transformers','vllm','numpy','aiohttp','compressed_tensors'))"
        subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True)
