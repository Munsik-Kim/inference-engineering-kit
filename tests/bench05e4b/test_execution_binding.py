"""CPU-only variable-unit executor controls; no live acquisition is simulated as measured."""
from copy import deepcopy
import errno
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inference_kit.core.io import digest,read,sha,write
from inference_kit.core.state import atomic
from inference_kit.suites import suite_execution as execute, suite_live
from inference_kit.suites.prepared_live import binding_layout
from bench05e3.test_execution import fixture as original_fixture
from bench05e4b.test_binding import fixture as inputs_fixture


def executor_fixture(root):
    original_fixture(root);context=read(root/'suite-execution-context.json')
    names=['serve-bf16-task0','serve-rtn-task0','serve-bf16-task1','serve-rtn-task1','serve-bf16-task2','serve-rtn-task2','token-bf16','token-rtn']
    job_ids=[n+'-attempt1' for n in names[:6]]+['main-bf16-attempt1','main-rtn-attempt1']
    context.update(binding_profile='prepared_smol_live_v1',ledger_task_id='BENCH05-E4B',expected_gpu_jobs=len(names),
                   allowed_ledger_job_ids=job_ids,execution_unit_ids=names)
    atomic(root/'suite-execution-context.json',context)
    ledger=read(root/'budget-continuation.json');ledger['task_id']='BENCH05-E4B';atomic(root/'budget-continuation.json',ledger)
    units=[{'node_id':'validate','operation_kind':'validate_prepared_inputs','dependencies':[],'required':True,'counts':{'documents':9}}]
    for n in names:
        kind='serve_group' if n.startswith('serve') else 'token_group'
        counts={'inference_POSTs':3} if kind=='serve_group' else {'main_target_observations':5,'main_full_observations':1,'forward_calls':3}
        units.append({'node_id':n,'operation_kind':kind,'dependencies':['validate'],'required':True,'counts':counts})
    units.append({'node_id':'report','operation_kind':'report','dependencies':names,'required':True,'counts':{'documents':9}})
    plan=read(root/'EXECUTION_PLAN.json');plan.pop('execution_plan_identity');plan['units']=units
    # This synthetic dispatcher has no renderer, but its report operation must
    # still bind immutable intent. Real runs always freeze this Suite input;
    # absence must not become a permissive legacy fallback in product code.
    write(root/'SUITE_PLAN.json',{'kind':'synthetic_executor_suite_intent_v1',
        'evidence_kind':'SYNTHETIC_EXECUTOR_TEST','semantic':{'suite':{'outputs':{
            'index':True,'report':True,'historical_source_ids':[],
            'report_packaging':{'schema_version':1,'mode':'single_report_v1'}}}}})
    plan['immutable_inputs']['SUITE_PLAN.json']={}
    for n in plan['immutable_inputs']:
        plan['immutable_inputs'][n]={'bytes':(root/n).stat().st_size,'sha256':sha(root/n)}
    plan['execution_plan_identity']=digest(plan);atomic(root/'EXECUTION_PLAN.json',plan)
    state=read(root/'checkpoint.json');state.update(execution_plan_identity=plan['execution_plan_identity'],nodes={n['node_id']:{'status':'PENDING','attempts':0} for n in units});atomic(root/'checkpoint.json',state)
    return names


def synthetic_dispatch(calls):
    def handler(root,context,node,state,allow_gpu,output):
        calls.append(node['node_id']);output.mkdir(parents=True,exist_ok=True)
        write(output/'EVIDENCE.json',{'evidence_kind':'SYNTHETIC_EXECUTOR_TEST','unit':node['node_id']})
        if node['operation_kind'] in execute.GPU_KINDS:
            name=node['node_id'];job=('main-'+name[len('token-'):] if name.startswith('token-') else name)+'-attempt1'
            ledger=read(root/'budget-continuation.json');ledger['jobs'].append({'id':job,'wall_seconds':1.});ledger['used_seconds']+=1.;atomic(root/'budget-continuation.json',ledger)
            return {'status':'COMPLETE','owned_cleanup_verified':True,'counts':deepcopy(node['counts']),'evidence_kind':'SYNTHETIC_EXECUTOR_TEST'},output
        return {'status':'PASS','evidence_kind':'SYNTHETIC_EXECUTOR_TEST'},output
    return handler


class PreparedExecutionTests(unittest.TestCase):
    def test_exact_eight_units_cover_typed_nodes_once_and_missing_unit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,loaded=inputs_fixture(Path(tmp),3);layout=binding_layout(plan,loaded)
            context={'binding_profile':'prepared_smol_live_v1','serving_groups':layout['serving_groups'],
              'runtime_plan':{'arms':loaded['model_metadata']['arms']},'execution_unit_ids':[n['node_id'] for n in layout['execution_units']], 'expected_gpu_jobs':len(layout['execution_units'])}
            units=execute.execution_units(plan,context)
            self.assertEqual(sum(n['operation_kind'] in execute.GPU_KINDS for n in units),8)
            self.assertEqual(len({p for n in units for p in n['provides']}),len(plan['semantic']['graph']['nodes']))
            context['serving_groups'].pop()
            with self.assertRaisesRegex(ValueError,'coverage'):execute.execution_units(plan,context)

    def test_permission_denial_precedes_gpu_query_and_ledger_change(self):
        context={'binding_profile':'prepared_smol_live_v1','runtime_plan':{'port':19001,'suite':{'budget':{}}}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite_live,'resources',return_value={'ready':True}), \
             patch.object(suite_live,'verify_live_context',return_value=True), \
             patch.object(suite_live.socket,'socket',side_effect=PermissionError(errno.EPERM,'fixture denied')), \
             patch('inference_kit.suites.bench05a_run.check_gpu',side_effect=AssertionError('GPU queried')):
            gate=suite_live.preflight(context,tmp)
            self.assertEqual(gate['status'],'BLOCKED_LOOPBACK');self.assertEqual(gate['stage'],'socket_create')
            self.assertEqual(gate['gpu_queries'],0);self.assertEqual(gate['ledger_changes'],0)

    def test_port_collision_distinguished_and_handle_closed(self):
        from unittest.mock import Mock
        handle=Mock();handle.bind.side_effect=OSError(errno.EADDRINUSE,'in use')
        with patch.object(suite_live.socket,'socket',return_value=handle):gate=suite_live.loopback_preflight(19000)
        self.assertEqual(gate['reason'],'PORT_IN_USE');self.assertEqual(gate['stage'],'bind');handle.close.assert_called_once();handle.listen.assert_not_called()
        with self.assertRaises(ValueError):suite_live.loopback_preflight(True)

    def test_blocked_gpu_gate_has_no_dispatch_reservation_or_budget_debit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';executor_fixture(root);before=sha(root/'budget-continuation.json');calls=[]
            cpu=synthetic_dispatch(calls);real=execute._dispatch
            def dispatch(*args):
                return cpu(*args) if args[2]['operation_kind']=='validate_prepared_inputs' else real(*args)
            gate={'status':'BLOCKED_LOOPBACK','reason':'PERMISSION_DENIED','gpu_queries':0,'ledger_changes':0}
            with patch.object(execute,'_dispatch',side_effect=dispatch),patch.object(suite_live,'preflight',return_value=gate), \
                 patch.object(execute,'_reserve_campaign',side_effect=AssertionError('Reservation on blocker')), \
                 patch.object(suite_live,'execute_acquisition',side_effect=AssertionError('Acquisition on blocker')):
                result=execute.resume(root,root/'local.json',True)
            self.assertEqual(result['status'],'BLOCKED_LOOPBACK');self.assertEqual(calls,['validate'])
            self.assertEqual(sha(root/'budget-continuation.json'),before)
            self.assertEqual(read(root/'checkpoint.json')['nodes']['serve-bf16-task0']['status'],'BLOCKED')
            self.assertFalse((root/'nodes/serve-bf16-task0/COMPLETION.json').exists())

    def test_synthetic_eight_unit_pause_resume_noop_and_cli_never_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';names=executor_fixture(root);calls=[]
            with patch.object(execute,'_dispatch',side_effect=synthetic_dispatch(calls)):
                a=execute.resume(root,root/'local.json',True,stop_after_node=names[0]);self.assertEqual(a['status'],'PAUSED_BOUNDARY')
                b=execute.resume(root,root/'local.json',True);self.assertEqual(b['status'],'COMPLETE_VALIDATED')
            self.assertEqual(len(calls),10);self.assertEqual(len(set(calls)),10)
            hashes={n:sha(root/n) for n in ('checkpoint.json','budget-continuation.json')}
            with patch.object(execute,'_dispatch',side_effect=AssertionError('No-op dispatch')),patch.object(suite_live,'preflight',side_effect=AssertionError('No-op preflight')):
                self.assertTrue(execute.resume(root,allow_gpu=False)['noop'])
            command=[sys.executable,'-B','-m','inference_kit','bench','suite','resume','--run',str(root)]
            child=subprocess.run(command,capture_output=True,check=True);self.assertTrue(json.loads(child.stdout)['noop'])
            self.assertEqual({n:sha(root/n) for n in hashes},hashes)
            ledger=read(root/'budget-continuation.json');self.assertEqual(len(ledger['jobs']),8);self.assertEqual(ledger['used_seconds'],108.)
            self.assertEqual(execute.status(root)['status'],'COMPLETE_VALIDATED')

    def test_ledger_duplicate_or_unknown_group_never_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';executor_fixture(root);context=read(root/'suite-execution-context.json');ledger=read(root/'budget-continuation.json')
            ledger['jobs']=[{'id':'not-a-frozen-group','wall_seconds':1.}];ledger['used_seconds']=101.;atomic(root/'budget-continuation.json',ledger)
            with self.assertRaisesRegex(ValueError,'unknown job'):execute._verify_ledger(context,root)


class ServingUnitKeyTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_arm_distinct_unit_keys_preserve_both_progress_and_directories(self):
        from inference_kit.suites import bench05b1_run as serving
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'ledger.json',{'jobs':[]})
            plan={'suite':{'budget':{'max_gpu_processes':8}}};arm={'arm_id':'bf16'};progress={'jobs':{}}
            with patch.object(serving,'check_gpu',return_value={'status':'BLOCKED_SHARED_GPU'}), \
                 patch.object(serving,'server_launch',side_effect=AssertionError('No launch')):
                for name in ('serve-bf16-taskA','serve-bf16-taskB'):
                    self.assertFalse(await serving.serving_job(plan,root,root/'ledger.json',progress,arm,
                        request_driver=lambda *a:None,collector_contract=suite_live.mixed_contract(),job_key=name))
            self.assertEqual(set(progress['jobs']),{'serve-bf16-taskA','serve-bf16-taskB'})
            self.assertEqual(len({v['directory'] for v in progress['jobs'].values()}),2)
            self.assertEqual(read(root/'ledger.json'),{'jobs':[]})
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                await serving.serving_job(plan,root,root/'ledger.json',progress,arm,
                    request_driver=lambda *a:None,collector_contract=suite_live.mixed_contract(),job_key='../escape')
