"""Maintained STEP03 metric/selector parity contracts."""
import copy
import unittest
from inference_kit.core.metrics import parse,counter_delta,gauge_area
from inference_kit.core.assess_multi import assess

class MetricsTests(unittest.TestCase):
    def snapshot(self, waiting=1, counter=4):
        text=f'''# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{engine="0",model_name="serving-lab"}} {waiting}
# HELP vllm:num_preemptions_total Cumulative preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{{engine="0",model_name="serving-lab"}} {counter}
'''
        return parse(text)
    def test_missing_is_not_zero(self):
        result=parse('')
        self.assertEqual(result['roles']['waiting']['status'],'NOT_AVAILABLE')
        self.assertIsNone(counter_delta(result,result,'preemptions')['delta'])
    def test_counter_reset_is_not_success(self):
        result=counter_delta(self.snapshot(counter=5),self.snapshot(counter=1),'preemptions')
        self.assertEqual(result['status'],'COUNTER_RESET');self.assertIsNone(result['delta'])
    def test_observed_counter_zero_differs_from_missing(self):
        result=counter_delta(self.snapshot(),self.snapshot(),'preemptions')
        self.assertEqual(result['status'],'OBSERVED');self.assertEqual(result['delta'],0)
    def test_counter_label_change_is_not_a_difference(self):
        before=self.snapshot();after=self.snapshot(counter=5)
        for r in after['roles']['preemptions']['samples']:r['labels']['engine']='1'
        self.assertEqual(counter_delta(before,after,'preemptions')['status'],'LABEL_IDENTITY_CHANGED')
    def test_waiting_is_time_weighted_not_sample_count(self):
        observations=[{'monotonic':t,'http_status':200,'parsed':self.snapshot(v)} for t,v in ((0,1),(1,3),(4,0))]
        result=gauge_area(observations,'waiting',0,4)
        self.assertEqual(result['integral'],10);self.assertEqual(result['time_weighted_mean'],2.5)
    def test_failed_poll_preserves_gap(self):
        observations=[{'monotonic':0,'http_status':500,'parsed':{}},{'monotonic':2,'http_status':200,'parsed':self.snapshot()},{'monotonic':4,'http_status':200,'parsed':self.snapshot()}]
        result=gauge_area(observations,'waiting',0,4)
        self.assertEqual(result['covered_seconds'],2);self.assertEqual(result['coverage_fraction'],.5)

class SelectorTests(unittest.TestCase):
    def fixture(self):
        spec={'demo_policy':{'memory':{'scope':'whole_device_mib','phase_scope':'performance_with_startup','max_mib':8192},'latency_constraints':[{'input_length':v,'output_tokens':128,'concurrency':4,'metric':'ttft_seconds','statistic':'p95','max_seconds':.2 if v==128 else 1.0} for v in (128,1024)],'quality':{'delta_floor':-.02,'absolute_floor':.4}}};identity={k:'fixed-'+k for k in ('gpu_uuid','runtime_identity','workload_identity','instrumentation_identity','fixed_profile_identity','tokenizer_identity')}
        req={'schema_version':3,'policy':'DEMO','model_id':'qwen','model_revision':'pinned','comparison_group_identity':'experiment',
             'identity':identity,'treatment_axes':['weight_artifact','kv_pool_capacity'],'memory':spec['demo_policy']['memory'],
             'latency_constraints':copy.deepcopy(spec['demo_policy']['latency_constraints']),'quality':spec['demo_policy']['quality'],
             'allow_retained_quality':True,'quality_evidence':'retained_checkpoint','evidence_level':'exploratory_point'}
        row={'configuration_id':'D','arm':'D','model_id':'qwen','model_revision':'pinned','comparison_group_identity':'experiment','identity':copy.deepcopy(identity),'treatment_axes':req['treatment_axes'],
             'kind':'live_measurement','measurement_status':'VALID_LIVE_PROFILE','validity':{'requests_complete':True,'owned_resources_clean':True,'server_rounds':3,'parent_returncodes':[0,0,0],'comparison_status':'NO_INTERFERENCE_OBSERVED_WITHIN_OBSERVATION_LIMITS'},
             'memory':{'scope':'whole_device_mib','phase_scope':'performance_with_startup','max_mib':7000},
             'performance':{'cells':[{'input_length':l,'output_tokens':128,'concurrency':4,'ttft_seconds':{'p95':.1}} for l in (128,1024)]},
             'retained_quality':{'acc_norm':.45,'delta':-.01,'delta_ci95':[-.03,.01],'absolute_acc_norm_ci95':[.4,.5]},
             'quality_transfer':{'identity_status':'MATCHED_ARTIFACT_RUNTIME_TOKENIZER_SCORING','quality_source_profile':'previous'},
             'cache_sanity':{'status':'COMPLETE','same_weight_prediction_match':True},'exact_profile_quality':'NOT_VERIFIED'}
        return {'schema_version':3,'kind':'verified_result_index','results':[row]},req,row
    def test_all_conditions_produce_only_conditional_point_candidate(self):
        index,req,row=self.fixture();result=assess(index,req)
        self.assertEqual(result['status'],'CONDITIONAL_OBSERVED_CANDIDATE');self.assertEqual(result['pareto_candidates'],['D'])
        self.assertEqual(result['candidates'][0]['overall_deployment_evidence']['exact_profile_quality'],'NOT_VERIFIED')
    def test_long_latency_failure_is_not_hidden_by_short_success(self):
        index,req,row=self.fixture();row['performance']['cells'][1]['ttft_seconds']['p95']=3
        self.assertEqual(assess(index,req)['status'],'NO_FEASIBLE_CANDIDATE')
    def test_unobserved_cell_is_unknown(self):
        index,req,row=self.fixture();row['performance']['cells'].pop()
        result=assess(index,req);self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE');self.assertEqual(result['candidates'][0]['operational_constraints']['checks']['latency_1']['status'],'UNKNOWN')
    def test_axes_can_vary_but_fixed_runtime_cannot(self):
        index,req,row=self.fixture();other=copy.deepcopy(row);other.update(configuration_id='B',arm='B',execution={'kv_tokens':8192,'method':'bf16'});index['results'].append(other)
        self.assertEqual(set(assess(index,req)['pareto_candidates']),{'D','B'})
        row['identity']['runtime_identity']='different';other['identity']['fixed_profile_identity']='different'
        self.assertEqual(assess(index,req)['status'],'INSUFFICIENT_EVIDENCE')
    def test_memory_scope_and_phase_are_checked(self):
        for key,value in (('scope','process_mib'),('phase_scope','sanity_only')):
            index,req,row=self.fixture();row['memory'][key]=value
            self.assertEqual(assess(index,req)['status'],'INSUFFICIENT_EVIDENCE')
    def test_interval_boundary_remains_unknown(self):
        index,req,row=self.fixture();req['evidence_level']='screen_with_interval'
        result=assess(index,req);self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE');self.assertEqual(result['candidates'][0]['overall_deployment_evidence']['checks']['quality_loss']['status'],'UNKNOWN')
    def test_exact_profile_and_forbidden_transfer_cannot_promote_retained_quality(self):
        for change in ({'quality_evidence':'exact_profile'},{'allow_retained_quality':False}):
            index,req,row=self.fixture();req.update(change)
            self.assertEqual(assess(index,req)['status'],'INSUFFICIENT_EVIDENCE')
    def test_partial_historical_or_interference_cannot_be_candidate(self):
        for change in ({'measurement_status':'PARTIAL_UNCERTIFIED'},{'kind':'historical'},{'validity':{'comparison_status':'SHARED_GPU_INTERFERENCE'}}):
            index,req,row=self.fixture();row.update(change)
            self.assertEqual(assess(index,req)['status'],'INSUFFICIENT_EVIDENCE')
