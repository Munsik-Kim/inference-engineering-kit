"""Public CPU API shared by the CLI and external Python consumers."""
import copy
from importlib.resources import files
import json
from pathlib import Path
from .core.io import read,write,require,digest,sha,loads
from .core.state import new_output,derived_output,preserve_failure
from .core.evidence import open_bundle,case_root,verify_manifest,export_identity
from .core.parity import equal
from .core.index import check_index


def suite_plan(suite, sources, local, output):
    from .suites.suite_plan import plan
    return plan(suite, sources, local, output)


def suite_validate(plan, sources, output, local=None):
    from .suites.suite_plan import validate
    return validate(plan, sources, output, local)


def suite_describe(plan, output):
    from .suites.suite_plan import describe
    return describe(plan, output)


def suite_bind(plan, local, output, metadata_only=False):
    from .suites.suite_execution import bind
    return bind(plan, local, output, metadata_only)


def suite_run(plan, local, output, allow_gpu=False, stop_after_node=None):
    from .suites.suite_execution import run
    return run(plan,local,output,allow_gpu,stop_after_node)


def suite_status(run, output=None):
    from .suites.suite_execution import status
    return status(run,output)


def suite_resume(run, local=None, allow_gpu=False, stop_after_node=None):
    from .suites.suite_execution import resume
    return resume(run,local,allow_gpu,stop_after_node)


def suite_replay(run, output, historical_index=None, report_packaging=None):
    from .suites.suite_execution import replay
    return replay(run,output,historical_index=historical_index,report_packaging=report_packaging)


def suite_postprocess(run, output, historical_index=None, analysis_revision=None, stop_after=None,
                      report_packaging=None):
    from .analysis.suite_postprocess import postprocess
    return postprocess(run, output, historical_index=historical_index,
                       analysis_revision=analysis_revision, stop_after=stop_after,
                       report_packaging=report_packaging)


def suite_postprocess_resume(analysis, stop_after=None):
    from .analysis.suite_postprocess import resume
    return resume(analysis, stop_after=stop_after)


def suite_postprocess_status(analysis):
    from .analysis.suite_postprocess import status
    return status(analysis)


def index_build(sources, db, output):
    from .analysis.result_index import build_index
    return build_index(sources, db, output)


def index_add(sources, db, output):
    from .analysis.result_index import add_sources
    return add_sources(sources, db, output)


def index_query(db, output, task=None, root_map=None):
    from .analysis.result_index import query_index
    return query_index(db, output, task, root_map)


def index_report(db, output, root_map=None, report_packaging=None):
    from .analysis.result_index import report_index
    return report_index(db, output, root_map, packaging=report_packaging)


def index_report_validate(report, output=None):
    """Read and verify the closed report collection without model/runtime access."""
    from .analysis.report_packaging import validate_report_collection
    result = validate_report_collection(report)
    if output is not None:
        write(new_output(output)/'REPORT_COLLECTION_VALIDATION.json', result)
    return result


def plan(spec,local=None):
    from .core.config import make_plan
    return make_plan(spec,local)


def _record(path_or_value):
    if isinstance(path_or_value,(str,Path)):
        path=Path(path_or_value)
        if path.suffix=='.jsonl':
            return [loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        return read(path)
    return path_or_value


def compare(baseline,candidate,output=None,replicates=5000,seed=909223):
    require(type(replicates) is int and 1<=replicates<=5000,'Invalid bootstrap budget')
    baseline,candidate=_record(baseline),_record(candidate)
    if isinstance(baseline,dict) and isinstance(candidate,dict):
        from .core.quality import paired_quality
        for key in ('kind','task','split','primary_metric','auxiliary_metric','normalization','runtime_identity','tokenizer_identity'):
            require(baseline.get(key)==candidate.get(key),'Incompatible quality '+key)
        result=paired_quality(baseline,candidate,replicates,seed)
    else:
        from ._vendor.diova_compare.core import compare as compare_records
        result=compare_records(baseline,candidate,replicates=replicates,seed=seed)
    if output:write(new_output(output)/'comparison.json',result)
    return result


def load_index(path):
    path=Path(path).absolute();index=read(path)
    if index.get('evidence_scope')=='research_dev_only':return index
    if index.get('evidence_kind')=='synthetic_test':
        require(index['results_identity']==digest(index['results']),'Synthetic index identity differs')
        return index
    binding=index.get('evidence_binding');require(binding,'Explicit evidence bundle binding required; historical absolute path is metadata only')
    bundle=Path(binding['bundle']);bundle=bundle if bundle.is_absolute() else path.parent/bundle
    with open_bundle(bundle,expected_sha256=binding.get('sha256')) as root:
        selected=case_root(root,binding['case']);check_index(index,root,selected)
    return index


def assess(index,requirements,output=None):
    from .core.assess import assess as evaluate
    index=load_index(index) if isinstance(index,(str,Path)) else index
    requirements=_record(requirements)
    if index.get('evidence_scope')=='research_dev_only':
        result={'status':'INSUFFICIENT_EVIDENCE','evidence_scope':'research_dev_only',
                'production_recommendation':'NOT_ELIGIBLE','reason':'Research DEV cannot be promoted to deployment evidence'}
    else:result=evaluate(index,requirements)
    if output:write(new_output(output)/'assessment.json',result)
    return result


def verify(bundle,output,expected_sha256=None):
    out=derived_output(bundle,output)
    with preserve_failure(out),open_bundle(bundle,expected_sha256=expected_sha256) as root:
        result=verify_manifest(root);indexes=0;index_members=0;archival_issues=[]
        paths=list(root.rglob('results-index.json'));primary=None
        # A STEP03 bundle contains only the retained portion of older STEP02 evidence.
        # Its archived STEP02 index is not the current experiment's complete manifest.
        for case in ('step04','step03','step02'):
            try:selected=case_root(root,case)
            except ValueError:continue
            primary={selected/'results-index.json'} if case!='step04' else set()
            break
        if primary is None:primary=set(paths)
        for p in paths:
            value=read(p)
            if value.get('kind')!='verified_result_index':continue
            try:check_index(value,root,p.parent)
            except ValueError as exc:
                if p in primary:raise
                archival_issues.append({'index':str(p.relative_to(root)),'status':'ARCHIVAL_REFERENCES_NOT_FULLY_VERIFIED','reason':str(exc)})
                continue
            indexes+=1;index_members+=len(value['evidence_manifest'])
        result.update(indexes_verified=indexes,index_references_verified=index_members,
                      reference_coverage='PRIMARY_VERIFIED_ARCHIVAL_PARTIAL' if archival_issues else 'ALL_DECLARED_INDEX_REFERENCES_VERIFIED' if paths else 'NO_RESULT_INDEX_IN_INPUT',
                      archival_reference_issues=archival_issues,
                      archive_byte_identity='MATCHED_EXPECTED_SHA256' if expected_sha256 else 'SHA256_RECORDED_NOT_EXTERNAL_AUTHENTICATION',
                      archive_sha256=sha(Path(bundle)) if Path(bundle).is_file() else None,
                      sanitized_export_identity=export_identity(root),numeric_recalculation='NOT_RUN_USE_REPLAY',
                      new_gpu_runs=0,downloads=0)
    write(out/'verification.json',result);return result


def replay(case,bundle,output,expected_sha256=None):
    require(case in ('step02','step03','step04'),'Unsupported frozen recipe')
    out=derived_output(bundle,output);bundle=Path(bundle).expanduser().absolute()
    with preserve_failure(out),open_bundle(bundle,expected_sha256=expected_sha256) as root:
        verified=verify_manifest(root)
        if (root/'recorded-fixture.json').is_file():
            from .experiments.recorded import replay_fixture
            result=replay_fixture(read(root/'recorded-fixture.json'),case)
            result['manifest_verification']=verified;write(out/'replay.json',result);return result
        selected=case_root(root,case)
        if case=='step04':
            from .experiments.step04 import analyze
            summary=analyze(selected,out)
            for name in ('comparisons.json','decision.json','local-summary.json','summary.json'):
                equal(read(out/name),read(selected/name))
            result={'status':'CPU_REPLAY_PARITY_PASS','case':case,'historical_quality_documents_per_arm':256,
                    'historical_main_http':summary['new_main_http'],'historical_sentinel_http':summary['new_sentinel_http'],
                    'historical_research_decision':summary['research_decision'],'independent_holdout_sources':32}
            index={'schema_version':4,'kind':'research_diagnostic_result_index','evidence_scope':'research_dev_only',
                   'production_recommendation':'NOT_ELIGIBLE','summary':'summary.json'}
        elif case=='step03':
            from .experiments.step03 import analyze
            frozen=read(selected/'PLAN_FROZEN.json');old=check_index(read(selected/'results-index.json'),root,selected)
            computed=analyze(frozen['spec'],selected,out)
            equal(computed['index']['results'],old['results'])
            equal(read(out/'comparisons.json'),read(selected/'comparisons.json'))
            for name in ('point','interval','exact-profile','no-retained'):
                equal(read(out/'assessments'/(name+'.json')),read(selected/'assessments'/(name+'.json')))
            result={'status':'CPU_REPLAY_PARITY_PASS','case':case,'historical_server_rounds':len(computed['rounds']),
                    'historical_performance_requests':sum(c['n_timed_requests']+4 for r in computed['rounds'] for c in r['cells']),
                    'historical_assessment_statuses':{k:v['status'] for k,v in computed['assessments'].items()}}
            index=computed['index']
        else:
            from .experiments.step02 import replay as replay_step02
            result,index=replay_step02(root,selected,out)
        if case!='step04':
            # Rebinding is a new derived view; original index/evidence/source hashes remain in the input.
            portable=copy.deepcopy(index)
            portable['historical_evidence_root']=portable['evidence_root'];portable['evidence_root']='${BOUND_EVIDENCE_ROOT}'
            portable['evidence_binding']={'bundle':str(bundle),'sha256':sha(bundle) if bundle.is_file() else None,'case':case}
            if (out/'results-index.json').exists():
                # Preserve the calculated legacy view before publishing a separately named portable index.
                write(out/'portable-index.json',portable)
            else:write(out/'portable-index.json',portable)
        else:write(out/'research-index.json',index)
        result.update(analysis_kind='historical_cpu_reanalysis',new_gpu_runs=0,new_model_downloads=0,
                      input_archive_sha256=sha(bundle) if bundle.is_file() else None,manifest_verification=verified,
                      sanitized_export_identity=export_identity(root),
                      float_tolerance={'abs':1e-10,'rel':1e-9,'integers_booleans_status':'exact'},
                      gpu_support='HISTORICAL_GPU_EVIDENCE; maintained GPU path NOT_RERUN')
    write(out/'replay.json',result);return result


def demo(output):
    out=new_output(output);fixture=json.loads(files('inference_kit.resources').joinpath('demo.json').read_text(encoding='utf-8'))
    write(out/'baseline.json',fixture['baseline']);write(out/'candidate.json',fixture['candidate'])
    write(out/'index.json',fixture['index']);write(out/'requirements.json',fixture['requirements'])
    comp=compare(fixture['baseline'],fixture['candidate'],out/'comparison',replicates=200)
    assessment=assess(fixture['index'],fixture['requirements'],out/'assessment')
    result={'status':'CPU_SYNTHETIC_DEMO_COMPLETE','evidence_kind':'synthetic_test','inference':False,
            'source':'installed wheel package resource','assessment_status':assessment['status'],'new_gpu_runs':0}
    write(out/'demo.json',result);return result


def tokens_export(sources,output):
    """CPU-only portable token view from explicitly declared preserved sources."""
    from .analysis.tokens import export_tokens
    return export_tokens(sources,output)


def tokens_compare(view,baseline,candidate,output):
    """CPU-only same-prefix metrics and separate free-generation comparisons."""
    from .analysis.tokens import compare_tokens
    return compare_tokens(view,baseline,candidate,output)


def tokens_plan(spec,inputs,models,local,output):
    """CPU-only configured token capture plan over preserved prepared inputs."""
    from .suites.token_probe import plan
    return plan(spec,inputs,models,local,output)


def tokens_run(plan,output):
    """Explicit opt-in token-only GPU capture; CLI requires --allow-gpu."""
    from .suites.token_probe import run
    return run(plan,output)


def structure_plan(spec,inputs,models,local,output):
    """CPU-only multi-layer diagnostic plan over unchanged prepared inputs."""
    from .suites.structure_probe import plan
    return plan(spec,inputs,models,local,output)


def structure_run(plan,output):
    """Explicit GPU structure acquisition; CLI also requires --allow-gpu."""
    from .suites.structure_probe import run
    return run(plan,output)


def structure_replay(run,output):
    """CPU-only complete structure/control/ON token replay from saved raw."""
    from .analysis.structure import replay_structure
    return replay_structure(run,output)


def bench_plan(suite,models,local,output):
    from .suites.bench05a_plan import plan
    return plan(suite,models,local,output)


def bench_run(plan,output):
    """Explicitly authorized GPU operation; CLI additionally requires --allow-gpu."""
    from .suites.bench05a_run import run
    return run(plan,output)


def bench_replay(run,output):
    from .suites.bench05a_replay import replay
    return replay(run,output)


def bench_b1_plan(suite,models,local,output):
    from .suites.bench05b1_plan import plan
    return plan(suite,models,local,output)


def bench_b1_run(plan,output):
    from .suites.bench05b1_run import run
    return run(plan,output)


def bench_b1_replay(run,output):
    from .suites.bench05b1_replay import replay
    return replay(run,output)


def bench_b1_score(sample,generation,output=None):
    from .tasks.tech_record import score
    result=score(_record(sample),_record(generation))
    if output is not None:write(new_output(output)/'score.json',result)
    return result


def bench_b2_plan(suite,models,local,output):
    from .suites.bench05b2_plan import plan
    return plan(suite,models,local,output)


def bench_b2_run(plan,output):
    from .suites.bench05b2_run import run
    return run(plan,output)


def bench_b2_replay(run,output):
    from .suites.bench05b2_replay import replay
    return replay(run,output)


def bench_b2_score(sample,generation,output=None):
    from .tasks.contract_policy import diagnostics
    result=diagnostics(_record(sample),_record(generation))
    if output is not None:write(new_output(output)/'score.json',result)
    return result


def bench_b3_plan(suite,models,local,output):
    from .suites.bench05b3_plan import plan
    return plan(suite,models,local,output)


def bench_b3_run(plan,output):
    from .suites.bench05b3_run import run
    return run(plan,output)


def bench_b3_replay(run,output):
    from .suites.bench05b3_replay import replay
    return replay(run,output)


def bench_b3_score(sample,choices,output=None):
    from .tasks.registry import get_task
    result=get_task('arc_challenge','1.0').score(_record(sample),_record(choices))
    if output is not None:write(new_output(output)/'score.json',result)
    return result


def task_adapter(task_id,version=None):
    """Explicit CPU task registry; adapters do not load models or own processes."""
    from .tasks.registry import get_task
    return get_task(task_id,version)


def bench_b4_plan(suite,models,local,output):
    from .suites.bench05b4_plan import plan
    return plan(suite,models,local,output)


def bench_b4_run(plan,output):
    from .suites.bench05b4_run import run
    return run(plan,output)


def bench_b4_replay(run,output):
    from .suites.bench05b4_replay import replay
    return replay(run,output)


def bench_b4_score(sample,generation,output=None):
    result=task_adapter('context_evidence_retrieval','0.1').score(_record(sample),_record(generation) if generation is not None else None)
    if output is not None:write(new_output(output)/'score.json',result)
    return result


def inputs_prepare(spec, input, tokenizer_local, output):
    from .probes.input_preparation import prepare
    return prepare(spec, input, tokenizer_local, output)


def inputs_validate(prepared, output, tokenizer_local=None):
    from .probes.input_preparation import validate
    return validate(prepared, output, tokenizer_local)


def inputs_describe(prepared, output):
    from .probes.input_preparation import describe
    return describe(prepared, output)
