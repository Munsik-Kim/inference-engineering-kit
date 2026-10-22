"""One explicit CLI over the public Python API; CPU commands import no engine SDK."""
import argparse
import json
import sqlite3
from pathlib import Path
import sys
import zipfile
from . import api
from . import __version__


def main(argv=None):
    p=argparse.ArgumentParser(prog='inference-kit',description=__doc__)
    p.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    commands=p.add_subparsers(dest='command',required=True)
    bench=commands.add_parser('bench',help='BENCH05-A explicit plan/run and CPU raw replay')
    actions=bench.add_subparsers(dest='bench_action',required=True)
    inputs=actions.add_parser('inputs',help='Explicit local tokenizer preparation and CPU prepared validation')
    inputs_actions=inputs.add_subparsers(dest='inputs_action',required=True)
    ip=inputs_actions.add_parser('prepare')
    for name in ('spec','input','tokenizer-local','output'):ip.add_argument('--'+name,required=True,type=Path)
    iv=inputs_actions.add_parser('validate')
    for name in ('prepared','output'):iv.add_argument('--'+name,required=True,type=Path)
    iv.add_argument('--tokenizer-local',type=Path)
    ide=inputs_actions.add_parser('describe')
    for name in ('prepared','output'):ide.add_argument('--'+name,required=True,type=Path)
    suite=actions.add_parser('suite',help='Prepared Suite compiler, explicit E3 execution and CPU replay')
    suite_actions=suite.add_subparsers(dest='suite_action',required=True)
    suite_plan=suite_actions.add_parser('plan')
    for name in ('suite','sources','local','output'):suite_plan.add_argument('--'+name,required=True,type=Path)
    suite_validate=suite_actions.add_parser('validate')
    for name in ('plan','sources','output'):suite_validate.add_argument('--'+name,required=True,type=Path)
    suite_validate.add_argument('--local',type=Path)
    suite_describe=suite_actions.add_parser('describe')
    for name in ('plan','output'):suite_describe.add_argument('--'+name,required=True,type=Path)
    suite_bind=suite_actions.add_parser('bind',help='CPU prepared live layout or local binding inspection; no launch')
    for name in ('plan','local','output'):suite_bind.add_argument('--'+name,required=True,type=Path)
    suite_bind.add_argument('--metadata-only',action='store_true')
    suite_run=suite_actions.add_parser('run')
    for name in ('plan','local','output'):suite_run.add_argument('--'+name,required=True,type=Path)
    suite_run.add_argument('--allow-gpu',action='store_true');suite_run.add_argument('--stop-after-node')
    suite_resume=suite_actions.add_parser('resume');suite_resume.add_argument('--run',required=True,type=Path)
    suite_resume.add_argument('--local',type=Path);suite_resume.add_argument('--allow-gpu',action='store_true');suite_resume.add_argument('--stop-after-node')
    for action in ('status','replay'):
        sub=suite_actions.add_parser(action)
        for name in ('run','output'):sub.add_argument('--'+name,required=True,type=Path)
        if action=='replay':
            sub.add_argument('--historical-index',type=Path)
            sub.add_argument('--report-packaging',type=Path,help='Explicit JSON report packaging policy; no execution authorization')
    cpu=suite_actions.add_parser('postprocess',help='New CPU analysis revision over immutable acquired raw; no live migration')
    for name in ('run','output'):cpu.add_argument('--'+name,required=True,type=Path)
    cpu.add_argument('--historical-index',type=Path);cpu.add_argument('--analysis-revision')
    cpu.add_argument('--stop-after',choices=('score_task','analyze_tokens','index_add','report'))
    cpu.add_argument('--report-packaging',type=Path,help='Freeze an explicit JSON report packaging policy in this new CPU revision')
    cr=suite_actions.add_parser('postprocess-resume',help='Resume only this CPU analysis workspace')
    cr.add_argument('--analysis',required=True,type=Path)
    cr.add_argument('--stop-after',choices=('score_task','analyze_tokens','index_add','report'))
    cs=suite_actions.add_parser('postprocess-status')
    cs.add_argument('--analysis',required=True,type=Path);cs.add_argument('--output',required=True,type=Path)
    index=actions.add_parser('index',help='CPU-only evidence registration and offline report')
    index_actions=index.add_subparsers(dest='index_action',required=True)
    report_check=index_actions.add_parser('validate-report',help='Verify every report collection part/member after publication or relocation')
    report_check.add_argument('--report',required=True,type=Path)
    report_check.add_argument('--output',required=True,type=Path)
    for action in ('build','add'):
        sub=index_actions.add_parser(action)
        for name in ('sources','db','output'):sub.add_argument('--'+name,required=True,type=Path)
    for action in ('query','report'):
        sub=index_actions.add_parser(action)
        for name in ('db','output'):sub.add_argument('--'+name,required=True,type=Path)
        sub.add_argument('--root-map',type=Path,help='Explicit JSON mapping old physical roots to relocated roots')
        if action=='query':sub.add_argument('--task')
        else:sub.add_argument('--report-packaging',type=Path,help='Explicit bounded report packaging JSON; omitted retains legacy single-report limits')
    structure=actions.add_parser('structure',help='D1 explicit multi-layer capture and CPU raw replay')
    structure_actions=structure.add_subparsers(dest='structure_action',required=True)
    structure_plan=structure_actions.add_parser('plan')
    for name in ('spec','inputs','models','local','output'):structure_plan.add_argument('--'+name,required=True,type=Path)
    structure_run=structure_actions.add_parser('run');structure_run.add_argument('--plan',required=True,type=Path);structure_run.add_argument('--output',required=True,type=Path);structure_run.add_argument('--allow-gpu',action='store_true')
    structure_replay=structure_actions.add_parser('replay');structure_replay.add_argument('--run',required=True,type=Path);structure_replay.add_argument('--output',required=True,type=Path)
    tokens=actions.add_parser('tokens',help='CPU-only common token view and same-prefix comparison')
    token_actions=tokens.add_subparsers(dest='tokens_action',required=True)
    token_plan=token_actions.add_parser('plan',help='CPU-only configured token capture preflight')
    for name in ('spec','inputs','models','local','output'):token_plan.add_argument('--'+name,required=True,type=Path)
    token_run=token_actions.add_parser('run',help='Explicit token-only GPU acquisition; no serving')
    token_run.add_argument('--plan',required=True,type=Path);token_run.add_argument('--output',required=True,type=Path);token_run.add_argument('--allow-gpu',action='store_true')
    token_export=token_actions.add_parser('export')
    token_export.add_argument('--sources',required=True,type=Path);token_export.add_argument('--output',required=True,type=Path)
    token_compare=token_actions.add_parser('compare')
    token_compare.add_argument('--view',required=True,type=Path);token_compare.add_argument('--baseline',required=True);token_compare.add_argument('--candidate',required=True);token_compare.add_argument('--output',required=True,type=Path)
    b1=actions.add_parser('b1',help='BENCH05-B1 tech_record_extract v0.1 only')
    b1_actions=b1.add_subparsers(dest='b1_action',required=True)
    b1p=b1_actions.add_parser('plan');b1p.add_argument('--suite',required=True,type=Path);b1p.add_argument('--models',required=True,type=Path);b1p.add_argument('--local',required=True,type=Path);b1p.add_argument('--output',required=True,type=Path)
    b1r=b1_actions.add_parser('run');b1r.add_argument('--plan',required=True,type=Path);b1r.add_argument('--output',required=True,type=Path);b1r.add_argument('--allow-gpu',action='store_true')
    b1re=b1_actions.add_parser('replay');b1re.add_argument('--run',required=True,type=Path);b1re.add_argument('--output',required=True,type=Path)
    b1s=b1_actions.add_parser('score');b1s.add_argument('--sample',required=True,type=Path);b1s.add_argument('--generation',required=True,type=Path);b1s.add_argument('--output',required=True,type=Path)
    b2=actions.add_parser('b2',help='BENCH05-B2 fixed evidence/enum contract comparison')
    b2_actions=b2.add_subparsers(dest='b2_action',required=True)
    b2p=b2_actions.add_parser('plan');b2p.add_argument('--suite',required=True,type=Path);b2p.add_argument('--models',required=True,type=Path);b2p.add_argument('--local',required=True,type=Path);b2p.add_argument('--output',required=True,type=Path)
    b2r=b2_actions.add_parser('run');b2r.add_argument('--plan',required=True,type=Path);b2r.add_argument('--output',required=True,type=Path);b2r.add_argument('--allow-gpu',action='store_true')
    b2re=b2_actions.add_parser('replay');b2re.add_argument('--run',required=True,type=Path);b2re.add_argument('--output',required=True,type=Path)
    b2s=b2_actions.add_parser('score');b2s.add_argument('--sample',required=True,type=Path);b2s.add_argument('--generation',required=True,type=Path);b2s.add_argument('--output',required=True,type=Path)
    b3=actions.add_parser('b3',help='BENCH05-B3 frozen ARC choice likelihood regression')
    b3a=b3.add_subparsers(dest='b3_action',required=True)
    b3p=b3a.add_parser('plan')
    for name in ('suite','models','local','output'):b3p.add_argument('--'+name,required=True,type=Path)
    b3r=b3a.add_parser('run');b3r.add_argument('--plan',required=True,type=Path);b3r.add_argument('--output',required=True,type=Path);b3r.add_argument('--allow-gpu',action='store_true')
    b3re=b3a.add_parser('replay');b3re.add_argument('--run',required=True,type=Path);b3re.add_argument('--output',required=True,type=Path)
    b3s=b3a.add_parser('score')
    for name in ('sample','choices','output'):b3s.add_argument('--'+name,required=True,type=Path)
    b4=actions.add_parser('b4',help='BENCH05-B4 closed-context answer/evidence retrieval')
    b4a=b4.add_subparsers(dest='b4_action',required=True)
    b4p=b4a.add_parser('plan')
    for name in ('suite','models','local','output'):b4p.add_argument('--'+name,required=True,type=Path)
    b4r=b4a.add_parser('run');b4r.add_argument('--plan',required=True,type=Path);b4r.add_argument('--output',required=True,type=Path);b4r.add_argument('--allow-gpu',action='store_true')
    b4re=b4a.add_parser('replay');b4re.add_argument('--run',required=True,type=Path);b4re.add_argument('--output',required=True,type=Path)
    b4s=b4a.add_parser('score')
    for name in ('sample','generation','output'):b4s.add_argument('--'+name,required=True,type=Path)
    bp=actions.add_parser('plan');bp.add_argument('--suite',required=True,type=Path);bp.add_argument('--models',required=True,type=Path);bp.add_argument('--local',required=True,type=Path);bp.add_argument('--output',required=True,type=Path)
    br=actions.add_parser('run');br.add_argument('--plan',required=True,type=Path);br.add_argument('--output',required=True,type=Path);br.add_argument('--allow-gpu',action='store_true')
    replay_bench=actions.add_parser('replay');replay_bench.add_argument('--run',required=True,type=Path);replay_bench.add_argument('--output',required=True,type=Path)
    demo=commands.add_parser('demo');demo.add_argument('--output',required=True,type=Path)
    plan=commands.add_parser('plan');plan.add_argument('--spec',required=True,type=Path);plan.add_argument('--local',type=Path)
    for name in ('verify','replay'):
        c=commands.add_parser(name);c.add_argument('--bundle',required=True,type=Path);c.add_argument('--output',required=True,type=Path);c.add_argument('--expected-sha256')
        if name=='replay':c.add_argument('--case',choices=('step02','step03','step04'),required=True)
    compare=commands.add_parser('compare');compare.add_argument('--baseline',required=True,type=Path);compare.add_argument('--candidate',required=True,type=Path);compare.add_argument('--output',required=True,type=Path);compare.add_argument('--replicates',type=int,default=5000)
    assess=commands.add_parser('assess');assess.add_argument('--index',required=True,type=Path);assess.add_argument('--requirements',required=True,type=Path);assess.add_argument('--output',required=True,type=Path)
    run=commands.add_parser('run');run.add_argument('--spec',required=True,type=Path);run.add_argument('--local',required=True,type=Path);run.add_argument('--output',required=True,type=Path);run.add_argument('--allow-gpu',action='store_true')
    compose=commands.add_parser('compose');compose.add_argument('--base',required=True,type=Path);compose.add_argument('--intervention',required=True,type=Path);compose.add_argument('--modules',nargs='*',required=True);compose.add_argument('--base-receipt',required=True,type=Path);compose.add_argument('--intervention-receipt',required=True,type=Path);compose.add_argument('--output',required=True,type=Path);compose.add_argument('--allow-tensor-runtime',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.command=='bench':
            if a.bench_action=='inputs':
                if a.inputs_action=='prepare':r=api.inputs_prepare(a.spec,a.input,a.tokenizer_local,a.output)
                elif a.inputs_action=='validate':r=api.inputs_validate(a.prepared,a.output,a.tokenizer_local)
                else:r=api.inputs_describe(a.prepared,a.output)
            elif a.bench_action=='suite':
                if a.suite_action=='plan':
                    full=api.suite_plan(a.suite,a.sources,a.local,a.output)
                    r={k:full[k] for k in ('status','plan_identity','execution_count','acquisition_count','GPU_queries','model_loads','tokenizer_calls','network_requests','ledger_changes')}
                    r['counts']=full['semantic']['counts']
                elif a.suite_action=='validate':r=api.suite_validate(a.plan,a.sources,a.output,a.local)
                elif a.suite_action=='describe':r=api.suite_describe(a.plan,a.output)
                elif a.suite_action=='bind':r=api.suite_bind(a.plan,a.local,a.output,a.metadata_only)
                elif a.suite_action=='run':r=api.suite_run(a.plan,a.local,a.output,a.allow_gpu,a.stop_after_node)
                elif a.suite_action=='resume':r=api.suite_resume(a.run,a.local,a.allow_gpu,a.stop_after_node)
                elif a.suite_action=='status':r=api.suite_status(a.run,a.output)
                elif a.suite_action=='postprocess':
                    from .core.io import read
                    r=api.suite_postprocess(a.run,a.output,a.historical_index,a.analysis_revision,a.stop_after,
                                            read(a.report_packaging) if a.report_packaging else None)
                elif a.suite_action=='postprocess-resume':r=api.suite_postprocess_resume(a.analysis,a.stop_after)
                elif a.suite_action=='postprocess-status':
                    r=api.suite_postprocess_status(a.analysis)
                    from .core.io import write
                    from .core.state import new_output
                    write(new_output(a.output)/'STATUS.json',r)
                else:
                    from .core.io import read
                    r=api.suite_replay(a.run,a.output,a.historical_index,
                                       read(a.report_packaging) if a.report_packaging else None)
            elif a.bench_action=='index':
                if a.index_action=='build':r=api.index_build(a.sources,a.db,a.output)
                elif a.index_action=='add':r=api.index_add(a.sources,a.db,a.output)
                elif a.index_action=='validate-report':r=api.index_report_validate(a.report,a.output)
                else:
                    from .core.io import read
                    roots=read(a.root_map) if a.root_map else None
                    if a.index_action=='query':r=api.index_query(a.db,a.output,a.task,roots)
                    else:r=api.index_report(a.db,a.output,roots,
                                             read(a.report_packaging) if a.report_packaging else None)
            elif a.bench_action=='structure':
                if a.structure_action=='plan':
                    full=api.structure_plan(a.spec,a.inputs,a.models,a.local,a.output)
                    r={k:full[k] for k in ('task','status','GPU_actions','model_loads','tokenizer_loads','gpu_allowance_seconds','counts','resource_estimates','plan_identity')}
                elif a.structure_action=='run':
                    if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required; structure plan/replay are CPU only')
                    r=api.structure_run(a.plan,a.output)
                else:r=api.structure_replay(a.run,a.output)
            elif a.bench_action=='tokens':
                if a.tokens_action=='export':r=api.tokens_export(a.sources,a.output)
                elif a.tokens_action=='plan':
                    full=api.tokens_plan(a.spec,a.inputs,a.models,a.local,a.output)
                    r={k:full[k] for k in ('task','status','GPU_actions','model_loads','tokenizer_loads','gpu_allowance_seconds','counts','resource_estimates','plan_identity')}
                elif a.tokens_action=='run':
                    if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required; tokens plan/export/compare are CPU only')
                    r=api.tokens_run(a.plan,a.output)
                else:
                    full=api.tokens_compare(a.view,a.baseline,a.candidate,a.output)
                    r={k:full[k] for k in ('task','baseline','candidate','view_identity','coverage','compatibility','ENGINEERING','GPU_actions','model_loads','downloads','ledger_changes','REMOTE_CI')}
            elif a.bench_action=='b1':
                if a.b1_action=='plan':
                    full=api.bench_b1_plan(a.suite,a.models,a.local,a.output)
                    r={k:full[k] for k in ('task','status','gpu_actions','model_loads','network_actions','gpu_allowance_seconds','counts','plan_identity')}
                elif a.b1_action=='run':
                    if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required for B1; score/plan/replay are CPU only')
                    r=api.bench_b1_run(a.plan,a.output)
                elif a.b1_action=='score':r=api.bench_b1_score(a.sample,a.generation,a.output)
                else:r=api.bench_b1_replay(a.run,a.output)
            elif a.bench_action=='b2':
                if a.b2_action=='plan':
                    full=api.bench_b2_plan(a.suite,a.models,a.local,a.output)
                    r={k:full[k] for k in ('task','status','gpu_actions','model_loads','network_actions','gpu_allowance_seconds','counts','plan_identity')}
                elif a.b2_action=='run':
                    if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required for B2; score/plan/replay are CPU only')
                    r=api.bench_b2_run(a.plan,a.output)
                elif a.b2_action=='score':r=api.bench_b2_score(a.sample,a.generation,a.output)
                else:r=api.bench_b2_replay(a.run,a.output)
            elif a.bench_action=='b3':
                if a.b3_action=='plan':
                    full=api.bench_b3_plan(a.suite,a.models,a.local,a.output)
                    r={k:full[k] for k in ('task','status','gpu_actions','model_loads','network_actions','gpu_allowance_seconds','counts','plan_identity')}
                elif a.b3_action=='run':
                    if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required for B3; CPU plan/replay/score never load a model')
                    r=api.bench_b3_run(a.plan,a.output)
                elif a.b3_action=='score':r=api.bench_b3_score(a.sample,a.choices,a.output)
                else:r=api.bench_b3_replay(a.run,a.output)
            elif a.bench_action=='b4':
                if a.b4_action=='plan':
                    full=api.bench_b4_plan(a.suite,a.models,a.local,a.output)
                    r={k:full[k] for k in ('task','status','gpu_actions','model_loads','network_actions','gpu_allowance_seconds','counts','plan_identity')}
                elif a.b4_action=='run':
                    if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required for B4; plan/score/replay are CPU only')
                    r=api.bench_b4_run(a.plan,a.output)
                elif a.b4_action=='score':r=api.bench_b4_score(a.sample,a.generation,a.output)
                else:r=api.bench_b4_replay(a.run,a.output)
            elif a.bench_action=='plan':
                full=api.bench_plan(a.suite,a.models,a.local,a.output)
                r={k:full[k] for k in ('task','status','gpu_actions','model_loads','network_actions','gpu_allowance_seconds','counts','plan_identity')}
            elif a.bench_action=='run':
                if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required; bench plan/replay never launch a model')
                r=api.bench_run(a.plan,a.output)
            else:r=api.bench_replay(a.run,a.output)
        elif a.command=='demo':r=api.demo(a.output)
        elif a.command=='plan':r=api.plan(a.spec,a.local)
        elif a.command=='verify':r=api.verify(a.bundle,a.output,a.expected_sha256)
        elif a.command=='replay':r=api.replay(a.case,a.bundle,a.output,a.expected_sha256)
        elif a.command=='compare':r=api.compare(a.baseline,a.candidate,a.output,a.replicates)
        elif a.command=='assess':r=api.assess(a.index,a.requirements,a.output)
        elif a.command=='run':
            if not a.allow_gpu:raise ValueError('Explicit --allow-gpu required; CPU plan is separate')
            from .adapters.run import run
            r=run(a.spec,a.local,a.output)
        else:
            if not a.allow_tensor_runtime:raise ValueError('Explicit --allow-tensor-runtime required')
            from .adapters.tensors import compose
            r=compose(a.base,a.intervention,a.modules,a.output,a.base_receipt,a.intervention_receipt)
        print(json.dumps(r,ensure_ascii=True));return 0
    except (ValueError,OSError,KeyError,TypeError,ImportError,RuntimeError,TimeoutError,zipfile.BadZipFile,sqlite3.Error) as exc:
        print(json.dumps({'status':'BLOCKED_OR_INVALID','error_type':type(exc).__name__,'error':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
