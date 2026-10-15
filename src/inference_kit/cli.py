"""One explicit CLI over the public Python API; CPU commands import no engine SDK."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
from . import api


def main(argv=None):
    p=argparse.ArgumentParser(prog='inference-kit',description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
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
        if a.command=='demo':r=api.demo(a.output)
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
        print(json.dumps(r,ensure_ascii=False));return 0
    except (ValueError,OSError,KeyError,TypeError,ImportError,RuntimeError,TimeoutError,zipfile.BadZipFile) as exc:
        print(json.dumps({'status':'BLOCKED_OR_INVALID','error_type':type(exc).__name__,'error':str(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
