"""Installed B1 consumer. Explicit raw copied before in-process checkout/SDK read gate."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',required=True,type=Path);parser.add_argument('--workspace',required=True,type=Path);parser.add_argument('--forbidden-root',action='append',required=True);a=parser.parse_args()
    root=a.workspace.absolute();root.mkdir(parents=True,exist_ok=False);recorded=root/'recorded';recorded.mkdir()
    manifest=json.loads((a.fixture/'RAW_MANIFEST.json').read_text(encoding='utf-8'))
    for name in [*manifest['files'],'RAW_MANIFEST.json']:
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        src=a.fixture/p;assert src.is_file() and not src.is_symlink();dest=recorded/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    forbidden=[os.path.abspath(p) for p in a.forbidden_root]
    for name in ('PYTHONPATH','PYTHONHOME'):os.environ.pop(name,None)
    os.chdir(root);sys.path[:]=[p for p in sys.path if p and not any(os.path.abspath(p)==f or os.path.abspath(p).startswith(f+os.sep) for f in forbidden)]
    sdks=('torch','vllm','transformers','numpy','aiohttp','compressed_tensors')
    def audit(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(values[0]))
            if any(path==f or path.startswith(f+os.sep) for f in forbidden):raise AssertionError('Forbidden checkout read: '+path)
        if event=='import' and values and str(values[0]).split('.')[0] in sdks:raise AssertionError('CPU consumer SDK import')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import bench_b1_replay,bench_b1_score
    from inference_kit.cli import main as cli
    from inference_kit.core.io import read,write
    origin=Path(inference_kit.__file__).absolute();assert 'site-packages' in origin.parts
    result=bench_b1_replay(recorded,root/'api-replay');commands=[]
    def command(argv,expected=0):
        stdout,stderr=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:code=cli(argv)
            except SystemExit as exc:code=exc.code
        assert code==expected,(argv,code,stderr.getvalue());commands.append({'argv':argv,'returncode':code})
        return stdout.getvalue(),stderr.getvalue()
    # MetricSpec uses tuples in Python and arrays on the JSON wire; scalar/count/status stay exact.
    serialized_result=json.loads(json.dumps(result,allow_nan=False))
    command(['bench','b1','--help']);text,_=command(['bench','b1','replay','--run','recorded','--output','cli-replay']);assert json.loads(text)==serialized_result
    command(['bench','b1','replay','--run','recorded','--output','cli-replay'],2)
    plan=read(recorded/'PLAN_FROZEN.json');job=next(v for k,v in read(recorded/'progress.json')['jobs'].items() if k.startswith('serve-'))
    generation=json.loads((recorded/job['directory']/'generations.jsonl').read_text(encoding='utf-8').splitlines()[0]);sample=next(r for r in plan['data']['rows'] if r['sample_id']==generation['sample_id'])
    write(root/'sample.json',sample);write(root/'generation.json',generation);scored=bench_b1_score(sample,generation)
    text,_=command(['bench','b1','score','--sample','sample.json','--generation','generation.json','--output','cli-score']);assert json.loads(text)==scored==read(root/'cli-score/score.json')
    console=Path(sys.executable).parent/'inference-kit'
    child=subprocess.run([str(console),'bench','b1','score','--sample','sample.json','--generation','generation.json','--output','console-score'],capture_output=True,encoding='utf-8',timeout=30)
    assert child.returncode==0 and json.loads(child.stdout)==scored
    imported=[n for n in sys.modules if n.split('.')[0] in sdks];assert not imported
    receipt={'status':'B1_INDEPENDENT_WHEEL_CONSUMER_PASS','package_version':inference_kit.__version__,'package_import_origin':str(origin),
      'interpreter':sys.executable,'commands':commands,'external_console_score':'PASS','sdk_imports':imported,'checkout_reads':'DENIED_BY_IN_PROCESS_AUDIT_HOOK',
      'audit_scope':'Not inherited by console subprocess; not an OS sandbox','explicit_raw_copy_before_gate':True,
      'summary':serialized_result,'api_cli_parity':'JSON tuple/array representation normalized; scalar/count/status exact','gpu_actions':0,'downloads':0}
    write(root/'CONSUMER_RECEIPT.json',receipt);print(json.dumps({'status':receipt['status'],'version':inference_kit.__version__,'probe':result['probe']},ensure_ascii=True))


if __name__=='__main__':main()
