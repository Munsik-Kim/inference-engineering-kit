"""Run with an ordinarily installed wheel, outside checkout; deny checkout reads.

The explicit recorded fixture/example are copied before the deny gate. Package
operations below cannot read either checkout. No PYTHONPATH or SDK is needed.
"""
import argparse
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True,type=Path);p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--example',required=True,type=Path);p.add_argument('--forbidden-root',action='append',required=True);a=p.parse_args()
    root=a.workspace.absolute();root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.fixture,root/'recorded');shutil.copyfile(a.example,root/'consumer_project.py')
    forbidden=[os.path.abspath(s) for s in a.forbidden_root]
    for name in ('PYTHONPATH','PYTHONHOME'):os.environ.pop(name,None)
    os.chdir(root)
    sys.path[:]=[s for s in sys.path if s and not any(os.path.abspath(s)==f or os.path.abspath(s).startswith(f+os.sep) for f in forbidden)]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(args[0]))
            if any(path==f or path.startswith(f+os.sep) for f in forbidden):raise AssertionError('Consumer accessed forbidden checkout: '+path)
        if event=='import' and args and str(args[0]).split('.')[0] in ('torch','vllm','transformers'):
            raise AssertionError('CPU consumer imported an engine SDK')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit.cli import main as cli
    origin=Path(inference_kit.__file__).absolute()
    assert 'site-packages' in origin.parts and not any(str(origin).startswith(f+os.sep) for f in forbidden),origin
    commands=[]
    def command(argv,expected=0):
        stdout,stderr=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            try:rc=cli(argv)
            except SystemExit as e:rc=e.code
        assert rc==expected,(argv,rc,stderr.getvalue())
        commands.append({'argv':['inference-kit',*argv],'returncode':rc})
    command(['--help']);command(['demo','--output','demo'])
    command(['compare','--baseline','demo/baseline.json','--candidate','demo/candidate.json','--output','compare'])
    command(['assess','--index','demo/index.json','--requirements','demo/requirements.json','--output','assess'])
    command(['verify','--bundle','recorded','--output','verify'])
    command(['replay','--case','step04','--bundle','recorded','--output','replay'])
    spec={'schema_version':1,'backend':'recorded','model':{'model_id':'own-synthetic','revision':'2'*40,'architecture':'synthetic'},
          'method':{'name':'synthetic','format':'native','activation_dtype':'bfloat16','kv_dtype':'bfloat16'},
          'profile':{'context':512,'max_num_seqs':2,'max_num_batched_tokens':512,'kv_tokens':1024,'output_tokens':16,'concurrency':2,
                     'shutdown_timeout':10,'cleanup_grace':25,'chunked_prefill':True,'eager':True,'prefix_caching':False,'async_scheduling':False,'seed':71},
          'paths':{'workload':'own-data.json','output':'own-run'},'budget':{'wall_seconds':120,'cleanup_reserve_seconds':60,'automatic_retries':0}}
    (root/'own-spec.json').write_text(json.dumps(spec))
    command(['plan','--spec','own-spec.json'])
    command(['demo','--output','demo'],2)
    sys.argv=['consumer_project.py','--output','api-consumer'];runpy.run_path(str(root/'consumer_project.py'),run_name='__main__')
    console=subprocess.run([str(Path(sys.executable).parent/'inference-kit'),'--help'],capture_output=True,text=True,cwd=root,timeout=10)
    assert console.returncode==0 and 'assess' in console.stdout
    forbidden_imports=[n for n in sys.modules if n.split('.')[0] in ('torch','vllm','transformers','aiohttp')];assert not forbidden_imports
    result={'status':'INDEPENDENT_WHEEL_CONSUMER_PASS','package_version':inference_kit.__version__,'import_origin':str(origin),
            'python':sys.version,'commands':commands,'console_entrypoint':'PASS','editable':False,'original_checkout_reads':'DENIED_BY_AUDIT_HOOK',
            'forbidden_roots':forbidden,'explicit_fixture':'2-document recorded slice copied before deny gate',
            'engine_sdk_imports':forbidden_imports,'core_modified_for_custom_workload':False,'new_gpu_runs':0,'model_downloads':0}
    (root/'CONSUMER_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
