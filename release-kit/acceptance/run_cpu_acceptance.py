"""Explicit installed-wheel CPU acceptance, no archive code discovery."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mode',required=True,choices=('api','module','console'))
    a=p.parse_args();kit=a.kit.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('New output directory required')
    if out==kit or kit/'fixtures'==out or (kit/'fixtures') in out.parents:
        raise ValueError('Output cannot replace kit/fixtures')
    out.mkdir(parents=True)
    import inference_kit
    from inference_kit.analysis.tokens import source_identity
    from inference_kit.core.parity import equal
    origin=Path(inference_kit.__file__).resolve()
    if 'site-packages' not in origin.parts:raise ValueError('Ordinary wheel install required')
    expected=json.loads((kit/'RELEASE_IDENTITY.json').read_text(encoding='utf-8'))
    assert inference_kit.__version__==(kit/'VERSION').read_text().strip()==expected['version']
    assert importlib.metadata.version('inference-engineering-kit')==inference_kit.__version__
    assert source_identity()==expected['python_source_digest']
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    def command(argv,name):
        result=subprocess.run(argv,cwd=kit,env=env,capture_output=True,text=True,encoding='utf-8')
        (out/(name+'.stdout')).write_text(result.stdout,encoding='utf-8')
        (out/(name+'.stderr')).write_text(result.stderr,encoding='utf-8')
        commands.append({'name':name,'argv':argv,'returncode':result.returncode})
        if result.returncode:raise AssertionError(name+': '+result.stderr[-3000:])
    for filename,name in [('replay_saved.py','saved'),('replay_plans.py','plans'),('replay_publication.py','publication')]:
        command([sys.executable,'-I','-B',str(kit/'acceptance'/filename),'--kit',str(kit),'--output',str(out/name),'--mode',a.mode],name)
    command([sys.executable,'-I','-B',str(kit/'acceptance/negative_contracts.py'),'--kit',str(kit),'--output',str(out/'negative-contracts')],'negative-contracts')
    command([sys.executable,'-I','-B',str(kit/'acceptance/copy_error_regression.py'),'-v'],'copy-ownership')
    ml=[m for m in sys.modules if m.split('.')[0] in {'torch','transformers','vllm','tokenizers','pynvml','numpy'}]
    assert not ml
    value={'status':'CPU_QUICKSTART_ACCEPTANCE_PASS','mode':a.mode,'version':inference_kit.__version__,'source_digest':source_identity(),'package_origin':str(origin),'metadata_version':importlib.metadata.version('inference-engineering-kit'),'commands':commands,'ml_modules_loaded_parent':ml,'guard_scope':'Each helper records its in-process guard. Actual CLI children are separate processes; this is not an OS sandbox.','acquisition_scope':'Saved actual A/B slices; synthetic D publication/index demo explicitly separate. No new model measurement.','gpu_query':0,'model_load':0,'inference_post':0,'fresh_tokenizer':0,'campaign_debit':0}
    (out/'CPU_ACCEPTANCE_RECEIPT.json').write_text(json.dumps(value,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,ensure_ascii=True))

if __name__=='__main__':main()
