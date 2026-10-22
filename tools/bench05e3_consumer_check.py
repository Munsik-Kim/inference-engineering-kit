"""Explicit installed-wheel E3 full-raw or declared MINI CPU replay consumer.

No archive code discovery. Console is checked separately before an in-process
checkout/SDK/network/subprocess audit hook; the hook is not an OS sandbox.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(path,root):return path==root or root in path.parents


def main():
    parser=argparse.ArgumentParser()
    for name in ('run','output','expected-origin'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--expected-version',default='0.1.1.dev13');parser.add_argument('--deny-root',action='append',default=[],type=Path)
    args=parser.parse_args();assert sys.flags.isolated
    source=args.run.absolute();output=args.output.absolute();denied=[p.absolute() for p in args.deny_root]
    assert source.is_dir() and not source.is_symlink() and not output.exists()
    assert not any(inside(source,p) or inside(output,p) for p in denied)
    output.mkdir(parents=True)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    console=Path(sys.executable).parent/'inference-kit';assert console.is_file()
    command=[str(console),'bench','suite','replay','--run',str(source),'--output',str(output/'console')]
    child=subprocess.run(command,cwd=output,env=env,capture_output=True,timeout=240)
    assert child.returncode==0,child.stderr.decode('ascii',errors='backslashreplace')
    child_value=json.loads(child.stdout.decode('ascii'))
    counts={'checkout':0,'SDK':0,'network':0,'subprocess':0};denied_reads=[]
    forbidden={'torch','numpy','transformers','vllm','aiohttp','compressed_tensors','tokenizers','safetensors'}
    def guard(event,values):
        if event in ('open','os.scandir','os.listdir') and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=Path(os.path.abspath(os.fsdecode(values[0])))
            if any(inside(path,p) for p in denied):
                counts['checkout']+=1;denied_reads.append({'event':event,'path':str(path)});raise PermissionError('Original checkout read blocked')
        if event=='import' and values and str(values[0]).split('.')[0] in forbidden:counts['SDK']+=1;raise ImportError('ML/HTTP SDK blocked')
        if event in ('socket.__new__','socket.connect','socket.bind','socket.getaddrinfo'):counts['network']+=1;raise PermissionError('Network blocked')
        if event in ('subprocess.Popen','os.system','os.posix_spawn'):counts['subprocess']+=1;raise PermissionError('Subprocess blocked')
    # Trusted replay uses descriptor-relative temporary-directory cleanup.
    # Keep the process cwd outside the denied checkout for relative audit names.
    os.chdir(output)
    sys.addaudithook(guard)
    import inference_kit
    from inference_kit import suite_replay
    origin=Path(inference_kit.__file__).absolute();assert inside(origin,args.expected_origin.absolute()) and 'site-packages' in origin.parts
    assert inference_kit.__version__==args.expected_version and not forbidden.intersection(n.split('.')[0] for n in sys.modules)
    value=suite_replay(source,output/'api')
    # Source/runtime paths and analysis revision locators are provenance. Compare
    # saved scalar/count/status/type/order fields, never broad stringify/sorting.
    def semantic(folder):
        paths=['score_task/document-scores.jsonl','task-scores/document-scores.jsonl','scores/document-scores.jsonl',
               'analyze_tokens/token-metrics/token-comparisons.jsonl','token-metrics/token-comparisons.jsonl','tokens/token-metrics/token-comparisons.jsonl']
        found={}
        for relative in paths:
            path=folder/relative
            if path.is_file():
                with path.open(encoding='utf-8') as stream:found[relative]=[json.loads(line) for line in stream if line.strip()]
        assert found,'No replay document/token outputs found'
        return found
    from inference_kit.core.parity import equal
    left,right=semantic(output/'api'),semantic(output/'console');assert set(left)==set(right)
    for name in left:equal(left[name],right[name],path=name)
    receipt={'status':'PASS','version':inference_kit.__version__,'package_import_origin':str(origin),'interpreter':sys.executable,
        'API_console_result_status':[value.get('status'),child_value.get('status')],
        'exact_or_float_tolerance_parity':list(left),'row_counts':{k:len(v) for k,v in left.items()},'guard':counts,
        'blocked_checkout_attempts':denied_reads,
        'audit_scope':'In-process API only; console subprocess checked separately before hook, not OS/child sandbox',
        'GPU_queries':0,'model_loads':0,'tokenizer_calls':0,'network_requests':0,'new_acquisitions':0,
        'scope':'Declared full new raw or MINI slice only; MINI never certifies complete live resume/full110/44 coverage'}
    (output/'CONSUMER_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=True))

if __name__=='__main__':main()
