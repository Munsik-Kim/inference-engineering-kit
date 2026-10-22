"""Explicit installed-wheel E4A prepared-contract consumer; no retokenization.

Copies only declared/pinned prepared members. Separate module/console subprocess
checks precede the in-process checkout/SDK/network/launch audit hook. The hook is
not an OS sandbox and does not protect or automatically propagate to children.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys


def inside(path, root):
    return path == root or root in path.parents


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(name):
    path = PurePosixPath(name)
    assert name and not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name and str(path) == name
    return path


def copy_prepared(manifest, output):
    """Copy a complete bundle inventory or an explicitly standalone source set."""
    import hashlib
    def file_hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=manifest.absolute();root=manifest.parent.resolve();value=read(manifest)
    assert manifest.is_file() and not manifest.is_symlink()
    if value['kind']=='prepared_input_bundle_v1':
        entries=value['files'];sources=value['source_manifest']
    else:
        assert value['kind']=='suite_prepared_sources_v1'
        entries=[{'path':s['path'],'sha256':s['sha256']} for s in value['sources']];sources=manifest.name
    names=[item['path'] for item in entries]
    assert names and len(names)==len(set(names))
    assert manifest.name not in names
    output.mkdir(parents=True)
    for item in entries:
        relative=safe(item['path']);origin=root/relative
        assert origin.is_file() and not origin.is_symlink() and inside(origin.resolve(),root)
        assert file_hash(origin)==item['sha256']
        assert 'bytes' not in item or type(item['bytes']) is int and origin.stat().st_size==item['bytes']
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target)
    shutil.copyfile(manifest,output/manifest.name)
    return output/manifest.name,output/safe(sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','suite','local','output','expected-origin'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--expected-version',default='0.1.1.dev15')
    parser.add_argument('--deny-root',action='append',type=Path,default=[])
    args=parser.parse_args();assert sys.flags.isolated,'Run this consumer using python -I'
    os.environ.pop('PYTHONPATH',None);os.environ.pop('PYTHONHOME',None)
    output=args.output.absolute();assert not output.exists();output.mkdir(parents=True)
    denied=[p.absolute() for p in args.deny_root]
    assert not any(inside(output,root) for root in denied),'Consumer output must be outside original checkout'
    prepared,sources=copy_prepared(args.prepared,output/'input')
    suite=output/'SUITE.json';local=output/'LOCAL_METADATA_ONLY.json'
    shutil.copyfile(args.suite,suite);shutil.copyfile(args.local,local)
    local_value=read(local)
    assert local_value=={'schema_version':1,'artifact_check':'not_requested','artifact_roots':{},'runtime_check':'not_run','authorization':'NONE'},'Consumer requires metadata-only local binding, no artifact/ledger access'
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    child_receipts=[]
    def command(arguments, *, console=False):
        prefix=[str(Path(sys.executable).parent/'inference-kit')] if console else [sys.executable,'-I','-B','-m','inference_kit']
        child=subprocess.run(prefix+arguments,cwd=output,env=env,capture_output=True,timeout=180)
        assert child.returncode==0,child.stderr.decode('ascii',errors='backslashreplace')
        value=json.loads(child.stdout.decode('ascii'))
        child_receipts.append({'entry':'console' if console else 'python -I -m inference_kit','arguments':arguments,'returncode':child.returncode})
        return value
    cli_root=output/'cli'
    cli_input=command(['bench','inputs','validate','--prepared',str(prepared),'--output',str(cli_root/'inputs-validate')])
    cli_description=command(['bench','inputs','describe','--prepared',str(prepared),'--output',str(cli_root/'inputs-describe')])
    cli_plan=command(['bench','suite','plan','--suite',str(suite),'--sources',str(sources),'--local',str(local),'--output',str(cli_root/'plan')])
    cli_validation=command(['bench','suite','validate','--plan',str(cli_root/'plan/FROZEN_PLAN.json'),'--sources',str(sources),'--output',str(cli_root/'validate')])
    cli_plan_description=command(['bench','suite','describe','--plan',str(cli_root/'plan/FROZEN_PLAN.json'),'--output',str(cli_root/'describe')])
    console=Path(sys.executable).parent/'inference-kit';assert console.is_file(),'Ordinary wheel console entry is required'
    console_input=command(['bench','inputs','validate','--prepared',str(prepared),'--output',str(output/'console-inputs-validate')],console=True)
    counts={'checkout':0,'SDK':0,'network':0,'launch':0};blocked=[]
    forbidden={'torch','transformers','tokenizers','numpy','vllm','aiohttp','compressed_tensors','safetensors','tensorflow'}
    def guard(event,values):
        if event in ('open','os.scandir','os.listdir') and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=Path(os.path.abspath(os.fsdecode(values[0])))
            if any(inside(path,root) for root in denied):
                counts['checkout']+=1;blocked.append({'event':event,'path':str(path)});raise PermissionError('Checkout read blocked')
        if event=='import' and values and str(values[0]).split('.')[0] in forbidden:
            counts['SDK']+=1;raise ImportError('SDK import blocked')
        if event in ('socket.__new__','socket.connect','socket.bind','socket.getaddrinfo'):
            counts['network']+=1;raise PermissionError('Network blocked')
        if event in ('subprocess.Popen','os.system','os.posix_spawn'):
            counts['launch']+=1;raise PermissionError('Launch blocked')
    # Descriptor-relative cleanup audit names must resolve outside the checkout.
    os.chdir(output);sys.addaudithook(guard)
    import inference_kit
    from inference_kit import api
    from inference_kit.core.parity import equal
    from inference_kit.core.io import digest
    from inference_kit.suites.bench05a_plan import source_identity
    origin=Path(inference_kit.__file__).absolute()
    assert inside(origin,args.expected_origin.absolute()) and 'site-packages' in origin.parts
    assert inference_kit.__version__==args.expected_version
    assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)
    api_root=output/'api'
    api_input=api.inputs_validate(prepared,api_root/'inputs-validate')
    api_description=api.inputs_describe(prepared,api_root/'inputs-describe')
    api_plan=api.suite_plan(suite,sources,local,api_root/'plan')
    api_validation=api.suite_validate(api_root/'plan/FROZEN_PLAN.json',sources,api_root/'validate')
    api_plan_description=api.suite_describe(api_root/'plan/FROZEN_PLAN.json',api_root/'describe')
    equal(api_input,cli_input,path='inputs_validate');equal(api_description,cli_description,path='inputs_describe')
    equal(api_input,console_input,path='separate_console_inputs_validate')
    equal(api_validation,cli_validation,path='suite_validate');equal(api_plan_description,cli_plan_description,path='suite_describe')
    assert api_plan['plan_identity']==cli_plan['plan_identity']
    equal(api_plan['semantic']['counts'],cli_plan['counts'],path='suite_counts')
    assert api_plan['status']=='VALIDATED_NOT_EXECUTED' and api_plan['execution_count']==api_plan['acquisition_count']==0
    assert api_plan['semantic']['input_execution_scope']['live_binding']=='LIVE_BINDING_NOT_VALIDATED'
    assert not any(counts.values()),counts
    receipt={'status':'PASS','package_version':inference_kit.__version__,'package_import_origin':str(origin),'interpreter':sys.executable,
        'package_source_identity':digest(source_identity()),
        'isolated':bool(sys.flags.isolated),'PYTHONPATH_present':'PYTHONPATH' in os.environ,'PYTHONHOME_present':'PYTHONHOME' in os.environ,
        'prepared_scope':api_input['status'],'sample_count':api_input['sample_count'],'task_count':api_input['task_count'],
        'plan_identity':api_plan['plan_identity'],'counts':api_plan['semantic']['counts'],
        'API_CLI_console_parity':'PASS; exact ID/count/status/type/order with documented float comparison',
        'artifact_presence':'ARTIFACT_PRESENCE_NOT_CHECKED','live_binding':'LIVE_BINDING_NOT_VALIDATED',
        'retokenization':'NOT_RUN; saved prepared contracts only','GPU_queries':0,'model_loads':0,'inference_POSTs':0,'tokenizer_loads':0,'ledger_writes':0,
        'guard':counts,'blocked_attempts':blocked,
        'audit_scope':'In-process API only; separate module/console subprocesses checked before hook, not OS sandbox or child-hook inheritance',
        'subprocess_receipts':child_receipts}
    (output/'CONSUMER_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=True))

if __name__=='__main__':main()
