"""Installed B4 consumer; copied raw only, no checkout fallback or ML imports.

The in-process audit hook does not protect the separately checked console child
or act as an operating-system sandbox.
"""
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
    parser=argparse.ArgumentParser()
    for name in ('run','workspace'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--forbidden-root',action='append',required=True)
    args=parser.parse_args();root=args.workspace.absolute();root.mkdir(parents=True,exist_ok=False)
    recorded=root/'recorded';recorded.mkdir()
    manifest=json.loads((args.run/'RAW_MANIFEST.json').read_text(encoding='utf-8'))
    names=[*manifest['files'],'RAW_MANIFEST.json'];seen=set()
    for name in names:
        part=Path(name)
        assert not part.is_absolute() and '..' not in part.parts and '\\' not in name
        assert name.casefold() not in seen;seen.add(name.casefold())
        source=args.run/part;assert source.is_file() and not source.is_symlink()
        destination=recorded/part;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    forbidden=[os.path.abspath(value) for value in args.forbidden_root]
    for name in ('PYTHONPATH','PYTHONHOME'):os.environ.pop(name,None)
    os.chdir(root)
    sys.path[:]=[value for value in sys.path if value and not any(os.path.abspath(value)==f or os.path.abspath(value).startswith(f+os.sep) for f in forbidden)]
    sdk=('torch','vllm','transformers','numpy','aiohttp','compressed_tensors')
    def audit(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            filename=os.path.abspath(os.fsdecode(values[0]))
            if any(filename==f or filename.startswith(f+os.sep) for f in forbidden):raise AssertionError('Forbidden checkout read: '+filename)
        if event=='import' and values and str(values[0]).split('.')[0] in sdk:raise AssertionError('CPU SDK import forbidden')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import bench_b4_replay,bench_b4_score,task_adapter
    from inference_kit.core.io import read,write
    from inference_kit.cli import main as cli
    origin=Path(inference_kit.__file__).absolute()
    assert 'site-packages' in origin.parts and origin.is_relative_to(Path(sys.prefix))
    registry={name:task_adapter(name).request_kind for name in ('tech_record_extract','arc_challenge','context_evidence_retrieval')}
    assert registry=={'tech_record_extract':'generate_until','arc_challenge':'choice_loglikelihood','context_evidence_retrieval':'generate_until'}
    calls=[]
    def command(argv,expected=0):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:code=cli(argv)
            except SystemExit as exc:code=exc.code
        assert code==expected,(code,err.getvalue());calls.append({'args':argv,'code':code})
        return out.getvalue(),err.getvalue()
    summary=bench_b4_replay(recorded,root/'api-replay')
    command(['bench','b4','--help'])
    text,_=command(['bench','b4','replay','--run','recorded','--output','cli-replay'])
    assert json.loads(text)==json.loads(json.dumps(summary))
    command(['bench','b4','replay','--run','recorded','--output','cli-replay'],2)
    plan=read(recorded/'PLAN_FROZEN.json');adapter=task_adapter('context_evidence_retrieval','0.1')
    sample=plan['data']['rows'][0];generation=None;score_evidence='CPU_POSITIVE_CONTROL_ONLY'
    for filename in sorted(recorded.glob('attempts/serve-*-attempt1/task-generations.jsonl')):
        rows=[json.loads(line) for line in filename.read_text(encoding='utf-8').splitlines() if line]
        if rows:
            generation=rows[0];sample=next(row for row in plan['data']['rows'] if row['sample_id']==generation['sample_id'])
            score_evidence='CURRENT_RAW_GENERATION';break
    if generation is None:
        generation={'task_id':sample['task_id'],'task_version':sample['task_version'],'sample_id':sample['sample_id'],
                    'sample_identity':sample['sample_identity'],'request_kind':'generate_until',
                    'output_text':adapter.reference(sample)['text'],'transport_complete':True,'termination':'NATURAL_STOP',
                    'request_latency_http':.1,'evidence_kind':'CPU_POSITIVE_CONTROL_ONLY'}
    target=bench_b4_score(sample,generation)
    write(root/'sample.json',sample);write(root/'generation.json',generation)
    text,_=command(['bench','b4','score','--sample','sample.json','--generation','generation.json','--output','cli-score'])
    assert json.loads(text)==target
    child=subprocess.run([str(Path(sys.executable).parent/'inference-kit'),'bench','b4','score',
                          '--sample','sample.json','--generation','generation.json','--output','console-score'],
                         capture_output=True,encoding='utf-8',timeout=30)
    assert child.returncode==0,(child.returncode,child.stderr)
    assert json.loads(child.stdout)==target
    assert not any(name.split('.')[0] in sdk for name in sys.modules)
    receipt={'status':'B4_INSTALLED_CPU_CONSUMER_PASS','package_version':inference_kit.__version__,
             'package_import_origin':str(origin),'interpreter':sys.executable,'isolated_mode':bool(sys.flags.isolated),
             'registry':registry,'commands':calls,'summary':summary,'score_evidence':score_evidence,
             'raw_copy_before_gate':True,'checkout_reads':'DENIED_IN_PROCESS_AUDIT_HOOK','sdk_imports':[],
             'console_subprocess':'PASS; audit hook not inherited; no OS sandbox claim',
             'json_representation':'Only tuple/array serialization normalized via JSON; values, order, duplicates, null and types unchanged',
             'new_gpu_runs':0,'downloads':0}
    write(root/'CONSUMER_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'version':inference_kit.__version__,
                      'score_evidence':score_evidence,'current_live_status':summary['live_status']},ensure_ascii=True))


if __name__=='__main__':main()
