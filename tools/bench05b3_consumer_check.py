"""Installed B3 consumer: recorded-only inputs, checkout/ML imports denied in process."""
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
    p=argparse.ArgumentParser()
    for name in ('run','historical','workspace'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--forbidden-root',action='append',required=True);a=p.parse_args();root=a.workspace.absolute();root.mkdir(parents=True,exist_ok=False)
    recorded=root/'recorded';recorded.mkdir();manifest=json.loads((a.run/'RAW_MANIFEST.json').read_text(encoding='utf-8'))
    for name in [*manifest['files'],'RAW_MANIFEST.json']:
        part=Path(name);assert not part.is_absolute() and '..' not in part.parts
        source=a.run/part;assert source.is_file() and not source.is_symlink();dest=recorded/part;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    shutil.copyfile(a.historical,root/'historical-arc.json')
    forbidden=[os.path.abspath(f) for f in a.forbidden_root]
    for name in ('PYTHONPATH','PYTHONHOME'):os.environ.pop(name,None)
    os.chdir(root);sys.path[:]=[p for p in sys.path if p and not any(os.path.abspath(p)==f or os.path.abspath(p).startswith(f+os.sep) for f in forbidden)]
    sdk=('torch','vllm','transformers','numpy','aiohttp','compressed_tensors')
    def audit(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(values[0]))
            if any(path==f or path.startswith(f+os.sep) for f in forbidden):raise AssertionError('Forbidden checkout read: '+path)
        if event=='import' and values and str(values[0]).split('.')[0] in sdk:raise AssertionError('CPU SDK import forbidden')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import bench_b3_replay,bench_b3_score,task_adapter
    from inference_kit.core.io import read,write,digest
    from inference_kit.core.quality import likelihood_from_response
    from inference_kit.cli import main as cli
    origin=Path(inference_kit.__file__).absolute();assert 'site-packages' in origin.parts and origin.is_relative_to(Path(sys.prefix))
    summary=bench_b3_replay(recorded,root/'api-replay');calls=[]
    def command(args,expected=0):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:code=cli(args)
            except SystemExit as exc:code=exc.code
        assert code==expected,(code,err.getvalue());calls.append({'args':args,'code':code});return out.getvalue(),err.getvalue()
    command(['bench','b3','--help']);text,_=command(['bench','b3','replay','--run','recorded','--output','cli-replay']);assert json.loads(text)==json.loads(json.dumps(summary))
    command(['bench','b3','replay','--run','recorded','--output','cli-replay'],2)
    fixture=read(root/'historical-arc.json');adapter=task_adapter('arc_challenge','1.0');computed=[]
    for arm in ('bf16','rtn'):
        for sample in fixture['documents']:
            rows=[]
            for req in sample['requests']:
                record=next(r for r in fixture['raw'][arm] if r['sample_id']==sample['sample_id'] and r['choice_index']==req['choice_index'])
                assert record['request']['prompt']==req['input_ids'] and record['request_identity']==digest(record['request'])
                assert record['context_ids']==req['context_ids'] and record['continuation_ids']==req['continuation_ids']
                ll,tokens=likelihood_from_response(record['response'],req)
                assert abs(ll-record['loglikelihood'])<=1e-10 and tokens==record['continuation_token_logprobs']
                rows.append({'status':'COMPLETE','sample_id':sample['sample_id'],'choice_index':req['choice_index'],'choice_loglikelihood':ll,'continuation_token_count':len(tokens)})
            scored=adapter.score(sample,rows);expected=fixture['expected'][arm][sample['sample_id']]
            for key in ('acc','acc_norm','prediction_acc','prediction_acc_norm','choice_loglikelihood','choice_normalized_scores'):assert scored[key]==expected[key],key
            assert scored['document_serial_elapsed_seconds'] is None
            computed.append({'arm':arm,'sample_id':sample['sample_id'],'acc':scored['acc'],'acc_norm':scored['acc_norm'],'evidence_kind':'historical_native_likelihood_recalculation','new_gpu':False})
            if arm=='bf16' and sample==fixture['documents'][0]:
                write(root/'sample.json',sample);write(root/'choices.json',rows);target=scored
    text,_=command(['bench','b3','score','--sample','sample.json','--choices','choices.json','--output','cli-score']);assert json.loads(text)==target==bench_b3_score(read(root/'sample.json'),read(root/'choices.json'))
    child=subprocess.run([str(Path(sys.executable).parent/'inference-kit'),'bench','b3','score','--sample','sample.json','--choices','choices.json','--output','console-score'],capture_output=True,encoding='utf-8',timeout=30)
    assert child.returncode==0 and json.loads(child.stdout)==target
    assert not any(n.split('.')[0] in sdk for n in sys.modules)
    receipt={'status':'B3_INSTALLED_CPU_CONSUMER_PASS','package_version':inference_kit.__version__,'package_import_origin':str(origin),'interpreter':sys.executable,'commands':calls,
      'current_run_summary':summary,'historical_arc_parity_documents':computed,'current_live_status':summary['live_status'],'historical_is_not_current_measurement':True,
      'raw_copy_before_gate':True,'checkout_reads':'DENIED_IN_PROCESS_AUDIT_HOOK','sdk_imports':[],'console_subprocess':'PASS; audit hook not inherited, no OS sandbox claim','new_gpu_runs':0,'downloads':0}
    write(root/'CONSUMER_RECEIPT.json',receipt);print(json.dumps({'status':receipt['status'],'version':inference_kit.__version__,'current_live_status':summary['live_status'],'historical_documents':len(computed)},ensure_ascii=True))

if __name__=='__main__':main()
