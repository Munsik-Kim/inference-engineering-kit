"""Installed C2 CPU consumer: explicitly copied raw -> token view -> comparison.

Usage: python -I -B bench05c2_consumer_check.py --run RUN --output NEW_WORKSPACE
       --forbidden-root TOOLKIT_CHECKOUT --forbidden-root ORIGINAL_LAB

The audit hook is in-process only. The separately verified console child does
not inherit the hook and this helper does not claim an OS sandbox.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def strict_json(path):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('Duplicate JSON key: '+key)
            value[key]=item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Invalid JSON constant: '+value)))


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def copy_raw(source,destination):
    source=Path(source).absolute()
    if not source.is_dir() or source.is_symlink():raise ValueError('Explicit non-symlink raw directory required')
    manifest=strict_json(source/'RAW_MANIFEST.json')
    if manifest.get('scope')!='token_probe_raw_v1' or not isinstance(manifest.get('files'),dict):raise ValueError('C2 raw manifest required')
    files=list(manifest['files'].items())+[('RAW_MANIFEST.json',{'bytes':(source/'RAW_MANIFEST.json').stat().st_size,'sha256':file_sha(source/'RAW_MANIFEST.json')})]
    destination.mkdir(parents=True,exist_ok=False);seen=set();total=0
    if len(files)>10000:raise ValueError('Consumer file count limit')
    for name,entry in files:
        path=Path(name)
        if not isinstance(name,str) or not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or path.as_posix()!=name:
            raise ValueError('Unsafe raw member')
        if name.casefold() in seen:raise ValueError('Duplicate raw member')
        seen.add(name.casefold());member=source/path
        if not member.is_file() or member.is_symlink() or not member.resolve().is_relative_to(source.resolve()):raise ValueError('Missing/unsafe raw member')
        size=entry['bytes']
        if type(size) is not int or not 0<=size<=64*1024**2:raise ValueError('Consumer member size limit')
        if member.stat().st_size!=size or file_sha(member)!=entry['sha256']:raise ValueError('Original raw member identity mismatch')
        total+=size
        if total>512*1024**2:raise ValueError('Consumer raw total size limit')
        target=destination/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(member,target)
        if target.stat().st_size!=size or file_sha(target)!=entry['sha256']:raise ValueError('Copied raw identity mismatch')
    return {'manifest_sha256':file_sha(destination/'RAW_MANIFEST.json'),'members':len(seen),'bytes':total,
            'scope':'full explicitly manifested raw; no model/cache/environment directory copied'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output','--workspace',dest='output',required=True,type=Path)
    parser.add_argument('--forbidden-root',action='append',required=True)
    parser.add_argument('--baseline',default='bf16');parser.add_argument('--candidate',default='rtn')
    parser.add_argument('--expected-version')
    args=parser.parse_args()
    if not sys.flags.isolated:raise ValueError('Invoke the installed CPU interpreter with -I')
    root=args.output.absolute();forbidden=[os.path.abspath(p) for p in args.forbidden_root]
    if any(str(root)==p or str(root).startswith(p+os.sep) for p in forbidden):raise ValueError('Consumer output must be outside original checkouts')
    root.mkdir(parents=True,exist_ok=False);recorded=root/'recorded';copied=copy_raw(args.run,recorded)
    plan=strict_json(recorded/'PLAN_FROZEN.json');source={'source_id':'c2_configured_consumer','adapter':'token_probe_raw_v1',
        'path':str(recorded),'raw_manifest_sha256':copied['manifest_sha256']}
    sources=root/'copied-sources.json'
    with sources.open('x',encoding='utf-8') as handle:
        json.dump({'schema_version':1,'kind':'token_input_sources','sources':[source]},handle,ensure_ascii=False,allow_nan=False,indent=2)
        handle.write('\n')
    for name in ('PYTHONPATH','PYTHONHOME'):os.environ.pop(name,None)
    os.chdir(root)
    sys.path[:]=[p for p in sys.path if p and not any(os.path.abspath(p)==f or os.path.abspath(p).startswith(f+os.sep) for f in forbidden)]
    sdk=('torch','vllm','transformers','tensorflow','numpy','aiohttp','compressed_tensors','safetensors','accelerate')
    permitted={'console':False};audit_counts={'checkout':0,'ml_sdk':0,'network':0,'unexpected_subprocess':0}
    def audit(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            path=os.path.abspath(os.fsdecode(values[0]))
            if any(path==f or path.startswith(f+os.sep) for f in forbidden):
                audit_counts['checkout']+=1;raise AssertionError('Consumer original checkout read forbidden')
        if event=='import' and values and str(values[0]).split('.')[0] in sdk:
            audit_counts['ml_sdk']+=1;raise AssertionError('Consumer ML SDK import forbidden')
        if event in ('socket.connect','socket.getaddrinfo','socket.gethostbyname'):
            audit_counts['network']+=1;raise AssertionError('Consumer network forbidden')
        if event in ('subprocess.Popen','os.system','os.posix_spawn') and not permitted['console']:
            audit_counts['unexpected_subprocess']+=1;raise AssertionError('Unexpected consumer subprocess forbidden')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import tokens_export,tokens_compare
    from inference_kit.cli import main as cli
    from inference_kit.core.io import read,write
    from inference_kit.core.parity import equal
    origin=Path(inference_kit.__file__).absolute()
    assert 'site-packages' in origin.parts and origin.is_relative_to(Path(sys.prefix)),str(origin)
    if args.expected_version:assert inference_kit.__version__==args.expected_version
    calls=[]
    def json_shape(value):return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    def command(argv,expected=0):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:code=cli(argv)
            except SystemExit as exc:code=exc.code
        assert code==expected,(code,err.getvalue());calls.append({'args':argv,'code':code})
        return out.getvalue()
    exported=tokens_export(sources,root/'api-export')
    summary=tokens_compare(root/'api-export',args.baseline,args.candidate,root/'api-compare')
    assert exported['coverage']['generation_records']==0
    token_records=[json.loads(line) for line in (root/'api-export/token_records.jsonl').read_text(encoding='utf-8').splitlines() if line]
    expected_tasks={row['task_id'] for row in plan['samples']['rows']}
    assert {row['task_id'] for row in token_records}==expected_tasks
    expected_positions=sum(len(row['targets']) for row in plan['samples']['rows'])
    expected_full=sum(len(row['full_vocab_target_positions']) for row in plan['samples']['rows'])
    rows=[json.loads(line) for line in (root/'api-compare/token-comparisons.jsonl').read_text(encoding='utf-8').splitlines() if line]
    assert len(rows)==expected_positions
    progress=read(recorded/'progress.json');complete=all(progress.get('jobs',{}).get('main-'+arm,{}).get('status')=='COMPLETE' for arm in (args.baseline,args.candidate))
    if complete:
        assert summary['coverage']['measured']==expected_positions and summary['coverage']['incomparable']==0 and summary['coverage']['unknown']==0
    full=[row for row in rows if row['coverage']['full_logits']]
    if complete:assert len(full)==expected_full
    tolerance=plan['probe_spec']['consistency_policy']['absolute_tolerance_nat']
    for row in full:
        for side in ('reference','candidate'):
            check=row['metrics']['full_consistency'][side]
            assert check['status']=='MEASURED' and check['within_declared_tolerance'] is True and check['absolute_tolerance_nat']==tolerance
        assert row['metrics']['full_distribution']['status']=='MEASURED'
    for row in token_records:
        if row['status']=='MEASURED':assert row['target_logprob']<=0
        else:assert row['target_logprob'] is None
    command(['bench','tokens','--help'])
    text=command(['bench','tokens','export','--sources','copied-sources.json','--output','cli-export'])
    equal(json.loads(text),json_shape(exported));equal(read(root/'api-export/TOKEN_VIEW_MANIFEST.json'),read(root/'cli-export/TOKEN_VIEW_MANIFEST.json'))
    command(['bench','tokens','export','--sources','copied-sources.json','--output','cli-export'],2)
    text=command(['bench','tokens','compare','--view','api-export','--baseline',args.baseline,'--candidate',args.candidate,'--output','cli-compare'])
    equal(read(root/'cli-compare/summary.json'),json_shape(summary))
    selected={key:json_shape(summary)[key] for key in json.loads(text)};equal(json.loads(text),selected)
    command(['bench','tokens','compare','--view','api-export','--baseline',args.baseline,'--candidate',args.candidate,'--output','cli-compare'],2)
    argv=[str(Path(sys.executable).parent/'inference-kit'),'bench','tokens','compare','--view','api-export','--baseline',args.baseline,'--candidate',args.candidate,'--output','console-compare']
    permitted['console']=True
    try:child=subprocess.run(argv,capture_output=True,encoding='utf-8',timeout=120)
    finally:permitted['console']=False
    assert child.returncode==0,(child.returncode,child.stderr)
    equal(read(root/'console-compare/summary.json'),json_shape(summary));equal(json.loads(child.stdout),selected)
    assert not any(name.split('.')[0] in sdk for name in sys.modules)
    receipt={'status':'C2_INSTALLED_CPU_RAW_EXPORT_COMPARE_CONSUMER_PASS','package_version':inference_kit.__version__,
      'package_import_origin':str(origin),'interpreter':sys.executable,'isolated_mode':True,'raw_copy_before_guard':copied,
      'raw_jobs_complete':complete,'task_ids':sorted(expected_tasks),'requested_pair_positions':expected_positions,
      'measured_pair_positions':summary['coverage']['measured'],'requested_full_pairs':expected_full,'measured_full_pairs':len(full),
      'same_pass_tolerance_nat':tolerance,'same_pass_reference_candidate_checks':len(full)*2,'generation_records':0,
      'api_export_coverage':exported['coverage'],'api_compare_coverage':summary['coverage'],'commands':calls,'console_command':argv,
      'parity':'API/CLI export manifest and semantic numeric summary parity; separate console summary parity; immutable output refusal',
      'float_tolerance':{'absolute':1e-10,'relative':1e-9},'exact':'types,IDs,status,counts; tuple/JSON-array representation only normalized',
      'audit_counts':audit_counts,'checkout':'DENIED_IN_PROCESS_AFTER_EXPLICIT_COPY','ML_SDK_imports':[],'network':'DENIED_IN_PROCESS',
      'subprocess_scope':'One explicit console child allowed; hook is not inherited, no OS sandbox claim',
      'GPU_actions':0,'model_loads':0,'tokenizer_loads':0,'downloads':0}
    write(root/'CONSUMER_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'version':inference_kit.__version__,'package_import_origin':str(origin),
                     'task_ids':receipt['task_ids'],'paired_positions':receipt['measured_pair_positions'],
                     'full_pairs':receipt['measured_full_pairs'],'same_pass_checks':receipt['same_pass_reference_candidate_checks']},ensure_ascii=True))


if __name__=='__main__':main()
