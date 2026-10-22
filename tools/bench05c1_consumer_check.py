"""Independent installed-wheel C1 consumer over explicitly copied raw and view.

Run using the chosen CPU environment's ``python -I``. The in-process audit hook
blocks source checkout reads, ML SDK imports, network, and unexpected subprocess
creation. It is not an OS sandbox and is NOT inherited by console subprocesses.
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
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def safe_member(name):
    part=Path(name)
    if not isinstance(name,str) or not name or part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name or part.as_posix()!=name:
        raise ValueError('Unsafe copied member')
    return part


def copy_manifest_directory(source,destination,manifest_name,*,max_total=256*1024*1024):
    source=Path(source).absolute()
    if not source.is_dir() or source.is_symlink():raise ValueError('Consumer requires an explicit local run/view directory')
    manifest=strict_json(source/manifest_name);entries=manifest['files'];seen=set();total=0
    if not isinstance(entries,dict):raise ValueError('Manifest file table required')
    destination.mkdir(parents=True,exist_ok=False)
    for name,entry in list(entries.items())+[(manifest_name,{'bytes':(source/manifest_name).stat().st_size,'sha256':file_sha(source/manifest_name)})]:
        relative=safe_member(name)
        if name.casefold() in seen:raise ValueError('Duplicate copied member')
        seen.add(name.casefold());member=source/relative
        if not member.is_file() or member.is_symlink() or not member.resolve().is_relative_to(source.resolve()):raise ValueError('Missing/unsafe source member')
        if type(entry['bytes']) is not int or entry['bytes']<0 or member.stat().st_size!=entry['bytes'] or file_sha(member)!=entry['sha256']:raise ValueError('Source member identity differs')
        total+=entry['bytes']
        if total>max_total:raise ValueError('Consumer raw copy size limit')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(member,target)
        if target.stat().st_size!=entry['bytes'] or file_sha(target)!=entry['sha256']:raise ValueError('Copied member identity differs')
    return {'manifest':manifest_name,'manifest_sha256':file_sha(destination/manifest_name),'members':len(seen),'bytes':total}


def main():
    parser=argparse.ArgumentParser()
    for name in ('sources','view','workspace'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--forbidden-root',action='append',required=True)
    args=parser.parse_args();root=args.workspace.absolute();root.mkdir(parents=True,exist_ok=False)
    if not sys.flags.isolated:raise ValueError('Consumer must be invoked with Python -I')
    inputs=strict_json(args.sources)
    if inputs.get('schema_version')!=1 or inputs.get('kind')!='token_input_sources':raise ValueError('Expected explicit C1 source manifest')
    selected={}
    with (args.view/'token_records.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():continue
            record=json.loads(line)
            if record.get('acquisition_source')=='diagnostic' and record.get('status') in ('MEASURED',None):
                selected.setdefault(record['source_id'],record['sample_id'])
    # Existing sources use diagnostic acquisition labels that include path type.
    if len(selected)<len(inputs['sources']):
        with (args.view/'token_records.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():continue
                record=json.loads(line)
                if record.get('backend_policy')=='diagnostic':selected.setdefault(record['source_id'],record['sample_id'])
    copied=[];specs=[]
    for index,spec in enumerate(inputs['sources']):
        if spec['source_id'] not in selected:raise ValueError('Missing diagnostic document for source consumer slice')
        path=Path(spec['path']);path=path if path.is_absolute() else args.sources.parent/path
        target=root/'raw'/('source'+str(index+1))
        receipt=copy_manifest_directory(path,target,'RAW_MANIFEST.json')
        value=dict(spec,path=str(target),sample_ids=[selected[spec['source_id']]])
        if spec['adapter']=='bench05b2':value['policies']=['P00']
        specs.append(value);copied.append(dict(receipt,source_id=spec['source_id'],selected_sample_id=selected[spec['source_id']],copied_to=str(target)))
    view_root=root/'recorded-view';view_copy=copy_manifest_directory(args.view,view_root,'TOKEN_VIEW_MANIFEST.json')
    local_sources=root/'copied-sources.json'
    with local_sources.open('x',encoding='utf-8') as handle:
        json.dump(dict(inputs,sources=specs),handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write('\n')
    forbidden=[os.path.abspath(value) for value in args.forbidden_root]
    if any(str(root)==value or str(root).startswith(value+os.sep) for value in forbidden):raise ValueError('Consumer workspace must be outside forbidden checkout roots')
    for name in ('PYTHONPATH','PYTHONHOME'):os.environ.pop(name,None)
    os.chdir(root)
    sys.path[:]=[value for value in sys.path if value and not any(os.path.abspath(value)==f or os.path.abspath(value).startswith(f+os.sep) for f in forbidden)]
    sdk=('torch','vllm','transformers','tensorflow','numpy','aiohttp','compressed_tensors','safetensors','accelerate')
    allowed_child={'enabled':False};audit_counts={'network_attempts':0,'blocked_checkout_attempts':0,'ml_sdk_attempts':0,'unexpected_subprocess_attempts':0}
    def audit(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            filename=os.path.abspath(os.fsdecode(values[0]))
            if any(filename==f or filename.startswith(f+os.sep) for f in forbidden):
                audit_counts['blocked_checkout_attempts']+=1;raise AssertionError('Forbidden original checkout read')
        if event=='import' and values and str(values[0]).split('.')[0] in sdk:
            audit_counts['ml_sdk_attempts']+=1;raise AssertionError('CPU consumer ML SDK import forbidden')
        if event in ('socket.connect','socket.getaddrinfo','socket.gethostbyname'):
            audit_counts['network_attempts']+=1;raise AssertionError('CPU consumer network forbidden')
        if event in ('subprocess.Popen','os.system','os.posix_spawn') and not allowed_child['enabled']:
            audit_counts['unexpected_subprocess_attempts']+=1;raise AssertionError('CPU consumer unexpected process creation')
    sys.addaudithook(audit)
    import inference_kit
    from inference_kit import tokens_export,tokens_compare
    from inference_kit.cli import main as cli
    from inference_kit.core.io import read,write
    from inference_kit.core.parity import equal
    origin=Path(inference_kit.__file__).absolute()
    assert 'site-packages' in origin.parts and origin.is_relative_to(Path(sys.prefix)),str(origin)
    assert inference_kit.__version__=='0.1.1.dev8',inference_kit.__version__
    calls=[]
    def json_shape(value):return json.loads(json.dumps(value,ensure_ascii=False,allow_nan=False))
    def command(argv,expected=0):
        out,err=io.StringIO(),io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            try:code=cli(argv)
            except SystemExit as exc:code=exc.code
        assert code==expected,(code,err.getvalue());calls.append({'args':argv,'code':code})
        return out.getvalue(),err.getvalue()
    exported=tokens_export(local_sources,root/'api-export')
    export_summary=tokens_compare(root/'api-export','bf16','rtn',root/'api-export-compare')
    assert exported['coverage']['records']>0 and export_summary['coverage']['measured']>0
    actual_sources=read(root/'api-export/SOURCES.json')
    assert {record['source_id'] for record in actual_sources}=={spec['source_id'] for spec in specs}
    free=[json.loads(line) for line in (root/'api-export/generation_records.jsonl').read_text(encoding='utf-8').splitlines() if line]
    assert all(record['sample_id']==selected[record['source_id']] for record in free)
    command(['bench','tokens','--help'])
    text,_=command(['bench','tokens','export','--sources','copied-sources.json','--output','cli-export'])
    equal(json.loads(text),json_shape(exported))
    # Both exports use identical copied sources and code; numeric/token identity
    # must match. View file metadata timestamps are outside its content identity.
    equal(read(root/'api-export/TOKEN_VIEW_MANIFEST.json'),read(root/'cli-export/TOKEN_VIEW_MANIFEST.json'))
    command(['bench','tokens','export','--sources','copied-sources.json','--output','cli-export'],2)
    api=tokens_compare(view_root,'bf16','rtn',root/'api-compare')
    text,_=command(['bench','tokens','compare','--view','recorded-view','--baseline','bf16','--candidate','rtn','--output','cli-compare'])
    cli_summary=read(root/'cli-compare/summary.json');equal(cli_summary,json_shape(api))
    selected_cli={key:json_shape(api)[key] for key in json.loads(text)};equal(json.loads(text),selected_cli)
    command(['bench','tokens','compare','--view','recorded-view','--baseline','bf16','--candidate','rtn','--output','cli-compare'],2)
    console=Path(sys.executable).parent/'inference-kit'
    argv=[str(console),'bench','tokens','compare','--view','recorded-view','--baseline','bf16','--candidate','rtn','--output','console-compare']
    allowed_child['enabled']=True
    try:child=subprocess.run(argv,capture_output=True,encoding='utf-8',timeout=120)
    finally:allowed_child['enabled']=False
    assert child.returncode==0,(child.returncode,child.stderr)
    equal(read(root/'console-compare/summary.json'),json_shape(api));equal(json.loads(child.stdout),selected_cli)
    assert not any(name.split('.')[0] in sdk for name in sys.modules)
    receipt={'status':'C1_INSTALLED_CPU_EXPORT_COMPARE_CONSUMER_PASS','package_version':inference_kit.__version__,
      'package_import_origin':str(origin),'interpreter':sys.executable,'isolated_mode':bool(sys.flags.isolated),
      'raw_copy_before_guard':copied,'view_copy_before_guard':view_copy,'selected_actual_document_per_task':selected,
      'export_source_scope':'one prespecified existing diagnostic document per task; B2 policy P00; full manifests verified before filtering',
      'api_export_coverage':exported['coverage'],'api_export_compare_coverage':export_summary['coverage'],
      'prebuilt_view_compare_coverage':api['coverage'],'commands':calls,'console_command':argv,
      'comparisons':'API/CLI export manifest parity; API/in-process CLI/separate console numeric summary parity; immutable output refusal',
      'parity_tolerance':{'float_abs':1e-10,'float_rel':1e-9,'integer_boolean_status_identity':'exact'},
      'json_representation':'Only tuple -> JSON array normalization; order, duplicate keys, numeric values, strings, null and types preserved',
      'audit_counts':audit_counts,'ML_SDK_imports':[],'checkout_reads':'denied by in-process audit hook after explicit copy',
      'network':'denied by in-process audit hook','subprocess':'only one explicit console child allowed',
      'audit_limits':'in-process hook is not an OS sandbox and is not inherited by the separately checked console child',
      'GPU_actions':0,'model_loads':0,'tokenizer_loads':0,'downloads':0}
    write(root/'CONSUMER_RECEIPT.json',receipt)
    print(json.dumps({'status':receipt['status'],'version':inference_kit.__version__,'package_import_origin':str(origin),
                      'export_coverage':exported['coverage'],'compare_coverage':api['coverage']},ensure_ascii=True,allow_nan=False))


if __name__=='__main__':main()
