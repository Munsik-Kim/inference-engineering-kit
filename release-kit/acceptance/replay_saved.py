#!/usr/bin/env python3
"""Explicit installed-candidate CPU paths A/B; no archive-code discovery.

api uses public functions. module/console invoke actual fixed product commands
in new processes. The audit hook applies only to this process, not those children.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('api','module','console'))
    args=parser.parse_args(); kit=args.kit.resolve(); output=args.output.absolute()
    if output.exists():raise ValueError('New output required')
    if output.is_relative_to(kit/'fixtures'):raise ValueError('Output inside immutable fixture refused')
    expected=json.loads((kit/'acceptance/EXPECTED_SAVED_RESULTS.json').read_text(encoding='utf-8'))
    release=json.loads((kit/'RELEASE_IDENTITY.json').read_text(encoding='utf-8'))
    qwen=kit/'fixtures/qwen-native'; token=kit/'fixtures/smol-token'; structure=kit/'fixtures/smol-structure'
    helper=kit/'acceptance/replay_structure_slice.py'
    qout=output/'qwen'; tout=output/'smol-token'; sout=output/'smol-structure'
    base=[str(Path(sys.executable)), '-I', '-B']
    entry=base+['-m','inference_kit'] if args.mode=='module' else [str(Path(sys.executable).parent/'inference-kit')]
    commands=[]
    if args.mode!='api':
        commands=[entry+['bench','suite','replay','--run',str(qwen),'--output',str(qout)],
                  entry+['bench','tokens','compare','--view',str(token),'--baseline','bf16','--candidate','rtn','--output',str(tout)],
                  base+[str(helper),'--fixture',str(structure),'--output',str(sout)]]
    allowed={tuple(command) for command in commands}
    guards={'ML_SDK_import_attempts':0,'socket_attempts':0,'unapproved_subprocess_attempts':0,
            'approved_CPU_child_launches':0,'model_weight_open_attempts':0,'absolute_checkout_code_attempts':0}
    def guard(event, values):
        if event=='import' and values and str(values[0]).split('.')[0] in {'torch','transformers','tokenizers','numpy','vllm','aiohttp','compressed_tensors','safetensors'}:
            guards['ML_SDK_import_attempts']+=1;raise ImportError('Saved CPU example refuses ML SDK')
        if event in {'socket.__new__','socket.connect','socket.bind','socket.getaddrinfo'}:
            guards['socket_attempts']+=1;raise PermissionError('Saved CPU example refuses sockets')
        if event=='subprocess.Popen':
            if tuple(values[1]) not in allowed:
                guards['unapproved_subprocess_attempts']+=1;raise PermissionError('Only explicit fixed CPU CLI commands allowed')
            guards['approved_CPU_child_launches']+=1
        if event in {'os.system','os.posix_spawn'}:
            guards['unapproved_subprocess_attempts']+=1;raise PermissionError('Unapproved command launch refused')
        if event=='open' and values and isinstance(values[0],(str,bytes)):
            path=Path(os.fsdecode(values[0]))
            if path.suffix in {'.safetensors','.pt','.pth'}:
                guards['model_weight_open_attempts']+=1;raise PermissionError('Model/weight access refused')
            # Restrict actual checkout source imports without misinterpreting
            # dir_fd-relative temporary cleanup paths as absolute code reads.
            if path.is_absolute() and 'inference-engineering-kit' in path.parts and any(part in path.parts for part in ('src','tests','tools')):
                guards['absolute_checkout_code_attempts']+=1;raise PermissionError('Original checkout code access refused')
    sys.addaudithook(guard);sys.dont_write_bytecode=True
    import inference_kit
    from inference_kit import api
    from inference_kit.analysis.tokens import source_identity, jsonl
    from inference_kit.core.io import read, require, sha, write, digest
    from inference_kit.core.parity import equal
    origin=Path(inference_kit.__file__).resolve()
    require('site-packages' in origin.parts, 'Ordinary installed candidate wheel required')
    require(inference_kit.__version__==release['version'] and source_identity()==release['python_source_digest'], 'Release version/source mismatch')
    require(sys.flags.isolated, 'Run with python -I')
    pins={str(p.relative_to(kit/'fixtures')):{'bytes':p.stat().st_size,'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns}
          for p in sorted((kit/'fixtures').rglob('*')) if p.is_file()}
    output.mkdir(parents=True)
    cli=[]
    if args.mode=='api':
        api.suite_replay(qwen,qout)
        api.tokens_compare(token,'bf16','rtn',tout)
        spec=importlib.util.spec_from_file_location('kit_explicit_structure_slice',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.replay(structure,sout)
    else:
        env=dict(os.environ)
        for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
        env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for number, command in enumerate(commands,1):
            result=subprocess.run(command,cwd=kit,env=env,text=True,encoding='utf-8',capture_output=True,timeout=180)
            (output/f'command-{number}.stdout.json').write_text(result.stdout,encoding='utf-8')
            (output/f'command-{number}.stderr.log').write_text(result.stderr,encoding='utf-8')
            require(result.returncode==0, 'CPU command failed: '+str(command)+'\n'+result.stderr)
            require(all(ord(c)<128 for c in result.stdout), 'CLI stdout is not ASCII-safe JSON')
            cli.append({'argv':command,'returncode':result.returncode,'ASCII_safe_JSON':True})
    # Read products back from disk, not just their return/status strings.
    qresult=read(qout/'CPU_REPLAY_RECEIPT.json');qscore=read(qout/'score_task/summary.json')
    qrecords=list(jsonl(qout/'analyze_tokens/token-view/token_records.jsonl'))
    qpairs=list(jsonl(qout/'analyze_tokens/token-metrics/token-comparisons.jsonl'))
    qaggregates=read(qout/'analyze_tokens/token-metrics/aggregates.json')
    require(qresult['status']=='CPU_SLICE_REPLAYED' and qscore['all_required_task_inputs_complete'], 'Qwen slice incomplete')
    require(qscore['inference_submissions']==qscore['inference_responses']==expected['Qwen']['request_rows'] and
            qscore['document_score_rows']==expected['Qwen']['document_score_rows'], 'Task denominator differs')
    equal(qscore['tasks'],expected['Qwen']['tasks']);equal(qaggregates,expected['Qwen']['native_aggregates'])
    require(len(qrecords)==expected['Qwen']['native_scalar_rows'] and len(qpairs)==expected['Qwen']['native_pairs'], 'Native scalar/pair count differs')
    require(qresult['full_vectors']==0 and all(r['full_logits'] is None for r in qrecords), 'Uncollected full distribution fabricated')
    context=read(qwen/'suite-execution-context.json');raw=[r for job in context['serving_jobs'].values() for r in jsonl(qwen/job['raw_path'])]
    require(len(raw)==expected['Qwen']['request_rows'] and len({r['request_id']for r in raw})==len(raw), 'Request identity differs')
    output_tokens=sum(r.get('usage',r.get('response',{}).get('usage',{}))['completion_tokens'] for r in raw)
    require(output_tokens==expected['Qwen']['output_tokens'], 'Saved output token usage differs')
    tresult=read(tout/'summary.json');tpairs=list(jsonl(tout/'token-comparisons.jsonl'));taggregates=read(tout/'aggregates.json')
    require(tresult['coverage']['requested_pair_coordinates']==tresult['coverage']['measured']==expected['Smol']['scalar_pairs'] and
            tresult['coverage']['unknown']==tresult['coverage']['incomparable']==0,'Smol paired coverage differs')
    require(len(tpairs)==expected['Smol']['scalar_pairs'] and sum(r['coverage']['full_logits'] for r in tpairs)==expected['Smol']['full_pairs'], 'Smol full slice coverage differs')
    equal(taggregates,expected['Smol']['aggregates'])
    structure_result=read(sout/'REPLAY_RECEIPT.json')
    require(structure_result['physical_observations']==24 and structure_result['cross_arm_rows']==12 and structure_result['same_arm_reconstruction_rows']==6,'Structure slice coverage differs')
    for name,pin in pins.items():
        p=kit/'fixtures'/name;require(p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'] and p.stat().st_mtime_ns==pin['mtime_ns'], 'Read-only fixture changed: '+name)
    semantics={'Qwen':{'tasks':qscore['tasks'],'native_aggregates':qaggregates,'requests':len(raw),'output_tokens':output_tokens,'native_scalar':len(qrecords),'native_pairs':len(qpairs),'full_vectors':0},
               'Smol':{'aggregates':taggregates,'scalar_pairs':len(tpairs),'full_pairs':sum(r['coverage']['full_logits'] for r in tpairs)},
               'structure':{'cross_arm':read(sout/'STRUCTURE_COMPARISONS.json'),'same_arm_AV':read(sout/'SAME_ARM_AV.json')}}
    write(output/'SEMANTIC_RESULTS.json',semantics)
    receipt={'status':'PASS','mode':args.mode,'package_version':inference_kit.__version__,'package_origin':str(origin),
             'source_digest':source_identity(),'isolated':bool(sys.flags.isolated),'semantic_identity':digest(semantics),
             'Qwen':{'HTTP':len(raw),'output_tokens':output_tokens,'native_scalar':len(qrecords),'native_pairs':len(qpairs),'full_distribution':'NOT_COLLECTED'},
             'Smol':{'scalar_pairs':len(tpairs),'full_pairs':sum(r['coverage']['full_logits'] for r in tpairs),'separate_structure_acquisition':True},
             'structure_slice':structure_result,'raw_pins_unchanged':len(pins),'guards':guards,'commands':cli,
             'guard_scope':'Parent process only; direct product CLI children do not inherit audit hook and are separately checked through their saved output/count/identity. Not an OS sandbox.',
             'child_audit_hook':'NOT_INHERITED' if cli else 'NO_CHILDREN','expected_projections':'Exact task summary tasks and entire C1 aggregate objects; no broad metadata-ignore policy',
             'float_tolerance':expected['float_tolerance'],'new_acquisitions':0,'GPU_queries':0,'model_loads':0,'tokenizer_calls':0,'inference_POSTs':0,'campaign_debit':0,
             'not_recertified':['GPU execution','Fresh tokenization','Live resume/no-op','Full historical DB','Full D1 controls']}
    write(output/'SAVED_REPLAY_RECEIPT.json',receipt)
    print(json.dumps({'status':'PASS','mode':args.mode,'semantic_identity':receipt['semantic_identity'],'Qwen':receipt['Qwen'],'Smol':receipt['Smol'],'raw_pins_unchanged':len(pins)},ensure_ascii=True,allow_nan=False))

if __name__=='__main__':main()
