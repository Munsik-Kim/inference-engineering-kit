"""Public C1 export/compare orchestration over versioned, portable token views.

Source adapters know historical paths. This module pairs coordinates, streams
numeric blobs, and writes common outputs; it does not know task scoring rules.
"""
import collections
import csv
from contextlib import contextmanager
from importlib.resources import files
import json
from pathlib import Path
import shutil

from ..core.io import read,write,loads,sha,digest,require,safe_name
from ..core.evidence import open_bundle,inspect_directory
from ..core.state import derived_output,preserve_failure
from ..probes.blobs import load as load_blob

SCHEMA='common_token_view_v1'


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield loads(line)


def write_jsonl(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def source_identity():
    package=files('inference_kit')
    return digest({str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts})


def metric_contract():
    return {'schema_version':1,'metric_family':'common_tokens_c1','metric_version':'1','log_base':'e','unit':'nat',
      'target_nll':'-raw target logprob; delta=candidate-reference',
      'aggregation':'Token weighted and source-document macro separately per task/policy/acquisition/reference scope/field; no combined task score',
      'full_distribution':'Exact KL(reference||candidate), JS=.5KL(p||m)+.5KL(q||m), m=(p+q)/2; only complete stored vocabulary',
      'top_k':'Intersection size/effective_k, not Jaccard; requested k=1,5,10; effective_k=min(k,vocabulary); descending logit then ascending ID',
      'rank':'1-based descending logit, token ID ascending for ties',
      'target_margin':'target logit minus maximum other vocabulary logit; distinct from choice margin',
      'fields':'Preserved literal-token overlap tags; a boundary token may belong to multiple fields; no retokenization',
      'selection':'Selected observed positions are not whole-reference likelihood or corpus perplexity',
      'generation':'Same-prompt output trajectory comparison only; no teacher-forced pairing after divergence',
      'generation_positions':'0-based actual generated-output indices; distinct from absolute teacher-forced target t/logit t-1',
      'missing':'UNKNOWN/INCOMPARABLE/null with explicit requested and measured coverage, never zero filled',
      'native_diagnostic':'Separate acquisition panels; instrumented/decoded diagnostic pair explicitly allowed, no deployed-kernel equivalence',
      'free_generation_audit_tokens':'Likelihood audit generation excluded from free-answer comparisons',
      'GPU_actions':0,'model_loads':0,'tokenizer_loads':0}


@contextmanager
def source_root(spec,base):
    path=Path(spec['path']).expanduser();path=path if path.is_absolute() else base/path
    with open_bundle(path,expected_sha256=spec.get('sha256')) as root:
        if (root/'RAW_MANIFEST.json').is_file() and (root/'PLAN_FROZEN.json').is_file():selected=root
        else:
            expected={'bench05b2':('BENCH05-B2-CONTRACT',),'bench05b3':('BENCH05-B3',),'bench05b4':('BENCH05-B4',),'token_probe_raw_v1':('BENCH05-C2','BENCH05-D1')}[spec['adapter']]
            candidates=[p.parent for p in root.rglob('RAW_MANIFEST.json') if (p.parent/'PLAN_FROZEN.json').is_file()
                        and read(p.parent/'PLAN_FROZEN.json').get('task') in expected]
            require(len(candidates)==1,'Expected one explicit matching source run');selected=candidates[0]
        yield selected,path


def export_tokens(sources,output):
    """Normalize explicit historical sources into a self-contained CPU token view."""
    from .token_sources import collect_source
    from .generation_sources import collect_generations
    path=Path(sources).expanduser().absolute();manifest=read(path)
    require(manifest.get('schema_version')==1 and manifest.get('kind')=='token_input_sources','Unsupported input manifest')
    specs=manifest.get('sources');require(isinstance(specs,list) and specs,'Explicit source list required')
    require(len({s['source_id'] for s in specs})==len(specs),'Duplicate source panel ID')
    require(all(s['adapter'] in ('bench05b2','bench05b3','bench05b4','token_probe_raw_v1') for s in specs),'Unsupported source adapter')
    out=derived_output(path,output)
    with preserve_failure(out):
        tokens=[];sequences={};generations=[];source_records=[]
        def blob_sink(directory,metadata):
            # Validate numeric bytes before portable copying; process a single blob at a time.
            load_blob(directory,metadata)
            relative='blobs/'+metadata['sha256']+'.f32';target=out/relative
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(directory)/safe_name(metadata['path']),target)
            require(sha(target)==metadata['sha256'],'Copied numeric identity mismatch')
            return dict(metadata,path=relative)
        for spec in specs:
            with source_root(spec,path.parent) as (root,original):
                bound=dict(spec,path=str(root))
                if spec['adapter']=='token_probe_raw_v1':
                    from .token_probe_source import collect_source as collect_probe_source
                    result=collect_probe_source(bound,blob_sink)
                else:result=collect_source(bound,blob_sink)
                source=result['source'];source.update(input_path_as_metadata=str(original),input_archive_sha256=sha(original) if original.is_file() else None)
                source_records.append(source);tokens.extend(result['tokens'])
                for sequence in result['sequences']:
                    key=sequence['sequence_id'];require(key not in sequences or sequences[key]==sequence,'Duplicate sequence identity conflict');sequences[key]=sequence
                if spec['adapter'] in ('bench05b2','bench05b4'):generations.extend(collect_generations(bound,read(root/'PLAN_FROZEN.json'),source))
        require(len(tokens)<=200000,'Token view record limit')
        for rows,key in ((tokens,'record_id'),(generations,'record_id')):require(len({r[key] for r in rows})==len(rows),'Duplicate global record ID')
        require(all(t['sequence_id'] in sequences for t in tokens),'Missing sequence reference')
        write_jsonl(out/'token_records.jsonl',tokens);write_jsonl(out/'sequences.jsonl',list(sequences.values()));write_jsonl(out/'generation_records.jsonl',generations)
        write(out/'INPUT_SOURCES.json',manifest);write(out/'SOURCES.json',source_records);write(out/'METRIC_CONTRACT.json',metric_contract())
        coverage={'status':'EXPORTED_OBSERVED_NOT_NEW_MEASUREMENTS','records':len(tokens),'sequences':len(sequences),'generation_records':len(generations),
          'panels':[{'source_id':s['source_id'],'coverage':s.get('coverage'),'task_id':s.get('task_id')} for s in source_records],
          'per_acquisition':dict(collections.Counter(r['acquisition_source'] for r in tokens)),
          'full_logits_records':sum(r.get('full_logits') is not None for r in tokens),'GPU_actions':0}
        write(out/'COVERAGE.json',coverage)
        inventory={p.relative_to(out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
        view={'schema_version':1,'kind':SCHEMA,'analysis_source_digest':source_identity(),'input_manifest_sha256':sha(path),
              'source_panels':source_records,'files':inventory,'view_identity':digest(inventory),'new_observations':False,'GPU_actions':0}
        write(out/'TOKEN_VIEW_MANIFEST.json',view)
        return {'status':'TOKEN_VIEW_EXPORTED','view_identity':view['view_identity'],'coverage':coverage,'analysis_source_digest':view['analysis_source_digest']}


def verify_view(root):
    view=read(root/'TOKEN_VIEW_MANIFEST.json');require(view['kind']==SCHEMA and view['schema_version']==1,'Wrong token view schema')
    require(digest(view['files'])==view['view_identity'],'Token view manifest identity differs')
    actual={e['name'] for e in inspect_directory(root)}
    require(actual==set(view['files'])|{'TOKEN_VIEW_MANIFEST.json'},'Token view unmanifested/missing file')
    for name,entry in view['files'].items():
        p=root/safe_name(name);require(p.is_file() and not p.is_symlink() and sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes'],'Token view member identity mismatch')
    return view


def pair_key(row):
    return tuple(digest(row.get(k)) for k in ('source_id','task_id','task_version','dataset_identity','split','sample_id','policy_id','choice_index','reference_scope','backend_policy','token_position'))


def generation_key(row):
    return tuple(digest(row.get(k)) for k in ('source_id','task_id','task_version','sample_id','policy_id'))


def csv_output(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def compare_tokens(view,baseline,candidate,output):
    """Compute common metrics using only a portable token view, never live models."""
    from ..metrics.token_diagnostics import compare_token,aggregate_token_comparisons
    from ..metrics.generation_compare import compare_generation
    require(isinstance(baseline,str) and isinstance(candidate,str) and baseline and candidate and baseline!=candidate,'Distinct explicit arm labels required')
    out=derived_output(view,output)
    with preserve_failure(out),open_bundle(view) as root:
        if not (root/'TOKEN_VIEW_MANIFEST.json').is_file():
            found=list(root.rglob('TOKEN_VIEW_MANIFEST.json'));require(len(found)==1,'Expected one token view');root=found[0].parent
        manifest=verify_view(root);sequences={}
        for row in jsonl(root/'sequences.jsonl'):
            require(row['sequence_id'] not in sequences,'Duplicate sequence ID');sequences[row['sequence_id']]=row
        sides={baseline:{},candidate:{}}
        for row in jsonl(root/'token_records.jsonl'):
            if row['arm_id'] not in sides:continue
            key=pair_key(row);require(key not in sides[row['arm_id']],'Duplicate comparison coordinate');sides[row['arm_id']][key]=row
        results=[]
        for key in sorted(set(sides[baseline])|set(sides[candidate])):
            a,b=sides[baseline].get(key),sides[candidate].get(key)
            def values(row):return load_blob(root,row['full_logits']) if row and row.get('full_logits') is not None else None
            row=compare_token(a,b,sequences.get(a['sequence_id']) if a else None,sequences.get(b['sequence_id']) if b else None,
                              left_logits=values(a),right_logits=values(b))
            row['source_id']=(a or b)['source_id'];results.append(row)
        aggregate=aggregate_token_comparisons(results);generations={baseline:{},candidate:{}}
        for row in jsonl(root/'generation_records.jsonl'):
            if row['arm_id'] not in generations:continue
            key=generation_key(row);require(key not in generations[row['arm_id']],'Duplicate free generation coordinate');generations[row['arm_id']][key]=row
        free=[]
        for key in sorted(set(generations[baseline])|set(generations[candidate])):
            a,b=generations[baseline].get(key),generations[candidate].get(key)
            if not a or not b:row={'status':'UNKNOWN','reasons':['MISSING_GENERATION_PAIR'],'source_id':(a or b)['source_id'],'sample_id':(a or b)['sample_id']}
            else:row=compare_generation(a,b)
            free.append(row)
        write_jsonl(out/'token-comparisons.jsonl',results);write_jsonl(out/'free-generation-comparisons.jsonl',free)
        write(out/'aggregates.json',aggregate);csv_output(out/'per-field.csv',aggregate['groups'])
        document_groups=collections.defaultdict(list)
        for row in results:
            identity=row.get('identity',{});document_groups[digest({k:identity.get(k) for k in ('source_id','task_id','task_version','dataset_identity','split','sample_id','policy_id','choice_index','reference_scope','backend_policy')})].append(row)
        documents=[dict(aggregate_token_comparisons(rows),document_identity={k:rows[0]['identity'].get(k) for k in ('source_id','task_id','task_version','dataset_identity','split','sample_id','policy_id','choice_index','reference_scope','backend_policy')}) for rows in document_groups.values()]
        write(out/'per-document.json',documents);csv_output(out/'per-document.csv',[dict(group,**d['document_identity']) for d in documents for group in d['groups'] if group.get('field_tag')=='__all__'])
        compatibility={'status_counts':dict(collections.Counter(r['status'] for r in results)),'reasons':dict(collections.Counter(reason for r in results for reason in r.get('reasons',[]))),
                       'native_diagnostic_bridge':'SEPARATE_PANELS_NOT_AUTOMATICALLY_PAIRED','cross_policy':'NOT_DIRECTLY_PAIRED; policy part of coordinate identity'}
        write(out/'COMPATIBILITY.json',compatibility);write(out/'METRIC_CONTRACT.json',metric_contract())
        coverage={'requested_pair_coordinates':len(results),'measured':sum(r['status']=='MEASURED' for r in results),'unknown':sum(r['status']=='UNKNOWN' for r in results),
                  'incomparable':sum(r['status']=='INCOMPARABLE' for r in results),'free_generation_pairs':len(free),'source_coverage':read(root/'COVERAGE.json')}
        write(out/'COVERAGE.json',coverage)
        summary={'schema_version':1,'task':'BENCH05-C1','baseline':baseline,'candidate':candidate,'view_identity':manifest['view_identity'],
          'acquisition_source_panels':manifest['source_panels'],'analysis_source_digest':source_identity(),'view_analysis_source_digest':manifest['analysis_source_digest'],
          'coverage':coverage,'compatibility':compatibility,'aggregates':aggregate,'free_generation_status':dict(collections.Counter(r['status'] for r in free)),
          'ENGINEERING':'COMMON_TOKEN_ANALYSIS_CPU','INPUT_COVERAGE':'PRESERVED_SELECTED_OBSERVATIONS_ONLY','COMPARABILITY':'VERSIONED_SAME_PREFIX_PANELS',
          'MODEL_QUALITY':'NOT_CERTIFIED; retained HOLD_CLIPPING_VARIANT and prior INCONCLUSIVE','GPU_actions':0,'model_loads':0,'tokenizer_loads':0,'downloads':0,'ledger_changes':0,'REMOTE_CI':'NOT_RUN'}
        write(out/'summary.json',summary)
        text=['# BENCH05-C1 공통 토큰 분석','','저장된 관측의 CPU 파생 분석입니다. 새 모델 점수·배포 적합성·전체 참조 perplexity 인증이 아닙니다.',
              '',f"Baseline `{baseline}`, candidate `{candidate}`. 비교 좌표 {len(results)}, measured {coverage['measured']}, unknown {coverage['unknown']}, incomparable {coverage['incomparable']}.",
              '', 'Native/diagnostic, policy, reference scope, task별 panel을 유지합니다. 자유 생성 분기 이후를 teacher-forced 위치로 비교하지 않습니다.']
        (out/'REPORT_KO.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
        return summary
