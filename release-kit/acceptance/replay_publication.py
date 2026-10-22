"""CPU-only synthetic full Suite publication/state acceptance.

This explicit fixture is SYNTHETIC_CONTRACT_TEST, separate from actual Qwen and
Smol slices. It contains no live acquisition and needs no checkout or tokenizer.
Module and console modes invoke actual installed commands in fresh children.
The parent audit hook does not protect those child processes or the OS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=True, allow_nan=False,
                                    sort_keys=True, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pins(root):
    return {p.relative_to(root).as_posix(): [p.stat().st_size, sha(p), p.stat().st_mtime_ns]
            for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('api', 'module', 'console'))
    args = parser.parse_args()
    kit, output = args.kit.absolute(), args.output.absolute()
    if output.exists():
        raise ValueError('Refusing existing acceptance output')
    output.mkdir(parents=True)
    denied = []
    def guard(event, values):
        if event == 'import' and any(str(values[0]) == name or str(values[0]).startswith(name + '.')
                                     for name in ('torch','transformers','tokenizers','vllm','numpy','aiohttp')):
            denied.append(str(values[0])); raise RuntimeError('ML/HTTP SDK import blocked')
        if event.startswith('socket.'):
            denied.append(event); raise RuntimeError('Socket use blocked')
    sys.addaudithook(guard)
    import inference_kit
    from inference_kit import api
    from inference_kit.analysis.tokens import source_identity
    from inference_kit.analysis.result_index import canonical_export
    from inference_kit.analysis.suite_results import validate_current_index
    from inference_kit.analysis.evidence_publication import publish_suite_evidence
    from inference_kit.core.evidence import Limits
    library = Path(inference_kit.__file__).absolute()
    if 'site-packages' not in library.parts:
        raise ValueError('Requires ordinarily installed candidate site-packages')
    fixture = kit/'fixtures/publication-demo'
    scope = read(fixture/'FIXTURE_SCOPE.json')
    assert scope['kind'] == 'SYNTHETIC_CONTRACT_TEST'
    fixture_pin = pins(fixture)
    raw, analysis = output/'mixed-raw', output/'cpu-analysis'
    shutil.copytree(fixture, raw)
    # Deliberately exceed the unchanged reader bound outside the raw manifest.
    sparse = raw/'derived/unmanifested-index.sqlite'; sparse.parent.mkdir()
    with sparse.open('wb') as stream: stream.truncate(Limits().file_bytes + 1)
    manifest = read(raw/'RUN_MANIFEST.json')
    raw_pin = {name: sha(raw/name) for name in [*manifest['files'], 'RUN_MANIFEST.json']}
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    console = Path(sys.executable).with_name('inference-kit')
    commands = []
    def child(command, name):
        run = subprocess.run(command, cwd=kit, env=env, capture_output=True, text=True, encoding='utf-8')
        (output/(name+'.stdout')).write_text(run.stdout, encoding='utf-8')
        (output/(name+'.stderr')).write_text(run.stderr, encoding='utf-8')
        commands.append({'name':name, 'argv':command, 'returncode':run.returncode,
                         'guard_scope':'Explicit independent CPU child; parent audit hook is not inherited'})
        if run.returncode: raise AssertionError(name + ': ' + run.stderr)
        return json.loads(run.stdout)
    def invoke(action, values, name, new_process=False):
        if args.mode == 'api' and not new_process:
            if action == 'postprocess':
                return api.suite_postprocess(values['run'], values['output'],
                    analysis_revision=values['analysis-revision'], stop_after=values.get('stop-after'))
            if action == 'postprocess-resume': return api.suite_postprocess_resume(values['analysis'])
            if action == 'postprocess-status': return api.suite_postprocess_status(values['analysis'])
            raise AssertionError(action)
        if args.mode == 'api':
            # A separate actual Python process calls the installed public API.
            program = ('import json; from inference_kit import api; '
                       'print(json.dumps(api.suite_postprocess_resume(' + repr(str(analysis)) + ')))')
            return child([sys.executable,'-I','-B','-c',program], name)
        prefix = [sys.executable,'-I','-B','-m','inference_kit'] if args.mode == 'module' else [str(console)]
        command = prefix + ['bench','suite',action]
        for key, value in values.items(): command += ['--'+key, str(value)]
        return child(command, name)
    first = invoke('postprocess', {'run':raw,'output':analysis,
        'analysis-revision':'v1-release01-synthetic-publication','stop-after':'score_task'}, 'pause')
    assert first['status'] == 'CPU_PAUSED_BOUNDARY'
    score_marker = sha(analysis/'completions/score_task.json')
    last = invoke('postprocess-resume', {'analysis':analysis}, 'new-process-resume', new_process=True)
    assert last['status'] == 'CPU_POSTPROCESS_COMPLETE'
    assert sha(analysis/'completions/score_task.json') == score_marker
    index_output = analysis/last['nodes']['index_add']['output']
    report_output = analysis/last['nodes']['report']['output']
    publication = read(index_output/'EVIDENCE_PUBLICATION_RECEIPT.json')
    published = Path(publication['evidence_root'])
    assert publication['status'] == 'RAW_HASH_VERIFIED' and not published.is_relative_to(raw)
    assert not (published/'derived').exists()
    assert set(pins(published)) == set(manifest['files']) | {'RUN_MANIFEST.json'}
    assert sha(published/'RUN_MANIFEST.json') == sha(raw/'RUN_MANIFEST.json')
    db = index_output/'index.sqlite'; canonical = canonical_export(db)
    assert len(canonical['sources']) == 1
    assert canonical['sources'][0]['validation']['status'] == 'RAW_HASH_VERIFIED'
    assert len(canonical['acquisitions']) == 8
    current = validate_current_index(db, raw, publication=publication)
    evidence_before = pins(published)
    # Repeat public publisher is a validated no-op over the original mixed root.
    repeated = publish_suite_evidence(raw, published.parent)
    assert repeated['reused'] is True and pins(published) == evidence_before
    # Additional large DERIVED output is outside the immutable publication.
    growth = output/'derived-growth/index.sqlite'; growth.parent.mkdir()
    with growth.open('wb') as stream: stream.truncate(Limits().file_bytes + 1)
    assert canonical_export(db) == canonical and pins(published) == evidence_before
    # Validate repeat add against an SQLite backup, keeping completed workspace immutable.
    backup = output/'idempotence.sqlite'
    with sqlite3.connect(db.as_uri()+'?mode=ro', uri=True) as old, sqlite3.connect(backup) as new:
        old.backup(new)
    added = api.index_add(index_output/'INPUT_SOURCES.json', backup, output/'reregister')
    assert added['before'] == added['after']
    assert canonical_export(backup) == canonical
    rebuilt = output/'rebuilt.sqlite'
    api.index_build(index_output/'INPUT_SOURCES.json', rebuilt, output/'rebuild')
    assert canonical_export(rebuilt) == canonical
    # Only audit events and temporary lock activity may change during completed no-op.
    immutable = {str(p.relative_to(analysis)): [p.stat().st_size,sha(p),p.stat().st_mtime_ns]
                 for parent in (analysis/'stages',analysis/'completions')
                 for p in parent.rglob('*') if p.is_file()}
    checkpoint = [sha(analysis/'checkpoint.json'), (analysis/'checkpoint.json').stat().st_mtime_ns]
    noop = invoke('postprocess-resume', {'analysis':analysis}, 'completed-noop')
    status_values = {'analysis':analysis}
    if args.mode != 'api': status_values['output'] = output/'status'
    status = invoke('postprocess-status', status_values, 'completed-status')
    assert noop['noop'] is True and noop['checkpoint_reconciled'] is False
    assert status['status'] == 'CPU_POSTPROCESS_COMPLETE'
    assert checkpoint == [sha(analysis/'checkpoint.json'), (analysis/'checkpoint.json').stat().st_mtime_ns]
    assert immutable == {str(p.relative_to(analysis)): [p.stat().st_size,sha(p),p.stat().st_mtime_ns]
                 for parent in (analysis/'stages',analysis/'completions')
                 for p in parent.rglob('*') if p.is_file()}
    assert canonical_export(db) == canonical and pins(published) == evidence_before
    # A copied, corrupt raw parent is still rejected, despite the old valid publication.
    corrupt = output/'tampered-copy'; shutil.copytree(fixture, corrupt)
    name = next(name for name in manifest['files'] if name.endswith('task-responses.jsonl'))
    with (corrupt/name).open('ab') as stream: stream.write(b' ')
    try:
        publish_suite_evidence(corrupt, output/'rejected-publication')
    except ValueError as error:
        tamper_error = str(error)
    else:
        raise AssertionError('Corrupted raw unexpectedly published')
    assert not (output/'rejected-publication').exists()
    assert {name:sha(raw/name) for name in raw_pin} == raw_pin
    assert pins(fixture) == fixture_pin and not denied
    result = {'schema_version':1, 'status':'CPU_PUBLICATION_ACCEPTANCE_PASS',
        'scope':'SYNTHETIC_CONTRACT_TEST_NOT_LIVE_ACQUISITION', 'mode':args.mode,
        'package_version':inference_kit.__version__, 'package_origin':str(library),
        'source_digest':source_identity(), 'raw_manifest_sha256':sha(raw/'RUN_MANIFEST.json'),
        'expected':scope, 'publication':{key:publication[key] for key in
            ('status','manifest_sha256','canonical_source_id','members','bytes','reused')},
        'current_source_validation':current, 'index_counts':{key:len(canonical[key]) for key in
            ('sources','acquisitions','analyses','samples','executions','measurements','evidence','relations')},
        'new_process_resume':True, 'score_completion_unchanged':True,
        'reregister_count_unchanged':True,'rebuild_canonical_equal':True,
        'completed_noop':True,'checkpoint_bytes_mtime_unchanged':True,
        'completion_evidence_db_immutable_files':len(immutable), 'derived_growth_bytes':growth.stat().st_size,
        'tamper_refused':tamper_error,'fixture_unchanged':True,
        'report_entry':str(report_output/'index.html'), 'report_bytes':(report_output/'index.html').stat().st_size,
        'commands':commands,'guard_scope':'This process only; CLI/API subprocess calls explicitly recorded, not an OS sandbox',
        'denied_calls':denied,'GPU_queries':0,'model_loads':0,'tokenizer_calls':0,'inference_POSTs':0,'campaign_debit':0,
        'new_actual_acquisitions':0}
    write(output/'PUBLICATION_ACCEPTANCE.json',result)
    print(json.dumps({'status':result['status'],'mode':args.mode,'receipt':str(output/'PUBLICATION_ACCEPTANCE.json')},ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
