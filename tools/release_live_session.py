"""Reviewed CPU-only live-session preparation; never launches a model or edits a ledger.

The native task clock retains the existing 3600/3000-second contract. A separate
session receipt prevents a fresh task clock from extending the enclosing session.
Run ``check`` before each explicit run/resume. Existing run and source checks remain
in the installed product; this helper does not grant GPU or socket authorization.
"""
import argparse
import datetime
import hashlib
import math
from pathlib import Path
import platform
import time
import uuid

from inference_kit.analysis.tokens import source_identity
from inference_kit.core.io import digest, read, require, sha, write


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _host_boot():
    boot = Path('/proc/sys/kernel/random/boot_id')
    require(boot.is_file(), 'LIVE_CLOCK_UNSUPPORTED_HOST: Linux boot identity unavailable')
    return {'host_identity': hashlib.sha256(platform.node().encode('utf-8')).hexdigest(),
            'boot_identity': hashlib.sha256(boot.read_bytes()).hexdigest()}


def _sealed(value):
    return value | {'identity': digest(value)}


def _verify_seal(value):
    require(isinstance(value, dict) and isinstance(value.get('identity'), str), 'Missing session/clock identity')
    require(value['identity'] == digest({k: v for k, v in value.items() if k != 'identity'}),
            'Session/clock identity changed')


def _placeholder(value):
    if isinstance(value, str):
        require(value.strip() and '<' not in value and '>' not in value and
                'CHANGE_ME' not in value and '/absolute/path/' not in value,
                'UNRESOLVED_LOCAL_BINDING: explicit value required')
    elif isinstance(value, dict):
        for item in value.values(): _placeholder(item)
    elif isinstance(value, list):
        for item in value: _placeholder(item)


def start_session(output, started_utc=None):
    observed_utc = datetime.datetime.now(datetime.timezone.utc)
    observed_mono = time.monotonic()
    origin = datetime.datetime.fromisoformat(started_utc) if started_utc else observed_utc
    require(origin.tzinfo is not None, 'Session origin requires an explicit timezone')
    origin = origin.astimezone(datetime.timezone.utc)
    elapsed = (observed_utc-origin).total_seconds()
    require(0 <= elapsed < 3480, 'Session origin is future/expired or cleanup reserve unavailable')
    start = observed_mono-elapsed
    value = _sealed({'schema_version': 1, 'kind': 'release_live_session_v1',
        'session_id': uuid.uuid4().hex, 'start_utc': origin.isoformat(), 'start_monotonic': start,
        'observed_utc': observed_utc.isoformat(), 'observed_monotonic': observed_mono,
        'monotonic_origin': 'UTC_DELTA_MAPPING_AT_RECEIPT_CREATION' if started_utc else 'DIRECT_CURRENT_OBSERVATION',
        'deadline_monotonic': start + 3600, 'no_new_gpu_after': start + 3000,
        'cleanup_reserve_seconds': 120, 'gpu_wall_cap_seconds': 900,
        'source_digest': source_identity(), **_host_boot(), 'authorization': 'NONE',
        'gpu_queries': 0, 'model_loads': 0, 'inference_POSTs': 0, 'ledger_changes': 0})
    write(Path(output), value)
    return value


def check_session(session):
    value = read(Path(session)); _verify_seal(value)
    require(value.get('kind') == 'release_live_session_v1' and value.get('schema_version') == 1,
            'Unsupported release session receipt')
    require(value['source_digest'] == source_identity(), 'Session package source changed')
    require(all(value.get(k) == v for k, v in _host_boot().items()), 'Session belongs to another host/boot')
    start = value.get('start_monotonic')
    require(type(start) in (int, float) and math.isfinite(start), 'Invalid session monotonic time')
    require(value.get('deadline_monotonic') == start + 3600 and
            value.get('no_new_gpu_after') == start + 3000 and
            value.get('cleanup_reserve_seconds') == 120 and value.get('gpu_wall_cap_seconds') == 900,
            'Release session bounds changed')
    now = time.monotonic()
    origin = datetime.datetime.fromisoformat(value['start_utc'])
    require(origin.tzinfo is not None and (datetime.datetime.now(datetime.timezone.utc)-origin).total_seconds() < 3000,
            'SESSION_EXPIRED_OR_CLEANUP_RESERVE: enclosing UTC launch deadline reached')
    require(start <= now < value['no_new_gpu_after'] and now < value['deadline_monotonic'] - 120,
            'SESSION_EXPIRED_OR_CLEANUP_RESERVE: use a separately authorized fresh session; do not edit this clock')
    return value


def make_clock(session, output):
    outer = check_session(session); start = time.monotonic()
    value = _sealed({'schema_version': 1, 'kind': 'release_task_clock_v1',
        'start_utc': _utc(), 'start_monotonic': start, 'deadline_monotonic': start + 3600,
        'no_new_gpu_after': start + 3000, 'cleanup_reserve_seconds': 120,
        'enclosing_session_id': outer['session_id'], 'enclosing_session_identity': outer['identity'],
        'enclosing_deadline_monotonic': outer['deadline_monotonic'],
        'enclosing_no_new_gpu_after': outer['no_new_gpu_after'], **_host_boot(),
        'source_digest': source_identity(), 'authorization': 'NONE', 'ledger_changes': 0})
    write(Path(output), value)
    return value


def check_clock(session, clock):
    outer = check_session(session); value = read(Path(clock)); _verify_seal(value)
    require(value.get('kind') == 'release_task_clock_v1' and value.get('schema_version') == 1,
            'Unsupported fresh task clock')
    require(value.get('source_digest') == source_identity() and
            all(value.get(k) == v for k, v in _host_boot().items()), 'Clock package/host/boot changed')
    require(value.get('enclosing_session_id') == outer['session_id'] and
            value.get('enclosing_session_identity') == outer['identity'] and
            value.get('enclosing_deadline_monotonic') == outer['deadline_monotonic'] and
            value.get('enclosing_no_new_gpu_after') == outer['no_new_gpu_after'],
            'Clock belongs to another release session')
    start = value.get('start_monotonic')
    require(type(start) in (int, float) and math.isfinite(start) and
            outer['start_monotonic'] <= start <= time.monotonic() and
            value.get('deadline_monotonic') == start + 3600 and
            value.get('no_new_gpu_after') == start + 3000 and value.get('cleanup_reserve_seconds') == 120,
            'Task clock bounds changed')
    require(time.monotonic() < value['no_new_gpu_after'] and
            time.monotonic() < value['deadline_monotonic'] - 120, 'Task clock expired')
    return value


def resolve_local(binding, prepared, session, clock, output):
    """Write a new explicit binding; preserve original assets, templates and ledgers."""
    check_clock(session, clock)
    value = read(Path(binding))
    # These three fields are generated below, rather than user-supplied secrets.
    _placeholder({k:v for k,v in value.items() if k not in ('prepared_manifest','prepared_manifest_sha256','task_clock')})
    require(value.get('kind') == 'prepared_suite_live_binding_v1' and value.get('profile') in
            ('prepared_smol_live_v1', 'prepared_qwen_serve_v1'),
            'CPU_LOCAL_OR_UNSUPPORTED_PROFILE: full explicit live binding required')
    require(type(value.get('gpu')) is int and value['gpu'] >= 0 and
            type(value.get('port')) is int and 1024 <= value['port'] <= 65535 and
            isinstance(value.get('gpu_uuid'), str) and value['gpu_uuid'].startswith('GPU-'),
            'Explicit approved GPU index/UUID and localhost port required')
    for name in ('artifact_receipt', 'runtime_metadata', 'runtime_python', 'server_python',
                 'package_site', 'prior_ledger', 'campaign_root'):
        require(isinstance(value.get(name), str), 'Missing local binding path: ' + name)
        path = Path(value[name]).expanduser().absolute()
        require(path.exists(), 'Missing local binding path: ' + name)
        value[name] = str(path)
    prepared = Path(prepared).expanduser().absolute()
    require(prepared.is_file() and prepared.name == 'PREPARED_MANIFEST.json',
            'Full PREPARED_MANIFEST.json required; ancestry/SLICE is not live authority')
    value['prepared_manifest'] = str(prepared)
    value['prepared_manifest_sha256'] = sha(prepared)
    value['task_clock'] = str(Path(clock).absolute())
    if value.get('historical_index') is not None:
        path = Path(value['historical_index']).expanduser().absolute()
        require(path.is_file(), 'Historical index must be explicit existing read-only input')
        value['historical_index'] = str(path)
    write(Path(output), value)
    return {'status': 'CPU_LOCAL_BOUND_NOT_AUTHORIZED', 'output': str(Path(output).absolute()),
            'sha256': sha(Path(output)), 'GPU_queries': 0, 'model_loads': 0, 'ledger_changes': 0}


def make_suite(template, sources, output):
    """Select all validated prepared records; preserve template task/model policies."""
    from copy import deepcopy
    from inference_kit.suites.suite_inputs import load_sources
    from inference_kit.suites.suite_plan import compile_suite, validate_spec
    loaded = load_sources(sources); spec = deepcopy(read(Path(template)))
    validate_spec(spec)
    by_task = {(s['task_id'],s['task_version']):s for s in loaded['sources']}
    require(len(by_task) == len(loaded['sources']) == len(spec['tasks']), 'Prepared/template task inventory differs')
    posts, cap = 0, 0
    for task in spec['tasks']:
        source = by_task.get((task['task_id'],task['task_version']))
        require(source is not None and source['samples'], 'Missing nonempty prepared task')
        task['source_id'] = source['source_id']
        task['sample_ids'] = [row['sample_id'] for row in source['samples']]
        for row in source['samples']:
            if source['request_kind'] == 'generate_until':
                posts += 1; cap += row['generation']['wire_payload']['max_tokens']
            else:
                # Existing all-choice likelihood contract has one audit token per request.
                from inference_kit.tasks.registry import get_task
                from inference_kit.adapters.likelihood import request_body
                adapter = get_task(source['task_id'], source['task_version'])
                metadata = loaded['model_metadata']
                requests = adapter.requests(row['authored_sample'], 'release-static-suite',
                    metadata['tokenizer_identity'], metadata['token_id_space'], metadata['runtime_identities']['serving'])
                posts += len(requests)
                cap += sum(request_body(request)['max_tokens'] for request in requests)
    spec['suite_id'] = 'V1-RELEASE-01-'+spec['models'][0]['profile']
    spec['budget']['max_inference_posts'] = posts*len(spec['models'])
    spec['budget']['max_output_tokens'] = cap*len(spec['models'])
    compiled = compile_suite(spec, loaded, {'schema_version':1,'artifact_check':'not_requested','artifact_roots':{},'authorization':'NONE','runtime_check':'not_run'})
    require(compiled['status'] == 'VALIDATED_NOT_EXECUTED', 'Required prepared operations blocked')
    require(compiled['semantic']['counts']['inference_POSTs'] == spec['budget']['max_inference_posts'] and
            compiled['semantic']['counts']['output_token_upper'] == spec['budget']['max_output_tokens'],
            'Derived request/output bounds differ from product compiler')
    write(Path(output), spec)
    return {'status':'SUITE_SPEC_DERIVED_NOT_EXECUTED','source_manifest_sha256':sha(Path(sources)),
            'template_sha256':sha(Path(template)), 'suite_sha256':sha(Path(output)),
            'counts':compiled['semantic']['counts'],'GPU_queries':0,'tokenizer_calls':0,'ledger_changes':0}


def first_serving_unit(binding):
    value = read(Path(binding))
    require(value.get('status') == 'CPU_BOUND_NOT_EXECUTED',
            'Full CPU live binding required; metadata-only layout is not ready')
    units = value.get('execution_units')
    require(isinstance(units, list) and units and all(isinstance(u, dict) for u in units),
            'INVALID_EMPTY_PLAN: execution units missing')
    ids = [u.get('node_id') for u in units]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)),
            'Invalid/duplicate execution unit identity')
    serving = [u for u in units if u.get('operation_kind') == 'serve_group']
    require(serving and serving[0].get('required') is True and
            type(serving[0].get('counts', {}).get('inference_POSTs')) is int and
            serving[0]['counts']['inference_POSTs'] > 0, 'No nonempty required serving execution unit')
    return serving[0]['node_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    new = actions.add_parser('session', help='new enclosing 3600-second session; no authorization')
    new.add_argument('--output', type=Path, required=True)
    new.add_argument('--started-utc', help='actual earlier enclosing session UTC; never resets the session deadline')
    clock = actions.add_parser('clock', help='new native task clock immediately before live binding')
    clock.add_argument('--session', type=Path, required=True); clock.add_argument('--output', type=Path, required=True)
    check = actions.add_parser('check', help='verify package/host/boot/clock/outer session before explicit run/resume')
    check.add_argument('--session', type=Path, required=True); check.add_argument('--clock', type=Path, required=True)
    local = actions.add_parser('local', help='write new resolved local binding without editing its input')
    for name in ('binding', 'prepared', 'session', 'clock', 'output'): local.add_argument('--'+name, type=Path, required=True)
    suite = actions.add_parser('suite', help='derive exact all-record SuiteSpec from validated prepared sources')
    for name in ('template', 'sources', 'output'): suite.add_argument('--'+name, type=Path, required=True)
    first = actions.add_parser('first-unit', help='print actual first required serving node ID from full binding')
    first.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'session': result = start_session(args.output, args.started_utc)
    elif args.action == 'clock': result = make_clock(args.session, args.output)
    elif args.action == 'check': result = {'status': 'CLOCK_VALID_CURRENT_SESSION', 'identity': check_clock(args.session, args.clock)['identity'], 'authorization': 'NONE'}
    elif args.action == 'suite': result = make_suite(args.template, args.sources, args.output)
    elif args.action == 'local': result = resolve_local(args.binding, args.prepared, args.session, args.clock, args.output)
    else:
        print(first_serving_unit(args.binding)); return
    import json
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
