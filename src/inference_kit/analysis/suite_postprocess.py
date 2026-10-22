"""A resumable CPU analysis revision over immutable, previously acquired raw.

This is not live-run migration. It dispatches only the four existing CPU
operations, with separate source identity, completion receipts and workspace.
"""
from pathlib import Path

from ..core.io import read, write, require, digest, sha
from ..core.state import new_output, atomic
from ..suites.suite_execution import _lock, _event, _events, _inventory, _verify_files, _path, _utc
from .tokens import source_identity
from .suite_results import verify_run, execute_cpu

OPERATIONS = ('score_task', 'analyze_tokens', 'index_add', 'report')
KIND = 'suite_cpu_postprocess_v1'


def _revision(value):
    require(isinstance(value, str) and value.strip() == value and 0 < len(value) <= 128
            and not any(ord(c) < 32 for c in value), 'Invalid CPU analysis revision')
    return value


def _plan(root):
    record = read(_path(root, 'ANALYSIS_PLAN.json')); payload = dict(record)
    identity = payload.pop('record_identity')
    require(digest(payload) == identity, 'CPU analysis plan changed')
    semantic = record['semantic']
    require(record['kind'] == KIND and record['analysis_plan_identity'] == digest(semantic),
            'CPU analysis semantic identity changed')
    require(semantic['operations'] == list(OPERATIONS), 'Only the fixed CPU operations are permitted')
    _revision(semantic['analysis_revision'])
    if 'report_packaging' in semantic:
        from .report_packaging import validate_report_packaging
        require(semantic['report_packaging'] == validate_report_packaging(semantic['report_packaging']),
                'CPU report packaging policy changed')
    require(semantic['analyzer_source_digest'] == source_identity(),
            'CPU analyzer source changed; resume requires the original installed source')
    raw = Path(record['locators']['raw']); parent, _, _, compiled, _ = verify_run(raw)
    require(sha(parent/'RUN_MANIFEST.json') == semantic['parent_manifest_sha256'] and
            compiled['plan_identity'] == semantic['parent_plan_identity'], 'CPU parent evidence changed')
    historical = record['locators'].get('historical_index')
    if historical is not None:
        path = Path(historical)
        require(path.is_file() and not path.is_symlink() and sha(path) == semantic['historical_index_sha256'],
                'CPU historical index input changed')
    else:
        require(semantic['historical_index_sha256'] is None, 'Missing CPU historical index binding')
    _events(root)
    return record


def _completion(root, plan, name, parents):
    path = _path(root, 'completions/'+name+'.json')
    value = read(path); check = dict(value); identity = check.pop('completion_identity')
    require(digest(check) == identity and value['analysis_plan_identity'] == plan['analysis_plan_identity']
            and value['operation'] == name and value['status'] == 'CPU_COMPLETE_VALIDATED',
            'CPU completion identity changed')
    require(value['dependency_identity'] == digest(parents), 'CPU completion dependencies changed')
    require(value['result'].get('status') == 'COMPLETE' and type(value['attempt']) is int
            and value['attempt'] > 0, 'CPU completion has incomplete coverage or attempt')
    require(value['files'], 'EMPTY_REQUIRED_OUTPUT: CPU completion evidence absent')
    _verify_files(root, value['files'])
    target = _path(root, value['output'])
    require(target.is_dir() and _inventory(root, target) == value['files'], 'CPU output inventory differs')
    if name == 'index_add':
        from .suite_results import validate_index_completion
        validate_index_completion(target, plan['locators']['raw'])
    if name == 'report':
        from .suite_results import report_packaging_for_run, validate_report_completion
        policy = plan['semantic'].get('report_packaging', report_packaging_for_run(plan['locators']['raw']))
        validate_report_completion(target, policy)
    return value


def _drive(root, stop_after=None):
    require(stop_after is None or stop_after in OPERATIONS, 'Unknown CPU stop boundary')
    plan = _plan(root); state = read(_path(root, 'checkpoint.json'))
    require(state['analysis_plan_identity'] == plan['analysis_plan_identity'] and
            set(state['nodes']) == set(OPERATIONS), 'CPU checkpoint plan coverage differs')
    require(all(type(entry.get('attempts')) is int and entry['attempts'] >= 0
                for entry in state['nodes'].values()), 'Invalid CPU checkpoint attempt')
    original = digest(state); completions = {}; stages = {}
    historical = plan['locators'].get('historical_index')
    if historical is not None:
        stages.update(historical_index=historical,
                      historical_index_sha256=plan['semantic']['historical_index_sha256'])
    stages['analysis_revision'] = plan['semantic']['analysis_revision']
    if 'report_packaging' in plan['semantic']:
        stages['report_packaging'] = plan['semantic']['report_packaging']
    pending = []
    for name in OPERATIONS:
        entry = state['nodes'][name]; marker = root/'completions'/(name+'.json')
        if marker.exists():
            require(not pending, 'CPU completion lacks a required earlier dependency')
            completion = _completion(root, plan, name, completions)
            completions[name] = completion['completion_identity']
            stages[name] = root/completion['output']
            require(entry['attempts'] == completion['attempt'], 'CPU completion attempt differs')
            entry.update(status='CPU_COMPLETE_VALIDATED', output=completion['output'],
                         completion='completions/'+name+'.json')
        else:
            require(entry['status'] != 'CPU_COMPLETE_VALIDATED', 'MISSING_EVIDENCE: CPU completion absent')
            pending.append(name)
    if not pending:
        state['status'] = 'CPU_POSTPROCESS_COMPLETE'
        reconciled = digest(state) != original
        if reconciled:
            atomic(root/'checkpoint.json', state)
        _event(root, 'CPU_NOOP', checkpoint_reconciled=reconciled)
        return _result(root, state, noop=True, checkpoint_reconciled=reconciled)
    state['controller_sessions'] += 1; state['status'] = 'CPU_RUNNING'
    atomic(root/'checkpoint.json', state); _event(root, 'CPU_CONTROLLER_START', pending=pending)
    for name in pending:
        entry = state['nodes'][name]; entry['attempts'] += 1; entry['status'] = 'CPU_RUNNING'
        atomic(root/'checkpoint.json', state)
        output = root/'stages'/name/('attempt'+str(entry['attempts']))
        _event(root, 'CPU_OPERATION_START', operation=name, attempt=entry['attempts'])
        try:
            result = execute_cpu(name, Path(plan['locators']['raw']), output, stage_outputs=stages)
            require(result.get('status') == 'COMPLETE', 'CPU operation did not produce complete verified coverage')
            files = _inventory(root, output)
            require(files, 'EMPTY_REQUIRED_OUTPUT: CPU operation returned no evidence')
            completion = {'schema_version': 1, 'status': 'CPU_COMPLETE_VALIDATED', 'operation': name,
                'analysis_plan_identity': plan['analysis_plan_identity'], 'dependency_identity': digest(completions),
                'attempt': entry['attempts'], 'output': output.relative_to(root).as_posix(), 'files': files,
                'result': result, 'new_acquisitions': 0, 'new_GPU_actions': 0, 'completed_utc': _utc()}
            completion['completion_identity'] = digest(completion)
            write(root/'completions'/(name+'.json'), completion)
            entry.update(status='CPU_COMPLETE_VALIDATED', output=completion['output'],
                         completion='completions/'+name+'.json')
            completions[name] = completion['completion_identity']; stages[name] = output
            atomic(root/'checkpoint.json', state)
            _event(root, 'CPU_OPERATION_COMPLETE', operation=name, completion_identity=completion['completion_identity'])
        except BaseException as error:
            # A committed receipt is authoritative if only checkpoint update
            # failed. The next resume validates it before reconciling state.
            entry['status'] = 'CPU_FAILED'; state['status'] = 'CPU_POSTPROCESS_FAILED'
            atomic(root/'checkpoint.json', state)
            _event(root, 'CPU_OPERATION_FAILED', operation=name, error_type=type(error).__name__, error=str(error))
            raise
        if stop_after == name and name != OPERATIONS[-1]:
            state['status'] = 'CPU_PAUSED_BOUNDARY'; atomic(root/'checkpoint.json', state)
            _event(root, 'CPU_PAUSED_BOUNDARY', after=name)
            return _result(root, state, noop=False)
    state['status'] = 'CPU_POSTPROCESS_COMPLETE'; atomic(root/'checkpoint.json', state)
    _event(root, 'CPU_POSTPROCESS_COMPLETE')
    return _result(root, state, noop=False)


def _result(root, state, **details):
    return {'schema_version': 1, 'status': state['status'], 'analysis_plan_identity': state['analysis_plan_identity'],
        'analysis_workspace': str(root), 'nodes': state['nodes'], 'controller_sessions': state['controller_sessions'],
        'new_acquisitions': 0, 'GPU_queries': 0, 'model_loads': 0, 'tokenizer_calls': 0,
        'inference_POSTs': 0, 'campaign_debit': 0, 'historical_live_controller': 'UNCHANGED', **details}


def postprocess(run, output, *, historical_index=None, analysis_revision=None, stop_after=None,
                report_packaging=None):
    """Start a new CPU revision; never resume or modify the parent live run."""
    parent, _, _, compiled, _ = verify_run(run); source = source_identity(); parent_sha = sha(parent/'RUN_MANIFEST.json')
    output = Path(output).absolute()
    require(not output.resolve().is_relative_to(parent.resolve()), 'CPU output must be outside immutable parent evidence')
    revision = _revision(analysis_revision) if analysis_revision is not None else 'cpu-v1:'+digest([parent_sha, source])
    history = Path(historical_index).absolute() if historical_index is not None else None
    if history is not None:
        require(history.is_file() and not history.is_symlink(), 'Missing explicit historical index')
    require(stop_after is None or stop_after in OPERATIONS, 'Unknown CPU stop boundary')
    semantic = {'analyzer': KIND, 'analyzer_source_digest': source, 'analysis_revision': revision,
        'parent_manifest_sha256': parent_sha, 'parent_plan_identity': compiled['plan_identity'],
        'parent_acquisition_source_digest': compiled['semantic']['compiler']['source_digest'],
        'historical_index_sha256': sha(history) if history is not None else None, 'operations': list(OPERATIONS)}
    from .report_packaging import validate_report_packaging
    from .suite_results import report_packaging_for_run
    semantic['report_packaging'] = validate_report_packaging(
        report_packaging if report_packaging is not None else report_packaging_for_run(parent))
    root = new_output(output)
    plan = {'kind': KIND, 'schema_version': 1, 'semantic': semantic, 'analysis_plan_identity': digest(semantic),
        'locators': {'raw': str(parent), 'historical_index': str(history) if history else None}, 'created_utc': _utc()}
    plan['record_identity'] = digest(plan); write(root/'ANALYSIS_PLAN.json', plan)
    atomic(root/'checkpoint.json', {'schema_version': 1, 'analysis_plan_identity': plan['analysis_plan_identity'],
        'status': 'CPU_PENDING', 'controller_sessions': 0,
        'nodes': {kind: {'status': 'CPU_PENDING', 'attempts': 0} for kind in OPERATIONS}})
    with _lock(root):
        return _drive(root, stop_after)


def resume(workspace, *, stop_after=None):
    root = Path(workspace).absolute()
    require(root.is_dir() and not root.is_symlink(), 'Missing or unsafe CPU analysis workspace')
    with _lock(root):
        return _drive(root, stop_after)


def status(workspace):
    """Read and validate current CPU completions without rewriting state."""
    root = Path(workspace).absolute()
    with _lock(root):
        plan = _plan(root); state = read(_path(root, 'checkpoint.json')); completed = {}
        require(state['analysis_plan_identity'] == plan['analysis_plan_identity'] and
                set(state['nodes']) == set(OPERATIONS), 'CPU checkpoint coverage differs')
        require(all(type(entry.get('attempts')) is int and entry['attempts'] >= 0
                    for entry in state['nodes'].values()), 'Invalid CPU checkpoint attempt')
        for name in OPERATIONS:
            if state['nodes'][name]['status'] == 'CPU_COMPLETE_VALIDATED':
                completion = _completion(root, plan, name, completed)
                require(state['nodes'][name]['attempts'] == completion['attempt'], 'CPU completion attempt differs')
                completed[name] = completion['completion_identity']
        require(state['status'] != 'CPU_POSTPROCESS_COMPLETE' or len(completed) == len(OPERATIONS),
                'CPU complete status lacks required completion evidence')
        return _result(root, state, verified_completed=list(completed))
