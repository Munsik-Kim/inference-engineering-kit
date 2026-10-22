"""Rebuildable, local evidence index. Raw evidence remains the authority.

Only explicit built-in data adapters run. No SQL, archive code, model or network
execution is accepted from a source manifest.
"""
from contextlib import closing, ExitStack
import json
import math
from pathlib import Path
import sqlite3

from ..core.io import read, write, loads, digest, sha, require, safe_name
from ..core.evidence import open_bundle
from ..core.state import new_output

SCHEMA = 1
TABLES = ('sources', 'acquisitions', 'analyses', 'samples', 'executions',
          'measurements', 'evidence', 'relations')


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _connect(path, writable=False):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'Missing/unsafe index database')
    connection = sqlite3.connect(path.as_uri() + ('?mode=rw' if writable else '?mode=ro'), uri=True)
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('PRAGMA trusted_schema=OFF')
    require(connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA, 'Unsupported index schema')
    require(not connection.execute("SELECT name FROM sqlite_master WHERE type IN ('trigger','view')").fetchall(), 'Index triggers/views are forbidden')
    if not writable: connection.execute('PRAGMA query_only=ON')
    return connection


def _create(path):
    path = Path(path).absolute()
    require(not path.exists() and not path.is_symlink(), 'Refusing existing index database')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a raced existing database from being modified.
    with path.open('xb'): pass
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.executescript('''
            CREATE TABLE sources(id TEXT PRIMARY KEY, content_identity TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE aliases(alias TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id));
            CREATE TABLE locators(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), payload TEXT NOT NULL);
            CREATE TABLE acquisitions(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), payload TEXT NOT NULL);
            CREATE TABLE analyses(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), acquisition_id TEXT NOT NULL REFERENCES acquisitions(id), payload TEXT NOT NULL);
            CREATE TABLE samples(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), acquisition_id TEXT NOT NULL REFERENCES acquisitions(id), task_id TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE executions(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), acquisition_id TEXT NOT NULL REFERENCES acquisitions(id), payload TEXT NOT NULL);
            CREATE TABLE measurements(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), analysis_id TEXT NOT NULL REFERENCES analyses(id), sample_id TEXT REFERENCES samples(id), execution_id TEXT REFERENCES executions(id), task_id TEXT NOT NULL, metric TEXT NOT NULL, status TEXT NOT NULL, scope TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE evidence(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), relative_path TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(source_id,relative_path));
            CREATE TABLE relations(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), kind TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX measurement_task ON measurements(task_id,metric,scope);
            PRAGMA user_version=1;
        ''')
        connection.commit()
    return path


def _insert(connection, table, row):
    cols = list(row)
    old = connection.execute('SELECT ' + ','.join(cols) + ' FROM ' + table + ' WHERE id=?', (row['id'],)).fetchone()
    if old is not None:
        require(tuple(row[c] for c in cols) == old, 'SOURCE_ID_CONFLICT: ' + table)
        return
    connection.execute('INSERT INTO ' + table + '(' + ','.join(cols) + ') VALUES (' + ','.join('?' for _ in cols) + ')', tuple(row[c] for c in cols))


def _alias_identity(alias, scope=None):
    """Separate a caller's display alias from its explicit source namespace.

    Unscoped aliases retain the original schema-1 meaning. Scoped aliases use a
    deterministic, reversible encoding; neither spelling changes source IDs,
    acquisition IDs, or analysis IDs. A collision inside one scope still fails.
    """
    require(isinstance(alias, str) and alias, 'Explicit source alias required')
    require(scope is None or isinstance(scope, str) and 0 < len(scope) <= 256,
            'Invalid source alias scope')
    require(scope is not None or not alias.startswith('scoped-alias-v1:'),
            'Reserved scoped source alias encoding')
    return alias if scope is None else 'scoped-alias-v1:' + _json([scope, alias])


def _store(connection, envelope, alias, alias_scope=None):
    sid = envelope['source_id']
    content = envelope['content_identity']
    storage_alias = _alias_identity(alias, alias_scope)
    old_alias = connection.execute('SELECT source_id FROM aliases WHERE alias=?', (storage_alias,)).fetchone()
    require(old_alias is None or old_alias[0] == sid, 'SOURCE_ID_CONFLICT: declared source alias')
    source_payload = envelope.get('payload', {})
    _insert(connection, 'sources', {'id':sid, 'content_identity':content, 'payload':_json(source_payload)})
    if old_alias is None:
        connection.execute('INSERT INTO aliases VALUES (?,?)', (storage_alias, sid))
    locator = envelope['locator']
    _insert(connection, 'locators', {'id':digest([sid, locator]), 'source_id':sid, 'payload':_json(locator)})
    for table in TABLES[1:]:
        seen = set()
        for raw in envelope.get(table, []):
            require(raw['id'] not in seen, 'Duplicate envelope ID: ' + table); seen.add(raw['id'])
            obj = dict(raw)
            if table == 'evidence': safe_name(obj['relative_path'])
            if table == 'measurements':
                value = obj.get('value')
                require(value is None or type(value) in (int, float, bool, str), 'Non-scalar measurement value')
                require(type(value) is not float or math.isfinite(value), 'Nonfinite measurement')
                require(not (obj.get('status') in ('MISSING', 'UNKNOWN', 'NOT_APPLICABLE', 'INVALID_ACQUISITION') and value is not None), 'Missing/invalid measurement must be null')
            columns = {'id':obj['id'], 'source_id':sid}
            for name in {'acquisitions':(), 'analyses':('acquisition_id',), 'samples':('acquisition_id','task_id'),
                         'executions':('acquisition_id',), 'measurements':('analysis_id','sample_id','execution_id','task_id','metric','status','scope'),
                         'evidence':('relative_path',), 'relations':('kind','status')}[table]:
                columns[name] = obj.get(name)
            columns['payload'] = _json(obj)
            _insert(connection, table, columns)


def _refresh_relations(connection):
    """Prove historical lineage without reclassifying it as a paired execution."""
    sources = {sid:loads(payload) for sid,payload in connection.execute('SELECT id,payload FROM sources')}
    contents = {sid:content for sid,content in connection.execute('SELECT id,content_identity FROM sources')}
    samples = [(sid, source, loads(payload)) for sid,source,payload in connection.execute('SELECT id,source_id,payload FROM samples')]
    for left_id, source_id, left in samples:
        lp = left.get('payload', {}); prepared = lp.get('prepared', {})
        if sources[source_id].get('adapter') not in ('structure_d1','token_c2'): continue
        scopes = [scope for (scope,) in connection.execute('SELECT DISTINCT scope FROM measurements WHERE sample_id=? AND scope IN (?,?,?)',
                    (left_id,'structure_pair','projection_pair','token_pair'))]
        if scopes:
            relation = {'id':digest([left_id,'WITHIN_RUN_MODEL_PAIR']), 'kind':'WITHIN_RUN_MODEL_PAIR',
                'status':'EXISTING_COMPARATOR_CONTRACT_VERIFIED', 'left_id':left_id, 'right_id':left_id,
                'payload':{'scopes':sorted(scopes),'verification':'Existing C1/D1 readers check full prefix/coordinate/mask identities before computing metrics',
                           'treatment':'artifact/effective weight difference','deployment_equivalence':'NOT_ESTABLISHED'}}
            _insert(connection,'relations',{'id':relation['id'],'source_id':source_id,'kind':relation['kind'],'status':relation['status'],'payload':_json(relation)})
        for right_id, right_source, right in samples:
            rp = right.get('payload', {}); rprepared = rp.get('prepared', {})
            if left_id == right_id or lp.get('task_id') != rp.get('task_id') or lp.get('sample_id') != rp.get('sample_id'): continue
            historical = sources[right_source].get('adapter', '').startswith('task_')
            if historical:
                checks = {'source_manifest':prepared.get('source_bundle_identity') == contents[right_source],
                          'source_plan':prepared.get('source_plan_identity') == sources[right_source].get('plan_identity')}
                for key in ('task_version','source_id','prompt_hash','reference_hash','tokenizer_identity','token_id_space'):
                    checks[key] = lp.get(key) is not None and lp.get(key) == rp.get(key)
                kind = 'HISTORICAL_OUTCOME_LINK'; status = 'PROVEN_RELATED_ONLY' if all(checks.values()) else 'BLOCKED_INSUFFICIENT_IDENTITY'
                reason = 'Different SERVE and teacher-forced diagnostic executions; no quality/internal-state equivalence'
            elif sources[source_id].get('adapter') == 'structure_d1' and sources[right_source].get('adapter') == 'token_c2':
                checks = {key:prepared.get(key) is not None and prepared.get(key) == rprepared.get(key)
                          for key in ('probe_input_identity','sequence_identity','input_ids','prompt_hash','reference_hash','tokenizer_identity','token_id_space')}
                # Tokenizer identity may reside at the source sample level.
                for key in ('tokenizer_identity','token_id_space'):
                    checks[key] = lp.get(key) is not None and lp.get(key) == rp.get(key)
                kind = 'COMMON_COORDINATES_ONLY'; status = 'RELATED_NO_AUTOMATIC_DELTA' if all(checks.values()) else 'INCOMPARABLE'
                reason = 'C2 token-only versus D1 ON instrumentation/pass; common input does not authorize subtracting aggregate means'
            else: continue
            relation = {'id':digest([left_id,right_id,kind]), 'kind':kind, 'status':status,
                        'left_id':left_id,'right_id':right_id,'delta_permitted':False,
                        'payload':{'checks':checks,'reason':reason}}
            if kind == 'COMMON_COORDINATES_ONLY':
                lpos = {t['token_position'] for t in prepared.get('targets', [])}
                rpos = {t['token_position'] for t in rprepared.get('targets', [])}
                relation['payload']['coordinate_coverage'] = {'common':sorted(lpos&rpos),
                    'left_only':sorted(lpos-rpos),'right_only':sorted(rpos-lpos),
                    'left_requested':len(lpos),'right_requested':len(rpos)}
            _insert(connection,'relations',{'id':relation['id'],'source_id':source_id,'kind':kind,'status':status,'payload':_json(relation)})


def counts(connection):
    return {t:connection.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in TABLES}


def add_sources(sources, db, output):
    """Verify each source before its atomic transaction; failed imports leave no partial source."""
    from .index_sources import load_source
    path = Path(sources).absolute(); manifest = read(path)
    require(manifest.get('schema_version') == 1 and isinstance(manifest.get('sources'), list) and manifest['sources'], 'Explicit index sources schema required')
    aliases = [_alias_identity(s['source_id'], s.get('alias_scope')) for s in manifest['sources']]
    require(len(set(aliases)) == len(aliases), 'Duplicate declared source ID')
    out = new_output(output); receipts = []
    with closing(_connect(db, True)) as connection:
        before = counts(connection)
        for spec in manifest['sources']:
            try:
                envelope = load_source(spec, path.parent)
                for review in manifest.get('reviews', []):
                    if review.get('source_alias') != spec['source_id']: continue
                    if review.get('alias_scope') != spec.get('alias_scope'): continue
                    review_path = path.parent / safe_name(review['path'])
                    require(review_path.is_file() and not review_path.is_symlink() and sha(review_path) == review['sha256'], 'Review receipt identity changed')
                    data = read(review_path)
                    envelope['relations'].append({'id':digest([envelope['source_id'],'EVIDENCE_REVIEW',review['sha256']]),
                        'kind':'EVIDENCE_REVIEW','status':'REVIEWED_WITH_STATED_LIMITS',
                        'payload':{'review_sha256':review['sha256'],'review':data,'new_acquisition':False}})
                with connection:
                    _store(connection, envelope, spec['source_id'], spec.get('alias_scope'))
                    _refresh_relations(connection)
                receipts.append({'source_alias':spec['source_id'], 'alias_scope':spec.get('alias_scope'),
                    'alias_storage_key':_alias_identity(spec['source_id'], spec.get('alias_scope')),
                    'source_id':envelope['source_id'], 'status':'REGISTERED', 'counts_after':counts(connection)})
            except Exception as error:
                receipts.append({'source_alias':spec['source_id'], 'status':'IMPORT_REJECTED_ROLLED_BACK', 'reason':str(error)})
                write(out/'IMPORT_RECEIPT.json', {'schema_version':SCHEMA, 'before':before, 'after':counts(connection), 'sources':receipts, 'GPU_actions':0})
                raise
        result = {'schema_version':SCHEMA, 'status':'REGISTERED', 'before':before, 'after':counts(connection), 'sources':receipts, 'GPU_actions':0, 'network_actions':0}
    write(out/'IMPORT_RECEIPT.json', result)
    return result


def build_index(sources, db, output):
    _create(db)
    return add_sources(sources, db, output)


def _mapped_path(path, root_map):
    original = Path(path)
    matches = []
    for old, new in (root_map or {}).items():
        base = Path(old)
        if original == base or base in original.parents:
            matches.append((len(base.parts), Path(new) / original.relative_to(base)))
    return max(matches, key=lambda item:item[0])[1] if matches else original


def validate_locator(locator, root_map=None):
    """Recheck content, without executing any file in the source."""
    path = _mapped_path(locator['path'], root_map)
    if not path.exists(): return {'status':'MISSING_EVIDENCE', 'reason':'Bound source is unavailable'}
    try:
        require(not path.is_symlink(), 'Source symlink')
        if path.is_file() and locator.get('archive_sha256'):
            require(sha(path) == locator['archive_sha256'], 'Archive content changed')
        with open_bundle(path) as root:
            selected = root / safe_name(locator['relative_root']) if locator.get('relative_root') else root
            require(selected.resolve().is_relative_to(root.resolve()), 'Source root escape')
            inventory = locator['files']; require(inventory, 'Empty evidence inventory')
            for name, identity in inventory.items():
                member = selected / safe_name(name)
                require(member.is_file() and member.resolve().is_relative_to(selected.resolve()), 'Missing/escaped evidence member')
                require(not any(p.is_symlink() for p in (member, *member.parents) if p != selected.parent), 'Evidence symlink')
                require(member.stat().st_size == identity['bytes'] and sha(member) == identity['sha256'], 'Evidence content changed: ' + name)
        return {'status':'RAW_HASH_VERIFIED', 'verified_files':len(inventory)}
    except (ValueError, OSError, KeyError) as error:
        return {'status':'STALE', 'reason':str(error)}


def canonical_export(db, task=None, root_map=None, validate=True):
    require(task is None or isinstance(task, str), 'Invalid task filter')
    result = {'schema_version':SCHEMA, 'kind':'local_evidence_index', 'new_gpu_measurements':0}
    with closing(_connect(db)) as connection:
        for table in TABLES:
            rows = []
            for row in connection.execute('SELECT * FROM ' + table + ' ORDER BY id'):
                names = [d[0] for d in connection.execute('SELECT * FROM '+table+' LIMIT 0').description]
                obj = dict(zip(names, row)); payload = loads(obj.pop('payload'))
                if table == 'sources': obj['payload'] = payload
                else: obj = {**payload, **obj}
                rows.append(obj)
            result[table] = rows
        if task:
            require(any(r['task_id'] == task for r in result['samples']), 'Unknown task filter')
            result['samples'] = [r for r in result['samples'] if r['task_id'] == task]
            result['measurements'] = [r for r in result['measurements'] if r['task_id'] == task]
        if validate:
            validations = {}
            for sid, payload in connection.execute('SELECT source_id,payload FROM locators ORDER BY id'):
                state = validate_locator(loads(payload), root_map)
                validations.setdefault(sid, []).append(state)
            for source in result['sources']:
                states = validations.get(source['id'], [])
                good = next((s for s in states if s['status'] == 'RAW_HASH_VERIFIED'), None)
                source['validation'] = good or {'status':'STALE' if any(s['status']=='STALE' for s in states) else 'MISSING_EVIDENCE', 'locators':states}
            statuses = {s['id']:s['validation']['status'] for s in result['sources']}
            for row in result['measurements']:
                row['evidence_status'] = statuses[row['source_id']]
                if row['evidence_status'] != 'RAW_HASH_VERIFIED':
                    row['value'] = None; row['status'] = row['evidence_status']
    result['logical_identity'] = digest({k:v for k,v in result.items() if k != 'logical_identity'})
    return result


def query_index(db, output, task=None, root_map=None):
    out = new_output(output); export = canonical_export(db, task, root_map)
    write(out/'canonical-export.json', export)
    result = {'status':'EXPORTED', 'logical_identity':export['logical_identity'], 'counts':{t:len(export[t]) for t in TABLES}, 'GPU_actions':0}
    write(out/'QUERY_RECEIPT.json', result); return result


def report_index(db, output, root_map=None, packaging=None):
    from .index_report import render_report
    from .report_packaging import validate_report_packaging
    policy = validate_report_packaging(packaging)
    export = canonical_export(db, root_map=root_map)
    export['compatibility'] = {'relations':[{'id':r['id'],'kind':r['kind'],'status':r['status']} for r in export['relations']],
                              'negative_control':_negative_comparison_control(export),
                              'cross_run_delta':'NOT_AUTOMATICALLY_AUTHORIZED', 'combined_task_score':'NOT_DEFINED'}
    # Only already manifested raw members become portable links. Bounded copies
    # contain data, not executables; no inferred model/runtime path is followed.
    with closing(_connect(db)) as connection, ExitStack() as stack:
        roots = {}
        for sid, payload in connection.execute('SELECT source_id,payload FROM locators ORDER BY id'):
            if sid in roots: continue
            locator = loads(payload)
            if validate_locator(locator, root_map)['status'] != 'RAW_HASH_VERIFIED': continue
            bundle = stack.enter_context(open_bundle(_mapped_path(locator['path'], root_map)))
            roots[sid] = bundle / safe_name(locator['relative_root']) if locator.get('relative_root') else bundle
        total = 0
        for evidence in export['evidence']:
            source = evidence['source_id']; relative = evidence['relative_path']
            if source not in roots:
                evidence['copy_exclusion_status'] = next((s['validation']['status'] for s in export['sources'] if s['id'] == source), 'MISSING_EVIDENCE')
                evidence['copy_exclusion_reason'] = 'No currently hash-verified source locator'
                continue
            # Numeric data and frozen JSON/HTTP/log receipts only. Never Python,
            # shell, HTML, pickle, model weights or arbitrary archive members.
            if Path(relative).suffix.lower() not in ('.json','.jsonl','.csv','.txt','.log','.f32'):
                evidence['copy_exclusion_status'] = 'UNSUPPORTED_PORTABLE_TYPE'
                evidence['copy_exclusion_reason'] = 'Excluded executable/archive or unsupported evidence type'
                continue
            total += evidence['bytes']
            if policy['mode'] == 'single_report_v1':
                require(total <= 256*1024**2, 'Portable evidence byte cap exceeded')
            evidence['approved_copy'] = {'source_root':str(roots[source]), 'relative_path':relative,
                'portable_path':'evidence/'+source+'/'+relative, 'sha256':evidence['sha256'], 'max_bytes':64*1024**2}
        result = render_report(export, output, packaging=policy)
        if policy['mode'] == 'single_report_v1':
            write(Path(output)/'COMPATIBILITY_REPORT.json', export['compatibility'])
    result['raw_copy_scope'] = 'Explicit hash-verified source manifest data members only; excluded executable types are not copied'
    return result


def _negative_comparison_control(export):
    """An explicit rejection check using actual different selected head axes."""
    grouped = {}
    for row in export['measurements']:
        c = row.get('coordinates', {})
        if row['scope'] != 'structure_pair' or row['metric'] != 'attention_row.js_nat': continue
        key = (row['source_id'],row['sample_id'],c.get('layer'),c.get('query_position'))
        if key in grouped and grouped[key].get('coordinates', {}).get('head') != c.get('head'):
            first = grouped[key]
            left = {'task_id':first['task_id'],'metric':first['metric'],'coordinate_identity':digest(first['coordinates'])}
            right = {'task_id':row['task_id'],'metric':row['metric'],'coordinate_identity':digest(c)}
            return {'kind':'DECLARED_NEGATIVE_CONTROL_NOT_A_MODEL_MEASUREMENT',
                    'left_measurement':first['id'],'right_measurement':row['id'],
                    **compatibility(left,right), 'test_expectation':'INCOMPARABLE: different head axes'}
        grouped[key] = row
    return {'status':'NOT_AVAILABLE','reason':'No two different actual head coordinates in this index'}


def compatibility(left, right, mode='WITHIN_RUN_MODEL_PAIR'):
    """Connecting samples does not grant permission to subtract measurements."""
    invariant = ('task_id','task_version','dataset_identity','sample_id','policy_id','reference_role',
                 'prompt_hash','reference_hash','sequence_identity','tokenizer_identity','token_id_space',
                 'metric','metric_version','scope','coordinate_identity','probability_kind')
    reasons = [key+' differs or missing' for key in invariant if key not in left or key not in right or left[key] != right[key]]
    if mode == 'WITHIN_RUN_MODEL_PAIR':
        for key in ('acquisition_id','runtime_identity','pass_id'):
            if not left.get(key) or left.get(key) != right.get(key): reasons.append(key+' differs or missing')
    else:
        reasons.append('Cross-run/trajectory comparison requires an explicit reviewed contract; related display only')
    return {'compare_mode':mode, 'status':'INCOMPARABLE' if reasons else 'COMPARABLE', 'reasons':reasons,
            'artifact_axis':'allowed treatment', 'delta_permitted':not reasons}
