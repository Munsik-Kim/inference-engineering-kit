"""Finite, manifest-closed offline report collections; no raw or metric rewriting.

The legacy report has a 256 MiB evidence budget.  A collection is an explicit
different policy: at most two individually bounded volumes, with independently
bounded evidence, presentation, and total output.  No file is split or deduped
across source identities.  Atomic publication covers a directory rename, not
power-loss durability or a transaction with the caller's index database.
"""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile

from ..core.io import digest, fields, read, require, safe_name, sha, write


MIB = 1024 ** 2
LIMITS = {'volume_payload_bytes': 256 * MIB, 'max_volumes': 2,
          'evidence_payload_bytes': 512 * MIB, 'presentation_bytes': 128 * MIB,
          'collection_bytes': 640 * MIB, 'member_bytes': 64 * MIB,
          'max_members': 10000}
_HEX = frozenset('0123456789abcdef')


def validate_report_packaging(policy=None):
    """Normalize the versioned policy. User bounds can only decrease defaults."""
    if policy is None:
        return {'schema_version': 1, 'mode': 'single_report_v1'}
    fields(policy, ('schema_version', 'mode'), LIMITS)
    require(type(policy['schema_version']) is int and policy['schema_version'] == 1,
            'Unsupported report packaging schema')
    require(policy['mode'] in ('single_report_v1', 'bounded_collection_v1'),
            'Unsupported report packaging mode')
    if policy['mode'] == 'single_report_v1':
        require(set(policy) == {'schema_version', 'mode'}, 'Legacy report bounds are fixed')
        return dict(policy)
    result = {'schema_version': 1, 'mode': 'bounded_collection_v1'}
    for key, limit in LIMITS.items():
        value = policy.get(key, limit)
        require(type(value) is int and 0 < value <= limit,
                'Invalid/downward-only report packaging bound: ' + key)
        result[key] = value
    return result


def _hash(value):
    require(isinstance(value, str) and len(value) == 64 and set(value) <= _HEX,
            'Invalid report member SHA256')
    return value


def _file(root, relative):
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), 'Missing/unsafe report collection root')
    require(not any(p.is_symlink() for p in root.parents), 'Report collection parent symlink refused')
    current = root
    for component in Path(safe_name(relative)).parts:
        current = current / component
        require(not current.is_symlink(), 'Report collection symlink refused')
    require(current.is_file() and current.resolve().is_relative_to(root.resolve()),
            'Missing/escaped report collection member: ' + relative)
    return current


def _identity(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def _verify_file(root, relative, identity):
    require(isinstance(identity, dict) and type(identity.get('bytes')) is int and identity['bytes'] >= 0,
            'Invalid report member byte count')
    _hash(identity.get('sha256'))
    member = _file(root, relative)
    require(member.stat().st_size == identity['bytes'] and sha(member) == identity['sha256'],
            'Report collection member content changed: ' + relative)
    return member


def _all_files(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        require(not path.is_symlink(), 'Report collection symlink refused')
        if path.is_file(): result[path.relative_to(root).as_posix()] = path
        else: require(path.is_dir(), 'Unsupported report collection member type')
    return result


def _inventory(export, policy):
    """Freeze logical keys before any output or copy; order is registration-independent."""
    entries, excluded = [], []
    identifiers, keys, source_paths = set(), set(), set()
    for row in export.get('evidence', []):
        identifier = row.get('id')
        require(isinstance(identifier, str) and identifier and identifier not in identifiers,
                'Duplicate/missing report evidence ID')
        identifiers.add(identifier)
        declaration = row.get('approved_copy')
        if declaration is None:
            excluded.append({'evidence_id': identifier, 'source_id': row.get('source_id'),
                             'relative_path': row.get('relative_path'),
                             'status': row.get('copy_exclusion_status', row.get('status', 'NOT_APPROVED')),
                             'reason': row.get('copy_exclusion_reason', 'No verified approved copy')})
            continue
        require(isinstance(declaration, dict), 'Invalid approved report evidence declaration')
        source = safe_name(row.get('source_id', ''))
        relative = safe_name(declaration['relative_path'])
        require(row.get('relative_path', relative) == relative, 'Report evidence relative path mismatch')
        identity = _hash(declaration.get('sha256'))
        size = row.get('bytes')
        # Direct render callers may omit a redundant size; the actual approved
        # source is checked before planning and copied against this same pin.
        from .index_report import _bounded_file
        original = _bounded_file(declaration['source_root'], relative)
        actual = original.stat().st_size
        require(size is None or type(size) is int and size == actual,
                'Approved report evidence byte count mismatch')
        require(actual <= policy['member_bytes'], 'Report evidence member byte cap exceeded')
        require(actual <= policy['volume_payload_bytes'], 'Report evidence cannot fit a bounded volume')
        key = (source, relative, identity)
        require(key not in keys and (source, relative) not in source_paths,
                'Conflicting/duplicate logical report evidence key')
        keys.add(key); source_paths.add((source, relative))
        entries.append({'evidence_id': identifier, 'source_id': source,
                        'relative_path': relative, 'sha256': identity, 'bytes': actual})
    entries.sort(key=lambda e: (e['source_id'], e['relative_path'], e['sha256']))
    excluded.sort(key=lambda e: e['evidence_id'])
    require(len(entries) <= policy['max_members'], 'Report evidence member count cap exceeded')
    total = sum(e['bytes'] for e in entries)
    require(total <= policy['evidence_payload_bytes'], 'Report collection evidence payload cap exceeded')
    require(total <= policy['collection_bytes'], 'Report collection total byte cap exceeded')
    volume, used = 1, 0
    for entry in entries:
        if used + entry['bytes'] > policy['volume_payload_bytes']:
            volume += 1; used = 0
        require(volume <= policy['max_volumes'], 'Report collection volume count cap exceeded')
        part = 'part-' + str(volume).zfill(4)
        entry['part'] = part
        entry['path'] = 'evidence/' + part + '/' + entry['source_id'] + '/' + entry['relative_path']
        safe_name(entry['path'])
        used += entry['bytes']
    return entries, excluded


def render_collection(export, output, packaging):
    """Stage, verify, then atomically publish every bounded volume and presentation file."""
    from .index_report import render_report, _bounded_json
    from .evidence_publication import _publication_lock
    policy = validate_report_packaging(packaging)
    require(policy['mode'] == 'bounded_collection_v1', 'Collection report policy required')
    entries, excluded = _inventory(export, policy)
    target = Path(output).expanduser().absolute()
    require(not target.exists() and not target.is_symlink(), 'Refuse overwrite existing report collection')
    require(not any(p.is_symlink() for p in target.parents), 'Report collection parent symlink refused')
    for row in export.get('evidence', []):
        declaration = row.get('approved_copy')
        if declaration:
            source = Path(declaration['source_root']).absolute()
            require(not target.resolve().is_relative_to(source.resolve()),
                    'Report output must be outside immutable evidence')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the full finite presentation allowance as well as planned raw.
    require(shutil.disk_usage(target.parent).free >= sum(e['bytes'] for e in entries) + policy['presentation_bytes'],
            'Insufficient disk for bounded report collection')
    # The existing exclusive publication lock serializes cooperating product
    # publishers. It is not protection against an unrelated process deliberately
    # replacing paths outside that protocol, nor power-loss durability.
    with _publication_lock(target.parent, 'report-' + digest(str(target))):
        require(not target.exists() and not target.is_symlink(), 'Refuse existing report collection')
        return _render_collection_locked(export, target, entries, excluded, policy, render_report, _bounded_json)


def _render_collection_locked(export, target, entries, excluded, policy, render_report, write_json):
    staging = Path(tempfile.mkdtemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent))
    temporary = staging / 'collection'
    presentation_budget = [min(policy['presentation_bytes'], policy['collection_bytes'] - sum(e['bytes'] for e in entries)), 0]
    try:
        prepared = deepcopy(export)
        mapping = {e['evidence_id']: e for e in entries}
        for row in prepared.get('evidence', []):
            if row['id'] in mapping:
                row['approved_copy']['portable_path'] = mapping[row['id']]['path']
                row['approved_copy']['max_bytes'] = policy['member_bytes']
        result = render_report(prepared, temporary, _presentation_budget=presentation_budget)
        if 'compatibility' in export:
            write_json(temporary / 'COMPATIBILITY_REPORT.json', export['compatibility'], presentation_budget)
        parts = []
        for name in sorted({e['part'] for e in entries}):
            members = [dict(e) for e in entries if e['part'] == name]
            part = {'schema_version': 1, 'kind': 'report_evidence_part_v1', 'part': name,
                    'members': members, 'payload_bytes': sum(e['bytes'] for e in members)}
            relative = 'evidence/' + name + '/PART_MANIFEST.json'
            write_json(temporary / relative, part, presentation_budget)
            parts.append({'part': name, 'path': relative, **_identity(temporary / relative),
                          'payload_bytes': part['payload_bytes'], 'member_count': len(members)})
        raw_paths = {e['path'] for e in entries}
        presentation = {name: _identity(path) for name, path in _all_files(temporary).items()
                        if name not in raw_paths}
        manifest = {'schema_version': 1, 'kind': 'bounded_report_collection_v1',
                    'packaging': policy, 'entrypoint': 'index.html',
                    'logical_inventory_identity': digest(entries),
                    'logical_evidence': entries, 'excluded_evidence': excluded,
                    'logical_member_count': len(entries), 'logical_payload_bytes': sum(e['bytes'] for e in entries),
                    'stored_payload_bytes': sum(e['bytes'] for e in entries), 'content_deduplication': False,
                    'parts': parts, 'presentation': presentation,
                    'presentation_bytes_excluding_root_manifest': sum(v['bytes'] for v in presentation.values()),
                    'closure': 'Root pins part and presentation hashes; parts pin every raw member. Root does not hash itself.',
                    'publication_atomicity': 'Exclusive cooperating-publisher lock and same-filesystem directory rename; not a database transaction or power-loss durability guarantee.'}
        write_json(temporary / 'REPORT_COLLECTION_MANIFEST.json', manifest, presentation_budget)
        verification = validate_report_collection(temporary)
        require(not target.exists() and not target.is_symlink(), 'Refuse report collection destination race')
        os.rename(temporary, target)
        result['packaging'] = policy
        result['collection'] = verification
        result['collection_manifest'] = 'REPORT_COLLECTION_MANIFEST.json'
        return result
    finally:
        shutil.rmtree(staging)


def validate_report_collection(output):
    """Recheck this locator's complete closure; an alternative good locator cannot substitute."""
    root = Path(output).expanduser().absolute()
    manifest_path = _file(root, 'REPORT_COLLECTION_MANIFEST.json')
    manifest = read(manifest_path)
    require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1 and manifest.get('kind') == 'bounded_report_collection_v1',
            'Unsupported report collection manifest')
    policy = validate_report_packaging(manifest.get('packaging'))
    require(policy['mode'] == 'bounded_collection_v1', 'Missing bounded collection policy')
    entries = manifest.get('logical_evidence')
    require(isinstance(entries, list) and len(entries) <= policy['max_members'], 'Invalid report logical inventory')
    require(entries == sorted(entries, key=lambda e: (e['source_id'], e['relative_path'], e['sha256'])) and
            digest(entries) == manifest.get('logical_inventory_identity'), 'Report logical inventory changed')
    require(len({e['evidence_id'] for e in entries}) == len(entries), 'Duplicate report evidence ID')
    require(len({(e['source_id'], e['relative_path']) for e in entries}) == len(entries), 'Conflicting report source path')
    excluded = manifest.get('excluded_evidence')
    require(isinstance(excluded, list) and all(isinstance(e, dict) and isinstance(e.get('evidence_id'), str) and e['evidence_id'] for e in excluded),
            'Missing/invalid excluded report evidence states')
    require(len({e['evidence_id'] for e in excluded}) == len(excluded) and not
            {e['evidence_id'] for e in excluded} & {e['evidence_id'] for e in entries},
            'Duplicate/conflicting excluded report evidence ID')
    require(manifest.get('content_deduplication') is False, 'Unsupported report deduplication policy')
    parts = manifest.get('parts')
    require(isinstance(parts, list) and len(parts) <= policy['max_volumes'], 'Report collection volume cap exceeded')
    require([p['part'] for p in parts] == ['part-' + str(i + 1).zfill(4) for i in range(len(parts))],
            'Noncanonical report part set')
    expected = {'REPORT_COLLECTION_MANIFEST.json'}
    from_parts = []
    for part in parts:
        require(part['path'] == 'evidence/' + part['part'] + '/PART_MANIFEST.json', 'Invalid report part path')
        path = _verify_file(root, part['path'], part); expected.add(part['path'])
        record = read(path)
        require(type(record.get('schema_version')) is int and record['schema_version'] == 1 and record.get('kind') == 'report_evidence_part_v1' and
                record.get('part') == part['part'], 'Invalid report part manifest')
        members = record.get('members')
        require(isinstance(members, list) and members, 'Empty/invalid report evidence part')
        used = 0
        for member in members:
            require(member.get('part') == part['part'] and
                    member['path'] == 'evidence/' + part['part'] + '/' + safe_name(member['source_id']) + '/' + safe_name(member['relative_path']),
                    'Invalid report member mapping')
            require(type(member.get('bytes')) is int and 0 <= member['bytes'] <= policy['member_bytes'],
                    'Report evidence member byte cap exceeded')
            require(member['path'] not in expected, 'Duplicate report physical member')
            _verify_file(root, member['path'], member)
            expected.add(member['path']); used += member['bytes']
        require(used <= policy['volume_payload_bytes'] and used == record.get('payload_bytes') == part.get('payload_bytes'),
                'Report part payload cap/count mismatch')
        require(len(members) == part.get('member_count'), 'Report part member count mismatch')
        from_parts.extend(members)
    require(from_parts == entries, 'Report root/part logical coverage mismatch')
    total = sum(e['bytes'] for e in entries)
    require(len(entries) == manifest.get('logical_member_count') and total == manifest.get('logical_payload_bytes') ==
            manifest.get('stored_payload_bytes') and total <= policy['evidence_payload_bytes'],
            'Report collection payload count/cap mismatch')
    presentation = manifest.get('presentation')
    require(isinstance(presentation, dict) and {'index.html', 'display.csv', 'REPORT_MANIFEST.json'} <= set(presentation),
            'Missing report presentation closure')
    for relative, identity in presentation.items():
        require(relative not in {e['path'] for e in entries} and relative != 'REPORT_COLLECTION_MANIFEST.json',
                'Report evidence/presentation overlap')
        _verify_file(root, relative, identity); expected.add(relative)
    require(sum(v['bytes'] for v in presentation.values()) == manifest.get('presentation_bytes_excluding_root_manifest'),
            'Report presentation byte count mismatch')
    actual = _all_files(root)
    require(set(actual) == expected, 'Unmanifested/missing report collection files')
    presentation_bytes = sum(p.stat().st_size for name, p in actual.items() if name not in {e['path'] for e in entries})
    require(presentation_bytes <= policy['presentation_bytes'], 'Report presentation byte cap exceeded')
    require(total + presentation_bytes <= policy['collection_bytes'], 'Report collection total byte cap exceeded')
    rendered = read(root / 'REPORT_MANIFEST.json')
    require(rendered.get('portable_evidence') == {e['evidence_id']: {'path': e['path'], 'sha256': e['sha256'], 'bytes': e['bytes']} for e in entries},
            'Report presentation/raw mapping mismatch')
    return {'status': 'REPORT_COLLECTION_VERIFIED', 'packaging': policy,
            'manifest_sha256': sha(manifest_path), 'logical_inventory_identity': manifest['logical_inventory_identity'],
            'logical_member_count': len(entries), 'logical_payload_bytes': total, 'stored_payload_bytes': total,
            'volume_count': len(parts), 'volume_payload_bytes': [p['payload_bytes'] for p in parts],
            'presentation_bytes': presentation_bytes, 'collection_bytes': total + presentation_bytes,
            'excluded_member_count': len(manifest['excluded_evidence'])}
