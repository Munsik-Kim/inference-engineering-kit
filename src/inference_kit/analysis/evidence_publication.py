"""Publish the verified immutable Suite subset outside mutable derived outputs.

This directory-only adapter never broadens the bounded evidence reader.  A
publication preserves the original manifest and member bytes; its path is a
locator, not a new acquisition.  Receipts belong outside the published root.
"""
import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from ..core.evidence import Limits, inspect_directory, member_name
from ..core.io import digest, fields, read, require, sha


KIND = 'verified_suite_raw_publication_v1'
MANIFEST = 'RUN_MANIFEST.json'


def _plain_path(path):
    path = Path(path).expanduser().absolute()
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'Symlink in publication path')
    return path


def _fingerprint(path):
    value = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(value.st_mode), 'Evidence member is not a regular file')
    return tuple(getattr(value, key) for key in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def _member(root, name):
    require(member_name(name) == name, 'Non-normalized raw member path')
    require(name != MANIFEST, 'Raw manifest cannot declare itself')
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), 'Evidence path escapes source')
    for current in (path, *path.parents):
        require(not current.is_symlink(), 'Symlink in declared raw evidence')
        if current == root:
            break
    require(path.is_file(), 'MISSING_EVIDENCE: ' + name)
    _fingerprint(path)
    return path


def _inventory(root, expected_sha256, limits):
    path = root / MANIFEST
    require(path.is_file() and not path.is_symlink(), 'MISSING_EVIDENCE: ' + MANIFEST)
    manifest_stat = _fingerprint(path)
    require(manifest_stat[2] <= limits.file_bytes, 'Evidence file limit')
    manifest_sha = sha(path)
    if expected_sha256 is not None:
        require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256),
                'Invalid expected raw manifest SHA256')
        require(manifest_sha == expected_sha256, 'Raw manifest identity changed')
    manifest = read(path)
    fields(manifest, ('schema_version', 'scope', 'files'), ('kind', 'immutable', 'derived_slice_kind'))
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1,
            'Unsupported Suite raw manifest schema')
    if 'kind' in manifest:
        require(manifest['kind'] == 'suite_e3_raw_v1' and manifest.get('immutable') is True,
                'Unsupported or mutable Suite raw manifest')
    else:
        require(manifest['scope'] in ('suite_execution_raw_v1', 'suite_e3_task_slice_v1'),
                'Unsupported Suite raw scope')
    require('derived_slice_kind' not in manifest,
            'SLICE_ONLY: explicit task/token slices cannot publish parent acquisition evidence')
    entries = manifest['files']
    require(isinstance(entries, dict) and entries, 'Empty Suite raw manifest')
    require(len(entries) + 1 <= limits.members, 'Evidence member limit')
    names = set(); fingerprints = {}; total = manifest_stat[2]
    for name, entry in entries.items():
        normalized = member_name(name)
        require(normalized == name, 'Non-normalized raw member path')
        require(normalized.casefold() not in names, 'Duplicate normalized raw member')
        names.add(normalized.casefold())
        fields(entry, ('bytes', 'sha256'))
        require(type(entry['bytes']) is int and 0 <= entry['bytes'] <= limits.file_bytes,
                'Evidence file limit or invalid byte count')
        require(isinstance(entry['sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['sha256']),
                'Malformed raw member SHA256')
        item = _member(root, name); fingerprints[name] = _fingerprint(item)
        require(fingerprints[name][2] == entry['bytes'] and sha(item) == entry['sha256'],
                'Raw evidence content changed: ' + name)
        total += entry['bytes']
    require(total <= limits.total_bytes, 'Evidence size limit')
    if manifest.get('kind') == 'suite_e3_raw_v1':
        require({'EXECUTION_PLAN.json', 'budget-continuation.json'} <= entries.keys(),
                'Suite raw manifest lacks execution/ledger dependencies')
    # A manifest inside the declared subset cannot point to omitted raw.  Do not
    # rewrite its relative locators or silently accept a partial parent manifest.
    for name in entries:
        if Path(name).name != 'RAW_MANIFEST.json':
            continue
        child = read(root / name); nested = child.get('files')
        require(isinstance(nested, dict) and nested, 'Empty nested raw manifest')
        for relative, entry in nested.items():
            require(member_name(relative) == relative, 'Non-normalized nested raw member path')
            joined = (Path(name).parent / relative).as_posix()
            require(joined in entries and entries[joined] == entry,
                    'MISSING_EVIDENCE: nested raw dependency ' + joined)
    # This validates actual task/prepared identities, compiled operation input
    # identities, and raw/journal membership; a self-consistent arbitrary JSON
    # inventory alone is insufficient to authorize publication.
    from .suite_results import verify_run
    _, _, _, compiled, _ = verify_run(root)
    if manifest.get('kind') == 'suite_e3_raw_v1':
        from ..suites.suite_execution import _frozen
        frozen = _frozen(root, current_source=False)
        require(frozen['compiled_plan_identity'] == compiled['plan_identity'],
                'Execution and Suite frozen plan identity differ')
        require(all(name in entries and entries[name] == entry
                    for name, entry in frozen['immutable_inputs'].items()),
                'MISSING_EVIDENCE: frozen execution input dependency')
    require(_fingerprint(path) == manifest_stat and sha(path) == manifest_sha,
            'Raw manifest changed during validation')
    fingerprints[MANIFEST] = manifest_stat
    inventory = dict(entries)
    inventory[MANIFEST] = {'bytes': manifest_stat[2], 'sha256': manifest_sha}
    return manifest_sha, inventory, fingerprints


def _copy_member(source, target, entry, fingerprint):
    """Bounded streaming copy with file identity checked before and after read."""
    require(_fingerprint(source) == fingerprint, 'Raw evidence changed before copy')
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        reader = os.fdopen(descriptor, 'rb')
        descriptor = None
        with reader, target.open('xb') as writer:
            value = os.fstat(reader.fileno())
            opened = tuple(getattr(value, key) for key in
                           ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
            require(opened == fingerprint, 'Raw evidence replaced before read')
            result = hashlib.sha256(); count = 0
            while True:
                block = reader.read(min(1024 ** 2, entry['bytes'] - count + 1))
                if not block:
                    break
                count += len(block)
                require(count <= entry['bytes'], 'Raw evidence grew during copy')
                writer.write(block); result.update(block)
            require(count == entry['bytes'] and result.hexdigest() == entry['sha256'],
                    'Raw evidence changed during copy')
            writer.flush(); os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
            require(tuple(getattr(after, key) for key in
                          ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')) == fingerprint,
                    'Raw evidence changed during read')
    finally:
        if descriptor is not None:
            os.close(descriptor)
    require(_fingerprint(source) == fingerprint, 'Raw evidence replaced during copy')


def _verify_published(path, manifest_sha, inventory, limits):
    require(path.is_dir() and not path.is_symlink(), 'SOURCE_ID_CONFLICT: publication path')
    actual = inspect_directory(path, limits)
    require({v['name'] for v in actual} == set(inventory),
            'SOURCE_ID_CONFLICT: publication contains different members')
    for name, entry in inventory.items():
        member = path / name
        require(member.stat().st_size == entry['bytes'] and sha(member) == entry['sha256'],
                'SOURCE_ID_CONFLICT: published raw content changed: ' + name)
    # Retain both bounded reader and frozen dependency validation after copying.
    _inventory(path, manifest_sha, limits)


@contextmanager
def _publication_lock(parent, key):
    """Serialize publishers of one content key; never steal a stale lock."""
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / ('.' + key + '.publishing.lock')
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError('PUBLICATION_IN_PROGRESS: exclusive publication lock exists') from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def publish_suite_evidence(source_root, output_root, *, manifest_sha256=None, limits=Limits()):
    """Verify and atomically publish full Suite raw under ``output_root/SHA``.

    Return a receipt for the caller to persist outside evidence.  Existing exact
    publications are reused without changing bytes or timestamps.  No completion
    or acquisition is synthesized and no old source/version resume is attempted.
    """
    defaults = Limits()
    require(isinstance(limits, Limits) and all(
        type(getattr(limits, name)) is int and 0 < getattr(limits, name) <= getattr(defaults, name)
        for name in ('members', 'file_bytes', 'total_bytes', 'compression_ratio')),
        'Publication limits must be positive integer bounds no greater than defaults')
    source = _plain_path(source_root); parent = _plain_path(output_root)
    require(source.is_dir(), 'Explicit Suite source directory required')
    require(not parent.resolve().is_relative_to(source.resolve()),
            'Publication output must be outside source evidence')
    manifest_sha, inventory, fingerprints = _inventory(source, manifest_sha256, limits)
    target = parent / manifest_sha
    require(not source.resolve().is_relative_to(target.resolve()),
            'Publication cannot contain its source')
    with _publication_lock(parent, manifest_sha):
        if target.exists() or target.is_symlink():
            _verify_published(target, manifest_sha, inventory, limits); reused = True
        else:
            temporary = Path(tempfile.mkdtemp(prefix='.' + manifest_sha + '-', suffix='.publishing', dir=parent))
            try:
                for name, entry in sorted(inventory.items()):
                    _copy_member(source / name, temporary / name, entry, fingerprints[name])
                # The entire source subset must remain stable, not only each file
                # at read time. Unmanifested derived outputs are ignored.
                for name, expected in fingerprints.items():
                    require(_fingerprint(source / name) == expected,
                            'Raw evidence changed during publication: ' + name)
                _inventory(source, manifest_sha, limits)
                _verify_published(temporary, manifest_sha, inventory, limits)
                require(not target.exists() and not target.is_symlink(),
                        'SOURCE_ID_CONFLICT: publication target appeared during copy')
                os.rename(temporary, target); reused = False
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    from .. import __version__
    from .tokens import source_identity
    return {'schema_version': 1, 'kind': KIND, 'status': 'RAW_HASH_VERIFIED',
            'publisher_package_version': __version__, 'publisher_source_digest': source_identity(),
            'source_root': str(source), 'evidence_root': str(target),
            'manifest_path': MANIFEST, 'manifest_sha256': manifest_sha,
            'canonical_source_id': digest({'adapter': 'suite_e3', 'raw_manifest': manifest_sha}),
            'reused': reused, 'members': len(inventory),
            'publication_scope': ('DECLARED_LEGACY_SLICE' if read(source / MANIFEST)['scope'] == 'suite_e3_task_slice_v1' else 'FULL_SUITE_RAW_MANIFEST'),
            'bytes': sum(v['bytes'] for v in inventory.values()),
            'files': inventory, 'excluded': 'All files absent from the original frozen raw manifest; derived output is not certified',
            'publication_mode': 'STREAM_COPY_ATOMIC_DIRECTORY_RENAME',
            'new_acquisitions': 0, 'new_GPU_actions': 0,
            'limits': {'members': limits.members, 'file_bytes': limits.file_bytes, 'total_bytes': limits.total_bytes}}
