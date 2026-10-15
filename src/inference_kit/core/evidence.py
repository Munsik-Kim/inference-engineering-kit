"""Bounded data-only archive reader. Archive Python is never imported or executed."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile
from .io import read,require,sha,write


@dataclass(frozen=True)
class Limits:
    members:int=10000
    total_bytes:int=512*1024**2
    file_bytes:int=64*1024**2
    compression_ratio:int=2000


def member_name(name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name and '\x00' not in name,'Unsafe archive name')
    p=PurePosixPath(name)
    require(not p.is_absolute() and str(p)!='.' and '..' not in p.parts and str(p)==name.rstrip('/'),'Archive path traversal/absolute path')
    return str(p)


def inspect_zip(path,limits=Limits()):
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();require(len(entries)<=limits.members,'Archive member limit')
        require(sum(e.file_size for e in entries)<=limits.total_bytes,'Archive size limit')
        names=set();out=[]
        for e in entries:
            name=member_name(e.filename);key=name.casefold()
            require(key not in names,'Duplicate archive member');names.add(key)
            mode=e.external_attr>>16
            require(not stat.S_ISLNK(mode) and not (mode and stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)),'Archive symlink/special file')
            require(not e.flag_bits&1,'Encrypted archive unsupported')
            require(e.file_size<=limits.file_bytes,'Archive individual file limit')
            require(e.file_size<=max(1,e.compress_size)*limits.compression_ratio,'Archive compression limit')
            out.append({'name':name,'bytes':e.file_size,'directory':e.is_dir()})
        return out


def inspect_directory(path,limits=Limits()):
    path=Path(path);out=[]
    for p in sorted(path.rglob('*')):
        require(not p.is_symlink(),'Symlink in evidence directory')
        if p.is_file():
            require(p.stat().st_size<=limits.file_bytes,'Evidence file limit')
            out.append({'name':member_name(str(p.relative_to(path))),'bytes':p.stat().st_size,'directory':False})
    require(len(out)<=limits.members and sum(e['bytes'] for e in out)<=limits.total_bytes,'Evidence limits exceeded')
    return out


@contextmanager
def open_bundle(bundle,limits=Limits(),expected_sha256=None):
    bundle=Path(bundle).expanduser().absolute()
    require(bundle.exists() and not bundle.is_symlink(),'Missing/symlink evidence bundle')
    if bundle.is_dir():
        require(expected_sha256 is None,'Archive SHA256 cannot authenticate a directory; supply a ZIP or validate its manifest')
        inspect_directory(bundle,limits);yield bundle;return
    if expected_sha256:require(sha(bundle)==expected_sha256,'Archive byte identity mismatch')
    entries=inspect_zip(bundle,limits)
    with tempfile.TemporaryDirectory(prefix='inference-kit-evidence-') as temp:
        root=Path(temp)
        with zipfile.ZipFile(bundle) as z:
            for entry in entries:
                p=root/entry['name'];require(p.resolve().is_relative_to(root),'Archive escapes extraction')
                if entry['directory']:p.mkdir(parents=True,exist_ok=True);continue
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(entry['name']) as src,p.open('xb') as dst:
                    remaining=entry['bytes']
                    while remaining:
                        block=src.read(min(1024**2,remaining));require(block,'Truncated archive member');dst.write(block);remaining-=len(block)
                    require(not src.read(1),'Archive declared size mismatch')
        yield root


def case_root(root,case):
    root=Path(root)
    marker={'step04':'PROTOCOL_FROZEN.json','step03':'PLAN_FROZEN.json','step02':'results-index.json'}[case]
    matches=[]
    for p in root.rglob(marker):
        rel=p.relative_to(root)
        if any(x in rel.parts for x in ('source','sources','source-snapshot','execution-source-v1','replay-source')):continue
        value=read(p)
        if case=='step02':
            if value.get('schema_version')==2 and value.get('kind')=='verified_result_index':matches.append(p.parent)
        elif value.get('task')==case or value.get('spec',{}).get('task')==case:matches.append(p.parent)
    require(matches,'Missing '+case+' evidence root')
    if case=='step02':
        repeats=[p for p in matches if p.name.startswith('step02_qwen_w4_repeat_')]
        if len(repeats)==1:return repeats[0]
        original=[p for p in matches if p.name.startswith('step02_')]
        require(len(original)==1,'Ambiguous STEP02 index; explicit bundle required');return original[0]
    require(len(matches)==1,'Ambiguous frozen case root');return matches[0]


def binding(root,historical):
    """Rebind an old path as metadata; never fall back to that absolute location."""
    root=Path(root);p=PurePosixPath(str(historical));parts=p.parts
    candidates=[]
    for i,part in enumerate(parts):
        if re.fullmatch(r'step0[234]_[A-Za-z0-9_]+',part):
            tail=Path(*parts[i:])
            candidates.extend([root/tail,root/'evidence'/tail])
            for d in root.glob('campaign-history/*'):
                if d.name==part:candidates.append(d/Path(*parts[i+1:]))
    found=[c for c in candidates if c.exists() and c.resolve().is_relative_to(root.resolve())]
    unique=list(dict.fromkeys(found));require(len(unique)==1,'Missing/ambiguous explicit historical binding')
    return unique[0]


def verify_manifest(root):
    root=Path(root);verified=[];manifests=[]
    for name in ('archive-manifest.json','BUNDLE_MANIFEST.json','MANIFEST.json','manifest.json'):
        p=root/name
        if not p.is_file():continue
        obj=read(p);entries=obj.get('files') if isinstance(obj,dict) else None
        if not isinstance(entries,dict):continue
        for relative,entry in entries.items():
            require(isinstance(entry,dict) and re.fullmatch(r'[0-9a-f]{64}',entry.get('sha256','')),'Malformed manifest identity')
            relative=member_name(relative);f=root/relative
            require(f.is_file() and f.resolve().is_relative_to(root.resolve()),'Manifest missing/unsafe reference')
            require(sha(f)==entry['sha256'],'Manifest content corrupted: '+relative)
            if 'bytes' in entry:require(f.stat().st_size==entry['bytes'],'Manifest byte size mismatch')
            verified.append(relative)
        manifests.append(name)
    return {'status':'MANIFEST_CONTENT_VERIFIED' if verified else 'INVENTORY_ONLY_NO_RECOGNIZED_MANIFEST',
            'verified_members':len(set(verified)),'manifests':manifests,'python_executed':False,
            'authenticity':'Hashes require a trusted expected archive SHA; self-consistent metadata alone is not authenticity'}


def export_identity(root):
    receipt=Path(root)/'EXPORT_RECEIPT.json'
    if not receipt.exists():return {'status':'NO_SANITIZATION_RECEIPT_IN_INPUT'}
    value=read(receipt)
    return {'status':'SANITIZED_EXPORT_DECLARED','original_archive_sha256':value['original_archive_sha256'],
            'removed_files':sorted(value['removed']),'numeric_raw_changed':value['numeric_raw_changed'],
            'limitation':'Receipt is covered by the export manifest; original archive byte equivalence is not claimed'}
