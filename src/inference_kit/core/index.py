"""Verify an index against a locally bound bundle, never an old absolute home path."""
from pathlib import Path
from .io import read,require,sha,digest
from .evidence import member_name


def check_index(index,bundle_root,index_root):
    require(index.get('kind')=='verified_result_index' and index.get('schema_version') in (2,3),'Unsupported/unverified result index')
    require(index.get('results_identity')==digest(index['results']),'Result index summary identity differs')
    require(index.get('evidence_manifest'),'Missing evidence manifest')
    root=Path(bundle_root);index_root=Path(index_root)
    # Old evidence_root identifies scope, never an accessible default path.
    historical=Path(index.get('historical_evidence_root',index['evidence_root']));base=index_root if historical.name==index_root.name else index_root.parent
    for name,identity in index['evidence_manifest'].items():
        name=member_name(name);p=base/name
        require(p.resolve().is_relative_to(root.resolve()) and p.is_file(),'Missing/unsafe bound index reference '+name)
        require(p.stat().st_size==identity['bytes'] and sha(p)==identity['sha256'],'Bound evidence identity differs '+name)
    return index
