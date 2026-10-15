"""Review/export existing evidence; excludes private local.json without editing original.

Use only for explicitly reviewed local archives. No archive code is executed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from inference_kit.core.evidence import inspect_zip
from review_files import scan

def export(source,destination,receipt_path):
    source,destination=Path(source),Path(destination)
    if destination.exists() or Path(receipt_path).exists():raise ValueError('Refuse export overwrite')
    inspect_zip(source);before=scan(source)
    excluded=[f['file'] for f in before['findings'] if f['rule']=='excluded_name_or_payload' and Path(f['file']).name=='local.json']
    unreviewed=[f for f in before['findings'] if f['file'] not in excluded]
    if unreviewed:raise ValueError('Unreviewed findings; see source scan receipt')
    original_sha=hashlib.sha256(source.read_bytes()).hexdigest();removed={};payload={}
    with zipfile.ZipFile(source) as z:
        for entry in z.infolist():
            if entry.is_dir():continue
            blob=z.read(entry)
            if entry.filename in excluded:
                removed[entry.filename]={'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'reason':'Personal local runtime configuration excluded'}
            else:payload[entry.filename]=blob
    if removed:
        original_manifest=payload.pop('BUNDLE_MANIFEST.json')
        payload['provenance/original-BUNDLE_MANIFEST.json']=original_manifest
        redaction={'schema_version':1,'derived_export':True,'original_archive_sha256':original_sha,
                   'removed':removed,'modified_fields':[], 'numeric_raw_changed':False,
                   'historical_source_hashes':'Original identities unchanged; snapshot omits named personal configurations',
                   'original_manifest':'provenance/original-BUNDLE_MANIFEST.json'}
        payload['EXPORT_RECEIPT.json']=(json.dumps(redaction,indent=2)+'\n').encode()
        manifest={'schema_version':3,'files':{n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in sorted(payload.items())},
                  'derived_from_archive_sha256':original_sha,'original_manifest_preserved':True}
        payload['BUNDLE_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
        destination.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for n,b in sorted(payload.items()):z.writestr(n,b)
    else:
        import shutil
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    after=scan(destination)
    if after['findings']:raise ValueError('Export review findings')
    receipt={'asset_name':destination.name,'bytes':destination.stat().st_size,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
             'original_archive_sha256':original_sha,'derived':bool(removed),'excluded_files':removed,
             'original_source_modified':False,'numerical_raw_changed':False,'scan':after,
             'archive_identity':'SANITIZED_EXPORT' if removed else 'BYTE_IDENTICAL_ORIGINAL',
             'historical_gpu_relabelled':False}
    with Path(receipt_path).open('x') as f:json.dump(receipt,f,indent=2)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    r=export(a.source,a.output,a.receipt);print(json.dumps({k:v for k,v in r.items() if k!='scan'}))
