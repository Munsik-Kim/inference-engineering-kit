"""Bounded heuristic review; emits finding locations and rule names, never secret values.

Not a comprehensive security certification. Read ZIP/wheel/sdist contents as data only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile

RULES={
 'github_token':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
 'hf_token':r'hf_[A-Za-z0-9]{30,}',
 'openai_key':r'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}',
 'aws_access_key':r'(?:AKIA|ASIA)[A-Z0-9]{16}',
 'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 'bearer_literal':r'(?i)Authorization[\"\x27]?\s*[:=]\s*[\"\x27]Bearer\s+[A-Za-z0-9_.-]{20,}',
 'url_credentials':r'https?://[^\s/:@]+:[^\s/@]+@',
 'signed_url':r'(?i)[?&](?:X-Amz-Signature|X-Goog-Signature|access_token|api_key|signature)=[A-Za-z0-9%._-]{15,}',
}
FORBIDDEN={'.env','local.json','id_rsa','id_ed25519','credentials','cookies.txt'}
BAD_SUFFIX={'.safetensors','.pt','.pth','.ckpt','.pem','.p12','.pfx'}

def inspect(name,blob):
    p=Path(name);findings=[]
    if p.name in FORBIDDEN or p.suffix.lower() in BAD_SUFFIX or any(v in p.parts for v in ('site-packages','.venv','venv','.git')):
        findings.append({'file':name,'rule':'excluded_name_or_payload'})
    if len(blob)>64*1024**2:findings.append({'file':name,'rule':'large_file_requires_review'})
    try:text=blob.decode('utf-8')
    except UnicodeError:return findings
    for rule,pattern in RULES.items():
        for match in re.finditer(pattern,text):findings.append({'file':name,'rule':rule,'line':text.count('\n',0,match.start())+1})
    return findings

def scan(path):
    path=Path(path);entries=[];findings=[]
    def one(name,blob):
        entries.append({'name':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()});findings.extend(inspect(name,blob))
    if path.is_dir():
        for p in sorted(path.rglob('*')):
            if any(v in p.relative_to(path).parts for v in ('.git','__pycache__','build','dist')) or '.egg-info' in str(p):continue
            if p.is_symlink():findings.append({'file':str(p.relative_to(path)),'rule':'symlink'});continue
            if p.is_file():one(str(p.relative_to(path)),p.read_bytes())
    elif zipfile.is_zipfile(path):
        from inference_kit.core.evidence import inspect_zip
        inspect_zip(path)
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir():one(info.filename,z.read(info))
    elif tarfile.is_tarfile(path):
        from inference_kit.core.evidence import member_name
        with tarfile.open(path) as tar:
            members=tar.getmembers();assert len(members)<10000 and sum(m.size for m in members)<512*1024**2
            for m in members:
                member_name(m.name);assert m.isdir() or m.isfile(),'Tar symlink/special file'
                if m.isfile():
                    assert m.size<64*1024**2;one(m.name,tar.extractfile(m).read())
    else:one(path.name,path.read_bytes())
    return {'status':'HEURISTIC_REVIEW_CLEAR' if not findings else 'REVIEW_REQUIRED','files_reviewed':len(entries),
            'files':entries,'findings':findings,'scanner':'extension/path, bounded archive inventory, token/key/header/query patterns; no gitleaks installed',
            'limits':'No completeness/security guarantee. Public dataset/token IDs are not anonymous. Historical paths remain metadata.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--output',required=True);a=p.parse_args()
    result=scan(a.path);out=Path(a.output)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files',)}));raise SystemExit(bool(result['findings']))
