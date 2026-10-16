"""Bounded little-endian float32 blobs. No pickle, SDK or archive execution."""
from array import array
import math
from pathlib import Path
import sys
from ..core.io import require, sha, safe_name

MAX_BYTES=64*1024**2

def size(shape):
    require(isinstance(shape,list) and shape and all(type(d) is int and d>0 for d in shape),'Invalid blob shape')
    n=math.prod(shape);require(n*4<=MAX_BYTES,'Numeric blob byte cap');return n


def save(root,name,values,shape):
    require(safe_name(name)==name and '/' not in name,'Unsafe numeric name')
    require(len(values)==size(shape) and all(math.isfinite(x) for x in values),'Invalid/nonfinite numeric values')
    a=array('f',values);require(all(math.isfinite(x) for x in a),'Float32 cast overflow')
    if sys.byteorder!='little':a.byteswap()
    p=Path(root)/'blobs'/(name+'.f32');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:a.tofile(f)
    return {'path':str(p.relative_to(root)),'dtype':'float32','shape':shape,'endianness':'little','bytes':len(a)*4,'sha256':sha(p)}


def load(root,metadata):
    require(set(metadata)=={'path','dtype','shape','endianness','bytes','sha256'},'Unknown blob metadata')
    require(metadata['dtype']=='float32' and metadata['endianness']=='little','Unsupported numeric representation')
    n=size(metadata['shape']);require(metadata['bytes']==n*4,'Numeric shape/byte mismatch')
    name=safe_name(metadata['path']);root=Path(root);p=root/name
    require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) and p.is_file(),'Missing/unsafe blob')
    require(p.stat().st_size==metadata['bytes'] and sha(p)==metadata['sha256'],'Blob size/hash mismatch')
    a=array('f')
    with p.open('rb') as f:a.fromfile(f,n)
    if sys.byteorder!='little':a.byteswap()
    require(all(math.isfinite(v) for v in a),'NaN/inf numeric blob')
    return list(a)
