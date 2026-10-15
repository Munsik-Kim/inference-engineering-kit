"""New outputs, immutable raw records, atomic mutable state and linked accounting."""
import datetime
from contextlib import contextmanager
import os
from pathlib import Path
import uuid
from .io import read, write, require, sha, digest


def new_output(path):
    path=Path(path).expanduser().absolute()
    require(not path.exists() and not path.is_symlink(),'Refuse overwrite existing output')
    path.mkdir(parents=True,exist_ok=False)
    return path


def derived_output(bundle,output):
    source=Path(bundle).expanduser().absolute()
    target=Path(output).expanduser().absolute()
    require(not (source.is_dir() and target.resolve().is_relative_to(source.resolve())), 'Derived output must be outside immutable evidence')
    return new_output(target)


@contextmanager
def preserve_failure(output):
    try:yield
    except BaseException as exc:
        path=Path(output)/'failure.json'
        if not path.exists():write(path,{'status':'FAILED_OR_INVALID_DERIVATION','error_type':type(exc).__name__,'error':str(exc),'new_gpu_runs':0})
        raise


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        write(temporary,value);os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


class Attempt:
    def __init__(self,output,freeze,parent=None):
        self.path=new_output(output);self.identity=digest(freeze)
        if parent:
            previous=read(Path(parent)/'attempt.json')
            require(previous['identity']==self.identity,'Resume config/source identity differs')
        self.record={'schema_version':1,'attempt_id':uuid.uuid4().hex,'identity':self.identity,
                     'parent':str(parent) if parent else None,'status':'RUNNING','stages':[],
                     'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(self.path/'freeze.json',freeze);atomic(self.path/'attempt.json',self.record)
    def stage(self,name,status,**details):
        self.record['stages'].append({'name':name,'status':status,**details});atomic(self.path/'attempt.json',self.record)
    def finish(self,status):
        self.record['status']=status;atomic(self.path/'attempt.json',self.record)


def cumulative(record,resolver,seen=None):
    """Resolver maps an explicit historical ledger reference to a local evidence file."""
    require(record.get('running') is None,'Unresolved preceding GPU job')
    seen=set() if seen is None else seen
    previous=0;link=record.get('inherited_from')
    if link:
        key=link['path'];require(key not in seen,'Ledger cycle');seen.add(key)
        p=Path(resolver(key));require(sha(p)==link['sha256'],'Prior ledger identity changed')
        previous=cumulative(read(p),resolver,seen)
        require(abs(previous-link['used_seconds'])<1e-6,'Inherited GPU time differs')
    jobs=record.get('jobs',[]);require(all(j['wall_seconds']>=0 for j in jobs),'Negative accounting')
    used=previous+sum(j['wall_seconds'] for j in jobs)
    require(abs(used-record['used_seconds'])<1e-6,'Budget double counted or missing')
    return used


class Budget:
    def __init__(self,path):self.path=Path(path)
    def begin(self,identifier,cap_seconds):
        r=read(self.path)
        require(r['running'] is None,'Unresolved job')
        require(not any(j['id']==identifier for j in r['jobs']),'Duplicate attempt ID')
        remaining=r['cap_seconds']-r['used_seconds']
        require(remaining>0 and 0<cap_seconds<=remaining,'Budget exhausted')
        r['running']={'id':identifier,'cap_seconds':cap_seconds};atomic(self.path,r)
    def end(self,seconds,**details):
        r=read(self.path);require(r['running'] is not None and seconds>=0,'No running job/negative time')
        r['jobs'].append(r['running']|{'wall_seconds':seconds}|details)
        r['used_seconds']+=seconds;r['running']=None;atomic(self.path,r)
