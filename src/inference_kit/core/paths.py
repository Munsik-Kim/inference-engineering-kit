"""Explicit project/artifact/data/run bindings, independent of cwd and source layout."""
from dataclasses import dataclass
import os
from pathlib import Path
import re
from .io import read,require

ROOT_KEYS=('PROJECT_ROOT','ARTIFACT_ROOT','DATA_ROOT','RUNS_ROOT')
SECRET=re.compile(r'^(authorization|password|secret|api[_-]?key|access[_-]?token|hf_token|github_token)$',re.I)


def masked(value):
    if isinstance(value,dict):return {k:('<MASKED>' if SECRET.fullmatch(k) else masked(v)) for k,v in value.items()}
    if isinstance(value,list):return [masked(v) for v in value]
    return value


@dataclass(frozen=True)
class Paths:
    roots:dict
    settings_base:Path
    @classmethod
    def load(cls,config_file,local_file=None):
        base=Path(config_file).expanduser().absolute().parent
        local=read(Path(local_file)) if local_file else {}
        binding_base=Path(local_file).expanduser().absolute().parent if local_file else base
        supplied=local.get('roots',{});require(set(supplied)<=set(ROOT_KEYS),'Unknown path root')
        roots={}
        for key in ROOT_KEYS:
            raw=supplied.get(key,'.' if key=='PROJECT_ROOT' else key.removesuffix('_ROOT').lower())
            p=Path(raw).expanduser();roots[key]=p if p.is_absolute() else binding_base/p
        return cls(roots,base)
    def resolve(self,value,base=None):
        require(isinstance(value,str) and value,'Missing path')
        expanded=value
        for key,p in self.roots.items():expanded=expanded.replace('${'+key+'}',str(p))
        require('${' not in expanded and not re.search(r'^[a-z]+://',expanded),'Unknown path binding or remote path')
        p=Path(expanded).expanduser()
        # Preserve a runtime venv symlink spelling. Identity is a separate operation.
        return p.absolute() if p.is_absolute() else ((base or self.settings_base)/p).absolute()
