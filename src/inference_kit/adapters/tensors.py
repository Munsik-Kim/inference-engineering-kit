"""Exact module-bundle intervention for the validated symmetric pack-quantized format.

No tensor library is imported until compose(). This adapter does not quantize or fit.
"""
import contextlib
from pathlib import Path
import shutil
from ..core.io import read,write,require,digest,sha
from ..core.state import new_output

BUNDLE=('weight_packed','weight_scale','weight_shape')


def module_for_key(key,modules):
    matches=[m for m in modules if key.startswith(m+'.')]
    require(len(matches)<=1,'Duplicate/overlapping module prefix')
    return matches[0] if matches else None


def choose_keys(keys,modules,known_modules):
    require(len(modules)==len(set(modules)),'Duplicate intervention module')
    require(set(modules)<=set(known_modules),'Missing exact intervention module')
    selected={key for key in keys if module_for_key(key,modules) and key.rsplit('.',1)[-1] in BUNDLE}
    require(len(selected)==len(modules)*len(BUNDLE),'Missing complete tensor bundle')
    return selected


def check_scheme(config):
    q=config.get('quantization_config',{});groups=q.get('config_groups',{})
    require(q.get('quant_method')=='compressed-tensors' and q.get('format')=='pack-quantized' and len(groups)==1,'Unsupported compressed-tensors format')
    group=next(iter(groups.values()));w=group['weights']
    require(group['input_activations'] is None and group['output_activations'] is None,'Unsupported activation quantization')
    require(w['num_bits']==4 and w['type']=='int' and w['symmetric'] is True and w['strategy']=='group'
            and w['group_size']==128 and w.get('actorder') is None,'Unsupported group128 symmetric INT4 scheme')
    return group


class Reader:
    def __init__(self,path,stack,safe_open):
        path=Path(path);self.handles={};self.shards={}
        for p in sorted(path.glob('*.safetensors')):
            f=stack.enter_context(safe_open(p,framework='pt',device='cpu'))
            for key in f.keys():
                require(key not in self.handles,'Duplicate shard key');self.handles[key]=f;self.shards[key]=p.name
        require(self.handles,'Missing safetensors payload')
        index=path/'model.safetensors.index.json'
        if index.exists():require(read(index)['weight_map']==self.shards,'Shard index/key mismatch')
    def get(self,key):return self.handles[key].get_tensor(key)


def compose(base,intervention,modules,output,base_receipt,intervention_receipt):
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor
        from compressed_tensors.quantization import QuantizationScheme
    except ImportError as exc:
        raise RuntimeError('Tensor adapter requires an explicitly prepared torch/safetensors/compressed-tensors0.18 runtime; CPU analysis remains available') from exc
    base,intervention=Path(base),Path(intervention)
    left,right=read(Path(base_receipt)),read(Path(intervention_receipt))
    require(all(left.get(k) and left[k]==right.get(k) for k in ('model_id','model_revision')),'Model/revision identity mismatch')
    a,b=read(base/'config.json'),read(intervention/'config.json')
    group=check_scheme(a);require(a==b and group==check_scheme(b),'Architecture/scheme metadata mismatch')
    scheme=QuantizationScheme(**group)
    require(set(PackedQuantizationCompressor.compression_param_names(scheme))==set(BUNDLE),'Unreviewed tensor metadata suffixes')
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        require((base/name).exists()==(intervention/name).exists(),'Tokenizer file set differs')
        if (base/name).exists():require(sha(base/name)==sha(intervention/name),'Tokenizer identity mismatch')
    out=new_output(output);before={str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for folder in (base,intervention) for p in folder.iterdir() if p.is_file()}
    provenance={}
    with contextlib.ExitStack() as stack:
        l,r=Reader(base,stack,safe_open),Reader(intervention,stack,safe_open)
        require(l.handles.keys()==r.handles.keys(),'Tensor key sets differ')
        known={k.removesuffix('.weight_packed') for k in l.handles if k.endswith('.weight_packed')}
        selected=choose_keys(l.handles,modules,known)
        quantized={m+'.'+suffix for m in known for suffix in BUNDLE}
        require(quantized<=l.handles.keys(),'Missing quantization bundle')
        for m in known:
            for source in (l,r):
                shape=source.get(m+'.weight_shape');packed=source.get(m+'.weight_packed');scale=source.get(m+'.weight_scale')
                dims=shape.tolist()
                require(shape.dtype==torch.int64 and len(dims)==2 and dims[0]>0 and dims[1]>0 and dims[1]%128==0,'Invalid weight_shape')
                require(packed.dtype==torch.int32 and list(packed.shape)==[dims[0],dims[1]//8],'Packed shape/dtype mismatch')
                require(scale.dtype==torch.bfloat16 and list(scale.shape)==[dims[0],dims[1]//128]
                        and bool(torch.isfinite(scale).all()) and bool((scale>0).all()),'Scale metadata shape/dtype/value mismatch')
            require(torch.equal(l.get(m+'.weight_shape'),r.get(m+'.weight_shape')),'Original weight shape differs')
            related={k for k in l.handles if k.startswith(m+'.')}
            require(related<={m+'.'+suffix for suffix in BUNDLE}|{m+'.bias'},'Unexpected module metadata/dense fallback')
        tensors={}
        for key in sorted(l.handles):
            x,y=l.get(key),r.get(key)
            require(x.shape==y.shape and x.dtype==y.dtype,'Tensor shape/dtype differs')
            if key not in quantized:require(torch.equal(x,y),'Native tensor differs outside intervention')
            chosen=r if key in selected else l;tensors[key]=chosen.get(key).clone().contiguous()
            provenance[key]={'source':'intervention' if key in selected else 'base','shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.numel()*x.element_size()}
        save_file(tensors,str(out/'model.safetensors'),metadata={'format':'pt'});del tensors
        saved=Reader(out,stack,safe_open)
        for key in l.handles:require(torch.equal(saved.get(key),(r if key in selected else l).get(key)),'Saved tensor provenance differs')
    for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        if (base/name).exists():shutil.copyfile(base/name,out/name)
    require(before=={p:{'bytes':Path(p).stat().st_size,'mtime_ns':Path(p).stat().st_mtime_ns} for p in before},'Source file modified')
    result={'schema_version':1,'status':'PACKED_COMPOSITION_CPU_TENSORWISE_VERIFIED','model_id':left['model_id'],'model_revision':left['model_revision'],
            'intervention_modules':list(modules),'tensor_provenance':provenance,'source_receipts':{'base':sha(Path(base_receipt)),'intervention':sha(Path(intervention_receipt))},
            'packed_file_bytes':(out/'model.safetensors').stat().st_size,'packed_file_sha256':sha(out/'model.safetensors'),
            'writable_hardlinks':False,'quantization_refit':False,'gpu_loader_verification':'NOT_RUN',
            'support':'Validated format adapter; tiny CPU fixture verification does not certify new GPU artifacts'}
    write(out/'composition.json',result);return result
