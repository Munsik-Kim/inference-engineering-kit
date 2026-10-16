"""Selected native Q/K/V observer. No full attention matrix is formed here."""
from ..core.io import require


def selected_torch_row(module,q,k,v,mask,head,query_index,scaling,causal=None):
    import torch
    require(q.ndim==k.ndim==v.ndim==4 and q.shape[0]==k.shape[0]==1,'Observer requires batch1 B,H,T,D')
    require(q.shape[2]==k.shape[2] and q.shape[1]%k.shape[1]==0,'Prefill length/head mapping mismatch')
    kv=head//(q.shape[1]//k.shape[1]);length=k.shape[2]
    scores=(q[0,head,query_index].float()@k[0,kv].float().T)*scaling
    allowed=torch.ones(length,dtype=torch.bool,device=q.device)
    if mask is not None:
        require(mask.ndim==4 and mask.shape[-2:]==(q.shape[2],length),'Unreviewed native attention mask axes')
        row_mask=mask[0,head if mask.shape[1]>1 else 0,query_index,:length]
        if row_mask.dtype==torch.bool:allowed &= row_mask
        else:
            require(row_mask.is_floating_point(),'Invalid native additive mask')
            allowed &= torch.isfinite(row_mask)&(row_mask>torch.finfo(row_mask.dtype).min/2)
            scores=scores+row_mask.float()
    else:
        causal=getattr(module,'is_causal',True) if causal is None else causal
        if causal:allowed &= torch.arange(length,device=q.device)<=query_index
    require(not bool(allowed[query_index+1:].any()),'Native path exposes future reference tokens')
    if not bool(allowed.any()):return None,None,allowed,kv
    scores=scores.masked_fill(~allowed,-torch.inf)
    row=torch.softmax(scores,dim=-1,dtype=torch.float32)
    reconstructed=row@v[0,kv].float()
    return row,reconstructed,allowed,kv


class AttentionObserver:
    """Temporary interface override calls the original SDPA and returns it unchanged.

    Only the selected module is collected. Mapping and projection hook are restored
    even on callback/model exceptions. There is no library/site-packages mutation.
    """
    def __init__(self,interface,module,selection,consume):
        self.interface,self.module,self.selection,self.consume=interface,module,selection,consume
        self.restored=False;self.calls=0;self.no_inplace_changes=True;self.hook=None
    def __enter__(self):
        self.previous=self.interface['sdpa']
        local=getattr(self.interface,'_local_mapping',None)
        self.had_local=True if local is None else 'sdpa' in local
        def observed(module,q,k,v,mask,**kwargs):
            original=(q,k,v,mask);versions=[getattr(x,'_version',None) for x in original]
            result=self.previous(module,q,k,v,mask,**kwargs)
            if module is self.module:
                self.calls+=1;native=result[0]
                require(tuple(native.shape)==(1,q.shape[2],q.shape[1],q.shape[3]),'Installed native SDPA output axes differ')
                for head in self.selection['heads']:
                    for query_index in self.selection['queries']:
                        row,reconstructed,allowed,kv=selected_torch_row(module,q,k,v,mask,head,query_index,kwargs['scaling'],kwargs.get('is_causal'))
                        coord={'layer':self.selection['layer'],'head':head,'kv_head':kv,'query_position':query_index,'key_positions':list(range(k.shape[2])),'allowed_keys':allowed.cpu().tolist()}
                        details={'qk_point':'native_post_RoPE; Smol has no QK norm/softcap','scaling':kwargs['scaling'],
                                 'probability_source':'reconstructed_from_native_qk','normalization_dtype':'float32','native_qkv_dtype':str(q.dtype),
                                 'mask_source':'native_sdpa_mask' if mask is not None else 'native_sdpa_is_causal','key_scope':'all native keys, masked keys retained as zeros'}
                        if row is None:self.consume('attention_row',coord,None,details|{'status':'NOT_APPLICABLE'})
                        else:
                            self.consume('attention_row',coord,row,details)
                            self.consume('attention_output',coord,native[0,query_index,head],details|{'output_source':'native_returned_SDPA_AV','raw_dtype':str(native.dtype)})
                            self.consume('reconstructed_AV',coord,reconstructed,details|{'output_source':'reconstructed_AV_from_FP32_row_and_native_V'})
            require(versions==[getattr(x,'_version',None) for x in original],'Observer changed native input/mask in-place')
            return result
        self.interface['sdpa']=observed
        def projection(module,args,output):
            version=getattr(output,'_version',None)
            for query_index in self.selection['queries']:
                self.consume('output_projection',{'layer':self.selection['layer'],'query_position':query_index},output[0,query_index],
                             {'output_source':'native_o_proj_forward_hook','raw_dtype':str(output.dtype)})
            require(version==getattr(output,'_version',None),'Observer modified projection output')
        try:self.hook=self.module.o_proj.register_forward_hook(projection)
        except BaseException:
            self.__exit__(None,None,None);raise
        return self
    def __exit__(self,*args):
        if self.hook is not None:self.hook.remove()
        if self.had_local:self.interface['sdpa']=self.previous
        else:del self.interface['sdpa']
        self.restored=self.interface['sdpa'] is self.previous
        require(self.restored,'Attention interface restoration failed')
