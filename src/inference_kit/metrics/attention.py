"""CPU metrics over matched full-key rows and saved attention outputs."""
import math
from ..core.io import require
from ..probes.contracts import MetricSpec

SPECS=[MetricSpec('attention_js','1',('attention_row',),('sample','prefix','layer','head','query','keys','mask','row_policy'),
 '.5 KL(a||m)+.5 KL(b||m); m=(a+b)/2','nat','diagnostic_only','measured full-key rows'),
 MetricSpec('attention_output_nmse','1',('attention_output',),('sample','prefix','layer','head','query'),
 'sum((candidate-reference)^2)/sum(reference^2)','ratio','diagnostic_only','measured vectors; near-zero denominator remains undefined')]


def row_metrics(a,b,allowed,tolerance=2e-6):
    require(len(a)==len(b)==len(allowed) and a,'Attention key/mask axis mismatch')
    require(any(allowed),'All-masked attention row: no normalized distribution')
    require(all(type(m) is bool for m in allowed),'Mask must be boolean allowed-key semantics')
    for row in (a,b):
        require(all(math.isfinite(x) and x>=0 for x in row),'Invalid probability')
        require(abs(math.fsum(row)-1)<=tolerance,'Full-key row does not sum to one')
        require(all(x==0 for x,m in zip(row,allowed) if not m),'Nonzero causal/padding mass')
    # Float32 row sums can round; keep saved values, report that rounding without renormalizing a key subset.
    mix=[(x+y)/2 for x,y in zip(a,b)]
    js=.5*math.fsum(x*math.log(x/m) for x,m in zip(a,mix) if x>0)+.5*math.fsum(y*math.log(y/m) for y,m in zip(b,mix) if y>0)
    entropy=lambda r:-math.fsum(x*math.log(x) for x in r if x>0)
    return {'l1':math.fsum(abs(x-y) for x,y in zip(a,b)),'js_nat':js,'entropy_reference_nat':entropy(a),
            'entropy_candidate_nat':entropy(b),'row_sum_reference':math.fsum(a),'row_sum_candidate':math.fsum(b),'valid_keys':sum(allowed)}


def vector_metrics(a,b,min_energy=1e-12):
    require(len(a)==len(b) and a and all(math.isfinite(x) for x in a+b),'Output vector shape/nonfinite')
    ea,eb=math.fsum(x*x for x in a),math.fsum(y*y for y in b);sse=math.fsum((y-x)**2 for x,y in zip(a,b))
    nmse=sse/ea if ea>min_energy else None
    cosine=math.fsum(x*y for x,y in zip(a,b))/math.sqrt(ea*eb) if ea>min_energy and eb>min_energy else None
    return {'sse':sse,'reference_energy':ea,'candidate_energy':eb,'nmse':nmse,'cosine':cosine,
            'nmse_status':'MEASURED' if nmse is not None else 'NOT_APPLICABLE_NEAR_ZERO_REFERENCE',
            'cosine_status':'MEASURED' if cosine is not None else 'NOT_APPLICABLE_ZERO_OR_NEAR_ZERO_VECTOR','min_energy':min_energy}


def selected_row_oracle(q,keys,values,query_index,padding,scale):
    """Small stdlib oracle only; real collection uses native post-RoPE Q/K/V."""
    allowed=[i<=query_index and bool(padding[i]) for i in range(len(keys))]
    if not any(allowed):return {'status':'NOT_APPLICABLE','row':None,'output':None,'allowed':allowed}
    scores=[math.fsum(x*y for x,y in zip(q,k))*scale if ok else -math.inf for k,ok in zip(keys,allowed)]
    maximum=max(scores);weights=[math.exp(v-maximum) if ok else 0. for v,ok in zip(scores,allowed)];den=math.fsum(weights)
    row=[x/den for x in weights];output=[math.fsum(p*v[d] for p,v in zip(row,values)) for d in range(len(values[0]))]
    return {'status':'MEASURED','row':row,'output':output,'allowed':allowed}
