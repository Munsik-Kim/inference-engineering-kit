"""Selected-position raw token metrics, natural logs; not corpus perplexity."""
import math
from ..core.io import require
from ..probes.contracts import MetricSpec

SPECS=[
 MetricSpec('target_nll','1',('target_logprob',),('sample','prefix','token_id_space','target_position'),'-log P(target|prefix)','nat','lower','mean over measured target positions'),
 MetricSpec('full_kl_ref_candidate','1',('full_logits',),('sample','prefix','token_id_space','vocabulary_size'),'sum p_ref*(log p_ref-log p_candidate)','nat','diagnostic_only','selected positions only'),
 MetricSpec('full_js','1',('full_logits',),('sample','prefix','token_id_space','vocabulary_size'),'.5 KL(p||m)+.5 KL(q||m), m=(p+q)/2','nat','diagnostic_only','selected positions only')]


def log_softmax(logits):
    require(logits and all(math.isfinite(v) for v in logits),'Invalid full logits')
    maximum=max(logits);normalizer=maximum+math.log(math.fsum(math.exp(v-maximum) for v in logits))
    return [v-normalizer for v in logits]


def nll(logprob):
    require(type(logprob) in (float,int) and math.isfinite(logprob) and logprob<=0,'Invalid target logprob')
    return -logprob


def distribution_metrics(left,right,full_vocabulary=True):
    require(full_vocabulary is True and len(left)==len(right) and len(left)>1,'Need same complete vocabulary, not top-k')
    a,b=log_softmax(left),log_softmax(right)
    p,q=[math.exp(v) for v in a],[math.exp(v) for v in b]
    log_mix=[max(u,v)+(math.log1p(math.exp(-abs(u-v)))-math.log(2.)) for u,v in zip(a,b)]
    kl=math.fsum(x*(u-v) for x,u,v in zip(p,a,b))
    js=.5*math.fsum(x*(u-m) for x,u,m in zip(p,a,log_mix) if x>0)+.5*math.fsum(y*(v-m) for y,v,m in zip(q,b,log_mix) if y>0)
    return {'kl_ref_candidate_nat':kl,'js_nat':js,'entropy_reference_nat':-math.fsum(x*u for x,u in zip(p,a)),
            'entropy_candidate_nat':-math.fsum(y*v for y,v in zip(q,b)), 'vocabulary_size':len(a),'log_base':'e','full_vocabulary':True}
