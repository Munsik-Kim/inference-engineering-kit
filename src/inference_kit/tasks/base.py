"""Small typed task boundary. Executors collect; adapters render and score."""
from dataclasses import dataclass
from typing import Protocol
from ..core.io import require, digest

KINDS = {'generate_until', 'choice_loglikelihood'}

@dataclass(frozen=True)
class TaskRequest:
    request_id: str
    task_id: str
    task_version: str
    request_kind: str
    sample_id: str
    payload: dict
    identity: dict

    def validate(self):
        require(self.request_kind in KINDS, 'Unsupported request kind')
        require(all(isinstance(x,str) and x for x in (self.request_id,self.task_id,self.task_version,self.sample_id)), 'Missing request identity')
        require(self.identity.get('task_id') == self.task_id and self.identity.get('task_version') == self.task_version, 'Task request identity mismatch')
        if self.request_kind == 'choice_loglikelihood':
            require(type(self.payload.get('choice_index')) is int, 'Likelihood choice index missing')
            ids=self.payload.get('input_ids');ctx=self.payload.get('context_ids');cont=self.payload.get('continuation_ids')
            require(isinstance(ctx,list) and ctx and isinstance(cont,list) and cont and ids==ctx+cont, 'Likelihood boundary/input mismatch')
            require(all(type(x) is int and x>=0 for x in ids), 'Invalid token ID')
        else:
            require(isinstance(self.payload.get('token_ids'),list) and self.payload['token_ids'], 'Generation token IDs missing')
        return self

class TaskAdapter(Protocol):
    task_id: str
    task_version: str
    request_kind: str
    scorer_version: str
    def requests(self, sample, execution, tokenizer_identity): ...
    def reference(self, sample): ...
    def score(self, sample, records): ...
    def metric_specs(self): ...


def compatible(left, right):
    """Only weight artifact is an intended treatment in B3."""
    for k in ('task_id','task_version','split','sample_id','prompt_identity','tokenizer_identity','scorer_version','source_identity','runtime_identity'):
        require(left.get(k)==right.get(k) and left.get(k) is not None, 'Incompatible task comparison: '+k)
    return True
