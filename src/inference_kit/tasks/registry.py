"""Explicit local registry; no dynamic code loading or runtime imports."""
from ..core.io import require
from .base import TaskRequest

class TechRecordAdapter:
    task_id='tech_record_extract';task_version='0.1';request_kind='generate_until'
    @property
    def scorer_version(self):
        from .tech_record import SCORER
        return SCORER
    def requests(self,sample,execution,tokenizer_identity):
        require(sample.get('task_id',self.task_id)==self.task_id,'Wrong generation task')
        identity={'task_id':self.task_id,'task_version':self.task_version,'tokenizer_identity':tokenizer_identity}
        return [TaskRequest(execution+'|'+sample['sample_id'],self.task_id,self.task_version,self.request_kind,sample['sample_id'],{'token_ids':sample['generation_input_ids']},identity).validate()]
    def render_prompt(self,sample):
        from .tech_record import render_prompt
        return render_prompt(sample)
    def reference(self,sample):
        from .tech_record import canonical_reference
        return canonical_reference(sample)
    def score(self,sample,records):
        from .tech_record import score
        require(records is None or isinstance(records,dict) and records.get('request_kind','generate_until')=='generate_until','Wrong collector for generation scorer')
        return score(sample,records)
    def metric_specs(self):return ['strict_task_success','task_success_within_10s']


def get_task(task_id, version=None):
    if task_id=='tech_record_extract':adapter=TechRecordAdapter()
    elif task_id=='arc_challenge':
        from .arc_challenge import ArcChallengeAdapter
        adapter=ArcChallengeAdapter()
    elif task_id=='context_evidence_retrieval':
        from .context_evidence import ContextEvidenceAdapter
        adapter=ContextEvidenceAdapter()
    else:raise ValueError('Unsupported task: '+str(task_id))
    require(version is None or version==adapter.task_version,'Unsupported task version')
    return adapter
