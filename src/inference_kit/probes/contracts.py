"""BENCH observation contract v1, usable without Torch/NumPy/Transformers."""
from dataclasses import dataclass, asdict
import math
from ..core.io import require, digest, safe_name

SCHEMA = 'inference-kit-observation-v1'
STATES = {'PLANNED_NOT_VERIFIED','MEASURED','UNSUPPORTED','NOT_APPLICABLE','NOT_RUN',
          'BUDGET_SKIPPED','FAILED','INCOMPARABLE','BLOCKED_SHARED_GPU','UNAVAILABLE'}
KINDS = {'live_model_diagnostic','synthetic_contract'}
LABELS = {'DEPLOYED_PATH_OBSERVATION','INSTRUMENTED_PATH','DECODED_WEIGHT_DIAGNOSTIC','REIMPLEMENTED_OR_SYNTHETIC'}


@dataclass(frozen=True)
class Sample:
    task_id: str
    task_version: str
    sample_id: str
    source_id: str
    prompt: str
    prompt_hash: str
    reference_text: str
    reference_source: str
    tokenization: dict
    correctness_meaning: str = 'Diagnostic continuation only; not a gold accuracy benchmark'

    def validate(self):
        require(self.task_id and self.task_version and self.source_id,'Missing task/source identity')
        require(safe_name(self.sample_id)==self.sample_id and '/' not in self.sample_id,'Unsafe sample ID')
        require(self.prompt_hash==digest(self.prompt),'Prompt identity differs')
        require(self.reference_source=='author_provided_diagnostic_text','Unregistered diagnostic reference source')
        require(self.prompt and self.reference_text,'Empty diagnostic text')
        return asdict(self)


@dataclass(frozen=True)
class Execution:
    run_id: str
    attempt_id: str
    arm_id: str
    model_id: str
    model_revision: str
    artifact_identity: str
    recipe_identity: str
    tokenizer_identity: str
    runtime_identity: str
    execution_mode: str
    effective_weight_dtype: str
    backend: str
    evidence_kind: str
    status: str
    raw_path: str

    def validate(self):
        require(self.execution_mode in LABELS and self.evidence_kind in KINDS,'Unknown execution label/kind')
        require(self.status in STATES,'Unknown execution status')
        for name in ('run_id','attempt_id','arm_id','model_id','model_revision','artifact_identity','recipe_identity','tokenizer_identity','runtime_identity'):
            require(isinstance(getattr(self,name),str) and getattr(self,name),'Missing execution '+name)
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    kind: str
    execution_id: str
    sample_id: str
    evidence_kind: str
    label: str
    status: str
    coordinates: dict
    numeric: dict | None
    details: dict
    schema: str = SCHEMA

    def validate(self):
        require(self.schema==SCHEMA and self.evidence_kind in KINDS and self.label in LABELS,'Invalid observation contract')
        require(self.status in STATES and self.kind in ('target_logprob','full_logits','attention_row','attention_output','reconstructed_AV','output_projection'),'Unknown observation kind/state')
        require(self.coordinates.get('prefix_hash') and self.coordinates.get('token_id_space'),'Missing prefix/token-ID space')
        if self.status=='MEASURED':require(isinstance(self.numeric,dict),'Measured observation needs a numeric blob')
        else:require(self.numeric is None,'Unmeasured observation cannot contain manufactured numeric data')
        if self.evidence_kind=='synthetic_contract':require(self.label=='REIMPLEMENTED_OR_SYNTHETIC','Synthetic evidence cannot be a live label')
        return asdict(self)


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    version: str
    required_observations: tuple
    matching_axes: tuple
    formula: str
    units: str
    direction: str
    aggregation: str
    missing_rule: str = 'UNKNOWN; never impute zero'
    log_base: str = 'e'

    def record(self):return asdict(self)


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    value: float | None
    status: str
    evidence: list
    measured_count: int
    requested_count: int
    denominator: str
    scope: str

    def record(self):
        require(self.status in STATES,'Unknown metric state')
        require(0<=self.measured_count<=self.requested_count,'Invalid metric coverage')
        if self.status=='MEASURED':require(type(self.value) in (float,int) and math.isfinite(self.value),'Invalid numeric metric')
        else:require(self.value is None,'Unknown metric must not be zero-filled')
        return asdict(self)


def target_positions(ids, boundary, special_ids=(), maximum=16):
    require(type(boundary) is int and 0<boundary<len(ids),'Invalid first continuation token boundary')
    require(type(maximum) is int and 0<maximum<=16,'Target position cap')
    positions=[t for t in range(boundary,len(ids)) if ids[t] not in special_ids][:maximum]
    require(positions,'No non-special continuation targets')
    return [{'token_position':t,'logit_position':t-1,'target_token_id':ids[t]} for t in positions]


def attention_coordinates(n_layers,n_heads,n_kv_heads,boundary,targets):
    require(n_layers>0 and n_heads>0 and n_heads%n_kv_heads==0,'Invalid head structure')
    layer=n_layers//2;heads=sorted(set((0,n_heads//2)))
    queries=sorted(set((boundary-1,targets[-1]['token_position']-1)))
    require(min(queries)>=0,'Invalid causal attention query')
    return {'layer':layer,'heads':heads,'queries':queries,'query_to_kv_head':{str(h):h//(n_heads//n_kv_heads) for h in heads}}
