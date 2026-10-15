# 다른 프로젝트에서 사용

wheel 일반 설치 후 checkout 밖에서 실행할 수 있습니다. 기본 기능에 editable/PYTHONPATH/원본 sibling 경로가 필요하지 않습니다. [consumer_project.py](../examples/consumer_project.py)는 자체 synthetic 업무/profile을 추가해 비교/판정을 실제 호출합니다. fixture는 테스트 자료이며 업무 품질 근거가 아닙니다.

```python
from inference_kit import compare, assess, plan
result = compare("my-baseline.json", "my-candidate.json", output="results/paired-01")
decision = assess("bound-index.json", "requirements.json", output="results/decision-01")
execution_plan = plan("spec.json", "local.json")
```

1. 저장된 결과는 `schema_version=1` paired record 또는 검증된 live_quality document 기록으로 전달합니다. task/version/prompt/gold/evidence_kind와 pairing을 맞추세요. acc_norm view를 acc로 바꾸지 않습니다.
2. 요구조건은 기존 STEP02 v2 또는 STEP03 v3 계약을 사용합니다. v3은 GPU/runtime/workload/instrumentation/fixed-profile/tokenizer identity, treatment axes, memory scope/phase, 여러 latency 조건과 quality transfer 정책을 요구합니다. 예시는 각 case replay의 requirements에 있습니다.
3. 실제 evidence index는 `replay`가 만든 `portable-index.json`을 사용합니다. bundle path+SHA를 명시해 archive manifest를 재검증합니다. 원본 evidence_root 문자열을 현재 접근 경로로 사용하지 않습니다. Python API의 메모리 index는 caller가 검증한 입력을 전달하는 경계입니다.
4. model/profile 변경은 새 spec/local을 만듭니다. vLLM은 qwen3/llama architecture, BF16 native 또는 검증한 group128 symmetric INT4 pack-quantized에 제한됩니다. 명시적 artifact receipt의 model/revision/recipe/config/tokenizer hash가 맞아야 합니다. model 이름 등록만으로 GPU 지원이 확인되지는 않습니다.
5. 새 metric은 독립 module에서 sample별 값과 단위·방향·분모·pairing·missing 규칙을 정의하고 core comparison record로 연결하세요. 기존 acc_norm/latency 이름의 의미를 덮어쓰지 않습니다. 지원되지 않는 metric은 거절합니다.

실제 모델 실행이 필요할 때만 준비된 runtime에서 `inference-kit run --spec spec.json --local local.json --output runs/attempt1 --allow-gpu`를 명시합니다. server는 loopback, TP1, activation/KV BF16, 고정 eager/prefix/async 정책입니다. profile 밖의 설정이나 다른 runtime tuple은 먼저 별도 계약 테스트/검토가 필요합니다. 이 문서의 명령은 이번 GPU 재실행 기록이 아닙니다.

packed 조합은 별도 준비된 tensor runtime에서 `python -m inference_kit compose --base <기존RTN> --intervention <기존CLIP> --modules <정확한모듈> --base-receipt <json> --intervention-receipt <json> --output <새artifact> --allow-tensor-runtime`입니다. base/intervention receipt는 동일 model_id/model_revision을 가져야 합니다. R00/R11은 기존 값, R10/R01은 전체 packed/scale/shape bundle을 선택합니다. INT4를 BF16으로 복원하는 ablation이 아닙니다. 조합 후 별도 GPU loader 검증은 여전히 필요합니다.

다음 확장 경로는 **실제 업무 품질 adapter 하나**입니다. 고객/프로젝트의 입력·gold/실패 정의·허용 지연/품질을 먼저 받아 별도 task/version record로 연결합니다. STEP05·모델 평가·새 데이터 수집은 이번 릴리스에 포함하지 않습니다.
