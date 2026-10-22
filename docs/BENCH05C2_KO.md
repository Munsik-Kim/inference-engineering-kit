# BENCH05-C2: 설정으로 선택하는 token 전용 관측

`ProbeSpec`으로 참조 token과 full-vocabulary 위치 수를 정하고, 이미 동결된 입력을 같은 token ID 그대로 관측한다. task별 입력 해석은 준비 adapter에 두고 공통 worker는 token forward·저장만 담당한다. 새 prompt, tokenizer, 자유 생성, serving 서버, attention 관측은 추가하지 않는다.

## 입력과 선택 계약

`probes/token_inputs.py`는 C1 `INPUT_SOURCES.json`의 명시적인 B2 P00·B3·B4 source와 해당 frozen plan을 읽는다. pinned raw manifest와 그 안의 plan 크기·SHA·identity를 확인한다. 응답·attention blob을 읽어 입력을 고르는 과정은 없다.

| Task | 보존하는 외부 참조 | 선택 범위 |
|---|---|---|
| tech_record_extract | 저자 gold의 canonical JSON | B2 P00의 기존 진단 4문항 |
| arc_challenge | 외부 answerKey의 gold choice | B3의 기존 진단 4문항; harness가 이동한 선행 공백 포함 |
| context_evidence_retrieval | 저자 gold의 ANSWER/EVIDENCE 두 줄 | B4의 기존 진단 4문항 |

각 task에서 `sha256(task_id + "|" + sample_id)` 오름차순의 처음 2개를 선택한다. MAIN의 입력은 총 6개다. 모델 결과로 문항을 선택하지 않는다. 기존 prompt/reference·UTF-8/codepoint offset·mask·선택 좌표와 원래 출처를 보존하며, chat template를 다시 적용하거나 재토큰화하지 않는다. offset이 없으면 `MAPPING_UNAVAILABLE`로 남기고 원래 token ID·boundary·mask를 사용한다.

위치 선택 정책은 `preserved_field_round_robin_uniform_v1`이다.

1. 유효한 기존 target을 원래 순서대로 cap까지 보존한다.
2. 여유가 있으면 첫 참조 token, field의 양 끝, field 이름순 round-robin 위치를 추가한다.
3. 남은 유효 참조 위치에서 정수 floor 규칙으로 균등 채운다. t=0, special/padding 위치는 제외하고 중복·가짜 padding으로 수를 채우지 않는다.
4. Full-vocabulary 위치는 새 target에 포함된 기존 full 위치를 먼저 보존한 뒤 같은 균등 규칙으로 채운다.

`reference_role_filter`는 실제 선택 대상을 필터링한다. field가 겹치는 token도 전체 target에서는 한 번만 센다. cap은 요청 상한이며 continuation이 짧으면 실제 관측 수가 더 작다.

| 설정 | 문항당 target/full cap | 현재 고정 6입력의 실제 target/full, arm당 | numeric raw 예상, 두 arm |
|---|---:|---:|---:|
| MAIN | 32 / 8 | 110 / 44 | 17,302,384 bytes |
| SMALL | 8 / 2 | 44 / 12 | 4,718,944 bytes |

이 표는 저장된 입력에서 CPU로 계산한 계획값이다. JSON·log 크기를 포함하지 않으며 모델 속도나 품질 측정값이 아니다. 두 설정은 동일한 입력에서 실제 위치·배열 크기를 다르게 선택한다. SMALL은 CPU 계약 검증용이며 자동 추가 GPU 실험을 뜻하지 않는다. 입력 길이 상한은 640이며 초과하면 잘라서 실행하지 않고 차단한다.

## 실제 CLI와 Python API

현재 `bench tokens`의 하위 명령은 `plan`, `run`, `export`, `compare`다. 각 `--help`를 실제 확인했다. 아래 경로는 사용자가 준비한 로컬 자료로 바꾼다. `local.example.json`의 origin 사용량 placeholder도 검산한 **숫자**로 바꾸며 campaign을 0으로 초기화하지 않는다.

```bash
inference-kit bench tokens plan \
  --spec configs/bench05c2/main.json \
  --inputs /prepared/C1/INPUT_SOURCES.json \
  --models configs/bench05a/pair.json \
  --local /prepared/C2/local.json --output /runs/C2/plan-main

inference-kit bench tokens run \
  --plan /runs/C2/plan-main/PLAN_FROZEN.json \
  --output /runs/C2/run-attempt1 --allow-gpu
```

`plan`은 GPU·모델·tokenizer·네트워크를 사용하지 않는다. `run`은 새 task clock, 누적 ledger, 정확한 artifact/runtime/설치 package site, 실제 GPU ownership과 자원을 검사한다. 실패 attempt는 보존하며 source/spec/input 변경 후에는 새 freeze가 필요하다. 공유·미확인 compute PID가 있으면 launch하지 않는다.

새 raw를 공통 token view로 내보내는 source 파일은 계획 입력 manifest와 별개다.

```json
{
  "schema_version": 1,
  "kind": "token_input_sources",
  "sources": [{
    "source_id": "c2_configured",
    "adapter": "token_probe_raw_v1",
    "path": "/runs/C2/run-attempt1",
    "raw_manifest_sha256": "REPLACE_WITH_COMPLETE_RAW_MANIFEST_SHA256"
  }]
}
```

```bash
inference-kit bench tokens export \
  --sources /prepared/C2/export-sources.json --output /runs/C2/view
inference-kit bench tokens compare \
  --view /runs/C2/view --baseline bf16 --candidate rtn --output /runs/C2/comparison
```

`export`와 `compare`는 저장 자료만 읽는 CPU 재계산 경로다. 존재하는 output을 덮어쓰지 않는다. 임의 archive Python을 실행하지 않으며 원시 blob/hash·prefix·vocabulary·task/reference·선택 계약을 검증한다.

```python
from inference_kit import tokens_plan, tokens_export, tokens_compare

plan = tokens_plan("main.json", "INPUT_SOURCES.json", "pair.json",
                   "local.json", "new-plan")
view = tokens_export("export-sources.json", "new-view")
comparison = tokens_compare("new-view", "bf16", "rtn", "new-comparison")
```

명시적인 GPU 실행의 Python 입구는 `tokens_run(plan_path, new_output)`이다. CPU 소비자 예제에서는 호출하지 않는다. C2에는 별도의 `tokens replay` 명령이 없으며 위 export→compare가 raw 재계산이다.

## 관측 의미와 제한

BF16은 `INSTRUMENTED_PATH`, RTN은 기존 packed artifact를 검증해 해석한 `DECODED_WEIGHT_DIAGNOSTIC`이다. 후자는 effective BF16 진단 forward이며 Marlin 내부 관측·INT4 serving 메모리·배포 동등성을 뜻하지 않는다. 이번 worker는 선택 hidden 위치만 native LM head에 넘겨 target logprob와 선택 full logits를 저장한다. attention observer·새 native likelihood bridge·OFF/ON 대조는 실행하지 않는다.

NLL·ΔNLL, raw top1, 저장된 완전 vocabulary에서만 KL/JS·entropy·top-k/rank/margin을 계산한다. 서로 다른 task/reference/policy/acquisition을 종합 점수로 평균하지 않는다. 문항·token 분모와 token-weighted/document-macro 평균을 구분한다.

새 C2의 same-pass consistency 계약은 `same_pass_saved_f32_v1`, 절대 허용차 `2e-5 nat`이다. 같은 pass의 저장된 float32 full logits를 CPU log-softmax한 target 값과 저장된 float32 scalar를 비교한다. 과거 C1 source의 acquisition plan에 허용차가 선언되지 않았으면 그 source는 계속 `UNKNOWN`이다. 새 기준을 과거 raw에 소급해 인증하지 않는다.

실제 실행 coverage·부모 종료·소유 자원 정리·native worker exit·GPU 시간은 해당 run receipt를 따른다. CPU 계약 검증을 GPU 성공으로 표시하지 않는다. 기존 `HOLD_CLIPPING_VARIANT`, 품질 `INCONCLUSIVE`, diagnostic/serving 비동등성을 유지한다. 원격 CI는 별도 실행 전까지 `NOT_RUN`이며, 이 단계는 새 품질 benchmark·production/SLA·고객 검증이 아니다.
