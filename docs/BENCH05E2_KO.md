# BENCH05-E2: 준비된 입력의 공통 실행계획

E2는 세 task와 관측 preset을 하나의 `SuiteSpec`에서 읽어, 검증 가능한 동결 계획으로 만드는 CPU 기능이다. 모델을 실행하는 기능은 이번에 추가하지 않았다. 계획 생성은 E1의 acquisition·metric 수를 늘리지 않는다.

지원 task는 `tech_record_extract/0.1`, `arc_challenge/1.0`, `context_evidence_retrieval/0.1`이다. 기존 TaskAdapter의 생성·선택지 likelihood·채점 역할과 C2/D1 준비 입력·선정 계약을 재사용한다. 새 task/model의 이름을 등록하는 것만으로 지원이 입증되지는 않는다.

## 설정과 파일의 역할

- **SuiteSpec**: 사용할 task·정확한 sample ID·arm 역할·실행 종류·관측·metric·예산·출력 의도를 선언한다. `configs/bench05e2/`의 세 JSON이 실제 예다.
- **sources**: 이미 준비된 경량 입력 SLICE의 위치와 SHA를 명시한다. `suite_prepared_sources_v1` manifest는 `prepared_suite_slice_v1` 파일을 가리킨다. 각 slice는 parent raw manifest/plan/source hash, 포함 sample과 제외 수, 원래 source/gold/token 좌표를 유지한다.
- **local**: 물리 artifact 위치·점검 정책·실행 권한의 별도 binding이다. `configs/bench05e2/local.example.json`은 artifact 점검을 요청하지 않으며 runtime 점검도 실행하지 않는다. 개인 local 설정을 배포할 필요가 없다.
- **FROZEN_PLAN.json**: 위 입력을 해석한 의미 identity와 typed operation·의존성·입출력·수량·미실행 이유다. 생성시각이나 출력 폴더만 바뀌면 의미 identity가 바뀌지 않는다. 입력 내용·metric·좌표·컴파일러 source가 바뀌면 다시 검증해야 한다.

현재 input mode는 `prepared_manifest`다. 보존된 token IDs·boundary·mask·offset을 검증하며 tokenizer를 호출하지 않는다. 원시 텍스트만 주면 `NEEDS_PREPARATION`으로 남는다. slice 검증은 parent manifest 전체가 재검증됐다는 뜻이 아니다.

자연 생성은 B2 P00/B4의 **실제 prompt-only request payload**를 사용한다. 진단용 prompt+gold joint sequence를 잘라 생성 입력으로 대체하지 않는다. ARC task 평가는 원래 순서의 모든 choice를 계획하며, gold-choice-only 진단 입력으로 `acc/acc_norm`을 계산할 수 있다고 표시하지 않는다. 생성과 teacher-forced 진단은 별도 acquisition이다.

## 실제 API와 CLI

출력은 새 경로여야 한다. 아래 `INPUTS`는 명시 복사한 prepared slice manifest가 있는 위치다.

```bash
INPUTS=/absolute/copied/inputs
OUT=/absolute/new/e2-plan
inference-kit bench suite plan --suite configs/bench05e2/task_pair_smoke.json --sources "$INPUTS/SOURCES.json" --local configs/bench05e2/local.example.json --output "$OUT"
inference-kit bench suite validate --plan "$OUT/FROZEN_PLAN.json" --sources "$INPUTS/SOURCES.json" --local configs/bench05e2/local.example.json --output /absolute/new/e2-validation
inference-kit bench suite describe --plan "$OUT/FROZEN_PLAN.json" --output /absolute/new/e2-description
```

Python에서도 같은 구현을 호출한다.

```python
from inference_kit import suite_plan, suite_validate, suite_describe

plan = suite_plan(suite_file, sources_file, local_file, new_plan_directory)
check = suite_validate(new_plan_directory / "FROZEN_PLAN.json", sources_file,
                       new_validation_directory, local=local_file)
description = suite_describe(new_plan_directory / "FROZEN_PLAN.json",
                             new_description_directory)
```

`bench suite run`은 제공하지 않는다. 기존 `bench tokens/structure run`이나 B wrapper의 의미를 바꾸지 않는다. plan/validate/describe는 모델·GPU·HTTP inference·외부 네트워크 없이 동작한다.

## 세 예제의 계획 수량

아래는 E2가 보존된 실제 6문항을 읽어 계산하는 **계획 상한**이며 새 실측 결과가 아니다. task별 두 문항과 두 arm을 사용한다.

|계획|모델 작업·수량|주 관측과 control|
|---|---|---|
|`task_pair_smoke.json`|24 POST, 출력 상한 1,040 token = 자연 생성 1,024 + ARC audit 16|tech/retrieval 생성과 ARC 모든 choice likelihood; 기존 recipe별 6개 serving group|
|`diagnostics_pair_token.json`|worker 2개, forward 12회|paired target 110/full 44; structure는 `NOT_REQUESTED`|
|`diagnostics_pair_structure.json`|worker 2개, forward 36회|OFF1/OFF2/ON; ON paired target 110/full 44. arm당 attention/native AV/reconstructed AV 각 72, head 없는 projection 36|

ARC POST는 실제 choice 수를 합한다. 이 예제의 두 ARC 문항은 각각 네 choice지만 일반 계약은 그 수를 고정하지 않는다. 작은 audit generation은 선택지 likelihood/NLL에서 제외한다.

Structure preset은 layer 0/12/23·query head 0/16·두 query role의 보존된 D1 정책을 해석한다. head 없는 `o_proj` 벡터를 head 수만큼 복제하지 않는다. OFF1/OFF2는 integrity control이며 문항 수나 ON 주 token 분모에 합산하지 않는다. 같은 ON pass와 정확히 같은 target/full 좌표를 제공한다는 계약이 있을 때만 별도 token collector를 생략한다.

## 의존성·실행 권한·거절

Operation은 제한된 enum의 데이터다. 준비 입력 검증, SERVE 생성/likelihood, token/structure 수집, task 채점, 분석, index 등록, 보고서가 논리 입출력과 dependency를 가진다. 임의 shell/Python·외부 plugin을 실행하지 않는다. 순환·없는 parent·중복 node/output·arm/sample/prefix 불일치·scorer 입력 누락은 거절한다. 상위 결과가 실패하면 dependent 작업은 `BLOCKED_DEPENDENCY`다.

빈 필수 task/model/sample/job은 `INVALID_EMPTY_PLAN`이다. 모든 필수 작업이 차단되면 `BLOCKED_REQUIRED_OPERATIONS`이며 READY로 올리지 않는다. 선택 관측 OFF는 `NOT_REQUESTED`다. 불완전한 준비자료는 missing/invalid 상태로 남고 0값으로 채우지 않는다.

유효한 정적 계획의 `VALIDATED_NOT_EXECUTED`는 실행 성공이 아니다. GPU node는 새 권한과 당시 runtime/resource preflight를 필요로 한다. E2의 `execution_count=0`, `acquisition_count=0`, authorization은 `NONE`이다. GPU free/busy·port·runtime importability를 조회하지 않는다.

기본 소비자 검증은 `CONTRACT_VALIDATED_ONLY`/`ARTIFACT_PRESENCE_NOT_CHECKED` 범위다. 선택적인 `stat_only` binding은 경로·config 파일의 존재만 확인하고 가중치를 읽거나 identity를 재해시하지 않는다. 이것도 live-ready 인증이 아니다. 과거 ledger는 읽기 전용 snapshot이며 새 실행 권한·예산 예약이 아니다.

E1 historical 결과는 관련 문항 링크일 뿐 새 작업의 cache hit가 아니다. RTN packed SERVE와 `DECODED_WEIGHT_DIAGNOSTIC`/effective BF16 진단을 병합하지 않는다. 기존 HOLD_CLIPPING_VARIANT·품질 INCONCLUSIVE·합성 strict0·ARC REGRESSION_REUSE를 유지한다.

다음 E3는 이 동결 계획을 기존 executor와 연결하는 별도 단계다. 자동 실행·resume·새 모델 지원·원시 텍스트 준비는 E2 완료 범위가 아니다.
