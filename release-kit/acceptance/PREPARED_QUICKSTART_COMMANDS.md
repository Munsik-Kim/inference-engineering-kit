# C: 저장 prepared → 공통 plan (CPU, tokenizer 호출 없음)

키트 root에서 일반 설치한 후보 interpreter/console을 사용한다. 각 output은 새 경로다. fixture의 `SOURCES.json`과 세 source member의 저장 hash/ID·raw/gold/token/offset/mask를 검증한다. `ancestry/PARENT_PREPARED_MANIFEST.json`은 과거 full 준비 inventory의 근거이며 제외된 runtime receipt를 통과시키는 manifest가 아니다. 원래 full parent를 `bench inputs validate`에 전달하지 않는다. 현재 계약 검사는 공개 `bench suite plan/validate`가 모든 active SOURCES member를 읽는다.

```sh
"$PY" -I -B -m inference_kit bench suite plan --suite examples/smol-task.suite.json --sources fixtures/prepared/smol/SOURCES.json --local examples/CPU_LOCAL.json --output "$OUT/smol-task-plan"
"$PY" -I -B -m inference_kit bench suite validate --plan "$OUT/smol-task-plan/FROZEN_PLAN.json" --sources fixtures/prepared/smol/SOURCES.json --output "$OUT/smol-task-check"
"$PY" -I -B -m inference_kit bench suite describe --plan "$OUT/smol-task-plan/FROZEN_PLAN.json" --output "$OUT/smol-task-description"
"$PY" -I -B -m inference_kit bench suite plan --suite examples/smol-token.suite.json --sources fixtures/prepared/smol/SOURCES.json --local examples/CPU_LOCAL.json --output "$OUT/smol-token-plan"
"$PY" -I -B -m inference_kit bench suite validate --plan "$OUT/smol-token-plan/FROZEN_PLAN.json" --sources fixtures/prepared/smol/SOURCES.json --output "$OUT/smol-token-check"
"$PY" -I -B -m inference_kit bench suite describe --plan "$OUT/smol-token-plan/FROZEN_PLAN.json" --output "$OUT/smol-token-description"
"$PY" -I -B -m inference_kit bench suite plan --suite examples/qwen-task.suite.json --sources fixtures/prepared/qwen/SOURCES.json --local examples/CPU_LOCAL.json --output "$OUT/qwen-task-plan"
"$PY" -I -B -m inference_kit bench suite validate --plan "$OUT/qwen-task-plan/FROZEN_PLAN.json" --sources fixtures/prepared/qwen/SOURCES.json --output "$OUT/qwen-task-check"
"$PY" -I -B -m inference_kit bench suite describe --plan "$OUT/qwen-task-plan/FROZEN_PLAN.json" --output "$OUT/qwen-task-description"
```

설치 console은 `"$CONSOLE" bench suite ...`로 같은 인자를 사용한다. Python API는 `inference_kit.suite_plan(suite,sources,local,output)`, `suite_validate(plan,sources,output)`, `suite_describe(plan,output)`다. 파일을 옮기면 새 후보 source에서 plan을 새로 생성하거나 `suite_validate`에 새 SOURCES locator를 명시한다. 옛 source의 실행계획을 force-resume하지 않는다.

|예제|문서|요청/관측 계획|실행 근거|
|---|---|---|---|
|Smol task|3|12POST/최대520출력token/6serving jobs|계획만; POST0|
|Smol token-only|3|2workers/6forward,74scalar/24fullvector =37/12pair|계획만; forward0; structure NOT_REQUESTED|
|Qwen serving-only|3|12POST/최대520출력token/6jobs,38native ARC scalar =19pair|계획만; POST0; full NOT_COLLECTED|

Qwen SOURCES는 기존 MAIN9의 source3개를 bytes 그대로 포함하고 suite는 고정 첫1문서/task만 명시 선택한다. Smol은 과거 독립 준비 subset3다. 문항/target를 재토큰화하거나 다른 모델의 ID로 맞추지 않는다. CPU_LOCAL은 ARTIFACT_PRESENCE_NOT_CHECKED, runtime/GPU CHECK_NOT_RUN, authorizationNONE다. 현재 Smol live binding은 combined SERVE+token이므로 위 task-only/token-only CPU plan을 각각 live-ready로 읽지 않는다.
