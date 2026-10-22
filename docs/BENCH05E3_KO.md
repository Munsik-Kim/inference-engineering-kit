# BENCH05-E3: 동결 Suite의 제한 실행과 근거 기반 재개

E3는 E2의 준비 입력·typed operation 계획을 기존 owned serving/token worker 및 CPU 채점·분석·index/report에 연결한다. 같은 세 task의 보존된 6입력과 Smol pair만 지원하는 제한된 recipe다. 범용 scheduler, raw-text 준비, 새로운 task/model 구조 지원은 구현하지 않는다.

E2의 계획은 acquisition이 아니다. 과거 E2 계획·source hash를 보존하고, 현재 source로 새 계획과 별도 실행 identity를 동결한다. 과거 source 검사를 끄거나 E2의 `execution_count=0`을 실행 결과로 고치지 않는다.

## 공개 명령과 API

`plan/validate/describe`는 기존 CPU 계약을 유지한다. 추가된 `run/status/resume/replay`는 다음과 같다. 경로는 사용자 로컬의 명시적 위치이며 각 새 output은 비어 있어야 한다. 아래는 실행 방법이며 실제 측정 완료를 뜻하지 않는다.

```bash
R=/absolute/bench05e3-report
RUN="$R/run-attempt1"
LOCAL=/absolute/private-local.json

inference-kit bench suite run --plan "$R/frozen-plan/FROZEN_PLAN.json" --local "$LOCAL" --output "$RUN" --allow-gpu --stop-after-node serve-bf16
inference-kit bench suite status --run "$RUN" --output /absolute/new/status-check
inference-kit bench suite resume --run "$RUN" --local "$LOCAL" --allow-gpu
inference-kit bench suite replay --run "$RUN" --output /absolute/new/cpu-replay
# 완료가 검증된 동일 run의 재개는 no-op이며 새 GPU 권한이 필요하지 않다.
inference-kit bench suite resume --run "$RUN"
```

```python
from inference_kit import suite_run, suite_status, suite_resume, suite_replay

paused = suite_run(plan_file, local_file, new_run_directory,
                   allow_gpu=True, stop_after_node="serve-bf16")
state = suite_status(new_run_directory, new_status_directory)
completed = suite_resume(new_run_directory, local=local_file, allow_gpu=True)
recalculated = suite_replay(new_run_directory, new_replay_directory)
noop = suite_resume(new_run_directory)  # 검증된 완료 run에 한함
```

`run`과 pending GPU 작업의 `resume`는 명시적인 `--allow-gpu`가 필요하다. `status/replay`, 완료 no-op resume는 GPU를 조회하거나 모델·tokenizer를 로딩하지 않는다. API/CLI의 기본 import도 ML SDK를 요구하지 않는다.

## 계획과 실제 실행 단위

보존된 6문항의 합성 기술 기록 2개·ARC 2개·문맥 검색 2개를 그대로 사용한다. 실제 생성 payload와 teacher-forced joint는 별도 입력이며, ARC는 원래 순서의 모든 choice를 평가한다. 새 prompt·gold·tokenizer 준비는 없다.

이 recipe의 **계획값**은 24 inference POST, 생성 상한 1,040 token이다. 그중 자연 생성 상한은 1,024, ARC likelihood에 제외되는 audit generation은 16 token이다. 진단은 paired target 110/full 44 좌표를 계획한다. 이는 실행 전 수량이며 실제 수행·누락·예산은 run receipt로 확인한다.

E2는 recipe 경계를 유지한 6개 logical serving operation group을 계획했다. E3는 동일한 serving 설정을 확인한 경우에만 arm별 서버 1개가 여러 task request를 처리하도록 묶는다. 원래 typed output을 각 group의 `provides`에 명시하므로 operation을 조용히 생략하지 않는다. 이 recipe는 SERVE 2개 + token worker 2개, 총 4개 acquisition job을 계획한다. 임의의 Suite에 최소 서버 수를 보장하지 않는다.

관측은 token-only다. Structure/attention/AV/o_proj는 `NOT_REQUESTED`이며 새 D 관측을 수행하지 않는다. RTN SERVE는 실제 packed artifact 경로이고, 별도 RTN token 진단은 `DECODED_WEIGHT_DIAGNOSTIC`/effective BF16이다. 둘을 같은 배포 내부 관측으로 합치지 않는다.

## 안전 경계와 재개

실행 단위는 고정된 내부 adapter enum이다. 임의 shell command·Python module 경로·pickle을 입력으로 실행하지 않는다. typed plan의 모든 node는 정확히 하나의 실행 단위가 명시적으로 제공해야 한다.

각 node는 시작 journal, 독립 attempt 경로, raw evidence, 완료 receipt를 가진다. `COMPLETE_VALIDATED`는 단순 checkpoint 문자열이 아니다. 완료 receipt의 hash·plan/input/dependency identity, nonempty 파일 집합·bytes/SHA, GPU coverage와 owned cleanup 확인이 필요하다. source·준비 입력·raw가 바뀌거나 없어지면 재개를 거절한다. 완료 receipt가 먼저 저장되고 checkpoint가 갱신되기 전에 중단된 경우에는 그 receipt를 재검증해 복구한다.

`--stop-after-node serve-bf16`은 owned serving 종료·정리와 완료 근거가 검증된 경계에서만 멈춘다. 실행 중 임의 위치의 pause/resume나 exactly-once HTTP/GPU 제출 복구를 보장하지 않는다. GPU node가 RUNNING/FAILED인데 durable completion이 없으면 제출 여부를 UNKNOWN으로 남기고 자동 재전송하지 않는다. 낮은 점수도 재시도 이유가 아니다. CPU 파생 stage는 검증된 immutable 입력에서 새 attempt로 계산할 수 있지만 새 GPU acquisition과는 구분한다.

run lock은 동시 controller를 거절하며 정상·예외 종료에 해제된다. hash-chain journal과 checkpoint는 mutable 제어 기록이고 raw manifest는 별도의 immutable 근거다. 완료 run의 no-op resume는 기존 완료 파일과 ledger를 다시 확인하고 새 worker/POST/ledger debit 0을 기록한다.

## 예산·출처·검산

과거 campaign ledger를 읽고 parent path/SHA/used를 확인한 새 continuation만 만든다. parent ledger는 바꾸지 않는다. child cap·origin·parent·job ID·양수 wall cost·누적 사용량을 검증하며, 완료 GPU receipt에는 당시 ledger prefix를 보존한다. 과거 계획이나 잔여 예산이 새 실행 권한은 아니다. shared campaign 예약은 서로 다른 새 run이 같은 parent ledger를 몰래 중복 소비하는 것을 차단한다.

실행 직전의 CPU RAM/disk·artifact/runtime identity·현재 GPU ownership은 별도 preflight다. GPU 상태가 불확실하거나 타인의 compute가 보이면 새 제출을 차단하며 타인의 프로세스를 종료하지 않는다. 준비 입력의 길이·dtype·관측 수량을 줄여 강제로 실행하지 않는다.

CPU replay는 저장된 raw HTTP/SSE·likelihood token·선택 numeric blob과 기존 task scorer/C1 token 계산기를 사용한다. 모델 출력과 gold가 같다는 가정이나 summary 숫자의 단순 재출력으로 대체하지 않는다. ARC `acc/acc_norm`, JSON strict task, 두 줄 검색 strict task는 서로 다른 metric·분모다. 신규 2문항 결과를 과거 24/64문항 결과와 합산하지 않으며 E1의 historical panel과 현재 run panel을 구분한다.

HOLD_CLIPPING_VARIANT·이전 품질 INCONCLUSIVE·합성 strict0·ARC REGRESSION_REUSE·native worker exit UNKNOWN 등 과거 판정은 보존한다. 이번은 실행·재개·근거 전달 경로 검증이다. 배포 동등성, 일반 품질 우열, production SLO, 오류의 인과 기전, 고객 적합성을 입증하지 않는다.
