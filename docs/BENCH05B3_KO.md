# BENCH05-B3: 두 번째 task의 공통 연결

0.1.1.dev6은 명시적인 두 task adapter를 제공한다. `task_adapter('tech_record_extract','0.1')`은 `generate_until`, `task_adapter('arc_challenge','1.0')`은 `choice_loglikelihood`다. task는 요청과 참조·채점을 만들며, typed executor는 수집기에 전달한다. 기존 owned serving에 request driver를 주입하므로 loader, lifecycle, GPU ownership, budget, token/attention 수식은 복제하지 않는다. B1/B2 frozen wrapper와 strict scorer는 유지한다.

```python
from inference_kit import task_adapter, bench_b3_replay
arc = task_adapter('arc_challenge', '1.0')
requests = arc.requests(prepared_arc_document, 'run|attempt|bf16', tokenizer_identity,
                        token_id_space, serving_runtime_identity)
# 모든 choice의 saved HTTP를 검증·decode한 뒤에만 문항을 채점한다.
score = arc.score(prepared_arc_document, decoded_choice_records)
result = bench_b3_replay('/explicit/recorded-run-or.zip', '/new/recomputed')
```

실제 CLI는 `inference-kit bench b3 plan --suite ... --models ... --local ... --output ...`, `bench b3 run --plan ... --output ... --allow-gpu`, `bench b3 replay --run ... --output ...`, `bench b3 score --sample ... --choices ... --output ...`다. plan/replay/score는 stdlib만 사용한다. plan은 로컬 파일·ledger를 검사하며 모델이나 GPU를 로딩하지 않는다. tokenizer 준비는 별도 opt-in `python -m inference_kit.probes.arc_tokens`로 기존 진단 runtime에서 수행한다. run은 현재 검증한 Smol pair 전용 recipe다.

ARC는 과거 준비 pool의 원래 sample ID, split, answerKey, choice 순서 및 joint token ID를 유지한다. hash 선택 SMOKE4+REGRESSION64는 과거 test의 **REGRESSION_REUSE**이며 새 TEST가 아니다. 기존 prompt는 이미 chat-rendered다. 추가 template 적용은 0회다. harness가 context의 trailing whitespace를 continuation으로 옮긴 경계도 유지한다. acc는 continuation logprob 합의 argmax, acc_norm은 원래 choice 문자열 character 길이로 나눈 값의 argmax다. 동률은 첫 index다. 선택지별 audit1 생성은 score/NLL에 포함하지 않는다. 한 choice라도 누락되면 문항은 INCOMPLETE이며 argmax를 만들지 않는다.

native HTTP의 모든 continuation token은 t, logit t−1, token ID, prefix hash와 request identity를 갖는다. HTTP 전체 수신 시간은 TTFT가 아니다. token-weighted와 document-macro NLL을 구분한다. 선택지 argmax 일치와 vocabulary next-token argmax는 다른 metric이다. native full-vocabulary 분포는 수집하지 않는다.

선택 ARC 4문항의 gold choice는 외부 answerKey에서 온다. 기존 collector는 BF16 INSTRUMENTED_PATH, RTN DECODED_WEIGHT_DIAGNOSTIC으로 실행한다. QK 기반 확률 재구성과 native SDPA AV를 구분한다. native와 진단 NLL bridge는 측정 범위만 기록하고 동등성을 인증하지 않는다. 새 ARC OFF/ON 대조는 수행하지 않는다. attention 차이는 정답 변화의 인과 설명이 아니다.

새 adapter는 명시적인 registry 항목·task version·request kind·reference source를 정의하고 CPU 계약 테스트를 추가한다. 임의 dynamic code loader나 모든 모델 지원은 없다. 두 task 점수를 평균한 종합 점수는 제공하지 않는다. B2 historical strict JSON 결과와 B3 live ARC 결과는 별도 행이다. HOLD_CLIPPING_VARIANT, 기존 품질 INCONCLUSIVE, worker native exit UNKNOWN을 유지한다. 배포/고객 적합성은 NOT_VALIDATED다.

2026-10-16 첫 CPU 작업 종료 시에는 GPU preflight의 `BLOCKED_SHARED_GPU`였다. 당시 모델 서버/진단 worker 및 추론 POST는 모두0이다. 소유자를 확인할 수 없는 compute PID가 관측돼 제출을 중단한 원시 기록은 `reports/bench05b3_arc_adapter_20261016_142044/`에 그대로 보존한다.

같은 날 사용자의 명시적인 재개 지시 후 같은 dev6 wheel/source digest `c71a3bfd884e182d407820af2c3fac8271e04f128a2efa51c6d5dff04680ed3c`로 고정 계획을 실행했다. 새 결과는 `reports/bench05b3_arc_live_resume_20261016_171610/`이다. 두 serving 서버에서 총544 POST(SMOKE32/REGRESSION512), 오류0, 두 diagnostic worker에서 arm당27 target 위치/16 attention 행을 관측했다. REGRESSION64에서 BF16 acc/acc_norm은23/64·23/64, RTN은16/64·25/64다. 정규화와 raw 판정이 달라 범용 양자화 우위를 주장하지 않는다. native↔diagnostic NLL 최대 절대 차이는 BF16 0.364379/RTN 1.607199 nat이며 경로 동등성을 주장하지 않는다. GPU job wall 합계68.621717초, 캠페인4597.717952/7200초다. 원시값 CPU 재계산·외부 설치 wheel 소비자 검증은 새 report에 별도 기록하며 과거 차단 결과를 덮어쓰지 않는다.

likelihood 요청은 HTTP 제출 직전에 immutable journal에 request/body/identity와 경계를 남긴다. 취소·중단으로 완성 raw response가 없는 선택지도 journal에서 INCOMPLETE로 복구하며, 남은 선택지로 argmax를 만들지 않는다. 종료 시각을 관측하지 못한 경우 latency는 UNKNOWN이다. 이 journal은 서버가 실제 요청을 받았다는 증명이 아니라 client submit attempt의 근거다.
