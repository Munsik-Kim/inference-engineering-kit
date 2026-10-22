# BENCH05-B1 — tech_record_extract v0.1

dev3(PR1/main970f949c)의 UTF-8·ASCII CLI·테스트0건 실패·합성 자원 분리·설치 wheel 계약을 유지한 로컬 dev4 추가 기능이다. 기존 A wrapper는4문항/128토큰/plain completion 정책을 유지한다. B1은 별도 task/preparation ID, chat template1회, prompt384/reference128/결합512토큰으로 검사한다.

`tasks.tech_record`는 문서 렌더링, canonical gold/필드 값 span, 독립 원문-gold 검사와 순수 CPU 채점을 담당한다. HTTP·모델 로딩을 호출하지 않는다. source events에서 만든 문서의 최종 수정/누락을 별도의 원문 행 parser로 검증하며, 모델 응답 채점은 동결 gold와 비교한다. 합성 문서는 parser로도 풀 수 있는 작은 개발 자료다.

strict JSON은 전체 응답의 앞뒤 공백만 제거한다. 중복 key/NaN/Infinity/code fence/추가 설명/잘못된 타입을 거절한다. 숫자 문자열·bool을 number로 바꾸지 않고 Decimal absolute tolerance1e-6을 사용한다. 정보 없음은 명시적 null이며 key 누락과 다르다. unsupported_value는 이 문서의 온도/상태 gold와 다른 non-null 값에 한정하고, 다른 대상/초기 source 값과 문서에 없는 불일치를 구분한다. 자유 생성 전체의 hallucination rate가 아니다.

`natural_generation_v1`은 기존 fixed length와 별도다. temperature0/top_p1/n1, 중립 penalty, max192, ignore_eos=False, guided decoding/repair/추가 stop/min_tokens/재시도 없음. 자연 EOS/length cap/빈 출력/잘림/HTTP 오류를 분리하며 JSON 오류·오답으로 문항을 빼지 않는다. SMOKE4는 본 점수에 합산하지 않는다. 정해진24문항 분모와 NOT_RUN coverage를 유지한다.

지연 ID는 `request_latency_http=response_complete-http_submit`, `request_latency_task=response_complete-queued`, `ttft_http=first_content-http_submit`이다. 기존 slot 기준 TTFT를 재정의하지 않는다. `task_success_within_10s`는 HTTP 지연을 사용한 합성 DEMO 정책이며 고객 SLA가 아니다. TPOT는 실제 completion token 수를 분모로 한 요청 평균 추정이다.

probe는 결과와 무관한 seed909261 규칙으로 기본en/수정ko/누락en/유사IDko 각1개를 고른다. canonical JSON의 온도/상태 값 span과 겹치는 token을 우선하고 남은 자리를 균일 선택해 최대16위치를 기록한다. 값의 JSON 문자열 quote/null/숫자 literal을 span에 포함한다. key 이름은 값 span이 아니다. 나머지 A의 동일 중간 layer/head/query 선택과 blob·weight loader·attention metric을 재사용한다. full logits와 저장 target의 허용차2e-5는 기존 A와 같다. row sum 허용차2e-6, output near-zero energy1e-12를 유지한다.

BF16/RTN 자연 생성은 원래 artifact와 serving runtime의 실제 배포 경로다. RTN 내부 진단은 실제 packed 값을 decode한 **DECODED_WEIGHT_DIAGNOSTIC**이며 compressed kernel 내부 관측/INT4 VRAM 측정이 아니다. A의 기존 bridge 차이를 원인 분석하거나 이번 profile의 동등성 인증으로 사용하지 않는다.

## 실제 인터페이스

```bash
inference-kit bench b1 plan --suite suite.json --models pair.json --local private-local.json --output new-plan
inference-kit bench b1 run --plan new-plan/PLAN_FROZEN.json --output new-run --allow-gpu
inference-kit bench b1 replay --run new-run --output new-replay
inference-kit bench b1 score --sample one-sample.json --generation one-generation.json --output new-score
```

`plan/score/replay`는 Torch/vLLM/Transformers/NumPy/aiohttp나 GPU를 요구하지 않는다. tokenizer 준비만 별도 기존 runtime에서 명시적으로 수행한다.

```bash
python -m inference_kit.probes.task_tokens --data data.json --tokenizer /prepared/pinned-tokenizer --config /prepared/bf16/config.json --output prepared.json --runtime-output diagnostic-runtime.json
```

API는 `bench_b1_plan`, `bench_b1_run`, `bench_b1_replay`, `bench_b1_score`이다. wheel에 동봉한 `inference_kit.resources/bench05b1-data.json`은 직접 작성한28개 문서와 gold이며, 실제 응답이나 모델이 아니다. local 경로는 configs/bench05b1/local.example.json의 root binding으로 명시한다. 실제 local.json은 배포하지 않는다. GPU controller와 worker 모두 동결한 동일 package_site를 사용해야 한다.

새 요청·관측은 즉시 UTF-8 JSONL과 little-endian float32 blob으로 기록한다. 원시는 새 attempt/immutable manifest, budget/progress는 atomic snapshot이다. replay는 archive 내부 Python을 실행하지 않고 raw SSE·numeric blob에서 score/paired/sample-linked/CSV/Markdown을 계산한다. ZIP path/size/symlink 검사를 재사용한다. 큰 연구 raw와 B1은 별도 provenance이며 기존 HOLD_CLIPPING_VARIANT/품질 INCONCLUSIVE를 유지한다. B1 원격 CI는 NOT_RUN이다.
