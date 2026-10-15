# STEP03 — Weight × KV pool 개입과 여러 업무 조건

문제: L1024/C4의 TTFT 대기가 작은 KV pool 때문인지, weight quantization과 구분해야 했습니다.

구현: BF16/W4 × KV4096/8192 네 arm, 실제 argv/시작 capacity 확인, 3시간블록 교차 순서, SSE/queue/Prometheus/NVML 원시 수집, 같은 블록의 두 요인 차이. [experiments/step03](../../src/inference_kit/experiments/step03.py), [metrics](../../src/inference_kit/core/metrics.py), [다중 조건 selector](../../src/inference_kit/core/assess_multi.py)를 재사용할 수 있습니다.

설계 선택: KV는 모두 BF16이며 pool capacity만 바꿨습니다. context2048/batched_tokens2048/C4/output128은 고정입니다. TTFT와 client queue/HTTP E2E/TPOT, gauge time integral과 histogram, performance startup 포함 max와 sanity/task max를 구분했습니다. 384timed+96warmup=480요청과 256sanity HTTP가 기록됐습니다. 고유 performance prompt vector는32개입니다.

대표 과거 결과(이번 raw CPU 재계산; worst-round p95 seconds):

| Arm | Weight/KV | whole-device max MiB | L128 TTFT p95 | L1024 TTFT p95 | L1024 TPOT p95 |
|---|---|---:|---:|---:|---:|
| A | bf16/4096 | 10877 | 0.065114 | 3.222303 | 0.028320 |
| B | bf16/8192 | 11377 | 0.073857 | 0.321234 | 0.030236 |
| C | gptq_w4/4096 | 5773 | 0.072650 | 2.542301 | 0.024417 |
| D | gptq_w4/8192 | 6285 | 0.076353 | 0.325364 | 0.030884 |

BF16과 W4 모두 L1024/C4 대기와 TTFT가 줄어 해당 profile 안에서 MECHANISM_SUPPORTED_WITHIN_PROFILE입니다. 이는 작은 pool 설정 문제의 근거이며 W4만 가능한 해결책이 아닙니다. W4의 메모리는 낮지만 K8192의 W4가 BF16보다 느린 cell도 있습니다. 추가 cache memory를 쓴 개선을 동일 메모리 알고리즘 우위로 부르지 않습니다.

DEMO max8192MiB/짧은TTFT0.2s/긴TTFT1s/relative-loss2%p/absolute0.4 정책: W4 K8192가 retained quality를 가정한 point 조건부 후보입니다. interval/exact-profile/no-retained는 INSUFFICIENT_EVIDENCE입니다. W4의 과거 품질 INCONCLUSIVE는 그대로이며 DEV16 cache sanity가 exact-profile 품질 인증을 대신하지 않습니다. 실제8GiB GPU, 혼합 arrival workload, production tail SLA는 미검증입니다.

실패와 수정: immutable writer/progress 경계를 구분한 원본 correction과 관측 한계를 asset에 보존합니다. 이번 export는 개인 local.json2개만 제외하고 raw 숫자·원본 source hash는 보존했습니다. 새/원본 manifest를 구분해 두었습니다.

재현: `inference-kit replay --case step03 --bundle STEP03_SANITIZED_EVIDENCE_20261015.zip --output runs/step03-01`; 이어 `assess --index runs/step03-01/portable-index.json --requirements runs/step03-01/requirements/point.json --output runs/decision-01`.

원시 request/metrics/memory/argv/paired difference와 frozen profile은 [release index](../../evidence/index.json), 기계 생성 요약은 [summary.json](summary.json)입니다. 과거 실험을 새 패키지 GPU 실행으로 재라벨링하지 않습니다.
