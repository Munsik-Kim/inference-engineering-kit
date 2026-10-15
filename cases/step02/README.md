# STEP02 — 두 모델을 같은 실행·품질 계약에 연결

문제: Qwen/Smol과 BF16/weight-only artifact의 실행·측정·품질 의미를 이어붙이는 반복 작업이 필요했습니다.

구현: 명시적 모델/recipe/경로 identity, artifact→HTTP request→원시 likelihood→공식 acc/acc_norm→paired 비교→요구조건 screen. suite의 기존 함수를 [core/quality](../../src/inference_kit/core/quality.py), [SSE adapter](../../src/inference_kit/adapters/http.py), [선택기](../../src/inference_kit/core/assess.py)로 이식했습니다. CPU replay는 모델을 로딩하지 않습니다.

설계 선택: smokes와 품질·성능을 분리하고, normalization과 token boundary를 보존했습니다. parent exit/owned cleanup/native UNKNOWN/kernel selected-log는 별도입니다. 새 패키지의 경로는 local/spec 기준이며 원래 /home 경로를 참조하지 않습니다.

실패와 수정: 초기 Qwen W4의 공유 GPU 영향 attempt는 PARTIAL_UNCERTIFIED/SHARED_GPU_INTERFERENCE, 관측 max14370MiB로 남았습니다. 최신 W4는 별도 3서버·222요청(192 timed/30warmup), 오류0입니다. 서로 다른 background의 옛 BF16과 메모리 절감률을 통제 비교로 만들지 않습니다.

대표 결과(과거 ARC-Challenge TEST 고정256 subset, 이번 재채점):

| 모델/방법 | acc_norm | acc |
|---|---:|---:|
| qwen3_4b--bf16 | 0.44921875 | 0.41796875 |
| qwen3_4b--gptq_w4 | 0.43359375 | 0.40234375 |
| smollm2_live--bf16 | 0.41796875 | 0.41015625 |
| smollm2_live--gptq_w4 | 0.39453125 | 0.33984375 |
| smollm2_live--rtn_w4 | 0.42187500 | 0.34765625 |

Qwen W4−BF16은 −1.5625%p, 기존95%구간[−5.46875,+2.34375]%p로 −2%p 경계를 걸칩니다. INCONCLUSIVE를 유지합니다. 공식 전체 benchmark 점수가 아닙니다. Smol clipping은 당시 DEV에서 worse여서 TEST/performance 후보가 아니었습니다.

재사용 API: `inference_kit.replay('step02', bundle, fresh_output)`, `compare`, `assess`. `inference-kit replay --case step02 --bundle STEP02_QWEN_W4_UPDATED_20261014.zip --output runs/step02-01`을 사용합니다.

[summary.json](summary.json)은 전체5구성/12판정/옛 partial을 기계 결과에서 생성했습니다. raw·frozen data/runtime/source·실패·correction은 [evidence index](../../evidence/index.json)의 STEP02 asset에 있습니다. 이번 GPU 실행은0이며 과거 recipe의 재현과 maintained GPU 경로 검증은 다릅니다.
