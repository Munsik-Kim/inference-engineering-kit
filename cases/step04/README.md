# STEP04 — 국소 오차와 task 품질을 분리한 packed 개입

문제: 과거 CAL의 합산 SSE 개선이 H=model.layers.7.mlp.down_proj에 집중됐지만 최종 품질 개선과 연결되는지는 몰랐습니다. H는 과거 관측으로 정한 사후 표적입니다.

구현: R00/R10/R01/R11은 H와 나머지 R의 clipping on/off이며 모두 INT4입니다. weight_packed/weight_scale/weight_shape를 묶어 같은 scheme 원본 사이에서 교체했습니다. native embedding/lm_head/norm과 tokenizer/config, tensor 출처·shape·저장/재로딩을 검증했습니다. [tensor adapter](../../src/inference_kit/adapters/tensors.py), [고정 recipe](../../src/inference_kit/experiments/step04_recipe.py), [분석](../../src/inference_kit/experiments/step04.py)을 제공합니다.

설계 선택: frozen ratios를 재탐색하지 않고, WikiText holdout32source의 공통 BF16 activation 국소 오차와 새 ARC train DEV256의 실제 packed likelihood6구성을 구분했습니다. bootstrap5000은 같은 문항을 공동 재표집했고 sentinel8문항은 본 평균에서 제외했습니다. TEST/성능 평가를 추가하지 않았습니다.

대표 과거 결과(이번 raw HTTP 재계산):

| Arm | acc_norm 정답/256 | acc_norm | acc |
|---|---:|---:|---:|
| R00 | 99 | 0.38671875 | 0.35937500 |
| R10 | 97 | 0.37890625 | 0.36328125 |
| R01 | 94 | 0.36718750 | 0.33593750 |
| R11 | 98 | 0.38281250 | 0.33593750 |
| BF16_REF | 98 | 0.38281250 | 0.38281250 |
| GPTQ_REF | 95 | 0.37109375 | 0.31250000 |

H의 holdout SSE 감소는 88.5619%로32/32source에서 남았고, 전체 SSE 감소의 99.3753%를 차지했습니다. 하지만 H만 적용한 R10은 RTN보다 정답2개 적었고 R01/R11도 +2%p screen을 통과하지 못했습니다. 결론은 **HOLD_CLIPPING_VARIANT**입니다. 국소 목적함수 개선을 최종 정확도 개선으로 부르지 않습니다. 층간 오차 전파·특정 q/k 원인·전체 모델 상호작용 부재를 입증하지 않습니다.

실패와 수정: 첫 local attempt의 자체 CUDA context 종료 gate 실패와 corrected attempt2를 모두 보존합니다. 원본 GPU 실행 source-v1, 이후 source-snapshot, CPU replay-source는 다른 identity입니다. native worker exit UNKNOWN, kernel Marlin/FLASH_ATTN selected-log, semaphore 경고도 유지합니다. 과거111개 CPU 테스트를 이번 패키징 테스트 수에 합치지 않습니다.

재현: `inference-kit replay --case step04 --bundle STEP04_SMOL_CLIPPING_ABLATION_20261015.zip --output runs/step04-01`. API도 `replay('step04', bundle, fresh_output)`입니다. [summary.json](summary.json)은6arm과 음성 연구 판정을 모두 포함합니다. 원시 HTTP/국소module×source/manifest/recipe/source/failed attempt는 [release index](../../evidence/index.json)에 연결됩니다.

재사용 가치는 같은 packed format의 모듈 출처·국소/최종 효과를 분리하는 진단 기능입니다. 고객 수요/지불 의사, 배포 품질/속도, 새 알고리즘 우위는 확인되지 않았습니다. STEP04는 research_dev_only/추천 불가입니다. 후속은 업무 품질 adapter 하나이며 clipping 탐색을 이번에 재개하지 않습니다.
