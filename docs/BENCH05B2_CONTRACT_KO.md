# BENCH05-B2-CONTRACT

로컬 dev5의 고정 prompt 비교 recipe다. tech_record_extract 0.1의 gold·strict scorer는 바꾸지 않는다. B1의 원래 instruction에도 evidence object/행 ID/enum/null 규칙이 있다. B2는 누락 수정이 아니라 추가 명시화·강조 block의 효과이며 길이·위치 효과를 완전히 분리하지 않는다.

`tasks.contract_policy`의 P00/P10/P01/P11은 evidence와 status block의 2×2다. P00는 B1 render_prompt를 그대로 호출하고, P10은 E, P01은 S, P11은 동일 E→S를 추가한다. 새 문서에는 새 사실·표현·행 배치를 쓰며 같은 계열의 작은 합성24문서라는 한계를 유지한다. model-visible instruction에 gold는 없다. canonical gold는 teacher-forced 진단에만 쓴다.

준비된 기존 로컬 tokenizer/runtime으로 실행하는 명시적 `python -m inference_kit.probes.contract_tokens --help`는 모델을 로딩하지 않는다. B2의 prompt512/ref128/joint640 profile은 B1의384/128/512와 A의128을 변경하지 않는다. offset/field span/선택 attention collector는 재사용한다. 실제 artifact나 runtime이 없으면 다운로드 대신 차단한다.

```bash
inference-kit bench b2 plan --suite configs/bench05b2/suite.json --models configs/bench05a/pair.json --local /path/to/private/local.json --output new-plan
inference-kit bench b2 run --plan new-plan/PLAN_FROZEN.json --output new-run --allow-gpu
inference-kit bench b2 replay --run new-run --output new-derived
inference-kit bench b2 score --sample one-sample.json --generation one-generation.json --output new-score
```

Python API는 `bench_b2_plan/run/replay/score`다. plan/replay/score/base import는 Torch/vLLM/TF/NumPy/aiohttp를 필요로 하지 않는다. run만 기존 HTTP runtime/명시적 GPU opt-in을 사용한다. 상대 경로는 설정의 명시적 root로 해석한다. 같은 output의 덮어쓰기를 거절한다.

모델 서버는 BF16/실제 packed RTN 각1회다. document별 policy 순서를 균형 회전하며 양 arm에 같은112요청 순서를 적용한다. 2×4×(SMOKE4+EVAL24)=224POST, 본 평가192POST지만 고유 평가문서는24개다. 출력 상한은43,008token이다. 서버를 공유하는 policy별 VRAM을 독립 측정으로 부르지 않는다. 지연은 C1 단일 서버 기술통계다.

`bench05b2_replay`는 저장 SSE를 검산하고 같은 scorer로 재채점한다. enum_valid·evidence 키/타입/행 존재·core values는 추가 진단이며 strict task 성공을 대체하지 않는다. source/gold/scorer/runtime/tokenizer identity를 검사한 `declared_prompt_intervention`에서만 다른 prompt를 비교한다. 같은 policy 안의 weight 비교는 prompt도 같아야 한다. E/S 효과, 상호작용, P11−P00와 각 policy의 RTN−BF16을 모두 남긴다. NOT_RUN을 오답·완료로 바꾸거나 분모에서 숨기지 않는다.

P00/P11만 사전 선택4문서의 추가 관측을 허용한다. 각 policy 안에서만 BF16↔RTN token/attention 차이를 계산한다. 서로 다른 prefix/key axis를 맞춰 JS를 만들지 않는다. RTN 내부는 DECODED_WEIGHT_DIAGNOSTIC이고 실제 packed SERVE와 동등성을 주장하지 않는다. 이번 결과·예산·실행/미실행 범위는 timestamped REPORT/receipt가 근거이며 과거 A/B1 관측으로 coverage를 채우지 않는다.

새 policy를 추가하려면 명시적 E/S 선언·고정 문구·identity·순서·새 데이터/실행 계획과 회귀가 필요하다. 현재 recipe는 위 네 policy만 지원한다. 이번 비교를 마치면 점수와 무관하게 종료한다. 다음 후보는 기존 ARC 과제를 동일 TaskAdapter/Sample/Metric 경로에 연결하는 한 단계이며 자동 실행하지 않는다.
