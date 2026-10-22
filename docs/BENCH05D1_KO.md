# BENCH05-D1: 설정 기반 여러 층의 attention·출력 진단

D1은 C2에서 동결한 세 task × 두 문서의 준비 입력을 그대로 사용해, 선택 layer/head/query의 저장 관측을 CPU에서 비교하는 기능입니다. 새 문서·tokenizer 준비·서빙·업무 품질 평가는 추가하지 않습니다. 기존 token-only `bench tokens plan/run`의 attention OFF 기본값은 유지합니다.

`StructureProbeSpec`의 MAIN은 실제 model config로 첫 층·중간 층·마지막 층, 첫 query head·중간 query head, `last_prompt`·`before_last_reference` query를 정합니다. 동일한 물리 query가 두 role에 해당하면 한 번 저장하고 alias를 남깁니다. SMALL은 중간 층·head 0·last_prompt의 CPU 계획만 확인합니다. 선택 좌표나 허용차를 모델 결과에 따라 변경하지 않습니다.

## 실제 인터페이스

```bash
inference-kit bench structure --help
inference-kit bench structure plan \
  --spec configs/bench05d1/main.json \
  --inputs /path/to/C2/PROBE_INPUTS_FROZEN.json \
  --models configs/bench05a/pair.json --local /private/local.json \
  --output /new/d1-plan
inference-kit bench structure run \
  --plan /new/d1-plan/PLAN_FROZEN.json --output /new/d1-run --allow-gpu
inference-kit bench structure replay --run /new/d1-run --output /new/d1-replay
inference-kit bench structure replay --run /path/to/D1_REPRO.zip --output /new/d1-zip-replay
```

Plan은 prepared sequence·config·원본 identity·ledger·자원 한도를 CPU에서 검사합니다. 위 입력 경로는 자리표시자이며 실제 사용 경로와 실행 영수증은 각 run의 `HANDOFF_KO.md`에 기록합니다. 실제 local 파일은 배포하지 않습니다. Run만 명시적 GPU 실행이며 replay는 저장된 raw/numeric blob만 읽습니다.

Python API도 같은 implementation을 사용합니다.

```python
from inference_kit import structure_plan, structure_run, structure_replay

# 준비된 입력·모델·local 설정의 명시적 CPU 계획
# frozen = structure_plan(spec, inputs, models, local, new_plan)
# structure_run(new_plan / "PLAN_FROZEN.json", new_run)  # 명시적 GPU 작업
summary = structure_replay("/path/to/D1_REPRO.zip", "/new/cpu-replay")
```

## 관측의 의미

한 worker가 모델을 한 번 적재하고 각 문서에서 OFF1 → OFF2 → ON을 수행합니다. MAIN의 ON만 주 TokenView와 structure 비교에 들어가며 OFF 두 pass는 관측 간섭 확인용입니다. 두 arm·6문서·3 pass는 36 forward이고 36개의 독립 문서가 아닙니다.

| 관측 | 의미 | 축 |
|---|---|---|
| attention_row | native post-RoPE Q/K·실제 scale/mask로 선택 행만 FP32 재구성 | layer/head/query × 전체 key |
| attention_output | 원래 SDPA를 한 번 호출한 실제 native 반환 AV | layer/head/query × head dimension |
| reconstructed_AV | 저장할 FP32 확률 행과 native V의 별도 곱 | layer/head/query × head dimension |
| output_projection | 실제 o_proj 반환, 모든 head 결합 후·residual 덧셈 전 | layer/query × hidden dimension, head 축 없음 |

현재 Smol MAIN의 layer는 0/12/23, head는 0/16입니다. 문서별 두 query가 서로 다르면 arm당 attention 행/native AV/reconstructed AV 각각 72개, o_proj 36개입니다. 실제 count·중복 alias·미수집 상태는 frozen selection과 raw를 기준으로 보고합니다. Token은 C2 MAIN의 선택 110·full 44 좌표를 pass마다 유지합니다. 확률 행을 fused kernel이 직접 반환했다고 표현하지 않습니다.

원래 SDPA dispatch는 한 개만 임시 교체하며 module identity로 선택 layer를 구분합니다. 선택 o_proj hook만 설치하고 정상 종료·등록 실패·callback/forward 예외에 복원합니다. 소비자는 선택 부분의 detached CPU 복사본을 받습니다. 전체 `[layers, heads, T, T]` 확률이나 전체 `[T,V]` logits를 만들고 잘라 저장하는 경로를 추가하지 않습니다.

BF16은 `INSTRUMENTED_PATH`, RTN은 실제 packed artifact를 엄격하게 decode한 `DECODED_WEIGHT_DIAGNOSTIC/effective BF16`입니다. Native AV는 현재 진단 runtime의 SDPA 반환이며 vLLM/Marlin 내부 출력이 아닙니다. RTN 진단 메모리는 INT4 서빙 메모리 절감 측정이 아닙니다.

## 고정된 검사와 CPU 재계산

- OFF1↔OFF2 및 OFF1↔ON: 선택 target logprob와 선택 full logits의 최대 절대 차이 `1e-5`.
- 같은 pass의 저장 scalar↔저장 FP32 logits에서 CPU log-softmax[target]: `2e-5 nat`.
- 동일 raw의 CPU metric parity: absolute `1e-10`, relative `1e-9`; ID/count/status/null/type/order는 exact.
- 구조 비교는 기존 attention L1/JS/entropy 및 vector NMSE/cosine 정의를 재사용합니다. Reference energy `≤1e-12`의 정의되지 않은 값을 정상 0이나 cosine 1로 채우지 않습니다.
- o_proj는 별도 layer/query 표입니다. 같은-arm reconstructed/native AV 차이는 BF16↔RTN 차이와 별도 구현 진단입니다.
- Token ON은 기존 C1 `tokens_export`/`tokens_compare`를 사용합니다. OFF pass를 token 분모에 합치지 않습니다.
- 근거 content span은 이미 검증된 metadata가 있을 때만 사용합니다. NONE/no-answer는 `NOT_APPLICABLE`, 부재 metadata는 `MISSING_SPAN_METADATA`이며 mass 0으로 바꾸지 않습니다.

모델 한 쌍의 end-to-end hidden 차이를 비교하므로 특정 층이 task 오류의 원인이라는 주장은 하지 않습니다. 여러 head/query를 여러 독립 문서로 세거나 서로 다른 task의 지표를 단일 품질 점수로 평균하지 않습니다. 새 배포 동등성·task 정답·인과 기전은 별도 미평가 상태이며 기존 `HOLD_CLIPPING_VARIANT`와 `INCONCLUSIVE`를 유지합니다.

## 독립 설치 소비자 검사

`tools/bench05d1_consumer_check.py`는 새 base CPU venv에 일반 설치한 wheel을 `-I`로 사용합니다. 명시한 raw manifest의 모든 파일을 checkout 밖에 복사·해시 확인한 뒤 원본 checkout 읽기, ML SDK import, 네트워크를 in-process audit hook으로 차단하고 API/CLI replay를 검사합니다. 별도 console subprocess는 같은 파생 파일들의 byte parity를 확인하지만 audit hook을 상속하지 않습니다. OS sandbox 검증으로 표현하지 않습니다.

```bash
/path/to/base-venv/bin/python -I -B tools/bench05d1_consumer_check.py \
  --run /path/to/d1-run --output /new/outside-checkout-consumer \
  --forbidden-root /path/to/toolkit-checkout \
  --forbidden-root /path/to/original-lab --expected-version 0.1.1.dev10 \
  --expected-analysis /path/to/original-d1-replay
```

`--expected-analysis`는 이동 전 결과와 raw 유래 숫자·상태·ID·순서를 비교합니다. 경로에 의존하는 token view fingerprint 및 원본 위치 metadata 세 필드만 명시적으로 정규화합니다. Archive의 Python/pickle은 자동 실행하지 않습니다. 전체 과거 unit suite를 ZIP만으로 실행한다는 주장은 하지 않으며, 배포 영수증에는 전체 신규 raw 포함 범위와 sealed ZIP의 installed-wheel replay 결과를 따로 남깁니다.

D1 remote CI는 `NOT_RUN`입니다. 이 단계는 여러 층의 제한된 관측과 CPU 재계산이며 D 전체·원시 텍스트 준비·새 구조/모델 지원·E dashboard 완료가 아닙니다.
