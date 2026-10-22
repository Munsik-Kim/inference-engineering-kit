# 보존된 개발·배포 이력

아래는 과거 버전의 원문 기록이다. 0.2.0의 현재 수용시험 결과나 설치 명령으로 소급하지 않는다. 역사적 NOT_RUN/PARTIAL/STALE와 원래 source는 보존한다.

## 보존된 역사적 개발·배포 기록

아래의 날짜/버전별 실행·설치 예는 원래 기록입니다. 과거 release 다운로드 명령은 이번 후보의 설치 절차가 아닙니다. 새 후보의 CPU/실측 검증으로 소급하지 않습니다.


LLM 추론 실험의 실행·계측·품질 비교·packed artifact 진단을 재사용하고, 저장된 원시 기록에서 결과를 다시 계산하는 Python toolkit.

모델/recipe/profile과 로컬 경로를 분리하고, 요청 완료·품질·프로세스 정리·kernel 관측·배포 근거를 각각 기록합니다. 원본 연구 저장소 없이 설치해 CPU 검산과 비교를 수행할 수 있습니다. 기본 설치에는 Torch/vLLM/Transformers가 없습니다. 내부 비공개 사용을 위한 첫 엔지니어링 릴리스입니다.

English: A small Python toolkit for portable inference experiment plumbing, raw-record replay, paired comparison, constraint assessment and packed-tensor provenance. The development build includes selected token/attention diagnostics, historically measured with 0.1.1.dev2 on one Smol BF16/decoded-RTN pair, and CPU distribution fixes in 0.1.1.dev3. No accuracy/speed or production claim is made. The private 0.1.0 release remains unchanged.

로컬 C2는 보존된 세 task의 준비 토큰 입력을 설정 기반 token-only worker에 연결합니다. Smol BF16/decoded RTN의 새 6문항 관측은 기존 C1 TokenView와 비교기로 CPU 재계산되며, 새 attention·서빙·업무 정확도 평가는 포함하지 않습니다. [C2 계약과 실제 명령](docs/BENCH05C2_KO.md)을 참조하세요. 이 개발본의 원격 게시·CI는 NOT_RUN입니다.

로컬 D1은 같은 준비 입력과 token 좌표에 설정 기반 다층 attention·native SDPA AV·o_proj 관측을 연결합니다. OFF1/OFF2/ON 대조와 raw CPU 재계산을 분리하며, RTN은 decoded BF16 진단 경로입니다. [D1 계약](docs/BENCH05D1_KO.md)을 따르고 실제 완료 범위는 timestamp report의 receipt로 확인합니다. 새 업무 점수·packed kernel 동등성·임의 모델 지원을 주장하지 않습니다.

## BENCH05-B1: 업무 JSON 생성·채점·진단 연결

로컬 dev4에는 `tech_record_extract v0.1` 과제 하나를 추가했습니다. 직접 작성한 SMOKE4/DEV24 문서의 자연 생성 응답을 엄격히 채점하고, 응답을 보기 전에 선택한4문항의 고정 참조를 기존 token/attention collector에 연결합니다. 실제 packed RTN 서비스와 decoded 진단은 서로 다른 실행입니다. [B1 계약·사용법](docs/BENCH05B1_KO.md)을 따르며 B2 이후 과제나 배포 품질 인증을 포함하지 않습니다. 현재 B1은 원격 CI/게시하지 않았습니다.

```bash
inference-kit bench b1 --help
inference-kit bench b1 plan --suite configs/bench05b1/suite.json --models configs/bench05a/pair.json --local /path/to/private-b1-local.json --output runs/b1-plan-new
inference-kit bench b1 replay --run /path/to/b1-recorded-run --output runs/b1-replay-new
```

## CPU 배포 수정

`0.1.1.dev3`는 UTF-8 파일 처리, 테스트0건 실패 판정, 설치 wheel 검증과 3버전 CPU workflow를 보강합니다. Python3.10/3.11/3.12에서 각각 source·wheel123개 계약을 검증했고, 깨끗한 CPU venv의 demo/compare/assess/recorded replay도 통과했습니다. 원격 실행 상태와 wheel 자산은 [수동 CPU workflow](https://github.com/Munsik-Kim/inference-engineering-kit/actions/workflows/cpu.yml)에서 확인합니다. [검증 범위](docs/CPU_RELEASE_FIX_KO.md)의 로컬 task 기록과 원격 실행을 구분합니다. 게시 과정에서 발견한 합성 fixture의 호스트 자원 의존성도 수정했습니다. 현재124개 계약과 첫 CI 실패 보존은 [게시 검증 기록](docs/CPU_PUBLICATION_KO.md)을 따릅니다. 기존 v0.1.0 릴리스는 유지합니다.

## BENCH05-A: 실제 토큰·attention 관측

현재 `0.1.1.dev3`의 `bench` 이름공간은 설정 동결 → 순차 진단 → 원시 numeric blob의 CPU 재계산을 연결합니다. 아래 실제 SmolLM2-1.7B GPU 관측은 이전 `0.1.1.dev2`에서 같은 pinned tokenizer/입력으로 BF16과 기존 RTN을 실행한 기록입니다. RTN은 실제 packed/scale/shape를 해석한 **DECODED_WEIGHT_DIAGNOSTIC**입니다. INT4 kernel 내부 관측이나 INT4 VRAM 측정으로 해석하지 않습니다. 확률 행은 native post-RoPE Q/K에서 재구성하고, attention output은 원래 SDPA 반환값에서 읽습니다.

```bash
# 준비된 로컬 wheel만 일반 설치합니다. 의존성/모델 다운로드는 없습니다.
python -m pip install --no-index --no-deps /path/to/inference_engineering_kit-0.1.1.dev3-py3-none-any.whl
inference-kit bench --help
inference-kit bench plan --suite configs/bench05a/suite.json --models configs/bench05a/pair.json --local /path/to/private-local.json --output runs/new-plan
inference-kit bench replay --run /path/to/recorded-bench05a-run --output runs/new-replay
```

`plan`에는 기존 artifact receipt·준비된 token manifest·명시적 runtime·연결된 campaign ledger가 필요합니다. 모델을 로딩하거나 GPU를 확인한 것으로 표시하지 않습니다. `bench run --plan runs/new-plan/PLAN_FROZEN.json --output runs/new-run --allow-gpu`만 실제 GPU 작업을 시작합니다. source/config가 동결 이후 바뀌거나 외부 compute PID/예산 문제가 있으면 중단합니다. 자동 재시도는 없습니다.

```python
from inference_kit import bench_plan, bench_replay

planned = bench_plan("suite.json", "pair.json", "private-local.json", "new-plan")
observed = bench_replay("recorded-bench05a-run", "new-replay")
```

전체 관측에는 4개 합성 문항×2 arm, arm당 참조 23위치·선택 attention 16행·full vocabulary 15위치가 있습니다. 이 짧은 continuation NLL/attention 차이는 일반 품질·언어 능력·사고 과정의 정답이 아닙니다. 별도 vLLM bridge는 경로 간 차이를 측정했으며 동등성을 주장하지 않습니다. 이 기능만 새 wheel GPU 검증 범위이며 기존 `run` 전체를 재인증하지 않습니다. 세부 계약·재현과 작은 실제 fixture 사용은 [BENCH05-A 안내](docs/BENCH05A_KO.md)를 따릅니다. 당시 GPU raw와 dev2 wheel은 로컬에 보존합니다. 이번 dev3 소스 게시에서는 GPU를 재실행하지 않으며 기존 원격 release를 변경하지 않습니다.

## 설치와 CPU Quickstart

Python 3.10 이상을 대상으로 하며 Linux/WSL Python 3.10·3.11·3.12에서 CPU 검증했습니다. 아래는 보존한 v0.1.0 release의 Quickstart입니다. 현재 dev3 wheel은 위 CPU workflow의 명시적 artifact 다운로드나 검토한 로컬 build로 준비합니다. 릴리스의 wheel을 인증된 계정으로 명시적으로 받거나 이 저장소를 clone해 wheel을 만드세요. 자동 다운로드는 없습니다.

```bash
gh release download v0.1.0-internal --repo Munsik-Kim/inference-engineering-kit --pattern '*.whl' --dir ./downloads
python3 -m venv .venv
.venv/bin/python -m pip install ./downloads/inference_engineering_kit-0.1.0-py3-none-any.whl
.venv/bin/inference-kit --help
.venv/bin/inference-kit demo --output ./runs/demo-01
.venv/bin/inference-kit compare --baseline ./runs/demo-01/baseline.json --candidate ./runs/demo-01/candidate.json --output ./runs/compare-01
.venv/bin/inference-kit assess --index ./runs/demo-01/index.json --requirements ./runs/demo-01/requirements.json --output ./runs/assess-01
```

Draft release는 권한 있는 계정에만 보입니다. 업로드 자산 목록/해시는 [evidence/index.json](evidence/index.json)을 확인하세요. `demo`는 wheel resource의 자체 synthetic 기록을 사용하며 추론 성능을 측정하지 않습니다. 예상 판정은 `INSUFFICIENT_EVIDENCE`입니다. 같은 output 재사용은 거절합니다.

저장소에서 제공하는 작은 과거 기록은 명시적 bundle 입력으로 검산합니다. 두 문항 slice이며 전체 실험 재현으로 부르지 않습니다.

```bash
inference-kit plan --spec configs/examples/cpu-spec.json --local configs/examples/local.example.json
inference-kit verify --bundle tests/fixtures/recorded-step04 --output runs/verify-01
inference-kit replay --case step04 --bundle tests/fixtures/recorded-step04 --output runs/recorded-01
python examples/consumer_project.py --output runs/consumer-01
```

## API와 CLI

```python
from inference_kit import compare, assess

comparison = compare("baseline.json", "candidate.json", output="new-comparison")
assessment = assess("portable-index.json", "requirements.json", output="new-assessment")
```

CLI는 같은 Python API를 호출합니다. `plan/help/demo/verify/replay/compare/assess`는 GPU·모델 로딩·네트워크를 시작하지 않습니다. `verify`는 bounded archive/manifest/index를 확인하고, `replay`는 설치된 코드로 원시 응답을 다시 채점합니다. archive의 Python은 자동 실행하지 않습니다. 전체 STEP02~04 자산 사용법은 [재현 안내](docs/REPRODUCIBILITY_KO.md)에 있습니다.

`run --allow-gpu`는 명시적 spec/local/runtime/GPU UUID/output/budget과 기존 로컬 artifact를 요구하는 opt-in 실행 경로입니다. `compose --allow-tensor-runtime`은 준비된 별도 tensor runtime에서만 작동합니다. **0.1.0 릴리스**에서는 실제 모델과 GPU를 실행하지 않았습니다. 그 일반 `run` 경로의 상태는 `REFACTORED_CPU_CONTRACT_VERIFIED_GPU_NOT_RERUN`입니다. BENCH05-A의 제한된 새 관측은 위 별도 범위입니다.

| 기능 | 현재 검증 | 경계 |
|---|---|---|
| 독립 wheel CPU demo/compare/assess/plan | 외부 consumer에서 검증 | synthetic은 실측이 아님 |
| STEP02/03/04 replay | 전체 공개 원시 subset 응답·수치 parity | 과거 GPU source와 새 소스 구분 |
| SSE/owned lifecycle/NVML/Prometheus | CPU 계약·loopback·subprocess | 새 GPU 관측 없음 |
| 다중 workload AND 선택기 | STEP02 v2/STEP03 v3 | retained quality는 조건부, missing은 UNKNOWN |
| packed module compose | tiny CPU tensor 저장·재로딩 | symmetric INT4 group128 pack-quantized adapter만 |
| vLLM run | CPU launch 계약/opt-in 검증 | historical Qwen3/SmolLM2 경로, 새 wheel GPU 미실행 |

## 코드와 근거의 흐름

`maintained source → frozen recipe → immutable raw → derived comparison → assessment`

새 코드는 [src/inference_kit](src/inference_kit)에 있고, 고정 recipe는 [experiments](src/inference_kit/experiments)에 있습니다. 과거 raw와 당시 source/config/실패 attempt는 별도 draft release 자산에 있습니다. Git에는 작은 recorded fixture·요약·출처를 둡니다. 원래 `/home/...` 경로는 historical metadata이며 현재 파일 lookup으로 사용하지 않습니다.

대표 사례:

- [STEP02](cases/step02/README.md): 두 모델·여러 weight 방법을 공통 HTTP/likelihood 계약에 연결하고 기존 실패와 후속 재측정을 구분. [채점/paired API](src/inference_kit/core/quality.py).
- [STEP03](cases/step03/README.md): weight×KV pool 2×2의 실제 설정/대기/메모리 검증과 여러 업무 조건 AND screen. [재계산](src/inference_kit/experiments/step03.py), [선택기](src/inference_kit/core/assess_multi.py).
- [STEP04](cases/step04/README.md): packed/scale/shape 출처를 보존한 모듈 개입, 독립 입력에서 국소/최종 품질 분리. 결론 `HOLD_CLIPPING_VARIANT` 보존. [tensor adapter](src/inference_kit/adapters/tensors.py), [분석](src/inference_kit/experiments/step04.py).

## 지원과 제한

임의 모델·engine·quantization을 지원한다고 주장하지 않습니다. ARC subset의 acc_norm과 acc를 구분하고, point/interval 정책, retained/exact-profile 품질, whole-device/process/allocator memory를 섞지 않습니다. worker native exit는 `UNKNOWN`, kernel은 선택 로그 수준이며 profiler 검증이 아닙니다. STEP04는 `research_dev_only`로 배포 추천 대상이 아닙니다. 8GiB screen은 실제 8GiB GPU 검증이 아닙니다.

[구조](docs/ARCHITECTURE_KO.md) · [재사용 cookbook](docs/REUSE_COOKBOOK_KO.md) · [데이터 계약](docs/DATA_CONTRACTS_KO.md) · [지원표](docs/SUPPORT_MATRIX_KO.md) · [이식/source-map](docs/MIGRATION_KO.md) · [출처/라이선스](THIRD_PARTY_NOTICES.md)

테스트는 `python -B tools/test_cpu.py --output <새 receipt.json>`으로 실행합니다. HTTP fixture 테스트에는 선택 의존성 `.[http]`가 필요합니다. tensor 테스트는 기존 별도 runtime이 없으면 명시적 skip입니다. GitHub Actions는 `workflow_dispatch`만 제공하며 이번 게시에서 실행하지 않았습니다 (`CI_NOT_RUN`).

BENCH05-B2-CONTRACT의 로컬 dev5 추가 기능은 [고정 evidence×enum 정책 비교](docs/BENCH05B2_CONTRACT_KO.md)를 참조한다. B1 scorer와 과거 결과를 유지하며 `bench b2 plan/run/replay/score`로 새 합성 문서의 8개 cell을 연결한다. 새 B2 원격 CI는 NOT_RUN이고, 실제 실행 범위는 로컬 timestamped receipt로 구분한다.

BENCH05-B3 로컬 개발본(0.1.1.dev6)은 ARC `choice_loglikelihood`와 기존 tech_record `generate_until`을 명시적 TaskAdapter로 연결한다. 원시 HTTP와 선택 token/attention의 재계산 계약·실행 범위는 [BENCH05B3_KO.md](docs/BENCH05B3_KO.md)에 설명한다. 기존 test subset 재사용이며 새 품질 인증이 아니다. B3 원격 CI/게시를 수행하지 않는다.


BENCH05-B4 로컬 dev7은 세 번째 `context_evidence_retrieval/0.1` TaskAdapter를 기존 generate_until 실행·선택 token/attention·CPU replay에 연결한다. 새로운 요청 종류나 일반 RAG 서비스는 아니다. 사용 계약은 [B4 문맥·근거 검색](docs/BENCH05B4_KO.md), 고정 실측은 `reports/bench05b4_context_evidence_20261019_105257/`를 따른다.


BENCH05-C1 개발판 0.1.1.dev8: `bench tokens export/compare`로 세 task의 보존 token·선택 full logits·자유 생성 분기를 CPU에서 분석한다. Native/diagnostic과 policy·gold scope를 분리하며 새 GPU 실행은 없다. 계약·설치·한계는 [BENCH05C1_KO](docs/BENCH05C1_KO.md)를 참고한다.
