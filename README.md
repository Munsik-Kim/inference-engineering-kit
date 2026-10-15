# inference-engineering-kit 0.1.0

LLM 추론 실험의 실행·계측·품질 비교·packed artifact 진단을 재사용하고, 저장된 원시 기록에서 결과를 다시 계산하는 Python toolkit.

모델/recipe/profile과 로컬 경로를 분리하고, 요청 완료·품질·프로세스 정리·kernel 관측·배포 근거를 각각 기록합니다. 원본 연구 저장소 없이 설치해 CPU 검산과 비교를 수행할 수 있습니다. 기본 설치에는 Torch/vLLM/Transformers가 없습니다. 내부 비공개 사용을 위한 첫 엔지니어링 릴리스입니다.

English: A small Python toolkit for portable inference experiment plumbing, raw-record replay, paired comparison, constraint assessment and packed-tensor provenance. CPU portability is verified separately from historical GPU experiments. No new accuracy/speed claim is made.

## 설치와 CPU Quickstart

Python 3.10 이상을 대상으로 하며 이번 로컬 검증은 Linux/WSL Python 3.12에서 수행했습니다. 릴리스의 wheel을 인증된 계정으로 명시적으로 받거나 이 저장소를 clone해 wheel을 만드세요. 자동 다운로드는 없습니다.

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

`run --allow-gpu`는 명시적 spec/local/runtime/GPU UUID/output/budget과 기존 로컬 artifact를 요구하는 opt-in 실행 경로입니다. `compose --allow-tensor-runtime`은 준비된 별도 tensor runtime에서만 작동합니다. 이번 릴리스에서는 실제 모델과 GPU를 실행하지 않았습니다. vLLM 실행 경로 상태는 `REFACTORED_CPU_CONTRACT_VERIFIED_GPU_NOT_RERUN`입니다.

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
