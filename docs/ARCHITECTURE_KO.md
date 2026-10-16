# 구조와 의존성

공개 API는 `inference_kit.{plan,verify,replay,compare,assess,demo}`입니다. 입력은 경로 또는 compare/assess에서 명시적인 메모리 객체이며 CLI도 같은 함수를 호출합니다. `core`는 stdlib만 사용합니다. 패키지 import에는 HTTP/tensor/engine SDK가 필요 없습니다.

| 경계 | 역할 | 입력 → 출력 |
|---|---|---|
| core/config, paths | spec v1, 명시적 root, mask, profile 검증 | JSON/local → 미실행 plan |
| core/io, state, evidence, index | immutable raw, atomic state, linked ledger, 안전한 archive와 reference | 파일/bundle → 검증 receipt |
| core/quality, vendor | 기존 likelihood와 공식 acc/acc_norm, paired correctness | raw HTTP/token 경계 → document scores/intervals |
| core/metrics | HELP/TYPE/label 기반 Prometheus, reset/gap 보존 | text/time series → 범위가 있는 통계 |
| core/assess, assess_multi | 기존 v2와 v3 정책 | bound index + 요구조건 → PASS/FAIL/UNKNOWN와 후보 |
| adapters/http, protocol, owned | SSE 경계, 명시적 HTTP, 소유 group 정리 | local endpoint/argv → raw records/lifecycle |
| adapters/nvml, vllm, run | opt-in 단일 서버/장비 관측/시간 제한 | prepared local inputs → 새 attempt |
| adapters/tensors | 검증 형식의 정확한 module bundle 교체 | 두 artifact + receipt → 새 artifact/출처 map |
| experiments/step02~04 | 각 고정 실험 재계산 | 원본 bundle → 별도 derived output |

`Path(__file__).parents[...]`, 원본 `tools` import, sys.path sibling 주입을 제거했습니다. demo만 `importlib.resources`로 설치된 synthetic resource를 사용합니다. diova-compare core 0.1.1을 명시적 vendor namespace에 포함했습니다.

GPU `run`은 optional HTTP client와 외부 server Python을 분리합니다. server runtime tuple이 다르면 거절하고 설치하지 않습니다. CUDA 초기화는 이 외부 opt-in 서버에서만 일어날 수 있습니다. `plan`은 작은 config/tokenizer identity만 검사하며 server를 실행하지 않습니다. `run`은 외부 프로세스가 보이면 요청을 중단하고 자기 group만 정리합니다. WSL의 process 관측 한계는 남습니다.

배포는 단일 Python 패키지이며 DB/UI/telemetry는 없습니다. 큰 raw, GPU 당시 execution source, 후속 correction source는 release 자산에 남습니다. 새 패키지는 이 역사적 source hash를 대신하지 않습니다.
# BENCH05-A 추가 경계

`probes.contracts/blobs`와 `metrics`는 stdlib CPU 계층이다. `models.capabilities`는 model+artifact+runtime별 준비 가능성과 실측을 구분한다. `suites.bench05a_*`가 계획·owned 순차 실행·재계산을 연결한다. Torch/Transformers/CT는 명시적으로 실행한 `probes.local_transformer/weights/attention` worker 안에서만 사용한다. 기존 experiments와 결과 schema를 새 계층으로 교체하지 않는다. [관측 좌표·실행 방식·재사용](BENCH05A_KO.md).
