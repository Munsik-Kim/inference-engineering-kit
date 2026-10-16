# 0.1.0 릴리스의 지원·검증 수준

게시 상태 안내: 아래 미게시·CI NOT_RUN 문구는 각 로컬 task 종료 당시 기록입니다. 이번 dev3의 원격 게시와 실제 3버전 실행 상태는 [CPU workflow](https://github.com/Munsik-Kim/inference-engineering-kit/actions/workflows/cpu.yml)의 run/commit별 결과를 확인합니다. 기존 릴리스 자산과 historical GPU source identity는 유지합니다.

| 영역 | Historical GPU evidence | 이번 릴리스 CPU 검증 | 미검증/미이식 |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 BF16/GPTQ W4 | STEP02, STEP03 | raw response 재채점·timing/selector parity | 새 wheel GPU 재실행 없음 |
| SmolLM2-1.7B-Instruct BF16/RTN/GPTQ | STEP02/04 | raw likelihood와 pairing/contrast parity | 다른 모델·다른 recipe 미검증 |
| Smol R00/R10/R01/R11 | STEP04 loader/quality | tiny compose tensor 출처/저장·재로딩, full raw replay | 새로운 실제 artifact 생성/forward 없음 |
| vLLM managed server | 과거 0.29.0/Torch2.13.0/Transformers5.17.0/CT0.17.0 | argv/identity/budget/owned CPU contracts | maintained GPU NOT_RERUN |
| tensor adapter runtime | build CT0.18.0의 packed 표현 | 기존 별도 runtime에서 tiny CPU fixtures | 기본 설치 시 torch/CT 없음 |
| SGLang | 이전 lab transport 자산 | synthetic HTTP parser 계약 | kit managed SGLang run 미지원 |
| CPU interpreter | 해당 없음 | Python3.10/3.11/3.12 Linux/WSL CPU 계약·일반 wheel 설치 | 다른 OS·Python3.13+ 미검증 |
| CI | 해당 없음 | 기존 main의 Python3.12 수동 run 성공 | 새 3버전 matrix는 로컬 검증, 원격 NOT_RUN |

Python 지원 선언은 >=3.10입니다. 설치 호환성 전체가 실측되었다는 뜻은 아닙니다. base deps는 stdlib, `http` extra만 aiohttp입니다. GPU 패키지 설치/upgrade/자동 다운로드는 하지 않습니다. GPU runtime 경로를 symlink resolve로 대체하지 않습니다.

run은 준비된 runtime tuple, qwen3/llama, native BF16/검증된 pack-quantized scheme만 받습니다. 정확한 artifact identity·tokenizer와 profile이 맞아도 새 GPU 재검증을 생략했다는 상태는 유지합니다. source packaging의 성공을 GPU 품질/연구 우위/배포 적합성으로 승격하지 않습니다.

원본 runtime 경로/버전/argv는 release historical receipt를 참조합니다. 개인 경로를 포함한 pip freeze를 새 lockfile로 배포하지 않습니다. CPU build 의존성은 전용 격리 venv에만 설치했으며 기존 연구 환경은 수정하지 않았습니다.

## BENCH05-A 로컬 0.1.1.dev2 추가 검증 (2026-10-15)

| 경로 | 이번 실측 | 해석 범위 |
|---|---|---|
| 새 wheel `bench run` → `probes.local_transformer` | RTX5080, Smol pinned BF16/RTN, smoke2+main2 GPU worker | BF16는 INSTRUMENTED_PATH, RTN은 DECODED_WEIGHT_DIAGNOSTIC |
| 선택 attention 행 | native post-RoPE Q/K의 FP32 재구성, layer12/head0·16, arm당16행 | native fused 확률 tensor 반환 아님; live MHA만 |
| attention output | 원래 SDPA 반환 AV 및 o_proj 후 vector, 별도 재구성 AV | 선택 좌표만; 모든 layer/모델 일반화 아님 |
| vLLM bridge2 server | BF16 및 원래 packed RTN, arm당4개 echo 요청 | 실제 Marlin 선택 로그, profiler 아님; 경로 간 logprob/argmax 차이 존재 |
| OFF/OFF/ON | 첫 영어2문항, 두 arm 모두 target/logit 최대 차이0 | 같은 프로세스/모델의 작은 제어 실험만 |
| `bench plan/replay` | 새 일반 설치 wheel, checkout 읽기 차단, SDK/NumPy 미수입 | raw CPU 재계산, GPU 재실행 아님 |

진단 runtime은 기존 Python/Torch2.13.0/Transformers5.14.1/CT0.18.0을 재사용했습니다. RTN의168Linear를 실제 library decode 후 strict 적재했으며 새 양자화나 모델 다운로드는 없습니다. bridge는 별도 기존 vLLM0.29.0/Transformers5.17.0/CT0.17.0 runtime을 사용했습니다. 이 두 경로의 수치 동등성은 주장하지 않습니다. worker native exit UNKNOWN, owned cleanup과 parent exit는 별도 기록했습니다.

이 추가 검증으로 0.1.0의 일반 `run`·compose·과거 STEP02~04 전체를 새 GPU 실행한 것으로 승격하지 않습니다. 이번 변경은 로컬 개발본이며 remote push/release/CI dispatch는 없습니다. [계약과 사용법](BENCH05A_KO.md)을 참조하세요.

## CPU 배포 수정본 0.1.1.dev3 (2026-10-16)

한글 JSON/JSONL·CSV·Markdown 파일은 UTF-8로 읽고 쓴다. CLI JSON은 ASCII escape를 사용하여 ASCII stdout에서도 같은 값을 전달한다. 테스트 0건은 실패이며 영수증에 실제 import 위치·버전·수집 상태·허용한 CI 문맥만 남긴다. 기본 CPU import는 ML SDK를 요구하지 않는다.

Python3.10.20/3.11.15/3.12.3에서 현재 123개 CPU 계약을 확인했다. 선택 tensor 의존성 유무에 따라 skip 수를 분리한다. 별도 일반 wheel 설치·소비자·원시 재계산의 상세 결과는 [CPU 배포 검증](CPU_RELEASE_FIX_KO.md)을 따른다.

기존 v0.1.0 main d6c0b4be의 [run 38705852280](https://github.com/Munsik-Kim/inference-engineering-kit/actions/runs/38705852280)은 Python3.12에서 성공했다. 이 기록을 새 수정본의 원격 검증으로 해석하지 않는다. 새 workflow는 수동 3버전 matrix, source 및 설치 wheel 계약, 외부 console 소비자, sdist 재빌드의 package bytes 대조, 7일 receipt 보관을 추가했다. checkout/setup-python은 공식 태그의 Node24 실행 SHA로 갱신했다. 새 workflow의 원격 실행은 NOT_RUN이다.

수정본 GPU 경로는 CPU 계약만 회귀 검증했다. BENCH05-A 0.1.1.dev2의 실제 GPU 관측과 소스 identity는 그대로 보존하며 dev3에서 측정했다고 바꾸지 않는다. 기존 연구 HOLD_CLIPPING_VARIANT·품질 INCONCLUSIVE도 유지한다.
