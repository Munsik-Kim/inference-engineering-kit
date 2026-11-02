<a id="inference-engineering-kit-021"></a>
# AI를 바꾸기 전에, 우리 업무로 먼저 시험하세요.

**Inference Engineering Kit — AI 모델 변경을 위한 품질검사 도구**

더 적은 용량으로 압축한 AI로 바꾸면, 우리 업무도 그대로 잘 처리할까요? Inference Engineering Kit는 지원하는 로컬 AI 모델에 같은 과제를 수행하게 하고, 답변의 정확성·근거·출력 형식을 비교하는 설치형 Python 도구입니다. 원본 모델과 압축 모델의 결과를 나란히 확인하고, 어떤 문항의 답이 달라졌는지 원래 입력과 응답까지 연결해 살펴볼 수 있습니다.

지원하는 세 과제의 **정해진 입력 형식과 외부 정답 기준**에 맞춰 자료를 준비합니다. 자유로운 업무 문서를 넣기만 하면 정답과 평가 기준까지 자동으로 만드는 도구는 아닙니다.

점수만 남기지 않습니다. 실행 조건과 원시 결과를 보관해, 저장된 자료를 **모델 재실행 없이** 다시 채점하고 분석할 수 있습니다. AI 도입·모델 교체를 검토하는 개발팀, 작은 R&D팀, 로컬 모델을 비교하는 연구자를 위한 도구이며, 설치·자료 준비·실행에는 개발 작업이 필요합니다.

현재 정식 사용판은 **v0.2.1**입니다. [제품을 쉬운 말로 살펴보기](docs/PRODUCT_OVERVIEW_KO.md) · [바로 시작하기](START_HERE_KO.md) · [지원 범위](SUPPORT_MATRIX_KO.md)

## 무엇을 확인할 수 있나요?

- **같은 업무의 답 비교:** 사용자가 준비한 자료와 정답 기준으로 원본·압축 구성의 정답, 근거, 형식을 같은 규칙으로 채점합니다.
- **달라진 문항 추적:** 점수와 함께 원래 입력·응답·실행 조건을 보존하고, 지원 경로에서는 토큰 단위 진단을 확인합니다. 토큰은 모델이 문장을 처리하는 작은 단위입니다.
- **저장 결과 재사용:** 기본 CPU 설치로 원시근거 검증, 재채점·비교, 근거 게시, index·오프라인 보고서, 준비 입력의 계획 검사를 수행합니다. 이 경로는 모델을 다시 실행하지 않습니다.

## 현재 지원 범위와 상태

| 대상 | 사용할 수 있는 범위 | 함께 알아둘 점 |
|---|---|---|
| 공통 과제 | 기록의 필드 추출, 주어진 문맥에서 답·근거 찾기, ARC 객관식 | 기존 세 task의 입력·정답 계약을 따릅니다. 임의 업무를 자동으로 채점하지 않습니다. |
| SmolLM2-1.7B-Instruct 특정 BF16/RTN 구성 | 세 task 실행, 선택 토큰·전체 분포 진단 | RTN 내부 진단은 decoded/effective BF16이며 실제 packed 실행 커널 관측과 다릅니다. |
| Smol 선택 구조 진단 | 별도 standalone attention·AV·projection 관측 | 공통 Suite의 임의 모델 구조 분석 기능은 아닙니다. |
| Qwen3-4B-Instruct-2507 특정 BF16/GPTQ 구성 | 세 task 실행, ARC native 토큰 점수 분석 | Qwen 전체 분포·attention·decoded probe는 지원 범위 밖입니다. |

정확한 artifact·tokenizer·runtime·입력 한도와 검증 축은 [지원표](SUPPORT_MATRIX_KO.md)와 [기계 판독 정본](SUPPORT_MATRIX.json)을 따릅니다. 실행 메모리·시간의 보존 사례와 측정 경로도 있지만, 모든 모델·과제의 모든 비용 지표가 하나의 Suite에서 자동 검증된다는 뜻은 아닙니다.

검증은 개발자 자체 수용시험 범위입니다. 일반 모델 품질, 양자화 우위, 배포 동등성, SLA, 독립 고객 성과를 보증하지 않습니다. 미게시 로컬 개발 후보 `0.2.2.dev1`의 기록은 정식 `v0.2.1`과 별도이며, 아래 설치 경로는 정식판만 가리킵니다.

<a id="비공개-배포와-첫-cpu-실행"></a>
## 배포물 받기와 첫 CPU 실행

기존 [공개 v0.2.1 release](https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1)의 **키트 → SHA256 확인 → 작은 CPU 재계산** 순서로 시작하세요. 모델·tokenizer·ML SDK 없이 저장된 응답을 다시 채점할 수 있습니다.

1. [사용 키트 ZIP](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/inference-engineering-kit-0.2.1-kit.zip)과 [SHA256SUMS.txt](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/SHA256SUMS.txt)를 새 폴더에 받습니다.
2. [파일 지문 확인 방법](START_HERE_KO.md#배포물-받기와-sha256-확인)으로 검산하고 새 폴더에 풉니다. `gh` 인증은 공개 다운로드의 필수 조건이 아닙니다.
3. **풀린 키트 루트에서** 다음을 실행합니다. Python 3.10+와 venv/pip가 준비되어 있어야 하고 환경·출력 경로는 새로 만들어야 합니다.

```sh
python3 -m venv ../inference-kit-cpu
../inference-kit-cpu/bin/python -m pip install --no-index --no-deps dist/inference_engineering_kit-0.2.1-py3-none-any.whl
../inference-kit-cpu/bin/python -I -B -m inference_kit bench suite replay --run fixtures/qwen-native --output ../qwen-replay
```

`../qwen-replay/CPU_REPLAY_RECEIPT.json`에서 `CPU_SLICE_REPLAYED`와 검사 범위를, `score_task/summary.json`에서 재채점 결과를 읽습니다. 이것은 키트에 포함된 작은 Qwen 응답 기록의 CPU 재계산이며 새 모델 요청이 아닙니다. 전체 raw·전체 어휘 분포·과거 서버 종료까지 재현하는 명령은 아닙니다.

[다음 CPU 예제와 정확한 분모](QUICKSTART_CPU_KO.md) · [내 입력과 지원 과제](SUPPORT_MATRIX_KO.md). Kit는 비교 기능에 diova-compare를 재사용하지만, 독립 diova-compare처럼 두 답변 파일만 비교하는 도구와 제품 범위가 같지는 않습니다. 기본 wheel은 표준 라이브러리 CPU 기능이며, HTTP 경로의 `aiohttp`와 모델 runtime은 별도 의존성입니다.

### 현재 안내와 보존된 배포 이력

v0.2.1은 원래 비공개 배포에서 검증된 뒤 공개 접근이 확인된 배포판입니다. 동결된 키트의 PRIVATE·인증 필수·옛 pointer 문구는 당시 기록으로 남겼습니다. 현재 시작 경로는 이 README와 [START_HERE](START_HERE_KO.md)를 따르며, kit·wheel·tag·기존 영수증의 bytes는 바꾸지 않습니다. 최신 source main의 문서 수정과 새 release는 다른 상태입니다.

최신 release 판정은 **[ACTIVE_RELEASE_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/ACTIVE_RELEASE_ACCEPTANCE_2.json)** → [RELEASE_DECISION_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/RELEASE_DECISION_ACCEPTANCE_2.json) → [LIVE_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/LIVE_ACCEPTANCE_2.json) 및 pointer에 선언된 관련 근거를 따릅니다. 일반 이름의 `RELEASE_DECISION.json`·`LIVE_ACCEPTANCE_RECEIPT.json`과 옛 pointer는 보존된 과거 BLOCKED/PARTIAL 기록일 수 있습니다. 자산 이름만으로 최신 판정을 고르지 마세요.

### 새 자료로 모델을 실행하려면

1. 기존 task 계약에 맞는 문서·질문·외부 정답 기준을 준비합니다.
2. 지원 모델의 pinned tokenizer로 입력을 준비하고 계획을 검증합니다.
3. 별도 모델·runtime·현재 권한·자원·예산 조건을 확인한 뒤 명시적으로 실행합니다.
4. 같은 규칙으로 채점한 결과와 원시근거, 미수집 진단 범위를 함께 읽습니다.

실제 명령과 외부 조건은 [Live Quickstart](QUICKSTART_LIVE_KO.md)에 있습니다. 계획 파일은 실행 권한이 아닙니다. 모델 weights·tokenizer 전체 자산·ML SDK·credential·실제 local 설정은 배포물에 포함하지 않습니다. 누적 보고서를 옮길 때는 [보고서 포장 계약](REPORT_PACKAGING_KO.md)에 따라 모든 evidence volume과 manifest를 함께 이동합니다.

## 상세 문서와 한계

[CPU 명령](QUICKSTART_CPU_KO.md) · [Live 명령](QUICKSTART_LIVE_KO.md) · [지원표](SUPPORT_MATRIX_KO.md) · [알려진 제한](KNOWN_LIMITATIONS_KO.md) · [보고서 포장](REPORT_PACKAGING_KO.md) · [수용시험 명세](release-kit/acceptance/ACCEPTANCE_PLAN_V1.json)

NFC 정규화에 따른 일부 원문 span 매핑 제한, namespace 사이의 동일 sample ID 거절, 같은 동결 source에서만 가능한 재개, 불명확한 제출의 자동 재전송 금지를 유지합니다. 저장 자료 재계산이 과거 GPU 실행을 새로 검증한 것은 아닙니다. 과거 PARTIAL·STALE·UNKNOWN·INCONCLUSIVE와 `HOLD_CLIPPING_VARIANT` 판정도 원래 이력에 남습니다.

[구조](docs/ARCHITECTURE_KO.md) · [데이터 계약](docs/DATA_CONTRACTS_KO.md) · [보존 이력](docs/HISTORY_KO.md) · [변경 이력](CHANGELOG_KO.md) · [라이선스·출처](THIRD_PARTY_NOTICES.md)

## 사용 범위와 문제 보고

문제는 이 저장소의 Issues에 package version, source digest, 실행 명령, 기술 실패와 모델 오답의 구분, 민감 경로를 제거한 작은 재현 자료를 남겨 주세요. 공개 Issues·댓글·첨부는 누구나 읽을 수 있습니다. credential·개인 local binding·고객/개인 데이터·전체 모델·전체 raw는 첨부하지 않습니다. 원래 raw와 실패 상태를 덮어쓰지 마세요.

소스 검사는 `python -B tools/test_cpu.py --output <새 receipt.json>`으로 실행합니다. 현재 `.github/workflows/cpu.yml`은 수동 `workflow_dispatch`이며 PR·main push마다 자동 회귀를 실행하지 않습니다. 과거 수동 CI 결과는 최신 main의 자동 통과 표시가 아닙니다. HTTP mock 테스트의 선택 HTTP 의존성과 기본 wheel의 CPU 의존성을 구분합니다. 현재 릴리스의 실제 source·wheel·CI·live 결과는 위 active release pointer를 따르며, 과거 테스트 수를 현재 PASS로 쓰지 않습니다.

DIOVA는 [공개 포트폴리오](https://munsik-kim.github.io/inference-lab/ko/index.html)의 브랜드, [inference-lab](https://github.com/Munsik-Kim/inference-lab)은 공개 사례 저장소입니다. 그 안의 diova-compare는 공개된 별도 CPU 결과 비교 도구입니다. 이 설치형 제품과 동일한 제품·검증 범위로 합치지 않습니다.
