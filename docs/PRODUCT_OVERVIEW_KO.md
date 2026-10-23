# Inference Engineering Kit 제품 소개

## AI를 바꾸기 전에, 우리 업무로 먼저 시험하세요

모델을 더 적은 용량으로 표현하는 **양자화**는 로컬 AI를 운영할 때 검토하는 방법 중 하나입니다. 파일이 작아졌다는 사실만으로 중요한 업무의 답까지 같다고 판단할 수는 없습니다. 같은 문서에서 다른 대상의 값을 가져오거나, 내용은 맞아도 정해진 출력 형식을 지키지 못할 수 있습니다.

Inference Engineering Kit는 지원하는 원본 모델과 압축 모델에 같은 과제를 수행하게 하고, 답의 정확성·근거·출력 형식을 같은 규칙으로 비교하는 개발자용 설치형 Python 도구입니다. AI 도입·모델 교체를 검토하는 개발팀, 작은 R&D팀, 로컬 모델을 비교하는 연구자가 사용할 시험 근거를 만듭니다. 자료는 지원하는 세 과제의 정해진 입력 형식과 외부 정답 기준에 맞춰 준비합니다.

## 어떻게 사용하나요?

| 단계 | 사용자가 준비하거나 확인하는 것 |
|---|---|
| 1. 자료와 기준 준비 | 기존 task 계약에 맞는 문서·질문·정답·근거를 준비합니다. 정답은 사용자가 제공하거나 지원 dataset/task 규칙에 있어야 합니다. |
| 2. 두 구성 실행 | 각 모델의 pinned tokenizer로 입력을 준비하고, 모델·runtime·권한·자원·예산을 확인한 뒤 명시적으로 실행합니다. |
| 3. 같은 규칙으로 비교 | 정답 여부, 필요한 필드·근거·출력 형식과 문항별 유지·손실·획득을 확인합니다. |
| 4. 결과와 근거 확인 | 점수에서 원래 입력·응답·실행 조건으로 이동하고, 저장 자료를 모델 재실행 없이 다시 채점·분석합니다. |

저장 자료를 다시 계산하는 **CPU replay**는 모델을 다시 실행하는 것이 아닙니다. 보존된 응답·관측값의 포함 범위 안에서 결과를 다시 확인하는 절차입니다. 실행 비용과 내부 상태의 진단도 지원 경로에서 확인할 수 있지만, 모든 모델·과제의 모든 지표가 통합된 것은 아닙니다.

## 현재 할 수 있는 일

세 가지 기존 과제를 제공합니다. **기록 추출**은 문서에서 지정 필드와 근거를 찾고, **문맥·근거 검색**은 주어진 문맥 안에서 답과 근거를 찾으며, **ARC 객관식**은 모든 선택지의 점수를 같은 계약으로 비교합니다. 자유로운 새 업무 schema나 네 번째 task를 자동으로 만들어 주지는 않습니다.

- **CPU 공통:** 저장 원시근거 검증·재채점·비교, 불변 근거 게시, index·오프라인 보고서, 준비 입력의 validate·describe·plan.
- **SmolLM2-1.7B-Instruct 특정 BF16/RTN:** 세 task 실행과 선택 토큰·전체 분포 진단. 별도 standalone 선택 attention·AV·projection 진단도 있습니다. RTN 내부 진단은 decoded/effective BF16으로, packed 커널 내부 관측과 구분합니다.
- **Qwen3-4B-Instruct-2507 특정 BF16/GPTQ:** 세 task 실행과 ARC의 native 토큰 점수 분석. 전체 어휘 분포·attention·decoded probe는 이 경로에 없습니다.

토큰은 모델이 문장을 처리하는 작은 단위입니다. 토큰 진단은 지원된 좌표와 확률 계약에서 읽어야 하며, 서로 다른 tokenizer를 쓰는 모델의 점수를 직접 정렬하거나 하나의 지능 점수로 합치지 않습니다. 정확한 artifact·runtime·입력 한도는 [지원표](../SUPPORT_MATRIX_KO.md)와 [기계 판독 정본](../SUPPORT_MATRIX.json)을 따릅니다.

## 이용 형태와 시작 경로

현재 정식 사용판은 **v0.2.1**입니다. 설치와 자료 준비·운영에는 개발 작업이 필요합니다. 일반 직원이 클릭 한 번으로 쓰는 SaaS나 새 LLM, 자동 압축 최적화 엔진을 제공하는 형태는 아닙니다.

1. [START_HERE의 다운로드·SHA 확인](../START_HERE_KO.md#배포물-받기와-sha256-확인)에서 [정식 release](https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1)의 키트를 받습니다. 저장소가 PUBLIC인 경우 웹 다운로드에는 로그인이 필요하지 않습니다. `gh`는 선택 경로입니다.
2. [ACTIVE_RELEASE_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/ACTIVE_RELEASE_ACCEPTANCE_2.json) → [RELEASE_DECISION_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/RELEASE_DECISION_ACCEPTANCE_2.json) → [LIVE_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/LIVE_ACCEPTANCE_2.json)와 관련 근거를 확인하고, [CPU Quickstart](../QUICKSTART_CPU_KO.md)에서 저장 예제를 먼저 실행합니다. generic `RELEASE_DECISION.json`·`LIVE_ACCEPTANCE_RECEIPT.json`은 과거 BLOCKED/PARTIAL 기록일 수 있습니다. 기본 CPU wheel은 외부 의존성 없이 동작하며 모델을 실행하지 않습니다.
3. 새 자료를 실제 모델에 처리하려면 [Live Quickstart](../QUICKSTART_LIVE_KO.md)의 외부 자산·현재 권한·예산 조건과 명시적 실행 opt-in을 확인합니다.

모델 weights·tokenizer 전체 자산·ML runtime·개인 local 설정은 별도로 준비해야 합니다. 개발 중인 미게시 로컬 후보 `0.2.2.dev1`은 정식 `v0.2.1`과 별개입니다. 후보의 실행 기록을 정식판의 새 실측이나 정식 `0.2.2` 배포로 옮기지 않습니다.

**2026-10-23 접근 안내:** `v0.2.1`은 원래 비공개 배포로 검증됐습니다. 저장소의 접근 범위를 바꾸더라도 동결된 키트 안의 PRIVATE·옛 pointer 문구와 기존 자산은 당시 bytes 그대로 보존합니다. 현재 시작 경로는 웹의 START_HERE를 따르며, 이번 접근 변경을 새 모델 실측·새 패키지 출시로 해석하지 않습니다.

검증은 개발자 자체 수용시험 범위입니다. 일반화 성능·배포 동등성·SLA·독립 고객 성과는 별도 미확립 범위이며, 공개된 소스·패키지·저장 fixture의 범위를 전체 모델 자산이나 모든 실행 자료의 공개로 확대하지 않습니다. NFC 원문 매핑, namespace ID, 동일-source 재개 등의 [알려진 제한](../KNOWN_LIMITATIONS_KO.md)을 기능과 함께 읽어 주세요.

## 공개 사례와의 관계

**DIOVA**는 [공개 포트폴리오](https://munsik-kim.github.io/inference-lab/ko/index.html)의 브랜드이고, **inference-lab**은 [공개 사례 저장소](https://github.com/Munsik-Kim/inference-lab)입니다. **diova-compare**는 그 포트폴리오에서 소개하는 공개된 별도 CPU 결과 비교 도구입니다. **Inference Engineering Kit**는 여기 설명한 설치형 도구입니다. 각각의 구현과 공개 검증 범위는 다릅니다.

[README로 돌아가기](../README.md) · [기술 구조](ARCHITECTURE_KO.md) · [개발·실패 이력](HISTORY_KO.md) · [라이선스·출처](../THIRD_PARTY_NOTICES.md)
