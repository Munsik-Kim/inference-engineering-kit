# 0.2.1 지원 범위

정본은 [SUPPORT_MATRIX.json](SUPPORT_MATRIX.json)이다. 구현·historical live·이전 rc1 CPU·현재 release CPU·현재 release live·외부 조건·OS/Python 축을 분리한다. 이 동결 문서는 결과를 미리 PASS로 선언하지 않는다. 실제0.2.1 commit/wheel의 통과 여부는 release의 RELEASE_DECISION/CPU_TESTS/REMOTE_CI/LIVE_ACCEPTANCE 영수증을 따른다.

| 기능/모델 | 구현 범위 | 현재 live 수용 gate | 한계 |
|---|---|---|---|
| CPU 공통 | 저장 raw 검증/replay, 세 task 채점, compatible TokenView, publication, index/report, prepared plan | CPU 소비자·이동·API/module/console | stdlib base wheel. 작은 actual SLICE와 synthetic state test를 분리 |
| SmolLM2-1.7B BF16/RTN | 세 task serving/native likelihood + selected teacher-forced token/full combined Suite | L1 MAIN9 | packed SERVE와 decoded/effective BF16 diagnostic 별개 |
| Smol selected structure | standalone attention/native AV/o_proj·OFF1/OFF2/ON | L3 별도 보존6입력 | Suite structure 통합·임의 모델 지원 아님 |
| Qwen3-4B-Instruct-2507 BF16/GPTQ W4A16 | 세 task serving-only + all-choice native ARC scalar | L2 MAIN9 | full-vocab/KL/JS/entropy/rank/attention/GQA/decoded probe 미지원·미수집 |
| 원문 준비 | 두 pinned tokenizer와 기존 세 task의 명시 local-only prepare | L1/L2 새 CPU preparation | 외부 local assets/runtime 필요. 저장 prepared 검증과 구분 |
| STEP02~04 | 보존 recipe/작은 saved replay | 이번 신규 live gate 아님 | 과거 성능·KV·clipping을 현재 통합 Suite 지원으로 합치지 않음 |

CPU 선언은 Linux/WSL의 준비된 Python3.10/3.11/3.12 검증 범위다. 각 버전의 실제 실행/환경 차단은 현재 영수증에 따른다. py3-none-any wheel을 모든 OS live 검증으로 해석하지 않는다. Live는 지정된 Linux/WSL host의 pinned runtime·artifact·tokenizer·현재 권한/자원·ledger 조건에 한정한다.

동일 bare sample ID의 namespace 충돌은 명시 거절한다. Qwen NFC 두 문서 source-span unavailable, same-source resume만 허용, UNKNOWN submission 자동 retry 금지, 과거 Smol PARTIAL/final noop NOT_RUN/E3 STALE 및 HOLD_CLIPPING_VARIANT/INCONCLUSIVE는 유지한다. 세 task9문항 engineering 수용 결과가 일반 품질·고객 검증·배포 동등성을 보증하지 않는다.

report의 기본 single 모드는 evidence256MiB이고, 명시적인 bounded collection 모드는 volume256MiB×최대2/payload512MiB/presentation128MiB/전체640MiB다. 수집·metric 지원을 늘리는 기능이 아니라 손실 없는 유한 포장이다. [공개 계약](REPORT_PACKAGING_KO.md)의 전체 part 이동·closure 검증을 따른다.
