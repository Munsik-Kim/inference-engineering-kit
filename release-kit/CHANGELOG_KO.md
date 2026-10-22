# 변경 이력

## 0.2.1 — bounded 누적 보고서 수용시험

기존 단일 보고서의256MiB raw cap을 유지하며, 명시적인 `bounded_collection_v1` 포장 모드를 추가한다. 하나의 오프라인 진입점과 최대2개의256MiB evidence volume으로 필수 누적 근거를 손실 없이 보존한다. 전체 raw payload512MiB, presentation128MiB, 전체 output640MiB와 기존 단일 member/개수 한도를 별도로 검사한다. root/part/member closure는 완료와 no-op에도 검증한다.

이는 새 모델·task·metric 추가가 아니다. acquisition/analysis/metric identity·수치·분모와 과거 STALE/PARTIAL/null은 유지한다. 기존0.2.0 wheel/draft/실패 run과 자산은 변경하지 않는다.0.2.1의 정식 게시 여부는 동일 새 wheel의 G1~G8 외부 영수증에 따르며 CPU 교정 결과로 live 성공을 대체하지 않는다.

## 0.2.0 — 비공개 draft/실패 수용시험 (보존)

0.2.0rc1의 세 task, Smol BF16/RTN combined task+token Suite와 standalone 선택 구조, Qwen BF16/GPTQ serving/native ARC scalar, 공통 CPU replay·publication·index/report를 비공개 설치형 배포로 묶는다. 기능 확장 대신 README/유지 문서·명시 fresh-clock/첫 serving-unit helper·설치형 수용시험·고정 source/wheel/원격 CI와 자산 연결을 검증한다. 실제 gate 통과 여부는 release 외부 영수증을 따른다.

rc1에서 수정한 `_copy_member` fd 소유권 오류, field membership, source-scoped alias, terminal checkpoint 복구, raw-only publisher는 유지한다. 이번의 새 발견으로 다시 세지 않는다. 새 task·모델·양자화·범용 구조 진단·고객 품질 보증은 추가하지 않는다.

## 0.2.0rc1 — V1-CANDIDATE-01 (보존)

기능 동결, 알려진 raw copy fd 예외 처리 수정, wheel/sdist와 실제 저장 SLICE·CPU 사용 키트를 만들었다. 당시 live/원격 CI/release publication은 NOT_RUN이었다. 후보 원본 wheel과 영수증은 변경하지 않는다.

이전 0.1.x 개발 이력은 `docs/HISTORY_KO.md`와 개별 BENCH 문서에 보존한다.
