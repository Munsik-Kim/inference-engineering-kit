# 역사적 rc1 후보 scope (보존)

이 문서는 V1-CANDIDATE-01 당시 범위다. 현재0.2.0 지원/수용시험 상태는 SUPPORT_MATRIX.json과 release RELEASE_DECISION 영수증을 따른다.

# V1-CANDIDATE-01 범위 동결

이 패키지는 명시 범위의 설치형 수용시험에 투입할 첫 사용판 후보다. 정식 v1 완료, 안정 API 보증, 새 live 수용 또는 배포 추천이 아니다. 기능 추가를 동결하고 복사 descriptor 소유권 오류의 최소 수정과 CPU 사용 확인을 수행한다.

정본은 [SUPPORT_MATRIX.json](SUPPORT_MATRIX.json)이며, 문서 표는 그 읽기용 요약이다. 각 기능은 구현 capability, 과거 실제 관측, 이번 후보 CPU 검사, 이번 후보 live 검사, 외부 자산 요구사항을 따로 읽는다. 후보 CPU 결과는 키트의 최종 CPU 수용 영수증으로 확정하며, 후보 live는 전부 NOT_RUN이다.

|범위|구현·과거 근거|이번 후보에서 확인할 범위|후보 live|
|---|---|---|---|
|CPU 공통|raw 검증, 기존 scorer/C1·구조 metric, publication, SQLite index/report, prepared planner|설치 wheel의 작은 실제 SLICE와 별도 빈 DB; 재등록·rebuild·relocation|NOT_RUN|
|Smol BF16/RTN SERVE|pinned Smol pair, 세 task generation/모든 choice likelihood; E4B의 9문서 실측|저장 자료와 CPU task plan|NOT_RUN|
|Smol 선택 token|BF16 INSTRUMENTED_PATH; RTN decoded weight/effective BF16; C2/E4B selected target/full|실제 저장 paired scalar/full 재계산, token-only plan|NOT_RUN|
|Smol 선택 structure|D1의 standalone collector, 선택 layer/head/query, probability/native AV/reconstructed AV/head 없는 o_proj|실제 저장 선택 구조 SLICE 재계산|NOT_RUN|
|Qwen BF16/GPTQ|Qwen3-4B-Instruct-2507 pinned BF16/GPTQModifier packed INT4/group128; serving-only, native ARC scalar|저장 응답 score/native TokenView와 serving-only plan|NOT_RUN|
|원문 준비|지원 세 task와 외부 gold; pinned tokenizer의 별도 prepare worker|저장 prepared 계약·경계·hash만 검사; 새 tokenizer 호출0|NOT_RUN|
|STEP02~04|과거 성능/KV/clipping recipe와 명시 replay 경로|이번 후보에서 모델/성능 재측정 없음|NOT_RUN|

현재 task는 tech_record_extract0.1, context_evidence_retrieval0.1, arc_challenge1.0이다. 기존 scorer와 prompt/gold 정책을 유지한다. 기본 context2048, 준비 prompt<=512/reference<=128/joint<=640 token, C1 직렬, 생성 cap은 tech192/retrieval64이며 ARC audit1token은 continuation likelihood 채점에서 제외한다. 더 긴 입력은 자동 자르지 않는다.

Smol packed RTN SERVE와 DECODED_WEIGHT_DIAGNOSTIC은 다른 경로다. standalone Smol structure collector가 있는 사실은 임의 모델의 Suite structure 통합을 뜻하지 않는다. Qwen full-vocabulary/attention/AV/o_proj/decoded probe는 이번 profile의 미수집·미지원이다. scalar를 full 지표0으로 채우지 않는다.

모델 간 task 표는 같은 원문/gold/scorer가 확인된 문항에서 조건을 표시해 병렬로 읽는다. Smol/Qwen tokenizer·prompt template·runtime이 다르므로 token NLL/PPL/KL·좌표를 직접 정렬하지 않는다. 새 모델·네 번째 task·양자화·UI 재설계·구조 관측 확대는 후보 범위 밖이다.

동일 sample_id가 다른 namespace에 존재하면 현재 live binder는 명시 거절한다. Qwen NFC로 두 문서의 전체 source-span 매핑은 unavailable이며 raw/참조를 재작성하지 않는다. 같은 source의 resume는 source/input/plan/runtime/evidence 일치를 요구하고 옛 live run을 새 source로 이관하지 않는다. historical PARTIAL/STALE/no-op NOT_RUN·UNKNOWN·INCONCLUSIVE를 보존한다.

base wheel은 stdlib CPU 경로를 제공한다. 모델·tokenizer assets·ML SDK·GPU·실제 binding/ledger·자원 권한은 외부 조건이며 키트에 포함하지 않는다. prepare는 명시적 tokenizer runtime에서만 SDK를 사용하고, live는 별도 opt-in과 현재 gate를 요구한다. 후보 패키징은 이 외부 조건의 충족을 인증하지 않는다.
