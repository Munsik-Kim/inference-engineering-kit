# BENCH05-B4: 제공된 문맥의 답·근거 검색

개발판0.1.1.dev7은 세 번째 TaskAdapter `context_evidence_retrieval/0.1`을 추가한다. 요청 종류는 기존 `generate_until`이다. 여섯 문단을 모두 제공하고 지정 대상/관계의 코드와 근거 ID를 찾으며, 없는 사실은 NOT_FOUND/NONE으로 답한다. 웹·벡터DB·embedding·RAG 서비스는 없다.

합성 SMOKE4/EVAL24는 영어·한국어 각12개 본 문서, 답 있음18/없음6, 근거P01..P06 각3개다. 원문·질문만 읽는 독립 parser와 controlled-template/always-NOT_FOUND baseline을 같은 scorer로 검사한다. 통제된 문법의 작은 DEV이고 고객·공식 TEST가 아니다. 첫 응답 전에 자료·문구·순서·scorer·probe를 동결한다. GPU 전 길이 수정은 원래 문구와 이유를 보존한다.

출력은 `ANSWER: <code>`와 `EVIDENCE: <Pxx>` 두 줄이다. CRLF/CR→LF, 바깥 ASCII 공백/빈줄만 제거하며 label 앞·colon 뒤 space/tab을 허용한다. 코드·sentinel은 exact이고 fence/prose/중복은 strict format 실패다. 유일한 독립 필드는 별도 채점한다. Pxx 문법, 실제 인용 존재, gold 근거 일치를 구분한다. strict는 전송 완료·자연 종료·format·답·근거가 모두 맞아야 한다. 10초는 HTTP submit~완료의 DEMO 기준이다.

```text
inference-kit bench b4 plan --suite suite.json --models pair.json --local local.json --output NEW_PLAN
inference-kit bench b4 run --plan NEW_PLAN/PLAN_FROZEN.json --output NEW_RUN --allow-gpu
inference-kit bench b4 replay --run RECORDED_RUN_OR_ZIP --output NEW_REPLAY
inference-kit bench b4 score --sample sample.json --generation generation.json --output NEW_SCORE
```

Python API는 같은 `bench_b4_plan/run/replay/score`, `task_adapter('context_evidence_retrieval','0.1')`이다. plan/score/replay와 기본 import는 모델·네트워크·ML SDK·GPU를 사용하지 않는다. tokenizer만 기존 별도 runtime의 `python -m inference_kit.probes.retrieval_tokens --data ... --tokenizer ... --config ... --output ... --runtime-output ...`로 준비한다. 개인 local은 Git에 넣지 않는다. 예제의 origin0은 placeholder이며 실제로는 최신 campaign ledger/사용량을 연결해야 한다.

기존 owned serving에 명시적 collector contract를 전달하여 natural64/EOS/stream을 실제 wire와 대조한다. B1 natural192 및 B3 likelihood audit1 기본 계약은 유지한다. HTTP 전 journal은 client submit attempt이며 서버 수신 증명이 아니다. SSE를 채점 전에 flush하고, incomplete 제출은 새 CPU replay에서 복원한다. raw/task/prompt/tokenizer/recipe/runtime/출처가 다르면 비교를 거절하며 세 task 점수를 평균하지 않는다.

선택4문항의 author gold 두 줄을 별도 TF 참조로 사용한다. B4 한도는 prompt512/reference64/joint576이고 과거 wrapper 한도는 바꾸지 않는다. 기존 layer12/head0·16·query 정책과 최대16 target/4 full-vocab 위치를 사용한다. 근거 CONTENT의 codepoint·UTF8·joint token span을 label/instruction/reference와 구분해 검사한다. 저장된 full-key row에서 겹치는 content token mass를 합산하며 boundary token은 전체 포함한다. 답 부재 mass는 NOT_APPLICABLE/null, query 조건은 last_prompt/reference_prefix로 나눈다.

BF16은 INSTRUMENTED_PATH, RTN은 DECODED_WEIGHT_DIAGNOSTIC이다. 선택 행은 native post-RoPE QK 재구성, AV는 원래 SDPA 반환값이다. NLL/JS/mass의 정답 인과성·packed/decoded 동등성을 주장하지 않는다. B4 bridge/OFF 대조는 추가하지 않는다. 실제 coverage·숫자·예산은 timestamp report를 따른다. C1 단일 서버 기술통계이며 production/customer NOT_VALIDATED다. 다음 BENCH05-C는 별도 작업이다.
