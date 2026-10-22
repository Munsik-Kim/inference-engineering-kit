# 알려진 제한과 유지하는 이력

- 현재 릴리스의 CPU/live 수용시험 상태는 별도 RELEASE_DECISION/LIVE_ACCEPTANCE_RECEIPT에 따른다. 과거 실측 source/runtime과 0.2.0 analyzer/source를 구분한다. gate 실패가 있으면 정식 게시하지 않는다.
- Smol RTN diagnostic은 DECODED_WEIGHT_DIAGNOSTIC/effective BF16이다. packed INT4 SERVE의 kernel과 동일하다고 쓰지 않는다.
- Qwen은 pinned BF16/GPTQ serving-only + native ARC target scalar다. full/KL/JS/entropy/rank·attention/decoded probe는 미수집이다.
- Qwen NFC normalization 때문에 e4a-tech-en-basic/e4a-tech-en-missing의 전체 source-span 매핑은 unavailable이다. 저장 raw/text/token/offset bytes를 변형하지 않는다.
- 동일한 sample_id가 다른 task/dataset namespace에 있으면 현재 live binder는 명시 거절한다. 조용히 overwrite하거나 임의 rename하지 않는다.
- 재개는 같은 frozen source의 검증된 동일 run에 한한다. cross-source migration/force-resume는 없다. UNKNOWN_SUBMISSION과 interrupted GPU job은 자동 재전송하지 않는다.
- Smol D1 structure는 standalone 선택 관측이다. arbitrary-model Suite structure, MLP/residual/MoE/state는 범위 밖이다.
- prepared fixture는 역사적 tokenizer 실행의 저장 결과다. 저장 prepared의 CPU 검증은 새 retokenization이 아니다. release live용 fresh preparation은 별도 lineage로 기록한다. FULL parent manifest를 누락 파일 허용으로 통과시키지 않는다.
- CPU 예제는 작은 실제 SLICE다. full 과거 DB/모든 task 분모/과거 GPU 완료 재인증으로 확장하지 않는다. active revision과 원래 acquisition을 별도 표시한다.
- historical dev16 controller PARTIAL, historical final live no-op NOT_RUN, historical E3 STALE locator, native worker exit UNKNOWN은 해당 원본 상태로 남긴다.
- HOLD_CLIPPING_VARIANT, 품질 INCONCLUSIVE, 역사적 synthetic strict0, ARC REGRESSION_REUSE를 보존한다. 새 관측 점수를 과거 strict0에 맞추지 않는다.
- 큰 역사적 HTML의 성능은 작은 CPU 예제로 측정하지 않는다. 브라우저 시각 검사 여부는 실제 receipt를 따른다.
- base wheel/kit에는 모델 weights·전체 tokenizer·ML runtime·credential·개인 local config가 없다. HTTP extra는 live 클라이언트용이며 기본 CPU 경로와 분리한다.
