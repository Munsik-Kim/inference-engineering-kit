# BENCH06-A: 기존 Qwen pair의 serving-only Suite 계약

기존 Qwen3-4B-Instruct-2507 BF16와 GPTQModifier W4 artifact를 명시적 준비 자료·Suite에 연결한다. GPTQ의 저장 형식은 compressed-tensors/pack-quantized, symmetric INT4/group128이며 RTN이 아니다. 정확한 revision·artifact receipt·구조·packing·tokenizer·runtime 조합을 검사한다. 모델 이름 등록만으로 지원을 부여하지 않는다.

`qwen3_4b_gptq_serving_v1`은 준비/정적 capability, `prepared_qwen_serve_v1`은 실제 binding profile이다. 기존 Smol profile과 frozen wrapper는 별도 계약을 유지한다. 지원 operation은 기존 세 task의 자연 생성·모든 ARC choice likelihood와 CPU score/analyze/index/report다. 선택 token/full-logit worker와 structure는 NOT_REQUESTED이며, full-vocabulary KL/JS/entropy/rank는 NOT_COLLECTED다.

원문·외부 gold·task body·scorer를 보존하고 pinned Qwen template를 한 번 적용한다. 생성 prompt-only와 teacher-forced reference는 분리한다. ARC trailing whitespace·joint encoding·choice 순서·문자 길이 정규화는 기존 계약을 사용한다. Tokenizer vocabulary151669와 model output axis151936은 별도 metadata다. 원래 raw의 ENGINEERING_INPUT_ONLY를 바꾸지 않고 신규 실행 scope ENGINEERING_REGRESSION_REUSE를 별도로 기록한다.

SuiteSpec metric `native_arc_tokens_c1_v1`은 실제 native ARC continuation scalar를 공통 TokenView/C1 비교기에 연결한다. GOLD_ONLY/NON_GOLD/ALL_CHOICES의 분모를 분리하며 audit 생성1token은 제외한다. 다른 tokenizer의 Smol NLL과 직접 비교하지 않는다. 같은 MAIN9 raw/gold의 task outcome을 모델별 deployment configuration으로 나란히 볼 수 있으며, template/runtime 차이를 양자화만의 효과로 해석하지 않는다.

제품 API/CLI는 기존 입구를 사용한다: `inputs_prepare`, `suite_plan/validate/describe/bind`, `suite_run/resume/status/replay`, `index_add/report`. 로컬 binding은 명시 파일이며 임의 module/shell/URL을 받지 않는다. GPU 실행은 별도 권한·현재 자원·소유권·최신 ledger·cleanup reserve를 매번 확인한다. 정상 완료 no-op는 추가 GPU/runtime/HTTP 전에 완료 evidence를 검증한다. 과거 source의 live run을 이관하거나 검사를 끄지 않는다.

Native MINI는 명시적 source slice이며 lossless HTTP/continuation metadata와 parent manifest ancestry를 보존한다. 저장 raw의 CPU score/C1 재계산만 수행한다. 개인 locator·모델·tokenizer 자산이 없는 MINI를 원본 live 환경이나 전체 역사 DB의 재인증으로 부르지 않는다.

실제 수행 명령·수량·통과/차단 상태·source/wheel identity와 소용량 인계는 해당 `reports/bench06a_qwen_portability_*`의 REPORT/HANDOFF를 따른다. 이 문서는 실제 GPU 완료를 선언하는 영수증이 아니다.

Pinned Qwen tokenizer의 NFC normalizer가 원래 NFD archive note를 내부 정규화하는 경우 원문이나 렌더 문자열을 preparer가 변경하지 않는다. 원래 native offset을 그대로 보존하고, 대응되지 않는 prompt combining codepoint와 normalize 여부·hash를 별도로 기록한다. 이런 문서의 전체 source span은 MAPPING_UNAVAILABLE_NORMALIZATION이다. 정확한 reference/field 및 ARC continuation boundary를 주장하려면 해당 구간은 normalization-stable이어야 한다. 이것은 모델/tokenizer 속성에 따른 지원 한계이며 공백/offset 검사를 무조건 통과시키는 옵션이 아니다. Smol의 기존 strict 경로는 유지한다.
