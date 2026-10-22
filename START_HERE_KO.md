# inference-engineering-kit 0.2.0 시작

이 키트는 비공개 저장소의 권한 있는 사용자용 0.2.0 배포물이다. 정식 게시 여부와 같은 wheel의 실제 통과 범위는 release의 `RELEASE_DECISION.json`을 먼저 확인한다. 버그 부재·임의 모델·일반 모델 품질 보증은 아니다.

1. `QUICKSTART_CPU_KO.md`에 따라 새 base venv에 wheel만 일반 설치한다. 모델·tokenizer·ML SDK는 필요 없다.
2. 실제 저장 Qwen 응답(A), Smol token/선택 구조(B), prepared 계획(C), 명시적 합성 publication/index/report 상태기계(D)를 실행한다. 각 coverage와 acquisition 출처가 다르다.
3. `SUPPORT_MATRIX_KO.md`와 정본 `SUPPORT_MATRIX.json`에서 구현·과거 실측·현재 release CPU·현재 release live·외부 조건을 확인한다.
4. 실제 실행이 필요한 경우 `QUICKSTART_LIVE_KO.md`의 외부 asset과 모든 경로 변수를 먼저 채운다. source/권한/현재 자원/ledger gate를 통과한 명시 opt-in만 모델을 실행한다.

wheel은 stdlib CPU 기능을 제공한다. 모델 weights, tokenizer 전체 자산, ML runtime, credential, 실제 local 설정은 포함하지 않는다. `examples/CPU_LOCAL.json`과 저장 prepared SLICE는 계획·계약용이며 live-ready가 아니다. 원래 BENCH 폴더를 찾을 필요 없이 키트의 CPU 명령은 자립한다.

`KNOWN_LIMITATIONS_KO.md`의 decoded RTN/NFC/namespace/동일 source 재개 경계와 과거 실패 상태를 유지한다. 새 모델과 네 번째 task는 이 릴리스 scope 밖이다.
