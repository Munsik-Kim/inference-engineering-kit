# 이식 범위와 source-map

현재 inference-lab working tree의 파일 bytes를 기준으로 이식했습니다. source HEAD만으로 미커밋 STEP02~04 구현 identity를 대체하지 않았습니다. [source-map.json](source-map.json)은 원본 함수/파일·SHA → 새 API·처리 상태·검증 연결을 기록하고 [source-map.initial.json](source-map.initial.json)은 추출 전 import/함수 조사 기록입니다.

공통 경계는 기존 modelpack immutable JSON/digest, suite quality/selector/metrics, lab HTTP/lifecycle/NVML, diova core입니다. sys.path 주입과 원본 ROOT lookup만 제거하고 numerical definition은 유지했습니다. STEP별 고정된 H·표본·bootstrap·treatment contract는 experiments에 남깁니다. STEP02 C4 raw response의 완료 순서는 sample/choice key로 복원하며 STEP04 C1은 저장 순서도 검사합니다.

원본 step04 adapter의 전이 의존성을 포함해 추출했고, 변경 파일13개만 옮기거나 ZIP 내 코드 버전을 합치지 않았습니다. execution-source-v1은 GPU 실행 당시, source-snapshot은 후속 수정 시점, replay-source는 CPU replay용입니다. maintained package source는 별도의 새 버전입니다. 과거 source/config hash를 새 패키지 hash로 바꾸지 않습니다.

미이식: acquisition/download, quantization training/GPTQ/clipping ratio 탐색, BF16 holdout hook 재실행, 원본의 모든 historical CLI/registry, Case001~012 전체, 모델/data cache. raw artifact identity/계산/recipe는 evidence에서 보존하지만 이번에 이를 재생성하지 않습니다. 모델·method 추가는 새 explicit spec/adapter 계약으로 검토하며 이름 등록만으로 지원 완료가 아닙니다.

이번 수정/생성은 새 target과 외부 delivery 폴더에만 있습니다. 원본 Git status와 선정 source/config hash를 처음/끝 대조합니다. 기존 untracked 작업과 reports는 그대로 유지합니다. 새 target은 독립 Git이며 원본 history/.git/credential을 복사하지 않습니다.
