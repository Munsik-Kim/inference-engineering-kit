# 데이터·상태 계약

`kit spec` v1은 model ID/revision/architecture, method recipe/format, profile, paths, budget을 분리합니다. 원본 STEP02/03/04 schema를 같은 spec으로 바꿔 썼다고 주장하지 않습니다. frozen replay는 각 legacy schema를 그대로 읽습니다.

경로는 `${PROJECT_ROOT}`, `${ARTIFACT_ROOT}`, `${DATA_ROOT}`, `${RUNS_ROOT}`를 명시적으로 바인딩합니다. roots는 local 파일의 부모 기준, 그 밖의 상대 경로는 spec 파일 부모 기준입니다. cwd와 원본 checkout을 lookup하지 않습니다. runtime Python symlink 표기는 보존합니다. local 예시는 placeholder이며 실제 local.json은 Git/자산에서 제외합니다.

identity는 canonical JSON SHA256(UTF-8/ensure_ascii=False)와 파일 bytes SHA256을 구분합니다. 기존 SSE wire hash의 ensure_ascii=True 정의도 보존합니다. model/tokenizer/recipe/profile/runtime/workload, STEP03 비교 group과 두 treatment axis를 분리합니다. 원본 config의 historical 경로는 메타데이터입니다. bundle 참조는 범위와 manifest를 확인하고 명시적 evidence binding 안에서만 풉니다. 해시 자체는 저자의 신원/진실성 인증이 아닙니다.

측정 정의:

- TTFT: HTTP 전송 후 첫 유효 content까지. 빈 SSE·usage·finish event를 content로 세지 않습니다.
- HTTP E2E: HTTP 전송부터 EOF까지, client semaphore wait 제외. task E2E는 client queue 포함.
- TPOT: `(마지막 content−첫 content)/(실제 completion tokens−1)`. 요청별 평균 추정이며 토큰별 ITL 분포가 아닙니다. 1 token이면 null.
- token 수: usage/검증 token ID, SSE event 개수 아님. timeout/오류를 버리고 유효 요청만 성능 평균을 만들지 않습니다.
- acc: summed continuation loglikelihood argmax. acc_norm: choice 원문 character 길이로 나눈 score argmax. 동률은 기존 first-index 규칙입니다. audit generation 1 token의 score는 제외합니다.
- paired 단위: sample/document ID. choice HTTP, sentinel, 같은 문서 반복을 독립 표본으로 세지 않습니다. STEP04 bootstrap은 네 arm을 같은 document index로 공동 재표집합니다.
- memory: `whole_device_mib`/`process_mib`/`allocator_mib`, phase_scope까지 일치해야 비교합니다. startup 포함 성능 max와 sanity/task max는 분리합니다. 누락은 null/NOT_MEASURED이며 artifact bytes로 대신하지 않습니다.
- Prometheus: 지원 HELP/TYPE/labels를 확인한 gauge/counter만 사용합니다. missing, label 변경, counter reset, poll gap을 0으로 바꾸지 않습니다. waiting 시간 적분과 histogram은 request별 queue로 연결하지 않습니다.

요청 완료, parent exit, owned cleanup/port, worker native exit, kernel level, quality status, 고객 검증을 별도 필드로 보존합니다. parent exit0은 worker exit0의 증거가 아닙니다. selected-log는 profiler 검증이 아닙니다. low-level semaphore 경고만으로 정상 raw를 폐기하거나 cleanup 성공을 만들어내지 않습니다.

raw output은 새 경로/immutable입니다. budget/progress만 atomic snapshot입니다. failed/partial은 새 attempt로 남기며 이전 ledger hash와 누적 합계를 확인합니다. kit의 새 단일 run budget을 기존 캠페인 예산 초기화로 사용하면 안 됩니다. 재개는 같은 frozen identity를 확인하고 새 attempt를 생성하는 API를 사용하며 자동 GPU 재시도는 없습니다.

STEP03 v3 `latency_constraints`는 전부 AND, 관측되지 않은 cell은 UNKNOWN입니다. retained quality는 허용을 명시해도 `CONDITIONAL_OBSERVED_CANDIDATE`이며 exact-profile 인증이 아닙니다. interval이 경계를 걸치면 INCONCLUSIVE/UNKNOWN입니다. STEP04 DEV의 production status는 항상 NOT_ELIGIBLE입니다.

숫자 parity 허용오차는 finite float에 abs=1e-10, rel=1e-9입니다. 정수·boolean·상태·ID·pairing·정오답은 정확히 같아야 합니다. NaN/inf는 거절합니다. JSON CRLF/LF 차이와 archive byte identity는 수치 일치와 별도입니다.
