# BENCH05-E1 로컬 근거 index

E1은 보존된 D1/C2 진단과 B2 P00/B3/B4 task 결과를 CPU에서 검산하고 검색하는 개발 기능이다. 원시파일은 불변이며 SQLite는 재생성 가능한 검색용 사본이다. 새로운 GPU 측정·업무 품질 평가·배포 적합성 검증은 수행하지 않는다.

공개 API는 `index_build`, `index_add`, `index_query`, `index_report`다. CLI는 같은 함수를 호출한다.

```bash
inference-kit bench index build --sources INPUT_SOURCES.json --db new.sqlite --output new-import
inference-kit bench index add --sources additional.json --db new.sqlite --output new-add
inference-kit bench index query --db new.sqlite --task arc_challenge --output new-query
inference-kit bench index report --db new.sqlite --output new-report
```

입력은 schema_version=1과 명시적인 sources 배열이다. 각 source는 source_id(선언 namespace), adapter, path를 가진다. 허용 adapter는 structure_d1, token_c2, task_b2_p00, task_b3, task_b4다. 모델이나 외부 Python loader는 등록할 수 없다. 파일/ZIP을 읽을 때 frozen plan과 모든 RAW_MANIFEST 파일의 bytes/SHA를 확인한다. 과거 raw를 새로 분석한 것은 acquisition이 늘어난 것이 아니다. sample_ids/splits/partitions 선택은 별도 analysis identity에 포함한다.

같은 원시자료의 directory/ZIP/다른 locator는 같은 source로 등록된다. 같은 문항을 C2와 D1에서 각각 실행한 것은 다른 acquisition이다. 같은 namespace에 다른 내용을 넣으면 SOURCE_ID_CONFLICT로 거절한다. source 단위 transaction이 실패하면 해당 source의 metric 일부만 남지 않는다. query/report는 현재 근거 hash를 다시 검사하고 STALE/MISSING_EVIDENCE 값을 null로 표시한다. 이전 physical root를 옮겼으면 `--root-map mapping.json`으로 `{ "old-root": "new-root" }`를 명시한다.

보고서는 서버·CDN·외부 script·fetch 없이 file://로 읽는다. 승인된 manifest 파일만 상대 링크로 복사하며 HTML과 표시용 CSV는 실행 가능한 텍스트를 escape한다. canonical raw/JSON 숫자는 바꾸지 않는다. 원본 raw에 있는 역사적 경로 문자열은 출처 metadata이며 파일 접근 fallback으로 쓰지 않는다.

D1 점검에서 빈 target 집합이 observer VERIFIED로 판정되는 reader 버그를 수정했다. 실제 D1의 nonempty 110 target/44 full pair에는 수치 영향이 없었다. 21개 metric 파일은 새 analyzer provenance만 분리한 비교에서 일치했다. Q/K/V 미보존 때문에 과거 GPU QK/AV 연산 자체를 CPU에서 독립 재구성했다고 주장하지 않는다.

과거 SERVE 결과와 별도 teacher-forced 진단은 HISTORICAL_OUTCOME_LINK다. C2 token-only와 D1 ON은 COMMON_COORDINATES_ONLY 관계로 표시하고 scalar delta를 자동 계산하지 않는다. RTN 내부 관측은 DECODED_WEIGHT_DIAGNOSTIC/effective BF16이다. OFF control, head별 AV, head 없는 o_proj 및 task별 지표/분모를 합치지 않는다. E 전체나 v0.2 완료를 의미하지 않는다.
