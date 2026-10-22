# 유한한 누적 보고서 포장

0.2.1은 기존 `single_report_v1` 보고서의256MiB evidence 한도를 유지한다. 누적 근거를 보존해야 하는 호출은 첫 실행 전에 `examples/REPORT_PACKAGING.json`의 명시적인 `bounded_collection_v1` 정책을 선택한다. 임의 상향이나 실행 중 policy 변경은 허용하지 않는다. 더 작은 양의 상한은 선택할 수 있으며 입력이 넘으면 거절한다.

| 구분 | 상한 | 의미 |
|---|---:|---|
| evidence volume | 256MiB | 원시파일은 나누지 않고 결정적으로 배치 |
| volume 개수 | 2 | 자동 무제한 part 생성 없음 |
| 전체 evidence payload | 512MiB | 모든 volume의 실제 raw byte 합 |
| presentation | 128MiB | HTML/CSV/root·part manifest 등 파생 파일 합 |
| collection 전체 | 640MiB | raw payload와 presentation의 합 |
| 단일 raw member | 64MiB | 기존 더 엄격한 검사 유지 |
| eligible member | 10,000 | 복사 허용 유형·경로·hash 검증 유지 |

256MiB보다 큰 raw를 분할 포장하면 전체 저장량도256MiB보다 크다. 이 기능은 전체 byte 제한을 숨기는 압축·생략 기능이 아니다. logical evidence의 source/acquisition/analysis/metric identity와 수치·분모는 변하지 않는다. historical STALE/unsupported evidence는 원래 상태와 null/사유를 유지하며 인증된 복사 대상으로 승격하지 않는다.

## 정책을 동결하는 위치

Suite 예제는 `outputs.report_packaging`에 같은 정책을 포함한다. `tools/release_live_session.py suite`는 선택한 template의 정책을 보존하여 새 Suite를 작성한다. 정책은 Suite semantic/execution identity의 일부이며 같은 frozen run의 실패 뒤 바꾸어 재개하지 않는다. 새 정책으로 재분석하려면 새 CPU 분석 workspace를 만든다.

이미 보유한 명시 DB의 보고서 또는 raw의 CPU 재분석은 설치 wheel의 공개 명령을 사용한다. `DB`, `RAW`, `KIT`, `WORK`, `PY`는 실제 절대경로이며 output은 새 디렉터리다.

```sh
"$PY" -I -B -m inference_kit bench index report --db "$DB" --report-packaging "$KIT/examples/REPORT_PACKAGING.json" --output "$WORK/collection"
"$PY" -I -B -m inference_kit bench suite replay --run "$RAW" --historical-index "$DB" --report-packaging "$KIT/examples/REPORT_PACKAGING.json" --output "$WORK/replay"
"$PY" -I -B -m inference_kit bench suite postprocess --run "$RAW" --analysis-revision my-cpu-revision --report-packaging "$KIT/examples/REPORT_PACKAGING.json" --output "$WORK/analysis"
```

API는 `index_report(..., report_packaging=policy)`, `suite_replay(..., report_packaging=policy)`, `suite_postprocess(..., report_packaging=policy)`이며 `policy`는 JSON을 읽은 dict다. `None`은 기존 단일 보고서 동작을 선택한다. CPU 재분석 성공은 과거 live controller나 새 wheel의 GPU 수용시험을 재인증하지 않는다.

## 이동·검증 단위

`index.html`, CSV, `REPORT_MANIFEST.json`, `REPORT_COLLECTION_MANIFEST.json`, 모든 `evidence/part-NNNN/PART_MANIFEST.json`과 member를 **collection 디렉터리째 함께 이동**한다. manifest 또는 HTML 하나만 전달하면 완전한 보고서가 아니다. ZIP은 먼저 풀고 `index.html`을 연다. 브라우저가 ZIP 내부를 직접 탐색한다고 가정하지 않는다. 링크는 collection 안의 상대경로이며 원본 raw의 절대경로를 새 내용으로 rewrite하지 않는다.

root는 part manifest의 hash를, part는 member의 path/size/hash를 연결한다. 자기 hash 순환은 없다. 제품 완료·resume/no-op 검사는 전체 closure를 다시 확인한다. part/member의 누락·다른 byte·symlink/path escape는 정상 과거 locator의 존재와 관계없이 실패한다. no-op는 새 모델 작업이 없다는 뜻이며 검증 생략을 뜻하지 않는다. 원시 source publication은 파생 DB/report와 계속 분리한다.

## 보존하는 실패와 수용 범위

기존0.2.0의 실제 누적 근거4177개/285388384bytes는 단일256MiB cap을 넘었다. 이는 보존 실패 자료의 CPU 교정 기준이지 새 live의 고정 count가 아니다. 기존0.2.0 draft·wheel·run·실패와 원래 historical391 STALE는 보존한다.0.2.1의 CPU correction, 다운로드 수용, L1/L2/L3 live, 최종 PRIVATE 게시는 각각 외부 RELEASE_DECISION/영수증에 따른다. 문서의 정책 설명은 미실행 gate를 PASS로 표시하지 않는다.
