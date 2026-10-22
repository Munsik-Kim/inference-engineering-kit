# 기존 획득 자료의 CPU 분석 revision

`bench suite postprocess`는 이미 보존한 Suite raw를 새 분석 workspace에서
채점 → token 분석 → index 등록 → 오프라인 report 순으로 처리한다.
모델·GPU·tokenizer·추론 HTTP·campaign ledger 작업은 없다.
과거 live checkpoint를 새 source로 재개하는 migration 기능이 아니다.

```sh
inference-kit bench suite postprocess --run <immutable-raw-directory> \
  --historical-index <readonly-prior.sqlite> --output <new-analysis-directory> \
  --stop-after score_task
inference-kit bench suite postprocess-resume --analysis <analysis-directory>
inference-kit bench suite postprocess-status --analysis <analysis-directory> --output <new-status-directory>
```

동일 함수는 `inference_kit.api.suite_postprocess`, `suite_postprocess_resume`,
`suite_postprocess_status`다. `--historical-index`는 선택적이며 새 계획에 SHA를
동결한 후 SQLite readonly backup을 사용한다. `--analysis-revision`은 선택적인
명시 이름(1–128자)이다. 기본 revision은 raw SHA와 현재 analyzer source로 결정한다.
별도 이름을 요청하면 기존 acquisition을 유지한 새 analysis가 된다.

계획은 네 CPU operation만 허용한다. 새 source에서 과거 분석 workspace의 resume를
자동 승인하지 않는다. 각 단계의 raw/dependency/output identity와 nonempty coverage를
검사하고 completion을 저장한 뒤 checkpoint를 atomic replace한다. 정상 완료의
반복 resume는 checkpoint·수치 출력·DB를 다시 쓰지 않고 append-only event만 남긴다.
완료 영수증과 checkpoint 사이의 공백은 검증 후 복구한다. 이 보장은 전원 장애에 대한
분산 transaction이나 과거 모든 버전의 자동 이관 인증이 아니다.

`token_probe_raw_v1` reader의 `configured_flat_field_membership_v2`는 동일한
legacy field key의 여러 origin을 보존하면서 해당 token의 membership를 한 번
내보낸다. 서로 다른 field overlap은 유지한다. 표현할 수 없는 structured namespace는
거절하며 C1 validator·수식·허용차를 완화하지 않는다. 물리 token 분모와 field별
membership 분모를 구분한다.

Suite index 입력은 표시 alias `suite-current`를 canonical source별 scope에서
해석한다. scope 내 충돌과 legacy global alias 충돌은 계속 거절한다. 동일 raw의
다른 locator는 새 acquisition이 아니며 현재 source/revision의 재등록도 중복되지 않는다.
과거와 현재의 analysis는 별도 ID로 병존한다. 여러 revision을 합쳐 task 분모나
종합 품질 점수를 만들지 않는다.

명시적인 MINI SLICE는 포함한 HTTP/token 범위만 `bench suite replay`로 재계산한다.
`suite_all_scalar_selected_full_slice_v1`은 독립 manifest·parent SHA·coverage를
검사하며 누락 parent 파일을 통과시키는 옵션이 없다. 이 slice를 전체 parent
acquisition으로 index 등록하는 것은 거절한다. full vocabulary는 보존된 위치에서만
계산하며 원래 full 범위의 수치를 slice에 복사하지 않는다.

실제 모델의 품질, 배포 kernel 동등성, 신규 모델 지원, raw-text 준비는 이 경로가
검증하지 않는다. RTN 내부 token 관측의 decoded/effective BF16 경로와 native
SERVE 경로는 계속 별도 acquisition이다.
