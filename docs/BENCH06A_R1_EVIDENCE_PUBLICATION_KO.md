# Suite 원시근거 자동 게시

공통 `execute_cpu('index_add', ...)`는 `RUN_MANIFEST.json`의 완전한 원시근거를 검증하여 파생 DB/report와 분리된 위치에 게시한 뒤 그 locator를 등록한다. 원래 획득 manifest·ID·bytes는 바꾸지 않는다. directory 전체를 읽는 기존 bounded reader와 그 상한도 그대로다.

Python API는 `inference_kit.analysis.evidence_publication.publish_suite_evidence(source_root, output_root, *, manifest_sha256=None, limits=Limits())`이다. 알려진 Suite manifest만 지원하며 임의 JSON inventory가 완전한 실험을 증명한다고 보지 않는다. 기존 prepared/frozen input·execution plan·nested manifest 의존성을 확인하고, 필수 파일 누락·잘못된 경로·symlink·수치/형식 오류를 거절한다. 명시한 source directory 밖의 `output_root/<manifest-sha>/`에 전체 선언 member와 원래 manifest를 stream copy한다. 선언 밖 파일은 인증하지 않는다. 정상 raw 자체가 기존 한도를 넘으면 지원하지 않는다.

복사는 원본 file identity/stat/hash를 전후 확인하며 temp sibling에서 완성한다. 모든 파일 검증 후 directory rename으로 게시한다. 같은 key가 이미 있으면 전체 목록/bytes/hash가 같을 때만 재사용한다. 다른 내용은 overwrite하지 않는다. 같은 key 게시의 exclusive lock을 사용하며 stale lock을 자동으로 훔치지 않는다. 실패하면 자신의 temp/lock만 정리한다. OS crash/power-loss 전체 내구성이나 악의적 다른 프로세스에 대한 filesystem sandbox를 인증하지 않는다. 불변성은 새 쓰기를 거절하고 내용을 매번 검증하는 계약이며, writable filesystem의 OS 보안 봉인이 아니다.

공통 후처리는 게시 receipt를 evidence 밖 `EVIDENCE_PUBLICATION_RECEIPT.json`에 남긴다. 새 CPU workspace는 stage의 형제 `published-evidence/`를 쓰고, live root 안의 index stage는 raw root 밖의 형제 publication root를 쓴다. 원본 source 안으로 재귀 복사하지 않는다. 새 DB의 `INPUT_SOURCES.json`은 게시본을 가리킨다. 이후 `CURRENT_SOURCE_VALIDATION.json`은 반드시 이번 게시 locator 자체가 `RAW_HASH_VERIFIED`임을 확인한다. 다른 과거 정상 locator로 이번 손상을 숨기지 않는다.

Report 전후와 CPU/live 완료 evidence를 다시 읽는 경로도 현재 게시본을 검사한다. current가 stale/missing이면 정상 완료를 반환하지 않는다. 과거 historical source의 stale 상태는 따로 보여주며 current의 검증으로 바꾸지 않는다. 숫자가 같은 다른 획득을 합치지 않고, locator 이동/재등록은 acquisition을 늘리지 않는다. 같은 raw를 새 source로 분석한 analysis revision은 명시적으로 별도 기록한다.

기존 공개 명령으로 사용할 수 있다. 기존 live run을 새 version으로 resume하지 않는다.

```sh
inference-kit bench suite postprocess --run <original-mixed-run> --output <new-cpu-workspace> --historical-index <readonly-parent.sqlite> --analysis-revision <explicit-new-revision> --stop-after score_task
inference-kit bench suite postprocess-resume --analysis <new-cpu-workspace>
inference-kit bench suite postprocess-status --analysis <new-cpu-workspace> --output <new-status-output>
inference-kit bench suite replay --run <original-mixed-run> --output <new-cpu-replay>
```

`postprocess`의 정확한 인자는 설치 CLI `--help`를 따른다. 기본 import/publisher/index/report/replay는 stdlib CPU 경로이며 모델·tokenizer·GPU·네트워크를 실행하지 않는다. 부분집합 manifest는 원래 전체 획득의 성공 증거가 아니다. 현재 금지된 native/full slice를 full parent acquisition으로 등록하는 제한은 유지한다. 기존 legacy slice 형식의 게시 receipt는 `DECLARED_LEGACY_SLICE`로 표시한다.
