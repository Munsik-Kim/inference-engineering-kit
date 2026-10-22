# BENCH05-E4A: 원문을 준비 입력과 Suite 계획으로 연결

E4A는 기존 세 task의 UTF-8 원문과 외부 정답을 로컬 pinned tokenizer로 준비한다. 생성 입력, ARC 선택지 likelihood 입력, 외부 정답을 공급하는 진단 입력은 별도 필드로 유지한다. 준비 결과의 검증과 Suite 계획은 stdlib CPU 경로이며, tokenizer SDK는 명시적인 `inputs prepare`의 별도 worker에서만 사용한다.

지원 task는 `tech_record_extract 0.1`, `arc_challenge 1.0`, `context_evidence_retrieval 0.1`이다. 기존 scorer·prompt 정책은 유지한다. 새 모델이나 임의 template를 등록하는 기능은 없다.

## 입력과 세 가지 역할

`raw_task_manifest_v1`의 필수 최상위 필드는 `schema_version=1`, `kind`, `dataset_id`, `usage_scope`, 비어 있지 않은 `rows`다. 각 row는 다음 필드를 가진다.

|필드|의미|
|---|---|
|`task_id`, `task_version`|기존 task 계약|
|`dataset_id`, `source_id`, `sample_id`, `split`, `usage_scope`|원문과 사용 범위. ARC 과거 pool은 `REGRESSION_REUSE`|
|`policy_id`, `reference_source`|허용된 기존 prompt와 외부 정답의 출처|
|`sample`|해당 task의 원문·질문·선택지·외부 gold와 기존 검산용 metadata|

JSONL도 받는다. 첫 줄은 `rows`를 제외한 manifest header이고 다음 줄부터 row다. 중복 JSON key, 비유한 수치, 중복 namespace, 잘못된 타입과 지원하지 않는 필드는 거절한다. 새로운 일반 문서 parser나 자동 정답 생성기는 없다.

- 자연 생성: renderer가 문서·대상을 읽어 user content를 만들고 chat template를 한 번 적용한다. prompt-only를 독립 tokenize한다. gold를 붙인 joint IDs에서 생성 입력을 잘라내지 않는다.
- ARC: 원래 문항·모든 선택지·label 순서와 외부 `answerKey`를 보존한다. 원래 Smol completion context를 이미 렌더된 것으로 취급한다. context 끝 공백을 continuation으로 옮기는 기존 경계 규칙과 원래 choice character 길이 분모를 유지한다.
- 진단: task가 제공한 canonical reference를 teacher forcing하는 별도 joint sequence다. 생성 요청의 답변도, 신규 품질 측정도 아니다.

자연 생성 문서를 고정하고 숨겨진 gold만 바꾸면 generation content는 불변이고 reference만 변한다. 실제 데이터 준비에서는 독립 source/gold 검산도 통과해야 한다. 질문의 답이 문서 안에 있다는 사실은 gold suffix 누출과 다르다.

## 준비 설정과 로컬 binding

기본 설정 예시는 다음과 같다.

```json
{
  "schema_version": 1,
  "kind": "input_preparation_spec_v1",
  "policy_id": "existing_tasks_local_tokenizer_v1",
  "limits": {"max_prompt_tokens": 512, "max_reference_tokens": 128,
             "max_joint_tokens": 640, "context_tokens": 2048},
  "selection": {"policy_id": "field_round_robin_uniform_v1",
                "max_targets": 16, "max_full": 4},
  "seed": 909284
}
```

Retrieval은 기존 reference 64/joint 576 상한도 함께 적용한다. 자동 truncation, padding, 문항 삭제는 없다. full OFF(`max_full=0`)는 `NOT_REQUESTED`이며 필수 target 0과 구분한다. t의 정답은 t−1 logit에 대응한다. codepoint/UTF-8 offset과 경계 crossing은 저장된 정확한 token 단위로 유지한다.

`local_tokenizer_binding_v1`은 개인 로컬 locator다. 필수 필드는 `schema_version`, `kind`, `runtime_python`, `package_site`, `tokenizer_path`, `config_path`, `revision`, `model_id`, `tokenizer_files`, `config_sha256`, `toolkit_source_identity`, `model_metadata`이며 `timeout_seconds`는 선택이다. 네 locator는 명시적 절대경로를 요구한다. 개인 binding 파일은 배포 묶음에 넣지 않는다.

tokenizer/config 파일은 SHA로 pin한다. 지원 pinned revision은 `31b70e2e869a7173562077fd711b654946d38674`다. worker는 일반 설치한 자기 wheel의 source bytes가 controller와 같은지 검사한다. 로컬 tokenizer는 `local_files_only=True`, `trust_remote_code=False`, fast backend로 읽으며 GPU는 숨긴다. tokenizer 또는 runtime이 없으면 다운로드나 모델 API 대체 없이 거절한다.

## 실제 API와 CLI

아래 파일명은 사용자 작업 폴더에 준비한 명시적 입력의 예다. 모든 output은 새 경로여야 한다.

```bash
inference-kit bench inputs prepare --spec preparation.json --input raw.json \
  --tokenizer-local tokenizer-local.json --output prepared-new
inference-kit bench inputs validate --prepared prepared-new/PREPARED_MANIFEST.json \
  --output prepared-check
inference-kit bench inputs describe --prepared prepared-new/PREPARED_MANIFEST.json \
  --output prepared-description
inference-kit bench suite plan --suite suite-token-only.json \
  --sources prepared-new/SOURCES.json --local local-metadata-only.json --output token-plan
inference-kit bench suite validate --plan token-plan/FROZEN_PLAN.json \
  --sources prepared-new/SOURCES.json --output token-plan-check
inference-kit bench suite describe --plan token-plan/FROZEN_PLAN.json --output token-plan-description
```

Python은 같은 구현을 사용한다.

```python
from inference_kit import api
api.inputs_prepare("preparation.json", "raw.json", "tokenizer-local.json", "prepared-new")
api.inputs_validate("prepared-new/PREPARED_MANIFEST.json", "prepared-check")
api.inputs_describe("prepared-new/PREPARED_MANIFEST.json", "prepared-description")
api.suite_plan("suite-token-only.json", "prepared-new/SOURCES.json",
               "local-metadata-only.json", "token-plan")
```

Suite token 설정은 `selection="prepared_reference_positions_v1"`을 사용한다. 자연 생성 계획은 기존 prompt-only wire를, ARC 계획은 모든 choice를 사용한다. `permissions`는 `gpu_execution/network/tokenizer` 모두 `NONE`이다. 모델 파일을 확인하지 않는 metadata-only local 예시는 다음과 같다.

```json
{"schema_version":1,"artifact_check":"not_requested","artifact_roots":{},
 "runtime_check":"not_run","authorization":"NONE"}
```

## 출력과 확인된 범위

준비 디렉터리는 `RAW_INPUT.json`, `PREPARATION_SPEC.json`, `SOURCES.json`, task별 `prepared_task_source_v1`, `PREPARED_MANIFEST.json`, 준비·runtime·offset 영수증을 포함한다. source manifest는 자체 bytes/hash를 검증하는 authority이며 가짜 과거 acquisition이나 completion이 아니다. staging을 모두 검증한 뒤 새 디렉터리에 atomic 게시한다. 실패 staging과 blocker는 보존하며 성공 manifest로 포장하지 않는다.

Golden 세 문항은 기존 렌더 텍스트·wire IDs·ARC 전체 choice IDs·진단 IDs·경계·마스크·offset의 150개 named field가 exact였다. 새로운 selection policy·source/runtime/time·locator는 별도 provenance다. source evidence content와 출력 reference literal은 다른 span이며, 별도 evidence ID label token span이 없는 경우 이를 만들었다고 주장하지 않는다.

이번 MAIN 9문서의 task pair는 36 POST/출력 상한 1,560을 계획했고 token-only는 18 forward/paired target 123/full 36을 계획했다. SUBSET 3문서는 각각 12 POST/상한 520, 6 forward/37/12다. 모두 계획값이며 새 inference POST는 0이다. 합성 여섯 문서는 `ENGINEERING_INPUT_ONLY`, ARC 세 문서는 `REGRESSION_REUSE`다.

새 준비 자료는 `PREPARED_CONTRACT_VALIDATED_ONLY`, 계획은 `VALIDATED_NOT_EXECUTED`다. **`LIVE_BINDING_NOT_VALIDATED`**: E3의 기존 6문항·110/44 고정 live binder는 변경하지 않았다. 새 입력의 모델 실행, 구조 관측, 배포 동등성, 품질 일반화와 새 모델 지원은 미검증이다. MINI는 저장 prepared의 검증·계획을 재현하며 실제 재토큰화에는 명시한 로컬 tokenizer/runtime이 필요하다.
