# 0.2.1의 명시적 로컬 live 실행

기본 wheel의 CPU 성공과 live 권한·준비는 별개다. 이 절차는 이미 보유한 pinned Smol BF16/RTN 및 Qwen BF16/GPTQ만 사용한다. 다운로드·runtime 업그레이드·다른 프로세스 종료·source migration은 하지 않는다. MAIN9 원문/gold/scorer는 유지하며 모델별 tokenizer로 새 준비한다. release gate는 L1 Smol combined task+token, L2 Qwen serving/native ARC, L3 별도 Smol standalone structure다. 실패한 gate를 뒤늦게 제거하지 않는다.

## 경로와 생성 순서

아래 사용자 값은 승인된 실제 절대경로로 바꾼다. `<...>`가 남으면 helper가 거절한다. `PYTHON`은 pip/venv가 준비된 기존 CPU Python, `KIT`은 풀린 배포 키트다. `WORK`는 새 경로이며 원래 run을 재사용하지 않는다. 기존 runtime/site-packages는 건드리지 않는다.

```sh
PYTHON='/absolute/path/to/prepared/python'
KIT='/absolute/path/to/inference-kit-0.2.1'
WORK='/absolute/path/to/new-acceptance-work'
FAMILY=smol  # L1; L2는 별도 새 WORK에서 qwen
TOKENIZER_PRIOR='/absolute/path/to/verified-tokenizer-binding.json'
LOCAL_LIVE_INPUT='/absolute/path/to/filled-live-binding.json'
SESSION_STARTED_UTC='<actual-authorized-session-start-UTC>' # 실제 시작 ISO8601(+00:00)로 바꾼다; 미해결 placeholder는 거절
unset PYTHONPATH PYTHONHOME
"$PYTHON" -m venv "$WORK/controller"
PY="$WORK/controller/bin/python"
CONSOLE="$WORK/controller/bin/inference-kit"
"$PY" -m pip install --no-index --no-deps "$KIT/dist/inference_engineering_kit-0.2.1-py3-none-any.whl"
"$PY" -m pip install --no-index --no-deps --target "$WORK/worker-site" "$KIT/dist/inference_engineering_kit-0.2.1-py3-none-any.whl"
RAW="$KIT/fixtures/prepared/$FAMILY/RAW_INPUT.json"
PREPARATION_SPEC="$KIT/fixtures/prepared/$FAMILY/PREPARATION_SPEC.json"
TOKENIZER_LOCAL="$WORK/tokenizer-local.json"
SUITE_SPEC="$WORK/suite.json"
LOCAL_LIVE="$WORK/local-live.json"
SESSION="$WORK/session.json"
CLOCK="$WORK/task-clock.json"
"$PY" -I -B "$KIT/tools/release_live_session.py" session --started-utc "$SESSION_STARTED_UTC" --output "$SESSION"
```

`SESSION_STARTED_UTC`는 CI·승인 대기를 포함한 현재 전체 작업의 시작이다. 처음 명령을 시작할 때 UTC를 기록하고 그대로 쓴다. 새 task clock을 만들며 전체3600초/신규 launch3000초 상한을 연장하지 않는다. helper는 원래 UTC를 현재 monotonic에 대응시킨 방법·시각과 host/boot hash를 기록한다. 과거의 실제 monotonic을 복원했다고 주장하지 않는다. 현재 release 작업의 상한이 더 먼저 끝나면 그 상한이 우선한다. 별도 승인 없는 새 세션으로 예산/시간을 초기화하지 않는다.

`TOKENIZER_PRIOR`는 기존 검증된 tokenizer/runtime/config/file pin과 model_metadata를 가진 외부 입력이다. 아래 완전한 CPU 명령은 원본 binding을 바꾸지 않고, 새 worker-site의 같은 wheel source만 연결한 새 파일을 만든다. tokenizer/model을 로딩하지 않는다.

```sh
"$PY" -I -B - "$TOKENIZER_PRIOR" "$WORK/worker-site" "$TOKENIZER_LOCAL" <<'PY'
import sys
from pathlib import Path
from inference_kit.core.io import read,write,digest
from inference_kit.probes.input_preparation import validate_binding
from inference_kit.suites.bench05a_plan import source_identity
value=read(Path(sys.argv[1])); validate_binding(value,verify_package=False)
value['package_site']=str(Path(sys.argv[2]).absolute())
value['toolkit_source_identity']=digest(source_identity())
validate_binding(value); write(Path(sys.argv[3]),value)
PY
"$PY" -I -B -m inference_kit bench inputs prepare --spec "$PREPARATION_SPEC" --input "$RAW" --tokenizer-local "$TOKENIZER_LOCAL" --output "$WORK/prepared"
"$PY" -I -B -m inference_kit bench inputs validate --prepared "$WORK/prepared/PREPARED_MANIFEST.json" --output "$WORK/prepared-check"
"$PY" -I -B -m inference_kit bench inputs describe --prepared "$WORK/prepared/PREPARED_MANIFEST.json" --output "$WORK/prepared-description"
```

prepare만 지정된 별도 local-only CPU tokenizer runtime에서 tokenizer SDK를 실행한다. generation prompt-only와 teacher-forced gold를 독립 준비하고 ARC choice 전부를 보존한다. hidden gold, truncation/padding, 재렌더 template, score repair를 추가하지 않는다. 한국어/NFC span unavailable은 근거 없는 좌표로 채우지 않는다.

## 실제 자료에서 Suite와 binding 생성

`LOCAL_LIVE_INPUT`는 모델별 `examples/LOCAL_LIVE_SMOL.template.json` 또는 Qwen template을 사용자가 채운 새 파일이다. `prepared_manifest/prepared_manifest_sha256/task_clock` 세 항목은 아래 helper가 생성한다. 나머지 artifact/runtime Python, worker-site, parent ledger/campaign/reservation root, GPU index/UUID, 승인된127.0.0.1 port는 실제 값이어야 한다. 이번 L1 필수 수용시험에서는 검토한 누적 `historical_index`를 유지한다. current-only/빈 DB로 바꾸어 필수 누적 report gate를 대체하지 않는다. 새 일반 작업에서 역사 비교가 원래 필요 없을 때만 이 선택 필드를 제거한다. CPU_LOCAL을 여기에 쓰지 않는다.

L1의 Suite template은 `examples/smol-live-combined.suite.template.json`, L2는 `examples/qwen-task.suite.json`이다. helper는 새 SOURCES 전체의 sample/source ID를 읽고 요청/출력 cap을 계산한 뒤 기존 compiler로 대조한다. 과거6/110/44나 MAIN9 관측 수를 통과 상수로 넣지 않는다. 준비된 자료가 부족하면 중단한다. maintained Suite template의 `outputs.report_packaging`은 `examples/REPORT_PACKAGING.json`과 같은 명시적 bounded collection 정책이며 처음 계획할 때 동결한다. 실행 중 cap/mode를 바꾸지 않는다. 모든 evidence part와 root를 함께 전달하는 [포장 계약](REPORT_PACKAGING_KO.md)을 따른다.

```sh
SUITE_TEMPLATE="$KIT/examples/smol-live-combined.suite.template.json"
# L2: SUITE_TEMPLATE="$KIT/examples/qwen-task.suite.json"
"$PY" -I -B "$KIT/tools/release_live_session.py" suite --template "$SUITE_TEMPLATE" --sources "$WORK/prepared/SOURCES.json" --output "$SUITE_SPEC"
"$PY" -I -B -m inference_kit bench suite plan --suite "$SUITE_SPEC" --sources "$WORK/prepared/SOURCES.json" --local "$KIT/examples/CPU_LOCAL.json" --output "$WORK/plan"
"$PY" -I -B -m inference_kit bench suite validate --plan "$WORK/plan/FROZEN_PLAN.json" --sources "$WORK/prepared/SOURCES.json" --output "$WORK/plan-check"
"$PY" -I -B -m inference_kit bench suite describe --plan "$WORK/plan/FROZEN_PLAN.json" --output "$WORK/description"
# CI와 파일 준비가 끝난 뒤, 실제 live session 직전의 새 native clock
"$PY" -I -B "$KIT/tools/release_live_session.py" clock --session "$SESSION" --output "$CLOCK"
"$PY" -I -B "$KIT/tools/release_live_session.py" local --binding "$LOCAL_LIVE_INPUT" --prepared "$WORK/prepared/PREPARED_MANIFEST.json" --session "$SESSION" --clock "$CLOCK" --output "$LOCAL_LIVE"
"$PY" -I -B -m inference_kit bench suite bind --plan "$WORK/plan/FROZEN_PLAN.json" --local "$LOCAL_LIVE" --output "$WORK/binding"
FIRST_SERVING_UNIT=$("$PY" -I -B "$KIT/tools/release_live_session.py" first-unit --binding "$WORK/binding/BINDING_PLAN.json")
```

bind는 CPU 파일/config/stat 검사다. 모델/GPU/실제 runtime/socket readiness 인증이 아니다. 첫 launch 전 실제 controller와 child의 승인 범위에서 socket create/bind/listen/connect/cleanup, runtime importability·CPU RAM/disk, GPU UUID/외부 compute·소유권·VRAM, port, 최신 ledger와 cleanup120초를 확인한다. EPERM/EACCES는 정식 좁은 승인 또는 BLOCKED이며 EADDRINUSE와 구분한다. 다른 작업은 종료하지 않는다. 최신 parent ledger를 읽고 자기 continuation만 만든다. L2/L3 parent는 앞 profile 완료의 실제 continuation을 이어야 하며, 기존7200초 campaign과 통합900초/각 profile600초 중 더 작은 잔여를 지킨다. helper는 예산 예약·debit·GPU 조회를 하지 않는다.

## 정상 경계 pause, 새 process resume, 완료 no-op

각 explicit GPU run/resume 직전에 `check`를 실행한다. helper와 계획은 실행 권한이 아니며 실제 제품의 current source/input/completion/ledger/자원 검사도 유지된다. outer session 잔여가 job cap+cleanup보다 작으면 새 job을 시작하지 않는다. native task clock은3600/3000 계약을 유지하지만 enclosing session 상한이 더 엄격하게 우선한다.

```sh
"$PY" -I -B "$KIT/tools/release_live_session.py" check --session "$SESSION" --clock "$CLOCK"
"$PY" -I -B -m inference_kit bench suite run --plan "$WORK/plan/FROZEN_PLAN.json" --local "$LOCAL_LIVE" --output "$WORK/run" --allow-gpu --stop-after-node "$FIRST_SERVING_UNIT"
# 정상 pause/cleanup 후 새 controller process; source/plan/clock을 바꾸지 않는다
"$PY" -I -B "$KIT/tools/release_live_session.py" check --session "$SESSION" --clock "$CLOCK"
"$PY" -I -B -m inference_kit bench suite resume --run "$WORK/run" --local "$LOCAL_LIVE" --allow-gpu
"$PY" -I -B -m inference_kit bench suite status --run "$WORK/run" --output "$WORK/status-complete"
"$PY" -I -B -m inference_kit bench suite resume --run "$WORK/run"
"$CONSOLE" bench suite resume --run "$WORK/run"
"$PY" -I -B -m inference_kit bench suite replay --run "$WORK/run" --output "$WORK/replay"
```

완료 API 호출은 `inference_kit.suite_resume(WORK/'run')`, `suite_status`를 사용한다. disk checkpoint와 API/module/console이 COMPLETE_VALIDATED여야 한다. 정상 no-op에서는 새 POST/GPU조회/worker/model load/ledger 차감·DB중복이0이고 checkpoint/raw/completion/publication bytes는 불변이어야 한다. append-only events/lock은 별도다. 통합 run은 scorer→token/native C1→자동 raw-only publication→index/report current source RAW_HASH_VERIFIED까지 끝나야 한다. 수동 raw 복사/별도 보정으로 성공시키지 않는다. 오답은 기술 실패와 구분하고 자동 retry하지 않는다.

## L3: 별도 Smol standalone 선택 구조

이 gate는 Suite 구조 지원으로 확대하지 않는다. 사용자 외부 파일 `STRUCTURE_SPEC`, `STRUCTURE_INPUTS`, `STRUCTURE_MODELS`, `STRUCTURE_LOCAL_INPUT`를 기존 검증된6입력/3layer/2head/2query/OFF1·OFF2·ON recipe로 고정한다. 파일 경로를 실제 값으로 정의하고 새 `STRUCTURE_WORK`와 clock을 만든다. 기존 source/raw를 수정하지 않는다.

```sh
STRUCTURE_SPEC='/absolute/path/to/pinned-structure-spec.json'
STRUCTURE_INPUTS='/absolute/path/to/preserved-probe-inputs.json'
STRUCTURE_MODELS='/absolute/path/to/pinned-smol-pair.json'
STRUCTURE_LOCAL_INPUT='/absolute/path/to/filled-structure-local.json'
STRUCTURE_WORK='/absolute/path/to/new-structure-work'
STRUCTURE_CLOCK="$STRUCTURE_WORK/task-clock.json"
STRUCTURE_LOCAL="$STRUCTURE_WORK/local.json"
"$PY" -I -B "$KIT/tools/release_live_session.py" clock --session "$SESSION" --output "$STRUCTURE_CLOCK"
"$PY" -I -B - "$STRUCTURE_LOCAL_INPUT" "$STRUCTURE_CLOCK" "$STRUCTURE_LOCAL" <<'PY'
import sys
from pathlib import Path
from inference_kit.core.io import read,write
value=read(Path(sys.argv[1])); value['task_clock']=str(Path(sys.argv[2]).absolute())
write(Path(sys.argv[3]),value)
PY
"$PY" -I -B -m inference_kit bench structure plan --spec "$STRUCTURE_SPEC" --inputs "$STRUCTURE_INPUTS" --models "$STRUCTURE_MODELS" --local "$STRUCTURE_LOCAL" --output "$STRUCTURE_WORK/plan"
"$PY" -I -B "$KIT/tools/release_live_session.py" check --session "$SESSION" --clock "$STRUCTURE_CLOCK"
"$PY" -I -B -m inference_kit bench structure run --plan "$STRUCTURE_WORK/plan/PLAN_FROZEN.json" --output "$STRUCTURE_WORK/run" --allow-gpu
"$PY" -I -B -m inference_kit bench structure replay --run "$STRUCTURE_WORK/run" --output "$STRUCTURE_WORK/replay"
```

구조 local의 runtime/worker-site/최신 parent ledger/task origin은 현재 설치와 chain을 명시해야 한다. 원래 recipe의 작은600초 이하 상한·cleanup·OFF/OFF/ON observer 검증과 selected token/full 좌표를 유지한다. RTN은 DECODED_WEIGHT_DIAGNOSTIC/effective BF16이며 packed kernel 직접 관측이 아니다. L3 실패 후 gate를 삭제하지 않는다.
