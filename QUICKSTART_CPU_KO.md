# 다운로드 없는 CPU 빠른 시작

키트를 새 폴더에 풀고 그 폴더에서 실행한다. Python 3.10 이상과 로컬 pip/venv가 필요하다. 아래 pip는 키트의 wheel만 일반 설치하며 인터넷·의존성 다운로드를 쓰지 않는다. `PYTHON` placeholder를 실제 준비된 interpreter로 바꾼 뒤 실행한다. 시스템 Python에 ensurepip/venv가 없으면 기존의 다른 로컬 Python을 사용한다. 이 키트는 apt/pip 다운로드로 환경을 설치하지 않는다. 모델·tokenizer·ML SDK는 필요 없다.

```sh
# 먼저 지정: venv/ensurepip와 pip가 준비된 로컬 Python의 절대 경로
PYTHON=/absolute/path/to/prepared/python
unset PYTHONPATH PYTHONHOME
"$PYTHON" -m venv ../inference-kit-cpu
../inference-kit-cpu/bin/python -m pip install --no-index --no-deps dist/inference_engineering_kit-0.2.1-py3-none-any.whl
../inference-kit-cpu/bin/python -I -B -m inference_kit --version
../inference-kit-cpu/bin/inference-kit --version
```

두 출력은 `inference-kit 0.2.1`이다. 새 output 경로를 쓴다. 완료된 CPU workspace 반복은 D의 명시 resume가 검사한다. 같은 새 출력 폴더에 초기 명령을 덮어쓰지 않는다.

세 인터페이스에서 같은 경로를 검증하는 닫힌 진입점:

```sh
../inference-kit-cpu/bin/python -I -B acceptance/run_cpu_acceptance.py --kit . --output ../cpu-api --mode api
../inference-kit-cpu/bin/python -I -B acceptance/run_cpu_acceptance.py --kit . --output ../cpu-module --mode module
../inference-kit-cpu/bin/python -I -B acceptance/run_cpu_acceptance.py --kit . --output ../cpu-console --mode console
```

이 명시 script는 아래 네 고정 helper와 fd 회귀만 실행한다. archive 안 코드를 탐색 실행하거나 임의 module/shell을 받지 않는다. API 모드는 설치 public API, module/console 모드는 실제 새 프로세스 CLI를 사용한다. in-process SDK/socket 차단 hook은 자식이나 OS sandbox에 상속된다는 보장이 아니다. 실제 import origin과 각 명령을 영수증으로 남긴다.

## A. 실제 Qwen 응답

```sh
../inference-kit-cpu/bin/python -I -B -m inference_kit bench suite replay --run fixtures/qwen-native --output ../qwen-replay
```

36HTTP, output503token, native ARC226scalar/113pair의 저장 SLICE를 기존 scorer/C1으로 재계산한다. full 분포·attention은 NOT_COLLECTED다. index full-source importer는 이 partial parent run을 인증하지 않는다. 원래 acquisition·raw hash는 그대로이며 0.2.1 analyzer identity만 새로 기록된다.

## B. 실제 Smol 진단

```sh
../inference-kit-cpu/bin/python -I -B -m inference_kit bench tokens compare --view fixtures/smol-token --baseline bf16 --candidate rtn --output ../smol-token
../inference-kit-cpu/bin/python -I -B acceptance/replay_structure_slice.py --fixture fixtures/smol-structure --output ../smol-structure
```

scalar123pair, 전체 어휘3pair만 포함한다. 구조는 별도 D1 획득의24observation/12cross-arm·6same-arm 행이다. `replay_structure_slice.py`는 설치 제품의 blob/Observation/metric API를 호출하는 SLICE helper다. 전체 raw용 `bench structure replay` 명령을 대체하는 통합 기능으로 표시하지 않는다. [상세 명령과 범위](acceptance/SMOL_QUICKSTART_COMMANDS.md)를 따른다.

## C. 저장 prepared → 새 계획

```sh
../inference-kit-cpu/bin/python -I -B -m inference_kit bench suite plan --suite examples/qwen-task.suite.json --sources fixtures/prepared/qwen/SOURCES.json --local examples/CPU_LOCAL.json --output ../qwen-plan
../inference-kit-cpu/bin/python -I -B -m inference_kit bench suite validate --plan ../qwen-plan/FROZEN_PLAN.json --sources fixtures/prepared/qwen/SOURCES.json --output ../qwen-check
../inference-kit-cpu/bin/python -I -B -m inference_kit bench suite describe --plan ../qwen-plan/FROZEN_PLAN.json --output ../qwen-description
```

모델은 실행되지 않는다. Smol task 12POST/cap520, Smol token6forward/37target12full pair, Qwen task12POST/native19pair를 자료에서 계산한다. [세 계획의 실제 명령](acceptance/PREPARED_QUICKSTART_COMMANDS.md)에 나머지 명령이 있다. 저장 prepared 검사는 fresh tokenization이 아니며 parent FULL manifest를 빠진 파일과 함께 검증하지 않는다. 외부 artifact 존재는 NOT_CHECKED다.

## D. 공통 자동 게시·index/report·CPU 재개

```sh
../inference-kit-cpu/bin/python -I -B acceptance/replay_publication.py --kit . --output ../publication-api --mode api
```

별도 SYNTHETIC_CONTRACT_TEST 정적 run을 새 workspace로 복사해, public CPU postprocess를 score 경계에서 멈추고 새 controller로 resume한다. 자동 raw-only publication·현재 source 검증·빈 DB 등록·report·재등록/rebuild·완료 no-op를 검사한다. 명시한 `examples/REPORT_PACKAGING.json`으로 bounded collection을 만들고 root/part/member closure를 검증한다. 실제 A/B의 partial manifest를 full acquisition으로 위장하지 않는다. 새 실제 acquisition은0이다. raw-only 복사를 사용자가 수동 수행할 필요가 없다.

## 복사 실패·이동 검증

```sh
../inference-kit-cpu/bin/python -I -B acceptance/copy_error_regression.py -v
```

EEXIST/ENOSPC/EACCES와 fdopen 실패·정상/빈 파일·hash/길이 거절·public publisher 정리를 검사한다. 키트 전체를 다른 폴더로 복사한 뒤 같은 `--kit` 명령으로 replay/plan을 재실행할 수 있다. ID·bytes는 바꾸지 않고 물리 locator만 이동한다. 원본 fixture를 수정하지 않는다.

작은 report의 정적 상대링크·escape/local-only 검사는 영수증을 따른다. 브라우저가 없으면 시각 검증은 NOT_RUN이며 설치하지 않는다. CPU replay 성공은 과거 GPU cleanup·서버 동작의 재인증이 아니다. 현재 release live 결과는 별도 LIVE_ACCEPTANCE_RECEIPT에서 확인한다.

누적 실제 DB/전체 raw의 공개 CPU 명령과 모든 part의 이동 단위는 [REPORT_PACKAGING_KO.md](REPORT_PACKAGING_KO.md)를 따른다. 작은 키트의 SLICE가 full 누적285MB와 새 GPU를 재현하는 것은 아니다.
