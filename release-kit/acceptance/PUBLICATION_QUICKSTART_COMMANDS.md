# 원시근거 게시·CPU 재개 예제

이 경로의 `fixtures/publication-demo`는 **SYNTHETIC_CONTRACT_TEST**입니다. 모델·tokenizer 실행으로 얻은 자료가 아니며 실제 Qwen/Smol SLICE와 합산하지 않습니다. 세 task 각 1문서·두 arm의 12개 합성 HTTP row와 74개 합성 scalar/24개 full vector가 들어 있습니다. 기존 제품 compiler와 검증·scorer·C1·SQLite·보고서를 실행하기 위한 닫힌 작은 예제입니다.

일반 설치한 후보 Python에서 다음 명령을 실행합니다. `KIT`은 이 사용 키트의 실제 위치이고 `RESULTS`는 새 결과 위치입니다.

```bash
python -I -B "$KIT/acceptance/copy_error_regression.py" -v
python -I -B "$KIT/acceptance/replay_publication.py" --kit "$KIT" --output "$RESULTS/publication-api" --mode api
python -I -B "$KIT/acceptance/replay_publication.py" --kit "$KIT" --output "$RESULTS/publication-module" --mode module
python -I -B "$KIT/acceptance/replay_publication.py" --kit "$KIT" --output "$RESULTS/publication-console" --mode console
```

`replay_publication.py`는 원본 fixture를 새 결과 폴더에 복사하고, 원시 manifest에 없는 64MiB+1 byte sparse 파생 파일을 추가합니다. 이어서 **제품 공개 postprocess**를 `score_task` 뒤에서 멈추고 새 Python controller로 나머지 CPU 작업을 재개합니다. 원시근거 게시·등록·보고서가 공통 제품 경로에서 실행됩니다. 완료 뒤 no-op, 원시근거 불변, 재등록·새 DB rebuild 일치, 파생 파일 증가 무영향과 변조 거절을 검사합니다. sparse 파일은 검사용 결과 폴더에만 생성됩니다.

직접 호출되는 제품 CLI는 다음과 같습니다. 첫 명령의 출력은 새 디렉터리여야 합니다.

```bash
python -I -B -m inference_kit bench suite postprocess --run "$RAW" --output "$ANALYSIS" --analysis-revision v1-candidate01-synthetic-publication --stop-after score_task
python -I -B -m inference_kit bench suite postprocess-resume --analysis "$ANALYSIS"
python -I -B -m inference_kit bench suite postprocess-status --analysis "$ANALYSIS" --output "$NEW_STATUS"
```

`RAW`는 위 예제에서 생성된 `mixed-raw`이며, 원래 fixture에 직접 파생 자료를 쓰지 않습니다. 실제 원시근거를 사용자가 수동으로 raw-only 복사하거나 SQL을 수정할 필요가 없습니다. 상태는 `CPU_PAUSED_BOUNDARY` → `CPU_POSTPROCESS_COMPLETE`이고 current source는 `RAW_HASH_VERIFIED`여야 합니다. 새 실제 acquisition/GPU/model/tokenizer/추론/예산 사용은 모두 0입니다.

감사 hook은 예제 Python 프로세스 안의 SDK import·socket 호출을 검사합니다. module/console과 새 controller는 명시적으로 실행한 별도 CPU child이며 hook 상속·OS sandbox 보호를 주장하지 않습니다. 과거 live controller의 완료나 no-op를 다시 인증하는 예제가 아닙니다.
