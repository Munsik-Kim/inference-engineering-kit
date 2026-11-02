# inference-engineering-kit 0.2.1 시작

이 안내는 개발자·연구팀용 설치형 도구 **v0.2.1**의 시작 경로다. 지원하는 세 과제의 정해진 입력 형식과 외부 정답 기준을 사용한다. 버그 부재·임의 모델·모든 OS·일반 모델 품질을 보증하지 않는다.

먼저 **저장 응답의 CPU 재계산**으로 입력·출력 구조를 확인하세요. 모델 실행은 별도 단계입니다.

| 하고 싶은 일 | 시작 경로 | 필요한 것 |
|---|---|---|
| 기존 응답을 다시 채점 | 아래 첫 CPU 실행 | v0.2.1 키트, Python 3.10+, 새 출력 폴더 |
| 내 자료의 실행 계획 확인 | [CPU 계획 예제](QUICKSTART_CPU_KO.md#c-저장-prepared--새-계획) | 지원 task의 입력·외부 정답 계약 |
| 실제 모델 요청 | [Live 안내](QUICKSTART_LIVE_KO.md) | 별도 모델·tokenizer·runtime·HTTP 의존성, 명시 권한과 자원 |

현재 저장소와 v0.2.1 배포물은 공개되어 있습니다. 키트 안의 과거 PRIVATE 안내와 수용시험 이력은 당시 기록이며, 새 시작 안내는 이 웹 문서를 따릅니다.

## 배포물 받기와 SHA256 확인

1. [v0.2.1 release](https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1)에서 [사용 키트 ZIP](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/inference-engineering-kit-0.2.1-kit.zip)과 [SHA256SUMS.txt](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/SHA256SUMS.txt)를 같은 새 `downloads` 폴더에 저장한다. 기본 wheel은 키트의 `dist/` 안에 있으며, [wheel만 받기](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/inference_engineering_kit-0.2.1-py3-none-any.whl)도 가능하다. PyPI 설치를 안내하는 경로는 아니다.
2. 아래 명령으로 받은 키트의 SHA256을 release의 목록과 대조한다. 불일치하면 풀거나 설치하지 않는다.

```sh
python3 - <<'PY'
from pathlib import Path
import hashlib
root = Path("downloads")
name = "inference-engineering-kit-0.2.1-kit.zip"
rows = [line.split(maxsplit=1) for line in (root / "SHA256SUMS.txt").read_text().splitlines() if line.strip()]
expected = [digest for digest, filename in rows if filename.lstrip("*") == name]
if len(expected) != 1:
    raise SystemExit("Kit checksum is missing or ambiguous")
digest = hashlib.sha256()
with (root / name).open("rb") as reader:
    for block in iter(lambda: reader.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != expected[0]:
    raise SystemExit("Kit SHA256 MISMATCH")
print(name + ": OK")
PY
```

이 명령은 받은 키트 ZIP 하나만 확인한다. wheel·sdist를 별도로 받았다면 `name`을 그 파일명으로 바꾸어 각각 확인한다. 전체 자산이나 모델을 검증하는 명령은 아니다.

3. 확인한 키트를 새 폴더에 풀고, 아래 CPU 경로를 실행한다. `gh release download`는 선택 가능한 CLI 경로이며 공개 다운로드의 필수 인증 절차가 아니다.

```sh
python3 -m zipfile -e ./downloads/inference-engineering-kit-0.2.1-kit.zip ./inference-kit-0.2.1
```

## 첫 CPU 실행과 새 입력 준비

가장 작은 경로는 [README의 설치·Qwen 응답 재계산](README.md#배포물-받기와-첫-cpu-실행)입니다. 먼저 `CPU_REPLAY_RECEIPT.json`의 범위와 `score_task/summary.json`을 읽은 뒤 다른 예제로 이동하세요. 일반 clone의 루트에는 배포 wheel이 없으므로, 아래 설치 안내는 **검산 후 풀린 키트 루트**를 기준으로 합니다.

1. [QUICKSTART_CPU_KO.md](QUICKSTART_CPU_KO.md)에 따라 새 base venv에 wheel만 일반 설치한다. 모델·tokenizer·ML SDK는 필요 없다.
2. 실제 저장 Qwen 응답(A), Smol token/선택 구조(B), prepared 계획(C), 명시적 합성 publication/index/report 상태기계(D)를 실행한다. 각 coverage와 acquisition 출처가 다르다.
3. [SUPPORT_MATRIX_KO.md](SUPPORT_MATRIX_KO.md)와 정본 [SUPPORT_MATRIX.json](SUPPORT_MATRIX.json)에서 구현·과거 실측·현재 release CPU·현재 release live·외부 조건을 확인한다.
4. 실제 실행이 필요한 경우 [QUICKSTART_LIVE_KO.md](QUICKSTART_LIVE_KO.md)의 외부 asset과 모든 경로 변수를 먼저 채운다. source/권한/현재 자원/ledger gate를 통과한 명시 opt-in만 모델을 실행한다.

wheel은 stdlib CPU 기능을 제공한다. 모델 weights, tokenizer 전체 자산, ML runtime, credential, 실제 local 설정은 포함하지 않는다. `examples/CPU_LOCAL.json`과 저장 prepared SLICE는 계획·계약용이며 live-ready가 아니다. 원래 BENCH 폴더를 찾을 필요 없이 키트의 CPU 명령은 자립한다.

`KNOWN_LIMITATIONS_KO.md`의 decoded RTN/NFC/namespace/동일 source 재개 경계와 과거 실패 상태를 유지한다. 새 모델과 네 번째 task는 이 릴리스 scope 밖이다.

미게시 로컬 후보 `0.2.2.dev1`은 이 정식 `v0.2.1`과 별개다. 공개 접근만으로 새 후보의 코드·측정이 정식판에 반영되는 것은 아니다.

누적 보고서의 [포장 계약](REPORT_PACKAGING_KO.md)은 단일256MiB 모드와 별도 bounded collection의 payload/presentation/전체 상한을 구분한다. 모든 part를 함께 이동한다.

## 현재 판정부터 확인

[ACTIVE_RELEASE_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/ACTIVE_RELEASE_ACCEPTANCE_2.json) → [RELEASE_DECISION_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/RELEASE_DECISION_ACCEPTANCE_2.json) → [LIVE_ACCEPTANCE_2.json](https://github.com/Munsik-Kim/inference-engineering-kit/releases/download/v0.2.1/LIVE_ACCEPTANCE_2.json) 및 pointer에 선언된 관련 근거를 따른다. generic `RELEASE_DECISION.json`·`LIVE_ACCEPTANCE_RECEIPT.json`과 옛 pointer는 역사적 BLOCKED/PARTIAL을 가리킬 수 있다. 이 파일들은 보존하며 최신 판정으로 바꿔 읽지 않는다.


기존 kit·wheel·release 자산과 판정 파일을 교체하지 않습니다. 현재 source CPU workflow는 수동 실행이며, 최신 main의 자동 PR 회귀와 혼동하지 않습니다.
