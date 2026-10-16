# CPU 배포 수정 검증 — 0.1.1.dev3

게시 상태 안내: 아래 미게시·CI NOT_RUN 문구는 각 로컬 task 종료 당시 기록입니다. 이번 dev3의 원격 게시와 실제 3버전 실행 상태는 [CPU workflow](https://github.com/Munsik-Kim/inference-engineering-kit/actions/workflows/cpu.yml)의 run/commit별 결과를 확인합니다. 기존 릴리스 자산과 historical GPU source identity는 유지합니다.

2026-10-16 현재 작업본에서 확인한 배포 오류를 수정했다. 기존 BENCH05-A 변경을 보존한 개발 wheel이며, 기존 v0.1.0 릴리스 자산을 교체하지 않는다. 원격 push·commit·새 CI dispatch·모델 다운로드·GPU 실행은 없다.

## 수정한 동작

- JSON/JSONL 입출력과 BENCH05-A CSV/한국어 Markdown 보고서를 UTF-8로 고정했다. 관련 quality/STEP03/STEP04 raw reader에도 같은 계약을 적용했다. CLI JSON은 ASCII escape를 사용해 ASCII stdout에서도 의미가 같은 JSON을 전달한다.
- 테스트가 0개면 실패한다. 영수증은 수집 상태, 실제 package import 위치·버전·Python 버전과 허용한 GitHub 실행 문맥만 기록한다. 환경변수 전체를 수집하지 않는다. CI 환경변수는 자체 보고 정보이며 원격 실행의 독립 증명은 아니다.
- 수동 CPU workflow에 Python3.10/3.11/3.12 matrix, source와 설치 wheel의 전체 계약 테스트, 실제 외부 console 명령, sdist 재빌드의 package bytes 대조, 실패 시에도 7일간 receipt/wheel 보관을 추가했다.
- 기존 setuptools를 무조건 재사용하지 않도록 임시 runner의 build 의존성 범위를 명시했다. 로컬 연구 환경은 수정하지 않았다.
- Node20 경고가 발생했던 checkout/setup-python을 실제 공식 태그에서 확인한 Node24 SHA로 갱신했다. `workflow_dispatch`만 유지하며 권한은 `contents: read`, checkout credential은 비영속이다.

## 이번 로컬 실행 결과

| Python | source 계약 | 설치 wheel 계약 | 별도 base venv 소비자 |
|---|---:|---:|---|
| 3.10.20 | 111 pass / 12 optional skip | 111 pass / 12 optional skip | PASS |
| 3.11.15 | 123 pass / 0 skip | 123 pass / 0 skip | PASS |
| 3.12.3 | 111 pass / 12 optional skip | 111 pass / 12 optional skip | PASS |

고유 테스트123개(기존116+새 회귀7), 실행738회, 재실행615회, skip 실행48회이며 실패/오류0이다. 과거 릴리스·CPU audit·BENCH task의 실행 수를 합산하지 않았다. 새 회귀는 ASCII locale에서 한글 JSON 읽기/쓰기, JSONL 공개 비교, CLI 출력, quality/recipe raw reader, BENCH raw 재계산/한국어 보고서 및 테스트0건 실패를 확인한다.

base 소비자 venv에는 자체 wheel 하나만 일반 설치했다. 세 버전에서 `PYTHONPATH/PYTHONHOME`을 제거하고 원본 두 checkout 읽기와 ML SDK import를 audit hook으로 거절했다. demo/compare/assess/plan/verify/recorded replay/API 예제와 중복 output 거절을 검사했다. 실제 console 별도 프로세스도 help 및 5개 명령/거절 검사를 실행했다. audit hook은 해당 프로세스의 검사용이며 OS sandbox나 자식 프로세스까지의 읽기 차단으로 표현하지 않는다.

설치 wheel 계약의 HTTP/tensor 의존성은 기존 환경을 읽기 전용으로 재사용했다. 3.12 unit venv는 기존 CPU-only build 의존성 경로를 명시한 `.pth`를 사용했다. 깨끗한 base 소비자는 이 경로 없이 통과했으며 torch/vLLM/Transformers/NumPy/aiohttp가 설치되어 있지 않다. 선택 tensor 테스트의 skip은 GPU 모델 실행 실패가 아니다.

## 저장된 근거 재계산

일반 설치한 wheel에서 원본 source 읽기를 차단하고 다음을 새 output으로 재계산했다. frozen raw·기존 summary·과거 source identity를 수정하지 않았다. archive 내부 Python은 실행하지 않았다.

- STEP02: 과거 품질1,280문항, 최신 선택 성능222요청·3서버, pairing/선택기 판정 일치. 과거 shared-GPU attempt도 그대로 보존했다.
- STEP03: 과거12서버·480요청과 다중 조건의 조건부 후보/근거 부족 판정 일치. 일부 retained archive reference의 기존 누락 한계는 유지한다.
- STEP04: 과거 DEV256×6, 본HTTP6,144·sentinel192 및 holdout32source 재계산. `HOLD_CLIPPING_VARIANT` 유지.
- BENCH05-A: 실제 저장 raw345파일에서 token23쌍·attention16행·output16벡터·full-vocabulary15위치·bridge46위치를 재계산했다. 기존 summary와 정확히 같았다. 작은 소비자 fixture는 token23/attention16을 포함하지만 full distribution은 없어서 KL/JS를 채우지 않는다.

wheel과 sdist 재빌드의 package53개 멤버는 byte-wise 같았다. archive metadata까지의 byte 동일성은 요구하지 않는다. source/wheel 내용 일치와 모델·raw·환경·개인 local config 제외를 별도로 검사했다.

## 원격 상태와 남은 범위

기존 main d6c0b4be의 [CPU run38705852280](https://github.com/Munsik-Kim/inference-engineering-kit/actions/runs/38705852280)은 Python3.12에서 성공했다. 새 수정본과 3버전 workflow는 원격 미게시·미실행이다. YAML 구조·6개 shell block 구문과 공식 Action SHA/Node24 metadata를 로컬에서 검증했다. actionlint는 설치되어 있지 않아 실행하지 않았다.

GitHub 저장소 PRIVATE와 기존 릴리스 자산 보존을 읽기 전용으로 확인했다. 새 GPU 실행0, 모델/데이터/패키지 다운로드0, 기존 환경 변경0이다. 새 source hash의 GPU 경로는 CPU 계약 검증 범위이며, BENCH05-A dev2의 실제 관측을 dev3에서 측정한 것으로 바꾸지 않는다. 품질 `INCONCLUSIVE`와 연구 보류 판정을 유지한다. Python3.13+ 및 다른 OS, 새 matrix의 원격 성공은 아직 주장하지 않는다.

로컬 상세 receipt·wheel·task 시작 대비 patch와 인계 자료는 `/home/mskim/Codex_main/inference-kit-cpu-fix_20261016_080812/`에 있다. 새 workflow의 원격 확인에는 별도로 검토한 Git 반영이 필요하다.
