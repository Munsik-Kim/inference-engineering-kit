# inference-engineering-kit 0.2.0

세 가지 task의 준비 입력·Suite 계획·승인된 로컬 추론·저장 raw 재계산·비교·원시근거 게시·index/report를 연결하는 Python 도구다. 기본 wheel은 외부 의존성 없이 CPU에서 저장 자료를 검증하고 다시 채점한다. Smol과 Qwen의 지원 범위는 [지원표](SUPPORT_MATRIX_KO.md)에 따르며, artifact·tokenizer·runtime·권한과 실행 근거를 별도로 기록한다.

임의 모델·새 양자화·네 번째 task·공용 inference 서비스는 제공하지 않는다. 제한된 engineering acceptance의 성공은 모델 품질·비열등성·양자화 우위·SLA·안정 API의 영구 보증이 아니다. Smol 구조 관측은 standalone 선택 진단이고 Qwen은 serving/native ARC scalar 범위다. [알려진 제한](KNOWN_LIMITATIONS_KO.md)과 과거 PARTIAL/STALE/INCONCLUSIVE를 유지한다.

## 비공개 배포와 첫 CPU 실행

이 저장소와 release는 **PRIVATE**다. 기존 `gh` 인증과 저장소 접근권한이 필요하다. 정식 게시 여부·검증된 commit/asset SHA는 release의 `RELEASE_DECISION.json`과 `SHA256SUMS`로 확인한다. draft는 모든 gate 통과 전의 배포 준비 상태이며 정식 통과를 뜻하지 않는다.

```sh
gh release view v0.2.0 --repo Munsik-Kim/inference-engineering-kit
gh release download v0.2.0 --repo Munsik-Kim/inference-engineering-kit --pattern 'inference-engineering-kit-0.2.0-kit.zip' --dir ./downloads
python3 -m zipfile -e ./downloads/inference-engineering-kit-0.2.0-kit.zip ./inference-kit-0.2.0
```

풀린 키트의 [START_HERE_KO.md](START_HERE_KO.md) → [QUICKSTART_CPU_KO.md](QUICKSTART_CPU_KO.md)를 따른다. 새 venv에 `dist/inference_engineering_kit-0.2.0-py3-none-any.whl`만 `--no-index --no-deps`로 설치한다. 저장된 실제 Qwen 응답, Smol 진단 slice, prepared 계획 및 명시된 합성 publication 예제를 실행하며 새 모델·tokenizer를 호출하지 않는다. 저장소에서는 같은 자료가 `release-kit/`에 있다.

Live는 별도 pinned model/tokenizer/runtime, approved GPU·localhost·현재 자원·ledger가 필요하다. [QUICKSTART_LIVE_KO.md](QUICKSTART_LIVE_KO.md)는 모든 경로 변수와 실행 직전 clock·첫 unit 추출 helper를 정의한다. 계획 파일이 실행 권한을 부여하지 않는다. weights·tokenizer 자산·ML SDK·실제 local 설정은 배포물에 없다.

## 사용 범위와 문제 보고

[기계 판독 지원표](SUPPORT_MATRIX.json) · [CPU 명령](QUICKSTART_CPU_KO.md) · [Live 명령](QUICKSTART_LIVE_KO.md) · [수용시험 명세](release-kit/acceptance/ACCEPTANCE_PLAN_V1.json) · [변경 이력](CHANGELOG_KO.md) · [라이선스/출처](THIRD_PARTY_NOTICES.md)

문제는 이 PRIVATE 저장소의 Issues에 package version, source digest, 실행한 명령, 기술 실패/모델 오답의 구분과 민감 경로를 제거한 작은 receipt를 남긴다. credential·개인 local binding·전체 모델/raw는 첨부하지 않는다. 원래 raw와 실패 상태를 덮어쓰지 않는다.

소스 검사는 `python -B tools/test_cpu.py --output <새 receipt.json>`으로 실행한다. HTTP mock 테스트만 선택 HTTP 의존성이 필요하며 기본 wheel CPU 의존성과 구분한다. 현재 릴리스의 실제 source/wheel/CI/live 결과는 해당 release의 영수증에 기록한다. 과거 테스트 수를 현재 PASS로 쓰지 않는다.

이전 dev3/0.1.0 설치·BENCH 기술 설명은 [보존 이력](docs/HISTORY_KO.md), [구조](docs/ARCHITECTURE_KO.md), [데이터 계약](docs/DATA_CONTRACTS_KO.md)에 있다.
