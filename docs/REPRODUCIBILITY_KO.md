# 근거 검산과 재현

이번 릴리스의 새 GPU 실행/모델 다운로드는 0입니다. CPU raw 재계산은 과거 GPU 측정을 새 패키지로 반복 실행한 것이 아닙니다. 당시 runtime/argv/source hash, 후속 수정 source, 새 maintained source를 구분합니다.

Git의 `tests/fixtures/recorded-step04`는 outcome과 무관한 frozen 순서 앞 2문항, R00/R11 16응답만 가진 작은 recorded slice입니다. wheel은 synthetic demo resource만 포함합니다. recorded/full raw는 wheel/sdist에 넣지 않습니다.

권한 있는 계정으로 수동 취득:

```bash
gh release download v0.1.0-internal --repo Munsik-Kim/inference-engineering-kit --pattern '*.zip' --pattern 'SHA256SUMS.txt' --dir evidence-download
cd evidence-download
sha256sum -c SHA256SUMS.txt --ignore-missing
cd ..
inference-kit verify --bundle evidence-download/STEP04_SMOL_CLIPPING_ABLATION_20261015.zip --output runs/verify-step04-01
inference-kit replay --case step04 --bundle evidence-download/STEP04_SMOL_CLIPPING_ABLATION_20261015.zip --output runs/replay-step04-01
inference-kit replay --case step03 --bundle evidence-download/STEP03_SANITIZED_EVIDENCE_20261015.zip --output runs/replay-step03-01
inference-kit replay --case step02 --bundle evidence-download/STEP02_QWEN_W4_UPDATED_20261014.zip --output runs/replay-step02-01
inference-kit assess --index runs/replay-step03-01/portable-index.json --requirements runs/replay-step03-01/requirements/point.json --output runs/decision-step03-01
```

`--expected-sha256 <evidence/index.json의 값>`을 verify/replay에 추가하면 byte identity까지 검사합니다. index의 SHA를 신뢰할 수 있는 checkout/영수증과 대조해야 합니다. self-consistent manifest만으로 원본 신원을 인증하지 않습니다. 다운로드는 import/CI에서 자동 실행하지 않습니다.

안전 경계: ZIP member 수/압축후 비율/개별·전체 크기, duplicate/casefold, traversal/절대/Windows 경로, symlink/special file, declared size, manifest/reference hash를 검사합니다. archive 안 .py를 자동 import/실행하지 않습니다. package replay는 설치된 코드만 사용합니다. 결과는 새 output에 쓰며 frozen raw를 수정하지 않습니다.

STEP02 원본 ZIP은 root manifest가 없으므로 inventory-only 상태와 index의 선언된 evidence file 검증을 따로 기록합니다. STEP03 정제본은 개인 local.json 2개만 제외했습니다. original BUNDLE_MANIFEST는 provenance 안에 보존하고 새 BUNDLE_MANIFEST/EXPORT_RECEIPT를 생성했습니다. raw 숫자·원본 GPU source hash는 바꾸지 않았습니다. 원본 ZIP SHA와 정제 ZIP SHA는 서로 다릅니다.

새 reference 검사에서 STEP03 안에 보관된 과거 STEP02 index의 일부 참조 파일이 이 제한된 retained 묶음에 없음을 발견했습니다. 기존 VERIFY_CPU는 현재 STEP03의 evidence만 확인했습니다. 새 verify는 primary STEP03 index를 전부 확인하고, 누락된 archival index는 `PRIMARY_VERIFIED_ARCHIVAL_PARTIAL` 및 파일별 issue로 남깁니다. 과거 STEP02 전체를 이 asset 하나로 검증했다고 주장하지 않습니다. STEP02 전체 자산에서는 해당 참조를 별도로 검증합니다. primary 참조 손상은 오류로 차단합니다.

명시적으로 검토한 기존 VERIFY_CPU.py 3개는 이번에 fresh temporary extraction에서 별도로 실행했습니다. 이는 일반 archive 실행 기능이 아니며 자동화 API로 노출하지 않습니다. 원본 evidence 폴더에서 실행하면 출력이 생길 수 있으므로 직접 실행하지 마세요. verifier SHA/결과는 validation receipt에 남깁니다.

STEP02는 TEST1280 document의 raw likelihood/paired 결과와 최신 W4 3회·222요청/12판정을 대조합니다. STEP03는 12서버·480성능요청/64sanity평가/다중 요구조건4종을 재계산합니다. STEP04는 6×DEV256·6144본평가 응답/192sentinel 응답, 국소32source 집계/contrast/decision을 대조합니다. 정오답·상태·count는 정확히, finite float는 사전 고정 tolerance로 비교합니다.

새로운 모델/실측/신뢰구간 탐색은 수행하지 않습니다. 원본 실패 attempt, 공유 GPU 영향, worker UNKNOWN, semaphore warning, kernel selected-log, INCONCLUSIVE/HOLD가 보존됩니다. full raw와 최소 fixture는 각각 다른 범위이며 slice 성공을 full 재현으로 승격하지 않습니다.
