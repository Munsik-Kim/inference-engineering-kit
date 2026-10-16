# dev3 게시 검증 기록

CPU 수정본과 기존 BENCH05-A 의존 소스·테스트를 같은 비공개 저장소에 게시한다. 기존 v0.1.0 릴리스 자산과 historical GPU 원시 기록은 변경하지 않는다. 이번 게시에서 GPU 실행·모델 다운로드는 없다.

첫 [3버전 CI run38830424398](https://github.com/Munsik-Kim/inference-engineering-kit/actions/runs/38830424398)은 세 버전 모두 source 계약의 같은 테스트에서 실패했다. 작은 metadata fixture가 실제 모델의 RAM10GiB·디스크4GiB 조건을 그대로 상속했고, 준비 계획이 자원 부족으로 차단되어 shared-GPU 거절 경로에 도달하지 못했다. 로컬의 충분한 자원이 이 의존성을 가렸다. 실패 run과 receipt를 보존한다.

합성 fixture에만 작은 자원 조건을 부여했다. 실제 suite의 RAM/디스크 조건은 바꾸지 않았다. 별도 회귀는 요구 RAM이 가용량보다 클 때 계획을 BLOCKED로 기록하고 GPU 조회·소유 프로세스 시작·예산 변경 전에 실행을 거절하는지 확인한다. 새 테스트는124개(기존123+자원 차단 회귀1)이며 선택 tensor runtime이 없는 CPU runner에서는12개가 명시적으로 skip된다.

로컬 source와 일반 설치 wheel에서 해당4개 plan/run 계약을 통과했다. 최신 원격 3버전의 source·wheel·consumer·sdist 결과와 wheel 자산은 [수동 CPU workflow](https://github.com/Munsik-Kim/inference-engineering-kit/actions/workflows/cpu.yml)의 run/commit별 기록을 따른다. 과거123개 로컬 검증을 새 원격124개 검증과 합산하지 않는다. CI에는 작은 STEP04 recorded slice만 포함하며, 전체 STEP02~04와 실제 BENCH raw 재계산은 로컬 receipt의 별도 범위다.

Node24 Action 실행은 첫 run의 setup에서 성공했다. 이것은 새 GPU runtime 검증이나 배포 품질 인증이 아니다. HOLD_CLIPPING_VARIANT와 기존 품질 INCONCLUSIVE를 유지한다.
