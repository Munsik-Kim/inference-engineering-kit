# 이번 엔지니어링 릴리스의 검증

고유 CPU 테스트 81개를 검증했습니다. 총 실행 348회, 재실행 267회, optional tensor skip 실행 24회입니다. base suite에서는 tensor6개가 skip되며, 기존 별도 build runtime에서는 이6개도 CPU로 통과했습니다. 과거111개를 합산하지 않았습니다. 초기 sandbox loopback 제한·추출 import/type 처리·mock Content-Type 오류와 수동 수정 재검증 기록도 보존합니다. 자동 재시도0입니다.

일반 설치한 wheel을 별도 CPU venv와 /tmp consumer에서 실행했습니다. PYTHONPATH/PYTHONHOME을 제거하고 원본/target checkout 파일 읽기를 audit hook으로 거절한 상태에서 help/demo/plan/verify/recorded replay/compare/assess/자체 synthetic 업무 API를 검증했습니다. 같은 output은 거절했고 engine SDK import는 없었습니다. 전체 STEP02/03/04 raw replay도 설치 wheel만으로 source read를 차단하고 통과했습니다.

STEP02 TEST1280 document/최신 W4 222요청/12판정, STEP03 12서버·480요청/4판정, STEP04 DEV256×6·6144본응답/192sentinel/32holdoutsource의 scalar/count/pairing/정책을 대조했습니다. root manifest coverage와 archival reference 누락은 별도로 보고합니다. STEP03 내부 옛 STEP02 index 일부 참조는 retained subset에 없어 PRIMARY_VERIFIED_ARCHIVAL_PARTIAL입니다. 해당 STEP02 전체 자산에서는 검증됩니다.

reviewed legacy VERIFY_CPU.py3개는 임시 extraction에서 별도 실행해 모두 통과했습니다. package는 archive Python을 실행하지 않습니다. 새 수치 실험·TEST 확대·모델 다운로드·GPU 실행은 모두0입니다. maintained GPU 경로는 REFACTORED_CPU_CONTRACT_VERIFIED_GPU_NOT_RERUN입니다. GitHub CI는 CI_NOT_RUN이며 수동 workflow만 있습니다.

[VALIDATION.json](VALIDATION.json)에 범위와 결과를 연결했습니다. 게시 commit/PRIVATE/readback/clone/release asset 다운로드 검증은 target 바깥 PUBLISH_RECEIPT에 기록하므로 자기 commit SHA를 미리 넣지 않습니다.
