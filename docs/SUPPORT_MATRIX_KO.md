# 지원·검증 수준

| 영역 | Historical GPU evidence | 이번 릴리스 CPU 검증 | 미검증/미이식 |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 BF16/GPTQ W4 | STEP02, STEP03 | raw response 재채점·timing/selector parity | 새 wheel GPU 재실행 없음 |
| SmolLM2-1.7B-Instruct BF16/RTN/GPTQ | STEP02/04 | raw likelihood와 pairing/contrast parity | 다른 모델·다른 recipe 미검증 |
| Smol R00/R10/R01/R11 | STEP04 loader/quality | tiny compose tensor 출처/저장·재로딩, full raw replay | 새로운 실제 artifact 생성/forward 없음 |
| vLLM managed server | 과거 0.29.0/Torch2.13.0/Transformers5.17.0/CT0.17.0 | argv/identity/budget/owned CPU contracts | maintained GPU NOT_RERUN |
| tensor adapter runtime | build CT0.18.0의 packed 표현 | 기존 별도 runtime에서 tiny CPU fixtures | 기본 설치 시 torch/CT 없음 |
| SGLang | 이전 lab transport 자산 | synthetic HTTP parser 계약 | kit managed SGLang run 미지원 |
| CPU interpreter | 해당 없음 | Python3.12 Linux/WSL wheel consumer | 3.10/3.11/다른 OS 실측 미실행 |
| CI | 해당 없음 | 수동 workflow 파일/로컬 smoke | CI_NOT_RUN, 이번 원격 dispatch 없음 |

Python 지원 선언은 >=3.10입니다. 설치 호환성 전체가 실측되었다는 뜻은 아닙니다. base deps는 stdlib, `http` extra만 aiohttp입니다. GPU 패키지 설치/upgrade/자동 다운로드는 하지 않습니다. GPU runtime 경로를 symlink resolve로 대체하지 않습니다.

run은 준비된 runtime tuple, qwen3/llama, native BF16/검증된 pack-quantized scheme만 받습니다. 정확한 artifact identity·tokenizer와 profile이 맞아도 새 GPU 재검증을 생략했다는 상태는 유지합니다. source packaging의 성공을 GPU 품질/연구 우위/배포 적합성으로 승격하지 않습니다.

원본 runtime 경로/버전/argv는 release historical receipt를 참조합니다. 개인 경로를 포함한 pip freeze를 새 lockfile로 배포하지 않습니다. CPU build 의존성은 전용 격리 venv에만 설치했으며 기존 연구 환경은 수정하지 않았습니다.
