# CPU 경로 B: 저장 Smol 진단

`fixtures/smol-token`은 E4B의 실제 MAIN9 token-only 획득을 E4B-R1에서 교정한 C1 TokenView SLICE다. scalar 246행/123pair와 task별 결정적 1개 전체 어휘 pair, 총 3full pair만 포함한다. 9문서·두 arm의 scalar 범위와 전체 36full pair의 부분집합을 구분한다. 원래 ID·raw bytes·TokenView manifest는 그대로다.

키트 root에서 설치한 Python/console을 사용한다. 각 output은 새 경로여야 한다.

```sh
python -I -B -m inference_kit bench tokens compare --view fixtures/smol-token --baseline bf16 --candidate rtn --output outputs/smol-token-module
inference-kit bench tokens compare --view fixtures/smol-token --baseline bf16 --candidate rtn --output outputs/smol-token-console
```

같은 공개 API는 `inference_kit.tokens_compare('fixtures/smol-token', 'bf16', 'rtn', 'outputs/smol-token-api')`다. 세 호출의 numeric/ID/count/status 의미는 같고, 새 획득은 없다. scalar-only 위치에는 full KL/JS를 생성하지 않는다.

`fixtures/smol-structure`는 별도의 과거 D1 획득에서 고정 입력 순서의 task별 첫 문서, layer0/head0/last_prompt query를 선택했다. 3문서·24개 실제 numeric observation으로 attention/native AV/reconstructed AV/머리 축 없는 o_proj의 12개 BF16↔RTN 행과 6개 같은-arm AV 재구성을 검사한다. 원래 observation JSONL 전체와 parent manifest를 보존하여 선택한 ID·원시 bytes의 소속을 검사한다. 선택되지 않은 numeric 파일은 없다.

```sh
python -I -B acceptance/replay_structure_slice.py --fixture fixtures/smol-structure --output outputs/smol-structure
```

이 명시 helper는 설치 제품의 `Observation.validate`, `probes.blobs.load`, `metrics.attention.row_metrics/vector_metrics`를 호출한다. 수식을 복제하거나 모델·tokenizer를 실행하지 않는다. 과거 지표와 abs1e-10/rel1e-9로 비교하고 self-comparison·좌표 mismatch 거절도 확인한다.

구조 SLICE는 `bench structure replay`의 전체 raw run 입력이 아니다. 전체 `structure_replay`에는 별도로 보존한 완전한 D1 run/ZIP이 필요하다. 이 키트만으로 OFF/OFF/ON 통제·전체 D1 관측·live cleanup을 재인증하지 않는다. token과 structure를 같은 획득이나 같은 문항 집합으로 합치지 않는다. RTN 내부 진단은 DECODED_WEIGHT_DIAGNOSTIC/effective BF16이며 packed INT4 serving 내부와 동일하지 않다.
