# BENCH05-A 관측 계약과 재사용

이 개발본은 `diagnostic_continuation_v1`의 SmolLM2-1.7B BF16/원래 RTN 한 쌍을 대상으로 한다. BENCH05-B~E, BENCH06, ARC 업무 adapter나 임의 모델 지원은 포함하지 않는다. 과거 HOLD_CLIPPING_VARIANT 및 품질 INCONCLUSIVE는 유지한다.

## 코드 경계

| 책임 | 코드/API | CPU 기본 import |
|---|---|---|
| Sample/Execution/Observation/MetricSpec/MetricResult | `probes.contracts` dataclass | stdlib |
| dtype/shape/endianness/bytes/SHA가 있는 F32 blob | `probes.blobs` | stdlib array/struct 계열 |
| 모델·artifact·runtime 조합의 기능 | `models.capabilities` | source/config에 근거한 계획; 실측은 별도 receipt |
| 고정 profile·경로·ledger 검산 | `suites.bench05a_plan`, `bench_plan` | 모델/네트워크/서브프로세스 없음 |
| 소유 프로세스·시간·GPU 감시 | `suites.bench05a_run`, `bench_run` | 명시적 opt-in일 때만 외부 runtime 실행 |
| 토큰화·offset 준비 | `probes.prepare_tokens` | 별도 tokenizer runtime; CPU 명령에서만 lazy import |
| strict 모델 적재와 packed decode | `probes.weights` | worker에서만 Torch/CT/safetensors |
| 선택 post-RoPE Q/K/V와 원래 SDPA 출력 관측 | `probes.attention`, `probes.local_transformer` | worker에서만 Torch/Transformers |
| 원시 관측 재계산 | `metrics.token`, `metrics.attention`, `bench_replay` | stdlib, CUDA/NumPy 불필요 |

`bench_plan/bench_replay`는 CLI와 동일 API다. 관측 함수와 계산 함수는 분리되어 있고, generic metric에는 모델명·layer 번호가 없다. 첫 live adapter의 구조 gate는 Smol/MHA와 설치본 Transformers5.14.1·compressed-tensors0.18.0으로 한정된다. toy GQA 테스트는 live GQA 지원이 아니다.

## 준비와 실행

원본 연구 checkout의 import, sys.path 주입 또는 fallback lookup은 없다. artifact receipt의 BF16_REF/R00 레코드를 명시적으로 입력하며 모델/recipe/tokenizer identity와 실제 파일 조회를 구분한다. 기존 Python symlink 문자열을 그대로 사용한다. 상대경로는 local/suite의 명시적 root 기준이다. 개인 local 파일은 Git/ZIP에서 제외한다.

로컬 tokenizer를 가진 기존 runtime에서 다음 준비 명령만 별도로 실행한다. 모델·CUDA를 사용하지 않는다. 파일이 이미 있으면 덮어쓰지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /path/to/existing-runtime/bin/python -m inference_kit.probes.prepare_tokens --samples configs/bench05a/samples.json --tokenizer /path/to/pinned-tokenizer --config /path/to/bf16/config.json --output /path/to/new/prepared-samples.json --runtime-output /path/to/new/runtime.json
inference-kit bench plan --suite configs/bench05a/suite.json --models configs/bench05a/pair.json --local /path/to/private-local.json --output /path/to/new-plan
inference-kit bench run --plan /path/to/new-plan/PLAN_FROZEN.json --output /path/to/new-run --allow-gpu
inference-kit bench replay --run /path/to/new-run --output /path/to/new-replay
```

기존 artifact.json에서 확인한 identity/recipe/config SHA와 tokenizer metadata를 담은 `artifact-receipt.json`이 필요하다. 모델이 없으면 다운로드 없이 BLOCKED로 기록한다.

이번 고정 순서는 smoke2 worker→main2 worker→선택적 bridge2 server다. 최대8 process, 1200초와 실제 campaign 잔량 중 작은 값, 120초의 정리 여유를 지킨다. 자동 재시도는0이다. raw는 새 attempt에 쓰며 budget/progress만 atomic snapshot으로 갱신한다. 외부 GPU PID를 발견하면 새 요청을 멈추고 소유 group만 정리한다. WSL/NVML 관측으로 짧은 외부 작업이나 graphics 사용의 부재를 증명할 수는 없다.

## 入力と座標

合成英文2・韓文2、reference_source=`author_provided_diagnostic_text`。正答率の gold ではない。plain completionを結合してspecialなしでtokenizeする。prompt単独/continuation単独と結合token IDの差、BPE境界を跨ぐtoken、codepoint/UTF-8 byte offset、BOS/EOS/padding/loss maskを記録する。境界は「prompt末尾を超える最初のjoint token」で、そのtoken全体を評価する。

target token index `t` は logit index `t-1` で採点する。予測prefixは `input_ids[:t]`。token-ID space、vocabulary mapping、prefix hashが一致したときだけ比較する。今回の選択は中間layer12、head0/16、最後のprompt位置と最後のreferenceの直前。future/padding massは0、行は全key軸を保持する。意図したartifact差を同一性違反にはせず、その他の入力・観測規則の違いを拒否する。

## 観測の意味

BF16は `INSTRUMENTED_PATH`。RTNはライブラリの実pack-quantized decodeで168LinearをBF16 dense weightへ展開した `DECODED_WEIGHT_DIAGNOSTIC`。strict state loadでmissing/unexpectedとも0、native50tensorのBF16一致、tied embedding/lm_head、FP32 native RoPE buffer、代表3groupのinteger code×scaleとdecode値の一致を検査した。新規量子化・比率探索・dense artifact保存はしない。INT4圧縮kernelの内部/VRAMの証拠ではない。

選択したhidden位置だけを実モデルの最終norm後にLM headへ渡す。全[T,V]を生成後sliceする方式ではない。target log-softmaxはFP32、選択raw full logitsはfloat32 blob、sampling/top-p/temperature変換は使わない。

確率行はnative post-RoPE Q/KからFP32で再構成した `reconstructed_from_native_qk`。native SDPAをそのまま呼び、戻り値を書き換えず選択headのAVを読む。`reconstructed_AV`とnative-returned AV、output projection後の選択query vectorを別に保存する。fused kernelが確率行を物質化したという主張はしない。OFF1/OFF2/ONは最初の英文2件のみ。wrapper/interface/hookはfinallyで復元し、tensor versionと原本statを確認する。

vLLM bridgeは同じtoken IDsのecho continuation logprobを再採点する。これはdiagnosticの配備同等性認証ではない。raw logitsがないのでcross-backend full KLは計算できない。native worker exit UNKNOWN、kernel selected log、parent exit/owned cleanup/portを別記する。

## 計算・欠測

| 指標 | 定義/単位 |
|---|---|
| NLL、ΔNLL | `-log P(target|prefix)`、candidate−BF16、nat |
| raw top1 match | sampling前argmaxの一致率、観測位置のみ |
| full KL | `KL(reference || candidate)`、同一full vocabulary、nat |
| full JS | 平均分布への2KLの平均、対称、nat |
| entropy | `-Σp log p`、nat |
| attention L1/JS | 同じlayer/head/query/mask・全key行、FP32丸めの行和を記録 |
| output NMSE | `Σ(candidate−reference)² / Σreference²` |
| cosine | 内積/両ベクトルnormの積 |

energy ≤1e-12 のNMSE/cosineはundefinedとして値null。all-masked行も分布0にしない。unsupported/budget-excluded/failed/not-run/incomparableとmeasuredを分ける。full logitsを保存していない位置にKL/JSを作らない。複数tokenを独立文書として一般化せず、短いcontinuation NLLをcorpus perplexity/品質認証にしない。attention差は伝播したhiddenのend-to-end差であり、layer単独の因果効果・思考過程の正しさではない。

OFF/ON許容差1e-5、保存logitのCPU log-softmax対target許容差2e-5、zero-energy規則は実行前に固定した。bridgeには同等性許容差を設けず `BRIDGE_MEASURED_NO_EQUIVALENCE_CLAIM` を維持する。

## CPU再計算と小さい実測fixture

`bench replay`はdirectoryまたは安全検査したZIPをdataとして読み、raw manifest・blob SHA・座標・dtype・prefixを検査する。archiveのPythonを実行しない。結果は入力の外の新しいoutputへJSON/CSV/Markdownで保存する。既存outputは拒否する。

```bash
python examples/bench05a_consumer.py --run /path/to/recorded-run --output /path/to/new-analysis
python tools/export_bench05a_fixture.py --run /path/to/recorded-run --output /path/to/new-small-fixture
```

小さいfixtureは実token/attention/output blobのbyteを保持し、full vocabulary blob・bridge・smoke・memoryを省くderived view。元raw manifestのSHA、変更したJSONLの前後SHA、除外IDをDERIVATION.jsonに記す。元の実行ソースhashを新ソースhashに置き換えない。このslice単体ではfull KL/JSと完全なobserver-logit検査はNOT_RUN/INSUFFICIENT。完全版runにはそれらが残る。両方とも非公開ローカル資料である。

## 次のadapterへの境界

BENCH05-Bはtask adapterがSampleとtokenized manifestを作り、同じExecution/Observation/MetricSpecを渡す接続を追加する。今回の4入力やSmol layerをgeneric metricへ埋め込まない。新taskのgold/正規化/品質定義と内観測の診断値は分離する。新モデルのライブサポートは構造/runtimeの実観測後にのみ登録する。今回のproduction/customer評価はNOT_EVALUATED。
