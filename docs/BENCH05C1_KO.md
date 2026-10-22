# BENCH05-C1: 보존 자료의 공통 token 분석

개발판 0.1.1.dev8은 모델을 실행하지 않고 세 task의 실제 저장 기록을 공통 token view로 바꾼다. `analysis/token_sources.py`가 B2/B3/B4의 파일·선택지·policy 구조를 해석하고, `metrics/token_diagnostics.py`가 task 이름을 알지 않고 좌표 검사·수치·집계를 수행한다. `analysis/generation_sources.py`는 원시 SSE와 기존 scorer를 재사용하고 `metrics/generation_compare.py`는 생성 출력의 분기만 비교한다. 기존 collector·frozen replay·scorer·metric 수식은 유지한다.

```bash
inference-kit bench tokens export --sources INPUT_SOURCES.json --output NEW_VIEW
inference-kit bench tokens compare --view NEW_VIEW --baseline bf16 --candidate rtn --output NEW_ANALYSIS
```

Python API는 동일한 `tokens_export(sources, output)`, `tokens_compare(view, baseline, candidate, output)`다. 상대 입력 경로는 input manifest의 디렉터리 기준이다. 분석할 arm 이름은 인자로 지정하며 수식에 BF16/RTN을 고정하지 않는다. 출력이 있으면 덮어쓰지 않는다. view는 manifest와 numeric blob을 포함하므로 원본 checkout 없이 compare할 수 있다. 명시적 source ZIP은 기존 bounded reader로 읽고 archive의 Python은 실행하지 않는다. CPU 기능은 Torch/vLLM/Transformers/NumPy/aiohttp/CT·tokenizer·GPU·네트워크를 필요로 하지 않는다.

Token record는 원래 observation/request ID와 raw manifest·plan·execution provenance, task/version/data/sample/policy/choice, reference role, 정확한 input sequence와 prefix hash, t/logit t−1, tokenizer/vocabulary mapping, 확률 stage, dtype와 선택 위치를 연결한다. 파생 전역 ID는 archive ancestry를 포함한다. 서로 다른 acquisition source hash는 출처이며 자동으로 동일해야 할 비교 축이 아니다. artifact/recipe는 treatment로 허용하되, task/data/policy/prefix/token space/runtime/probability 의미가 다르면 INCOMPARABLE이다. 없는 관측은 UNKNOWN이며 0으로 채우지 않는다.

직접 비교는 동일 prefix의 teacher-forced 관측만 허용한다. Instrumented BF16과 decoded RTN은 같은 진단 runtime에서 비교할 수 있지만 packed kernel의 양자화 오차 인증이 아니다. Native ARC logprob와 diagnostic은 별도 panel이다. B2 P00/P11의 prefix·위치·문맥을 직접 빼지 않는다. ARC는 외부 answerKey의 gold와 non-gold, GOLD_ONLY/ALL_CHOICES, REGRESSION/SMOKE를 구분한다. REGRESSION은 이미 사용한 subset의 재사용이며 새 TEST가 아니다.

NLL·ΔNLL은 natural log(nat)이다. Full logits가 실제 저장된 위치에서만 정확한 KL(reference||candidate), JS, entropy, top-k overlap, target rank·margin을 계산한다. Top-k는 intersection/effective_k로 Jaccard가 아니며 k>vocab이면 effective_k=min(k,vocab)다. 동률은 작은 token ID 우선, rank는 1-based다. Target margin은 전체 어휘의 target logit−best other logit이며 ARC choice margin과 다르다. Target-only나 top-k transport는 full distribution이 아니다. 원래 acquisition tolerance가 명시되지 않은 경우 target scalar↔full logits 일관성 인증은 UNKNOWN으로 남긴다.

집계는 token-weighted와 source-document macro를 함께 표시하며 task/policy/acquisition/reference/field를 유지한다. Field는 보존된 value literal과 겹치는 token 태그다. 한 boundary token이 여러 field에 걸치면 중복 태그일 수 있어 field 합계를 전체 분모로 더하지 않는다. 출력 reference literal과 문맥의 evidence CONTENT span은 다르다. Offset이 없으면 새 tokenizer로 보충하지 않는다. 선택 위치 평균은 전체 JSON·문장 likelihood나 corpus perplexity가 아니다.

자유 생성은 실제 서버 token ID가 저장된 경우만 LCP·첫 token 분기를 계산한다. 위치는 생성 출력 내 0-based index이며 teacher-forced 절대 t와 다르다. 자연 종료·length cap·EOF/미완료와 token ID 미수집(TEXT_ONLY)을 구분한다. 길이가 먼저 끝나면 가짜 EOS를 삽입하지 않는다. 분기 뒤 두 출력 token을 같은 prefix의 확률 비교로 사용하지 않는다. 문자열·token 일치는 task 정답률이 아니며 정답 gain/loss는 원래 scorer와 join한다. ARC audit1 token은 자유 답변이 아니다.

새 compatible 모델 결과를 읽으려면 원본 request/observation을 공통 record와 sequence로 해석하는 명시적 source adapter가 필요하다. 같은 token mapping·확률 stage·runtime 정책을 검증해야 하며 임의 모델의 live 실행이나 cross-tokenizer 정렬을 지원한다고 선언하지 않는다. 모델 로딩·수집 일반화·C2/D/E·새 task는 이번 범위 밖이다. HOLD_CLIPPING_VARIANT, 기존 품질 INCONCLUSIVE, B1/B2/B4 strict0, worker native exit UNKNOWN을 유지한다. 새 연구 우위·고객 수요·배포 추천을 주장하지 않는다.
