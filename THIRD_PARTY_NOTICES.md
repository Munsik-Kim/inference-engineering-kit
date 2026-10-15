# 출처·라이선스·기여 경계

프로젝트 코드는 원본 Munsik-Kim/inference-lab의 적용 가능한 Apache-2.0 정책을 이어받습니다. 원본 LICENSE를 그대로 포함합니다. 코드별 원본 경로/bytes SHA와 변경 범위는 docs/source-map.json에 있습니다. 새롭게 다른 라이선스를 부여하지 않습니다.

`inference_kit._vendor.diova_compare.core`는 inference-lab/packages/diova-compare **0.1.1**의 stdlib core에서 가져왔습니다. original core SHA `a0cbf427dcde7449cf0da400f2880cd1b77e8bb44c60ae8df52aa5a9816d07e7`, Apache-2.0. vendor LICENSE를 함께 배포하고 수식을 새 구현으로 주장하지 않습니다. lab SSE parser/summary와 suite likelihood/selector/metrics/contrasts도 기존 구현의 import/경로 경계를 수정했습니다. source-map은 실제 copy와 작은 수정·미이식을 구분합니다.

GPTQ, Marlin, vLLM, Transformers, LLM Compressor, compressed-tensors, lm-evaluation-harness의 알고리즘·kernel·architecture·benchmark 정의는 각각 upstream 구현/정책에 속합니다. 이 toolkit은 오케스트레이션·계약·재계산·출처 추적·진단을 통합합니다. upstream 바이너리·모델 가중치는 포함하지 않습니다. Codex는 구현·추출·실행·테스트·문서 작성을 보조했으며 사용자 연구 설계/통합과 upstream 기여를 대신한 독자 알고리즘 저작을 주장하지 않습니다.

공개 데이터는 코드 라이선스와 별도입니다. Git recorded fixture와 private evidence에 선택된 ARC 문항/답/모델 응답이 포함되며 익명 자료가 아닙니다. token ID도 원문 복원이 가능합니다.

- **ARC-Challenge**: Allen Institute for AI의 AI2 ARC, `allenai/ai2_arc` revision `210d026faf9955653af8916fad021475a3f00453`, dataset metadata CC BY-SA 4.0. [원문](https://huggingface.co/datasets/allenai/ai2_arc), [AI2](https://allenai.org/data/arc), [조건](https://creativecommons.org/licenses/by-sa/4.0/). 문항 선택·chat template/tokenization·loglikelihood 생성과 raw 응답은 원본 데이터의 변형/측정으로 표시하고 source ID와 원본 조건을 유지합니다. synthetic fixture에는 ARC 품질 주장을 붙이지 않습니다.
- **WikiText**: Stephen Merity 등의 corpus 및 Wikipedia 출처, `Salesforce/wikitext`/wikitext-2-raw-v1 revision `b08601e04326c79dfdd32d625aee71d232d685c3`, 기존 dataset metadata CC BY-SA3.0/GFDL. [원문](https://huggingface.co/datasets/Salesforce/wikitext), [WikiText 설명](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/), [CC](https://creativecommons.org/licenses/by-sa/3.0/), [GFDL](https://www.gnu.org/licenses/fdl-1.3.html). raw 자료의 원 row/article attribution 단서는 acquisition/data manifest에 보존합니다. 자료 전체를 배포하지 않으며 개인별 문서 저자 추적을 새로 검증하지 않았습니다.
- 모델: [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) revision `31b70e2e869a7173562077fd711b654946d38674`, [Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) revision `cdbee75f17c01a7cc42f958dc650907174af0554`. 과거 acquisition의 Apache2 metadata/조건을 보존하며 이번에 모델을 내려받거나 가중치를 재배포하지 않았습니다.
- 평가 정의: [lm-evaluation-harness pinned source](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520). raw prompt/choice normalization과 subset ID는 frozen protocol을 따릅니다.

관련 공개 조건 확인은 기존 acquisition/asset DATA_LICENSES와 링크의 범위입니다. 비공개 저장소도 upstream 고지를 면제하지 않습니다. 새 artifact/고객 데이터·별도 자료 재배포 권한은 이 문서가 부여하지 않습니다. 외부 재배포가 필요하면 각 데이터의 share-alike/attribution 조건을 따로 검토해야 합니다.
