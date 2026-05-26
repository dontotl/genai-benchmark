# OCI GenAI Global Benchmark Plan

## 1. 목적

OCI Generative AI 온디맨드 모델을 대상으로, 여러 source runner 위치에서
여러 target region의 모델 endpoint를 호출했을 때의 체감 지연과 안정성을 비교한다.

현재 단계의 핵심 목적은 아래와 같다.

- 권역별 사용자가 어느 target region을 호출할 때 가장 빠른지 확인
- same-region 및 cross-region 호출의 latency 차이 확인
- Chat/Reasoning 및 NL2SQL workload에서 모델별 응답 속도 비교
- 4개 source runner와 3개 target region의 routing 판단 근거 확보
- 결과를 GitHub Pages dashboard로 공유 가능한 형태로 정리

## 2. 현재 완료 상태

- 3개 source runner 병렬 suite 실행 완료:
  - `ap-osaka-runner`
  - `us-chicago-runner`
  - `eu-frankfurt-runner`
- Target regions:
  - `ap-osaka-1`
  - `us-chicago-1`
  - `eu-frankfurt-1`
- Smoke settings:
  - `repeats=1`
  - `concurrency_levels=1`
  - `streaming=false`
- Result:
  - Attempts: `54`
  - Successes: `54`
  - Failures: `0`
- Dashboard:
  - `Runner Matrix` heatmap added
  - `Runner Ranking` summary cards added
  - GitHub Pages deployment confirmed

Published dashboard:

- `https://dontotl.github.io/genai-benchmark/`

## 3. 현재 Phase 범위

이번 phase에서는 범위를 의도적으로 좁힌다.

### 포함

- Chat/Reasoning workload 확장
- NL2SQL workload 추가
- Seoul source runner 추가
- 4-source / 3-target runner matrix 실행 계획
- Dashboard에 workload 설명 및 해석 문구 보강

### 제외

- Embedding benchmark
- Rerank benchmark
- Vision/multimodal benchmark
- Guard/safety benchmark
- Oracle AI Database Vector Search를 포함한 full RAG end-to-end benchmark
- 실제 DB에 SQL을 실행하는 NL2SQL 정확도 검증

Embedding/Rerank/Full RAG는 후속 phase로 분리한다. 현재 runner는 chat
endpoint 중심으로 구현되어 있으므로, Embedding/Rerank를 같은 phase에 넣으면
API 호출 방식과 결과 schema 확장이 커진다.

## 4. Source / Target Matrix

현재 target region은 기존 3개를 유지한다.

| Target Region | 역할 |
| --- | --- |
| `ap-osaka-1` | APAC target |
| `us-chicago-1` | US target |
| `eu-frankfurt-1` | Europe target |

Source runner는 4개로 확장한다.

| Source Runner | Runner Region | 목적 |
| --- | --- | --- |
| `ap-osaka-runner` | `ap-osaka-1` | APAC same-region 및 cross-region 비교 |
| `ap-seoul-runner` | `ap-seoul-1` | 한국/동북아 사용자 위치 기준 비교 |
| `us-chicago-runner` | `us-chicago-1` | US same-region 및 cross-region 비교 |
| `eu-frankfurt-runner` | `eu-frankfurt-1` | Europe same-region 및 cross-region 비교 |

Seoul은 target region으로 추가하지 않는다. 이번 phase에서는 source runner로만
추가해, 한국 위치의 client가 기존 3개 target region을 호출할 때의 latency를 본다.

예상 suite 형태:

```text
4 source runners x 3 target regions x selected chat models x selected workloads
```

## 5. 모델 범위

### 기본 비교 대상

- `openai.gpt-oss-20b`
- `google.gemini-2.5-flash`

### 확장 후보

아래 모델은 on-demand 지원 region을 확인한 뒤 명시적으로 포함한다.

- `openai.gpt-oss-120b`
- `google.gemini-2.5-pro`
- `google.gemini-2.5-flash-lite`
- Command A 계열
- Llama 계열
- Grok 계열

실행 단위가 커지는 것을 막기 위해, 4-source suite의 첫 실행은 모델 2~3개로
제한한다.

## 6. Workload Plan

### 6.1 Chat / Reasoning

목적:

- 일반 업무형 질문에서 모델별 응답 속도와 안정성을 비교한다.

권장 workload:

| Workload ID | 설명 |
| --- | --- |
| `chat-helpdesk` | 사용자의 간단한 문의에 상담원처럼 답하게 한다. |
| `summary-ko` | 긴 설명을 짧은 한국어 요약으로 줄이게 한다. |
| `code-debug` | 작은 코드 조각을 읽고 문제점과 수정 방향을 말하게 한다. |
| `reasoning-choice` | 여러 조건을 보고 가장 좋은 선택지를 고르게 한다. |
| `agentic-plan` | 실제 도구 실행 없이 작업 순서와 확인 항목을 계획하게 한다. |

### 6.2 NL2SQL

목적:

- 사람이 평소 말로 한 질문을 SQL query로 바꾸는 속도와 응답 안정성을 본다.

1차 범위:

- 실제 DB에는 연결하지 않는다.
- 모델에게 schema와 자연어 질문을 제공한다.
- SQL 생성 결과의 latency와 형식만 확인한다.

예시 schema:

```sql
customers(customer_id, name, region, segment)
orders(order_id, customer_id, order_date, status, total_amount)
order_items(order_id, product_id, quantity, unit_price)
products(product_id, name, category)
```

예시 질문:

- 2025년 지역별 총 매출을 높은 순서로 보여줘.
- 최근 30일 동안 주문 금액이 가장 큰 고객 10명을 찾아줘.
- 카테고리별 평균 주문 금액을 계산해줘.

가벼운 검증 기준:

- 응답에 `SELECT`가 포함되어 있는가
- `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER` 같은 위험 문구가 없는가
- 제공한 schema의 table name을 사용했는가

실제 SQL 실행, 정답 SQL 비교, 결과 row 검증은 후속 phase로 둔다.

## 7. Prompt Files

현재 파일:

- `prompts/sample_prompts.jsonl`

추가 계획:

- `prompts/chat_nl2sql_workloads.jsonl`

새 prompt file은 기존 chat runner가 읽을 수 있는 JSONL 형식을 유지한다.

예상 형식:

```json
{"id":"nl2sql-sales-analytics","messages":[{"role":"system","content":"Generate a safe SELECT SQL query only."},{"role":"user","content":"Schema: ... Question: ..."}]}
```

## 8. Runner / Config Plan

### 8.1 4-source suite config

추가 계획:

- `configs/ephemeral-4source-managed.example.json`

기준:

- 기존 `configs/ephemeral-3region-managed.example.json`에 Seoul runner를 추가한다.
- `target_regions`는 기존 3개를 유지한다.
- `parallelism`은 실제 실행 시 `4`를 사용한다.
- `existing_dynamic_group_update=false`를 유지한다.

Seoul runner 필수 값:

- `region`: `ap-seoul-1`
- `source_label`: `ap-seoul-runner`
- `availability_domain`: local config에서 실제 AD 사용
- `image_id`: local config에서 Seoul Oracle Linux 8 image OCID 사용

### 8.2 IAM / Dynamic Group

병렬 multi-runner suite에서는 per-instance Dynamic Group update를 사용하지 않는다.

필수 조건:

- 기존 Dynamic Group이 runner compartment를 포괄해야 한다.
- `network.existing_dynamic_group_update=false`
- 가능하면 `network.existing_policy_id=<existing-policy-ocid>`로 reusable policy를 사용한다.

## 9. 측정 항목

공통 항목:

- generated timestamp
- source runner label
- target region
- model id
- family
- workload/case id
- concurrency
- repeats
- success/failure
- latency seconds
- p95 latency
- p99 latency

Chat/NL2SQL 항목:

- input tokens
- output tokens
- total tokens
- end-to-end output tokens/sec
- response preview
- error type
- HTTP status
- request id

NL2SQL 추가 항목은 구현 phase에서 별도 schema 확장 여부를 결정한다. 1차 문서
계획에서는 기존 `case_id`와 `response_preview`만으로도 smoke 목적은 충족한다.

## 10. Dashboard Plan

현재 dashboard는 suite summary markdown을 읽어 `Runner Matrix`를 렌더링한다.

보강 계획:

- 4x3 matrix가 자연스럽게 보이는지 확인한다.
- `Runner Ranking` 제목은 실제 의미에 맞게 `Runner Summary`로 바꾸는 것을 검토한다.
- `Runner Matrix` 바로 아래에 해석 문구를 추가한다.

해석 문구에 포함할 내용:

- 어떤 모델을 테스트했는가
- 어떤 workload를 돌렸는가
- `summary-ko`, `table-en`, `ops-checklist`, NL2SQL이 각각 무엇을 의미하는가
- 숫자는 평균 latency이고 낮을수록 빠르다는 점
- smoke benchmark라 대규모 부하나 품질 평가 결론으로 과해석하지 말아야 한다는 점

워크로드 설명은 쉬운 한국어로 작성한다.

예시:

- `summary-ko`: 모델에게 긴 내용을 한국어로 짧게 요약하게 하는 테스트
- `table-en`: 모델에게 작은 모델과 큰 모델의 장단점을 영어 표로 정리하게 하는 테스트
- `ops-checklist`: 운영자가 점검표를 만들듯 확인 항목을 한국어 목록으로 정리하게 하는 테스트
- `nl2sql`: 사람이 말로 한 질문을 SQL query로 바꾸게 하는 테스트

## 11. 후속 Phase

### Phase B. Embedding

- Oracle AI Database docs 기반 짧은 문서 chunk를 embedding API에 보낸다.
- 벡터 DB 없이 API latency와 처리량만 측정한다.
- 문서는 원문 대량 복사가 아니라 paraphrase chunk + source URL 방식으로 준비한다.

### Phase C. Rerank

- Oracle AI Database docs 후보 문서 여러 개와 질문을 rerank API에 보낸다.
- expected top document id를 둬 간단한 품질 신호를 기록한다.
- on-demand 지원 rerank 모델만 catalog에 포함한다.

### Phase D. Full RAG

- Oracle AI Database Vector Search 또는 별도 vector DB를 붙인다.
- query embedding -> vector search -> rerank -> answer generation 전체 latency를 측정한다.
- 이 phase는 DB schema, VECTOR column, index, query plan까지 필요하므로 별도 작업으로 둔다.

### Phase E. Vision / Guard / Safety

- 이미지+텍스트 이해, multimodal search, content moderation, prompt injection detection은 별도 workload와 metric이 필요하다.
- 현재 phase에는 포함하지 않는다.

## 12. 리스크 및 확인 사항

- Seoul runner의 image OCID와 availability domain을 local config에 정확히 넣어야 한다.
- 4-source 병렬 실행은 VM/network 리소스가 동시에 4세트 생성되므로 quota를 확인해야 한다.
- 모델별 on-demand 지원 region이 다르면 skipped 조합이 늘어난다.
- Grok, Llama, Command 계열은 기본 catalog 및 experimental flag 정책을 다시 확인해야 한다.
- NL2SQL은 실제 DB 실행이 없으므로 정확도 결론은 제한적이다.
- Chat workload 수와 모델 수를 동시에 늘리면 요청 수와 비용이 빠르게 증가한다.

## 13. 다음 작업 순서

1. `PLAN.md`를 현재 phase 기준으로 확정한다.
2. `prompts/chat_nl2sql_workloads.jsonl`을 추가한다.
3. 4-source managed suite example config를 추가한다.
4. Seoul local config의 AD/image OCID를 확인한다.
5. 4-source suite dry-run을 실행한다.
6. 필요하면 dashboard 4x3 layout을 보강한다.
7. 실제 4-source / 3-target Chat/NL2SQL smoke를 실행한다.
8. 결과를 GitHub Pages dashboard에 배포한다.
