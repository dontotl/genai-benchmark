# OCI GenAI Global Benchmark Plan

## 1. 목적

OCI Generative AI의 `gpt-oss` 계열 모델과 `gemini` 계열 모델을 대상으로,
APAC, 미국, 유럽 권역 간 LLM 성능을 비교 측정한다.

이 benchmark의 핵심 목적은 다음과 같다.

- 권역별 사용자 체감 지연 차이 확인
- 리전 간 cross-region 호출 시 성능 저하 정도 확인
- 토큰 생성 처리량 비교
- 동시성 증가 시 안정성 저하 및 throttling 발생 여부 확인
- 측정 결과를 바탕으로 글로벌 서비스 아키텍처 최적화 방안 도출

## 2. 핵심 질문

- 사용자가 APAC, 미국, 유럽에 있을 때 각 권역에서 가장 빠른 응답 경로는 무엇인가
- 애플리케이션과 모델 endpoint가 같은 권역에 있을 때와 다른 권역에 있을 때 latency 차이는 얼마나 나는가
- 동일 프롬프트 기준으로 `gpt-oss`와 `gemini`의 처리량 차이는 어떻게 나타나는가
- 동시 요청이 늘어날수록 어떤 모델 또는 권역에서 실패율과 throttling이 먼저 증가하는가
- 글로벌 서비스 배치 시 active-active, regional affinity, fallback routing 중 어떤 구조가 가장 합리적인가

## 3. 범위

### 포함 범위

- 비교 대상 모델군
  - OCI GenAI `gpt-oss` 계열
  - OCI GenAI `gemini` 계열
- 비교 대상 권역
  - APAC
  - 미국
  - 유럽
- 비교 관점
  - same-region 호출
  - cross-region 호출
  - 단일 요청 성능
  - 동시성 증가 성능

### 제외 범위

- 응답 품질의 정성 평가
- 장문 fine-tuning 또는 batch inference 평가
- 비용 최적화 세부 산정
- 벡터 검색/RAG 포함 복합 아키텍처 성능

## 4. 측정 목표

### 4.1 사용자 체감 지연

주요 지표:

- end-to-end latency
- average latency
- p95 latency
- p99 latency

의미:

- 각 권역 사용자가 실제로 느끼는 응답 속도 차이를 판단
- 서비스 라우팅 기준 수립

### 4.2 cross-region 성능 저하

주요 지표:

- same-region 대비 latency 증가율
- same-region 대비 throughput 감소율
- 응답 실패율 차이

의미:

- 앱 서버와 모델 endpoint를 분리 배치할 때 허용 가능한 저하 범위 판단

### 4.3 토큰 생성 처리량

주요 지표:

- output tokens per second
- total tokens per request
- 모델별 평균 생성 속도

의미:

- 사용자 응답 속도와 backend capacity planning의 핵심 지표

### 4.4 동시성 및 안정성

주요 지표:

- concurrency level별 성공률
- timeout 비율
- HTTP 오류 비율
- throttling 발생 횟수
- 재시도 필요 비율

의미:

- 운영 트래픽 증가 시 병목 구간과 한계치 파악

## 5. 테스트 매트릭스

테스트는 아래 3개 축으로 설계한다.

### 축 A. 사용자 또는 앱 위치

- APAC
- 미국
- 유럽

### 축 B. 모델 위치

- APAC 내 대상 리전
- 미국 내 대상 리전
- 유럽 내 대상 리전

주의:

- 실제 테스트 대상 리전은 두 모델군이 모두 사용 가능한 리전 조합으로 확정한다.
- 리전 가용성은 benchmark 착수 직전에 별도 확인한다.

### 축 C. 부하 조건

- 단일 요청
- 저동시성
- 중동시성
- 고동시성

예시 매트릭스:

- APAC app -> APAC model
- APAC app -> US model
- APAC app -> EU model
- US app -> APAC model
- US app -> US model
- US app -> EU model
- EU app -> APAC model
- EU app -> US model
- EU app -> EU model

## 6. 테스트 시나리오

### 시나리오 1. Baseline

목적:

- 모델별, 권역별 기본 응답 성능 측정

방법:

- 동시성 1
- 고정 프롬프트 세트 사용
- 각 케이스를 반복 실행하여 평균과 p95 산출

### 시나리오 2. Cross-Region Comparison

목적:

- 권역 간 네트워크 거리와 라우팅 영향 측정

방법:

- 동일 앱 위치에서 여러 권역의 모델 endpoint 호출
- same-region 결과와 비교해 저하율 계산

### 시나리오 3. Throughput Comparison

목적:

- 모델별 토큰 생성 속도 비교

방법:

- 짧은 응답, 중간 응답, 긴 응답 프롬프트를 분리
- output token 수와 소요 시간으로 tokens/sec 계산

### 시나리오 4. Concurrency Ramp

목적:

- 동시성 증가 시 안정성 및 throttling 여부 확인

방법:

- 예: 1, 5, 10, 20, 50 동시성 단계별 증가
- 각 단계에서 성공률, 오류율, latency 분포 측정

### 시나리오 5. Sustained Load

목적:

- 짧은 burst가 아니라 지속 부하에서의 안정성 확인

방법:

- 정해진 concurrency를 일정 시간 유지
- 시간 경과에 따른 오류 증가, 지연 악화 여부 측정

## 7. 프롬프트 설계 원칙

- 모든 모델에 동일한 의미의 입력 사용
- 권역 비교를 위해 프롬프트 세트는 고정
- 짧은 응답/중간 응답/긴 응답 케이스를 분리
- 한국어와 영어 프롬프트를 함께 포함할지 여부는 별도 결정
- 품질 평가 목적이 아니므로 채점형 프롬프트보다 길이와 구조가 일정한 프롬프트를 우선 사용

권장 프롬프트 그룹:

- 짧은 요약형
- 표 생성형
- 체크리스트형
- 장문 설명형

## 8. 측정 항목 정의

필수 항목:

- request timestamp
- app region
- model region
- model name
- prompt id
- concurrency level
- latency seconds
- input tokens
- output tokens
- total tokens
- tokens per second
- status code 또는 예외 유형
- throttling 여부

집계 항목:

- 평균 latency
- p95 latency
- p99 latency
- 평균 output tokens/sec
- 성공률
- 오류율
- throttling rate

## 9. 구현 방향

### 9.1 도구 형태

초기 구현은 Python 기반 CLI benchmark 도구로 진행한다.

이유:

- 반복 실행과 결과 저장이 쉬움
- JSON/Markdown 리포트 자동 생성 가능
- concurrency 테스트 확장이 용이함

### 9.2 실행 구성

예상 구성:

- benchmark runner
- prompt dataset
- result collector
- summary report generator
- concurrency load executor

### 9.3 출력 산출물

- raw JSON 결과
- Markdown 요약 리포트
- 모델/권역별 비교표
- 아키텍처 권고안 초안

## 10. 단계별 추진 계획

### Phase 1. 설계 확정

- 대상 모델명 확정
- 대상 리전 확정
- 호출 인증 방식 확정
- 프롬프트 세트 확정
- 성공 기준과 실패 기준 정의

### Phase 2. Benchmark CLI 구현

- 단일 요청 benchmark 구현
- 결과 JSON 저장 구현
- Markdown summary 생성 구현
- same-region / cross-region 파라미터화

### Phase 3. Throughput 측정 추가

- output tokens/sec 계산
- 응답 길이별 프롬프트 그룹 분리
- 모델별 처리량 비교 리포트 추가

### Phase 4. Concurrency 테스트 추가

- 동시성 옵션 추가
- async 또는 worker 기반 부하 실행기 추가
- timeout, throttling, retry 지표 수집

### Phase 5. 분석 및 아키텍처 제안

- 권역별 결과 비교
- cross-region penalty 분석
- global routing 전략 제안
- DR/failover 고려안 정리

## 11. 성공 기준

- APAC, 미국, 유럽 권역 간 same-region / cross-region 비교 결과 확보
- `gpt-oss`와 `gemini` 모두에 대해 동일한 benchmark 포맷 적용
- latency, throughput, 안정성 지표가 한 리포트 체계로 정리됨
- 글로벌 서비스 배치에 대한 명확한 권고안 도출

## 12. 리스크 및 확인 필요 사항

- 모델 가용 리전이 두 모델군 간 완전히 일치하지 않을 수 있음
- 권역별 quota와 rate limit이 다를 수 있음
- 동일 시간대가 아니면 서비스 부하 차이로 결과 편차가 생길 수 있음
- 네트워크 품질과 VM 스펙이 결과에 영향을 줄 수 있음
- throttling이 애플리케이션 문제인지 서비스 제한인지 구분 필요

## 13. 의사결정 포인트

이 계획서 기준으로 다음 항목을 우선 확정해야 한다.

- APAC, 미국, 유럽에서 실제 사용할 리전 목록
- `gpt-oss`와 `gemini`에서 비교할 구체 모델명
- benchmark를 실행할 app node 위치
- 동시성 단계 정의
- 최종 보고서의 의사결정 기준

## 14. 다음 작업 제안

계획서 다음 단계는 아래 순서가 적절하다.

1. 대상 리전과 모델 조합 확정
2. benchmark 측정 항목 스키마 확정
3. 프롬프트 세트 작성
4. benchmark CLI 최소 기능 구현
5. concurrency 테스트 기능 확장
6. 결과 리포트 템플릿 작성
