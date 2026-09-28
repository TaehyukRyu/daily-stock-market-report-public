# AI 주식 투자 리포트 자동 생성 시스템

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![LangGraph](https://img.shields.io/badge/LangGraph-0.4-purple) ![LangChain](https://img.shields.io/badge/LangChain-0.3-green) ![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Opus/Sonnet/Haiku-orange) ![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-black) ![FastMCP](https://img.shields.io/badge/FastMCP-3.2-red) ![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-teal) ![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-181717?logo=github)

> **단일 LLM의 편향 문제를 "역할 분담 → 토론 → 규칙 기반 결정 → 사후 검증" 구조로 해결한다.**
> 매 거래일 새벽(KST), 무인 환경에서 한국 주식 110종목 중 **오늘 새 정보가 생긴 종목**을 골라 8명의 투표자(LLM 에이전트 7 + 규칙 에이전트 1)가 분석·토론하고, 통합 투자 리포트를 Notion에 자동 발행하는 자율 파이프라인입니다.

> **공개 버전 안내** — 이 저장소는 포트폴리오용 공개본입니다. 에이전트 프롬프트와 데이터 가공 로직은 `[REDACTED]`로, 실제 스크리닝 규칙 파일은 형식 예시(`src/config/screen_rules/example/`)로 대체했습니다. 설계 문서·백테스트 결과·운영 워크플로는 포함하지 않습니다.

---

## 📋 목차

1. [프로젝트 개요](#-프로젝트-개요)
2. [주요 특징](#-주요-특징)
3. [시스템 아키텍처](#-시스템-아키텍처)
4. [데이터 파이프라인](#-데이터-파이프라인)
5. [운영 결과](#-운영-결과)
6. [설치 및 실행](#-설치-및-실행)
7. [기술 스택](#-기술-스택)
8. [제한사항 및 향후 과제](#-제한사항-및-향후-과제)
9. [문의](#-문의)

---

## 🎯 프로젝트 개요

### 배경 및 필요성

단일 LLM에 투자 분석을 일임하면 그 모델의 **학습 편향과 환각**이 결과에 그대로 노출됩니다. 근거 추적이 어렵고, 잘못된 판단을 사후에 교정할 **폐루프(closed-loop) 학습 구조**도 없습니다. 운영해 보니 문제가 하나 더 있었습니다. **LLM은 프롬프트에 적힌 판정 규칙을 일관되게 지키지 않습니다.** 그리고 결과가 좋아 보여도 그것이 실력인지 운인지 구분할 장치가 없습니다.

본 시스템은 이를 다음과 같이 해결합니다.

| Before — Single LLM | After — 8 Voters + Debate + Rule + Verification |
|---|---|
| 한 모델의 편향이 결과 전체에 노출 | ① **역할 분담** — 거시·시장·정량·기술·이벤트·펀더멘털 7 도메인 + 규칙 기준선 1 |
| 근거 추적 어려움 | ② **토론** — 의견 경합 시 Bull vs Bear 2라운드 |
| LLM이 판정 규칙을 지키지 않음 | ③ **결정 분리** — 최종 방향은 파이썬 규칙, LLM은 근거·리스크 서술 |
| 잘못된 판단 교정 메커니즘 부재 | ④ **사후 학습** — D+5 채점 · 증거 게이트 · 판단 복기 메모리 |
| 운인지 실력인지 구분 불가 | ⑤ **통계 검증** — 무작위 대조군 · 규칙 사전 등록 · 다중검정 보정 |

### 프로젝트 목표

- **무인 자동화**: 사람의 개입 없이 매 거래일 통합 리포트를 자동 생성·발행
- **결정과 서술의 분리**: 방향은 결정론적 규칙이 정하고 LLM은 설명만 맡아 재현·감사 가능하게
- **증거 기반 자기 개선**: 표본 수와 통계 검정을 통과한 증거가 있을 때만 에이전트 가중치 변경
- **장애 복원력 + 날조 방지**: 실패는 격리하고, 데이터가 없으면 "없다"고 답하게 하는 스키마
- **저비용 운영**: 모델 역할 분담 + 실행당 LLM 예산 상한

### 프로젝트 정보

| 항목 | 내용 |
|---|---|
| 기준 시점 | 2026-09 |
| 실행 환경 | Python 3.11 / Ubuntu (CI) / Windows (개발) |
| 자동화 | GitHub Actions 워크플로 3종 (분석 · 채점 · 테스트) |
| 발행 채널 | Notion 하루 1건 통합 리포트 (실패 시 로컬 `.md` 파일 폴백) |
| 분석 유니버스 | 한국 주식 110종목 (분기 자동 갱신) → 당일 이벤트 종목만 분석 |
| 테스트 | 1,015건 (유료 API 호출 없는 904건은 push마다 CI 실행) |

---

## 🌟 주요 특징

### 1. 다중 LLM 역할 분담 (Multi-Model Architecture)

비용·성능·역할을 분리해 모델을 배치했습니다. 구조화 출력은 Pydantic `AnalysisReport` 스키마로 **강제**됩니다.

| 역할 | 모델 | 이유 |
|---|---|---|
| 전문가 7종 | `gpt-4o-mini` | 병렬 호출, 비용 우선 |
| 규칙 투표자 (`quant_rule_agent`) | 없음 (LLM 0콜) | "LLM이 규칙 하나보다 나은가"를 재는 비교 기준선 |
| Bull/Bear 토론 (`debate`) | `claude-sonnet-4-6` | 균형 성능 |
| 최종 서술 (`chief_strategist`) | `claude-opus-4-6` | 확정된 방향의 근거·리스크 서술, tool_use 강제 호출 |
| 뉴스 스크리닝 (`stage1c_news`) | `claude-haiku-4-5` | 시장 헤드라인 → 관련 종목 태깅 1회 호출 |
| 판단 복기 · 뉴스 감성 분류 | `gpt-4o-mini` | 저비용 보조 작업 |

### 2. 이벤트 기반 스크리닝 v2 (Screening Pipeline)

가격 패턴이 아니라 **오늘 새 정보가 생긴 종목**을 고릅니다. 장기 백테스트에서 시총 상위 종목군의 가격·거래량 팩터가 무작위 선정보다 낫다는 증거를 찾지 못해 이벤트 기반으로 전환했습니다.

| Stage | 모듈 | 역할 | 비용 |
|---|---|---|---|
| **1-A** 정량 | `stage1a_quant.py` | OHLCV 캐시 갱신 · 가격 필터 | 무료 |
| **1-B** 이벤트 | `stage1b_events.py` | 뉴스 발행량 급증 · 주요 DART 공시 · 감성 변화 | 무료 |
| **1-C** 뉴스 LLM | `stage1c_news.py` | 시장 헤드라인 → 관련 종목 태깅 (점수 아님) | LLM 1콜 |
| **2** 선정 | `stage2_scorer.py` | 유니버스 내 백분위 → 후보 → 필터 → 레짐별 상한 | 무료 |

- **선정 사유 전달**: 공시 제목·뉴스 급증·헤드라인 해석이 `screen_context`로 분석 단계와 리포트까지 전달됩니다.
- **억지 선정 금지**: 후보가 0개인 날은 종목을 만들어내지 않고 시장 개관만 발행합니다.
- **뉴스 관련도 필터**: 제목에 종목명·약칭이 없는 헤드라인은 저장하지 않습니다.
- **실패 격리**: 각 Stage는 try/except로 격리되어 한 단계가 실패해도 전체가 중단되지 않습니다.

### 3. LangGraph StateGraph 파이프라인

선정된 종목마다 순차 실행됩니다. 각 노드는 **개별 타임아웃**을 가지며 초과 시 빈 결과를 반환하고 다음 노드로 진행합니다. Notion 발행은 파이프라인 밖에서 `daily_runner`가 종목별 결과를 모아 **하루 1건**으로 처리합니다.

```
data_ingest → regime_detector → parallel_analysis → quality_gate
                                                         ↓
log_predictions ← report_formatter ← chief_strategist ← debate
                                     (+ signal_reconciliation)
```

| 노드 | 타임아웃 | 역할 |
|---|---|---|
| `data_ingest` | 30s | 주가·KOSPI·VIX·환율·국채·WTI 6 소스 병렬 수집 |
| `regime_detector` | 30s | Bull / Bear / Sideways / Volatile / Neutral 분류 |
| `parallel_analysis` | 360s | 투표자 8명 병렬 (전역 Semaphore 3 / KRX 2) |
| `quality_gate` | 30s | confidence≥0.6 ∧ reasoning≥3 ∧ sources≥2, 통과 부족 시 소프트 폴백 |
| `debate` | 90s | 의견이 경합할 때만 2라운드 토론 |
| `chief_strategist` | 120s | 방향 확정(파이썬 규칙) + 근거·리스크 서술(tool_use) + 과거 판단 교훈 주입 |
| `signal_reconciliation` | — | BUY 신호 정원 관리 (일 한도 · 섹터 한도) |
| `report_formatter` | 30s | 마크다운 7섹션 리포트 조립 |
| `log_predictions` | 30s | 투표자 8명 + chief 판단을 원장(`prediction_log`)에 기록 |

### 4. 최종 방향 결정 — LLM이 아니라 규칙이 정한다

초기 설계에서는 chief(opus)가 프롬프트에 적힌 판정 규칙에 따라 방향을 정했습니다. 운영 기록 40건을 같은 규칙으로 재현해 보니 규칙대로라면 **BUY 15 · SELL 20 · HOLD 5**가 나와야 했지만, LLM이 실제로 낸 판정은 **BUY 3 · SELL 0 · HOLD 37**이었습니다. 규칙을 지시문으로 주는 것만으로는 지켜지지 않는다는 것을 확인하고, 방향 결정을 결정론적 파이썬 함수(`chief_python.decide()`)로 분리했습니다.

- **방향**: 투표 결과를 파이썬 규칙으로 집계합니다. 같은 입력이면 항상 같은 방향이 나옵니다.
- **확신도**: LLM의 자기보고 값이 아니라 투표 결과로 계산합니다.
- **abstain**: `confidence=0.0`은 반대표가 아니라 **의견 없음**으로 보고 참여자에서 뺍니다.
- **서술 전담**: chief가 다른 방향을 제출해도 확정값으로 되돌리고, 그 사실을 `reasoning`에 남깁니다.

### 5. 8명의 투표자 (Multi-Perspective Analysis)

모든 에이전트는 공통 스키마 `AnalysisReport`(recommendation · confidence · reasoning · data_sources · prediction_basis · risk_factors · data_sufficient)를 출력합니다.

| 에이전트 | 분석 도메인 | 주 데이터 소스 |
|---|---|---|
| `macro_economist` | 거시경제 · 금리 · 환율 · 원자재 | FRED · BOK ECOS |
| `kr_market_specialist` | 한국 시장 · 수급 · 대주주 매매 | KRX MCP · DART |
| `us_market_specialist` | 미국 시장 · VIX · 국채 | yfinance · US 지표 |
| `quant_analyst` | PER/PBR · 모멘텀 | KRX MCP |
| `technical_analyst` | MA · RSI · MACD · 볼린저 · 일목균형표 | OHLCV · TA-Lib |
| `sentiment_analyst` | **종목 이벤트 해석** (공시 · 종목 헤드라인 · 선정 사유) | DART · NAVER 뉴스 |
| `fundamental_analyst` | 실적 · 컨센서스 · 밸류에이션 | KRX MCP · DART |
| `quant_rule_agent` | **LLM 없는 규칙 투표자** — LLM 기여도 측정 기준선 | OHLCV 캐시 |

`macro_economist`·`us_market_specialist`는 종목과 무관한 시장 판단이라 `daily_cache`로 **하루 1회만** 호출합니다.

### 6. 하이브리드 RAG (BM25 + Vector + RRF)

한국어 형태소 기반 BM25와 벡터 검색을 **앙상블(Reciprocal Rank Fusion)** 로 결합합니다.

```
query ─┬─ BM25 (rank-bm25 + Kiwi)
       └─ ChromaDB (text-embedding-3-small)
              ↓
       EnsembleRetriever (RRF) → context (TOTAL_TOKEN_BUDGET = 2,000)
```

- **에이전트별 primary/secondary 컬렉션 배분** — 도메인에 맞게 검색 대상 분리
- ※ 현재 운영 경로에는 ChromaDB 적재가 연결되어 있지 않습니다. 과거 판단의 교훈은 아래 **판단 복기 메모리**가 대신 주입합니다.

### 7. MCP 서버 직접 구현 (FastMCP 3.2.4)

데이터 수집 계층을 **3개 MCP 서버 + 32개 도구**로 직접 구축했습니다. 실패 시 예외 대신 `{"error": ...}` 반환으로 호출자를 격리합니다.

| MCP 서버 | 도구 수 | 제공 도구 (예시) |
|---|---|---|
| `krx_market` | 14 | `get_stock_price` · `get_investor_trends` · `get_analyst_reports` · `get_financials` · `get_convertible_bonds` · `get_earnings_calendar` |
| `news_economy` | 10 | `get_exchange_rate` · 뉴스 · 금리 · DXY · 금통위 · 종목 토론방 |
| `us_market` | 8 | `get_vix` · `get_treasury_yields` · `get_commodity_prices` · FedWatch |

수집이 실패한 항목은 프롬프트에 **"수집 실패"로 명시**해, LLM이 데이터 부재(값이 원래 없음)와 수집 실패를 혼동하지 않게 합니다.

### 8. 폐루프 피드백 학습 (Closed-Loop Feedback)

```
D        예측 기록 → prediction_log (투표자 8명 + chief)
D+1      chief 판단 복기 → 2~4문장 교훈 생성 → 다음 chief 프롬프트에 주입
D+5      실현 수익으로 채점 (BUY · SELL · HOLD 대칭 점수표)
         → 증거 게이트 통과 시에만 EMA 가중치 갱신
D+5/10/20  실현 수익 · KOSPI200 대비 초과수익 확정
```

- **채점 지평 D+1 → D+5**: 1~4주 포지션을 표방하면서 하루 등락으로 채점하던 불일치를 바로잡았습니다.
- **증거 게이트**: 표본 30개 **그리고** 평균 점수가 동전 던지기(0.5)와 통계적으로 다를 때(|z| ≥ 1.96)만 균등 가중치에서 벗어납니다. 매일 다시 판정하므로 한 번 통과해도 영구적이지 않습니다.
- **계층 폴백**: 레짐별 증거가 부족하면 레짐 무관(`All`) 가중치 → 그것도 없으면 균등 가중치를 씁니다.
- **판단 복기 메모리**: TradingAgents의 memory 방식을 참고해, 가중치가 증거를 모으는 동안 chief가 매일 백지에서 판단하는 공백을 메웁니다. 교훈은 **chief에게만** 주입해 투표자 간 비교가 오염되지 않게 합니다.

**동적 α 규칙** (`get_dynamic_alpha`):

| 조건 | α 값 | 의도 |
|---|---|---|
| Volatile 레짐 | 0.5 | 즉시 적응 |
| 최근 5거래일 내 레짐 전환 감지 | 0.5 | 빠른 전환 추종 |
| 동일 레짐 유지 (안정) | base_alpha (Bull 0.1, Bear 0.3, Sideways 0.2) | 노이즈 억제 |

### 9. 검증 인프라 — 백테스트 · 규칙 게이트

수익을 주장하기 전에 **"운보다 나은가"** 부터 검증합니다.

| 장치 | 모듈 | 무엇을 검증하나 |
|---|---|---|
| 백테스트 엔진 | `evaluation/backtest/` | 약 10년 과거 재현 · 거래비용 차감 · 노출 일치 벤치마크 · walk-forward |
| 규칙 게이트 | `screening/gate.py` | 사전 등록 규칙만 판정 · Bonferroni 보정 · OOS 붕괴 확인 · 무작위 대조군 백분위 |
| 무작위 대조군 | `evaluation/random_control.py` | 같은 조건에서 무작위로 고른 것보다 나은가 (실시간 누적) |
| LLM 기여도 | `evaluation/llm_contribution.py` | LLM 7명의 종합이 규칙 투표자 1명보다 나은가 |
| 확신도 보정 | `evaluation/calibration.py` | "확신도 0.8"이 실제로 80% 맞는가 |
| 섀도 A/B | `utils/shadow_log.py` | 모델 교체 후보를 운영과 병렬 실행해 선정 겹침률 비교 |

**규칙 사전 등록 흐름** — 결과를 본 뒤 파라미터를 조정해 통과시키는 것(p-hacking)을 구조적으로 막습니다.

```
규칙 JSON (7필드) → spec_hash 고정 (값 하나라도 바뀌면 새 규칙)
   → 분기별 가설 수 m 사전 선언 → Bonferroni 임계 t* 확정
   → 게이트 판정: draft → gated → shadow → active → retired
```

공개본에는 실제 규칙 대신 형식 예시 `src/config/screen_rules/example/example_rule.json`만 포함되어 있습니다.

### 10. 리스크 관리 — 변동성 기반 손절

손절폭은 LLM이 정하지 않습니다. 종목별 변동성(ATR14)으로 계산하고, 포지션 크기는 **1회 손절 시 시드 손실이 상한을 넘지 않도록** 줄입니다. 고정 손절폭이 종목 변동성과 무관하게 노이즈에 자주 걸리던 문제를 실측으로 확인하고 교체했습니다.

### 11. 다층 복원력 + 날조 방지 (Resilience by Design)

| 계층 | 패턴 | 도구 |
|---|---|---|
| ① 파이프라인 노드 | 노드별 asyncio 타임아웃, 실패 시 `{}` 반환 | `node_with_timeout` |
| ② 에이전트 호출 | 실패 시 `confidence=0.0` 폴백 보고서 | `ResilientChain` |
| ③ LLM API | Circuit Breaker + Retry (3회) + SDK 타임아웃 | `pybreaker` · `tenacity` |
| ④ 동기 라이브러리 | 스레드 분리 + SDK 레벨 타임아웃 | `pykrx` · `yfinance` |
| ⑤ 출력 스키마 | 위반 시 1회 피드백 재요청 → 근거 부족이면 `data_sufficient=false`로 abstain | `AnalysisReport` validator |
| ⑥ 실행 건강도 | "전원 폴백인데 성공처럼 보이는" 실행을 판정 → 종료 코드 1 → GitHub Issue 알림 | `daily_runner` |
| ⑦ 비용 | 실행당 LLM 예산 상한, 초과 시 종목 경계에서 중단 | `llm_budget` |

> **설계 원칙**: *"실패는 격리하고 폴백으로 흡수하되, 신뢰할 수 없는 산출물은 `confidence=0.0`으로 표시해 하류에서 제외한다. 데이터가 없으면 없다고 선언하게 만들고, 스키마가 거짓말을 압박하지 않도록 설계한다."*

---

## 🏛 시스템 아키텍처

### 5계층 구조 (Layered Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│  L5  피드백 / 검증                                                 │
│      prediction_logger · feedback_evaluator · decision_memory     │
│      evaluation: outcomes · random_control · calibration ·        │
│                  llm_contribution · screen_factors · backtest     │
├─────────────────────────────────────────────────────────────────┤
│  L4  오케스트레이션                                                │
│      daily_runner · pipeline (LangGraph StateGraph) · chief_python│
├─────────────────────────────────────────────────────────────────┤
│  L3  에이전트 / 추론                                               │
│      base_agent · 전문가 7 · quant_rule_agent · debate · chief    │
│      RAG: context_injection · hybrid_retriever · chroma_store     │
├─────────────────────────────────────────────────────────────────┤
│  L2  스크리닝 / 신호                                               │
│      screener · stage1a/1b_events/1c · stage2_scorer              │
│      rules (spec · registry) · gate · signals                     │
│      regime_detector · quality_gate · signal_reconciliation       │
├─────────────────────────────────────────────────────────────────┤
│  L1  데이터 / 인프라                                               │
│      MCP 서버 3종 · mention_db · ohlcv_cache · dart_events ·      │
│      flow_history · position_tracker · notion_publisher           │
├─────────────────────────────────────────────────────────────────┤
│  공통 횡단: resilience · security · llm_budget · mcp_client ·     │
│            graph_state · agent_output                             │
└─────────────────────────────────────────────────────────────────┘
```

### 배포 구조 — Serverless (GitHub Actions)

별도 서버 없이 GitHub Actions가 매일 코드를 실행합니다.

| 시각 (KST) | 작업 |
|---|---|
| 평일 새벽 | 스크리닝 → 종목별 분석 → Notion 통합 발행 |
| 평일 장 마감 후 | 실현 수익 확정 → D+5 채점 · 가중치 갱신 → chief 판단 복기 → 포지션 자동 청산 |
| push마다 | 유료 API를 호출하지 않는 테스트 904건 |

- **Secrets**: API 키 암호화 · 환경변수 주입 (`.env`는 저장소에 없음)
- **상태 보존**: 매 실행이 새 머신이므로 `mentions.db`를 캐시 + 백업 브랜치로 **이중 보관**
- **Artifacts**: 리포트·로그 7일 보존 · Notion 발행 실패 시 백업
- **cron 지연 대응**: GitHub 무료 티어 cron은 지연될 수 있어, 장중에 발행되면 리포트에 그 사실을 명시
- 공개 저장소에는 테스트 워크플로(`tests.yml`)만 포함되어 있습니다.

---

## 🔄 데이터 파이프라인

```
외부 소스 → 수집 (MCP + 크롤러) → SQLite 캐시 → 이벤트 스크리닝 → AI 에이전트 분석 → Notion
                                       ↑                                    ↓
                                       └──── D+1 복기 · D+5 채점 ←──── prediction_log
```

### 데이터 소스

| 소스 | 용도 |
|---|---|
| pykrx (KRX) | 종목 OHLCV · 시총 · 영업일 |
| yfinance | KOSPI(^KS11) · VIX(^VIX) · 글로벌 지수 |
| NAVER 금융 / 검색 API | 헤드라인 · PER/PBR · 종목 뉴스 · 수급 이력 |
| 한경 컨센서스 | 애널리스트 리포트 · 목표주가 |
| DART OpenAPI | 공시 목록 일괄 수집 · CB/BW · 유상증자 · 실적 캘린더 |
| FRED · BOK ECOS | 거시지표 (금리 등) |

### 저장 구조

| 저장소 | 형태 | 내용 |
|---|---|---|
| `data/mentions.db` | SQLite (16 tables) | 캐시(OHLCV · 뉴스) · 원장(예측 · 포지션 · 가중치 · chief 판단 · 공시) · 채점(실현 수익 · 대조군 · 스크리닝 후보 · 뉴스 보관) |
| `data/chroma_db/` | ChromaDB | RAG 지식 문서 임베딩 |
| `data/signals/signals_*.json` | JSON | 일별 신호 레지스트리 |
| `data/reports/` | Markdown | Notion 발행 실패 시 백업 |

---

## 📈 운영 결과

### LangSmith 7일 누적 모니터링 (2026-05 기준)

| 지표 | 값 | 해석 |
|---|---|---|
| 총 LLM 호출 | **335** | 7일 누적 |
| 오류율 | **4 %** | 초기 1일차 외부 API 장애 → Circuit Breaker 격리 후 사실상 0% |
| Latency P50 | **5.15s** | 절반의 호출이 이 시간 이내 |
| Latency P99 | **201s** | parallel_analysis + debate + chief_strategist 누적 (360s 예산 내 통제) |
| 토큰 사용량 | **3.53 M** | 7일 누적 |
| LLM 비용 | **$ 0.52** | 일평균 **$ 0.07** — 모델 역할 분담의 효과 |

### 비용 구성 (2026-09-13 실행 실측, 4종목)

| 구성요소 | 비용 | 비중 |
|---|---|---|
| chief_strategist (opus) | $0.262 | **92%** |
| 전문가 7명 (gpt-4o-mini) | $0.015 | 5% |
| 뉴스 스크린 (haiku) | $0.006 | 2% |
| 합계 | **$0.284** | 실행당 예산 상한 내 |

비용의 대부분이 서술 단계에서 나오므로, 방향 결정을 파이썬으로 분리한 뒤 서술 모델만 교체하는 저비용 모드(`CHIEF_MODE=python_sonnet`)를 준비해 두었습니다.

### 안정성 / 자동화

- **무인 자동 실행**: 평일 cron 트리거 · 사용자 개입 없음
- **소프트 폴백**: 에이전트 일부 실패 · Notion 실패 시에도 리포트 발행 유지 (로컬 백업)
- **거짓 성공 차단**: 폴백만으로 채워진 실행은 건강도 판정에서 실패 처리 후 Issue로 알림
- **테스트**: 1,015건 — 유료 API를 호출하지 않는 904건은 push마다 자동 실행 (공개본은 프롬프트 검증 28건이 `[REDACTED]`로 인해 skip)

### 출력 예시

| 발행 채널 | 형태 |
|---|---|
| Notion DB `Daily Stock Market Report` | 하루 1건 통합 리포트 페이지 |
| 통합 리포트 | 7섹션 마크다운 (포트폴리오 현황 · 오늘의 액션 플랜 · 시장 지표 · AI 분석단 의견 · 에이전트별 상세(접기) · 주요 리스크 등) |
| 종목별 선정 사유 | 공시 제목 · 뉴스 급증 · 헤드라인 해석 한 줄 |

---

## 🚀 설치 및 실행

### 환경 요구사항

- **Python 3.11** (`asyncio.timeout` 사용)
- **TA-Lib C 라이브러리** (선행 설치 필요)
  - Ubuntu: 소스 컴파일 (`.github/workflows/tests.yml` 참조)
  - macOS: `brew install ta-lib`
- **API 키**: OpenAI · Anthropic · Notion · FRED · BOK ECOS · DART · NAVER 검색 · KRX

> 공개본은 프롬프트가 `[REDACTED]`이므로 실행은 되지만 LLM 분석 결과는 원본 시스템과 다릅니다.

### 1. 클론 및 가상환경

```bash
git clone https://github.com/TaehyukRyu/daily-stock-market-report-public.git
cd daily-stock-market-report-public

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. 의존성 설치

```bash
# 로컬
pip install --no-deps -r requirements-freeze.txt

# CI
pip install --no-deps -r requirements-ci-frozen.txt
```

> **참고**: `langchain==0.3.x` + `langchain-core==1.3.3` 조합은 pip 의존성 해석으로는 설치할 수 없어 `--no-deps` 옵션을 사용합니다.

### 3. 환경변수 설정

`.env` 파일을 생성하거나 GitHub Secrets에 등록합니다.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
NOTION_API_KEY=...
NOTION_DATABASE_ID=...
FRED_API_KEY=...
BOK_ECOS_API_KEY=...
DART_API_KEY=...
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
KRX_ID=...
KRX_PW=...
LLM_BUDGET_USD=5.0        # 선택 — 실행 1회 LLM 예산 상한
```

### 4. 실행 방법

```bash
# (A) 전체 일일 실행 (스크리닝 → 종목별 분석 → 통합 발행)
python -m src.daily_runner

# (B) 단일 종목 디버그 (스크리닝 건너뜀, 기본 005930)
python -m src.graph.pipeline

# (C) 스크리닝만
python -m src.screening.screener

# (D) 분기 유니버스 재생성
python -m src.universe.universe_builder

# (E) 채점 · 복기 (장 마감 후)
python -m src.evaluation.outcomes          # D+5/10/20 실현 수익 확정
python -m src.data.feedback_evaluator      # D+5 채점 · 가중치 갱신
python -m src.data.decision_memory         # chief 판단 복기

# (F) 규칙 게이트 판정 (LLM 0콜)
python -m src.screening.gate <rule_id>

# (G) 포지션 조회
python -m src.data.position_tracker list
```

### 5. 테스트

```bash
pytest tests/ -m "not costly" -q    # 유료 API 호출 없음 — CI와 동일
pytest tests/ -q                    # 전체 (111건은 실제 유료 API 호출)
```

---

## 🛠 기술 스택

### Core

| 분류 | 기술 | 용도 |
|---|---|---|
| 언어 / 런타임 | **Python 3.11** | `asyncio.timeout` 사용 |
| 워크플로우 | **LangGraph 0.4.1** (`StateGraph`) | 선언적 파이프라인 |
| LLM 프레임워크 | **LangChain 0.3.x** · **Anthropic SDK** · **OpenAI SDK** | 구조화 출력 강제 (`with_structured_output`) |
| 검증 / 스키마 | **Pydantic 2.13** | `GraphState` · `AnalysisReport` (조건부 검증) |

### LLM

| 모델 | 사용처 |
|---|---|
| `gpt-4o-mini` | 전문가 7명 · 뉴스 감성 분류 · 판단 복기 |
| `claude-opus-4-6` | chief_strategist (최종 서술) |
| `claude-sonnet-4-6` | debate (Bull/Bear 토론) |
| `claude-haiku-4-5` | stage1c_news (뉴스 스크리닝) |
| `claude-sonnet-5` | 섀도 A/B · 저비용 chief 모드 (실험) |

### Data / RAG

| 기술 | 용도 |
|---|---|
| **FastMCP 3.2.4** | 데이터 수집 MCP 서버 (3개 · 32 tools) |
| **ChromaDB 1.5.9** | 벡터 검색 |
| **rank-bm25** | 키워드 검색 |
| **kiwipiepy 0.23** | 한국어 형태소 분석 |
| **tiktoken** | 토큰 카운팅 (예산 2,000) |
| **pykrx · yfinance** | 시장 데이터 |
| **TA-Lib · ta** | 기술적 지표 |

### Infra / Ops

| 기술 | 용도 |
|---|---|
| **GitHub Actions** | 서버리스 무인 실행 (분석 · 채점 · 테스트) |
| **SQLite** | `mentions.db` (16 tables) |
| **pybreaker** | Circuit Breaker |
| **tenacity** | Exponential Retry |
| **notion-client 2.3** | 리포트 발행 |
| **LangSmith** | LLM 호출 모니터링 / 비용 추적 |
| **pytest** | 1,015건 (유료 API 여부로 마커 자동 분류) |

---

## 📌 제한사항 및 향후 과제

### 현재 제한사항

#### 1. 실제 매매 미지원 (설계상 명시)
- 시스템은 **분석·추적만** 담당하며, 실제 증권사 주문 API 연동은 없음
- 진입 / 청산은 사용자가 직접 수행, 시스템은 `positions` 테이블로 기록만 유지

#### 2. 성과 검증 진행 중

측정 장치는 모두 갖췄지만 표본이 부족합니다. **"이 시스템이 수익을 내는가"에는 아직 답하지 않습니다.**

| 항목 | 상태 |
|---|---|
| 수익성 | D+5/10/20 실현 수익 표본 축적 중 — 결론까지 약 175거래일 필요 |
| 에이전트 가중치 | 증거 게이트 미통과 → 균등 가중치 유지 |
| 확신도 보정 | 표본 부족 → 포지션 비중 고정값 사용 |
| LLM 포함 백테스트 | 과거 뉴스 아카이브 축적 시작 단계 — 현재는 규칙층만 백테스트 가능 |

#### 3. 운영상 제약

- **cron 지연**: GitHub 무료 티어 cron은 SLA가 없어 장중 발행이 가능 (감지해 리포트에 표시)
- **운영 RAG 비활성**: 운영 경로에 ChromaDB 적재가 연결되어 있지 않음
- **포트폴리오 구성 없음**: 종목별 비중만 표시, 합계·현금 배분 로직 없음
- **종목 순차 처리**: 종목 수 증가 시 실행 시간 선형 증가
- **NAVER API 한도**: 일 25,000건 제한

### 향후 개선 방향

1. **규칙 운영 투입** — 게이트를 통과한 규칙만 shadow 관찰을 거쳐 운영 스크리닝에 투입
2. **에이전트 기여 감사** — 표본이 쌓이는 시점에 투표자별 D+5 성과로 구성 재검토
3. **포트폴리오 구성** — 종목 합계 비중·현금 배분 로직 추가
4. **비용 절감** — 서술 모델 교체(`CHIEF_MODE=python_sonnet`) 전환 검토
5. **병렬화** — 종목별 분석 병렬 실행

---

## 📮 문의

| 항목 | 내용 |
|---|---|
| 📧 Email | `<xogur1578@gmail.com>` |

---
