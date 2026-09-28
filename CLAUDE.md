# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 읽는 지침이다.
**문서와 코드가 다르면 코드가 진실이다.** 이 문서를 고칠 때는 반드시 코드를 직접 확인하고 고친다.

기준 시점: 2026-09 (비공개 운영 저장소 2026-09-25 커밋과 동기화된 공개본)

---

## ⚠️ 공개본 주의사항 — 먼저 읽을 것

이 저장소는 포트폴리오용 **공개본**이다. 운영 저장소와 다음이 다르다.

| 항목 | 공개본 상태 | 작업할 때 |
|---|---|---|
| 에이전트 프롬프트 16곳 | `"[REDACTED] Proprietary prompt engineering"` | 원문을 추측해 채우지 않는다 |
| 에이전트 `_format_prompt` 7개 | 본문이 `# [REDACTED]` + `return ""` | 같음 |
| 스크리닝 규칙 파일 | `src/config/screen_rules/example/example_rule.json` 1개 (형식 예시) | 실제 규칙은 운영 저장소에만 있다 |
| 프롬프트 내용 검증 테스트 28건 | `tests/conftest.py`의 `REDACTED_PROMPT_TESTS`로 skip | **고치려 하지 말 것** — 운영 저장소에서는 통과한다 |
| 운영 워크플로 (`ci.yml` · `feedback.yml`) | 없음 | 공개본에는 `tests.yml`(무료 테스트)만 있다 |
| 설계·감사 문서, 실험 결과 | 없음 (`docs/` `experiments/`). `doc/`에는 README 그림 생성 스크립트만 있다 | 그림 수정: `python doc/make_readme_svgs.py assets` |

공개본에서도 코드는 import·실행되지만, 프롬프트가 비어 있으므로 LLM 분석 결과는 운영 시스템과 다르다.

---

## 이 시스템이 하는 일

매일 새벽 GitHub Actions가 `src/daily_runner.py`를 돌린다.
유니버스 110종목을 스크리닝해 오늘 새 정보가 생긴 종목을 고르고, 종목마다 8명의 에이전트가 투표하고,
의견이 팽팽하면 Bull/Bear 토론을 붙이고, 파이썬 규칙이 최종 방향을 정한 뒤 chief_strategist가 근거와 리스크를 서술한다.
결과를 Notion에 하루 1건으로 발행하고, 장 마감 후 별도 워크플로가 예측을 채점해 에이전트 가중치를 갱신한다.

- **대상 투자자**: 중단기 매매자 (1주~1개월 포지션)
- **사람이 하는 일**: 리포트를 읽고 직접 매매. 시스템은 증권사 계좌에 접속하지 않는다.

핵심 성질 3가지.

1. **실패해도 멈추지 않는다.** 모든 노드에 타임아웃 래퍼(`node_with_timeout`)가 있고, 실패한 에이전트는 `confidence=0.0` 폴백 리포트가 된다.
2. **그래서 건강도 판정이 따로 있다.** `daily_runner._assess_run_health()`가 "전원 폴백인데 성공으로 보이는" 상태를 잡아 종료 코드 1을 낸다.
3. **성과 측정이 아직 진행 중이다.** 측정 장치는 다 깔렸지만 표본이 부족하다. "이 시스템이 수익을 내는가"는 아직 답이 없다.

---

## 실행 방법

### 운영 (PowerShell)

```powershell
# 전체 실행 — 스크리닝 → 종목별 분석 → Notion 통합 발행
.\.venv\Scripts\python -m src.daily_runner

# 단일 종목 디버그 (스크리닝 건너뜀, 기본 005930)
.\.venv\Scripts\python -m src.graph.pipeline

# 스크리닝만
.\.venv\Scripts\python -m src.screening.screener

# 유니버스 재구축
.\.venv\Scripts\python -m src.universe.universe_builder
```

### 평가·피드백

```powershell
.\.venv\Scripts\python -m src.evaluation.outcomes         # D+5/10/20 실현 수익 확정 (채점보다 먼저)
.\.venv\Scripts\python -m src.data.feedback_evaluator     # D+5 채점 + EMA 갱신
.\.venv\Scripts\python -m src.data.decision_memory        # chief 판단 D+1 복기 (OpenAI 1콜/건)
.\.venv\Scripts\python -m src.evaluation.performance      # 누적 성과 집계
.\.venv\Scripts\python -m src.evaluation.calibration      # 확신도 보정 표
.\.venv\Scripts\python -m src.evaluation.llm_contribution # LLM vs 규칙 투표자 비교
.\.venv\Scripts\python -m src.evaluation.screen_factors   # 스크리닝 팩터별 초과수익
.\.venv\Scripts\python -m src.data.prediction_logger      # 가중치 현황
.\.venv\Scripts\python -m src.data.position_tracker list  # 포지션 조회
.\.venv\Scripts\python -m src.utils.shadow_log            # 섀도 A/B 현황
.\.venv\Scripts\python -m src.screening.gate <rule_id>    # 규칙 게이트 판정 (LLM 0콜)
```

### 테스트

```powershell
.\.venv\Scripts\python -m pytest tests\ -m "not costly" -q   # 기본 — 904건(공개본은 28건 skip), 과금 0
.\.venv\Scripts\python -m pytest tests\ -q                   # 전체 1,015건 (costly 111건은 유료 API 호출)
.\.venv\Scripts\python -m pytest tests\test_pipeline.py -v    # 단일 파일
```

**⚠️ `pytest tests/`를 조건 없이 돌리지 말 것.** 이 스위트는 목이 아니라 실제 유료 API를 호출한다.
전체 실행은 1시간 이상 걸리고 OpenAI 호출이 과금된다. 분류는 `tests/conftest.py`의 `COSTLY_MODULES`.

### 린트·타입

```powershell
.\.venv\Scripts\python -m ruff check src\
.\.venv\Scripts\python -m mypy src\
```

### 진단

```powershell
# Circuit Breaker 상태
.\.venv\Scripts\python -c "from src.utils.resilience import get_breaker_status; import json; print(json.dumps(get_breaker_status(), indent=2, ensure_ascii=False))"

# DB 테이블 행 수
.\.venv\Scripts\python -c "import sqlite3; c=sqlite3.connect('data/mentions.db'); [print(f'{t[0]:28s} {c.execute(f\"SELECT COUNT(*) FROM {t[0]}\").fetchone()[0]:7d}행') for t in c.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\")]"
```

---

## 아키텍처

### 하루의 흐름

| 시각 (KST) | 무엇 | 워크플로 |
|---|---|---|
| 평일 새벽 | 스크리닝 → 종목별 분석 → Notion 발행 | 운영 저장소 (공개본에 없음) |
| 평일 장 마감 후 | D+5/10/20 확정 → D+5 채점·EMA 갱신 → chief 복기 → 포지션 자동 청산 | 운영 저장소 (공개본에 없음) |
| push마다 | `-m "not costly"` 테스트 | `.github/workflows/tests.yml` |

**GitHub 무료 티어 cron은 SLA가 없다.** 수 시간 지연될 수 있어 장중(09:00~15:30)에 발행되기도 하며,
그때는 `_market_session_warning()`이 리포트에 그 사실을 명시한다.

`src/scheduler.py`(APScheduler)는 코드로만 존재하고 **가동하지 않는다**. 실제 스케줄러는 GitHub Actions다.

### 1단계 — 스크리닝 (`src/screening/`) — v2

유니버스 110종목 중 **"오늘 새 정보가 생긴 종목"**을 고른다. 가격 패턴으로 고르지 않는다 —
장기 백테스트에서 시총 상위 종목군의 가격·거래량 팩터가 무작위 선정보다 낫다는 증거가 없었다.

| Stage | 파일 | 무엇 | 비용 |
|---|---|---|---|
| 1-A | `stage1a_quant.py` | ohlcv_cache 갱신. 스크리닝은 단기 수익률(필터)만 쓴다 | 무료 |
| 1-B | `stage1b_events.py` | `news_burst`(발행일 기준 뉴스 급증 배수) · `dart_event`(주요 공시 0/1) · `sent_delta`(기록) | 무료 |
| 1-C | `stage1c_news.py` | 시장 헤드라인 → 종목 지목. **점수 아님, 태그만** | haiku 1콜 |
| 2 | `stage2_scorer.py` | 유니버스 내 백분위 → 후보 → 필터 → 레짐 상한 | 무료 |

- **후보** = 뉴스 급증 백분위가 `NEWS_BURST_PCT_MIN` 이상이거나 주요 공시가 있는 종목. 후보가 0개면 시장 개관만 발행한다.
- **필터·상한** = `MAX_ABS_RET5`(단기 급등락 제외), `CAP_BY_REGIME`(레짐별 상한). `daily_runner`가 `MAX_CONFIRMED_TICKERS`와 min을 취한다.
- **워밍업**: `news_burst`는 v2 적용일(`eval_method_log.screen_v2`)로부터 기준선 기간이 찬 뒤에 켜진다. 그 전엔 공시만으로 후보.
- 뉴스는 `mention_tracker`가 **발행일(pubDate, KST)**로 저장한다. 공시는 `src/data/dart_events.py`가 날짜 창으로 한 번에 받는다(종목별 호출 아님).
- 매일 유니버스 110행이 `screen_candidates`에 팩터값과 함께 남고, `src/evaluation/screen_factors.py`가 팩터별 유니버스 대비 초과수익과 t를 낸다. 가중치 자동 갱신은 없다.
- 각 stage는 독립 try/except로 격리. 1-B가 실패해도 후보 0개로 Stage 2를 진행한다.

#### 스크리닝 규칙 파이프라인

새 스크리닝 가설을 운영에 넣기 전에 통계로 거르는 장치다. 운영 v2 스크리닝은 아직 이것으로 교체되지 않았다.

- 규칙 = `src/config/screen_rules/<분기>/*.json` (7필드). **해시 불변** — 값 하나 바꾸면 새 규칙, 새 분기. 공개본에는 `example/`만 있다.
- 레지스트리 `screen_rules`·`screen_rule_events`·`screen_quarters`·`screen_selections` (mentions.db, `MENTIONS_DB_PATH` 따름).
- 상태 기계: `draft → gated → shadow → active → retired` (탈락은 `rejected`).
- 게이트: `python -m src.screening.gate <rule_id>` / `--all-draft` / `--register-dir DIR --quarter Q` / `--declare-m N --quarter Q`. LLM 0콜.
  판정은 **리밸런스 날짜 단위 전체 기간** t ≥ Bonferroni t\*, OOS는 붕괴 여부만 확인. 벤치마크 둘(KOSPI200, 같은 창 유니버스 동일가중). 판정선은 `gate.THRESHOLDS`.
- **분기 m 사전 선언**: 첫 게이트 전에 그 분기의 가설 수를 고정해 통과선(t\*)이 사후에 바뀌지 않게 한다.
- 신호 `src/screening/signals/` — 백테스트·라이브 공용. 수급 이력은 `src/data/flow_history.py`(네이버, 자격증명 불필요).
- 게이트는 라이브와 같은 청산(t+1 시가 → H일 종가)을 시뮬레이션한다.
- 무작위 대조군은 (유니버스·H·N·기간·풀) 키로 `data/backtest/control_*.json`에 캐시.
- 뉴스 관련도 필터(`src/data/headline_relevance.py`)와 캐시 정합성 검사(`ohlcv_cache.find_suspect_tickers`)는 운영에 적용돼 있다.

### 2단계 — 종목별 파이프라인 (`src/graph/pipeline.py`)

LangGraph `StateGraph`. 확정 종목마다 순차로 1회씩 돈다.

```
data_ingest (30s)          주가·VIX·환율·KOSPI·미국채·WTI 6개 병렬
  → regime_detector (30s)  KOSPI MA20/60 + VIX → bull/bear/sideways/volatile/neutral
  → parallel_analysis (360s)  8명 투표. 전체 Semaphore(3), KRX Semaphore(2)
  → quality_gate (30s)     confidence≥0.6 AND reasoning≥3 AND data_sources≥2
  → debate (90s)           BUY·SELL 가중합이 팽팽할 때만
  → chief_strategist (120s)  방향 확정 + 서술 + signal_reconciliation 호출
  → report_formatter (30s)   v4.1 7섹션 마크다운
  → [notion_publish (30s)]   publish=True 일 때만 (단독 실행 시)
  → log_predictions (30s)    prediction_log + regime_history
```

- 모든 노드가 `node_with_timeout()`으로 감싸여 있다. 타임아웃 시 `{}`(빈 업데이트)를 반환하고 **다음 노드로 계속 간다**.
- **한계**: `asyncio.timeout`은 다음 `await` 지점에서만 취소한다. `asyncio.to_thread`로 띄운 동기 워커(pykrx·yfinance)는 죽일 수 없어 SDK 레벨 타임아웃을 따로 건다.
- **Notion 발행은 파이프라인 밖이다.** `daily_runner`가 `publish=False`로 종목별 파이프라인을 돌리고, 결과를 모아 **통합 리포트 1건**을 발행한다.

`GraphState` (`src/schemas/graph_state.py`) 누적 방식:

| 필드 | 방식 | 비고 |
|---|---|---|
| `analysis_reports` | `operator.add` 누적 | chief가 `[final]`만 반환해도 8명 리포트가 남아 있다 |
| `qualified_reports` | 교체 | Quality Gate 통과분. debate·chief가 **이것을 우선** 쓴다 |
| 나머지 | 교체 | `ticker` `screen_context` `market_data` `reconciled_signals` `final_strategy` `report_content` `current_regime` `debate_summary` `error_log` |

`screen_context`는 **오늘 이 종목이 뽑힌 사유**다 — 공시 제목·뉴스 급증 배수·1-C 해석.
`daily_runner`가 `build_screen_context(ticker, screen)`으로 만들어 `run_pipeline`에 넘긴다.
`sentiment_analyst`와 `chief_strategist`가 읽고, 리포트의 종목별 "선정 사유" 줄이 같은 문장을 쓴다.
단독 디버그 실행에서는 빈 dict다.

### 최종 방향 결정

**방향은 LLM이 아니라 파이썬 규칙이 정한다** (`chief_python.decide()`).

- 지정된 투표자들의 표를 규칙으로 집계한다. 투표자 목록과 집계 규칙은 `decide()`와 그 테스트(`tests/test_decision_rule.py`)가 기준이다.
- **확신도**는 LLM의 자기보고 confidence가 아니라 투표 결과로 계산한다.
- **abstain**: confidence 0.0은 '의견 없음'이다 — 참여자에서 뺀다(반대표가 아니다). 참여자가 0명이면 HOLD + confidence 0.0 → '판단 없음'이라 원장에 남지 않는다.
- 표를 던지지 않는 에이전트도 리포트와 chief 프롬프트에는 그대로 실리고 `prediction_log`에도 기록돼, 나중에 "누가 맞았나"를 잴 수 있다.

**왜 바꿨나**: 프롬프트에 판정 규칙을 적어 두었지만 운영 40건을 그 규칙으로 재현하면 BUY 15 / SELL 20 / HOLD 5인데,
opus가 실제로 낸 것은 **BUY 3 / SELL 0 / HOLD 37**이었다. 지시문으로 준 규칙은 지켜지지 않았다.

chief(opus)는 이제 방향을 설명하고 리스크를 쓰는 역할이다. 다른 방향을 제출해도
`run_chief_strategist`가 확정값으로 되돌리고 그 사실을 `reasoning`에 남긴다.
토론과 에이전트 가중치(`weight_context`)도 방향이 아니라 문장에만 영향을 준다.

### 8명의 투표자 + 종합

| 에이전트 | 역할 | 모델 | 비고 |
|---|---|---|---|
| `macro_economist` | 금리/환율/원자재/DXY | gpt-4o-mini | `daily_cache` — 하루 1콜 |
| `kr_market_specialist` | 수급/대주주매매 | gpt-4o-mini | KRX 세마포어 |
| `us_market_specialist` | S&P500/VIX/Treasury | gpt-4o-mini | `daily_cache` — 하루 1콜 |
| `quant_analyst` | PER/PBR/모멘텀 | gpt-4o-mini | KRX 세마포어 |
| `technical_analyst` | MA/RSI/MACD/볼린저/일목균형표 | gpt-4o-mini | KRX 세마포어 |
| `sentiment_analyst` | **종목 이벤트 해석** (공시 제목 + 종목 헤드라인 + 선정 사유) | gpt-4o-mini | MCP 호출 없음 |
| `fundamental_analyst` | 실적/컨센서스/DART | gpt-4o-mini | KRX 세마포어, timeout 180s |
| **`quant_rule_agent`** | **LLM 없는 8번째 투표자** | **없음** | ohlcv_cache만 읽는다 |
| `debate` (Bull/Bear) | 조건부 토론 | claude-sonnet-4-6 | 4콜 |
| `chief_strategist` | 확정된 방향의 설명·리스크 | claude-opus-4-6 | 방향은 `decide()`가 정함 |

**`quant_rule_agent`가 왜 있나**: "LLM 7명이 규칙 하나보다 나은가"를 재려면 같은 날 같은 종목에 대한
규칙의 답이 필요하다. 이것이 그 비교 기준선이고, `src/evaluation/llm_contribution.py`가 둘을 짝지어 집계한다.
`parallel_analysis` 안에서 나머지 7명과 같은 `gather`로 돈다.
chief 프롬프트도 "8개 투표자"로 맞췄다 (`tests/test_prompt_order_bias.py`가 고정 — 공개본에서는 skip).

**`CHIEF_MODE` 환경변수** (`src/agents/chief_python.py`):
- `legacy` (기본) — claude-opus-4-6이 서술 전부를 맡는다
- `python_sonnet` — 투표 산술은 파이썬이, 문장만 claude-sonnet-5가 쓴다 (비용 대폭 감소). 교체는 사람이 결정한다.

### 공통 출력 스키마 (`src/schemas/agent_output.py`)

모든 에이전트가 `AnalysisReport`를 반환한다. LangChain `with_structured_output()`으로 강제.

| 필드 | 조건 |
|---|---|
| `recommendation` | `BUY` / `SELL` / `HOLD` |
| `confidence` | 0.0–1.0 |
| `reasoning` | 3단계 이상 (Chain-of-Thought) |
| `data_sources` | 2개 이상 |
| `prediction_basis` | 2개 이상 (정량 근거) |
| `risk_factors` | 1개 이상 |
| `data_sufficient` | **false면 위 길이 조건이 전부 면제된다** |
| `ticker` / `ticker_name` | LLM이 채우지 않음. 파이프라인이 사후 주입 |
| `needs_review` | LLM 출력 파싱 실패 시 True (HOLD로 날조하지 않기 위한 표식) |
| 거래 파라미터 9개 | `entry_price` `stop_loss` `stop_loss_pct` `take_profit_1` `take_profit_2` `rr_ratio` `position_size_pct` `holding_period_weeks` `entry_strategy` — chief만 채움 |

**`data_sufficient`의 존재 이유**: 길이 조건을 무조건 강제하면, MCP 수집이 실패해 입력이 비었을 때도
LLM이 "정량적 근거 2개"를 만들어내야 스키마를 통과한다. 재시도 루프가 날조를 압박하는 구조였다.
→ `data_sufficient=false`라는 탈출구를 만들고 그때만 길이 요구를 푼다.
조건부 검증은 `@model_validator(mode="after")`로 구현했다. 스키마를 위반하면 `base_agent`가 위반 내용을 피드백으로 붙여 1회만 재요청한다.

**`ticker` 주입이 왜 중요한가**: 이 필드가 비어 있어서 `feedback_evaluator`가 예측을 전부 건너뛰고
채점이 한 건도 수행되지 않은 전례가 있다. 폴백 리포트에도 반드시 주입해야 한다.

**수집 실패 표기**: MCP 수집이 실패한 항목은 프롬프트에 "수집 실패"로 명시한다(`src/utils/mcp_result.py`).
값이 원래 없는 것과 수집에 실패한 것을 LLM이 혼동하지 않게 하기 위해서다.

### Quality Gate (`src/graph/quality_gate.py`)

```
통과 조건 (AND):  confidence ≥ 0.6  AND  reasoning ≥ 3  AND  data_sources ≥ 2
통과 < 2명이면:   소프트 폴백 — 필터를 풀되 confidence=0.0(abstain)은 계속 제외
```

`confidence=0.0`은 "관망 의견"이 아니라 **"의견 없음"**이다. 폴백에서도 투표에 넣지 않는다.

### 신호 정원 (`src/graph/signal_reconciliation.py`)

chief_strategist 직후 호출. 하루 매수 후보 **최대 5종목**, **섹터당 2종목**.
레지스트리는 `data/signals/signals_YYYY-MM-DD.json`에 저장.
BUY면 `positions` 테이블에 자동 기록되며, chief가 계산한 손절·목표가를 그대로 넘긴다
(넘기지 않으면 기본값으로 대체되어 리포트와 시스템이 불일치한다).

### 손절·포지션

**손절은 LLM이 정하지 않는다.** `src/utils/atr.py`가 `ohlcv_cache`에서 ATR14를 계산해 손절폭을 정하고,
봉이 모자라면 `DEFAULT_STOP_PCT`를 쓴다. `chief_python.apply_atr_stop()`이 `CHIEF_MODE` 두 경로에 공통으로 적용한다.
포지션 크기는 1회 손절 시 시드 손실이 `RISK_PER_TRADE_PCT`를 넘지 않도록 줄인다.

고정 손절폭이 종목 변동성과 무관하게 노이즈에 자주 걸리던 문제를 실측으로 확인하고 바꿨다.
**주의**: 이 변경은 "손절이 노이즈에 걸리는 문제"를 고치는 것이지 수익을 만드는 것이 아니다.

### 3단계 — 평가·피드백

| 장치 | 파일 | 무엇을 재나 | 필요 표본 |
|---|---|---|---|
| D+5 채점 | `data/feedback_evaluator.py` | 에이전트별 적중 → EMA 가중치 | 30표본 + z-검정 |
| chief 원장 | `data/prediction_logger.py` | chief 최종 판단을 `prediction_log`에 기록 — 시스템 출력을 재는 표본 | — |
| chief 교훈 메모리 | `data/decision_memory.py` | chief 판단을 D+1로 복기한 2~4문장 교훈 → 다음 chief 프롬프트에 주입 | 1건부터 작동 |
| D+5/10/20 | `evaluation/outcomes.py` | 실현 수익 + KOSPI200 대비 알파 | 약 175거래일 |
| 무작위 대조군 | `evaluation/random_control.py` | 운으로 찍은 것보다 나은가 | 누적 중 |
| 확신도 보정 | `evaluation/calibration.py` | "신뢰도 0.8"이 정말 80%인가 | 30건 |
| LLM 기여도 | `evaluation/llm_contribution.py` | LLM이 규칙 투표자보다 나은가 | — |
| 스크리닝 팩터 | `evaluation/screen_factors.py` | 팩터별 유니버스 대비 D+5/20 초과수익 | 30 날짜 |
| 백테스트 | `evaluation/backtest/` | 과거 재현 (거래비용 차감 · 노출 일치 벤치마크 · walk-forward) | 규칙층만 가능. LLM 포함은 불가 |

**채점 지평 D+5**: 표방 포지션이 1~4주인데 하루 등락으로 매긴 점수가 EMA를 움직이고 있어서 D+1에서 바꿨다.
근거는 `prediction_outcomes`의 확정 행이라 **`outcomes`가 `feedback_evaluator`보다 먼저 돌아야 한다**.
점수표는 BUY·SELL·HOLD 세 방향 대칭이다 (종전에는 HOLD만 부분점수가 없어 한쪽 방향만 내는 에이전트가 유리했다).
옛 D+1 점수는 `EVALUATED_D1_LEGACY`로 표시돼 증거 집계에서 빠지고, D+5가 확정되면 같은 행이 다시 채점된다.
이 이관은 `eval_method_log`의 `scoring_d5` 키로 한 번만 돈다.

**동적 EMA α**: Volatile 레짐 또는 5일 내 레짐 전환 → α=0.5. 안정 시 base_alpha (`BASE_REGIME_ALPHA`).
**증거 게이트**: 가중치가 균등(`INITIAL_WEIGHT` = 1/8)에서 벗어나려면 표본 30개(`WARMUP_SAMPLE_COUNT`)
**그리고** 평균 점수가 0.5(동전 던지기)와 통계적으로 다름(|z| ≥ 1.96, `has_weight_evidence()`)이 필요하다.
읽기·쓰기 양쪽이 같은 함수로 매일 다시 판정한다 — 한 번 통과해도 영구가 아니다.
**계층 폴백**: 레짐별 증거가 없으면 레짐 무관 `All` 행(`ALL_REGIME`)의 가중치, 그것도 없으면 균등.
레짐 5분할이 표본 요구량을 5배로 늘리는 문제를 "찰 때까지 합쳐 쓰는" 방식으로 푼다.

**chief 교훈 메모리** (TradingAgents `memory.py` 방식): EMA 가중치가 증거 게이트를 통과할 때까지
chief가 매일 백지에서 판단하는 공백을 메운다. `chief_decisions` 테이블에 기록 → 장 마감 후 **D+1** 채점 +
gpt-4o-mini 복기 → `chief_strategist_node`가 `get_past_context(ticker, as_of=오늘)`로 주입.
가중치 채점(D+5)과 달리 여기는 **D+1 그대로다** — 교훈은 매일 쌓여야 다음 날 프롬프트에 들어간다.
복기 LLM이 실패하면 채점만 저장하고 `resolved=0`으로 남긴다. `python -m src.data.decision_memory YYYY-MM-DD`로 재시도한다.
**chief에게만** 주입한다 — 투표자 8명에게 보여주면 `llm_contribution` 비교가 오염된다.
`as_of`는 백테스트용 시점 차단이다 (그 날짜 이후에 확정된 교훈은 안 보인다).

### 데이터 저장 — `data/mentions.db` (SQLite 단일 파일, 16 테이블)

| 그룹 | 테이블 |
|---|---|
| 캐시 | `ohlcv_cache` `mentions` `daily_mention_stats` |
| 원장 | `prediction_log` `positions` `regime_history` `agent_weights` `chief_decisions` `dart_events` |
| 채점 | `prediction_outcomes` `random_control_outcomes` `universe_snapshots` `screen_candidates` `news_archive` `llm_ab_log` `eval_method_log` |

`prediction_log`에는 **투표자 8명 + chief** 가 들어간다(`LOGGED_AGENTS`). `agent_weights`(EMA 가중치)는
**투표자 8명만**(`SCORED_AGENTS`) — chief는 기록·채점 대상이지 가중치 대상이 아니다.

**GitHub Actions는 매 실행이 새 머신이다.** 운영 워크플로는 `actions/cache`로 이 파일을 통째로 넘기고,
캐시가 사라질 때를 대비해 별도 브랜치에 최신 1부를 백업한다(이중 보관).
캐시가 끊기면 "언급량 급증"처럼 과거 기준선이 필요한 판단이 전부 무력해진다.
**캐시·백업 대상은 `mentions.db` 하나뿐이다.** `chroma_db/` `signals/` `reports/`는 매 실행 백지에서 시작한다.

### MCP 서버 (`src/mcp_servers/`) — 도구 32종

`krx_market`(14) / `news_economy`(10) / `us_market`(8). FastMCP 기반.
**별도 파이썬 프로세스로 뜬다** — "왜 env 키가 안 넘어가지" 류 사고가 이 경계에서 난다(`src/utils/mcp_client.py`).
파이프라인에서는 MCP 프로토콜이 아니라 도구 함수를 직접 호출한다.

### RAG (`src/rag/`)

ChromaDB + BM25 하이브리드(Kiwi 형태소 분석). 에이전트별 컬렉션·토큰 예산(총 2000)을
`AGENT_RAG_CONFIG`가 관리하고 `inject_context_into_prompt()`가 시스템 프롬프트에 주입한다.
컬렉션: `market_reports` `analyst_reports` `news_articles` `strategy_outcomes` `earnings_data`

**⚠️ 운영에서는 항상 비어 있다.** 운영 경로(`daily_runner`·`pipeline`·에이전트)에 ChromaDB 적재
호출이 없고, `data/chroma_db/`는 캐시 대상도 아니다. 검색 결과는 늘 0건이다.

### 비용

2026-09-13 실행 실측(4종목, 40콜, **$0.284**). 상한 `LLM_BUDGET_USD`(기본 5.0, 실행 1회, 초과 시 종목 경계에서 중단).

| 구성요소 | 비용 | 비중 |
|---|---|---|
| chief_strategist (opus) | $0.262 | **92%** |
| 전문가 7명 (gpt-4o-mini) | $0.015 | 5% |
| 뉴스 스크린 (haiku) | $0.006 | 2% |
| 감성 분류 (gpt-4o-mini) | $0.001 | 0.4% |

**비용을 줄이려면**: 종목 수를 줄이거나 `CHIEF_MODE=python_sonnet`.

---

## ⚠️ 절대 지킬 규칙

### [DEP-01] langchain-anthropic 설치 금지

설치하면 langchain-core 1.3.3 → 0.3.86 자동 다운그레이드가 발생해 시스템 전체가 깨진다.
**Anthropic 모델은 `anthropic` SDK를 직접 쓴다** (`AsyncAnthropic`).
`src/agents/base_agent.py`의 `ResilientChain`이 이를 처리하며 `chief_strategist`·`debate`도 같은 패턴이다.

### [DEP-02] langchain-core 1.3.3 고정

`requirements-freeze.txt`에 고정. langchain 0.3.25(0.3.x계) + langchain-core 1.3.3(1.x계) 조합은
pip 의존성 해석으로는 불가능하다. 새 환경 구축 시 반드시:

```powershell
pip install --no-deps -r requirements-freeze.txt   # 로컬
pip install --no-deps -r requirements-ci-frozen.txt # CI
```

### [DEP-03] pybreaker는 async 함수에 `.call_async()`

`pybreaker.call()`은 동기 함수 전용. async에서 잘못 쓰면 Circuit Breaker가 OPEN으로 전환되지 않는다.

### [SEC-01] API 키 로그 마스킹

`src/utils/security.py`의 `setup_secure_logging()`이 진입점에서 호출된다. OpenAI/Anthropic/Notion 키 자동 마스킹.

### [SEC-02] ticker 입력 검증

`run_pipeline()`에서 `validate_ticker()` 호출. 6자리 숫자 + 화이트리스트. SQL Injection 차단.

### [SEC-03] 리포트 발행 전 검증

`notion_publish` 노드에서 `validate_report()` 호출. None 포함·빈 리포트·필수 섹션 누락 시 발행 차단.
v4.1 리포트는 섹션 4의 `**최종 판단**` 라벨로 이 검증을 통과한다.

---

## 환경 설정

### Python

- **Python 3.11.9** (정확한 버전)
- 가상환경 `.venv/` — **영문 경로 필수** (한글 경로에서는 일부 패키지가 깨진다)
- TA-Lib은 나머지보다 **먼저** 설치한다:
  ```powershell
  pip install TA_Lib-0.4.32-cp311-cp311-win_amd64.whl
  pip install --no-deps -r requirements-freeze.txt
  ```
  Ubuntu(CI)에서는 패키지가 없어 소스 컴파일한다 (`.github/workflows/tests.yml` 참조).

### 주요 라이브러리 (실측 확인)

```
langchain 0.3.25   langchain-core 1.3.3 (고정)   langgraph 0.4.1
anthropic 0.50.0   openai 1.75.0                 fastmcp 3.2.4
chromadb 1.5.9     kiwipiepy 0.23.1              pydantic 2.13.3
pykrx 1.2.7        yfinance 1.3.0                notion-client 2.3.0
tenacity 9.1.4     pybreaker 1.4.1               APScheduler 3.11.0
pytest 8.3.5       pytest-asyncio 0.26.0
```

### 환경변수 — 코드가 실제로 읽는 것

| 키 | 쓰는 곳 |
|---|---|
| `OPENAI_API_KEY` | 전문가 7명 + 감성 분류 + chief 복기(decision_memory) |
| `ANTHROPIC_API_KEY` | chief_strategist + debate + 뉴스 스크린 |
| `FRED_API_KEY` `BOK_ECOS_API_KEY` | 거시경제 |
| `NAVER_CLIENT_ID` `NAVER_CLIENT_SECRET` | 뉴스/언급량 |
| `KRX_ID` `KRX_PW` `KRX_OpenAPI` | 한국 시장 (pykrx가 내부에서 읽음) |
| `DART_API_KEY` | 공시 |
| `NOTION_API_KEY` `NOTION_DATABASE_ID` | 리포트 발행 |
| `CHIEF_MODE` | `legacy`(기본) / `python_sonnet` |
| `LLM_BUDGET_USD` | 실행 1회 상한 (기본 5.0) |
| `SHADOW_MODE` | 섀도 A/B (기본 on, `off`로 끔) |
| `MENTIONS_DB_PATH` | DB 경로 오버라이드 (테스트용) |
| `LANGCHAIN_API_KEY` `LANGCHAIN_PROJECT` `LANGCHAIN_ENDPOINT` | LangSmith (선택) |

**⚠️ `.env.example`이 낡았다.** `NOTION_TOKEN`(코드는 `NOTION_API_KEY`를 읽는다), 쓰지 않는
`GOOGLE_API_KEY`/`HUGGINGFACE_API_KEY`가 남아 있고, 위 표의 절반이 빠져 있다.

---

## 디렉토리 구조

```
src/
├── daily_runner.py          ★ 운영 진입점 — 스크리닝·종목반복·건강도·통합발행
├── scheduler.py               APScheduler. 코드로만 존재, 미가동
├── agents/                    투표자 8명 + 토론 + 종합
│   ├── base_agent.py          ResilientChain, 스키마 위반 시 1회 재요청
│   ├── chief_strategist.py    claude-opus-4-6 + tool_use
│   ├── chief_python.py      ★ 방향 결정 규칙 decide() + CHIEF_MODE=python_sonnet
│   ├── quant_rule_agent.py  ★ LLM 없는 8번째 투표자
│   ├── debate.py              Bull/Bear
│   └── (전문가 7종)
├── screening/               ★ 110종목 → 오늘 새 정보가 있는 종목 (v2)
│   ├── screener.py            4단계 오케스트레이션
│   ├── stage1a_quant / stage1b_events ★ / stage1c_news / stage2_scorer(v2 ★)
│   ├── rules/ ★               RuleSpec(7필드·spec_hash) · 레지스트리·상태 기계
│   ├── signals/ ★             백테스트·라이브 공용 신호
│   └── gate.py ★              규칙 게이트 판정
├── config/
│   ├── universe_config.json
│   └── screen_rules/example/  규칙 형식 예시 (공개본)
├── graph/                     LangGraph 파이프라인과 산출물
│   ├── pipeline.py            StateGraph 정의
│   ├── quality_gate.py  regime_detector.py  signal_reconciliation.py
│   ├── report_formatter.py    v4.1 7섹션
│   └── notion_publisher.py    v4.0 블록 빌더
├── evaluation/              ★ 성과 측정
│   ├── outcomes.py            D+5/10/20 + 알파
│   ├── calibration.py  performance.py  llm_contribution.py
│   ├── random_control.py  news_archive.py  screen_factors.py ★
│   └── backtest/ (7파일)
├── data/                      수집·저장·채점
│   ├── mention_tracker.py  mention_db.py  ohlcv_cache.py  dart_events.py ★  flow_history.py ★
│   ├── prediction_logger.py  feedback_evaluator.py  decision_memory.py ★
│   └── position_tracker.py  sentiment_classifier.py  headline_relevance.py ★
├── mcp_servers/               krx_market(14) · news_economy(10) · us_market(8)
├── universe/                  universe_builder.py  filters.py
├── rag/                       chroma_store  hybrid_retriever(Kiwi+BM25)  context_injection
├── schemas/                   agent_output.py(AnalysisReport)  graph_state.py
└── utils/
    ├── resilience.py          Retry + CircuitBreaker + Timeout + Fallback
    ├── security.py            키 마스킹 + ticker/report 검증
    ├── atr.py ★               ATR 기반 손절폭
    └── mcp_client ★  mcp_result ★  shadow_log ★  llm_budget ★
        date_window ★  daily_cache ★  market_session ★

tests/                         1,015건 (not costly 904 / costly 111)
data/mentions.db               SQLite 16 테이블 (저장소에 없음, 실행 시 생성)
data/chroma_db/                ChromaDB
data/reports/                  리포트 markdown 백업 (Notion 실패 시 폴백)
data/signals/                  일별 신호 레지스트리
```

★ = 2026-09 신규. 프로젝트 초기 설계에 없던 것.

---

## 코드 작성 컨벤션

### Async 패턴

모든 노드와 에이전트는 `async def`다. `pipeline.ainvoke()` / `agent.ainvoke()`를 쓴다.
`invoke()`(동기)는 **작동하지 않는다**.

### Resilience 적용 패턴

새 외부 API 호출을 추가할 때:

```python
from src.utils.resilience import safe_call, krx_breaker

result = await safe_call(
    fetch_something,              # async 함수
    ticker="005930",              # func에 그대로 전달
    breaker=krx_breaker,          # None이면 CB 미적용
    timeout_seconds=60,
    fallback_value={"error": "수집 실패"},
    task_name="fetch_something",  # 로그용
)
```

`safe_call` 하나가 Retry(3회) + CircuitBreaker + Timeout + Fallback을 전부 처리한다.
실패해도 예외를 올리지 않고 `fallback_value`를 반환한다.

브레이커는 모듈 레벨 변수이고 서비스별로 설정이 다르다:

| 변수 | fail_max | reset |
|---|---|---|
| `openai_breaker` `anthropic_breaker` | 5 | 30초 |
| `dart_breaker` `fred_breaker` `krx_breaker` | 3 | 60초 |

### MCP 도구 추가 패턴

```python
@mcp.tool()
def new_tool(param: str) -> dict:
    """도구 설명 (LLM이 이 docstring을 보고 호출 여부를 결정한다)"""
    try:
        return {"result": ...}
    except Exception as e:
        return {"error": str(e)}   # 절대 raise 금지 — 파이프라인 중단 방지
```

### 에이전트 추가 패턴

`src/agents/`에 파일 추가 → `pipeline.py`의 `agent_names` 목록과 `gather` 호출에 등록 →
`prediction_logger`의 가중치 테이블에 자동 등록된다.

### 테스트 추가

LLM이나 외부 인증을 호출하는 테스트라면 **반드시** `tests/conftest.py`의 `COSTLY_MODULES`에 모듈명을 넣는다.
넣지 않으면 push마다 과금된다.

---

## 알려진 이슈

| ID | 내용 | 상태 |
|---|---|---|
| ISSUE-EVAL-01 | 수익성 미검증 — D+5/10/20 표본 거의 0 | 약 175거래일 필요 |
| ISSUE-EVAL-02 | 에이전트 가중치 전원 균등 — 증거 게이트(30표본 + z-검정) 미통과 | `All` 행은 8명 합산이라 먼저 찬다 |
| ISSUE-EVAL-03 | calibration 표 비어 있음 → 포지션 비중 고정값 | D+10 확정 30건 필요 |
| ISSUE-RAG-01 | 운영 RAG가 항상 비어 있음 — 적재 코드 없음 + 캐시 미포함 | chief 교훈은 decision_memory가 대체 |
| ISSUE-STRUCT-01 | 포트폴리오 구성 없음. 종목별 비중만 표시, 합계·현금 배분 로직 없음 | 설계 결정 필요 |
| ISSUE-OPS-01 | cron 지연으로 장중 발행 가능 | 감지해 리포트에 표시만 함 |
| ISSUE-OPS-02 | DART 간헐적 TCP connect 실패 | 경고로만 기록 |
| ISSUE-SCHEMA-01 | gpt-4o-mini가 `risk_factors`를 빠뜨려 재요청 발생 | 1회 재요청 후 실패 시 abstain |
| ISSUE-BT-01 | LLM 포함 백테스트 불가 (뉴스 아카이브 축적 시작 단계) | 아카이브가 쌓인 뒤 |
| ISSUE-TEST-01 | `test_*_selection_rationale_filled` 2건 실패 — LLM이 해당 필드를 비움 | costly 마크됨. 인프라 문제 아님 |
| ISSUE-DOC-01 | `.env.example`이 실제 환경변수와 불일치 | 위 환경변수 표 참조 |
| ISSUE-SCREEN-01 | 스크리닝 v2 워밍업 — `news_burst`가 기준선 기간이 찰 때까지 꺼져 있어 공시 종목만 후보 | 자동 해소 |

---

## 결정론에 대해

**같은 데이터로 다른 결과가 나올 수 있다. 의도된 동작이다.**

1. LLM은 완벽한 결정론이 아니다 (temperature=0이어도 GPU 부동소수점 차이)
2. 시간이 다르면 외부 데이터가 다르다 (pykrx·yfinance는 호출 시점 데이터)
3. DB 상태가 누적된다 (agent_weights, prediction_log가 매일 갱신)

→ "매일 새로운 시장 정보로 다시 판단하는 시스템"이므로 비결정성이 설계 의도다.
단, **최종 방향은 `decide()`가 정하므로 같은 투표 입력이면 같은 방향이 나온다.**

---

## 개발 원칙

1. **정확성 최우선** — 추측 금지. 검증 가능한 사실만 코드로 쓴다.
2. **Walking Skeleton** — 전체 흐름이 도는 것이 부분의 완벽함보다 먼저다.
3. **실패 격리** — 한 에이전트/도구의 실패가 전체를 멈추지 않는다.
4. **Fallback 필수** — 모든 외부 API는 캐시/기본값/에러 dict로 안전 반환한다.
5. **타입 검증** — Pydantic 스키마로 LLM 출력을 강제 구조화한다.
6. **날조 금지** — 데이터가 없으면 없다고 선언하게 만든다. 스키마가 거짓말을 압박하지 않도록 설계한다.
