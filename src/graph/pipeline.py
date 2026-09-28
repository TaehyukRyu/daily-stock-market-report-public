"""
AI 투자 리포트 파이프라인 v2.8

[변경사항]
  v2.1~v2.6: (생략 — 인수인계 문서 참조)
  v2.7: 복원력 추가 (node_with_timeout, 포지션 섹션 복원)
  v2.8: 보안 추가
        - setup_secure_logging(): API 키 로그 마스킹
        - validate_ticker(): 6자리 숫자 검증 + SQL Injection 차단
        - validate_report(): Notion 발행 전 필수 섹션 검증
"""

import asyncio
from langgraph.graph import StateGraph, END
from src.schemas.graph_state import GraphState
from src.schemas.agent_output import AnalysisReport
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2", 'false')
os.environ['LANGCHAIN_ENDPOINT']   = os.getenv("LANGCHAIN_ENDPOINT", "")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "")

# ── 보안 초기화 ───────────────────────────────────────────────────────────────
from src.utils.security import (
    setup_secure_logging,
    validate_ticker,
    validate_report,
    set_ticker_whitelist,
    InvalidTickerError,
    InvalidReportError,
)

setup_secure_logging()   # 모든 logger에 API 키 마스킹 필터 자동 적용

# ── 피드백 루프 초기화 ────────────────────────────────────────────────────────
from src.data.prediction_logger import (
    setup_feedback_system,
    log_agent_predictions,
    get_agent_weights,
)
from src.data.position_tracker import (
    setup_position_tracker,
    format_position_section,
)

setup_feedback_system()
setup_position_tracker()


# ──────────────────────────────────────────────────────────
# 노드 Timeout 래퍼 (v2.7)
# ──────────────────────────────────────────────────────────

def node_with_timeout(node_fn, timeout_seconds: float, node_name: str):
    """
    노드 단위 timeout 래퍼.

    한계: asyncio.timeout은 다음 await 지점에서만 task를 cancel한다.
    asyncio.to_thread(...)로 띄운 동기 worker thread는 OS 레벨에서 종료시킬
    수 없어, 외부 라이브러리(pykrx/yfinance/anthropic 등)가 무한히 매달리면
    worker thread는 계속 살아있다. 따라서 외부 라이브러리에 timeout 인자가
    있을 때는 해당 SDK 레벨에서도 잘라야 한다.
      - pykrx 동기 호출 → ThreadPoolExecutor.submit().result(timeout=N)
        (예: src/screening/stage1a_quant.py의 _backfill_one / _bulk_snapshot)
      - yfinance asyncio.to_thread 호출 → asyncio.wait_for(..., timeout=N)
        (data_ingest 노드에서 6개 호출 모두 20초 wait_for 적용)
      - anthropic SDK → Anthropic(api_key=..., timeout=N)
        (stage1c_news.py의 _call_llm에서 60초 timeout 적용)
    이 래퍼만으로는 worker thread를 끊을 수 없다는 점에 유의.
    """
    async def wrapper(state: GraphState) -> dict:
        try:
            async with asyncio.timeout(timeout_seconds):
                return await node_fn(state)
        except asyncio.TimeoutError:
            logger.error(f"[NodeTimeout] {node_name} — {timeout_seconds}초 초과")
            print(f"  ⏱️ [{node_name}] {timeout_seconds}초 초과 — 건너뜀")
            return {}
        except Exception as e:
            logger.error(f"[NodeError] {node_name}: {type(e).__name__}: {e}")
            print(f"  ❌ [{node_name}] 예외: {e}")
            return {}
    wrapper.__name__ = node_fn.__name__
    return wrapper


# ──────────────────────────────────────────
# 노드 1: 시장 스냅샷
# ──────────────────────────────────────────

async def data_ingest(state: GraphState) -> dict:
    print(f"\n[1/7] data_ingest — 종목: {state.ticker}")
    try:
        from src.mcp_servers.krx_market.server   import get_stock_price
        from src.mcp_servers.us_market.server    import (
            get_vix, get_treasury_yields, get_commodity_prices,
        )
        from src.mcp_servers.news_economy.server import get_exchange_rate
        import yfinance as yf

        def _fetch_kospi() -> float | None:
            try:
                hist = yf.Ticker("^KS11").history(period="5d")
                if not hist.empty:
                    return round(float(hist["Close"].iloc[-1]), 2)
            except Exception as e:
                logger.warning(f"[data_ingest] KOSPI 조회 실패: {e}")
            return None

        # 6개 외부 호출을 병렬로 (시장 지표 누락 방지 + 단일 지연 노출 회피).
        # 각 호출 20초 timeout — yfinance/pykrx 등 동기 라이브러리가 매달리면
        # gather가 무한 대기에 빠지지 않도록. TimeoutError는
        # gather(return_exceptions=True)에 의해 캡처되어 _safe 폴백으로 흡수된다.
        (
            stock_data, vix_data, exchange_data,
            kospi_value, treasury_data, commodity_data,
        ) = await asyncio.gather(
            asyncio.wait_for(asyncio.to_thread(get_stock_price, state.ticker, 5), timeout=20),
            asyncio.wait_for(asyncio.to_thread(get_vix),                          timeout=20),
            asyncio.wait_for(asyncio.to_thread(get_exchange_rate, 5),             timeout=20),
            asyncio.wait_for(asyncio.to_thread(_fetch_kospi),                     timeout=20),
            asyncio.wait_for(asyncio.to_thread(get_treasury_yields),              timeout=20),
            asyncio.wait_for(asyncio.to_thread(get_commodity_prices, ["WTI"]),    timeout=20),
            return_exceptions=True,
        )

        def _safe(v, default=None):
            return default if isinstance(v, Exception) else v

        stock_data    = _safe(stock_data, {})
        vix_data      = _safe(vix_data, {})
        exchange_data = _safe(exchange_data, {})
        kospi_value   = _safe(kospi_value)
        treasury_data = _safe(treasury_data, {}) or {}
        commodity_data = _safe(commodity_data, {}) or {}

        us_10y_value = (treasury_data.get("10y") or {}).get("close") if isinstance(treasury_data, dict) else None
        wti_value    = (commodity_data.get("WTI") or {}).get("price") if isinstance(commodity_data, dict) else None
        usd_krw_value = exchange_data.get("latest_rate") if isinstance(exchange_data, dict) else None

        return {"market_data": {
            "ticker":        state.ticker,
            "stock":         stock_data,
            "vix":           vix_data,
            "exchange_rate": exchange_data,
            # ── flat 시장 지표 (report_formatter._market_data_section 키와 일치) ──
            "kospi":         kospi_value,
            "us_10y_yield":  us_10y_value,
            "usd_krw":       usd_krw_value,
            "wti":           wti_value,
            "source":        "live",
        }}
    except Exception as e:
        print(f"  ⚠️ 스냅샷 수집 실패 (계속 진행): {e}")
        return {"market_data": {"ticker": state.ticker, "source": "fallback"}}


# ──────────────────────────────────────────
# 노드 1.5: 시장 레짐 탐지
# ──────────────────────────────────────────

from src.graph.regime_detector import regime_detector_node


# ──────────────────────────────────────────
# 노드 2: 8개 에이전트 병렬 실행 (LLM 7 + quant_rule_agent)
# ──────────────────────────────────────────

async def parallel_analysis(state: GraphState) -> dict:
    ticker = state.ticker
    print(f"\n[2/7] parallel_analysis — 8개 에이전트 병렬 실행 (최대 3개 동시)")
    print(f"       대상 종목: {ticker}")
    print(f"       현재 레짐: {state.current_regime.upper()}")

    from src.agents.macro_economist      import run_macro_economist
    from src.agents.kr_market_specialist import run_kr_market_specialist
    from src.agents.us_market_specialist import run_us_market_specialist
    from src.agents.quant_analyst        import run_quant_analyst
    from src.agents.technical_analyst    import run_technical_analyst
    from src.agents.sentiment_analyst    import run_sentiment_analyst
    from src.agents.fundamental_analyst  import run_fundamental_analyst
    from src.agents.quant_rule_agent     import run_quant_rule_agent   # G4: LLM 없는 규칙 투표자

    agent_names = [
        "macro_economist", "kr_market_specialist", "us_market_specialist",
        "quant_analyst", "technical_analyst", "sentiment_analyst", "fundamental_analyst",
        "quant_rule_agent",
    ]

    semaphore     = asyncio.Semaphore(3)
    krx_semaphore = asyncio.Semaphore(2)

    DEFAULT_AGENT_TIMEOUT = 120
    AGENT_TIMEOUTS = {
        # 외부 호출/크롤이 많은 에이전트는 여유를 더 줌
        "sentiment_analyst":    180,
        "fundamental_analyst":  180,
    }

    def _timeout_for(name: str) -> int:
        return AGENT_TIMEOUTS.get(name, DEFAULT_AGENT_TIMEOUT)

    async def run_with_semaphore(name, coro):
        t = _timeout_for(name)
        async with semaphore:
            try:
                return await asyncio.wait_for(coro, timeout=t)
            except asyncio.TimeoutError:
                logger.warning(f"[AgentTimeout] {name} — {t}초 초과")
                print(f"  ⏱️ {name} — 개별 타임아웃 ({t}초)")
                return None

    async def run_krx_agent(name, coro):
        t = _timeout_for(name)
        async with krx_semaphore:
            async with semaphore:
                try:
                    return await asyncio.wait_for(coro, timeout=t)
                except asyncio.TimeoutError:
                    logger.warning(f"[AgentTimeout] {name} — {t}초 초과")
                    print(f"  ⏱️ {name} — 개별 타임아웃 ({t}초)")
                    return None

    raw_results = await asyncio.gather(
        run_with_semaphore("macro_economist", run_macro_economist()),
        run_krx_agent("kr_market_specialist", run_kr_market_specialist(target_ticker=ticker)),
        run_with_semaphore("us_market_specialist", run_us_market_specialist()),
        run_krx_agent("quant_analyst", run_quant_analyst(target_ticker=ticker)),
        run_krx_agent("technical_analyst", run_technical_analyst(target_ticker=ticker)),
        # 종목 이벤트 해석자 — 오늘 이 종목이 뽑힌 사유(공시 제목·뉴스 급증)를 받는다 (v3)
        run_with_semaphore("sentiment_analyst",
                           run_sentiment_analyst(target_ticker=ticker,
                                                 screen_context=state.screen_context)),
        run_krx_agent("fundamental_analyst", run_fundamental_analyst(target_ticker=ticker)),
        run_with_semaphore("quant_rule_agent", run_quant_rule_agent(target_ticker=ticker)),
        return_exceptions=True,
    )

    reports = []
    errors  = []

    for name, result in zip(agent_names, raw_results):
        if result is None:
            t = _timeout_for(name)
            errors.append(f"{name}: timeout ({t}s)")
            reports.append(AnalysisReport(
                agent_name=name,
                confidence=0.0,
                recommendation="HOLD",
                reasoning=[
                    f"에이전트 개별 타임아웃: {t}초 초과",
                    "MCP 데이터 수집 또는 LLM 호출 중 시간 초과",
                    "이 보고서는 신뢰할 수 없으므로 chief_strategist 종합 시 제외 권장",
                ],
                data_sources=["timeout_fallback", "pipeline_fallback"],
                selection_rationale=None,
                prediction_basis=["타임아웃으로 인한 대체값", "pipeline_fallback"],
                risk_factors=["에이전트 타임아웃으로 분석 불가"],
            ))
        elif isinstance(result, Exception):
            print(f"  ❌ {name} 실패: {result}")
            errors.append(f"{name}: {str(result)}")
            reports.append(AnalysisReport(
                agent_name=name,
                confidence=0.0,
                recommendation="HOLD",
                reasoning=[
                    f"에이전트 실행 실패: {str(result)}",
                    "MCP 데이터 수집 또는 LLM 호출 중 오류 발생",
                    "이 보고서는 신뢰할 수 없으므로 chief_strategist 종합 시 제외 권장",
                ],
                data_sources=["error_fallback", "pipeline_fallback"],
                selection_rationale=None,
                prediction_basis=["오류로 인한 대체값", "pipeline_fallback"],
                risk_factors=["에이전트 오류로 분석 불가"],
            ))
        else:
            print(f"  ✅ {name} — {result.recommendation} (신뢰도 {result.confidence})")
            reports.append(result)

    # ── 종목 정보 주입 ────────────────────────────────────────────────
    # LLM은 ticker를 채우지 않는다(스키마 default=None). 파이프라인이 대상 종목을
    # 확정하므로 여기서 넣는다. 이게 없으면 prediction_logger가 빈 ticker를 저장하고
    # feedback_evaluator가 전부 건너뛰어 D+1 채점이 0건이 된다.
    # 폴백 리포트(타임아웃/예외)에도 넣어야 원장에 종목이 남는다.
    ticker_name = ((state.market_data.get("stock") or {}).get("name")) or ""
    for r in reports:
        r.ticker      = ticker
        r.ticker_name = ticker_name

    completed = sum(1 for r in reports if r.confidence > 0.0)
    print(f"  → {completed}/{len(agent_names)}개 에이전트 정상 완료")

    return {"analysis_reports": reports, "error_log": errors}


# ──────────────────────────────────────────
# 노드 2.5: Quality Gate
# ──────────────────────────────────────────

from src.graph.quality_gate import quality_gate_node


# ──────────────────────────────────────────
# 노드 3: Bull vs Bear 토론
# ──────────────────────────────────────────

from src.agents.debate import debate_node as _debate_node_original

async def debate_node(state: GraphState) -> dict:
    reports_for_debate = state.qualified_reports if state.qualified_reports else state.analysis_reports
    modified_state     = state.model_copy(update={"analysis_reports": reports_for_debate})
    return await _debate_node_original(modified_state)


# ──────────────────────────────────────────
# 노드 4: Chief Strategist
# ──────────────────────────────────────────

def _format_agent_weights_for_prompt(regime: str) -> str:
    try:
        weights = get_agent_weights(regime)
    except Exception as e:
        logger.warning(f"[pipeline] agent_weights 조회 실패: {e}")
        return "[에이전트 가중치 조회 실패 — 균등 취급]"

    if not weights:
        return "[에이전트 가중치 없음 — 균등 취급]"

    avg_weight     = sum(weights.values()) / len(weights)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    lines = [f"[에이전트 신뢰도 — {regime} 레짐 기준]"]
    for agent_name, w in sorted_weights:
        trend = "↑ 높음" if w > avg_weight * 1.2 else "↓ 낮음" if w < avg_weight * 0.8 else "→ 보통"
        lines.append(f"  - {agent_name:<32} {w:.4f}  {trend}")
    lines.append("")
    lines.append("✅ 신뢰도 높은 에이전트의 의견에 더 큰 비중을 두세요.")
    lines.append("⚠️ 단, 모든 에이전트가 동일 방향을 가리킬 때만 강한 포지션을 취하세요.")

    return "\n".join(lines)


async def chief_strategist_node(state: GraphState) -> dict:
    reports    = state.qualified_reports if state.qualified_reports else state.analysis_reports
    has_debate = bool(state.debate_summary)
    regime     = state.current_regime

    print(f"\n[4/7] chief_strategist — {len(reports)}개 보고서 종합")
    print(f"       레짐: {regime.upper()}")
    print(f"       토론: {'포함' if has_debate else '생략 (한쪽 우세)'}")

    weight_context = _format_agent_weights_for_prompt(regime)

    from src.data.prediction_logger import WARMUP_SAMPLE_COUNT, get_weight_summary, _normalize_regime
    normalized_regime = _normalize_regime(regime)
    summary = get_weight_summary()
    warmup_agents = [
        r["agent_name"] for r in summary
        if r["regime"] == normalized_regime and r["sample_count"] < WARMUP_SAMPLE_COUNT
    ]
    if warmup_agents:
        print(f"       ⏳ 워밍업 중 ({len(warmup_agents)}개 에이전트)")
    else:
        print(f"       ✅ EMA 가중치 적용 중")

    from src.agents.chief_strategist import run_chief_strategist

    current_price = float(
        (state.market_data.get("stock") or {}).get("latest_close") or 0
    )

    # 2026-09-18: 과거 판단 교훈 주입 — chief에게만. 실패해도 chief는 그대로 돈다.
    past_context = ""
    try:
        from datetime import date as _date
        from src.data.decision_memory import get_past_context
        past_context = get_past_context(state.ticker, as_of=_date.today().isoformat())
        if past_context:
            print("       📝 과거 교훈 주입")
    except Exception as e:
        logger.warning(f"[chief_strategist] 과거 교훈 조회 실패 (무시): {e}")

    final: AnalysisReport = await run_chief_strategist(
        reports        = reports,
        regime         = regime,
        debate_summary = state.debate_summary,
        weight_context = weight_context,
        current_price  = current_price,
        past_context   = past_context,
        screen_context = state.screen_context,
        ticker         = state.ticker,      # ATR 손절 계산 (src/utils/atr.py)
    )
    final.ticker      = state.ticker
    final.ticker_name = ((state.market_data.get("stock") or {}).get("name")) or ""
    print(f"  → 최종: {final.recommendation} (신뢰도 {final.confidence})")

    from src.graph.signal_reconciliation import run_signal_reconciliation, MAX_BUY_SIGNALS

    print(f"\n[4.5/7] signal_reconciliation")
    recon = run_signal_reconciliation(
        ticker         = state.ticker,
        recommendation = final.recommendation,
        confidence     = final.confidence,
    )

    current = recon["current"]
    if current and current.get("included"):
        print(f"  ✅ {state.ticker} → 매수 후보 {current['rank']}위")
    elif final.recommendation == "BUY":
        reason = current.get("exclusion_reason") if current else "미분류"
        print(f"  ⚠️ {state.ticker} BUY → 미선정 ({reason})")
    else:
        print(f"  → {state.ticker} {final.recommendation}")

    print(f"  → 오늘 매수 후보: {recon['buy_count']}/{MAX_BUY_SIGNALS}종목")

    # ── BUY 추천 시 포지션 자동 기록 ─────────────────────────────────────
    if final.recommendation == "BUY" and final.entry_price is not None:
        try:
            from src.data.position_tracker import add_position
            # stop_loss / target_price는 update_current_prices()가 손절·목표 도달을
            # 판정할 때 읽는 값이다. chief가 계산해 리포트에 표시한 값을 그대로 넘겨야
            # "리포트가 보여준 손절선"과 "시스템이 감시하는 손절선"이 일치한다.
            # 넘기지 않으면 add_position이 비율 기본값(-4%/+8%, CR-6)으로 대체해버린다.
            add_position(
                ticker               = state.ticker,
                entry_price          = float(final.entry_price),
                quantity             = 0,  # 시드 % 기반 추적
                stop_loss            = final.stop_loss,
                target_price         = final.take_profit_1,
                allocation_pct       = final.position_size_pct,
                take_profit_1        = final.take_profit_1,
                take_profit_2        = final.take_profit_2,
                holding_period_weeks = final.holding_period_weeks,
                rr_ratio             = final.rr_ratio,
                entry_rationale      = final.reasoning[0] if final.reasoning else "",
            )
            pos_pct = final.position_size_pct if final.position_size_pct is not None else 0
            print(f"  ✅ 포지션 기록: {state.ticker} 시드의 {pos_pct:.0f}%")
            if final.stop_loss is None:
                print(f"  ⚠️ chief가 stop_loss를 채우지 않음 → 기본값 대체됨 (리포트와 불일치 가능)")
        except Exception as e:
            logger.warning(f"[chief_strategist] 포지션 자동 기록 실패: {e}")

    return {
        "analysis_reports":   [final],
        "final_strategy":     final.recommendation,
        "reconciled_signals": recon["all_signals"],
    }


# ──────────────────────────────────────────
# 노드 5: 리포트 포맷팅 (v4.0 — src/graph/report_formatter.py)
# ──────────────────────────────────────────

from src.graph.report_formatter import report_formatter_node as report_formatter


# ──────────────────────────────────────────
# 노드 6: Notion 발행 (v2.8: 발행 전 검증 추가)
# ──────────────────────────────────────────

async def notion_publish(state: GraphState) -> dict:
    print(f"\n[6/7] notion_publish")

    # ── 발행 전 리포트 검증 ───────────────────────────────────────────────
    # 빈 리포트, None 포함, 필수 섹션 누락 시 발행 차단
    try:
        validate_report(state.report_content)
    except InvalidReportError as e:
        print(f"  ❌ 리포트 검증 실패 — 발행 차단: {e}")
        return {"error_log": list(state.error_log or []) + [f"리포트 검증 실패: {e}"]}

    from src.graph.notion_publisher import publish_to_notion

    chief_report = next(
        (r for r in state.analysis_reports if r.agent_name == "chief_strategist"), None
    )
    all_reports  = list(state.analysis_reports)

    # 포트폴리오 요약 로드 (실패해도 발행은 계속)
    portfolio_summary = None
    try:
        from src.data.position_tracker import get_portfolio_pct_summary
        portfolio_summary = get_portfolio_pct_summary()
    except Exception as e:
        logger.warning(f"[notion_publish] 포트폴리오 로드 실패: {e}")

    result = await publish_to_notion(
        report_content    = state.report_content,
        ticker            = state.ticker,
        regime            = state.current_regime,
        strategy          = state.final_strategy,
        chief_report      = chief_report,
        qualified_reports = list(state.qualified_reports or []),
        all_reports       = all_reports,
        debate_summary    = state.debate_summary or "",
        error_log         = list(state.error_log or []),
        portfolio_summary = portfolio_summary,
        market_data       = state.market_data or {},
    )

    if result["success"]:
        print(f"  ✅ 발행 완료: {result['url']}")
    else:
        print(f"  ⚠️ 발행 실패 — 로컬 저장됨: {result.get('fallback_path')}")

    print("─" * 60)
    print(state.report_content[:500], "..." if len(state.report_content) > 500 else "")
    print("─" * 60)
    return {}


# ──────────────────────────────────────────
# 노드 7: 예측 저장
# ──────────────────────────────────────────

async def log_predictions_node(state: GraphState) -> dict:
    print(f"\n[7/7] log_predictions — 오늘의 예측 저장")

    try:
        qualified_reports = list(state.qualified_reports or state.analysis_reports)
        regime            = state.current_regime or "Neutral"

        # chief 최종 판단도 원장에 남긴다 (doc/2026-09-21_agent-audit.md §2-6).
        # qualified_reports는 quality_gate가 chief 실행 **전에** 만든 목록이라 chief가 없다.
        # 이 행이 없으면 "시스템의 BUY를 따르면 버는가"를 잴 표본이 영원히 0이다.
        # confidence 0.0(needs_review·전원 폴백)은 판단이 아니라 '판단 없음'이라 제외한다
        # — decision_memory 기록 기준과 같다.
        chief_report = next(
            (r for r in state.analysis_reports if r.agent_name == "chief_strategist"), None
        )
        if chief_report is not None and chief_report.confidence > 0:
            if not any(r.agent_name == "chief_strategist" for r in qualified_reports):
                qualified_reports.append(chief_report)

        if not qualified_reports:
            print("  ⚠️ 저장할 qualified_reports 없음 — 건너뜀")
            return {}

        # 예측 시점 종가를 함께 저장 — 채점 때 되짚어 조회하지 않도록.
        # (KRX 장애 시 되짚기가 실패하면 채점 자체가 불가능해진다.)
        price_at_pred = (state.market_data.get("stock") or {}).get("latest_close")

        saved_count = log_agent_predictions(
            qualified_reports = qualified_reports,
            regime            = regime,
            price_at_pred     = float(price_at_pred) if price_at_pred else None,
        )

        from src.data.prediction_logger import record_regime, _normalize_regime
        record_regime(regime)

        # 2026-09-18: chief 판단을 decision_memory에 기록 (내일 feedback.yml이 복기한다).
        # confidence 0.0(needs_review·전원 폴백)은 "의견 없음"이라 기록하지 않는다.
        chief = next((r for r in state.analysis_reports if r.agent_name == "chief_strategist"), None)
        if chief is not None and chief.confidence > 0:
            try:
                from datetime import date as _date
                from src.data.decision_memory import store_decision
                store_decision(
                    pred_date      = _date.today().isoformat(),
                    ticker         = state.ticker,
                    ticker_name    = chief.ticker_name or "",
                    recommendation = chief.recommendation,
                    confidence     = chief.confidence,
                    regime         = _normalize_regime(regime),
                    reasoning      = list(chief.reasoning),
                    price_at_pred  = float(price_at_pred) if price_at_pred else None,
                )
                print("  ✅ chief 판단 기록 (decision_memory)")
            except Exception as e:
                logger.warning(f"[pipeline] chief 판단 기록 실패 (무시): {e}")

        print(f"  ✅ {saved_count}개 예측 저장 (레짐={_normalize_regime(regime)})")

    except Exception as e:
        logger.error(f"[pipeline] log_predictions_node 실패: {e}", exc_info=True)
        return {"error_log": list(state.error_log or []) + [f"prediction_logger 실패: {e}"]}

    return {}


# ──────────────────────────────────────────
# 그래프 조립
# ──────────────────────────────────────────

def build_pipeline(publish: bool = True) -> StateGraph:
    """파이프라인 그래프 빌드.

    publish=False 일 때 notion_publish 노드를 제외한다.
    daily_runner가 종목별 결과를 모아 1회만 발행할 때 사용.
    """
    graph = StateGraph(GraphState)

    graph.add_node("data_ingest",       node_with_timeout(data_ingest,            30,  "data_ingest"))
    graph.add_node("regime_detector",   node_with_timeout(regime_detector_node,   30,  "regime_detector"))
    graph.add_node("parallel_analysis", node_with_timeout(parallel_analysis,     360,  "parallel_analysis"))
    graph.add_node("quality_gate",      node_with_timeout(quality_gate_node,      30,  "quality_gate"))
    graph.add_node("debate",            node_with_timeout(debate_node,            90,  "debate"))
    graph.add_node("chief_strategist",  node_with_timeout(chief_strategist_node, 120,  "chief_strategist"))
    graph.add_node("report_formatter",  node_with_timeout(report_formatter,       30,  "report_formatter"))
    graph.add_node("log_predictions",   node_with_timeout(log_predictions_node,   30,  "log_predictions"))

    if publish:
        graph.add_node("notion_publish", node_with_timeout(notion_publish, 30, "notion_publish"))

    graph.set_entry_point("data_ingest")
    graph.add_edge("data_ingest",       "regime_detector")
    graph.add_edge("regime_detector",   "parallel_analysis")
    graph.add_edge("parallel_analysis", "quality_gate")
    graph.add_edge("quality_gate",      "debate")
    graph.add_edge("debate",            "chief_strategist")
    graph.add_edge("chief_strategist",  "report_formatter")

    if publish:
        graph.add_edge("report_formatter", "notion_publish")
        graph.add_edge("notion_publish",   "log_predictions")
    else:
        graph.add_edge("report_formatter", "log_predictions")

    graph.add_edge("log_predictions", END)

    return graph.compile()


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

async def run_pipeline(ticker: str = "005930", publish: bool = True,
                       screen_context: dict | None = None):
    """파이프라인 실행.

    publish=False면 notion_publish 노드를 건너뛰고 최종 state만 반환한다.
    daily_runner가 종목별로 호출해 결과를 모은 뒤 통합 리포트 1회 발행할 때 사용.

    screen_context: 오늘 이 종목이 뽑힌 사유 (src/screening/screen_context.py).
        None이면 빈 dict — 단독 디버그 실행 경로다.
    """
    # ── ticker 검증 (v2.8) ────────────────────────────────────────────────
    try:
        ticker = validate_ticker(ticker)
    except InvalidTickerError as e:
        print(f"❌ 유효하지 않은 ticker: {e}")
        return {}

    print("=" * 60)
    print(f"AI 투자 리포트 파이프라인 v2.8 — {ticker} (publish={publish})")
    print("=" * 60)

    pipeline = build_pipeline(publish=publish)

    initial_state = {
        "ticker":             ticker,
        "screen_context":     dict(screen_context or {}),
        "market_data":        {},
        "analysis_reports":   [],
        "qualified_reports":  [],
        "reconciled_signals": [],
        "final_strategy":     "",
        "report_content":     "",
        "current_regime":     "unknown",
        "debate_summary":     "",
        "error_log":          [],
    }

    result = await pipeline.ainvoke(initial_state)
    print("\n✅ 파이프라인 완료")
    return result


if __name__ == "__main__":
    asyncio.run(run_pipeline("005930"))