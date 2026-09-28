"""
src/daily_runner.py

매일 자동 실행되는 진입점.

[흐름]
  1. screener.run_screening() → confirmed/optional ticker 산출
  2. confirmed가 비어 있으면 optional까지 포함
  3. 선정 종목이 있으면: 각 ticker마다 run_pipeline(ticker) sequential await
  4. 선정 종목이 0개이면: 시장 개관 리포트(market-only)를 생성해 Notion 발행
  5. 종목별 성공/실패/소요시간 기록 후 총합 출력

pipeline.py, screener.py 등 기존 파일은 건드리지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv

from src.screening.screener import run_screening
from src.screening.screen_context import build_screen_context, selection_reason
from src.graph.pipeline import run_pipeline
from src.schemas.agent_output import AnalysisReport
from src.utils.llm_budget import get_budget, BudgetExceeded

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# 확정 0개일 때 optional에서 끌어올릴 상위 N개.
# Stage 2에서 점수 내림차순으로 정렬되어 들어오므로 상위 N개가 의미가 있다.
# 너무 크면 신호 약한 종목까지 전부 심층 분석되어 시간이 폭주한다.
# [삭제됨 — G5] MAX_OPTIONAL_FALLBACK = 2
#   확정 0개일 때 optional(점수 1)을 2개까지 분석하던 폴백. 산출량 보장 장치라 제거.
#   확정 0개 = 오늘 신호 없음 → 시장 전용 리포트.

# confirmed 분기의 종목 수 상한.
#
# [왜 필요한가]
#   이전에는 confirmed에 상한이 없어 스크리닝이 뽑는 만큼 전부 심층 분석했다.
#   2026-09-09 실행(#99) 실측: 확정 26개 → 26종목 분석 → 36분 소요.
#   종목당 LLM 호출은 에이전트 7 + 토론(조건부) + chief 1 이므로
#   26종목이면 하루 200~280콜이다. 유일한 제동장치가 CI의
#   timeout-minutes(120분)라 비용이 아니라 벽시계로만 막혀 있었다.
#
# [왜 8인가]
#   signal_reconciliation.MAX_BUY_SIGNALS = 5 — 하루에 채택되는 매수 신호는
#   최대 5종목이다. 그보다 많이 분석해도 뒤에서 버려진다.
#   BUY가 아닌 판정(HOLD/SELL)도 나오므로 여유를 둬 8로 잡는다.
#   Stage 2가 점수 내림차순으로 정렬해 주므로 상위 8개가 가장 신호가 강한 종목이다.
MAX_CONFIRMED_TICKERS = 8

# OHLCV 캐시가 이 일수 이상 갱신되지 않으면 정량 신호를 신뢰할 수 없다고 본다.
# (KRX 로그인 실패 시 _bulk_snapshot이 조용히 None을 반환하며 캐시가 멈춘다.
#  2026-07-24 ~ 09-07 46일간 이 상태가 감지되지 않은 전례가 있어 임계를 짧게 잡았다.)
MAX_OHLCV_STALE_DAYS = 5


def _ohlcv_staleness_days() -> Optional[int]:
    """ohlcv_cache 최신 날짜가 오늘로부터 며칠 뒤처졌는지. 조회 실패 시 None."""
    try:
        from src.data.ohlcv_cache import get_latest_cached_date
        latest = get_latest_cached_date()
        if not latest:
            return None
        return (datetime.now(KST).date() - datetime.strptime(latest, "%Y-%m-%d").date()).days
    except Exception as e:
        logger.warning(f"[DailyRunner] OHLCV 신선도 확인 실패: {e}")
        return None


def _market_session_warning() -> Optional[str]:
    """
    장중에 실행됐으면 경고 문구를 반환한다. 장 전/후면 None.

    [왜 필요한가]
      cron은 04:07 KST를 노리지만 GitHub 무료 티어는 SLA가 없어 밀린다.
      최근 10회 실측 지연은 102~491분. 최악의 경우 장중에 리포트가 나간다
      (2026-08-28 실행이 14:21 KST에 발행됐다).

      장중에 돌면 get_stock_price가 확정 종가가 아니라 **체결 중인 현재가**를
      돌려준다. 진입가·손절가가 그 가격으로 계산되는데, 리포트를 읽는 사람은
      장 시작 전 분석으로 오해하기 쉽다.

      스케줄만으로는 못 막으므로 실행 시각을 근거로 리포트가 스스로 밝히게 한다.
      실패로 처리하지는 않는다 — 리포트 자체는 유효하다.
    """
    now = datetime.now(KST)
    if now.weekday() >= 5:                     # 주말은 장이 없다
        return None
    minutes = now.hour * 60 + now.minute
    if 9 * 60 <= minutes <= 15 * 60 + 30:      # 09:00 ~ 15:30 KST
        return (
            f"장중 실행 ({now:%H:%M} KST) — 가격은 확정 종가가 아니라 "
            f"체결 중인 현재가다. 진입가·손절가가 장 마감까지 달라질 수 있다."
        )
    return None


# 건강도 판정이 오류 문자열에서 찾는 패턴 (2026-09-11 운영 복구)
CREDIT_ERROR_MARKERS = ("credit balance is too low", "insufficient_quota", "Incorrect API key")
MCP_KEY_MISSING_MARKER = "설정되지 않았습니다"

# DART(전자공시) 접속 실패 마커 (CR-3, 2026-09-14).
#
# [왜 실패가 아니라 경고인가 — 2026-09-14 재조정]
#   조사 결론은 "간헐적이고 운영 영향 없음"이었다. 키 문제도, 러너 IP 영구 차단도,
#   서비스 장애도 아니다(같은 날 18:05 실패 → 21:29·21:55 정상). 관측된 스케줄 실행
#   4건은 모두 정상이었고 실패는 수동 dispatch 2건뿐이다.
#
#   처음에는 reasons에 넣어 빨간불로 만들었는데 그러면 **정상 실행이 가끔 빨간불이 되고,
#   결국 빨간불을 무시하게 된다**. 경고가 소음이 되면 진짜 실패를 놓친다.
#   → warnings로 낮춘다. 리포트 건강도 줄에는 표시되고 종료 코드는 0이다.
#
#   영향: DART가 죽으면 fundamental_analyst의 공시 근거가 사라진다(8명 중 1명 약화).
#   승격 판단용으로 건수를 dart_fail_count에 남긴다. 스케줄 실행에서 반복되면
#   그때 reasons로 올린다.
DART_ERROR_MARKER = "opendart.fss.or.kr"


def _assess_run_health(per_ticker_results: list[dict], selected: list[str],
                       screen: dict | None = None) -> dict:
    """
    이번 실행이 실제로 분석을 수행했는지 판정한다.

    [왜 필요한가]
      에이전트가 전부 실패해도 폴백 리포트(confidence=0.0)가 생성되고
      Quality Gate 소프트 폴백이 그것을 통과시키므로, 파이프라인은 끝까지 돌고
      GitHub Actions는 success로 표시된다.
      실제로 2026-09-07 실행에서 OpenAI 401로 7개 에이전트가 전부 죽었는데도
      success로 기록되어 아무도 알아채지 못했다.

      여기서 "살아있는 에이전트 리포트 수"를 세어 0이면 degraded로 판정하고,
      호출자(__main__)가 프로세스를 비정상 종료시켜 실행이 빨간불이 되게 한다.

    Returns:
      {"healthy": bool, "live_agent_reports": int, "total_agent_reports": int,
       "ohlcv_stale_days": int|None, "reasons": [str, ...]}
    """
    live = total = 0
    for payload in per_ticker_results:
        for r in payload.get("all_reports") or []:
            if getattr(r, "agent_name", "") == "chief_strategist":
                continue
            total += 1
            if getattr(r, "confidence", 0.0) > 0.0:
                live += 1

    stale = _ohlcv_staleness_days()
    reasons: list[str] = []

    if selected and total > 0 and live == 0:
        reasons.append(
            f"전문가 에이전트 {total}개가 전부 실패 (confidence>0 인 리포트 0개) "
            f"— LLM 자격증명/네트워크 확인 필요"
        )
    if selected and not per_ticker_results:
        reasons.append(f"종목 {len(selected)}개를 선정했으나 분석 결과가 0건")
    if stale is not None and stale > MAX_OHLCV_STALE_DAYS:
        reasons.append(
            f"OHLCV 캐시가 {stale}일 뒤처짐 (임계 {MAX_OHLCV_STALE_DAYS}일) "
            f"— KRX 로그인 실패 시 조용히 멈춘다. 정량 신호를 신뢰할 수 없음"
        )

    # ── 2026-09-11 추가: 크레딧 400 / 헤드라인 0건 / MCP 키 누락 ──────────
    # 셋 다 "리포트는 나가지만 내용이 비어 있는" 성공 위장이다.
    all_errors: list[str] = []
    for payload in per_ticker_results:
        all_errors.extend(str(e) for e in (payload.get("error_log") or []))
    credit_hits = [e for e in all_errors if any(m in e for m in CREDIT_ERROR_MARKERS)]
    if credit_hits:
        reasons.append(
            f"LLM 자격증명/크레딧 오류 {len(credit_hits)}건 — 예: {credit_hits[0][:120]}"
        )

    if screen is not None and selected and screen.get("headline_count") == 0:
        reasons.append("Stage 1-C 헤드라인 0건 — 뉴스 수집 실패(페이지 구조 변경/차단). news_match 전부 False")

    try:
        from src.utils.mcp_result import recent_errors
        _mcp_errors = recent_errors()
        key_missing = sorted({e.split(":")[0] for e in _mcp_errors if MCP_KEY_MISSING_MARKER in e})
        dart_fails  = [e for e in _mcp_errors if DART_ERROR_MARKER in e]
    except Exception:
        key_missing, dart_fails = [], []

    if key_missing:
        reasons.append(
            f"MCP 서버가 API 키를 못 읽음 ({', '.join(key_missing[:6])}) — "
            f"서브프로세스 env 전달(src/utils/mcp_client.py) 또는 시크릿 확인"
        )

    # 경고는 reasons와 분리한다. reasons는 종료 코드를 바꾸지만
    # warnings는 "리포트는 유효하되 읽을 때 감안할 것"이라서 실패가 아니다.
    warnings: list[str] = []
    session = _market_session_warning()
    if session:
        warnings.append(session)

    # CR-3: 실패가 아니라 경고. 종료 코드는 바뀌지 않는다.
    if dart_fails:
        warnings.append(
            f"DART 접속 실패 {len(dart_fails)}건 — fundamental_analyst의 공시 근거가 비었다. "
            f"간헐적 장애로 조사됨(CR-3). 스케줄 실행에서 반복되면 실패로 승격할 것"
        )

    return {
        "healthy":             not reasons,
        "live_agent_reports":  live,
        "total_agent_reports": total,
        "ohlcv_stale_days":    stale,
        "reasons":             reasons,
        "warnings":            warnings,
        # 승격 판단용 카운트. 스케줄 실행 로그에서 이 값이 계속 0보다 크면 재검토한다.
        "dart_fail_count":     len(dart_fails),
    }


def _health_note(health: dict) -> str:
    """리포트 최상단에 넣을 건강도 한 줄.

    [왜 필요한가]
      건강도 판정은 지금까지 콘솔에만 나갔다(종료 코드로만 드러남). 리포트만 보는
      사람은 "에이전트가 전부 죽은 날의 리포트"와 정상 리포트를 구분할 수 없었다.
      한 줄이라도 리포트 안에 있어야 읽는 사람이 그날 수치를 믿을지 정한다.
    """
    live, total = health.get("live_agent_reports", 0), health.get("total_agent_reports", 0)
    if health.get("healthy", True):
        base = f"✅ 실행 정상 — 에이전트 {live}/{total} 응답" if total else "✅ 실행 정상 — 오늘 확정 종목 없음"
    else:
        base = "⛔ 이 리포트는 정상 분석으로 볼 수 없다 — " + " / ".join(health.get("reasons") or ["사유 불명"])
    stale = health.get("ohlcv_stale_days")
    if stale:
        base += f" | OHLCV {stale}일 지연"
    for w in health.get("warnings") or []:
        base += f" | ⚠️ {w}"
    return base


def _aggregate_strategy(per_ticker_results: list[dict]) -> str:
    """종목별 chief 판단을 모아 통합 전략 라벨 산출.

    우선순위: BUY > SELL > HOLD. 모두 분석 실패면 'NO_PICK'.
    """
    recs = [
        r["chief_report"].recommendation
        for r in per_ticker_results
        if r.get("chief_report") is not None
    ]
    if not recs:
        return "NO_PICK"
    if "BUY"  in recs: return "BUY"
    if "SELL" in recs: return "SELL"
    return "HOLD"


def _extract_pipeline_result(state, ticker: str) -> dict:
    """run_pipeline(publish=False) 반환 state에서 통합 발행에 필요한 페이로드 추출.

    state는 LangGraph가 반환한 dict (Pydantic GraphState의 직렬화).
    """
    if not state:
        return {
            "ticker":            ticker,
            "chief_report":      None,
            "qualified_reports": [],
            "all_reports":       [],
            "market_data":       {},
            "current_regime":    "unknown",
            "final_strategy":    "",
        }

    # state는 dict 또는 Pydantic 모델일 수 있음 — 양쪽 모두 지원
    def _g(key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    all_reports: list[AnalysisReport] = list(_g("analysis_reports") or [])
    chief = next((r for r in all_reports if r.agent_name == "chief_strategist"), None)

    return {
        "ticker":            ticker,
        "chief_report":      chief,
        "qualified_reports": list(_g("qualified_reports") or []),
        "all_reports":       all_reports,
        "market_data":       _g("market_data") or {},
        "current_regime":    _g("current_regime") or "unknown",
        "final_strategy":    _g("final_strategy") or "",
        "error_log":         list(_g("error_log") or []),     # 건강도 판정용 (크레딧 400 등)
    }


# ─────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """37.4 → '37초', 152.3 → '2분 32초'."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}초"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}분 {s}초"
    h, m = divmod(m, 60)
    return f"{h}시간 {m}분 {s}초"


# ─────────────────────────────────────────────────────────
# 종목 선정
# ─────────────────────────────────────────────────────────

def _select_tickers(screen_result: dict) -> tuple[list[str], str]:
    """
    confirmed만 분석 대상이다. 비어 있으면 [] — optional로 메우지 않는다 (G5).
    Returns: (선정 ticker 리스트, 사용된 모드 라벨)
    """
    confirmed = list(screen_result.get("confirmed") or [])

    if confirmed:
        # v2: 레짐 상한(bear·volatile 5)과 코드 상한(8) 중 작은 쪽
        cap = min(MAX_CONFIRMED_TICKERS, int(screen_result.get("cap") or MAX_CONFIRMED_TICKERS))
        picked = confirmed[:cap]
        if len(confirmed) > len(picked):
            # 잘라냈다는 사실을 남긴다. 조용히 자르면 "왜 26개 중 8개만 봤나"를
            # 나중에 추적할 수 없다.
            logger.info(
                f"[DailyRunner] 확정 {len(confirmed)}개 중 상위 {len(picked)}개만 분석 "
                f"(상한 cap={cap}). "
                f"제외: {', '.join(confirmed[len(picked):])}"
            )
            return picked, f"confirmed 상위 {len(picked)}/{len(confirmed)}개 (상한 적용)"
        return picked, "confirmed"

    return [], "선정 종목 없음 (확정 0개 — 신호 없음)"


# ─────────────────────────────────────────────────────────
# 시장 전용 리포트 (선정 종목 0개일 때 호출)
# ─────────────────────────────────────────────────────────

async def _collect_market_snapshot() -> dict:
    """KOSPI / VIX / 미10Y / USD-KRW / WTI를 병렬 수집해 dict로 반환.
    개별 항목 실패는 None으로 격리.
    """
    try:
        from src.mcp_servers.us_market.server    import (
            get_vix, get_treasury_yields, get_commodity_prices,
        )
        from src.mcp_servers.news_economy.server import get_exchange_rate
        import yfinance as yf
    except Exception as e:
        logger.warning(f"[market-only] 시장 데이터 모듈 import 실패: {e}")
        return {}

    def _kospi() -> Optional[float]:
        try:
            hist = yf.Ticker("^KS11").history(period="5d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"[market-only] KOSPI 조회 실패: {e}")
        return None

    # 각 외부 호출은 20초 timeout. wait_for의 TimeoutError는
    # gather(return_exceptions=True)에 의해 캡처되어 _safe 폴백으로 흡수된다.
    vix_v, ex_v, kospi_v, treasury_v, commodity_v = await asyncio.gather(
        asyncio.wait_for(asyncio.to_thread(get_vix),                       timeout=20),
        asyncio.wait_for(asyncio.to_thread(get_exchange_rate, 5),          timeout=20),
        asyncio.wait_for(asyncio.to_thread(_kospi),                        timeout=20),
        asyncio.wait_for(asyncio.to_thread(get_treasury_yields),           timeout=20),
        asyncio.wait_for(asyncio.to_thread(get_commodity_prices, ["WTI"]), timeout=20),
        return_exceptions=True,
    )

    def _safe(v, default=None):
        return default if isinstance(v, Exception) else v

    vix_v       = _safe(vix_v, {})
    ex_v        = _safe(ex_v, {})
    kospi_v     = _safe(kospi_v)
    treasury_v  = _safe(treasury_v, {}) or {}
    commodity_v = _safe(commodity_v, {}) or {}

    return {
        "kospi":        kospi_v,
        "vix":          vix_v,
        "exchange_rate": ex_v,
        "us_10y_yield": (treasury_v.get("10y") or {}).get("close") if isinstance(treasury_v, dict) else None,
        "usd_krw":      ex_v.get("latest_rate") if isinstance(ex_v, dict) else None,
        "wti":          (commodity_v.get("WTI") or {}).get("price") if isinstance(commodity_v, dict) else None,
        "source":       "live",
    }


async def _detect_regime_safe() -> str:
    """regime_detector 모듈을 사용해 현재 레짐 판정. 실패 시 'unknown'."""
    try:
        from src.graph.regime_detector import _fetch_market_indicators, classify_regime
        indicators = await asyncio.to_thread(_fetch_market_indicators)
        regime, _reason = classify_regime(indicators)
        return regime or "unknown"
    except Exception as e:
        logger.warning(f"[market-only] regime_detector 실패: {e}")
        return "unknown"


def _format_market_only_report(
    regime:      str,
    market:      dict,
    screen:      dict,
    portfolio:   Optional[dict],
    health_note: Optional[str] = None,
) -> str:
    """선정 종목 0개일 때 발행할 마크다운 리포트 본문."""
    from src.graph.report_formatter import (
        REGIME_KR, WEEKDAY_KR,
        _fmt_money, _fmt_pct, _scalar,
    )

    now     = datetime.now(KST)
    weekday = WEEKDAY_KR[now.weekday()]
    regime_label = REGIME_KR.get(regime.upper(), regime)

    title = f"# Daily stock market report_{now.strftime('%Y%m%d')}_{now.strftime('%H:%M')}_NO_PICK"
    lines = [
        title, "",
        "────────────────────────────────────────────────",
        f"🗓️ {now.strftime('%Y년 %m월 %d일')} ({weekday}) {now.strftime('%H:%M')}  |  "
        f"시장: {regime_label}",
        "────────────────────────────────────────────────",
        "",
    ]
    if health_note:
        lines += [f"> {health_note}", ""]
    # CR-10 ①: 왜 0개인가. 이게 없으면 독자가 시스템 고장으로 오해한다.
    from src.graph.report_formatter import no_pick_reason, upcoming_report_deadline, version_line
    lines += [f"> 신호 없음 사유 — {no_pick_reason(screen, regime)}", ""]
    _deadline = upcoming_report_deadline()
    if _deadline:                                   # CR-10 ⑪
        lines += [f"> 📅 {_deadline}", ""]

    # ── 섹션 1: 포트폴리오 (있으면) ────────────────────────────────
    lines.append("## 💼 내 포트폴리오 현황")
    lines.append("")
    if portfolio and portfolio.get("positions"):
        invested  = portfolio.get("invested_pct", 0.0)
        available = portfolio.get("available_pct", 0.0)
        cumulative = portfolio.get("cumulative_pnl_seed_pct", 0.0)
        lines.append(
            f"투자중: {invested:.1f}%  |  가용: {available:.1f}%  |  "
            f"누적 수익: {cumulative:+.2f}% of seed"
        )
        lines.append("")
        for p in portfolio["positions"]:
            name = f"{p['ticker_name']} ({p['ticker']})" if p.get("ticker_name") else p["ticker"]
            pnl  = _fmt_pct(p.get("position_pnl_pct"))
            lines.append(
                f"- {name} · 진입 {_fmt_money(p.get('entry_price'))} · "
                f"현재 {_fmt_money(p.get('current_price'))} · 수익률 {pnl}"
            )
        lines.append("")
    else:
        lines.append("보유 포지션 없음 (첫 거래 대기 중)")
        lines.append("")
    lines += ["---", ""]

    # ── 섹션 2: 오늘의 액션 플랜 (no pick) ─────────────────────────
    lines += [
        "## 🎯 오늘의 액션 플랜", "",
        "**오늘 신규 진입 추천 없음** — 스크리닝 단계에서 임계값을 통과한 종목이 없습니다.",
        "",
        "기존 포지션이 있다면 손절/익절 라인을 점검하세요. 신규 매수는 보류 권장.",
        "",
        "---", "",
    ]

    # ── 섹션 3: 시장 지표 ──────────────────────────────────────────
    lines += ["## 📊 시장 지표", ""]
    indicator_map = [
        ("kospi",        "KOSPI",         None),
        ("us_10y_yield", "미 10년 금리",   "rate"),
        ("vix",          "공포지수(VIX)", "vix"),
        ("usd_krw",      "원/달러 환율",   None),
        ("wti",          "WTI 유가",       None),
    ]
    any_row = False
    lines += ["| 지표 | 현재값 | 해석 |", "|------|--------|------|"]
    for key, name, interp_kind in indicator_map:
        raw = _scalar(market.get(key))
        if raw is None:
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            lines.append(f"| {name} | {raw} |  |")
            any_row = True
            continue
        interp = ""
        if interp_kind == "vix":
            interp = "공포" if num > 30 else ("주의" if num > 20 else "안정적")
        elif interp_kind == "rate":
            interp = "주식 부담↑" if num > 4.5 else "보통"
        value_str = f"{num:,.2f}" if num < 100 else f"{num:,.0f}"
        lines.append(f"| {name} | {value_str} | {interp} |")
        any_row = True
    if not any_row:
        lines.append("| - | - | 시장 지표 수집 실패 |")
    lines += ["", "---", ""]

    # ── 섹션 4: 스크리닝 결과 요약 ─────────────────────────────────
    lines += ["## 🔎 스크리닝 결과 요약", ""]
    lines.append(f"- 요약: {screen.get('summary', '-')}")
    if screen.get("warmup"):
        lines.append("- ⚠️ 뉴스 급증 팩터 워밍업 중 (발행일 기준 데이터 16일 필요) — 공시 이벤트만으로 후보 선정")

    scores = screen.get("scores") or {}
    try:
        from src.screening.stage2_scorer import NEWS_BURST_PCT_MIN as _pct_min
    except Exception:
        _pct_min = 0.8
    n_burst = sum(1 for v in scores.values() if (v.get("news_burst_pct") or 0.0) >= _pct_min)
    n_dart  = sum(1 for v in scores.values() if v.get("dart_event"))
    n_news  = sum(1 for v in scores.values() if v.get("news_match"))
    n_cand  = sum(1 for v in scores.values() if v.get("is_candidate"))
    n_filt  = sum(1 for v in scores.values() if v.get("filtered_reason"))
    lines.append(f"- 뉴스 급증 (유니버스 상위 20%): {n_burst}개")
    lines.append(f"- DART 주요 공시: {n_dart}개")
    lines.append(f"- 1-C 헤드라인 매칭 (태그): {n_news}개")
    lines.append(f"- 후보 {n_cand}개 / 필터 제외 {n_filt}개")
    lines.append("")

    top5 = sorted(scores.items(), key=lambda kv: (-(kv[1].get("score") or 0.0), kv[0]))[:5]
    if top5:
        lines.append("**참고: 점수 상위 5개**")
        lines.append("")
        lines += ["| 순위 | 종목 | 점수 | 팩터 |", "|-----|------|-----|------|"]
        for i, (tk, det) in enumerate(top5, 1):
            tags = []
            if det.get("news_burst") is not None:
                tags.append(f"뉴스 {det['news_burst']:.1f}배")
            if det.get("dart_event"):
                tags.append("공시: " + " / ".join((det.get("dart_titles") or [])[:1]))
            if det.get("news_match"):
                tags.append("1-C")
            if det.get("filtered_reason"):
                tags.append(f"제외({det['filtered_reason']})")
            lines.append(f"| {i} | {tk} | {det.get('score', 0):.2f} | {', '.join(tags) or '-'} |")
        lines.append("")
    lines += ["---", ""]

    # ── 면책 ───────────────────────────────────────────────────────
    lines += [
        "────────────────────────────────────────────────",
        f"생성: {now.strftime('%Y-%m-%d %H:%M:%S')} KST",
        f"버전: {version_line()}",                      # CR-10 ⑩
        "⚠️ AI 생성 분석 | 투자 조언 아님 | 최종 판단은 본인 책임",
        "────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


async def _publish_market_only_report(screen: dict, health_note: Optional[str] = None) -> dict:
    """
    선정 종목 0개일 때 호출.
    시장 스냅샷 + 스크리닝 결과 요약만 담은 리포트를 Notion에 발행한다.

    Returns:
      {"published": bool, "url": str | None, "fallback_path": str | None}
    """
    print("\n[DailyRunner] 시장 전용 리포트 생성 중 (no-pick)...")
    from src.graph.notion_publisher import publish_to_notion

    # ── DB 초기화 (CI 빈 환경 대비) ─────────────────────
    try:
        from src.data.prediction_logger import setup_feedback_system
        from src.data.position_tracker  import setup_position_tracker
        setup_feedback_system()
        setup_position_tracker()
    except Exception as e:
        logger.warning(f"[market-only] 보조 DB 초기화 실패 (계속 진행): {e}")

    market = await _collect_market_snapshot()
    regime = await _detect_regime_safe()

    portfolio = None
    try:
        from src.data.position_tracker import get_portfolio_pct_summary
        portfolio = get_portfolio_pct_summary()
    except Exception as e:
        logger.warning(f"[market-only] portfolio 조회 실패: {e}")

    report_md = _format_market_only_report(
        regime      = regime,
        market      = market,
        screen      = screen,
        portfolio   = portfolio,
        health_note = health_note,
    )

    # 로컬 백업 저장
    try:
        import os
        os.makedirs("data/reports", exist_ok=True)
        now = datetime.now(KST)
        local_path = (
            f"data/reports/Daily_stock_market_report_"
            f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M')}_NO_PICK.md"
        )
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"  ✅ 로컬 저장: {local_path}")
    except Exception as e:
        logger.warning(f"[market-only] 로컬 저장 실패: {e}")

    # Notion 발행 (chief_report=None → _markdown_to_blocks fallback 경로)
    result = await publish_to_notion(
        report_content    = report_md,
        ticker            = "MARKET",
        regime            = regime,
        strategy          = "NO_PICK",
        chief_report      = None,
        qualified_reports = None,
        all_reports       = None,
        debate_summary    = "",
        error_log         = None,
        portfolio_summary = portfolio,
        market_data       = market,
    )
    if result.get("success"):
        print(f"  ✅ Notion 발행: {result['url']}")
        return {"published": True, "url": result["url"], "fallback_path": None}
    else:
        print(f"  ⚠️ Notion 발행 실패 — 로컬 fallback: {result.get('fallback_path')}")
        return {
            "published":     False,
            "url":           None,
            "fallback_path": result.get("fallback_path"),
        }


# ─────────────────────────────────────────────────────────
# 통합 리포트 발행 (선정 종목 ≥ 1)
# ─────────────────────────────────────────────────────────

async def _publish_combined_report(
    per_ticker_results: list[dict],
    error_log:          list[str],
    health_note:        Optional[str] = None,
    screen:             Optional[dict] = None,
) -> dict:
    """종목별 분석 결과를 모아 하루 1개 통합 리포트를 발행한다.

    Notion 페이지 타이틀: {YYYYMMDD}_{ticker1}_{ticker2}_..._{strategy}
    """
    from src.graph.notion_publisher import publish_combined_to_notion

    print("\n[DailyRunner] 통합 리포트 발행 중...")

    # 시장 단위 정보는 첫 종목 결과 채택 (data_ingest/regime_detector가 종목별로
    # 중복 실행되지만 결과는 어차피 시장 동일)
    first       = per_ticker_results[0]
    regime      = first.get("current_regime") or "unknown"
    market_data = first.get("market_data") or {}

    # 포트폴리오 요약
    portfolio_summary = None
    try:
        from src.data.position_tracker import get_portfolio_pct_summary
        portfolio_summary = get_portfolio_pct_summary()
    except Exception as e:
        logger.warning(f"[combined] portfolio 조회 실패: {e}")

    # 통합 전략 라벨 & 타이틀
    strategy_label = _aggregate_strategy(per_ticker_results)
    now            = datetime.now(KST)
    ticker_part    = "_".join(r["ticker"] for r in per_ticker_results)
    title          = f"{now.strftime('%Y%m%d')}_{ticker_part}_{strategy_label}"

    result = await publish_combined_to_notion(
        per_ticker_results = per_ticker_results,
        regime             = regime,
        strategy           = strategy_label,
        title              = title,
        market_data        = market_data,
        portfolio_summary  = portfolio_summary,
        error_log          = error_log,
        health_note        = health_note,
        screen             = screen,
        fallback_content   = title,
    )

    if result.get("success"):
        print(f"  ✅ Notion 통합 발행: {result['url']}")
    else:
        print(f"  ⚠️ Notion 발행 실패 — 로컬 fallback: {result.get('fallback_path')}")

    return result


# ─────────────────────────────────────────────────────────
# mention_db 사전 적재 (stage1b 점수원 확보)
# ─────────────────────────────────────────────────────────

async def _warm_mention_db() -> None:
    """유니버스 전종목 × 1일치 mention을 적재.

    mention_db가 비면 stage1b의 news_burst가 전부 0이 되어 뉴스 급증 후보가 안 나온다. crawl_tickers는 다음을 이미 보유하므로 그대로 사용:
      - is_already_crawled 기반 오늘자 스킵
      - asyncio.Semaphore(max_concurrent) 동시성 제한
      - 개별 종목 실패 격리
      - NAVER API Circuit Breaker + Retry

    전체 실패는 격리한다 — mention 적재 실패가 데일리 중단으로 이어지지 않게.
    """
    try:
        from src.universe.universe_builder import load_universe
        from src.data.mention_tracker import crawl_tickers
        universe = list(load_universe())
        if not universe:
            print("[DailyRunner] mention 적재 스킵 — 유니버스 비어있음")
            return
        print(f"[DailyRunner] mention 적재 시작: 유니버스 {len(universe)}개 (동시 5)")
        t0 = time.monotonic()
        await crawl_tickers(universe, max_concurrent=5)
        print(f"[DailyRunner] mention 적재 완료 ({_fmt_duration(time.monotonic() - t0)})")
    except Exception as e:
        logger.warning(f"[DailyRunner] mention 적재 실패 (계속 진행): {e}", exc_info=True)
        print(f"[DailyRunner] ⚠️ mention 적재 실패 — screener 계속: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────

async def run_daily() -> dict:
    """
    오늘의 스크리닝 + 선정 종목별 심층 파이프라인.

    Returns:
      {
        "selected":  [ticker, ...],
        "mode":      "confirmed" | "optional ..." | "선정 종목 없음",
        "succeeded": [ticker, ...],
        "failed":    [{"ticker": ..., "error": ...}, ...],
        "total_seconds": float,
      }
    """
    overall_t0 = time.monotonic()

    # ── Step 0-a: 분기 유니버스 갱신 (CR-1, 결정 3-4b) ────────
    # 예전에는 scheduler.py(미가동 Docker)에만 있어 2026-05-10 이후 한 번도 안 돌았다.
    # 실패해도 그날 실행은 계속한다 — 기존 유니버스로 돌면 된다.
    try:
        from src.universe.universe_builder import build_universe, universe_refresh_due
        if universe_refresh_due():
            print("[DailyRunner] 분기 유니버스 갱신 시작")
            await asyncio.wait_for(asyncio.to_thread(build_universe), timeout=300)
    except Exception as e:
        logger.warning(f"[DailyRunner] 유니버스 갱신 실패(기존 유니버스로 계속): {e}")
        print(f"[DailyRunner] ⚠️ 유니버스 갱신 실패 — 기존 유니버스로 계속: {type(e).__name__}")

    # ── Step 0-b: mention_db 사전 적재 (stage1b 점수원 확보) ──
    await _warm_mention_db()

    # ── Step 1: 스크리닝 ──────────────────────────────────────
    # 레짐을 먼저 재서 Stage 2의 확정 최소 점수를 정한다 (G5: 약세·변동 장은 +1점).
    regime = await _detect_regime_safe()
    print(f"[DailyRunner] 레짐: {regime}")

    # run_screening은 sync. to_thread + wait_for(300)으로 외부 타임아웃 강제.
    try:
        screen = await asyncio.wait_for(
            asyncio.to_thread(run_screening, None, regime),
            timeout=300,
        )
    except asyncio.TimeoutError:
        logger.error("[DailyRunner] run_screening 300초 초과 — 빈 결과로 폴백")
        print("[DailyRunner] ⏱️ 스크리닝 300초 초과 — no-pick 처리")
        screen = {
            "confirmed":     [],
            "optional":      [],
            "summary":       "스크리닝 타임아웃 (300s)",
            "stage_results": {},
            "scores":        {},
        }
    selected, mode = _select_tickers(screen)

    print(
        f"[DailyRunner] 스크리닝 완료: "
        f"확정 {len(screen.get('confirmed') or [])}개 / "
        f"선택 {len(screen.get('optional') or [])}개"
    )

    # v2 §4: 유니버스 전 종목의 팩터값을 남긴다 — 선정 0개인 날도. screen_factors가 읽는다.
    # 첫 실행에 메서드 적용일을 기록한다 (INSERT OR IGNORE — 재실행해도 날짜가 안 밀린다).
    try:
        from src.evaluation.outcomes import record_screen_candidates, record_method_change, SCREEN_V2_KEY
        _today = datetime.now(KST).date().isoformat()
        record_method_change(SCREEN_V2_KEY, "정보 이벤트 스크린(v2): 발행일 뉴스 급증 + DART 공시", applied_date=_today)
        # sentiment_analyst가 재는 것이 바뀐 날. 이름(agent_weights·prediction_log 이력)은
        # 유지하되 의미가 달라졌으므로 전후 점수를 한 줄에 놓고 비교하면 안 된다.
        record_method_change(
            "sentiment_v3",
            "sentiment_analyst: 시장 헤드라인 30건 → 종목 이벤트(공시 제목 + 종목 헤드라인 + 선정 사유) 해석",
            applied_date=_today,
        )
        _rows = [{"ticker": t, **d} for t, d in (screen.get("scores") or {}).items()]
        _n = record_screen_candidates(_today, _rows, selected)
        print(f"[screen_candidates] {_n}행 기록 (후보 {sum(1 for r in _rows if r.get('is_candidate'))}개 / 분석 {len(selected)}개)")
    except Exception as e:
        logger.warning(f"[DailyRunner] 후보 기록 실패(계속): {e}")

    # 기록 전용(판단 무관). 선정 0개인 날도 남긴다 — 대조군과 뉴스 보관은 매일 쌓여야 의미가 있다.
    #   - universe_snapshots + random_control_outcomes: 같은 유니버스 무작위 5종목×100 (B-3 C3)
    #   - news_archive: Stage 1-C 헤드라인 원본 (LLM 포함 백테스트 조건 A-1 ①)
    try:
        from src.evaluation import random_control as _rc
        from src.evaluation import news_archive as _na
        from src.screening import stage1c_news as _s1c
        _today = datetime.now(KST).date().isoformat()
        _universe = list((screen.get("stage_results") or {}).get("universe") or [])
        if not _universe:
            from src.universe.universe_builder import load_universe
            _universe = list(load_universe())
        _n_snap = _rc.record_universe_snapshot(_today, _universe)
        _n_rc = _rc.register_random_control(_today, _universe)
        _n_news = _na.archive_headlines(_today, _s1c.LAST_HEADLINE_ITEMS)
        print(f"[random_control] 유니버스 {len(_universe)}종목 스냅샷 {_n_snap}행, 무작위 대조군 {_n_rc}행 등록")
        print(f"[news_archive] 헤드라인 {len(_s1c.LAST_HEADLINE_ITEMS)}건 중 신규 {_n_news}행 저장")
    except Exception as e:
        logger.warning(f"[DailyRunner] 대조군/뉴스 보관 실패(계속): {e}")

    if not selected:
        print(f"[DailyRunner] {mode} — 시장 전용 리포트로 대체 발행")
        # 발행 전에 건강도부터 판정한다 (CR-9). 종목이 0개여도 스크리닝 자체가
        # 고장난 것인지(헤드라인 0건·OHLCV 지연·MCP 키 누락) 리포트에 남겨야 한다.
        health = _assess_run_health([], [], screen)
        market_only_result = await _publish_market_only_report(screen, _health_note(health))
        return {
            "selected":         [],
            "mode":             mode,
            "succeeded":        [],
            "failed":           [],
            "market_only":      market_only_result,
            "total_seconds":    time.monotonic() - overall_t0,
        }

    print(f"[DailyRunner] 실행 대상: {len(selected)}개 ({mode}) → {', '.join(selected)}")

    # ── Step 2: ticker별 심층 분석 (발행은 하지 않음) ─────────
    succeeded: list[str] = []
    failed:    list[dict] = []
    per_ticker_results: list[dict] = []
    error_log_combined: list[str] = []

    budget_stopped = False
    for ticker in selected:
        # 종목 경계 = 예산을 끊기에 안전한 지점.
        # LLM 응답 직후에 끊으면 ResilientChain의 except Exception이 잡아
        # confidence=0.0 폴백으로 위장되므로, 반드시 여기서 확인한다.
        try:
            get_budget().check()
        except BudgetExceeded as e:
            logger.error(f"[DailyRunner] {e}")
            print(f"\n[DailyRunner] ⛔ {e}")
            print(f"[DailyRunner] 남은 {len(selected) - len(succeeded) - len(failed)}종목 분석 중단")
            budget_stopped = True
            break

        print(f"\n[DailyRunner] 심층 분석 시작: {ticker}")
        ticker_t0 = time.monotonic()
        try:
            # 종목별 10분 timeout. asyncio.TimeoutError는 Exception 서브클래스라
            # 아래 except에서 failed 리스트로 자동 분류된다.
            # 오늘 이 종목이 왜 뽑혔는지(공시 제목·뉴스 급증 배수·1-C 해석)를
            # 파이프라인 안으로 넘긴다. 이게 없으면 "새 정보가 생긴 종목"을 골라 놓고
            # 그 정보를 아무도 읽지 않는다 (doc/2026-09-21_agent-audit.md §2-2).
            state = await asyncio.wait_for(
                run_pipeline(ticker, publish=False,
                             screen_context=build_screen_context(ticker, screen)),
                timeout=600,
            )
            elapsed = time.monotonic() - ticker_t0
            payload = _extract_pipeline_result(state, ticker)
            per_ticker_results.append(payload)
            # 종목별 error_log 누적
            if isinstance(state, dict):
                error_log_combined += list(state.get("error_log") or [])
            print(f"[DailyRunner] 심층 분석 완료: {ticker} ({_fmt_duration(elapsed)})")
            succeeded.append(ticker)
        except Exception as e:
            elapsed = time.monotonic() - ticker_t0
            logger.error(f"[DailyRunner] {ticker} 실패: {type(e).__name__}: {e}", exc_info=True)
            print(f"[DailyRunner] 심층 분석 실패: {ticker} ({_fmt_duration(elapsed)}) — {type(e).__name__}: {e}")
            failed.append({"ticker": ticker, "error": f"{type(e).__name__}: {e}"})

    # ── 실행 건강도 판정 (발행 전) ────────────────────────────
    # 폴백 리포트로 끝까지 도는 "성공 위장"을 잡아낸다. 프로세스 종료 코드는 __main__이 결정한다.
    #
    # [왜 발행보다 먼저인가 — CR-9]
    #   예전에는 발행 뒤에 판정했다. 그래서 "에이전트가 전부 죽은 날의 리포트"가
    #   아무 표시 없이 Notion에 올라갔고, 읽는 사람은 GitHub Actions 빨간불을
    #   따로 확인해야만 그 사실을 알 수 있었다. 판정을 앞으로 옮겨 리포트 안에 한 줄 넣는다.
    #   budget_stopped는 종목 루프가 끝나야 확정되므로 여기가 가장 이른 시점이다.
    health = _assess_run_health(per_ticker_results, selected, screen)
    if budget_stopped:
        health["healthy"] = False
        health["reasons"].append(
            "LLM 예산 상한 초과로 분석이 중도 중단됨 — 종목 수 상한이나 "
            "재시도 루프를 점검할 것"
        )

    # ── Step 3: 통합 리포트 1회 발행 ──────────────────────────
    publish_result: Optional[dict] = None
    if per_ticker_results:
        publish_result = await _publish_combined_report(
            per_ticker_results = per_ticker_results,
            error_log          = error_log_combined,
            health_note        = _health_note(health),
            screen             = screen,
        )

    total = time.monotonic() - overall_t0
    print(
        f"\n[DailyRunner] 전체 완료: "
        f"{len(succeeded)}/{len(selected)}개 종목 "
        f"(실패 {len(failed)}개, 총 {_fmt_duration(total)})"
    )
    if failed:
        for f in failed:
            print(f"    ❌ {f['ticker']}: {f['error']}")

    print(get_budget().format_summary())
    try:
        from src.agents import base_agent as _ba
        print(f"[schema] 피드백 재요청 {_ba.SCHEMA_RETRY_COUNT}회 / 재요청 후에도 위반→abstain {_ba.SCHEMA_ABSTAIN_COUNT}회")
    except Exception:
        pass
    print(
        f"[DailyRunner] 건강도: 에이전트 {health['live_agent_reports']}/"
        f"{health['total_agent_reports']} 정상"
        + (f", OHLCV {health['ohlcv_stale_days']}일 지연" if health["ohlcv_stale_days"] is not None else "")
    )
    for w in health.get("warnings") or []:
        logger.warning(f"[DailyRunner] {w}")
        print(f"[DailyRunner] ⚠️ {w}")
    if not health["healthy"]:
        print("[DailyRunner] ⛔ 이번 실행은 정상 분석으로 볼 수 없음:")
        for r in health["reasons"]:
            logger.error(f"[DailyRunner] {r}")
            print(f"    - {r}")

    return {
        "selected":       selected,
        "mode":           mode,
        "succeeded":      succeeded,
        "failed":         failed,
        "publish_result": publish_result,
        "total_seconds":  total,
        "health":         health,
        "budget":         get_budget().snapshot(),
    }


# ─────────────────────────────────────────────────────────
# 단독 실행
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    load_dotenv()
    _result = asyncio.run(run_daily())

    # 리포트는 이미 발행됐지만, 분석이 실제로 이뤄지지 않았다면 종료 코드 1로 끝낸다.
    # → GitHub Actions가 빨간불로 표시하고, if: failure() 알림이 걸릴 수 있게 된다.
    #   (그전에는 전 에이전트가 죽어도 success로 기록되어 감지가 불가능했다.)
    _health = (_result or {}).get("health") or {}
    if _health and not _health.get("healthy", True):
        print("[DailyRunner] 종료 코드 1 — 실행 건강도 판정 실패")
        sys.exit(1)
