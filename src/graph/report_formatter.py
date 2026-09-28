"""
src/graph/report_formatter.py

리포트 포맷터 v4.1 (7섹션 구조)

  1. 💼 내 포트폴리오 현황
  2. 🎯 오늘의 액션 플랜  (기존 포지션 처리 + 신규 매수 추천)
  3. 📊 시장 지표
  4. 🤖 AI 분석단 의견  (chief_strategist 행에 '최종 판단' 라벨)
  5. 📋 에이전트별 상세 분석 (접기)
  6. ⚠️ 오늘의 주요 리스크  (top 5)
  7. 면책 및 생성 정보

[validate_report 호환]
  security.py validate_report()는 '최종 판단' 또는 '에이전트별 분석' 중
  하나라도 포함되어 있으면 통과. 섹션 4의 chief_strategist 행 라벨
  "**최종 판단**"이 들어가므로 통과됨.
"""

import re
import os
import logging
from collections import Counter
from datetime import datetime, timezone, timedelta, date
from typing import Optional

from src.schemas.agent_output import AnalysisReport

logger = logging.getLogger(__name__)

KST          = timezone(timedelta(hours=9))
WEEKDAY_KR   = ["월", "화", "수", "목", "금", "토", "일"]
SORT_ORDER   = {"SELL": 0, "HOLD": 1, "BUY": 2}

REGIME_KR = {
    "VOLATILE": "변동성 장세",
    "BULL":     "상승 장세",
    "BEAR":     "하락 장세",
    "NEUTRAL":  "중립 장세",
    "UNKNOWN":  "시장 파악 중",
}

AGENT_NAME_KR = {
    "kr_market_specialist":  "한국시장 전문가",
    "us_market_specialist":  "미국시장 전문가",
    "macro_economist":       "매크로 분석",
    "quant_analyst":         "퀀트 분석",
    "technical_analyst":     "기술적 분석",
    "sentiment_analyst":     "감성 분석",
    "fundamental_analyst":   "펀더멘털 분석",
    "chief_strategist":      "종합 판단",
}

SIGNAL_KR = {
    "BUY":  "🟢 매수",
    "HOLD": "🟡 관망",
    "SELL": "🔴 매도",
}

STRATEGY_KR = {
    "BUY":  "매수",
    "HOLD": "관망",
    "SELL": "매도",
}


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_money(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(v, digits: int = 2, signed: bool = True) -> str:
    if v is None:
        return "-"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if (signed and fv >= 0) else ("" if signed else "")
    return f"{sign}{fv:.{digits}f}%"


def _scalar(val):
    """market_data 값에서 표시 가능한 스칼라 추출."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        for k in ("value", "current", "price", "rate", "level",
                 "latest_close", "close", "vix"):
            if k in val and isinstance(val[k], (int, float, str)):
                return val[k]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 생성 함수
# ─────────────────────────────────────────────────────────────────────────────

def _header_block(regime: str) -> list[str]:
    now     = datetime.now(KST)
    weekday = WEEKDAY_KR[now.weekday()]
    regime_label = REGIME_KR.get(regime.upper(), regime)
    return [
        "────────────────────────────────────────────────",
        f"🗓️ {now.strftime('%Y년 %m월 %d일')} ({weekday}) {now.strftime('%H:%M')}  |  시장: {regime_label}",
        "────────────────────────────────────────────────",
        "",
    ]


def _portfolio_section(portfolio_summary: Optional[dict]) -> list[str]:
    """섹션 1: 💼 내 포트폴리오 현황"""
    lines = ["## 💼 내 포트폴리오 현황", ""]

    if not portfolio_summary or not portfolio_summary.get("positions"):
        lines += ["보유 포지션 없음 (첫 거래 대기 중)", "", "---", ""]
        return lines

    invested = portfolio_summary.get("invested_pct", 0.0)
    available = portfolio_summary.get("available_pct", 0.0)
    lines.append(
        f"투자중: {invested:.1f}%  |  가용: {available:.1f}%  |  "
        + seed_pnl_note(portfolio_summary)
    )
    lines.append("")

    lines += [
        "| 종목 | 매수일 | 매수가 | 현재가 | 배분 | 수익률 | 손절가 | 보유 |",
        "|------|--------|--------|--------|------|--------|--------|------|",
    ]
    for p in portfolio_summary["positions"]:
        name_part = f"{p['ticker_name']} ({p['ticker']})" if p.get("ticker_name") else p["ticker"]
        alert = p.get("alert")
        prefix = "⚠️ " if alert == "stop_loss" else ("🎯 " if alert == "target" else "")

        alloc_str = f"{p['allocation_pct']:.0f}%" if p.get("allocation_pct") is not None else "-"
        pnl_str   = _fmt_pct(p.get("position_pnl_pct"))
        # CR-5: D+10(≈14일) 경과는 자동 청산하지 않고 표시만 한다. 판단은 사람이 한다.
        holding   = f"{p['holding_days']}일" if p.get("holding_days") is not None else "-"
        if p.get("overdue"):
            holding += " ⏰"

        # [2026-09-15] 1차 익절·목표보유 열을 뺐다 (제안값·기록값이라 판단 정보가 없다).
        #   배분은 남긴다 — 제안이 아니라 실제 체결 비중이고 "얼마나 물려 있나"에 쓰인다.
        lines.append(
            f"| {prefix}{name_part} "
            f"| {p.get('entry_date') or '-'} "
            f"| {_fmt_money(p.get('entry_price'))} "
            f"| {_fmt_money(p.get('current_price'))} "
            f"| {alloc_str} "
            f"| {pnl_str} "
            f"| {_fmt_money(p.get('stop_loss_price'))} "
            f"| {holding} |"
        )
    lines.append("")

    closed = portfolio_summary.get("closed_recent") or []
    if closed:
        lines += [
            "📌 **청산 이력 (최근 5건)**",
            "",
            "| 종목 | 매수가 | 매도가 | 배분 | 포지션 손익 | 시드 손익 | 보유기간 |",
            "|------|--------|--------|------|------------|----------|---------|",
        ]
        for c in closed:
            alloc_str   = f"{c['allocation_pct']:.0f}%" if c.get("allocation_pct") is not None else "-"
            pos_pnl     = _fmt_pct(c.get("position_pnl_pct"))
            seed_pnl    = _fmt_pct(c.get("seed_pnl_pct"))
            seed_pnl    = f"{seed_pnl} of seed" if seed_pnl != "-" else "-"
            holding     = f"{c['holding_days']}일" if c.get("holding_days") is not None else "-"
            lines.append(
                f"| {c['ticker']} "
                f"| {_fmt_money(c.get('entry_price'))} "
                f"| {_fmt_money(c.get('close_price'))} "
                f"| {alloc_str} "
                f"| {pos_pnl} "
                f"| {seed_pnl} "
                f"| {holding} |"
            )
        lines.append("")

    lines += ["---", ""]
    return lines


def _action_plan_section(
    ticker: str,
    final: Optional[AnalysisReport],
    portfolio_summary: Optional[dict],
) -> list[str]:
    """섹션 2: 🎯 오늘의 액션 플랜"""
    lines = ["## 🎯 오늘의 액션 플랜", ""]

    # --- 기존 포지션 처리 -----------------------------------------------------
    lines += ["### 기존 포지션 처리", ""]

    positions = (portfolio_summary or {}).get("positions") or []
    if positions:
        lines += [
            "| 종목 | 권고 | 이유 (한줄) |",
            "|------|------|------------|",
        ]
        # [2026-09-15] 가격 신호(손절·목표선)를 권고 칸 맨 앞에 놓는다.
        #   예전에는 "오늘 분석 대상인가"만 봐서, 이미 손절선 아래인 종목도
        #   "⏳ 모니터링 (오늘 분석 대상 아님)"으로만 나왔다. 결정이 필요한 자리에
        #   결정에 필요한 사실이 없었다.
        for p in positions:
            name_part = f"{p['ticker_name']} ({p['ticker']})" if p.get("ticker_name") else p["ticker"]
            hit = position_alert(p)
            if p["ticker"] != ticker:
                action = "⏳ 모니터링"
                reason = "오늘 분석 대상 아님"
            elif final is None:
                action = "⏳ 보류"
                reason = "오늘 분석 결과 없음"
            elif final.recommendation == "HOLD":
                action = "🟡 유지"
                reason = headline_reason(final, 100) or "-"
            elif final.recommendation == "SELL":
                action = "🔴 매도 검토"
                reason = headline_reason(final, 100) or "-"
            else:  # BUY for existing position → 비중 확대
                action = "🟢 추가 매수 검토"
                reason = headline_reason(final, 100) or "-"
            if hit:
                grade, text = hit
                action = f"**{ALERT_ICON[grade]} {text}** — {action}"
            lines.append(f"| {name_part} | {action} | {reason} |")
        lines.append("")
    else:
        lines += ["보유 포지션 없음", ""]

    # --- 신규 매수 추천 -------------------------------------------------------
    available_pct = (portfolio_summary or {}).get("available_pct", 100.0)
    lines += [f"### 신규 매수 추천 (가용: {available_pct:.0f}%)", ""]

    if final and final.recommendation == "BUY" and final.entry_price is not None:
        # [2026-09-15] position_size_pct · take_profit_1/2 · rr_ratio를 본문에서 뺐다.
        #   전부 5% 고정 / R:R 2.0 항등식에서 기계적으로 나온 값이라 정보가 없는데
        #   숫자라서 정보처럼 읽힌다 (S5 D-2 경로3). DB에는 계속 기록된다.
        # 남기는 것: 진입가와 손절가 — 보유 중 판단에 실제로 쓰인다.
        sl_str = "-"
        if final.stop_loss is not None:
            pct = f" ({final.stop_loss_pct:+.1f}%)" if final.stop_loss_pct is not None else ""
            sl_str = f"{_fmt_money(final.stop_loss)}원{pct}"
        lines += [
            "| 종목 | 최종 판단 | 확신도 | 진입가 | 제안 손절가 |",
            "|------|----------|--------|--------|------------|",
            f"| {ticker} | 🟢 매수 | {final.confidence:.0%} | "
            f"{_fmt_money(final.entry_price)}원 | {sl_str} |",
            "",
        ]
        if final.reasoning:
            lines += [f"> 근거: {headline_reason(final, 140)}", ""]
    else:
        lines += ["오늘 신규 진입 추천 없음", ""]

    lines += [f"> ⚖️ {stop_loss_note(final)} {human_decides_note()}", ""]   # CR-10 ⑧
    lines += ["---", ""]
    return lines


def _market_data_section(market_data: Optional[dict]) -> list[str]:
    """섹션 3: 📊 시장 지표"""
    lines = ["## 📊 시장 지표", ""]

    if not market_data:
        lines += ["시장 지표 데이터 없음", "", "---", ""]
        return lines

    indicator_map = [
        ("kospi",        "KOSPI",         None),
        ("us_10y_yield", "미 10년 금리",   "rate"),
        ("vix",          "공포지수(VIX)", "vix"),
        ("usd_krw",      "원/달러 환율",   None),
        ("wti",          "WTI 유가",       None),
    ]

    rows = []
    for key, name, interp_kind in indicator_map:
        if key not in market_data:
            continue
        raw = _scalar(market_data.get(key))
        if raw is None:
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            rows.append((name, str(raw), ""))
            continue

        # 해석
        interp = ""
        if interp_kind == "vix":
            interp = "공포" if num > 30 else ("주의" if num > 20 else "안정적")
        elif interp_kind == "rate":
            interp = "주식 부담↑" if num > 4.5 else "보통"
        rows.append((name, f"{num:,.2f}" if num < 100 else f"{num:,.0f}", interp))

    if not rows:
        lines += ["시장 지표 데이터 없음", "", "---", ""]
        return lines

    lines += [
        "| 지표 | 현재값 | 해석 |",
        "|------|--------|------|",
    ]
    for name, value, interp in rows:
        lines.append(f"| {name} | {value} | {interp} |")
    lines += ["", "---", ""]
    return lines


def _agent_summary_section(
    agents: list[AnalysisReport],
    final:  Optional[AnalysisReport],
) -> list[str]:
    """섹션 4: 🤖 AI 분석단 의견 — chief 행에 '최종 판단' 라벨 포함."""
    lines = ["## 🤖 AI 분석단 의견", ""]
    lines += [
        "| 분석가 | 의견 | 신뢰도 | 핵심 근거 (한줄) | 비고 |",
        "|--------|------|--------|-----------------|------|",
    ]

    for r in sorted(agents, key=lambda x: SORT_ORDER.get(x.recommendation, 1)):
        name = AGENT_NAME_KR.get(r.agent_name, r.agent_name)
        sig  = SIGNAL_KR.get(r.recommendation, r.recommendation)
        conf = f"{r.confidence:.0%}"
        # [2026-09-15] 이상값이 든 문장은 요약에도 쓰지 않는다 (상세와 같은 기준)
        _clean, _ = filter_extreme_values(r.reasoning)
        reason = ((_clean or r.reasoning or ["-"])[0])[:100]
        if r.confidence == 0.0:
            note = "⚠️ 데이터없음"
        elif r.confidence < 0.6:
            note = "⚠️ 참고용"
        else:
            note = ""
        lines.append(f"| {name} | {sig} | {conf} | {reason} | {note} |")

    if final:
        lines.append("|---|---|---|---|---|")
        sig    = SIGNAL_KR.get(final.recommendation, final.recommendation)
        conf   = f"{final.confidence:.0%}"
        reason = headline_reason(final, 100) or "-"   # 투표 집계 줄은 건너뛴다
        note   = "—"
        if getattr(final, "needs_review", False):
            # HOLD로 날조된 것이 아니라 판단 자체가 없음을 드러낸다 (G5)
            sig, note = "⚠️ REVIEW", "사람 확인 필요"
        lines.append(f"| **최종 판단** | **{sig}** | **{conf}** | {reason} | {note} |")
        # CR-10 ④: 확신도 옆 보정 적중률. 미보정이면 그렇다고 못박는다.
        lines += ["", f"> 확신도 읽는 법 — {calibrated_confidence_note(final.confidence, final.agent_name)}"]
    _ab = abstain_summary(agents)                          # CR-10 ⑤
    if _ab:
        lines += ["", f"> 투표 — {_ab}"]

    lines += ["", "---", ""]
    return lines


def _details_section(
    agents: list[AnalysisReport],
    final:  Optional[AnalysisReport],
    qualified_names: set[str],
) -> list[str]:
    """섹션 5: 📋 에이전트별 상세 분석 (접기)"""
    lines = [
        "<details>",
        "<summary>📋 에이전트별 상세 분석 (클릭하여 펼치기)</summary>",
        "",
    ]

    # [2026-09-15] 시장 공통 에이전트는 상단 "오늘의 시장 배경"에서 한 번만 보여준다.
    detail_agents = [r for r in agents if r.agent_name not in MARKET_WIDE_AGENTS]
    for r in sorted(detail_agents, key=lambda x: SORT_ORDER.get(x.recommendation, 1)):
        gate = "✅" if r.agent_name in qualified_names else "⚠️ (QG 폴백)"
        sig  = SIGNAL_KR.get(r.recommendation, r.recommendation)
        name = AGENT_NAME_KR.get(r.agent_name, r.agent_name)
        lines.append(f"### {gate} {name} ({r.agent_name}) {sig} (신뢰도: {r.confidence:.2f})")
        # 상식 범위 밖 밸류에이션은 근거에서 빼고 뺐다는 사실을 밝힌다
        kept_reason, drop_a = filter_extreme_values(r.reasoning)
        kept_basis,  drop_b = filter_extreme_values(r.prediction_basis)
        for reason in kept_reason:
            lines.append(f"- {reason}")
        if kept_basis:
            lines += ["", "**근거:**"]
            for basis in kept_basis:
                lines.append(f"- {basis}")
        note = extreme_value_note(drop_a + drop_b)
        if note:
            lines += ["", f"> {note}"]
        if r.risk_factors:
            lines += ["", f"> ⚠️ 리스크: {' / '.join(r.risk_factors)}"]
        lines.append("")

    if final:
        lines.append("### 🎯 종합 판단 (chief_strategist)")
        for reason in final.reasoning:
            lines.append(f"- {reason}")
        if final.selection_rationale:
            lines += ["", f"**선정 종목:** {final.selection_rationale}"]
        lines.append("")

    lines += ["</details>", "", "---", ""]
    return lines


def _risk_section(
    agents: list[AnalysisReport],
    final:  Optional[AnalysisReport],
) -> list[str]:
    """섹션 6: ⚠️ 오늘의 주요 리스크 (top 5)"""
    lines = ["## ⚠️ 오늘의 주요 리스크", ""]
    lines += [f"> {debate_status(agents, '')}", ""]        # CR-10 ⑨

    unique_risks = real_risk_factors(agents + ([final] if final else []))

    if not unique_risks:
        lines += ["주요 리스크 없음", "", "---", ""]
        return lines

    for i, risk in enumerate(unique_risks[:5], 1):
        lines.append(f"{i}. {risk}")
    lines += ["", "---", ""]
    return lines


def _footer_section() -> list[str]:
    now = datetime.now(KST)
    return [
        "────────────────────────────────────────────────",
        f"생성: {now.strftime('%Y-%m-%d %H:%M:%S')} KST",
        "⚠️ AI 생성 분석 | 투자 조언 아님 | 최종 판단은 본인 책임",
        "────────────────────────────────────────────────",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def format_report_v4(
    ticker:             str,
    regime:             str,
    strategy:           str,
    final:              Optional[AnalysisReport],
    agents:             list[AnalysisReport],
    qualified_reports:  list[AnalysisReport],
    debate_summary:     str        = "",
    reconciled_signals: list[dict] | None = None,
    error_log:          list[str]  | None = None,
    portfolio_summary:  dict | None = None,
    market_data:        dict | None = None,
) -> str:
    """7섹션 v4.1 마크다운 리포트 생성."""
    qualified_names = {r.agent_name for r in (qualified_reports or [])}

    now = datetime.now(KST)
    title = f"# Daily stock market report_{now.strftime('%Y%m%d')}_{now.strftime('%H:%M')}_{strategy}"

    # [2026-09-15] 순서를 doc/2026-09-14_morning-workflow.md 1절의 읽는 순서에 맞췄다.
    #   건강도 → 최종 판단 → 액션 → 리스크 → 상세 → 시장 → 포트폴리오
    #   (예전: 포트폴리오 → 액션 → 시장 → 판단 → 상세 → 리스크)
    lines: list[str] = [title, ""]
    lines += [f"> 📌 {today_line([{'ticker': ticker, 'chief_report': final}], portfolio_summary)}", ""]
    lines += _header_block(regime)
    lines += _agent_summary_section(agents, final)
    lines += _action_plan_section(ticker, final, portfolio_summary)
    lines += _risk_section(agents, final)
    lines += _details_section(agents, final, qualified_names)
    lines += _market_data_section(market_data)
    lines += _portfolio_section(portfolio_summary)
    lines += _footer_section()

    return "\n".join(lines)


async def report_formatter_node(state) -> dict:
    """파이프라인 노드 — v4.1 포맷터."""
    print(f"\n[5/7] report_formatter v4.1")

    final  = next((r for r in state.analysis_reports if r.agent_name == "chief_strategist"), None)
    agents = [r for r in state.analysis_reports if r.agent_name != "chief_strategist"]

    # 포트폴리오 요약 로드 (실패해도 리포트 생성은 계속)
    portfolio_summary = None
    try:
        from src.data.position_tracker import get_portfolio_pct_summary
        portfolio_summary = get_portfolio_pct_summary()
    except Exception as e:
        logger.warning(f"[report_formatter] 포트폴리오 요약 로드 실패: {e}")

    report_content = format_report_v4(
        ticker             = state.ticker,
        regime             = state.current_regime,
        strategy           = state.final_strategy,
        final              = final,
        agents             = agents,
        qualified_reports  = state.qualified_reports or [],
        debate_summary     = state.debate_summary or "",
        reconciled_signals = state.reconciled_signals or [],
        error_log          = state.error_log or [],
        portfolio_summary  = portfolio_summary,
        market_data        = state.market_data or {},
    )

    try:
        now = datetime.now(KST)
        os.makedirs("data/reports", exist_ok=True)
        path = f"data/reports/Daily_stock_market_report_{now.strftime('%Y%m%d')}_{now.strftime('%H%M')}_{state.final_strategy}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"  ✅ 리포트 저장: {path}")
    except Exception as e:
        logger.warning(f"[report_formatter] 저장 실패: {e}")

    return {"report_content": report_content}


# ─────────────────────────────────────────────────────────────────────────────
# CR-10 결정 무관 항목 (2026-09-14): ⑦ 점수 분해 · ⑩ 버전 표기 · ⑪ 이벤트
# ─────────────────────────────────────────────────────────────────────────────

# 점수 항목 → 사람이 읽는 이름. stage2_scorer._score_one의 키와 1:1.
SCORE_LABELS: list[tuple[str, str, int]] = [
    ("media_price_divergence", "미디어-가격 괴리", 2),
    ("mention_spike",          "언급량 급증",      1),
    ("mention_buildup",        "언급량 누적",      1),
    ("sentiment_acceleration", "감성 가속",        1),
    ("news_match",             "뉴스 매치",        1),
]


def score_breakdown(detail: Optional[dict]) -> str:
    """⑦ 왜 이 종목이 뽑혔나 — 점수를 항목별로 펼친다.

    리서치 도구에서 가장 아쉬웠던 칸이다. 점수만 보이고 무엇이 켜졌는지 안 보이면
    독자가 추천을 검증할 수 없다. stage_results는 이미 있는데 리포트로 안 갔다.
    """
    if not detail:
        return "점수 분해 없음"
    parts = [f"{label} +{pt}" for key, label, pt in SCORE_LABELS if detail.get(key)]
    qs = int(detail.get("quant_score", 0) or 0)
    if qs:
        parts.append(f"정량신호 {detail.get('quant_signal_count', 0)}개 +{qs}")
    total = detail.get("total_score", 0)
    pc5 = detail.get("price_change_5d")
    tail = f" | 5일 {pc5*100:+.1f}%" if isinstance(pc5, (int, float)) else ""
    return f"{total}점 = " + (" · ".join(parts) if parts else "가점 없음") + tail


def version_line() -> str:
    """⑩ 이 리포트가 어느 코드·어느 모델로 나왔는지.

    나중에 "그날 리포트가 어느 코드였나"를 되짚을 수 있어야 성과를 코드 버전에
    귀속시킬 수 있다(S2 A-6). 조회 실패는 조용히 '미상'으로 둔다.
    """
    import os
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or "미상"
    except Exception:
        sha = "미상"

    chief_mode = os.getenv("CHIEF_MODE", "legacy")
    try:
        from src.agents.chief_strategist import CHIEF_MODEL
        from src.screening.stage1c_news import NEWS_SCREEN_MODEL
        from src.utils.shadow_log import SHADOW_POLICIES, shadow_active
        models = f"chief={CHIEF_MODEL} / news={NEWS_SCREEN_MODEL}"
        active = [n for n in SHADOW_POLICIES if shadow_active(n)]
        shadow = f"섀도 {', '.join(active)}" if active else "섀도 없음"
    except Exception:
        models, shadow = "모델 미상", "섀도 미상"

    return f"코드 {sha} | CHIEF_MODE={chief_mode} | {models} | {shadow}"


# ⑪ 정기보고서 제출 마감 (12월 결산법인 관례).
# krx_market.get_earnings_calendar와 같은 기준이지만 DART를 부르지 않는다 —
# DART는 간헐적으로 끊기고(CR-3), 리포트 발행이 거기 묶이면 안 된다.
# 한계: 종목별 실적 발표일이 아니라 시장 공통 마감일이다.
_DEADLINE_RULES = [("03-31", "전년 사업보고서"), ("05-15", "1분기 보고서"),
                   ("08-14", "반기보고서"), ("11-14", "3분기 보고서")]


def upcoming_report_deadline(today=None, within_days: int = 14) -> Optional[str]:
    """보유기간(기본 14일) 안에 정기보고서 제출 마감이 있으면 문구, 없으면 None."""
    from datetime import date as _date, datetime as _dt

    d = today or _dt.now(KST).date()
    if isinstance(d, _dt):
        d = d.date()
    for year in (d.year, d.year + 1):
        for mmdd, label in _DEADLINE_RULES:
            try:
                due = _date.fromisoformat(f"{year}-{mmdd}")
            except ValueError:
                continue
            gap = (due - d).days
            if 0 <= gap <= within_days:
                return f"{due.isoformat()} {label} 제출 마감 (D-{gap}) — 보유기간 내 실적 변동 가능"
    return None


def no_pick_reason(screen: Optional[dict], regime: str) -> str:
    """① 오늘 확정 0개인 이유를 수치로 설명한다.

    "왜 0개인가"가 없으면 독자가 시스템 고장으로 오해한다(G5로 0개 출력이 정상이 된 뒤의 숙제).
    """
    screen = screen or {}
    scores = screen.get("scores") or {}
    cmin   = screen.get("confirmed_min")
    regime_label = REGIME_KR.get((regime or "").upper(), regime or "미상")

    if not scores:
        return f"시장: {regime_label} | 점수 산출 실패 — 스크리닝이 정상 동작했는지 확인할 것"

    vals = [int((d or {}).get("total_score", 0)) for d in scores.values()]
    top  = max(vals) if vals else 0
    dist = Counter(vals)
    dist_str = ", ".join(f"{s}점 {dist[s]}개" for s in sorted(dist, reverse=True) if dist[s])
    cmin_str = f"{cmin}점" if cmin is not None else "미상"
    return (f"시장: {regime_label} | 확정 기준 {cmin_str} | 최고 점수 {top}점 | "
            f"분포: {dist_str} — 기준을 넘은 종목이 없다")


# ─────────────────────────────────────────────────────────────────────────────
# CR-10 ②③④⑧ (2026-09-14) — Q2·Q4 결정이 정해져 구현 가능해진 항목
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_context_line() -> Optional[str]:
    """② "이걸 사면 지수 대신 사는 것" — 최근 5/10/20거래일 벤치마크 수익률.

    [왜 필요한가]
      백테스트에서 무작위 5종목도 KODEX 200에 연 43~58%p 뒤졌다. 독자가 종목 추천을
      읽을 때 "대안은 지수였다"는 기준선을 모르면 추천의 의미를 잴 수 없다.
      결정 Q2(c): 리포트에는 지수 대비를 쓰고 승격 판정선은 무작위 대조군에 건다.

    yfinance 호출 1회. 실패하면 None을 돌려 리포트에서 줄을 뺀다.
    """
    try:
        import yfinance as yf
        from src.evaluation.outcomes import BENCHMARK
        hist = yf.Ticker(BENCHMARK).history(period="2mo")
        closes = [float(v) for v in hist["Close"].dropna()]
        if len(closes) < 21:
            return None
        last = closes[-1]
        parts = []
        for n in (5, 10, 20):
            prev = closes[-1 - n]
            parts.append(f"{n}일 {(last / prev - 1) * 100:+.1f}%")
        return (f"이 종목을 사면 KODEX 200({BENCHMARK}) 대신 사는 것이다. "
                f"지수 최근 " + " · ".join(parts))
    except Exception as e:                       # 네트워크·데이터 문제는 리포트를 막지 않는다
        logger.warning(f"[report] 벤치마크 수익률 조회 실패: {e}")
        return None


def recent_performance_lines() -> list[str]:
    """③ 최근 실현 성적 — 표본 단위는 하루(CR-8 / 결정 Q6-a).

    표본이 모자라면 수치 대신 "표본 N일 (최소 M)"을 보여준다. 적은 표본의 적중률을
    성적처럼 내보이면 없는 신호를 있는 것처럼 읽게 된다.
    """
    try:
        from src.evaluation.outcomes import HORIZONS, MIN_DAYS_FOR_TREND, summary
    except Exception:
        return []
    lines: list[str] = []
    for h in HORIZONS:
        try:
            s = summary(h)
        except Exception as e:
            logger.warning(f"[report] D+{h} 성적 조회 실패: {e}")
            continue
        n_days = s.get("n_days") or 0
        if n_days < MIN_DAYS_FOR_TREND:
            lines.append(f"- **D+{h}**: 표본 {n_days}일 (최소 {MIN_DAYS_FOR_TREND}일) — 아직 성적을 말할 수 없다")
            continue
        lines.append(
            f"- **D+{h}**: {n_days}일  평균 {_fmt_pct(s['avg_raw'])}  "
            f"알파 {_fmt_pct(s['avg_alpha'])}  비용차감 {_fmt_pct(s['avg_net'])}  "
            f"적중 {s['hit_rate']*100:.0f}% (손익분기 "
            + (f"{s['breakeven_win_rate']*100:.0f}%)" if s.get("breakeven_win_rate") is not None else "n/a)")
        )
    no_pick = 0
    try:
        from src.evaluation.outcomes import no_pick_days
        no_pick = no_pick_days()
    except Exception:
        pass
    if no_pick:
        lines.append(f"- 확정 0개였던 날 **{no_pick}일** (표본에서 제외, 결정 Q8-a)")
    return lines


def calibrated_confidence_note(confidence: Optional[float], agent: Optional[str] = None) -> str:
    """④ 확신도 옆 보정 적중률. 표본 미달이면 "미보정"이라고 못박는다.

    [왜 "미보정"을 명시하나]
      지금 모든 BUY의 배분은 5% 고정인데, 그 이유는 확신도가 아니라 **보정 표본 부족**이다.
      그 사실을 안 적으면 "확신도가 낮아서 5%"로 읽힌다.
    """
    if confidence is None:
        return "확신도 없음"
    try:
        from src.evaluation.calibration import MIN_SAMPLES, calibrated_hit_rate
        rate, tier = calibrated_hit_rate(float(confidence), agent=agent)
    except Exception:
        return f"{confidence:.0%} (보정 조회 실패)"
    if rate is None:
        return f"{confidence:.0%} — **미보정** (보정 표본 {MIN_SAMPLES}건 미만. 자기보고 값이다)"
    scope = "이 에이전트" if tier == "agent" else "전체"
    return f"{confidence:.0%} — 같은 구간 실제 적중 {rate*100:.0f}% ({scope} 기준)"


def human_decides_note(horizon: int = 10) -> str:
    """⑧ 손절·목표는 사람이 정한다 + 이 표본의 손익분기 승률.

    승률 하나만으로는 좋다·나쁘다를 말할 수 없다(결정 Q4-c). 손익분기선을 같이 보여야
    "적중 48%"가 이익인지 손실인지 판단된다.
    """
    # [2026-09-15] "사람이 정한다"는 stop_loss_note가 맡는다. 여기는 손익분기만 — 같은
    # 문장을 두 번 쓰면 읽는 사람이 둘 다 흘려 읽는다.
    base = "시스템은 가격 도달 시에만 청산을 표시하고, 시간 경과는 알림만 한다."
    try:
        from src.evaluation.outcomes import summary
        s = summary(horizon)
        be = s.get("breakeven_win_rate")
        if be is not None and s.get("n_days"):
            return base + (f" 이 표본(D+{horizon}, {s['n_days']}일)의 손익분기 승률은 "
                           f"**{be*100:.0f}%** — 적중률이 이보다 낮으면 비용을 못 넘는다.")
    except Exception as e:
        logger.warning(f"[report] 손익분기 승률 조회 실패: {e}")
    return base + " 손익분기 승률은 표본이 쌓이면 여기에 표시된다."


# ─────────────────────────────────────────────────────────────────────────────
# 2026-09-15 리포트 정직성·가독성 정리
#
# [왜 거래 파라미터를 본문에서 빼나 — S5 D-2 경로3]
#   S5 조사 결론: "공개 데이터 + 소수 종목 + 1~2주 보유"로 1~3등급 검증을 통과한
#   사례가 없다. 구조를 바꾸지 않기로 했으므로((b) 리서치 도구 유지), 리포트가
#   수익을 약속하는 것처럼 보이면 안 된다.
#   position_size_pct는 지금 전부 5% 고정이고, rr_ratio는 항등식 2.0이며,
#   take_profit은 그 5%와 2.0에서 기계적으로 나온 값이다. 셋 다 정보가 없는데
#   숫자라서 정보처럼 읽힌다. DB에는 계속 쌓는다 — (a) 전환 시 연속성과
#   옵션 A 섀도가 이 값을 쓴다.
# ─────────────────────────────────────────────────────────────────────────────

# 거래 파라미터 중 본문에서 뺀 것. DB 기록은 유지한다.
SUPPRESSED_TRADE_FIELDS = ("position_size_pct", "take_profit_1", "take_profit_2", "rr_ratio")


def stop_loss_note(final: Optional[AnalysisReport]) -> str:
    """손절가는 남기되 제안임을 못박는다 (CR-10 ⑧과 통합).

    손절선은 보유 중 판단에 실제로 쓰이므로 뺄 수 없다. 다만 시스템이 지키는
    약속이 아니라 계산된 제안이라는 것을 같은 줄에서 밝힌다.
    """
    base = "손절가는 **참고용 제안**이다. 실제 값과 청산 시점은 사람이 정한다."
    if final is None or final.stop_loss is None:
        return base
    pct = f" ({final.stop_loss_pct:+.1f}%)" if final.stop_loss_pct is not None else ""
    return f"제안 손절가 {_fmt_money(final.stop_loss)}원{pct} — " + base


def abstain_summary(agents: list[AnalysisReport]) -> Optional[str]:
    """⑤ 8명 중 몇 명이 왜 유보했나. 유보가 없으면 None.

    confidence 0.0은 "관망"이 아니라 "의견 없음"이다. 그 둘을 섞으면 투표 수가
    부풀어 보인다. 사유는 두 갈래로만 나눈다 — 데이터가 없었나, 근거가 모자랐나.
    """
    total = len(agents)
    if not total:
        return None
    no_data, thin = [], []
    for r in agents:
        if r.confidence > 0.0 and getattr(r, "data_sufficient", True):
            continue
        text = " ".join(r.reasoning or [])
        (no_data if ("폴백" in text or "실패" in text or "부족" in text) else thin).append(r.agent_name)
    n = len(no_data) + len(thin)
    if not n:
        return None
    parts = []
    if no_data:
        parts.append(f"데이터 없음 {len(no_data)}")
    if thin:
        parts.append(f"근거 부족 {len(thin)}")
    return f"{total}명 중 **{n}명 유보** ({', '.join(parts)}) — 실제 투표는 {total - n}명"


def debate_status(agents: list[AnalysisReport], debate_summary: str = "") -> str:
    """⑨ 토론이 열렸나 생략됐나. 생략이면 그 이유(가중합 차이)를 숫자로.

    토론은 찬반이 팽팽할 때만 열린다. 생략된 날 "반대 논거가 없다"가 아니라
    "한쪽이 우세해서 안 열었다"임을 밝혀야 독자가 리스크 섹션을 제대로 읽는다.
    """
    if debate_summary:
        return "Bull/Bear 토론 진행됨 — 아래 리스크에 반대 논거 반영"
    try:
        from src.agents.debate import DEBATE_THRESHOLD
        buy = sum(r.confidence for r in agents if r.recommendation == "BUY")
        sell = sum(r.confidence for r in agents if r.recommendation == "SELL")
        gap = abs(buy - sell)
        head = f"토론 생략 — BUY 가중합 {buy:.2f} vs SELL {sell:.2f}, 차이 {gap:.2f}"
        if gap >= DEBATE_THRESHOLD:
            # 설계상 생략: 한쪽이 충분히 우세해 토론할 쟁점이 없다
            return (f"{head} ≥ {DEBATE_THRESHOLD} (한쪽 우세). "
                    f"반대 논거는 리스크 항목에만 있다")
        # 차이가 작은데도 안 열렸다 = 한 진영에 에이전트가 없거나 토론이 실패한 것.
        # 이 경우는 "쟁점이 없어서"가 아니므로 구분해서 알린다.
        return (f"{head} < {DEBATE_THRESHOLD}인데도 토론이 열리지 않았다 — "
                f"한쪽 진영에 투표자가 없었거나 토론 호출이 실패했다. 반대 논거가 약할 수 있다")
    except Exception:
        return "토론 생략 (사유 확인 불가)"


def earnings_window_note(tickers: list[str], days_ahead: int = 14) -> Optional[str]:
    """⑪ 보유기간 안의 실적 발표·공시 일정.

    krx_market.get_earnings_calendar를 쓴다. DART가 간헐적으로 끊기므로(CR-3)
    실패하면 None을 돌려 줄을 뺀다 — 리포트 발행이 여기 묶이면 안 된다.
    """
    try:
        from src.mcp_servers.krx_market.server import get_earnings_calendar
        data = get_earnings_calendar(days_ahead=days_ahead)
        if not isinstance(data, dict) or data.get("error"):
            return None
        lines = []
        for d in (data.get("upcoming_deadlines") or [])[:2]:
            if isinstance(d, dict) and d.get("date"):
                lines.append(f"{d['date']} {d.get('description', '')}".strip())
        recent = data.get("recent_disclosures") or []
        hits = [r for r in recent if isinstance(r, dict) and str(r.get("stock_code", "")) in set(tickers)]
        if hits:
            lines.append(f"오늘 종목 관련 실적 공시 {len(hits)}건")
        return " · ".join(lines) if lines else None
    except Exception as e:
        logger.warning(f"[report] 실적 일정 조회 실패: {e}")
        return None


# 맨 위 요약 한 줄에 쓸 근거를 고른다.
#
# [왜 필요한가 — 2026-09-15 실제 발행본을 보고 추가]
#   chief의 reasoning[0]은 대개 "[투표 집계] BUY 가중합 0.00, SELL 가중합 1.40..."으로
#   시작한다. 가중합은 판단에 이른 계산 과정이지 판단 자체가 아니다. 그런데 이것이
#   리포트 맨 위 "최종 판단" 줄에 그대로 실려, 아침에 5초 보는 자리가 숫자 나열이 됐다.
#   게다가 110자에서 잘려 "SELL 가중합(1.40)이" 처럼 문장 중간에서 끊겼다.
#   집계는 종목 상세(펼침)에 이미 전문이 있으므로, 요약 줄에서는 결론을 고른다.
_TALLY_MARKERS = ("투표 집계", "투표집계", "가중합")
_CONCLUSION_MARKERS = ("[결론]", "결론:", "[종합]")


def headline_reason(final: Optional[AnalysisReport], limit: int = 110) -> str:
    """요약 줄에 쓸 근거 한 문장. 집계 줄은 건너뛰고 문장 경계에서 자른다."""
    if final is None or not final.reasoning:
        return ""
    picked = next((r for r in final.reasoning
                   if any(m in r[:12] for m in _CONCLUSION_MARKERS)), None)
    if picked is None:
        picked = next((r for r in final.reasoning
                       if not any(m in r[:20] for m in _TALLY_MARKERS)), None)
    if picked is None:
        picked = final.reasoning[0]
    # 말머리는 떼고 내용만. chief는 "[결론]"뿐 아니라 "**결론:**"으로도 쓴다
    # — 2026-09-14 발행본에서 "**결론:** 기술적 분석..."이 그대로 실렸다.
    for m in _CONCLUSION_MARKERS:
        for cand in (m, f"**{m}**", f"**{m.rstrip(':')}:**"):
            if picked.startswith(cand):
                picked = picked[len(cand):].strip()
                break
        else:
            continue
        break
    if len(picked) <= limit:
        return picked
    cut = picked[:limit]
    # 마지막 문장 경계에서 끊는다. 경계가 너무 앞이면 그냥 …을 붙인다.
    end = max(cut.rfind("다. "), cut.rfind(". "), cut.rfind("다."))
    return (cut[:end + 2].strip() if end >= limit // 2 else cut.rstrip() + "…")


# 포지션 상태 판정 — 한 곳에서만 한다.
#
# [왜 한 곳인가 — 2026-09-15 발행본에서 드러난 문제]
#   000500이 현재가 241,000 / 손절 241,300으로 이미 손절선 아래인데,
#   "오늘 한 줄"에는 "1개 손절·목표선 도달"이 뜨고 정작 결정하는 자리인
#   액션 플랜에는 "⏳ 모니터링 (오늘 분석 대상 아님)"만 나왔다.
#   두 곳이 각자 판정했기 때문이다. 같은 함수를 쓰면 어긋날 수 없다.
#
# [등급]
#   reached  손절가 이하 또는 목표가 이상 — 오늘 결정이 필요하다
#   near     그 선까지 NEAR_PCT 이내 — 내일 결정이 필요할 수 있다
#   overdue  D+10(HOLDING_ALERT_DAYS) 경과 — 시간이 지났을 뿐 가격 신호는 아니다
#   센 것 하나만 돌려준다. 셋을 다 붙이면 무엇이 급한지 안 보인다.
NEAR_PCT = 2.0

ALERT_ICON = {"reached": "🔴", "near": "🟠", "overdue": "⏰"}


# 밸류에이션 지표의 상식 범위 — 벗어나면 근거에서 빼고 "데이터 이상"으로 표시한다.
#
# [왜 — 2026-09-15 발행본]
#   삼성SDI PER 1161.57배가 "매우 높은 수준이며, 이는 고평가를 나타냅니다"라는
#   근거로 그대로 실렸다. PER 1161배는 "비싸다"가 아니라 "순이익이 거의 0"이라는
#   뜻이고, 그 구간에서 PER은 밸류에이션 신호가 아니다. 숫자가 있으니 판단이
#   있는 것처럼 보이는 것이 문제다.
#
# [범위 근거]
#   PER  0 초과 200 이하 — 한국 상장사 PER 중앙값은 10~15배대다. 200배는 순이익이
#        시총의 0.5% 미만이라는 뜻으로, 이익 사이클 판단은 PBR·EPS로 해야 한다.
#        음수·0은 적자라 PER 정의 자체가 성립하지 않는다.
#   PBR  0 초과 20 이하 — 자산 대비 20배는 자산이 사실상 의미 없는 사업 구조다.
#   배당수익률 0 이상 30 이하 — 30% 초과는 특별배당이거나 데이터 오류다.
#
#   근본 해결은 fundamental_analyst가 이상값을 LLM에 넘기지 않는 것이다.
#   여기(포맷 단계)는 이미 들어온 값이 리포트로 새어 나가는 것을 막는 마지막 관문이다.
VALUATION_RANGES = {
    "PER": (0.0, 200.0),
    "PBR": (0.0, 20.0),
    "배당수익률": (0.0, 30.0),
}
_METRIC_RE = re.compile(
    r"(PER|PBR|배당수익률)\s*(?:은|이|는|가|:)?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:배|%)"
)


# 종목과 무관하게 하루 1회 캐시되는 에이전트 — 종목마다 같은 문단이 반복된다.
#
# [왜 분리 — 2026-09-15 발행본]
#   4종목 리포트에서 macro_economist와 us_market_specialist의 문단이 각각 4번씩,
#   글자 하나 다르지 않게 반복됐다. 읽는 사람은 세 번째부터 건너뛰고, 그러다
#   종목 고유 근거까지 같이 건너뛴다.
#   투표 집계에서는 빼지 않는다 — 이들도 가중합에 실제로 들어가는 투표자다.
#   표시만 상단 "오늘의 시장 배경"으로 한 번 모은다.
MARKET_WIDE_AGENTS = ("macro_economist", "us_market_specialist")


def market_background(per_ticker_agents: list[AnalysisReport]) -> list[AnalysisReport]:
    """시장 공통 에이전트 보고서를 에이전트당 1건만 남겨 돌려준다."""
    seen: dict[str, AnalysisReport] = {}
    for r in per_ticker_agents:
        if r.agent_name in MARKET_WIDE_AGENTS and r.agent_name not in seen:
            seen[r.agent_name] = r
    return [seen[n] for n in MARKET_WIDE_AGENTS if n in seen]


def extreme_metrics(text: str) -> list[str]:
    """문장에 범위 밖 밸류에이션 수치가 있으면 ["PER 1161.57배"] 형태로."""
    out = []
    for name, raw in _METRIC_RE.findall(text or ""):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        low, high = VALUATION_RANGES[name]
        if not (low < val <= high):
            out.append(f"{name} {raw}{'%' if name == '배당수익률' else '배'}")
    return out


def filter_extreme_values(items: list[str]) -> tuple[list[str], list[str]]:
    """(살아남은 항목, 제외된 이상값 목록).

    항목 하나를 통째로 뺀다 — 문장 안에서 숫자만 지우면 "PER은 매우 높은 수준"이라는
    결론만 남아서 오히려 더 위험하다.
    """
    kept, dropped = [], []
    for it in items or []:
        bad = extreme_metrics(it)
        if bad:
            dropped.extend(bad)
        else:
            kept.append(it)
    return kept, dropped


def extreme_value_note(dropped: list[str]) -> Optional[str]:
    """제외된 것을 숨기지 않고 밝힌다. 조용히 지우면 왜 근거가 적은지 알 수 없다."""
    if not dropped:
        return None
    uniq = list(dict.fromkeys(dropped))
    return ("⚠️ 데이터 이상으로 근거에서 제외: " + ", ".join(uniq)
            + " — 상식 범위 밖 값이라 밸류에이션 신호로 쓸 수 없다")


def seed_pnl_note(portfolio_summary: Optional[dict]) -> str:
    """시드 손익 한 줄. 실현과 미실현을 나눠 쓴다.

    [왜 나누나 — 2026-09-15 발행본]
      보유 종목이 -3.12%, -5.12%인데 "누적 수익 +0.00% of seed"로 나왔다.
      cumulative_pnl_seed_pct는 청산분만 합산하므로 계산은 맞았지만,
      "누적 수익"이라는 이름 때문에 손실이 없는 것처럼 읽혔다.
      합치지 않는 이유: 확정된 손익과 되돌릴 수 있는 손익은 성격이 다르다.
    """
    ps = portfolio_summary or {}
    realized = ps.get("cumulative_pnl_seed_pct", 0.0) or 0.0
    unrealized = ps.get("unrealized_pnl_seed_pct")
    if unrealized is None:                     # 옛 호출부 호환
        return f"실현 손익: {realized:+.2f}% of seed"
    open_n = len(ps.get("positions") or [])
    priced = ps.get("priced_positions", open_n)
    miss = "" if priced >= open_n else f" (현재가 미수신 {open_n - priced}종목 제외)"
    return (f"실현 손익: {realized:+.2f}% of seed  |  "
            f"미실현: {unrealized:+.2f}% of seed{miss}")


def real_risk_factors(reports: list[AnalysisReport]) -> list[str]:
    """리스크 목록에 쓸 항목만. 중복 제거 순서는 유지한다.

    [왜 거르나 — 2026-09-14 발행본]
      "에이전트 오류로 분석 불가"가 오늘의 주요 리스크 5위에 올라왔다.
      confidence 0.0 폴백 보고서의 risk_factors인데, 그 보고서는 스스로
      "신뢰할 수 없으므로 Quality Gate에서 자동 제외됨"이라고 적는다.
      시스템 상태는 건강도 callout이 맡는다. 여기는 시장 리스크만 둔다.
    """
    seen: set[str] = set()
    out: list[str] = []
    for r in reports:
        if r is None or r.confidence == 0.0:
            continue
        for risk in r.risk_factors:
            if risk not in seen:
                seen.add(risk)
                out.append(risk)
    return out


def position_alert(p: dict) -> Optional[tuple[str, str]]:
    """(등급, 사람이 읽을 문구). 아무 상태도 아니면 None.

    position_tracker의 alert 필드(stop_loss/target)는 자동 청산이 쓰는 값이라
    건드리지 않고 읽기만 한다. near는 여기서만 계산한다.
    """
    cur = p.get("current_price")
    alert = p.get("alert")
    if alert == "stop_loss":
        return ("reached", f"손절선 도달 (현재 {_fmt_money(cur)}원 ≤ 손절 {_fmt_money(p.get('stop_loss_price'))}원)")
    if alert == "target":
        return ("reached", f"목표선 도달 (현재 {_fmt_money(cur)}원 ≥ 목표 {_fmt_money(p.get('target_price'))}원)")

    if cur:
        sl = p.get("stop_loss_price")
        if sl and 0 < (cur / sl - 1.0) * 100 <= NEAR_PCT:
            return ("near", f"손절선까지 {(cur / sl - 1.0) * 100:.1f}% (현재 {_fmt_money(cur)}원)")
        tp = p.get("target_price")
        if tp and 0 < (1.0 - cur / tp) * 100 <= NEAR_PCT:
            return ("near", f"목표선까지 {(1.0 - cur / tp) * 100:.1f}% (현재 {_fmt_money(cur)}원)")

    if p.get("overdue"):
        days = p.get("holding_days")
        return ("overdue", f"D+10 경과 (보유 {days}일)" if days else "D+10 경과")
    return None


def alert_counts(positions: list[dict]) -> dict[str, int]:
    """등급별 개수. today_line과 액션 플랜이 같은 숫자를 쓰게 한다."""
    out = {"reached": 0, "near": 0, "overdue": 0}
    for p in positions:
        hit = position_alert(p)
        if hit:
            out[hit[0]] += 1
    return out


def today_line(per_ticker: list[dict], portfolio_summary: Optional[dict],
               healthy: bool = True) -> str:
    """맨 위 한 줄. 이것만 읽어도 오늘 뭘 할지 알아야 한다.

    형식: "오늘: 신규 2종목 · 보유 3종목 중 1개 손절선 근접 · 시스템 정상"
    """
    buys = [r for r in per_ticker
            if (r.get("chief_report") is not None
                and r["chief_report"].recommendation == "BUY"
                and not getattr(r["chief_report"], "needs_review", False))]
    parts = [f"신규 {len(buys)}종목" if buys else "신규 추천 없음"]

    positions = (portfolio_summary or {}).get("positions") or []
    if positions:
        # [2026-09-15] 액션 플랜과 같은 판정을 쓴다 (alert_counts → position_alert)
        counts = alert_counts(positions)
        note = f"보유 {len(positions)}종목"
        flags = []
        if counts["reached"]:
            flags.append(f"{counts['reached']}개 손절·목표선 도달")
        if counts["near"]:
            flags.append(f"{counts['near']}개 근접")
        if counts["overdue"]:
            flags.append(f"{counts['overdue']}개 D+10 경과")
        if flags:
            note += " 중 " + ", ".join(flags)
        parts.append(note)
    else:
        parts.append("보유 없음")

    parts.append("시스템 정상" if healthy else "⛔ 시스템 이상 — 아래 확인")
    return "오늘: " + " · ".join(parts)
