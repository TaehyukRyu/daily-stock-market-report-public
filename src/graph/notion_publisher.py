"""
src/graph/notion_publisher.py

Notion 발행 모듈 v4.1 — L5 Report & Publishing (7섹션 구조)

블록 빌더 구조:
  - 헤더 callout (Daily stock market report | 날짜 | 시장 | 전략)
  - 섹션 1: 💼 내 포트폴리오 현황
  - 섹션 2: 🎯 오늘의 액션 플랜
  - 섹션 3: 📊 시장 지표
  - 섹션 4: 🤖 AI 분석단 의견
  - 섹션 5: 📋 에이전트별 상세 분석 (toggle)
  - 섹션 6: ⚠️ 오늘의 주요 리스크 (numbered list)
  - 면책 callout

구조화 데이터 없으면 _markdown_to_blocks() fallback (v3.0 호환).

[Notion API 제약]
  - rich_text content 최대 2000자 → 초과 시 자동 분할
  - children 한 번에 최대 100블록 → 초과 시 append_block_children으로 추가
"""

import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from notion_client import Client, APIResponseError
from dotenv import load_dotenv

from src.schemas.agent_output import AnalysisReport
from src.graph.report_formatter import (
    REGIME_KR, AGENT_NAME_KR, SIGNAL_KR, STRATEGY_KR,
    # 모듈 레벨 헬퍼가 쓰는 것들 — 함수 안 import로는 닿지 않는다
    MARKET_WIDE_AGENTS, filter_extreme_values, extreme_value_note,
    position_alert, seed_pnl_note, ALERT_ICON, headline_reason, real_risk_factors,
)

load_dotenv()


# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────

NOTION_API_KEY     = os.getenv("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

FALLBACK_DIR       = Path("data/reports")
MAX_TEXT_LENGTH    = 2000
MAX_BLOCKS_PER_REQ = 100

KST          = timezone(timedelta(hours=9))
SORT_ORDER   = {"SELL": 0, "HOLD": 1, "BUY": 2}


# ─────────────────────────────────────────────────────────
# rich_text 유틸
# ─────────────────────────────────────────────────────────

def _rich_text(content: str, bold: bool = False) -> list[dict]:
    if not content:
        return [{"type": "text", "text": {"content": ""}, "annotations": {"bold": False}}]
    chunks = [content[i: i + MAX_TEXT_LENGTH] for i in range(0, len(content), MAX_TEXT_LENGTH)]
    return [
        {"type": "text", "text": {"content": c}, "annotations": {"bold": bold}}
        for c in chunks
    ]


def _parse_bold(text: str) -> list[dict]:
    parts  = re.split(r"\*\*(.+?)\*\*", text)
    result = []
    for i, part in enumerate(parts):
        if not part:
            continue
        result.extend(_rich_text(part, bold=(i % 2 == 1)))
    return result if result else _rich_text(text)


# ─────────────────────────────────────────────────────────
# Block 생성 유틸
# ─────────────────────────────────────────────────────────

def _block(block_type: str, rich_text: list[dict]) -> dict:
    return {"object": "block", "type": block_type, block_type: {"rich_text": rich_text}}


def _heading(level: int, text: str) -> dict:
    t = f"heading_{level}"
    return {"object": "block", "type": t, t: {"rich_text": _rich_text(text)}}


def _bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": _parse_bold(text)}}


def _numbered(text: str) -> dict:
    return {"object": "block", "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": _parse_bold(text)}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _callout(text: str, emoji: str = "⚠️", color: str = "yellow_background") -> dict:
    return {
        "object": "block",
        "type":   "callout",
        "callout": {
            "rich_text": _rich_text(text),
            "icon":      {"type": "emoji", "emoji": emoji},
            "color":     color,
        },
    }


def _toggle(title: str, children: list[dict]) -> dict:
    return {
        "object": "block",
        "type":   "toggle",
        "toggle": {
            "rich_text": _rich_text(title),
            "children":  children,
        },
    }


def _paragraph(text: str) -> dict:
    return _block("paragraph", _parse_bold(text))


# ─────────────────────────────────────────────────────────
# 포맷 헬퍼
# ─────────────────────────────────────────────────────────

def _fmt_money(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def _scalar(val):
    if val is None:
        return None
    if isinstance(val, (int, float, str)):
        return val
    if isinstance(val, dict):
        for k in ("value", "current", "price", "rate", "level",
                 "latest_close", "close", "vix"):
            if k in val and isinstance(val[k], (int, float, str)):
                return val[k]
    return None


# ─────────────────────────────────────────────────────────
# v4.1 섹션 블록 빌더
# ─────────────────────────────────────────────────────────

def _build_portfolio_blocks(portfolio_summary: Optional[dict]) -> list[dict]:
    """섹션 1: 💼 내 포트폴리오 현황"""
    blocks: list[dict] = [_heading(2, "💼 내 포트폴리오 현황")]

    if not portfolio_summary or not portfolio_summary.get("positions"):
        blocks.append(_paragraph("보유 포지션 없음 (첫 거래 대기 중)"))
        return blocks

    invested  = portfolio_summary.get("invested_pct", 0.0)
    available = portfolio_summary.get("available_pct", 0.0)
    # [2026-09-15] 색은 실현+미실현 합으로 정한다. 실현만 보면 보유 손실이
    #   아무리 커도 청산 전까지 계속 노란불이라 경고 기능을 못 한다.
    total = ((portfolio_summary.get("cumulative_pnl_seed_pct") or 0.0)
             + (portfolio_summary.get("unrealized_pnl_seed_pct") or 0.0))
    if total > 0:
        color = "green_background"
    elif total < 0:
        color = "red_background"
    else:
        color = "yellow_background"

    blocks.append(_callout(
        f"투자중: {invested:.1f}%  |  가용: {available:.1f}%\n" + seed_pnl_note(portfolio_summary),
        emoji="💼",
        color=color,
    ))

    for p in portfolio_summary["positions"]:
        alert  = p.get("alert")
        prefix = "⚠️ " if alert == "stop_loss" else ("🎯 " if alert == "target" else "")
        # CR-5: 보유 기간 경과 알림. 청산은 하지 않는다.
        if p.get("overdue"):
            prefix = "⏰ " + prefix
        name   = f"{p['ticker_name']} ({p['ticker']})" if p.get("ticker_name") else p["ticker"]
        alloc  = f"{p['allocation_pct']:.0f}%" if p.get("allocation_pct") is not None else "-"
        pnl    = f"{p['position_pnl_pct']:+.2f}%" if p.get("position_pnl_pct") is not None else "-"
        sl     = _fmt_money(p.get("stop_loss_price"))
        holding = f"{p['holding_days']}일" if p.get("holding_days") is not None else "-"

        # [2026-09-15] 익절1을 뺐다. 배분은 남긴다 — 제안이 아니라 실제 체결 비중이다.
        blocks.append(_bullet(
            f"{prefix}**{name}** · 수익 {pnl} · 배분 {alloc} · 보유 {holding}\n"
            f"매수 {p.get('entry_date') or '-'} {_fmt_money(p.get('entry_price'))}원 → "
            f"현재 {_fmt_money(p.get('current_price'))}원 · 손절 {sl}원"
        ))

    closed = portfolio_summary.get("closed_recent") or []
    if closed:
        blocks.append(_heading(3, "📌 청산 이력 (최근 5건)"))
        for c in closed:
            pos_pnl = f"{c['position_pnl_pct']:+.2f}%" if c.get("position_pnl_pct") is not None else "-"
            seed_pnl = f"{c['seed_pnl_pct']:+.2f}%" if c.get("seed_pnl_pct") is not None else "-"
            holding = f"{c['holding_days']}일" if c.get("holding_days") is not None else "-"
            blocks.append(_bullet(
                f"**{c['ticker']}** | 매수 {_fmt_money(c.get('entry_price'))} → "
                f"매도 {_fmt_money(c.get('close_price'))} | "
                f"포지션 {pos_pnl} | 시드 {seed_pnl} | {holding} 보유"
            ))

    return blocks


def _build_action_plan_blocks(
    final: Optional[AnalysisReport],
    portfolio_summary: Optional[dict],
    ticker: str,
) -> list[dict]:
    """섹션 2: 🎯 오늘의 액션 플랜"""
    blocks: list[dict] = [_heading(2, "🎯 오늘의 액션 플랜")]

    # 기존 포지션 처리
    blocks.append(_heading(3, "기존 포지션 처리"))
    positions = (portfolio_summary or {}).get("positions") or []
    if positions:
        for p in positions:
            name = f"{p['ticker_name']} ({p['ticker']})" if p.get("ticker_name") else p["ticker"]
            if p["ticker"] != ticker:
                action = "⏳ 모니터링 (오늘 분석 대상 아님)"
                reason = ""
            elif final is None:
                action = "⏳ 보류"
                reason = ""
            elif final.recommendation == "HOLD":
                action = "🟡 유지"
                reason = final.reasoning[0][:100] if final.reasoning else ""
            elif final.recommendation == "SELL":
                action = "🔴 매도 검토"
                reason = final.reasoning[0][:100] if final.reasoning else ""
            else:
                action = "🟢 추가 매수 검토"
                reason = final.reasoning[0][:100] if final.reasoning else ""
            text = f"**{name}** — {action}"
            if reason:
                text += f" | {reason}"
            blocks.append(_bullet(text))
    else:
        blocks.append(_paragraph("보유 포지션 없음"))

    # 신규 매수 추천
    available_pct = (portfolio_summary or {}).get("available_pct", 100.0)
    blocks.append(_heading(3, f"신규 매수 추천 (가용: {available_pct:.0f}%)"))

    if final and final.recommendation == "BUY" and final.entry_price is not None:
        ep   = final.entry_price
        sl   = final.stop_loss
        sl_pct = final.stop_loss_pct
        tp1  = final.take_profit_1
        tp2  = final.take_profit_2
        rr   = final.rr_ratio
        pos  = final.position_size_pct
        wks  = final.holding_period_weeks

        lines = [f"🟢 {ticker} 매수 추천"]
        lines.append(f"진입가: {_fmt_money(ep)}원")
        if pos is not None:
            lines.append(f"배분: 시드의 {pos:.0f}%")
        if sl is not None and sl_pct is not None:
            lines.append(f"손절: {_fmt_money(sl)}원 ({sl_pct:+.1f}%)")
        if tp1 is not None:
            lines.append(f"1차 익절: {_fmt_money(tp1)}원")
        if tp2 is not None:
            lines.append(f"2차 익절: {_fmt_money(tp2)}원")
        if wks is not None:
            lines.append(f"예상 보유: {wks}주")
        if rr is not None:
            lines.append(f"R:R: {rr:.1f}")
        blocks.append(_callout("\n".join(lines), emoji="🟢", color="green_background"))

        # 현금 부족 경고
        if portfolio_summary and pos is not None and pos > available_pct:
            blocks.append(_callout(
                f"⚠️ 추천 배분({pos:.0f}%)이 가용 현금({available_pct:.0f}%)을 초과합니다",
                emoji="⚠️",
                color="red_background",
            ))
    else:
        blocks.append(_paragraph("오늘 신규 진입 추천 없음"))

    return blocks


def _build_market_data_blocks(market_data: Optional[dict]) -> list[dict]:
    """섹션 3: 📊 시장 지표"""
    blocks: list[dict] = [_heading(2, "📊 시장 지표")]

    if not market_data:
        blocks.append(_paragraph("시장 지표 데이터 없음"))
        return blocks

    indicator_map = [
        ("kospi",        "KOSPI",         None),
        ("us_10y_yield", "미 10년 금리",  "rate"),
        ("vix",          "공포지수(VIX)", "vix"),
        ("usd_krw",      "원/달러 환율",  None),
        ("wti",          "WTI 유가",      None),
    ]

    found_any = False
    for key, name, interp_kind in indicator_map:
        if key not in market_data:
            continue
        raw = _scalar(market_data.get(key))
        if raw is None:
            continue
        try:
            num = float(raw)
            value_str = f"{num:,.2f}" if num < 100 else f"{num:,.0f}"
        except (TypeError, ValueError):
            blocks.append(_bullet(f"**{name}**: {raw}"))
            found_any = True
            continue

        interp = ""
        if interp_kind == "vix":
            interp = "공포" if num > 30 else ("주의" if num > 20 else "안정적")
        elif interp_kind == "rate":
            interp = "주식 부담↑" if num > 4.5 else "보통"

        text = f"**{name}**: {value_str}" + (f" — {interp}" if interp else "")
        blocks.append(_bullet(text))
        found_any = True

    if not found_any:
        blocks.append(_paragraph("시장 지표 데이터 없음"))

    return blocks


def _ticker_label(ticker: str) -> str:
    """종목명(코드) 형식으로 라벨링. 조회 실패 시 코드만."""
    if not ticker:
        return ""
    try:
        from pykrx import stock
        name = stock.get_market_ticker_name(ticker)
        if name:
            return f"{name}({ticker})"
    except Exception:
        pass
    return ticker


def _build_chief_summary_line(chief: AnalysisReport, ticker: str) -> str:
    """chief_strategist 최종 판단 한 줄 생성 (recommendation별 포맷 분기)."""
    label = _ticker_label(ticker)
    sig   = SIGNAL_KR.get(chief.recommendation, chief.recommendation)
    conf  = f"{chief.confidence:.0%}"
    head  = f"**최종 판단** — **{label}** | " if label else "**최종 판단** — "
    parts = [f"{head}**{sig} {conf}**"]

    # [2026-09-15] 배분(시드의 N%)·목표가를 뺐다. 배분은 표본 부족이면 전부 5% 고정이고
    #   목표가는 R:R 2.0 항등식이라, 둘 다 숫자지만 판단 정보가 아니다. DB에는 남는다.
    if chief.recommendation == "BUY":
        if chief.entry_price is not None:
            parts.append(f"진입가 {_fmt_money(chief.entry_price)}원")
        if chief.stop_loss is not None:
            parts.append(f"제안 손절가 {_fmt_money(chief.stop_loss)}원")

    return " | ".join(parts)


def _build_agent_summary_blocks(
    agents: list[AnalysisReport],
    chief_report: Optional[AnalysisReport],
    qualified_names: set[str],
    ticker: str = "",
    heading_level: int = 2,
) -> list[dict]:
    """섹션 4: 🤖 AI 분석단 의견

    heading_level: 통합 리포트에서는 이 블록이 종목 toggle 안으로 들어간다.
      토글 안의 heading_2는 페이지 제목급으로 커져서 위계가 뒤집혀 보이므로 3을 넘긴다.
    """
    blocks: list[dict] = [_heading(heading_level, "🤖 AI 분석단 의견")]

    for r in sorted(agents, key=lambda x: SORT_ORDER.get(x.recommendation, 1)):
        name = AGENT_NAME_KR.get(r.agent_name, r.agent_name)
        sig  = SIGNAL_KR.get(r.recommendation, r.recommendation)
        # [2026-09-15] 이상값이 든 문장은 요약에도 쓰지 않는다. 상세에서만 빼면
        #   정작 사람이 먼저 읽는 한 줄에 "PER 1161.57배"가 그대로 남는다.
        _clean, _ = filter_extreme_values(r.reasoning)
        reason = ((_clean or r.reasoning or ["-"])[0])[:80]
        if r.confidence == 0.0:
            note = " ⚠️데이터없음"
        elif r.confidence < 0.6:
            note = " ⚠️참고용"
        else:
            note = ""
        blocks.append(_bullet(
            f"**{name}** — {sig} {r.confidence:.0%} | {reason}{note}"
        ))

    if chief_report:
        blocks.append(_divider())
        blocks.append(_bullet(_build_chief_summary_line(chief_report, ticker)))

    return blocks


def _build_rationale_toggle(
    agents: list[AnalysisReport],
    chief_report: Optional[AnalysisReport],
    qualified_names: set[str],
) -> dict:
    """섹션 5: 📋 에이전트별 상세 분석 (toggle 접기)"""
    children: list[dict] = []

    # [2026-09-15] 시장 공통 에이전트(macro·us_market)는 상단 "오늘의 시장 배경"으로
    #   옮겼다. 4종목이면 같은 문단이 4번 반복돼 종목 고유 근거까지 묻힌다.
    detail_agents = [r for r in agents if r.agent_name not in MARKET_WIDE_AGENTS]
    for r in sorted(detail_agents, key=lambda x: SORT_ORDER.get(x.recommendation, 1)):
        gate = "✅" if r.agent_name in qualified_names else "⚠️ (QG 폴백)"
        sig  = SIGNAL_KR.get(r.recommendation, r.recommendation)
        name = AGENT_NAME_KR.get(r.agent_name, r.agent_name)
        children.append(_heading(3, f"{gate} {name} ({r.agent_name}) {sig} (신뢰도: {r.confidence:.2f})"))
        kept_reason, drop_a = filter_extreme_values(r.reasoning)
        kept_basis,  drop_b = filter_extreme_values(r.prediction_basis)
        for reason in kept_reason:
            children.append(_bullet(reason))
        if kept_basis:
            children.append(_paragraph("**근거:**"))
            for basis in kept_basis:
                children.append(_bullet(basis))
        note = extreme_value_note(drop_a + drop_b)
        if note:
            children.append(_callout(note.replace("⚠️ ", ""), emoji="🚫", color="orange_background"))
        if r.risk_factors:
            children.append(_callout("리스크: " + " / ".join(r.risk_factors), emoji="⚠️"))
        children.append(_divider())

    if chief_report:
        children.append(_heading(3, "🎯 종합 판단 (chief_strategist)"))
        for reason in chief_report.reasoning:
            children.append(_bullet(reason))
        if chief_report.selection_rationale:
            children.append(_paragraph(f"**선정 종목:** {chief_report.selection_rationale}"))

    return _toggle("📋 에이전트별 상세 분석 (클릭하여 펼치기)", children)


def _build_risk_blocks(
    agents: list[AnalysisReport],
    chief_report: Optional[AnalysisReport],
) -> list[dict]:
    """섹션 6: ⚠️ 오늘의 주요 리스크 (top 5)"""
    blocks: list[dict] = [_heading(2, "⚠️ 오늘의 주요 리스크")]

    unique_risks = real_risk_factors(agents + ([chief_report] if chief_report else []))

    if not unique_risks:
        blocks.append(_paragraph("주요 리스크 없음"))
        return blocks

    for risk in unique_risks[:5]:
        blocks.append(_numbered(risk))

    return blocks


# ─────────────────────────────────────────────────────────
# v4.1 통합 빌더
# ─────────────────────────────────────────────────────────

def build_v4_blocks(
    ticker:            str,
    regime:            str,
    strategy:          str,
    chief_report:      Optional[AnalysisReport]       = None,
    qualified_reports: Optional[list[AnalysisReport]] = None,
    all_reports:       Optional[list[AnalysisReport]] = None,
    debate_summary:    str                             = "",
    error_log:         Optional[list[str]]            = None,
    portfolio_summary: Optional[dict]                  = None,
    market_data:       Optional[dict]                  = None,
) -> list[dict]:
    """v4.1 7섹션 Notion 블록 리스트 생성."""
    agents          = [r for r in (all_reports or []) if r.agent_name != "chief_strategist"]
    qualified_names = {r.agent_name for r in (qualified_reports or [])}

    blocks: list[dict] = []

    # 헤더 callout
    now     = datetime.now(KST)
    weekday = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
    regime_label   = REGIME_KR.get(regime.upper(), regime)
    strategy_label = STRATEGY_KR.get(strategy, strategy)
    blocks.append(_callout(
        f"Daily stock market report | {now.strftime('%Y.%m.%d')} ({weekday}) "
        f"{now.strftime('%H:%M')} KST | 시장: {regime_label} | 전략: {strategy_label}",
        emoji="📈",
        color="blue_background",
    ))
    blocks.append(_divider())

    # 섹션 1: 포트폴리오
    blocks += _build_portfolio_blocks(portfolio_summary)
    blocks.append(_divider())

    # 섹션 2: 액션 플랜
    blocks += _build_action_plan_blocks(chief_report, portfolio_summary, ticker)
    blocks.append(_divider())

    # 섹션 3: 시장 지표
    blocks += _build_market_data_blocks(market_data)
    blocks.append(_divider())

    # 섹션 4: AI 분석단 의견
    blocks += _build_agent_summary_blocks(agents, chief_report, qualified_names, ticker=ticker)
    blocks.append(_divider())

    # 섹션 5: 에이전트별 상세 분석 (toggle)
    if agents or chief_report:
        blocks.append(_build_rationale_toggle(agents, chief_report, qualified_names))
        blocks.append(_divider())

    # 섹션 6: 리스크
    blocks += _build_risk_blocks(agents, chief_report)
    blocks.append(_divider())

    return blocks


# ─────────────────────────────────────────────────────────
# 마크다운 → Block fallback (v3.0 호환)
# ─────────────────────────────────────────────────────────

_DIVIDER_RE = re.compile(r"^[─\-=]{3,}$")


def _markdown_to_blocks(text: str) -> list[dict]:
    blocks   = []
    lines    = text.split("\n")
    prev_emp = False

    for line in lines:
        s = line.strip()
        if not s:
            if not prev_emp:
                blocks.append(_block("paragraph", _rich_text("")))
            prev_emp = True
            continue
        prev_emp = False

        if s.startswith("# ") and not s.startswith("## "):
            blocks.append(_heading(1, s[2:].strip()))
        elif s.startswith("## ") and not s.startswith("### "):
            blocks.append(_heading(2, s[3:].strip()))
        elif s.startswith("### "):
            blocks.append(_heading(3, s[4:].strip()))
        elif s.startswith("- "):
            blocks.append(_bullet(s[2:].strip()))
        elif _DIVIDER_RE.match(s):
            blocks.append(_divider())
        elif s.startswith("⚠️"):
            blocks.append(_callout(s, emoji="⚠️"))
        else:
            blocks.append(_block("paragraph", _parse_bold(s)))

    return blocks


# ─────────────────────────────────────────────────────────
# 통합 리포트 빌더 (하루 1회 발행용)
# ─────────────────────────────────────────────────────────

def _build_combined_action_plan_blocks(
    per_ticker_results: list[dict],
    portfolio_summary:  Optional[dict],
) -> list[dict]:
    """통합 액션 플랜: 기존 포지션 + 신규 매수 추천 표."""
    blocks: list[dict] = [_heading(2, "🎯 오늘의 액션 플랜")]

    analyzed_map = {r["ticker"]: r.get("chief_report") for r in per_ticker_results}

    # ── 기존 포지션 처리 ─────────────────────────────────────
    blocks.append(_heading(3, "기존 포지션 처리"))
    positions = (portfolio_summary or {}).get("positions") or []
    if positions:
        # [2026-09-15] 가격 신호를 맨 앞에 세운다. 발행본에서 이미 손절선 아래인
        #   000500이 "⏳ 모니터링 (오늘 분석 대상 아님)"으로만 나왔다.
        #   판정은 report_formatter.position_alert 하나만 쓴다 — "오늘 한 줄"과 같은 함수.
        for p in positions:
            name = f"{p['ticker_name']} ({p['ticker']})" if p.get("ticker_name") else p["ticker"]
            chief = analyzed_map.get(p["ticker"])
            if chief is None:
                action, reason = "⏳ 모니터링", "오늘 분석 대상 아님"
            elif chief.recommendation == "HOLD":
                action, reason = "🟡 유지", headline_reason(chief, 100)
            elif chief.recommendation == "SELL":
                action, reason = "🔴 매도 검토", headline_reason(chief, 100)
            else:
                action, reason = "🟢 추가 매수 검토", headline_reason(chief, 100)
            hit = position_alert(p)
            if hit:
                grade, note = hit
                head = f"{ALERT_ICON[grade]} **{name} — {note}**"
                text = head + "\n" + action + (f" · {reason}" if reason else "")
            else:
                text = f"**{name}** — {action}" + (f" · {reason}" if reason else "")
            blocks.append(_bullet(text))
    else:
        blocks.append(_paragraph("보유 포지션 없음"))

    # ── 신규 매수 추천 (표) ───────────────────────────────────
    available_pct = (portfolio_summary or {}).get("available_pct", 100.0)
    buy_picks = [
        r for r in per_ticker_results
        if r.get("chief_report") is not None
        and r["chief_report"].recommendation == "BUY"
    ]

    # [2026-09-15] 0건인 날 heading조차 없어서 액션 플랜이 통째로 빈 것처럼 보였다.
    #   "오늘 한 줄"에만 적혀 있고 정작 액션 플랜에는 없으니 스크롤하다 놓친다.
    blocks.append(_heading(3, f"신규 매수 추천 (가용: {available_pct:.0f}%)"))
    if not buy_picks:
        blocks.append(_paragraph("오늘 신규 진입 추천 없음"))
    if buy_picks:
        # [2026-09-15] 목표가1(take_profit_1)을 뺐다 — R:R 2.0 항등식에서 나온 값이다.
        #   파이프로 6칸을 이어붙인 헤더 줄도 없앴다. Notion에는 markdown 표가 없어서
        #   "|"가 그대로 글자로 보이고, 좁은 화면에서는 아무 데서나 줄바꿈된다.
        #   종목당 한 줄 · 핵심 3개(진입·손절·확신도)만 남겨 모바일에서 한눈에 읽히게 했다.
        for r in buy_picks:
            chief = r["chief_report"]
            sig   = SIGNAL_KR.get(chief.recommendation, chief.recommendation)
            parts = [f"**{_ticker_label(r['ticker']) or r['ticker']}** {sig} {chief.confidence:.0%}"]
            if chief.entry_price is not None:
                parts.append(f"진입 {_fmt_money(chief.entry_price)}원")
            if chief.stop_loss is not None:
                sl_pct = f" ({chief.stop_loss_pct:+.1f}%)" if chief.stop_loss_pct is not None else ""
                parts.append(f"제안 손절 {_fmt_money(chief.stop_loss)}원{sl_pct}")
            blocks.append(_bullet(" · ".join(parts)))

    return blocks


def _build_per_ticker_section(payload: dict) -> list[dict]:
    """종목별 섹션: heading_2 + agent summary + rationale toggle."""
    ticker          = payload["ticker"]
    chief_report    = payload.get("chief_report")
    qualified       = payload.get("qualified_reports") or []
    all_reports     = payload.get("all_reports") or []
    qualified_names = {r.agent_name for r in qualified}
    agents          = [r for r in all_reports if r.agent_name != "chief_strategist"]

    # 헤더 라인
    if chief_report:
        sig    = SIGNAL_KR.get(chief_report.recommendation, chief_report.recommendation)
        header = f"📈 {ticker} — {sig} ({chief_report.confidence:.0%})"
    else:
        header = f"📈 {ticker} — 분석 실패"

    blocks: list[dict] = [_heading(2, header)]
    blocks += _build_agent_summary_blocks(agents, chief_report, qualified_names, ticker=ticker)

    if agents or chief_report:
        blocks.append(_build_rationale_toggle(agents, chief_report, qualified_names))

    return blocks


def build_v4_blocks_combined(
    per_ticker_results: list[dict],
    regime:             str,
    strategy:           str,
    portfolio_summary:  Optional[dict] = None,
    market_data:        Optional[dict] = None,
    error_log:          Optional[list[str]] = None,
    health_note:        Optional[str] = None,
    screen:             Optional[dict] = None,
) -> list[dict]:
    """하루 1회 발행용 통합 블록.

    per_ticker_results: [
      {
        "ticker":            str,
        "chief_report":      AnalysisReport | None,
        "qualified_reports": list[AnalysisReport],
        "all_reports":       list[AnalysisReport],  # chief 포함
      }, ...
    ]
    """
    from src.graph.report_formatter import (
        abstain_summary, benchmark_context_line, calibrated_confidence_note,
        debate_status, earnings_window_note, human_decides_note,
        no_pick_reason, recent_performance_lines, score_breakdown,
        market_background,
        stop_loss_note, today_line, upcoming_report_deadline,
    )

    blocks: list[dict] = []
    now     = datetime.now(KST)
    weekday = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
    regime_label = REGIME_KR.get(regime.upper(), regime)
    tickers      = [r["ticker"] for r in per_ticker_results]
    healthy      = not (health_note or "").startswith("⛔")
    _scores      = (screen or {}).get("scores") or {}

    # ── 0. 오늘 한 줄 ────────────────────────────────────────
    # 이것만 읽어도 오늘 뭘 할지 알아야 한다. 나머지는 전부 근거다.
    blocks.append(_callout(
        today_line(per_ticker_results, portfolio_summary, healthy),
        emoji="📌", color="blue_background",
    ))

    # ── 1. 건강도 ───────────────────────────────────────────
    # 아침 워크플로 1순위(morning-workflow 1절). 이 실행을 믿어도 되는지가
    # 아래 모든 수치의 전제다 (CR-9).
    if health_note:
        blocks.append(_callout(
            # [2026-09-15] health_note가 이미 "✅ 실행 정상..."으로 시작해서
            #   callout 아이콘과 겹쳐 "✅ ✅ 실행 정상"으로 보였다. 앞 기호를 뗀다.
            health_note.lstrip("✅⛔⚠️ ").strip(),
            emoji="⛔" if not healthy else "✅",
            color="red_background" if not healthy else "green_background",
        ))
    blocks.append(_paragraph(
        f"{now.strftime('%Y.%m.%d')} ({weekday}) {now.strftime('%H:%M')} KST · 시장 {regime_label}"
        + (f" · 종목 {', '.join(tickers)}" if tickers else "")
    ))
    blocks.append(_divider())

    # ── 2. 최종 판단 요약 ────────────────────────────────────
    blocks.append(_heading(2, "🎯 최종 판단"))
    if not per_ticker_results:
        # CR-10 ①: 0종목인 날은 "왜 0개인지"를 수치로. 없으면 고장으로 오해한다.
        blocks.append(_callout(no_pick_reason(screen, regime), emoji="🔍", color="gray_background"))
    else:
        for r in per_ticker_results:
            chief = r.get("chief_report")
            if chief is None:
                blocks.append(_bullet(f"**{r['ticker']}** — 분석 실패"))
                continue
            sig = SIGNAL_KR.get(chief.recommendation, chief.recommendation)
            if getattr(chief, "needs_review", False):
                sig = "⚠️ REVIEW (판단 없음)"
            # [2026-09-15] 종목코드만 쓰면 아침에 어느 회사인지 알 수 없다.
            #   근거는 headline_reason()이 투표 집계 줄을 건너뛰고 결론을 고른다.
            reason = headline_reason(chief)
            label = _ticker_label(r["ticker"]) or r["ticker"]
            blocks.append(_bullet(
                f"**{label}** {sig} ({chief.confidence:.0%})" + (f" — {reason}" if reason else "")))
            # [2026-09-21] 왜 이 종목이 오늘 올라왔는지를 판단 바로 밑에 붙인다.
            #   이게 없으면 독자는 "왜 하필 이 종목인가"를 알 수 없었다 (감사 §2-2).
            _detail = _scores.get(r["ticker"])
            if _detail:
                from src.screening.screen_context import build_screen_context, selection_reason
                _why = selection_reason(build_screen_context(r["ticker"], screen))
                if "선정 사유 없음" not in _why:
                    blocks.append(_bullet(f"↳ 선정 사유: {_why}"))
    blocks.append(_divider())

    # ── 2-b. 오늘의 시장 배경 ───────────────────────────────
    # macro_economist와 us_market_specialist는 하루 1회 캐시라 종목이 달라도 같은
    # 문단이 나온다. 4종목이면 같은 글이 4번 반복돼 종목 고유 근거까지 묻혔다.
    # 여기서 한 번만 보여주고 종목별 상세에서는 뺀다 (투표 집계에는 그대로 남는다).
    _bg_agents: list[AnalysisReport] = []
    for r in per_ticker_results:
        _bg_agents += [a for a in (r.get("all_reports") or []) if a.agent_name != "chief_strategist"]
    _bg = market_background(_bg_agents)
    if _bg:
        _bg_children: list[dict] = []
        for r in _bg:
            name = AGENT_NAME_KR.get(r.agent_name, r.agent_name)
            sig = SIGNAL_KR.get(r.recommendation, r.recommendation)
            _bg_children.append(_paragraph(f"**{name}** — {sig} {r.confidence:.0%}"))
            kept, dropped = filter_extreme_values(r.reasoning)
            for reason in kept:
                _bg_children.append(_bullet(reason))
            note = extreme_value_note(dropped)
            if note:
                _bg_children.append(_callout(note.replace("⚠️ ", ""), emoji="🚫",
                                             color="orange_background"))
        blocks.append(_toggle("🌏 오늘의 시장 배경 (종목 공통)", _bg_children))
        blocks.append(_divider())

    # ── 3. 액션 플랜 ────────────────────────────────────────
    blocks += _build_combined_action_plan_blocks(per_ticker_results, portfolio_summary)
    # 손절가는 남기되 제안임을 못박는다 (CR-10 ⑧ + S5 D-2 경로3)
    _first_buy = next((r["chief_report"] for r in per_ticker_results
                       if r.get("chief_report") is not None
                       and r["chief_report"].recommendation == "BUY"), None)
    # 두 문장을 줄바꿈으로 나눈다 — 한 줄로 이으면 좁은 화면에서 5줄짜리 글자 벽이 된다.
    blocks.append(_callout(
        (stop_loss_note(_first_buy) + "\n" + human_decides_note()).replace("**", ""),
        emoji="⚖️", color="gray_background",
    ))
    _events = " · ".join(x for x in (earnings_window_note(tickers),
                                     upcoming_report_deadline()) if x)
    if _events:                                              # CR-10 ⑪
        blocks.append(_callout(_events, emoji="📅", color="yellow_background"))
    blocks.append(_divider())

    # ── 4. 리스크 ───────────────────────────────────────────
    all_agents: list[AnalysisReport] = []
    all_chiefs: list[AnalysisReport] = []
    for r in per_ticker_results:
        all_agents += [a for a in (r.get("all_reports") or []) if a.agent_name != "chief_strategist"]
        if r.get("chief_report"):
            all_chiefs.append(r["chief_report"])

    blocks.append(_heading(2, "⚠️ 오늘의 주요 리스크"))
    # CR-10 ⑨: 토론이 열렸나 생략됐나. 생략된 날 "반대 논거가 없다"가 아님을 밝힌다.
    # [2026-09-15] 예전에는 전 종목 투표를 합쳐 "BUY 2.50 vs SELL 3.90"을 한 줄로 냈다.
    #   어느 종목 얘기인지 알 수 없는 숫자였다. 종목별 문장은 각 종목 toggle로 옮기고,
    #   여기에는 몇 종목에서 토론이 열렸는지만 남긴다 — 종목마다 긴 문장을 반복하면
    #   정작 리스크 목록이 아래로 밀린다.
    _opened = sum(1 for r in per_ticker_results
                  if "토론 생략" not in debate_status(
                      [a for a in (r.get("all_reports") or []) if a.agent_name != "chief_strategist"], ""))
    if per_ticker_results:
        blocks.append(_paragraph(
            f"토론 열림 {_opened}/{len(per_ticker_results)}종목 — "
            "생략된 종목은 구조화된 반대 논거가 없다. 종목별 사유는 아래 종목 상세에 있다."))
    unique_risks = real_risk_factors(all_agents + all_chiefs)
    for risk in unique_risks[:5]:
        blocks.append(_numbered(risk))
    if not unique_risks:
        blocks.append(_paragraph("주요 리스크 없음"))
    blocks.append(_divider())

    # ── 5. 종목별 상세 (toggle로 접는다) ─────────────────────
    if per_ticker_results:
        blocks.append(_heading(2, "📋 종목별 상세"))
        for payload in per_ticker_results:
            ticker = payload["ticker"]
            chief  = payload.get("chief_report")
            qnames = {r.agent_name for r in (payload.get("qualified_reports") or [])}
            agents = [a for a in (payload.get("all_reports") or [])
                      if a.agent_name != "chief_strategist"]
            sig  = SIGNAL_KR.get(chief.recommendation, chief.recommendation) if chief else "분석 실패"
            conf = f" ({chief.confidence:.0%})" if chief else ""

            children: list[dict] = []
            detail = _scores.get(ticker)
            if detail:                                       # CR-10 ⑦ — 뺀 자리에 들어간다
                children.append(_paragraph(f"선정 근거 — {score_breakdown(detail)}"))
            _ab = abstain_summary(agents)                    # CR-10 ⑤
            if _ab:
                children.append(_paragraph("투표 — " + _ab.replace("**", "")))
            if chief is not None:                            # CR-10 ④
                children.append(_paragraph(
                    "확신도 읽는 법 — "
                    + calibrated_confidence_note(chief.confidence, chief.agent_name).replace("**", "")))
            if agents:                                       # CR-10 ⑨ (종목별로)
                children.append(_paragraph(debate_status(agents, "")))
            children += _build_agent_summary_blocks(agents, chief, qnames, ticker=ticker,
                                                    heading_level=3)
            if agents or chief:
                children.append(_build_rationale_toggle(agents, chief, qnames))
            blocks.append(_toggle(f"{_ticker_label(ticker) or ticker} — {sig}{conf}", children))
        blocks.append(_divider())

    # ── 6. 시장 지표 (접는다) ───────────────────────────────
    blocks.append(_heading(2, "📊 시장 지표"))
    _bench = benchmark_context_line()                        # CR-10 ②
    if _bench:
        blocks.append(_paragraph(_bench))
    _mkt = _build_market_data_blocks(market_data)[1:]        # heading 제외
    _perf = recent_performance_lines()                       # CR-10 ③
    if _perf:
        _mkt.append(_paragraph("최근 실현 성적 (표본 단위 = 하루)"))
        for _line in _perf:
            _mkt.append(_bullet(_line.lstrip("- ").replace("**", "")))
    blocks.append(_toggle("지표·최근 성적 펼치기", _mkt or [_paragraph("시장 지표 없음")]))
    blocks.append(_divider())

    # ── 7. 포트폴리오 (접는다) ──────────────────────────────
    _pf = _build_portfolio_blocks(portfolio_summary)[1:]
    blocks.append(_toggle("💼 내 포트폴리오 현황", _pf or [_paragraph("보유 포지션 없음")]))

    return blocks


# ─────────────────────────────────────────────────────────
# 로컬 Fallback 저장
# ─────────────────────────────────────────────────────────

def _save_local_fallback(content: str, date_str: str, ticker: str) -> str:
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    filepath = FALLBACK_DIR / f"report_{date_str}_{ticker}.md"
    filepath.write_text(content, encoding="utf-8")
    return str(filepath)


# ─────────────────────────────────────────────────────────
# Notion 발행 메인 함수
# ─────────────────────────────────────────────────────────

def _create_notion_page(
    blocks:           list[dict],
    title:            str,
    fallback_content: str,
    fallback_ticker:  str,
) -> dict:
    """블록 리스트를 받아 실제 Notion 페이지를 생성하는 공통 헬퍼.

    면책 callout은 호출 측에서 이미 append된 상태로 들어와야 한다(원본 동작 유지하려면
    publish_to_notion / publish_combined_to_notion 양쪽에서 동일하게 면책을 추가하면 됨).
    """
    now      = datetime.now(KST)
    date_str = now.strftime("%Y-%m-%d")

    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        fallback = _save_local_fallback(fallback_content, date_str, fallback_ticker)
        return {
            "success":       False,
            "fallback_path": fallback,
            "error":         "NOTION_API_KEY 또는 NOTION_DATABASE_ID 미설정",
        }

    try:
        notion = Client(auth=NOTION_API_KEY)

        # DB title 속성 자동 탐지
        db_meta         = notion.databases.retrieve(database_id=NOTION_DATABASE_ID)
        title_prop_name = "Name"
        for prop_name, prop_meta in db_meta.get("properties", {}).items():
            if prop_meta.get("type") == "title":
                title_prop_name = prop_name
                break
        print(f"  → DB title 속성명: '{title_prop_name}'")

        # 페이지 생성 (첫 100블록)
        first_chunk = blocks[:MAX_BLOCKS_PER_REQ]
        page = notion.pages.create(
            parent     = {"database_id": NOTION_DATABASE_ID},
            properties = {title_prop_name: {"title": [{"text": {"content": title}}]}},
            children   = first_chunk,
        )

        page_id  = page["id"]
        page_url = page.get("url", f"https://notion.so/{page_id.replace('-', '')}")

        # 100블록 초과분 append
        remaining = blocks[MAX_BLOCKS_PER_REQ:]
        while remaining:
            chunk     = remaining[:MAX_BLOCKS_PER_REQ]
            remaining = remaining[MAX_BLOCKS_PER_REQ:]
            notion.blocks.children.append(block_id=page_id, children=chunk)

        print(f"  ✅ Notion 발행 완료: {page_url}")
        return {"success": True, "url": page_url}

    except APIResponseError as e:
        error_msg = f"Notion API 오류: {e.status} {e.code} — {e}"
        print(f"  ⚠️ {error_msg}")
        fallback = _save_local_fallback(fallback_content, date_str, fallback_ticker)
        print(f"  → 로컬 Fallback 저장: {fallback}")
        return {"success": False, "fallback_path": fallback, "error": error_msg}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"  ⚠️ Notion 발행 실패: {error_msg}")
        fallback = _save_local_fallback(fallback_content, date_str, fallback_ticker)
        print(f"  → 로컬 Fallback 저장: {fallback}")
        return {"success": False, "fallback_path": fallback, "error": error_msg}


def _disclaimer_callout() -> dict:
    now = datetime.now(KST)
    # CR-10 ⑩: 코드 SHA·모델·CHIEF_MODE·섀도 상태. 나중에 "그날 리포트가 어느 코드였나"를
    # 되짚을 수 있어야 성과를 코드 버전에 귀속시킬 수 있다.
    try:
        from src.graph.report_formatter import version_line
        version = version_line()
    except Exception:
        version = "버전 미상"
    return _callout(
        f"생성: {now.strftime('%Y-%m-%d %H:%M:%S')} KST | {version} | "
        "AI 생성 분석 | 투자 조언 아님 | 최종 판단은 본인 책임",
        emoji="⚠️",
    )


async def publish_to_notion(
    report_content:    str,
    ticker:            str,
    regime:            str,
    strategy:          str,
    chief_report:      Optional[AnalysisReport]       = None,
    qualified_reports: Optional[list[AnalysisReport]] = None,
    all_reports:       Optional[list[AnalysisReport]] = None,
    market_data:       Optional[dict]                 = None,
    debate_summary:    str                             = "",
    error_log:         Optional[list[str]]            = None,
    portfolio_summary: Optional[dict]                  = None,
    title_override:    Optional[str]                   = None,
) -> dict:
    """
    리포트를 Notion Database에 새 페이지로 발행합니다 (단일 ticker용).

    structured data(chief_report 등)가 있으면 v4.1 블록 빌더 사용.
    없으면 report_content(마크다운) → _markdown_to_blocks() fallback.

    title_override가 주어지면 그 문자열을 페이지 타이틀로 그대로 사용한다.

    Returns:
        성공: {"success": True,  "url": page_url}
        실패: {"success": False, "fallback_path": str, "error": str}
    """
    now      = datetime.now(KST)
    title    = title_override or (
        f"Daily stock market report_{now.strftime('%Y%m%d')}_{now.strftime('%H:%M')}_{strategy}"
    )

    # v4.1 구조화 블록 우선, fallback은 마크다운 파싱
    if chief_report is not None or all_reports:
        blocks = build_v4_blocks(
            ticker            = ticker,
            regime            = regime,
            strategy          = strategy,
            chief_report      = chief_report,
            qualified_reports = qualified_reports,
            all_reports       = all_reports,
            debate_summary    = debate_summary,
            error_log         = error_log,
            portfolio_summary = portfolio_summary,
            market_data       = market_data,
        )
    else:
        blocks = _markdown_to_blocks(report_content)

    blocks.append(_disclaimer_callout())

    return _create_notion_page(
        blocks           = blocks,
        title            = title,
        fallback_content = report_content,
        fallback_ticker  = ticker,
    )


async def publish_combined_to_notion(
    per_ticker_results: list[dict],
    regime:             str,
    strategy:           str,
    title:              str,
    market_data:        Optional[dict]      = None,
    portfolio_summary:  Optional[dict]      = None,
    error_log:          Optional[list[str]] = None,
    health_note:        Optional[str]        = None,
    screen:             Optional[dict]       = None,
    fallback_content:   str                  = "",
) -> dict:
    """하루 1회 통합 리포트를 Notion에 발행한다.

    per_ticker_results 각 원소:
      {
        "ticker":            str,
        "chief_report":      AnalysisReport | None,
        "qualified_reports": list[AnalysisReport],
        "all_reports":       list[AnalysisReport],  # chief 포함
      }
    """
    blocks = build_v4_blocks_combined(
        per_ticker_results = per_ticker_results,
        regime             = regime,
        strategy           = strategy,
        portfolio_summary  = portfolio_summary,
        market_data        = market_data,
        error_log          = error_log,
        health_note        = health_note,
        screen             = screen,
    )
    blocks.append(_disclaimer_callout())

    fallback_ticker = "_".join(r["ticker"] for r in per_ticker_results) or "COMBINED"

    return _create_notion_page(
        blocks           = blocks,
        title            = title,
        fallback_content = fallback_content or title,
        fallback_ticker  = fallback_ticker,
    )
