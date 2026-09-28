"""
src/agents/chief_python.py

옵션 A — Chief의 산술을 파이썬으로, 문장만 소형 모델로 (doc/2026-09-10_benchmark-analysis.md F-5).

[왜]
  S1 2-1: opus가 내는 거래 파라미터 9개 중 6개가 공식이고 투표 집계도 파이썬이 프롬프트에
  넣어 준다. TradingAgents는 같은 자리(신호 추출)를 LLM 0콜 정규식으로 처리한다
  (graph/signal_processing.py "no extra LLM call is needed"). 2026-09-09 실측에서 opus 8콜이
  하루 비용의 75%였다.

[동작 — CHIEF_MODE 환경변수]
  legacy (기본)   : opus tool_use 그대로. 단, 아래 파이썬 계산값을 섀도 로그에 baseline(opus 출력)과
                    나란히 기록해 일치율을 잰다. LLM 추가 호출 0, 비용 0.
  python_sonnet   : 투표·파라미터는 여기서 계산하고, reasoning·risk 등 문장만 claude-sonnet-5
                    1콜(≈1k 출력)로 받는다. 교체는 사람이 CHIEF_MODE를 바꿔서 한다.

[공식 — chief 시스템 프롬프트의 규칙을 그대로 옮김]
  BUY 가중합 > SELL 가중합 + 0.2 → BUY / SELL > BUY + 0.2 → SELL / 그 외 HOLD
  stop_loss      = entry × (1 + stop_loss_pct/100)          (stop_loss_pct 기본 -4)
  take_profit_1  = entry + (entry - stop_loss) × 2            (R:R 1:2)
  take_profit_2  = entry + (entry - stop_loss) × 3
  rr_ratio       = (tp1 - entry) / (entry - stop_loss)        (= 항등식 2.0)
  position_size  = calibration.position_size_for()            (G3; LLM 값 아님)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from anthropic import AsyncAnthropic

from src.evaluation.calibration import position_size_for
from src.schemas.agent_output import AnalysisReport
from src.utils.llm_budget import record_usage

logger = logging.getLogger(__name__)

NARRATIVE_MODEL      = "claude-sonnet-5"
NARRATIVE_MAX_TOKENS = 1024
VOTE_MARGIN          = 0.2
DEFAULT_STOP_PCT     = -4.0   # ohlcv_cache에 봉이 모자란 종목의 폴백 (ATR을 못 낼 때만)
MAX_CONFIDENCE       = 0.85

# 손절 범위 (2026-09-21). 종전 -3~-6%는 ATR 기준 손절을 전부 잘라냈다.
# 분석 종목 ATR14 중앙값이 7.3%라 2×ATR이 -14.6%다 (doc/2026-09-21_agent-audit.md §2-5).
MIN_STOP_PCT = -3.0
MAX_STOP_PCT = -15.0

# 한 거래에서 감수할 시드 대비 최대 손실(%).
# 손절폭이 -4% → -15%로 넓어지면 같은 포지션에서 손실이 3.75배가 된다.
# 손절폭이 넓은 종목은 적게 사서 손실 금액을 일정하게 맞춘다.
RISK_PER_TRADE_PCT = 0.5
MIN_POSITION_PCT   = 2.0
PARAM_KEYS           = ("entry_price", "stop_loss", "stop_loss_pct", "take_profit_1", "take_profit_2", "rr_ratio")


def chief_mode() -> str:
    raw = os.getenv("CHIEF_MODE", "legacy").strip().lower()
    return raw if raw in ("legacy", "python_sonnet") else "legacy"


# ── 산술 ────────────────────────────────────────────────────────────────

# 표를 던지는 에이전트. 서로 다른 데이터를 보는 넷만 남겼다
# (doc/2026-09-21_agent-audit.md §2-3).
#
# [왜 8명이 아닌가]
#   macro_economist·us_market_specialist는 daily_cache로 하루 1콜이라 종목을 모른다.
#   그날 모든 종목에 같은 표가 복사돼 종목 간 차이를 못 만들고 HOLD/SELL 질량만 키웠다.
#   technical_analyst·quant_analyst는 quant_rule_agent와 같은 OHLCV를 본다.
#   남는 독립 정보원은 수급·재무공시·이벤트·일봉규칙 넷이다.
#
#   이들은 투표에서만 빠진다. 리포트와 chief 프롬프트에는 그대로 실리고,
#   prediction_log에도 계속 기록돼 나중에 "누가 맞았나"를 잴 수 있다.
VOTING_AGENTS: tuple[str, ...] = (
    "kr_market_specialist",    # 외국인·기관 수급
    "fundamental_analyst",     # 재무·컨센서스·공시
    "sentiment_analyst",       # 종목 이벤트 (v3)
    "quant_rule_agent",        # 일봉 규칙, LLM 없음
)


def decide(reports: list[AnalysisReport]) -> tuple[str, float, dict]:
    """최종 방향을 정한다. **반대가 없으면 그 방향, 상충하면 HOLD.**

    [왜 가중합이 아닌가]
      옛 규칙은 "BUY 가중합 > SELL 가중합 + 0.2"였다. 가중치는 자기보고 confidence인데
      그 값이 실제 적중률과 맞는지 검증된 적이 없다(calibration 표본 부족).
      검증 안 된 수를 더하는 것보다 "반대가 있는가"를 세는 편이 30건 표본으로 판정 가능하다.

    [왜 다수결이 아닌가]
      3:1이어도 반대가 있으면 HOLD다. 신호가 상충할 때 진입하지 않는다는 원칙이
      표본이 없는 지금 단계에서 가장 방어 가능하다.

    Returns: (recommendation, confidence, detail)
      confidence = 동의 수 / 참여 수 × MAX_CONFIDENCE. 자기보고 값이 아니다.
      참여자가 없으면 ("HOLD", 0.0) — '관망'이 아니라 '판단 없음'이라 원장에 남지 않는다.
    """
    sides: dict[str, list[str]] = {"buy": [], "sell": [], "hold": []}
    seen: set[str] = set()
    for r in reports:
        name = r.agent_name
        if name not in VOTING_AGENTS or name in seen:
            continue
        if r.confidence <= 0.0:      # abstain은 '의견 없음' — 반대표가 아니다
            continue
        seen.add(name)
        sides[r.recommendation.lower()].append(name)

    participants = len(seen)
    detail = {
        "buy": sides["buy"], "sell": sides["sell"], "hold": sides["hold"],
        "absent": [a for a in VOTING_AGENTS if a not in seen],
        "participants": participants,
        "rule": "반대 0이면 그 방향, 상충하면 HOLD",
    }

    if participants == 0:
        return "HOLD", 0.0, detail

    if sides["buy"] and not sides["sell"]:
        rec, agree = "BUY", len(sides["buy"])
    elif sides["sell"] and not sides["buy"]:
        rec, agree = "SELL", len(sides["sell"])
    else:
        rec, agree = "HOLD", max(len(sides["hold"]), 1)

    return rec, round(agree / participants * MAX_CONFIDENCE, 4), detail


def format_decision(rec: str, confidence: float, detail: dict) -> str:
    """프롬프트 헤더와 reasoning에 들어갈 한 줄. 사람과 LLM이 같은 문장을 본다."""
    def names(key: str) -> str:
        return ", ".join(detail.get(key) or []) or "-"

    return (
        f"[결정] {rec} (확신도 {confidence:.2f}) — 규칙: {detail['rule']}. "
        f"투표 {detail['participants']}/{len(VOTING_AGENTS)}명 · "
        f"매수 {names('buy')} · 매도 {names('sell')} · 관망 {names('hold')}"
        + (f" · 기권 {names('absent')}" if detail.get("absent") else "")
    )


def compute_vote(reports: list[AnalysisReport]) -> tuple[str, float, dict]:
    """confidence 가중 투표. abstain(0.0)은 어디에도 안 들어간다.

    Returns: (recommendation, confidence, {"buy":w,"sell":w,"hold":w})
    confidence = 이긴 진영 가중합 / 전체 가중합, 상한 MAX_CONFIDENCE. 표가 없으면 0.0.
    """
    w = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
    for r in reports:
        if r.confidence <= 0.0:
            continue
        w[r.recommendation.lower()] = w.get(r.recommendation.lower(), 0.0) + r.confidence
    total = w["buy"] + w["sell"] + w["hold"]
    if total <= 0:
        return "HOLD", 0.0, w
    if w["buy"] > w["sell"] + VOTE_MARGIN:
        rec, win = "BUY", w["buy"]
    elif w["sell"] > w["buy"] + VOTE_MARGIN:
        rec, win = "SELL", w["sell"]
    else:
        rec, win = "HOLD", max(w["hold"], total - abs(w["buy"] - w["sell"]))
    conf = min(MAX_CONFIDENCE, round(win / total, 4))
    return rec, conf, w


def risk_scaled_position(base_pct: float, stop_pct: float) -> float:
    """손절폭을 감안한 포지션(%). 보정값과 리스크 한도 중 작은 쪽.

    한 거래 손실 = 포지션 × 손절폭. 이것이 시드의 RISK_PER_TRADE_PCT를 넘지 않게 한다.
    예: 손절 -15% → 0.5 / 15 × 100 = 3.3%. 손절 -4% → 12.5%(보정값 5%가 더 작아 5%).
    """
    stop = abs(float(stop_pct)) or abs(DEFAULT_STOP_PCT)
    limit = RISK_PER_TRADE_PCT / stop * 100.0
    return round(max(MIN_POSITION_PCT, min(float(base_pct), limit)), 1)


def compute_trade_params(entry_price: float, stop_loss_pct: float = DEFAULT_STOP_PCT) -> dict:
    """chief 프롬프트 [거래 파라미터 계산 규칙]의 6개 공식.

    stop_loss_pct는 ATR 기준으로 들어온다 (src/utils/atr.py). clamp 범위가
    -3~-6%였을 때는 ATR 손절이 전부 -6%로 잘려 고정 손절과 다를 바 없었다.
    """
    stop_loss_pct = float(max(MAX_STOP_PCT, min(MIN_STOP_PCT, stop_loss_pct)))
    stop_loss = round(entry_price * (1 + stop_loss_pct / 100), 2)
    risk = entry_price - stop_loss
    tp1 = round(entry_price + risk * 2, 2)
    tp2 = round(entry_price + risk * 3, 2)
    rr = round((tp1 - entry_price) / risk, 2) if risk else None
    return {"entry_price": float(entry_price), "stop_loss": stop_loss, "stop_loss_pct": stop_loss_pct,
            "take_profit_1": tp1, "take_profit_2": tp2, "rr_ratio": rr}


def params_from_report(r: AnalysisReport) -> dict:
    """섀도 비교용 — opus 출력에서 같은 6개 키만."""
    return {k: getattr(r, k) for k in PARAM_KEYS}


def apply_atr_stop(report: AnalysisReport, ticker: str, current_price: float) -> None:
    """BUY 리포트의 손절·익절·포지션을 ATR 기준으로 확정한다. CHIEF_MODE 두 경로 공통.

    LLM이 제안한 손절폭은 쓰지 않는다. 손절은 "몇 %가 적당한가"가 아니라
    "이 종목이 평소 하루에 얼마나 움직이나"로 정해야 하고, 그건 계산이지 판단이 아니다
    (doc/2026-09-21_agent-audit.md §2-5).

    실패해도 예외를 올리지 않는다 — 손절이 비면 position_tracker가 기본값으로 대체해
    리포트와 감시 기준이 어긋난다. 그 경우 종전 고정값으로 떨어진다.
    """
    if report.recommendation != "BUY" or not current_price or current_price <= 0:
        return

    from src.utils.atr import ATR_MULTIPLE, stop_pct_for

    llm_stop = report.stop_loss_pct
    try:
        stop_pct = stop_pct_for(ticker)
    except Exception as e:
        logger.warning(f"[chief] {ticker} ATR 손절 계산 실패 — 고정 {DEFAULT_STOP_PCT}% 사용: {e}")
        stop_pct = DEFAULT_STOP_PCT

    params = compute_trade_params(current_price, stop_pct)

    try:
        base, why = position_size_for(report.confidence, agent="chief_strategist")
    except Exception as e:
        logger.warning(f"[chief] 보정 조회 실패 — 기본 포지션 사용: {e}")
        base, why = 5.0, "보정 조회 실패 → 기본 5%"

    params["position_size_pct"] = risk_scaled_position(base, params["stop_loss_pct"])

    for key, value in params.items():
        setattr(report, key, value)

    report.reasoning = list(report.reasoning) + [
        f"[손절 {ATR_MULTIPLE:g}×ATR] {params['stop_loss_pct']:.1f}% "
        f"({params['stop_loss']:,.0f}원) — LLM 제안 "
        f"{f'{llm_stop:.1f}%' if llm_stop is not None else 'n/a'} 대신 종목 변동성 기준",
        f"[포지션] 보정 {base:.1f}% → {params['position_size_pct']:.1f}% "
        f"(한 거래 최대 손실 시드의 {RISK_PER_TRADE_PCT}%, {why})",
    ]


# ── 문장 생성 (sonnet-5) ─────────────────────────────────────────────────

NARRATIVE_TOOLS = [{
    "name": "submit_narrative",
    "description": "투표와 거래 파라미터는 이미 계산됐다. 근거 문장과 손절폭·보유기간·진입방식만 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning":            {"type": "array", "items": {"type": "string"}, "minItems": 3,
                                     "description": "투표 집계 + 토론 쟁점 + 결론, 3단계 이상"},
            "data_sources":         {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "prediction_basis":     {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "risk_factors":         {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "selection_rationale":  {"type": "string"},
            "stop_loss_pct":        {"type": "number", "description": "-6 ~ -3. 기술적 지지선이 명확하면 그 수준, 없으면 -4"},
            "holding_period_weeks": {"type": "integer", "description": "단기 1~2, 중기 3~4, 장기 6~8"},
            "entry_strategy":       {"type": "string", "enum": ["시장가", "분할매수", "지정가대기"]},
        },
        "required": ["reasoning", "data_sources", "prediction_basis", "risk_factors"],
    },
}]

NARRATIVE_SYSTEM = "[REDACTED] Proprietary prompt engineering"


def _parse_narrative(response) -> dict:
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_narrative":
            return block.input
    raise ValueError(f"submit_narrative 블록 없음. stop_reason={response.stop_reason}")


async def run_python_sonnet(
    reports: list[AnalysisReport],
    formatted_prompt: str,
    current_price: float = 0.0,
    client: Optional[AsyncAnthropic] = None,
    ticker: str = "",
) -> AnalysisReport:
    rec, conf, detail = decide(reports)
    header = (
        f"[파이썬 계산 결과 — 바꾸지 말 것]\n"
        f"  {format_decision(rec, conf, detail)}\n"
        f"  현재가: {current_price:,.0f}원\n\n"
    )
    client = client or AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.messages.create(
        model=NARRATIVE_MODEL, max_tokens=NARRATIVE_MAX_TOKENS, system=NARRATIVE_SYSTEM,
        tools=NARRATIVE_TOOLS, tool_choice={"type": "tool", "name": "submit_narrative"},
        messages=[{"role": "user", "content": header + formatted_prompt}],
    )
    record_usage(NARRATIVE_MODEL, getattr(response, "usage", None))
    text = _parse_narrative(response)

    reasoning = list(text["reasoning"]) + [format_decision(rec, conf, detail)]
    report = AnalysisReport(
        agent_name="chief_strategist",
        recommendation=rec, confidence=conf,
        reasoning=reasoning,
        data_sources=list(text["data_sources"]),
        prediction_basis=list(text["prediction_basis"]),
        risk_factors=list(text["risk_factors"]),
        selection_rationale=text.get("selection_rationale", ""),
        holding_period_weeks=text.get("holding_period_weeks") if rec == "BUY" else None,
        entry_strategy=text.get("entry_strategy") if rec == "BUY" else None,
    )
    # 손절·익절·포지션은 ATR 기준으로 여기서 채운다 (legacy 경로와 같은 함수).
    apply_atr_stop(report, ticker, current_price)
    return report
