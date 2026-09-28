"""
Chief Strategist Agent v4.0

[변경사항]
  v3.0: AsyncAnthropic(claude-opus-4-6) + tool_use 직접 호출
  v3.1: weight_context 파라미터 추가 (피드백 루프 연결)
  v4.0: 거래 파라미터 출력 추가 (리포트 v4.0 1단계)
        - current_price 파라미터 추가 (pipeline에서 최근 종가 주입)
        - tool_use 스키마에 entry_price / stop_loss / take_profit 등 9개 필드 추가
        - 시스템 프롬프트에 손절·익절 계산 규칙 주입
        - AnalysisReport 생성 시 거래 파라미터 매핑

[v4 거래 파라미터]
  BUY 판단 시에만 채워짐 (HOLD/SELL 시 None):
  - entry_price, stop_loss, stop_loss_pct
  - take_profit_1 (R:R 1:2), take_profit_2 (R:R 1:3)
  - rr_ratio, position_size_pct, holding_period_weeks, entry_strategy
"""

import logging
import os
from datetime import datetime
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

from src.schemas.agent_output import AnalysisReport
from src.rag.context_injection import get_context_for_agent, inject_context_into_prompt
from src.utils.llm_budget import record_usage
from src.utils.market_session import market_session_note
from src.agents.chief_python import (
    chief_mode, run_python_sonnet, compute_trade_params, params_from_report, DEFAULT_STOP_PCT,
    apply_atr_stop, decide, format_decision,
)
from src.utils.shadow_log import input_hash, log_shadow, shadow_active


# ─────────────────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────────────────

CHIEF_MODEL = "claude-opus-4-6"
MAX_TOKENS  = 2048


# ─────────────────────────────────────────────────────────
# 시스템 프롬프트 (v3.0과 동일)
# ─────────────────────────────────────────────────────────

CHIEF_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


# ─────────────────────────────────────────────────────────
# tool_use 스키마 정의 (v3.0과 동일)
# ─────────────────────────────────────────────────────────

CHIEF_TOOLS = [
    {
        "name": "submit_final_strategy",
        "description": (
            "확정된 방향에 대한 설명과 리스크를 제출합니다. "
            "분석이 완료되면 반드시 이 도구를 호출하여 결과를 제출해야 합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "string",
                    "enum": ["BUY", "SELL", "HOLD"],
                    "description": "[확정된 최종 방향]의 값을 그대로. 다르게 내도 시스템이 되돌린다.",
                },
                "confidence": {
                    "type": "number",
                    "description": "[확정된 최종 방향]의 확신도를 그대로. 다르게 내도 시스템이 되돌린다.",
                },
                "reasoning": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "확정 방향의 근거 → 토론 핵심 쟁점 → 결론 순서로 3개 이상",
                },
                "data_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "인용한 에이전트 이름 목록 (2개 이상)",
                },
                "prediction_basis": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "가장 강한 신호 2개 이상 (에이전트명 + 수치 포함)",
                },
                "risk_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "반대 의견의 핵심 논거 1개 이상",
                },
                "selection_rationale": {
                    "type": "string",
                    "description": "최종 주목 종목 1~2개와 그 이유",
                },
                "entry_price": {
                    "type": "number",
                    "description": "BUY 시 추천 진입가. 현재가 기준. HOLD/SELL 시 생략.",
                },
                "stop_loss": {
                    "type": "number",
                    "description": "BUY 시 손절가. entry_price × (1 + stop_loss_pct/100).",
                },
                "stop_loss_pct": {
                    "type": "number",
                    "description": "쓰이지 않음. 손절은 시스템이 ATR 기준으로 계산한다. 생략해도 된다.",
                },
                "take_profit_1": {
                    "type": "number",
                    "description": "BUY 시 1차 익절가. R:R 1:2 기준 (손절폭 × 2).",
                },
                "take_profit_2": {
                    "type": "number",
                    "description": "BUY 시 2차 익절가. R:R 1:3 기준 (손절폭 × 3).",
                },
                "rr_ratio": {
                    "type": "number",
                    "description": "Risk:Reward 비율. (take_profit_1 - entry_price) / (entry_price - stop_loss).",
                },
                "position_size_pct": {
                    "type": "number",
                    "description": "쓰이지 않음. 포지션은 시스템이 손절폭과 보정 적중률로 계산한다.",
                },
                "holding_period_weeks": {
                    "type": "integer",
                    "description": "BUY 시 예상 보유 기간 (주). 단기 1~2, 중기 3~4, 장기 6~8.",
                },
                "entry_strategy": {
                    "type": "string",
                    "enum": ["시장가", "분할매수", "지정가대기"],
                    "description": "BUY 시 진입 전략.",
                },
            },
            "required": [
                "recommendation",
                "confidence",
                "reasoning",
                "data_sources",
                "prediction_basis",
                "risk_factors",
                "selection_rationale",
            ],
        },
    }
]


# ─────────────────────────────────────────────────────────
# 프롬프트 포맷터
# [v3.1 변경] weight_context 파라미터 추가
# ─────────────────────────────────────────────────────────

def _format_reports_as_prompt(
    reports: list[AnalysisReport],
    regime: str,
    debate_summary: str,
    weight_context: str = "",
    current_price: float = 0.0,        # ← v4.0 추가
    past_context: str = "",            # ← 2026-09-18 추가: 과거 판단 교훈 (decision_memory)
    screen_context: dict | None = None,  # ← 2026-09-21 추가: 오늘 이 종목이 뽑힌 사유
) -> str:
    lines = [
        f"=== 현재 시장 레짐: {regime.upper()} ===\n",
    ]

    # ── 2026-09-21: 오늘 이 종목이 왜 후보에 올랐는가 (src/screening/screen_context.py)
    # 투표 집계보다 먼저 읽게 둔다. 유니버스 110종목 중 이 종목만 올라온 이유가
    # 판단의 출발점이기 때문이다 (doc/2026-09-21_agent-audit.md §2-2).
    if screen_context:
        from src.screening.screen_context import selection_reason
        lines.append("[오늘 이 종목이 선정된 사유]")
        lines.append(f"  {selection_reason(screen_context)}")
        lines.append("")

    if current_price > 0:
        lines.append(f"[현재가] {current_price:,.0f}원 (거래 파라미터 계산 기준)\n")

    # ── v3.1: 에이전트 신뢰도 섹션 (weight_context가 있을 때만 삽입) ──────────
    # weight_context 예시:
    #   [에이전트 신뢰도 — Bull 레짐 기준 최근 예측 정확도]
    #     - kr_market_specialist   0.2134  ↑ 높음
    #     - quant_analyst          0.1205  ↓ 낮음
    #   ✅ 신뢰도 높은 에이전트의 의견에 더 큰 비중을 두세요.
    #
    # 삽입 위치: 레짐 정보 직후, 투표 집계 직전
    # → 모델이 투표 집계를 읽기 전에 "어떤 에이전트를 더 신뢰해야 하는지"
    #   컨텍스트를 먼저 파악하게 하기 위함.
    #
    # 워밍업 기간(sample_count < 10)에는 pipeline.py에서 빈 문자열("")을 전달하므로
    # 이 섹션이 생략됨 → v3.0과 완전히 동일한 동작.
    if weight_context:
        lines.append(weight_context)
        lines.append("")

    # ── 2026-09-18: 과거 판단 교훈 (src/data/decision_memory.py) ──────────────
    # 신뢰도 섹션과 같은 자리. 투표 집계를 읽기 전에 "지난번엔 왜 틀렸나"를 먼저 보게 한다.
    # 교훈이 없으면 빈 문자열이 와서 섹션이 생략된다 → 이전과 동일한 프롬프트.
    if past_context:
        lines.append(past_context)
        lines.append("")

    lines.append(f"=== 전문가 애널리스트 {len(reports)}명의 분석 결과 ===\n")

    buy_weight  = sum(r.confidence for r in reports if r.recommendation == "BUY")
    sell_weight = sum(r.confidence for r in reports if r.recommendation == "SELL")
    hold_weight = sum(r.confidence for r in reports if r.recommendation == "HOLD")

    lines.append("[투표 집계 (confidence 가중합)]")
    lines.append(f"  BUY:  {buy_weight:.2f}")
    lines.append(f"  SELL: {sell_weight:.2f}")
    lines.append(f"  HOLD: {hold_weight:.2f}")
    lines.append("")

    for r in reports:
        lines.append(f"─── {r.agent_name} ───")
        lines.append(f"  판단: {r.recommendation} (확신도: {r.confidence:.2f})")
        lines.append("  추론:")
        for step in r.reasoning:
            lines.append(f"    • {step}")
        lines.append("  수치 근거:")
        for basis in r.prediction_basis:
            lines.append(f"    • {basis}")
        lines.append("  리스크:")
        for risk in r.risk_factors:
            lines.append(f"    • {risk}")
        if r.selection_rationale:
            lines.append(f"  주목 종목: {r.selection_rationale}")
        lines.append("")

    if debate_summary:
        lines.append("=" * 50)
        lines.append(debate_summary)
        lines.append("=" * 50)
        lines.append("")
        lines.append("=== 위 분석들과 토론 결과를 종합하여 최종 투자 전략을 도출해주세요 ===")
    else:
        lines.append(
            "=== 위 분석들을 종합하여 최종 투자 전략을 도출해주세요 ==="
            " (이번 라운드는 한쪽이 명확히 우세하여 토론 생략됨)"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# tool_use 응답 파싱 (v3.0과 동일)
# ─────────────────────────────────────────────────────────

def _parse_tool_use_response(response) -> dict:
    """
    Anthropic tool_use 응답에서 submit_final_strategy 호출 인자를 추출.

    응답 content 블록 구조:
      [{"type": "tool_use", "name": "submit_final_strategy", "input": {...}}, ...]
    """
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_final_strategy":
            return block.input

    raise ValueError(
        f"tool_use 블록 없음. stop_reason={response.stop_reason}, "
        f"content={response.content}"
    )


def _review_report(reason: str) -> AnalysisReport:
    """LLM 출력을 못 읽었을 때. recommendation은 스키마상 HOLD지만 needs_review=True로
    구분되어 리포트에 REVIEW로 표시되고, confidence 0.0이라 포지션 기록·신호 집계에서 빠진다.
    (TradingAgents signal_processing: "an unrecognizable decision yields REVIEW rather than
    a fabricated Hold, so a parsing failure is visible")"""
    return AnalysisReport(
        agent_name="chief_strategist",
        confidence=0.0,
        recommendation="HOLD",
        needs_review=True,
        reasoning=[
            f"[REVIEW] 종합 판단 응답을 해석할 수 없음: {reason[:200]}",
            "이 종목은 자동 판단이 아니라 사람의 확인이 필요하다",
            "HOLD로 표시되지만 '관망 판단'이 아니라 '판단 없음'이다",
        ],
        data_sources=["chief_parse_failure", "review"],
        prediction_basis=["응답 해석 실패로 근거 없음", "review"],
        risk_factors=["종합 판단 부재"],
    )


# ─────────────────────────────────────────────────────────
# 메인 함수
# [v3.1 변경] weight_context 파라미터 추가
# ─────────────────────────────────────────────────────────

async def run_chief_strategist(
    reports: list[AnalysisReport],
    regime: str = "neutral",
    debate_summary: str = "",
    weight_context: str = "",
    current_price: float = 0.0,        # ← v4.0 추가 (pipeline에서 최근 종가 주입)
    past_context: str = "",            # ← 2026-09-18 추가 (pipeline에서 decision_memory 조회)
    screen_context: dict | None = None,  # ← 2026-09-21 추가 (pipeline에서 스크리닝 사유 주입)
    ticker: str = "",                    # ← 2026-09-21 추가 (ATR 손절 계산에 필요)
) -> AnalysisReport:
    if not reports:
        logger.warning("[ChiefStrategist] 에이전트 결과 0건 — HOLD 폴백 반환")
        return AnalysisReport(
            agent_name="chief_strategist",
            confidence=0.0,
            recommendation="HOLD",
            reasoning=[
                "모든 에이전트가 타임아웃 또는 실패하여 분석 결과 없음",
                "데이터 부족으로 방향성 판단 불가 — 기본 HOLD 유지",
                "다음 파이프라인 실행 시 재분석 필요",
            ],
            data_sources=["no_agent_data", "fallback"],
            prediction_basis=["에이전트 결과 없음으로 인한 대체값", "fallback"],
            risk_factors=["전체 에이전트 실패 — 분석 근거 없음"],
        )

    # 1. 프롬프트 구성 (v4.0: current_price 추가)
    formatted = _format_reports_as_prompt(
        reports        = reports,
        regime         = regime,
        debate_summary = debate_summary,
        weight_context = weight_context,
        current_price  = current_price,    # ← v4.0 추가
        past_context   = past_context,     # ← 2026-09-18 추가
        screen_context = screen_context,   # ← 2026-09-21 추가
    )

    # 1-a-2. 최종 방향을 **파이썬 규칙이 먼저 정한다** (2026-09-21).
    #   옛 구조는 프롬프트에 "BUY 가중합 > SELL + 0.2 → BUY"라고 써 두고 opus가 지키기를
    #   기대했다. 운영 40건 재현 결과 그 규칙의 답은 BUY 15/SELL 20/HOLD 5인데 opus가
    #   실제로 낸 것은 BUY 3/SELL 0/HOLD 37이었다 — 규칙이 지켜지지 않았다.
    #   이제 방향·확신도는 decide()가 확정하고 opus는 그 이유를 쓴다
    #   (doc/2026-09-21_agent-audit.md §1-10).
    decided_rec, decided_conf, decision_detail = decide(reports)
    decision_line = format_decision(decided_rec, decided_conf, decision_detail)
    formatted = (
        "[확정된 최종 방향 — 바꿀 수 없습니다]\n"
        f"  {decision_line}\n"
        "  당신의 일은 이 방향이 나온 이유를 쓰고, 반대 논거를 risk_factors에 담는 것입니다.\n"
        "  recommendation·confidence를 다르게 제출해도 시스템이 위 값으로 되돌립니다.\n\n"
        + formatted
    )

    # 1-b. 장중 실행이면 "즉시 매수" 금지 주의를 붙인다 (G5)
    formatted += market_session_note()

    # 1-c. 옵션 A — CHIEF_MODE=python_sonnet 이면 산술은 파이썬, 문장만 sonnet-5 (src/agents/chief_python.py)
    if chief_mode() == "python_sonnet":
        report = await run_python_sonnet(reports, formatted, current_price=current_price, ticker=ticker)
        report.reasoning = list(report.reasoning) + ["[CHIEF_MODE=python_sonnet]"]
        return report

    # 2. RAG 컨텍스트 주입 (v3.0과 동일)
    rag_context = get_context_for_agent(
        agent_name="chief_strategist",
        state_vars={
            "regime": regime,
            "date":   datetime.now().strftime("%Y-%m-%d"),
        },
    )
    system_content = inject_context_into_prompt(CHIEF_SYSTEM_PROMPT, rag_context)

    # 3. Anthropic API 직접 호출 (v3.0과 동일)
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.messages.create(
        model       = CHIEF_MODEL,
        max_tokens  = MAX_TOKENS,
        system      = system_content,
        tools       = CHIEF_TOOLS,
        tool_choice = {"type": "tool", "name": "submit_final_strategy"},
        messages    = [{"role": "user", "content": formatted}],
    )

    # 토큰·비용 집계 (src/utils/llm_budget.py). opus는 단가가 가장 높아
    # 여기가 실행 비용의 큰 몫을 차지한다.
    record_usage(CHIEF_MODEL, getattr(response, "usage", None))

    # 4. tool_use 응답 → dict 파싱. 실패하면 HOLD로 날조하지 않고 REVIEW 표식 (G5)
    try:
        result_dict = _parse_tool_use_response(response)
    except ValueError as e:
        logger.error(f"[ChiefStrategist] 응답 해석 실패 → REVIEW: {e}")
        return _review_report(str(e))

    # 5. AnalysisReport 객체 생성 (v4.0: 거래 파라미터 매핑 추가)
    #   recommendation·confidence는 opus 응답이 아니라 decide() 결과를 쓴다.
    #   opus가 다른 방향을 제출해도 여기서 되돌린다.
    report = AnalysisReport(
        agent_name          = "chief_strategist",
        recommendation      = decided_rec,
        confidence          = decided_conf,
        reasoning           = result_dict["reasoning"],
        data_sources        = result_dict["data_sources"],
        prediction_basis    = result_dict["prediction_basis"],
        risk_factors        = result_dict["risk_factors"],
        selection_rationale = result_dict.get("selection_rationale", ""),
        # 거래 파라미터 (BUY 시 LLM이 채워줌, HOLD/SELL 시 None)
        entry_price          = result_dict.get("entry_price"),
        stop_loss            = result_dict.get("stop_loss"),
        stop_loss_pct        = result_dict.get("stop_loss_pct"),
        take_profit_1        = result_dict.get("take_profit_1"),
        take_profit_2        = result_dict.get("take_profit_2"),
        rr_ratio             = result_dict.get("rr_ratio"),
        position_size_pct    = result_dict.get("position_size_pct"),
        holding_period_weeks = result_dict.get("holding_period_weeks"),
        entry_strategy       = result_dict.get("entry_strategy"),
    )

    report.reasoning = list(report.reasoning) + [decision_line]
    _llm_rec = result_dict.get("recommendation")
    if _llm_rec and _llm_rec != decided_rec:
        logger.info(f"[ChiefStrategist] opus 제안 {_llm_rec} → 규칙 확정 {decided_rec}")
        report.reasoning.append(f"[방향 확정] LLM 제안 {_llm_rec}을 규칙 결과 {decided_rec}으로 대체")

    # 5-b. 옵션 A 섀도: opus가 낸 파라미터 6개 vs 파이썬 공식. candidate는 LLM 호출 없음(비용 0).
    #      python -m src.utils.shadow_log chief_python_sonnet 으로 일치율을 본다.
    _log_param_shadow(report, formatted, current_price)

    # 6. 손절·익절·포지션 확정 (2026-09-21).
    #    손절은 LLM이 제안한 -3~-6%가 아니라 종목 ATR 기준이다. 포지션은 보정 적중률
    #    (G3, src/evaluation/calibration.py)과 리스크 한도 중 작은 쪽이다.
    apply_atr_stop(report, ticker, current_price)

    return report


def _log_param_shadow(report: AnalysisReport, formatted: str, current_price: float) -> None:
    if not shadow_active("chief_python_sonnet"):
        return
    try:
        h = input_hash("chief", formatted)
        base = params_from_report(report)
        if report.recommendation == "BUY" and current_price > 0:
            cand = compute_trade_params(current_price, float(report.stop_loss_pct or DEFAULT_STOP_PCT))
        else:
            cand = {k: None for k in base}
        log_shadow("chief_python_sonnet", "baseline", CHIEF_MODEL, h,
                   {"params": base, "recommendation": report.recommendation}, 0, 0, 0)
        log_shadow("chief_python_sonnet", "candidate", "python-formula", h,
                   {"params": cand, "recommendation": report.recommendation}, 0, 0, 0)
    except Exception as e:
        logger.warning(f"[ChiefStrategist] 섀도 기록 실패(무시): {e}")
