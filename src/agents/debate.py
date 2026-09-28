"""
src/agents/debate.py

Bull vs Bear 토론 메커니즘 — L3 Agent Ensemble 확장

[v2 변경]
  langchain-anthropic 의존성 제거.
  anthropic.AsyncAnthropic 직접 사용 (이미 설치된 anthropic>=0.49.0 활용).
  이유: langchain-anthropic 0.3.x가 langchain-core 버전 충돌을 유발.
       anthropic 공식 클라이언트 직접 사용이 더 안정적이고 버전 독립적.
"""

import os
from anthropic import AsyncAnthropic

from src.schemas.agent_output import AnalysisReport
from src.schemas.graph_state import GraphState
from src.utils.llm_budget import record_usage


# ─────────────────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────────────────

DEBATE_THRESHOLD    = 1.0    # BUY-SELL 가중합 차이 < 이 값이면 토론 시작
MAX_ARGUMENT_TOKENS = 500    # 주장/반박 한 편당 최대 토큰
DEBATE_MODEL        = "claude-sonnet-4-6"


# ─────────────────────────────────────────────────────────
# 1. 토론 시작 조건
# ─────────────────────────────────────────────────────────

def should_debate(reports: list[AnalysisReport]) -> bool:
    """
    BUY 가중합과 SELL 가중합의 차이가 DEBATE_THRESHOLD 미만이면 True.

    한쪽이 명확히 우세하면 토론 불필요 (비용 절감 + 결론 변화 없음).
    경계 구간에서만 토론으로 논거 충돌을 표면화.
    """
    buy_w  = sum(r.confidence for r in reports if r.recommendation == "BUY")
    sell_w = sum(r.confidence for r in reports if r.recommendation == "SELL")
    return abs(buy_w - sell_w) < DEBATE_THRESHOLD


# ─────────────────────────────────────────────────────────
# 2. 진영별 논거 수집
# ─────────────────────────────────────────────────────────

def _collect_side_context(reports: list[AnalysisReport]) -> str:
    if not reports:
        return "해당 진영 에이전트 없음"

    lines = []
    for r in reports:
        lines.append(f"[{r.agent_name}] 확신도 {r.confidence:.2f}")
        for basis in r.prediction_basis[:2]:
            lines.append(f"  수치 근거: {basis}")
        if r.reasoning:
            lines.append(f"  핵심 주장: {r.reasoning[-1]}")
        if r.risk_factors:
            lines.append(f"  인정하는 리스크: {r.risk_factors[0]}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 3. LLM 텍스트 생성 (anthropic 직접 호출)
# ─────────────────────────────────────────────────────────

async def _generate_text(system_prompt: str, human_prompt: str) -> str:
    """
    anthropic.AsyncAnthropic으로 텍스트 생성.
    langchain 없이 공식 클라이언트 직접 사용.
    """
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = await client.messages.create(
        model=DEBATE_MODEL,
        max_tokens=MAX_ARGUMENT_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": human_prompt}],
    )
    record_usage(DEBATE_MODEL, getattr(message, "usage", None))
    return message.content[0].text.strip()


# ─────────────────────────────────────────────────────────
# 4. Round 1 — Bull 주장
# ─────────────────────────────────────────────────────────

async def _generate_bull_argument(
    bull_reports: list[AnalysisReport],
    bear_reports: list[AnalysisReport],
    regime: str,
) -> str:
    system = "[REDACTED] Proprietary prompt engineering"
    human = (
        f"현재 시장 레짐: {regime.upper()}\n\n"
        f"[Bull 진영 근거]\n{_collect_side_context(bull_reports)}\n\n"
        f"[Bear 진영 주요 우려 (선제 반박 필요)]\n{_collect_side_context(bear_reports)}\n\n"
        "매수 논거를 강력하게 제시하고, Bear의 우려를 선제적으로 반박하세요."
    )
    return await _generate_text(system, human)


# ─────────────────────────────────────────────────────────
# 5. Round 1 — Bear 반박
# ─────────────────────────────────────────────────────────

async def _generate_bear_rebuttal(
    bear_reports: list[AnalysisReport],
    bull_argument: str,
    regime: str,
) -> str:
    system = "[REDACTED] Proprietary prompt engineering"
    human = (
        f"현재 시장 레짐: {regime.upper()}\n\n"
        f"[Bull이 방금 한 주장]\n{bull_argument}\n\n"
        f"[Bear 진영 근거]\n{_collect_side_context(bear_reports)}\n\n"
        "Bull의 논거를 직접 반박하고 Bear 입장을 강력하게 주장하세요."
    )
    return await _generate_text(system, human)


# ─────────────────────────────────────────────────────────
# 6. Round 2 — Bull 재반박
# ─────────────────────────────────────────────────────────

async def _generate_bull_rebuttal(
    bull_reports: list[AnalysisReport],
    bear_rebuttal: str,
    regime: str,
) -> str:
    system = "[REDACTED] Proprietary prompt engineering"
    human = (
        f"현재 시장 레짐: {regime.upper()}\n\n"
        f"[Bear가 방금 한 반박]\n{bear_rebuttal}\n\n"
        f"[Bull 진영 추가 근거]\n{_collect_side_context(bull_reports)}\n\n"
        "Bear의 반박에 직접 응답하고 Bull 입장을 최종 방어하세요."
    )
    return await _generate_text(system, human)


# ─────────────────────────────────────────────────────────
# 7. HOLD 심판 평가
# ─────────────────────────────────────────────────────────

# 토론은 Bull→Bear→Bull 순서라 Bull이 2번 말하고 마지막 발언도 가진다.
# 심판이 "마지막에 말한 쪽" 또는 "많이 말한 쪽"에 끌리지 않도록 명시한다
# (TradingAgents research_manager의 "independent of which side spoke first or last" 발상, 2026-09-20).
HOLD_VERDICT_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


async def _generate_hold_verdict(
    hold_reports: list[AnalysisReport],
    bull_argument: str,
    bear_rebuttal: str,
    bull_rebuttal: str,
    regime: str,
) -> str:
    system = HOLD_VERDICT_SYSTEM_PROMPT
    human = (
        f"현재 시장 레짐: {regime.upper()}\n\n"
        f"[Bull 주장]\n{bull_argument}\n\n"
        f"[Bear 반박]\n{bear_rebuttal}\n\n"
        f"[Bull 재반박]\n{bull_rebuttal}\n\n"
        f"[중립(HOLD) 에이전트들의 관점]\n{_collect_side_context(hold_reports)}\n\n"
        "어느 쪽 논거가 더 설득력 있는지, 결론 내리기 어려운 핵심 이유를 평가하세요."
    )
    return await _generate_text(system, human)


# ─────────────────────────────────────────────────────────
# 8. 토론 요약 포맷팅
# ─────────────────────────────────────────────────────────

def _format_debate_summary(
    bull_argument: str,
    bear_rebuttal: str,
    bull_rebuttal: str,
    hold_verdict: str,
    buy_w: float,
    sell_w: float,
    hold_w: float,
) -> str:
    lines = [
        "=== 투자 토론 요약 (Bull vs Bear) ===\n",
        f"투표 집계: BUY {buy_w:.2f} / SELL {sell_w:.2f} / HOLD {hold_w:.2f}\n",
        "【Bull 주장 (Round 1)】",
        bull_argument,
        "",
        "【Bear 반박 (Round 1)】",
        bear_rebuttal,
        "",
        "【Bull 재반박 (Round 2)】",
        bull_rebuttal,
    ]
    if hold_verdict:
        lines.extend([
            "",
            "【중립 심판 평가 (HOLD 에이전트)】",
            hold_verdict,
        ])
    lines.append("\n=== 토론 종료 ===")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 9. 메인 함수
# ─────────────────────────────────────────────────────────

async def run_debate(reports: list[AnalysisReport], regime: str) -> str:
    buy_w  = sum(r.confidence for r in reports if r.recommendation == "BUY")
    sell_w = sum(r.confidence for r in reports if r.recommendation == "SELL")
    hold_w = sum(r.confidence for r in reports if r.recommendation == "HOLD")

    if not should_debate(reports):
        winner = "BUY" if buy_w > sell_w else "SELL"
        print(f"  → 토론 생략 (BUY {buy_w:.2f} vs SELL {sell_w:.2f}, 차이 {abs(buy_w-sell_w):.2f} ≥ {DEBATE_THRESHOLD})")
        print(f"     {winner} 진영 명확히 우세 — chief_strategist가 직접 종합")
        return ""

    print(f"  → 토론 시작 (BUY {buy_w:.2f} vs SELL {sell_w:.2f}, 차이 {abs(buy_w-sell_w):.2f} < {DEBATE_THRESHOLD})")

    bull_reports = [r for r in reports if r.recommendation == "BUY"]
    bear_reports = [r for r in reports if r.recommendation == "SELL"]
    hold_reports = [r for r in reports if r.recommendation == "HOLD"]

    print(f"     Bull {len(bull_reports)}명 | Bear {len(bear_reports)}명 | HOLD심판 {len(hold_reports)}명")

    if not bull_reports or not bear_reports:
        print("  → 토론 생략 (한쪽 진영에 에이전트 없음)")
        return ""

    print("  → Round 1: Bull 주장 생성 중...")
    bull_argument = await _generate_bull_argument(bull_reports, bear_reports, regime)

    print("  → Round 1: Bear 반박 생성 중...")
    bear_rebuttal = await _generate_bear_rebuttal(bear_reports, bull_argument, regime)

    print("  → Round 2: Bull 재반박 생성 중...")
    bull_rebuttal = await _generate_bull_rebuttal(bull_reports, bear_rebuttal, regime)

    hold_verdict = ""
    if hold_reports:
        print(f"  → HOLD 심판 평가 중... ({len(hold_reports)}명)")
        hold_verdict = await _generate_hold_verdict(
            hold_reports, bull_argument, bear_rebuttal, bull_rebuttal, regime
        )

    summary = _format_debate_summary(
        bull_argument, bear_rebuttal, bull_rebuttal, hold_verdict,
        buy_w, sell_w, hold_w,
    )
    print(f"  → 토론 완료 (요약 {len(summary)}자)")
    return summary


# ─────────────────────────────────────────────────────────
# 10. LangGraph 노드
# ─────────────────────────────────────────────────────────

async def debate_node(state: GraphState) -> dict:
    agent_reports = [r for r in state.analysis_reports if r.agent_name != "chief_strategist"]
    print(f"\n[2.5/5] debate_node — Bull vs Bear 토론")
    try:
        summary = await run_debate(agent_reports, state.current_regime)
        return {"debate_summary": summary}
    except Exception as e:
        print(f"  ⚠️ 토론 실패 (파이프라인 계속 진행): {type(e).__name__}: {e}")
        return {"debate_summary": ""}