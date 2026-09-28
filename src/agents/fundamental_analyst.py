"""
Fundamental Analyst Agent
담당: 기업 펀더멘털 분석 (재무/컨센서스/공시)
소비 MCP: krx_market (get_financials, get_consensus_estimates, get_dart_disclosure)

[v2 변경]
  FUNDAMENTAL_TICKERS 하드코딩 → load_universe() 교체
"""

import asyncio
import json
from datetime import datetime
from src.utils.mcp_client import mcp_client
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.base_agent import create_structured_agent
from src.schemas.agent_output import AnalysisReport
from src.rag.context_injection import get_context_for_agent, inject_context_into_prompt
from src.universe.universe_builder import load_universe
from src.utils.mcp_result import failure_note, parse_mcp_result


# ─────────────────────────────────────────────────────────
# 분석 대상 종목 — 유니버스에서 동적 로드
# ─────────────────────────────────────────────────────────

def _get_tickers(n: int = 10) -> list[str]:
    return load_universe()[:n]


FUNDAMENTAL_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


# ─────────────────────────────────────────────────────────
# Step 1: 데이터 수집
# ─────────────────────────────────────────────────────────

async def _collect_fundamental_data(target_ticker: str | None = None) -> dict:
    """
    대상 종목 1개의 재무/컨센서스/공시 데이터를 병렬 수집합니다.

    target_ticker가 None인 경우(단독 디버그 실행 등)에는 유니버스 1순위로 폴백.
    """
    tickers = [target_ticker] if target_ticker else _get_tickers()[:1]

    async with mcp_client("src/mcp_servers/krx_market/server.py") as krx_client:
        financials_tasks = [
            krx_client.call_tool("get_financials",          {"ticker": t})
            for t in tickers
        ]
        consensus_tasks = [
            krx_client.call_tool("get_consensus_estimates", {"ticker": t})
            for t in tickers
        ]
        dart_tasks = [
            krx_client.call_tool("get_dart_disclosure",     {"ticker": t, "days": 30})
            for t in tickers
        ]
        # 가격 폴백용: get_financials 실패해도 종가/등락률은 살린다
        price_tasks = [
            krx_client.call_tool("get_stock_price",         {"ticker": t, "days": 5})
            for t in tickers
        ]
        results = await asyncio.gather(
            *financials_tasks, *consensus_tasks, *dart_tasks, *price_tasks,
            return_exceptions=True,
        )

    n               = len(tickers)
    financials_raws = results[:n]
    consensus_raws  = results[n:n*2]
    dart_raws       = results[n*2:n*3]
    price_raws      = results[n*3:]

    def parse(raw, *, tool: str = "krx_market", context: str = "") -> dict:
        """MCP 결과 파싱. 오류는 삼키지 않고 로그에 남긴다 (src/utils/mcp_result.py)."""
        return parse_mcp_result(raw, tool=tool, context=context)

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        ticker_data[ticker] = {
            "financials": parse(financials_raws[i]),
            "consensus":  parse(consensus_raws[i]),
            "disclosure": parse(dart_raws[i]),
            "price":      parse(price_raws[i]),   # 폴백 데이터 (항상 시도)
        }

    return ticker_data


# ─────────────────────────────────────────────────────────
# Step 2: 프롬프트 포맷팅
# ─────────────────────────────────────────────────────────

def _format_prompt(ticker_data: dict) -> str:
    # [REDACTED] Proprietary data formatting logic
    return ""


# ─────────────────────────────────────────────────────────
# Step 3: 에이전트 실행
# ─────────────────────────────────────────────────────────

async def run_fundamental_analyst(target_ticker: str | None = None) -> AnalysisReport:
    """Fundamental Analyst 에이전트 실행 진입점.

    target_ticker: 파이프라인이 다루는 핵심 ticker. 주어지면 분석 대상에 강제 포함.
    """
    data      = await _collect_fundamental_data(target_ticker=target_ticker)
    formatted = _format_prompt(data)

    rag_context = get_context_for_agent(
        agent_name="fundamental_analyst",
        state_vars={
            "ticker": target_ticker or _get_tickers()[0],
            "date":   datetime.now().strftime("%Y-%m-%d"),
        },
    )
    system_content = inject_context_into_prompt(FUNDAMENTAL_SYSTEM_PROMPT, rag_context)

    agent = create_structured_agent(model="gpt-4o-mini")
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ]
    report: AnalysisReport = await agent.ainvoke(messages)
    report.agent_name = "fundamental_analyst"
    return report