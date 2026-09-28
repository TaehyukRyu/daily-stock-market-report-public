"""
Macro Economist Agent
담당: 거시경제 환경 분석 (금리/VIX/원자재/환율/기준금리/정책)
소비 MCP: us_market (get_treasury_yields, get_vix, get_commodity_prices)
          news_economy (get_exchange_rate, get_interest_rate, search_policy_news)
"""

import asyncio
import json
from datetime import datetime
from src.utils.mcp_client import mcp_client
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.base_agent import create_structured_agent
from src.schemas.agent_output import AnalysisReport
from src.rag.context_injection import get_context_for_agent, inject_context_into_prompt
from src.utils.mcp_result import failure_note, has_error, parse_mcp_result


# ─────────────────────────────────────────────────────────
# 시스템 프롬프트 (역할 / 판단 기준 / 출력 형식)
# ─────────────────────────────────────────────────────────

MACRO_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


# ─────────────────────────────────────────────────────────
# Step 1: 데이터 수집
# ─────────────────────────────────────────────────────────

async def _collect_macro_data() -> dict:
    """6개 MCP 도구를 병렬 호출하여 거시경제 데이터를 수집합니다."""

    # US Market MCP: 3개 도구 병렬 호출
    async with mcp_client("src/mcp_servers/us_market/server.py") as us_client:
        yields_raw, vix_raw, commodities_raw = await asyncio.gather(
            us_client.call_tool("get_treasury_yields", {}),
            us_client.call_tool("get_vix", {}),
            us_client.call_tool("get_commodity_prices", {
                "commodity_codes": ["WTI", "BRENT", "GOLD", "COPPER"]
            }),
        )

    # News & Economy MCP: 3개 도구 병렬 호출
    async with mcp_client("src/mcp_servers/news_economy/server.py") as news_client:
        exchange_raw, interest_raw, policy_raw = await asyncio.gather(
            news_client.call_tool("get_exchange_rate", {"days": 30}),
            news_client.call_tool("get_interest_rate", {"periods": 30}),
            news_client.call_tool("search_policy_news", {"max_results_per_query": 5}),
        )

    # FastMCP call_tool은 TextContent 객체를 반환하므로 JSON 파싱 필요
    def parse(raw, *, tool: str = "macro_mcp", context: str = "") -> dict:
        """MCP 결과 파싱. 오류는 삼키지 않고 로그에 남긴다 (src/utils/mcp_result.py)."""
        return parse_mcp_result(raw, tool=tool, context=context)

    return {
        "treasury_yields": parse(yields_raw),
        "vix":             parse(vix_raw),
        "commodities":     parse(commodities_raw),
        "exchange_rate":   parse(exchange_raw),
        "interest_rate":   parse(interest_raw),
        "policy_news":     parse(policy_raw),
    }


# ─────────────────────────────────────────────────────────
# Step 2: 프롬프트 포맷팅
# ─────────────────────────────────────────────────────────

def _format_prompt(data: dict) -> str:
    # [REDACTED] Proprietary data formatting logic
    return ""


# ─────────────────────────────────────────────────────────
# Step 3: 에이전트 실행 (수집 → 포맷 → RAG 주입 → LLM)
# ─────────────────────────────────────────────────────────

async def run_macro_economist() -> AnalysisReport:
    """종목 무관 → 하루 1회만 LLM 호출, 이후 종목은 복사본 재사용 (G4, src/utils/daily_cache.py)."""
    from src.utils.daily_cache import daily_cached
    return await daily_cached("macro_economist", _run_macro_economist_uncached)


async def _run_macro_economist_uncached() -> AnalysisReport:
    data = await _collect_macro_data()
    formatted = _format_prompt(data)

    # ── RAG 컨텍스트 주입 (market_reports + strategy_outcomes 컬렉션)
    rag_context = get_context_for_agent(
        agent_name="macro_economist",
        state_vars={"date": datetime.now().strftime("%Y-%m-%d")},
    )
    system_content = inject_context_into_prompt(MACRO_SYSTEM_PROMPT, rag_context)

    agent = create_structured_agent(model="gpt-4o-mini")
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ]
    report: AnalysisReport = await agent.ainvoke(messages)

    report.agent_name = "macro_economist"

    return report