import asyncio
import json
from datetime import datetime
from src.utils.mcp_client import mcp_client
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.base_agent import create_structured_agent
from src.schemas.agent_output import AnalysisReport
from src.rag.context_injection import get_context_for_agent, inject_context_into_prompt
from src.utils.mcp_result import failure_note, parse_mcp_result


US_MARKET_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"

BIGTECH_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]


def parse(raw, *, tool: str = "us_market", context: str = "") -> dict:
    """MCP 결과 파싱. 오류는 삼키지 않고 로그에 남긴다 (src/utils/mcp_result.py)."""
    return parse_mcp_result(raw, tool=tool, context=context)

async def _collect_us_market_data() -> dict:
    async with mcp_client("src/mcp_servers/us_market/server.py") as client:
        sp500_task    = client.call_tool("get_sp500_data",       {"days": 30})
        vix_task      = client.call_tool("get_vix",              {})           # ✅ Fix: days 파라미터 제거
        treasury_task = client.call_tool("get_treasury_yields",  {})           # ✅ Fix: days 파라미터 제거
        stock_tasks   = [
            client.call_tool("get_us_stock", {"symbol": t})                   # ✅ Fix: ticker→symbol, days 제거
            for t in BIGTECH_TICKERS
        ]

        results = await asyncio.gather(
            sp500_task, vix_task, treasury_task, *stock_tasks,
            return_exceptions=True
        )

    sp500_raw, vix_raw, treasury_raw = results[0], results[1], results[2]
    stock_raws = results[3:]

    # 예외도 parse에 넘긴다 — parse_mcp_result가 로그에 남기고 {"error": ...}로 바꾼다.
    # (이전: 예외면 {} → 로그도 없이 사라졌다)
    sp500    = parse(sp500_raw)
    vix      = parse(vix_raw)
    treasury = parse(treasury_raw)

    stocks = {}
    for ticker, raw in zip(BIGTECH_TICKERS, stock_raws):
        stocks[ticker] = parse(raw)

    return {
        "sp500":   sp500,
        "vix":     vix,
        "treasury": treasury,
        "bigtech": stocks,
    }


def _format_prompt(data: dict) -> str:
    # [REDACTED] Proprietary data formatting logic
    return ""


async def run_us_market_specialist() -> AnalysisReport:
    """종목 무관 → 하루 1회만 LLM 호출, 이후 종목은 복사본 재사용 (G4, src/utils/daily_cache.py)."""
    from src.utils.daily_cache import daily_cached
    return await daily_cached("us_market_specialist", _run_us_market_specialist_uncached)


async def _run_us_market_specialist_uncached() -> AnalysisReport:
    data      = await _collect_us_market_data()
    formatted = _format_prompt(data)

    # ── RAG 컨텍스트 주입 (market_reports + news_articles 컬렉션)
    rag_context    = get_context_for_agent(
        agent_name="us_market_specialist",
        state_vars={"date": datetime.now().strftime("%Y-%m-%d")},
    )
    system_content = inject_context_into_prompt(US_MARKET_SYSTEM_PROMPT, rag_context)

    agent = create_structured_agent(model="gpt-4o-mini")

    report: AnalysisReport = await agent.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ])

    report.agent_name = "us_market_specialist"

    return report