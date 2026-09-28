"""
KR Market Specialist Agent
담당: 한국 주식시장 분석 (종목별 주가/수급/애널리스트 리포트)
소비 MCP: krx_market (get_stock_price, get_investor_trends, get_analyst_reports)

[v2 변경]
  REPRESENTATIVE_TICKERS 하드코딩 → load_universe() 교체
  _get_tickers(n=10): 유니버스에서 시총 상위 N개 동적 로드
  universe_config.json 없으면 폴백 5개 자동 반환 (load_universe 내부 처리)
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
    """유니버스에서 시총 상위 N개 반환.
    universe_config.json 없으면 load_universe() 내부 폴백 5개 반환.
    """
    return load_universe()[:n]


# ─────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────

KR_MARKET_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


# ─────────────────────────────────────────────────────────
# Step 1: 데이터 수집
# ─────────────────────────────────────────────────────────

async def _collect_kr_market_data(target_ticker: str | None = None) -> dict:
    """대상 종목 1개 × 3개 도구 MCP 호출을 병렬로 실행합니다.

    target_ticker가 None인 경우(단독 디버그 실행 등)에는 유니버스 1순위로 폴백.
    """
    tickers = [target_ticker] if target_ticker else _get_tickers()[:1]

    async with mcp_client("src/mcp_servers/krx_market/server.py") as krx_client:
        price_tasks = [
            krx_client.call_tool("get_stock_price", {"ticker": t, "days": 10})
            for t in tickers
        ]
        trend_tasks = [
            krx_client.call_tool("get_investor_trends", {"ticker": t, "days": 5})
            for t in tickers
        ]
        report_tasks = [
            krx_client.call_tool("get_analyst_reports", {"ticker": t, "days": 90})
            for t in tickers
        ]

        # return_exceptions=True: 일부 MCP 도구가 ToolError(스키마 검증 실패)
        # 등으로 raise해도 다른 데이터는 살리고 진행 (개별 ticker만 손실)
        results = await asyncio.gather(
            *price_tasks, *trend_tasks, *report_tasks,
            return_exceptions=True,
        )

    n = len(tickers)
    price_raws  = results[:n]
    trend_raws  = results[n:n*2]
    report_raws = results[n*2:]

    def parse(raw, *, tool: str = "krx_market", context: str = "") -> dict:
        """MCP 결과 파싱. 오류는 삼키지 않고 로그에 남긴다 (src/utils/mcp_result.py)."""
        return parse_mcp_result(raw, tool=tool, context=context)

    ticker_data = {}
    for i, ticker in enumerate(tickers):
        ticker_data[ticker] = {
            "price":   parse(price_raws[i]),
            "trends":  parse(trend_raws[i]),
            "reports": parse(report_raws[i]),
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

async def run_kr_market_specialist(target_ticker: str | None = None) -> AnalysisReport:
    """KR Market Specialist 에이전트 실행 진입점.

    target_ticker: 파이프라인이 다루는 핵심 ticker. 이 종목만 단독 분석.
    """
    data      = await _collect_kr_market_data(target_ticker=target_ticker)
    formatted = _format_prompt(data)

    rag_context = get_context_for_agent(
        agent_name="kr_market_specialist",
        state_vars={
            "ticker": target_ticker or _get_tickers()[0],
            "date":   datetime.now().strftime("%Y-%m-%d"),
        },
    )
    system_content = inject_context_into_prompt(KR_MARKET_SYSTEM_PROMPT, rag_context)

    agent = create_structured_agent(model="gpt-4o-mini")
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ]
    report: AnalysisReport = await agent.ainvoke(messages)
    report.agent_name = "kr_market_specialist"
    return report


# ─────────────────────────────────────────────────────────
# 단독 디버그 실행
# ─────────────────────────────────────────────────────────

async def _debug_run():
    print("=" * 50)
    tickers = _get_tickers()
    print(f"분석 대상 {len(tickers)}개: {tickers}")

    print("\nStep 1: MCP 데이터 수집 시작")
    try:
        data = await _collect_kr_market_data()
        print("  ✅ 데이터 수집 성공")
        for ticker, d in data.items():
            price_ok  = bool(d.get("price")   and "latest_close" in d.get("price", {}))
            trend_ok  = bool(d.get("trends")  and d["trends"].get("data"))
            report_ok = bool(d.get("reports") and d["reports"].get("reports"))
            print(f"  {ticker}: price={price_ok}, trends={trend_ok}, reports={report_ok}")
    except Exception as e:
        print(f"  ❌ 데이터 수집 실패: {type(e).__name__}: {e}")
        return

    print("\nStep 2: 프롬프트 포맷팅")
    try:
        formatted = _format_prompt(data)
        print(f"  ✅ 포맷팅 성공 (길이: {len(formatted)}자)")
        print(formatted[:500])
    except Exception as e:
        print(f"  ❌ 포맷팅 실패: {type(e).__name__}: {e}")
        return

    print("\nStep 3: LLM 호출")
    try:
        result = await run_kr_market_specialist()
        print(f"  ✅ 성공: {result.recommendation} (신뢰도 {result.confidence})")
    except Exception as e:
        print(f"  ❌ LLM 실패: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(_debug_run())