"""
Quant Analyst Agent
담당: 정량 분석 (밸류에이션 + 모멘텀 + 목표주가 괴리율)
소비 MCP: krx_market (get_stock_price, get_analyst_reports)

[v2 변경]
  REPRESENTATIVE_TICKERS 하드코딩 → load_universe() 교체
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
from src.utils.mcp_result import parse_mcp_result


# ─────────────────────────────────────────────────────────
# 분석 대상 종목 — 유니버스에서 동적 로드
# ─────────────────────────────────────────────────────────

def _get_tickers(n: int = 10) -> list[str]:
    return load_universe()[:n]


QUANT_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


def parse(raw, *, tool: str = "krx_market", context: str = "") -> dict:
    """MCP 결과 파싱. 오류는 삼키지 않고 로그에 남긴다 (src/utils/mcp_result.py)."""
    return parse_mcp_result(raw, tool=tool, context=context)


def _calculate_momentum(price_data: dict) -> dict:
    """주가 데이터에서 모멘텀 지표 계산"""
    try:
        ohlcv = price_data.get("ohlcv", {})
        if not ohlcv or len(ohlcv) < 5:
            return {"error": "데이터 부족"}

        closes = [v["종가"] for v in ohlcv.values()]
        current       = closes[-1]
        price_20d_ago = closes[-20] if len(closes) >= 20 else closes[0]
        price_5d_ago  = closes[-5]  if len(closes) >= 5  else closes[0]

        volumes        = [v["거래량"] for v in ohlcv.values()]
        avg_volume_20d = (
            sum(volumes[-20:]) / len(volumes[-20:])
            if len(volumes) >= 20
            else sum(volumes) / len(volumes)
        )
        latest_volume = volumes[-1]

        return {
            "current_price":    current,
            "momentum_20d_pct": round((current - price_20d_ago) / price_20d_ago * 100, 2),
            "momentum_5d_pct":  round((current - price_5d_ago)  / price_5d_ago  * 100, 2),
            "volume_ratio":     round(latest_volume / avg_volume_20d, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _extract_target_price(analyst_data: dict) -> dict:
    """애널리스트 리포트에서 목표주가 괴리율 계산"""
    try:
        reports = analyst_data.get("reports", [])
        if not reports:
            return {"target_price": None, "upside_pct": None, "report_count": 0}

        target_prices = [
            r.get("target_price") for r in reports
            if r.get("target_price") and r.get("target_price") > 0
        ]
        if not target_prices:
            return {"target_price": None, "upside_pct": None, "report_count": len(reports)}

        avg_target    = sum(target_prices) / len(target_prices)
        current_price = analyst_data.get("current_price") or reports[0].get("current_price")

        upside = None
        if current_price and current_price > 0:
            upside = round((avg_target - current_price) / current_price * 100, 2)

        return {
            "target_price": round(avg_target),
            "upside_pct":   upside,
            "report_count": len(reports),
        }
    except Exception as e:
        return {"error": str(e)}


async def _collect_quant_data(target_ticker: str | None = None) -> dict:
    """대상 종목 1개의 정량 데이터(가격/애널리스트)를 수집합니다.

    target_ticker가 None인 경우(단독 디버그 실행 등)에는 유니버스 1순위로 폴백.
    """
    tickers = [target_ticker] if target_ticker else _get_tickers()[:1]

    async with mcp_client("src/mcp_servers/krx_market/server.py") as client:
        price_tasks = [
            client.call_tool("get_stock_price",    {"ticker": t, "days": 120})
            for t in tickers
        ]
        analyst_tasks = [
            client.call_tool("get_analyst_reports", {"ticker": t, "days": 90})
            for t in tickers
        ]
        results = await asyncio.gather(*price_tasks, *analyst_tasks, return_exceptions=True)

    n               = len(tickers)
    price_results   = results[:n]
    analyst_results = results[n:]

    quant_data = {}
    for i, ticker in enumerate(tickers):
        price_raw   = price_results[i]
        analyst_raw = analyst_results[i]

        # 예외도 parse에 넘긴다 — parse_mcp_result가 로그에 남긴다 (이전: {}로 조용히 사라짐)
        price_data   = parse(price_raw)
        analyst_data = parse(analyst_raw)

        quant_data[ticker] = {
            "name":     price_data.get("name", ticker),
            "momentum": _calculate_momentum(price_data),
            "analyst":  _extract_target_price(analyst_data),
        }

    return quant_data


def _format_prompt(data: dict) -> str:
    # [REDACTED] Proprietary data formatting logic
    return ""


async def run_quant_analyst(target_ticker: str | None = None) -> AnalysisReport:
    """Quant Analyst 에이전트 실행 진입점.

    target_ticker: 파이프라인이 다루는 핵심 ticker. 이 종목만 단독 분석.
    """
    data      = await _collect_quant_data(target_ticker=target_ticker)
    formatted = _format_prompt(data)

    rag_context = get_context_for_agent(
        agent_name="quant_analyst",
        state_vars={
            "ticker": target_ticker or _get_tickers()[0],
            "date":   datetime.now().strftime("%Y-%m-%d"),
        },
    )
    system_content = inject_context_into_prompt(QUANT_SYSTEM_PROMPT, rag_context)

    agent  = create_structured_agent(model="gpt-4o-mini")
    report: AnalysisReport = await agent.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ])
    report.agent_name = "quant_analyst"

    # LLM이 Optional 필드인 selection_rationale을 누락하는 경우가 있어 보강한다.
    # 종목 단위 분석이므로 항상 채워져 있어야 한다.
    if not report.selection_rationale:
        report.selection_rationale = (
            report.reasoning[-1] if report.reasoning
            else "정량 데이터 기준 종목 선정 보류"
        )

    return report