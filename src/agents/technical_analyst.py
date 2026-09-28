"""
Technical Analyst Agent
담당: 기술적 분석 (이동평균선 / RSI / MACD / 볼린저밴드 / 일목균형표)
소비 MCP: krx_market (get_stock_price)

[v2 변경]
  REPRESENTATIVE_TICKERS 하드코딩 → load_universe() 교체

[v5.4 변경]
  볼린저밴드(Bollinger Bands) 추가
  일목균형표(Ichimoku Cloud) 추가
"""

import asyncio
import json
from datetime import datetime
import pandas as pd
import ta
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


TECHNICAL_SYSTEM_PROMPT = "[REDACTED] Proprietary prompt engineering"


def parse(raw, *, tool: str = "krx_market", context: str = "") -> dict:
    """MCP 결과 파싱. 오류는 삼키지 않고 로그에 남긴다 (src/utils/mcp_result.py)."""
    return parse_mcp_result(raw, tool=tool, context=context)

def _calculate_indicators(price_data: dict) -> dict:
    """OHLCV 데이터 → MA/RSI/MACD/볼린저밴드/일목균형표 계산"""
    try:
        ohlcv = price_data.get("ohlcv", {})

        # 일목균형표는 최소 52일 필요 (기준선 26일 + 선행스팬 26일)
        MIN_DAYS = 52
        if not ohlcv or len(ohlcv) < MIN_DAYS:
            return {"error": f"데이터 부족 ({len(ohlcv)}일, 최소 {MIN_DAYS}일 필요)"}

        df = pd.DataFrame.from_dict(ohlcv, orient="index")
        df.index  = pd.to_datetime(df.index)
        df        = df.sort_index()
        df["close"]  = pd.to_numeric(df["종가"],   errors="coerce")
        df["high"]   = pd.to_numeric(df["고가"],   errors="coerce")
        df["low"]    = pd.to_numeric(df["저가"],   errors="coerce")
        df["volume"] = pd.to_numeric(df["거래량"], errors="coerce")
        df = df.dropna(subset=["close", "high", "low"])

        # ── 이동평균선 ────────────────────────────────────────────
        df["ma5"]  = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()

        # ── RSI ──────────────────────────────────────────────────
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

        # ── MACD ─────────────────────────────────────────────────
        macd = ta.trend.MACD(df["close"])
        df["macd"]        = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"]   = macd.macd_diff()

        # ── 볼린저밴드 (window=20, std=2) ────────────────────────
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"]  = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"]  = bb.bollinger_lband()
        df["bb_width"]  = bb.bollinger_wband()   # (upper-lower)/middle * 100
        df["bb_pct"]    = bb.bollinger_pband()   # (close-lower)/(upper-lower)

        # ── 일목균형표 (전환선9, 기준선26, 선행스팬B52) ───────────
        # ta 라이브러리 IchimokuIndicator 사용
        ichimoku = ta.trend.IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=9,    # 전환선
            window2=26,   # 기준선
            window3=52,   # 선행스팬B
        )
        df["tenkan"]      = ichimoku.ichimoku_conversion_line()  # 전환선
        df["kijun"]       = ichimoku.ichimoku_base_line()        # 기준선
        df["senkou_a"]    = ichimoku.ichimoku_a()                # 선행스팬A (26일 선행)
        df["senkou_b"]    = ichimoku.ichimoku_b()                # 선행스팬B (26일 선행)

        # 후행스팬: 당일 종가를 26일 뒤로 이동 (역방향으로 26일 전 데이터와 비교)
        df["chikou"]      = df["close"].shift(-26)               # 후행스팬 (26일 후행)
        df["close_26ago"] = df["close"].shift(26)                # 26일 전 종가

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        # ── 크로스 신호 ───────────────────────────────────────────
        golden_cross = prev["ma5"] <= prev["ma20"] and latest["ma5"] > latest["ma20"]
        dead_cross   = prev["ma5"] >= prev["ma20"] and latest["ma5"] < latest["ma20"]
        macd_golden  = (
            pd.notna(prev["macd"]) and pd.notna(prev["macd_signal"]) and
            pd.notna(latest["macd"]) and pd.notna(latest["macd_signal"]) and
            prev["macd"] <= prev["macd_signal"] and latest["macd"] > latest["macd_signal"]
        )
        macd_dead = (
            pd.notna(prev["macd"]) and pd.notna(prev["macd_signal"]) and
            pd.notna(latest["macd"]) and pd.notna(latest["macd_signal"]) and
            prev["macd"] >= prev["macd_signal"] and latest["macd"] < latest["macd_signal"]
        )

        # ── MA 정배열/역배열 ──────────────────────────────────────
        ma5_ok  = pd.notna(latest["ma5"])
        ma20_ok = pd.notna(latest["ma20"])
        ma60_ok = pd.notna(latest["ma60"])

        # ── 볼린저밴드 신호 ───────────────────────────────────────
        bb_signal = "중립"
        if pd.notna(latest["bb_pct"]):
            if latest["bb_pct"] >= 1.0:
                bb_signal = "과매수 (상단 돌파)"
            elif latest["bb_pct"] <= 0.0:
                bb_signal = "과매도 (하단 이탈)"
            elif latest["bb_pct"] >= 0.8:
                bb_signal = "상단 근접 (조정 경계)"
            elif latest["bb_pct"] <= 0.2:
                bb_signal = "하단 근접 (반등 기대)"

        bb_squeeze = (
            pd.notna(latest["bb_width"]) and
            pd.notna(df["bb_width"].rolling(20).mean().iloc[-1]) and
            latest["bb_width"] < df["bb_width"].rolling(20).mean().iloc[-1] * 0.7
        )

        # ── 일목균형표 신호 ───────────────────────────────────────
        cloud_top    = None
        cloud_bottom = None
        cloud_signal = "데이터 부족"
        tenkan_kijun_signal = "데이터 부족"

        # 현재 구름대는 26일 선행이므로, 현재 기준 → iloc[-27] 행의 senkou_a/b 사용
        if len(df) >= 27:
            current_cloud_row = df.iloc[-27]
            if pd.notna(current_cloud_row["senkou_a"]) and pd.notna(current_cloud_row["senkou_b"]):
                cloud_top    = max(float(current_cloud_row["senkou_a"]), float(current_cloud_row["senkou_b"]))
                cloud_bottom = min(float(current_cloud_row["senkou_a"]), float(current_cloud_row["senkou_b"]))
                close = float(latest["close"])
                if close > cloud_top:
                    cloud_signal = "구름 위 (강세)"
                elif close < cloud_bottom:
                    cloud_signal = "구름 아래 (약세)"
                else:
                    cloud_signal = "구름 속 (횡보/불확실)"

        if pd.notna(latest["tenkan"]) and pd.notna(latest["kijun"]):
            if latest["tenkan"] > latest["kijun"]:
                tenkan_kijun_signal = "전환선 > 기준선 (단기 상승)"
            elif latest["tenkan"] < latest["kijun"]:
                tenkan_kijun_signal = "전환선 < 기준선 (단기 하락)"
            else:
                tenkan_kijun_signal = "전환선 = 기준선 (방향성 탐색)"

        return {
            # ── 기본 가격 ──────────────────────────────────────
            "close":        round(float(latest["close"]), 0),

            # ── 이동평균선 ─────────────────────────────────────
            "ma5":          round(float(latest["ma5"]),  0) if ma5_ok  else None,
            "ma20":         round(float(latest["ma20"]), 0) if ma20_ok else None,
            "ma60":         round(float(latest["ma60"]), 0) if ma60_ok else None,
            "golden_cross": bool(golden_cross),
            "dead_cross":   bool(dead_cross),
            "ma_alignment": (
                "정배열" if (ma5_ok and ma20_ok and ma60_ok
                            and latest["ma5"] > latest["ma20"] > latest["ma60"])
                else "역배열" if (ma5_ok and ma20_ok and ma60_ok
                                and latest["ma5"] < latest["ma20"] < latest["ma60"])
                else "혼조"
            ),

            # ── RSI ────────────────────────────────────────────
            "rsi": round(float(latest["rsi"]), 2) if pd.notna(latest["rsi"]) else None,

            # ── MACD ───────────────────────────────────────────
            "macd":         round(float(latest["macd"]),        2) if pd.notna(latest["macd"])        else None,
            "macd_signal":  round(float(latest["macd_signal"]), 2) if pd.notna(latest["macd_signal"]) else None,
            "macd_hist":    round(float(latest["macd_hist"]),   2) if pd.notna(latest["macd_hist"])   else None,
            "macd_golden":  bool(macd_golden),
            "macd_dead":    bool(macd_dead),

            # ── 볼린저밴드 ─────────────────────────────────────
            "bb_upper":     round(float(latest["bb_upper"]),  0) if pd.notna(latest["bb_upper"])  else None,
            "bb_middle":    round(float(latest["bb_middle"]), 0) if pd.notna(latest["bb_middle"]) else None,
            "bb_lower":     round(float(latest["bb_lower"]),  0) if pd.notna(latest["bb_lower"])  else None,
            "bb_pct":       round(float(latest["bb_pct"]),    3) if pd.notna(latest["bb_pct"])    else None,
            "bb_width":     round(float(latest["bb_width"]),  3) if pd.notna(latest["bb_width"])  else None,
            "bb_signal":    bb_signal,
            "bb_squeeze":   bool(bb_squeeze),  # True = 변동성 수축 → 돌파 임박

            # ── 일목균형표 ─────────────────────────────────────
            "tenkan":       round(float(latest["tenkan"]), 0) if pd.notna(latest["tenkan"]) else None,
            "kijun":        round(float(latest["kijun"]),  0) if pd.notna(latest["kijun"])  else None,
            "cloud_top":    round(cloud_top,    0) if cloud_top    is not None else None,
            "cloud_bottom": round(cloud_bottom, 0) if cloud_bottom is not None else None,
            "cloud_signal": cloud_signal,         # 구름 위/아래/속
            "tenkan_kijun": tenkan_kijun_signal,  # 전환선 vs 기준선
        }
    except Exception as e:
        return {"error": str(e)}


async def _collect_technical_data(target_ticker: str | None = None) -> dict:
    """대상 종목 1개의 OHLCV 데이터로 기술적 지표를 계산합니다.

    target_ticker가 None인 경우(단독 디버그 실행 등)에는 유니버스 1순위로 폴백.
    """
    tickers = [target_ticker] if target_ticker else _get_tickers()[:1]

    # 일목균형표 52일 + 여유분 → 120일 유지
    async with mcp_client("src/mcp_servers/krx_market/server.py") as client:
        tasks = [
            client.call_tool("get_stock_price", {"ticker": t, "days": 120})
            for t in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    technical_data = {}
    for ticker, raw in zip(tickers, results):
        if isinstance(raw, Exception):
            technical_data[ticker] = parse(raw)   # 예외도 로그에 남기고 {"error": ...}로 받는다
            continue
        technical_data[ticker] = _calculate_indicators(parse(raw))

    return technical_data


def _format_prompt(data: dict) -> str:
    # [REDACTED] Proprietary data formatting logic
    return ""


async def run_technical_analyst(target_ticker: str | None = None) -> AnalysisReport:
    """Technical Analyst 에이전트 실행 진입점.

    target_ticker: 파이프라인이 다루는 핵심 ticker. 이 종목만 단독 분석.
    """
    data      = await _collect_technical_data(target_ticker=target_ticker)
    formatted = _format_prompt(data)

    rag_context = get_context_for_agent(
        agent_name="technical_analyst",
        state_vars={
            "ticker": target_ticker or _get_tickers()[0],
            "date":   datetime.now().strftime("%Y-%m-%d"),
        },
    )
    system_content = inject_context_into_prompt(TECHNICAL_SYSTEM_PROMPT, rag_context)

    agent  = create_structured_agent(model="gpt-4o-mini")
    report: AnalysisReport = await agent.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(content=formatted),
    ])
    report.agent_name = "technical_analyst"

    # LLM이 Optional 필드인 selection_rationale을 누락하는 경우가 있어 보강한다.
    # 종목 단위 분석이므로 항상 채워져 있어야 한다.
    if not report.selection_rationale:
        report.selection_rationale = (
            report.reasoning[-1] if report.reasoning
            else "기술적 지표 기준 종목 선정 보류"
        )

    return report