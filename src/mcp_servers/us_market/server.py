# ── stdio 서버 가드 (2026-09-11) ─────────────────────────────────────────
# 이 파일이 서브프로세스로 실행될 때 stdout은 JSON-RPC 채널이다. pykrx는 **import 시점**에
# "KRX 로그인 시도..."를 print()로 stdout에 찍으므로(2차 CI 실행에서도 80건 오염), 어떤 import보다
# 먼저 print를 stderr로 돌린다. 모듈로 import될 때(테스트)는 손대지 않는다.
if __name__ == "__main__":
    import builtins as _b, functools as _ft, sys as _sys
    _b.print = _ft.partial(_b.print, file=_sys.stderr)

from fastmcp import FastMCP
from dotenv import load_dotenv
import yfinance as yf
import os

load_dotenv(dotenv_path=".env")

mcp = FastMCP("us-market")

# ─────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────

def _ticker_info(symbol: str) -> dict:
    """yfinance Ticker 기본 정보 추출"""
    t = yf.Ticker(symbol)
    hist = t.history(period="5d")
    if hist.empty:
        return {"error": f"{symbol} 데이터 없음"}
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else latest
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    return {
        "symbol": symbol,
        "close": round(float(latest["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "volume": int(latest["Volume"]),
        "date": str(hist.index[-1].date()),
    }


# ─────────────────────────────────────────
# Tool 1: 미국 개별 종목
# ─────────────────────────────────────────

@mcp.tool()
def get_us_stock(symbol: str) -> dict:
    """미국 개별 종목 최근 주가 조회 (예: AAPL, NVDA, TSLA)"""
    return _ticker_info(symbol)


# ─────────────────────────────────────────
# Tool 2: S&P500 지수
# ─────────────────────────────────────────

@mcp.tool()
def get_sp500_data(days: int = 30) -> dict:
    """S&P500 지수 최근 데이터 조회"""
    t = yf.Ticker("^GSPC")
    hist = t.history(period=f"{days}d")
    if hist.empty:
        return {"error": "S&P500 데이터 없음"}
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else latest
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    return {
        "symbol": "^GSPC",
        "name": "S&P500",
        "close": round(float(latest["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "date": str(hist.index[-1].date()),
        "period_high": round(float(hist["High"].max()), 2),
        "period_low": round(float(hist["Low"].min()), 2),
    }


# ─────────────────────────────────────────
# Tool 3: 미국 국채 수익률
# ─────────────────────────────────────────

@mcp.tool()
def get_treasury_yields() -> dict:
    """미국 국채 수익률 조회 (2년, 10년, 30년)"""
    symbols = {
        "2y": "^IRX",
        "10y": "^TNX",
        "30y": "^TYX",
    }
    result = {}
    for label, sym in symbols.items():
        info = _ticker_info(sym)
        result[label] = info
    return result


# ─────────────────────────────────────────
# Tool 4: VIX 공포지수
# ─────────────────────────────────────────

@mcp.tool()
def get_vix() -> dict:
    """VIX 공포지수 조회"""
    return _ticker_info("^VIX")


# ─────────────────────────────────────────
# Tool 5: 원자재 가격 (FRED Tier1 + yfinance Tier2)
# ─────────────────────────────────────────

@mcp.tool()
def get_commodity_prices(commodity_codes: list[str] = None) -> dict:
    """
    원자재 가격 조회.
    에너지(WTI, BRENT, NG): FRED API (Tier1)
    귀금속/비철금속(GOLD, SILVER, COPPER): yfinance (Tier2)
    FRED 실패 시 yfinance로 Fallback.
    """
    if commodity_codes is None:
        commodity_codes = ["WTI", "BRENT", "GOLD", "SILVER", "COPPER", "DXY"]

    fred_map = {
        "WTI": "DCOILWTICO",
        "BRENT": "DCOILBRENTEU",
        "NG": "DHHNGSP",
        "DXY": "DTWEXBGS",
    }

    yf_map = {
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "COPPER": "HG=F",
        "WTI": "CL=F",
        "BRENT": "BZ=F",
        "DXY": "DX-Y.NYB",
    }

    result = {}

    for code in commodity_codes:
        code = code.upper()

        if code in fred_map:
            try:
                from fredapi import Fred
                fred = Fred(api_key=os.getenv("FRED_API_KEY"))
                series = fred.get_series(fred_map[code], observation_start="2025-01-01")
                series = series.dropna()
                if not series.empty:
                    latest_val = float(series.iloc[-1])
                    prev_val = float(series.iloc[-2]) if len(series) >= 2 else latest_val
                    change_pct = ((latest_val - prev_val) / prev_val) * 100
                    result[code] = {
                        "source": "FRED",
                        "price": round(latest_val, 2),
                        "change_pct": round(change_pct, 2),
                        "date": str(series.index[-1].date()),
                        "stale": False,
                    }
                    continue
            except Exception as e:
                result[code] = {"source": "FRED_FAILED", "error": str(e)}

        if code in yf_map:
            try:
                info = _ticker_info(yf_map[code])
                if "error" not in info:
                    result[code] = {
                        "source": "yfinance",
                        "price": info["close"],
                        "change_pct": info["change_pct"],
                        "date": info["date"],
                        "stale": False,
                    }
                else:
                    result[code] = {"source": "yfinance", "error": info["error"], "stale": True}
            except Exception as e:
                result[code] = {"source": "yfinance_failed", "error": str(e), "stale": True}
        elif code not in result:
            result[code] = {"error": f"{code} 지원하지 않는 코드"}

    return result


# ─────────────────────────────────────────
# [v5.2 신규] Tool: 중국 주요 지표
# ─────────────────────────────────────────

@mcp.tool()
def get_china_indicators() -> dict:
    """
    중국 주요 지표 조회.
    - 항셍지수(^HSI): 홍콩 증시, 중국 경기 프록시
    - 달러-위안화 환율(USDCNY=X): 위안화 강약 — 한국 수출주 영향
    """
    result = {}

    hang_seng = _ticker_info("^HSI")
    result["hang_seng"] = {
        **hang_seng,
        "note": "홍콩 항셍지수 — 중국 경기 선행 지표로 활용",
    }

    usd_cny = _ticker_info("USDCNY=X")
    result["usd_cny"] = {
        **usd_cny,
        "note": "달러-위안화 환율 — 수치 상승 = 위안화 약세 = 한국 수출주 경쟁력 유리",
    }

    return result


# ─────────────────────────────────────────
# [v5.2 신규] Tool: 한국 ETF 자금 흐름
# ─────────────────────────────────────────

@mcp.tool()
def get_korea_etf_flow(days: int = 30) -> dict:
    """
    글로벌 ETF 자금 흐름 간접 추적 (EWY — iShares MSCI South Korea ETF).
    외국인 자금의 한국 시장 진출입 방향성을 EWY 거래량으로 추정.
    """
    t = yf.Ticker("EWY")
    hist = t.history(period=f"{days}d")
    if hist.empty:
        return {"error": "EWY 데이터 없음"}

    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else latest
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    avg_vol_5d  = float(hist["Volume"].tail(5).mean())
    avg_vol_all = float(hist["Volume"].mean())
    vol_ratio = round(avg_vol_5d / avg_vol_all, 2) if avg_vol_all > 0 else 1.0

    if vol_ratio >= 1.2:
        signal = "외국인 수급 관심 증가 (매수 압력 가능성)"
    elif vol_ratio <= 0.8:
        signal = "외국인 수급 관심 감소 (이탈 압력 가능성)"
    else:
        signal = "중립 (평균 수준 거래량)"

    return {
        "symbol": "EWY",
        "name": "iShares MSCI South Korea ETF",
        "close": round(float(latest["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "volume_today": int(latest["Volume"]),
        "avg_vol_5d": int(avg_vol_5d),
        "avg_vol_30d": int(avg_vol_all),
        "vol_trend_ratio": vol_ratio,
        "signal": signal,
        "date": str(hist.index[-1].date()),
    }


# ─────────────────────────────────────────
# [v5.3 신규] Tool: Fed 금리 결정 확률
# ─────────────────────────────────────────

@mcp.tool()
def get_fedwatch_probabilities() -> dict:
    """
    Fed 금리 결정 확률 및 현재 금리 조회.

    데이터 소스:
      - 현재 금리: FRED DFEDTARL / DFEDTARU (Fed Target Rate 하단/상단)
      - 확률 추정: CME FedWatch 공개 API 시도 → 실패 시 SOFR OIS 기반 추정
      - FOMC 일정: 연준 공식 2026년 일정 (하드코딩, 매년 1월 공식 발표)

    반환값 해석:
      target_range: 현재 Fed Funds Rate 목표 범위 (예: "4.25%~4.50%")
      next_meeting: 다음 FOMC 회의 날짜
      probabilities: 다음 회의 금리 결정 확률 (인상/동결/인하)
        → CME API 실패 시 "데이터 없음"으로 fallback
    """
    import requests
    from datetime import datetime

    fred_api_key = os.getenv("FRED_API_KEY")

    # ── Step 1: 현재 Fed Target Rate (FRED) ──────────────────────
    current_lower = None
    current_upper = None

    if fred_api_key:
        try:
            for series_id, key in [("DFEDTARL", "lower"), ("DFEDTARU", "upper")]:
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": fred_api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1,
                    },
                    timeout=10,
                )
                obs = resp.json().get("observations", [])
                if obs and obs[0]["value"] != ".":
                    val = round(float(obs[0]["value"]), 2)
                    if key == "lower":
                        current_lower = val
                    else:
                        current_upper = val
        except Exception:
            pass

    target_range = (
        f"{current_lower}%~{current_upper}%"
        if current_lower is not None and current_upper is not None
        else "조회 실패"
    )

    # ── Step 2: 2026년 FOMC 예정일 (연준 공식 발표 기준) ──────────
    # 출처: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    FOMC_2026 = [
        "2026-01-28", "2026-03-18", "2026-05-06",
        "2026-06-17", "2026-07-29", "2026-09-16",
        "2026-10-28", "2026-12-09",
    ]

    today = datetime.now().strftime("%Y-%m-%d")
    upcoming = [d for d in FOMC_2026 if d >= today]
    next_meeting = upcoming[0] if upcoming else "2026년 일정 종료"
    days_to_next = None
    if upcoming:
        from datetime import date
        delta = date.fromisoformat(upcoming[0]) - date.today()
        days_to_next = delta.days

    # ── Step 3: CME FedWatch 확률 추정 ───────────────────────────
    # CME 공개 API 시도 (인증 불필요 엔드포인트)
    probabilities = None
    prob_source = None

    try:
        cme_resp = requests.get(
            "https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/V1/1/FedWatch",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        if cme_resp.status_code == 200:
            cme_data = cme_resp.json()
            # CME 응답 구조에 따라 파싱 (구조 변경 시 fallback)
            if isinstance(cme_data, dict) and "probabilities" in cme_data:
                probabilities = cme_data["probabilities"]
                prob_source = "CME FedWatch API"
    except Exception:
        pass

    # CME 실패 시: SOFR OIS 기반 간이 추정
    if probabilities is None and fred_api_key and current_upper is not None:
        try:
            # SOFR 1개월 선도금리로 다음 회의 기대 금리 추정
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "SOFR",
                    "api_key": fred_api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
                timeout=10,
            )
            sofr_obs = resp.json().get("observations", [])
            valid = [o for o in sofr_obs if o["value"] != "."]
            if valid:
                sofr_rate = float(valid[0]["value"])
                spread = round(sofr_rate - current_upper, 2)
                if spread >= 0.15:
                    probabilities = {"hike": "높음", "hold": "낮음", "cut": "없음"}
                elif spread <= -0.15:
                    probabilities = {"hike": "없음", "hold": "낮음", "cut": "높음"}
                else:
                    probabilities = {"hike": "낮음", "hold": "높음", "cut": "낮음"}
                prob_source = f"SOFR OIS 간이 추정 (SOFR={sofr_rate}%, Target Upper={current_upper}%)"
        except Exception:
            pass

    return {
        "target_range": target_range,
        "current_lower": current_lower,
        "current_upper": current_upper,
        "next_fomc": next_meeting,
        "days_to_next_fomc": days_to_next,
        "fomc_schedule_2026": FOMC_2026,
        "probabilities": probabilities or "CME API 및 SOFR 추정 모두 실패",
        "prob_source": prob_source or "없음",
        "note": "확률은 참고용 추정치입니다. 정확한 확률은 CME FedWatch 공식 페이지를 확인하세요.",
    }


if __name__ == "__main__":
    mcp.run()
