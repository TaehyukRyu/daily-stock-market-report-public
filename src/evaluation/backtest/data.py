"""
data.py — 가격·벤치마크·유니버스.

[출처]
  가격      yfinance {code}.KS/.KQ (auto_adjust=False → Adj Close·Volume 사용)
  벤치마크  yfinance ^KS200
  후보 풀   네이버 증권 시총 순위 API (현재 코스피 상위 300, 코스닥 상위 100)
            https://m.stock.naver.com/api/stocks/marketValue/{KOSPI|KOSDAQ}?page=N&pageSize=100
  pykrx는 로컬에 KRX 자격증명이 없어 과거 데이터를 못 받는다 (사전 등록 §1).

[survivorship 한계 — 사전 등록 §1]
  현재 상장 종목만 풀에 있다. 상장폐지·거래정지 종목 누락 → 성과 위 방향 편향.
  과거 시총 = AdjClose_t × 현재 주식수 (주식수 = 현재 시총 / 현재 종가).

캐시: data/backtest/ (git 미추적). 가격 parquet, 풀 JSON.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/backtest")
POOL_KOSPI_PAGES  = 3     # 상위 300
POOL_KOSDAQ_PAGES = 1     # 상위 100
KOSPI_TOP_N  = 100        # universe_builder와 동일
KOSDAQ_TOP_N = 20
# CR-1 (doc/2026-09-13_change-requests.md): yfinance ^KS200은 2026-07-16에서 끊긴다.
# 같은 지수를 추종하는 KODEX 200 ETF의 조정 시가/종가를 쓴다 (분배금 포함 → 전략 조정종가와 대칭).
BENCHMARK = "069500.KS"
BENCHMARK_LABEL = "KOSPI200 (KODEX 200 ETF 069500.KS, 조정가)"
_NAVER = "https://m.stock.naver.com/api/stocks/marketValue/{market}?page={page}&pageSize=100"
_H = {"User-Agent": "Mozilla/5.0"}


def _num(s) -> float:
    return float(str(s).replace(",", "").replace("+", ""))


# ── 후보 풀 ─────────────────────────────────────────────────────────────

def fetch_pool(cache_dir: Path = CACHE_DIR, refresh: bool = False,
               kospi_pages: int = POOL_KOSPI_PAGES, kosdaq_pages: int = POOL_KOSDAQ_PAGES,
               stocks_only: bool = False) -> pd.DataFrame:
    """현재 시총 순위 풀. columns: code, name, market, shares(현재 주식수, 억원/원 환산 무관 비율).
    캐시 파일명에 페이지 수를 넣는다 — 기본(3,1)은 기존 pool.json 그대로.
    (2026-09-23) 풀을 넓혔는데 옛 캐시를 조용히 읽는 일을 막는다.
    stocks_only: ETF·ETN을 뺀다(stockEndType == 'stock'). 코스피 목록 상위 800 중 주식은 480이다
    (2026-09-23 실측). 기본 False는 과거 결과 재현용."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    default = (kospi_pages, kosdaq_pages) == (POOL_KOSPI_PAGES, POOL_KOSDAQ_PAGES) and not stocks_only
    name = f"pool_k{kospi_pages}_q{kosdaq_pages}{'_stocks' if stocks_only else ''}.json"
    f = cache_dir / ("pool.json" if default else name)
    if f.exists() and not refresh:
        return pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
    rows = []
    for market, pages in (("KOSPI", kospi_pages), ("KOSDAQ", kosdaq_pages)):
        for p in range(1, pages + 1):
            d = requests.get(_NAVER.format(market=market, page=p), headers=_H, timeout=15).json()
            for s in d.get("stocks", []):
                if stocks_only and s.get("stockEndType") != "stock":
                    continue
                close = _num(s["closePrice"])
                mv = _num(s["marketValue"])          # 백만원 단위
                if close <= 0 or mv <= 0:
                    continue
                rows.append({
                    "code": s["itemCode"], "name": s["stockName"], "market": market,
                    "shares": mv * 1_000_000 / close,   # 현재 주식수(추정)
                    "marketValue_now": mv, "close_now": close,
                })
            time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates("code")
    f.write_text(json.dumps(df.to_dict("records"), ensure_ascii=False), encoding="utf-8")
    logger.info(f"[backtest.data] 풀 {len(df)}종목 (KOSPI {sum(df.market=='KOSPI')}, KOSDAQ {sum(df.market=='KOSDAQ')})")
    return df


def apply_universe_filters(pool: pd.DataFrame) -> pd.DataFrame:
    """현행 universe_builder 필터 그대로 (우선주·바이오). filters.py는 index=code, '종목명' 컬럼을 기대한다."""
    from src.universe.filters import filter_preferred_stocks, filter_bio_pharma
    df = pool.set_index("code").copy()
    df["종목명"] = df["name"]
    df = filter_preferred_stocks(df)
    df = filter_bio_pharma(df)
    return df.reset_index()


# ── 가격 ────────────────────────────────────────────────────────────────

def _symbol(code: str, market: str) -> str:
    return f"{code}.{'KS' if market == 'KOSPI' else 'KQ'}"


def fetch_prices(pool: pd.DataFrame, start: str, end: str,
                 cache_dir: Path = CACHE_DIR, refresh: bool = False,
                 tag: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """{'adj_close': DataFrame(date × code), 'open': ..., 'close': ..., 'volume': ...}
    tag: 풀 해시. 주면 캐시 파일명에 들어가 풀이 바뀌면 새로 받는다 (2026-09-23)."""
    import yfinance as yf
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (f"prices_{start}_{end}_{tag}.pkl" if tag else f"prices_{start}_{end}.pkl")
    if f.exists() and not refresh:
        wide = pd.read_pickle(f)
    else:
        symbols = [_symbol(c, m) for c, m in zip(pool["code"], pool["market"])]
        t0 = time.time()
        raw = yf.download(symbols, start=start, end=end, auto_adjust=False, progress=False, threads=True)
        logger.info(f"[backtest.data] yfinance {len(symbols)}종목 {time.time()-t0:.1f}s")
        # MultiIndex (field, symbol) → 평면화 후 저장
        wide = raw.copy()
        wide.columns = [f"{fld}|{sym}" for fld, sym in wide.columns]
        wide.to_pickle(f)
    sym2code = {_symbol(c, m): c for c, m in zip(pool["code"], pool["market"])}
    out: dict[str, pd.DataFrame] = {}
    for fld, key in (("Adj Close", "adj_close"), ("Open", "open"), ("Close", "close"), ("Volume", "volume")):
        cols = [c for c in wide.columns if c.startswith(f"{fld}|")]
        df = wide[cols].copy()
        df.columns = [sym2code.get(c.split("|", 1)[1], c.split("|", 1)[1]) for c in cols]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[key] = df.sort_index()
    # 시가는 조정 비율(Adj/Close)을 곱해 조정 시가로 (분할 정합성)
    ratio = out["adj_close"] / out["close"]
    out["adj_open"] = out["open"] * ratio
    return out


def fetch_benchmark(start: str, end: str, cache_dir: Path = CACHE_DIR, refresh: bool = False,
                    symbol: str = BENCHMARK) -> pd.DataFrame:
    """벤치마크 — columns: open, close (조정가: Adj Close 비율을 시가에도 적용)."""
    import yfinance as yf
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"bench_{symbol.replace('^','').replace('.','_')}_{start}_{end}.pkl"
    if f.exists() and not refresh:
        return pd.read_pickle(f)
    h = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    ratio = h["Adj Close"] / h["Close"]
    df = pd.DataFrame({"open": h["Open"] * ratio, "close": h["Adj Close"]}).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df.to_pickle(f)
    return df


# ── 유니버스 복원 ─────────────────────────────────────────────────────────

def monthly_universes(pool: pd.DataFrame, adj_close: pd.DataFrame,
                      kospi_n: int = KOSPI_TOP_N, kosdaq_n: int = KOSDAQ_TOP_N) -> dict[pd.Timestamp, list[str]]:
    """매월 첫 거래일 기준 시총(AdjClose × 현재 주식수) 순위로 코스피 top-N + 코스닥 top-M.
    필터(우선주·바이오)는 풀 단계에서 이미 적용된 상태를 기대한다."""
    shares = pool.set_index("code")["shares"]
    market = pool.set_index("code")["market"]
    codes = [c for c in adj_close.columns if c in shares.index]
    px = adj_close[codes]
    month_starts = px.groupby([px.index.year, px.index.month]).head(1).index
    out: dict[pd.Timestamp, list[str]] = {}
    for d in month_starts:
        cap = (px.loc[d] * shares[codes]).dropna()
        cap = cap[cap > 0]
        ks = cap[market[cap.index] == "KOSPI"].sort_values(ascending=False).head(kospi_n)
        kq = cap[market[cap.index] == "KOSDAQ"].sort_values(ascending=False).head(kosdaq_n)
        out[d] = sorted(set(ks.index) | set(kq.index))
    return out


def universe_on(date_: pd.Timestamp, universes: dict[pd.Timestamp, list[str]]) -> list[str]:
    """date_ 이전(포함) 가장 최근 월초 유니버스."""
    keys = [k for k in universes if k <= date_]
    if not keys:
        return []
    return universes[max(keys)]
