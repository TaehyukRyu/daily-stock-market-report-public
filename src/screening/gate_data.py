"""
게이트 데이터 — 월초 순위표, 규칙별 유니버스, 유니버스 동일가중 벤치마크 수익. 스펙 §2-2·§4.
가격·풀은 backtest/data.py를 쓴다. 여기서 외부 호출은 없다.

[풀을 코스피 800·코스닥 400으로 넓히는 이유 — 스펙 §4]
  풀은 "지금" 순위다. 2016년 중형이었다가 지금 300위 밖으로 떨어진 종목은 풀에 없어
  과거 유니버스에서 빠진다. 떨어진 종목이 빠지면 반전(하락 매수) 규칙이 실제보다 좋아 보인다.
"""
from __future__ import annotations

import contextlib
import hashlib
import io

import pandas as pd

from src.screening.rule_universe import rank_snapshot_from_cap, slice_universe

# 주식만(ETF·ETN 제외). 네이버 코스피 시총 목록은 약 40%가 ETF라 16페이지여야 주식 약 800개,
# 코스닥은 ETF가 없어 4페이지 = 400개 (2026-09-23 실측: 코스피 8p 주식 480 / 14p 756)
GATE_KOSPI_PAGES = 16
GATE_KOSDAQ_PAGES = 4


def pool_hash(pool: pd.DataFrame) -> str:
    raw = ",".join(sorted(map(str, pool["code"]))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def monthly_snapshots(pool: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> dict[pd.Timestamp, pd.DataFrame]:
    """매월 첫 거래일: 시총(조정종가 × 현재 주식수) 순위 + 20일 중앙 거래대금(원)."""
    p = pool.set_index("code")
    px = prices["adj_close"]
    codes = [c for c in px.columns if c in p.index]
    px = px[codes]
    turnover = (prices["close"][codes] * prices["volume"][codes]).rolling(20, min_periods=5).median()
    month_starts = px.groupby([px.index.year, px.index.month]).head(1).index
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in month_starts:
        cap = (px.loc[d] * p.loc[codes, "shares"]).dropna()
        cap = cap[cap > 0]
        out[d] = rank_snapshot_from_cap(cap, p.loc[cap.index, "market"], p.loc[cap.index, "name"],
                                        turnover.loc[d, cap.index])
    return out


def monthly_universes_for(spec_universe: dict, snapshots: dict[pd.Timestamp, pd.DataFrame]) -> dict[pd.Timestamp, list[str]]:
    """월마다 slice_universe. 제외 필터가 부를 때마다 찍는 제거 목록(130개월치)은 숨긴다."""
    with contextlib.redirect_stdout(io.StringIO()):
        return {d: slice_universe(spec_universe, snap) for d, snap in snapshots.items()}


def universe_ew_bench_r(prices: dict[str, pd.DataFrame], universes: dict[pd.Timestamp, list[str]],
                        periods: pd.DataFrame) -> pd.Series:
    """리밸런스 날짜마다: 그날 유니버스의 거래 가능 종목 전부를 t+1 시가에 사서 t+H 종가에 판
    동일가중 수익(비용 없음). 무작위 선정의 기대값이라, 이것과의 차이는 '유니버스 안에서 고른 효과'만 남는다.
    거래 가능 조건은 engine.simulate와 같다(진입 시가 > 0, 청산 종가 있음)."""
    from src.evaluation.backtest.data import universe_on
    ac, ao = prices["adj_close"], prices["adj_open"]
    out: dict[pd.Timestamp, float] = {}
    for row in periods.itertuples(index=False):
        members = [c for c in universe_on(row.signal_date, universes) if c in ao.columns]
        o, x = ao.loc[row.entry_date, members], ac.loc[row.exit_date, members]
        ok = o.notna() & x.notna() & (o > 0)
        out[row.signal_date] = float((x[ok] / o[ok] - 1.0).mean()) if ok.any() else float("nan")
    return pd.Series(out, dtype=float)


def with_bench(trades: pd.DataFrame, bench_r: pd.Series) -> pd.DataFrame:
    """같은 거래를 다른 벤치마크로 다시 채점. 시뮬을 한 번 더 돌리지 않는다."""
    out = trades.copy()
    if len(out):
        out["bench_r"] = pd.to_datetime(out["signal_date"]).map(bench_r)
        out["excess"] = out["r_net"] - out["bench_r"]
    return out
