"""
유니버스를 규칙 파라미터로. 스펙 §4.
snapshot 열: code, name, market(KOSPI|KOSDAQ), rank_in_market(1부터), turnover_20d(원).
백테스트(월초 복원)와 라이브(pykrx 월초 스냅샷, 3단계)가 같은 모양을 만들어 여기로 넣는다.
"""
from __future__ import annotations

import pandas as pd

from src.universe.filters import filter_bio_pharma, filter_preferred_stocks

_EXCLUDERS = {"preferred": filter_preferred_stocks, "bio": filter_bio_pharma}


def rank_snapshot_from_cap(cap: pd.Series, market: pd.Series, names: pd.Series,
                           turnover: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"cap": cap, "market": market, "name": names, "turnover_20d": turnover})
    df = df.dropna(subset=["cap", "market"])
    df = df[df["cap"] > 0]
    df["rank_in_market"] = df.groupby("market")["cap"].rank(ascending=False, method="first").astype(int)
    df.index.name = "code"
    return df.reset_index()[["code", "name", "market", "rank_in_market", "turnover_20d"]]


def slice_universe(universe_spec: dict, snapshot: pd.DataFrame) -> list[str]:
    parts = []
    for market, key in (("KOSPI", "kospi_rank"), ("KOSDAQ", "kosdaq_rank")):
        rng = universe_spec.get(key)
        if not rng:
            continue
        lo, hi = int(rng[0]), int(rng[1])
        parts.append(snapshot[(snapshot["market"] == market)
                              & (snapshot["rank_in_market"] >= lo) & (snapshot["rank_in_market"] <= hi)])
    if not parts:
        return []
    df = pd.concat(parts)
    floor = float(universe_spec.get("min_turnover_krw") or 0)
    if floor > 0:
        df = df[df["turnover_20d"].fillna(0) >= floor]
    if universe_spec.get("exclude"):
        idx = df.set_index("code").copy()
        idx["종목명"] = idx["name"]
        for name in universe_spec["exclude"]:
            idx = _EXCLUDERS[name](idx)
        df = idx.reset_index()
    return sorted(df["code"].tolist())
