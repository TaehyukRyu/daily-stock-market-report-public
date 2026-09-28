"""가격 신호. 조정 종가(adj_close) 기준, date까지만 본다."""
from __future__ import annotations

import numpy as np
import pandas as pd


LOOKBACK_ROWS = 65          # 라이브 frames_from_cache(days=65)와 같은 창


def _upto(frames: dict, key: str, date: pd.Timestamp) -> pd.DataFrame:
    """date까지, 마지막 65행만. 백테스트와 라이브가 같은 길이를 봐야 계산이 갈라지지 않는다."""
    return frames[key].loc[:date].iloc[-LOOKBACK_ROWS:]


def ret_20d(date, tickers, frames, params) -> dict[str, float]:
    px = _upto(frames, "adj_close", date)
    out: dict[str, float] = {}
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) >= 21:
            out[t] = float(s.iloc[-1] / s.iloc[-21] - 1.0)
    return out


def vol_20d(date, tickers, frames, params) -> dict[str, float]:
    px = _upto(frames, "adj_close", date)
    out: dict[str, float] = {}
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) >= 21:
            out[t] = float(np.log(s).diff().iloc[-20:].std(ddof=1))
    return out


def random_score(date, tickers, frames, params) -> dict[str, float]:
    """대조군. params['rng']가 있으면 그 생성기를 이어 쓴다(백테스트가 날짜마다 넘긴다)."""
    rng = params.get("rng")
    if rng is None:
        seed = params.get("seed")
        rng = np.random.default_rng(None if seed is None else int(seed))
    return {t: float(v) for t, v in zip(tickers, rng.random(len(tickers)))}


def filter_ret_5d_max(date, tickers, frames, params) -> set[str]:
    """5일 수익률이 max를 넘는 종목만 뺀다. 떨어진 종목은 남긴다(반전 규칙 보호)."""
    mx = float(params.get("max", 0.10))
    px = _upto(frames, "adj_close", date)
    keep: set[str] = set()
    for t in tickers:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) < 6 or (s.iloc[-1] / s.iloc[-6] - 1.0) <= mx:
            keep.add(t)
    return keep
