"""
백테스트·라이브 공용 신호 함수. 스펙 §1-2.

signal(date, tickers, frames, params) -> {ticker: score}. frames는 date까지만 본다.
filter(date, tickers, frames, params) -> 남길 ticker 집합.
백테스트는 날짜를 돌며 부르고 라이브는 오늘 한 번 부른다 — 같은 함수라 갈라질 수 없다.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.data.ohlcv_cache import get_ohlcv_series
from src.screening.signals import flow as _f
from src.screening.signals import price as _p

Frames = dict[str, pd.DataFrame]
SignalFn = Callable[[pd.Timestamp, list[str], Frames, dict], dict[str, float]]
FilterFn = Callable[[pd.Timestamp, list[str], Frames, dict], set[str]]

SIGNALS: dict[str, SignalFn] = {
    "ret_20d":        _p.ret_20d,
    "vol_20d":        _p.vol_20d,
    "random_score":   _p.random_score,
    "foreign_net_5d": _f.foreign_net_5d,
}
FILTERS: dict[str, FilterFn] = {"ret_5d_max": _p.filter_ret_5d_max}
SIGNAL_NEEDS: dict[str, set[str]] = {"foreign_net_5d": {"foreign_net", "flow_volume"}}


def frames_from_cache(tickers: list[str], days: int = 65, as_of: Optional[str] = None) -> Frames:
    """라이브 가격 — ohlcv_cache를 Frames로. 캐시는 pykrx 원가격이라 adj_close = close
    (분할 미조정 행은 Task 12 정합성 검사가 재백필한다)."""
    cols: dict[str, dict[str, pd.Series]] = {"open": {}, "close": {}, "volume": {}}
    for t in tickers:
        s = get_ohlcv_series(t, days=days, as_of=as_of)
        if not s:
            continue
        idx = pd.to_datetime([r["date"] for r in s])
        for k in cols:
            cols[k][t] = pd.Series([float(r[k]) for r in s], index=idx)
    out: Frames = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    out["adj_close"] = out["close"].copy()
    return out


def apply_rule(spec, date: pd.Timestamp, tickers: list[str], frames: Frames,
               rng: Optional[np.random.Generator] = None) -> list[tuple[str, float]]:
    """필터 → 신호 → candidate → ranking → 상위 n_picks. 없으면 []."""
    keep = set(tickers)
    for f in spec.filters:
        keep &= FILTERS[f["name"]](date, sorted(keep), frames, f.get("params") or {})
    if not keep:
        return []
    params = dict(spec.signal.get("params") or {})
    if rng is not None:
        params["rng"] = rng
    scores = SIGNALS[spec.signal["name"]](date, sorted(keep), frames, params)
    if not scores:
        return []
    vals = np.array(list(scores.values()), dtype=float)
    cand = spec.candidate
    if "percentile_max" in cand:
        cut = float(np.quantile(vals, float(cand["percentile_max"])))
        scores = {t: v for t, v in scores.items() if v <= cut}
    elif "percentile_min" in cand:
        cut = float(np.quantile(vals, float(cand["percentile_min"])))
        scores = {t: v for t, v in scores.items() if v >= cut}
    elif cand.get("event"):
        scores = {t: v for t, v in scores.items() if v > 0}
    desc = spec.ranking.strip().endswith("desc")
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1] if desc else kv[1], kv[0]))
    return ranked[: spec.n_picks]
