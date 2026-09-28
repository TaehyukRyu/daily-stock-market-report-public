"""수급 신호. 네이버 미조정 주식 수 — 같은 출처 거래량으로 나눠 액면분할에 불변(스펙 §4)."""
from __future__ import annotations


def foreign_net_5d(date, tickers, frames, params) -> dict[str, float]:
    """외국인 5일 순매수 합 ÷ 20일 평균 거래량."""
    fn = frames["foreign_net"].loc[:date].iloc[-65:]
    fv = frames["flow_volume"].loc[:date].iloc[-65:]
    out: dict[str, float] = {}
    for t in tickers:
        if t not in fn.columns or t not in fv.columns:
            continue
        a, v = fn[t].dropna(), fv[t].dropna()
        if len(a) < 5 or len(v) < 20:
            continue
        denom = float(v.iloc[-20:].mean())
        if denom > 0:
            out[t] = float(a.iloc[-5:].sum()) / denom
    return out
