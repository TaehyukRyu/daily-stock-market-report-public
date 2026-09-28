"""
metrics.py — CAGR·MDD·Sharpe·승률·손익비·초과수익·손익분기 승률.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float | None:
    if equity is None or len(equity) < 2 or equity.iloc[0] <= 0:
        return None
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return None
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float | None:
    if equity is None or len(equity) < 2:
        return None
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def sharpe(equity: pd.Series, rf: float = 0.0) -> float | None:
    if equity is None or len(equity) < 3:
        return None
    r = equity.pct_change().dropna()
    if r.std() == 0 or r.empty:
        return None
    return float((r.mean() - rf / TRADING_DAYS) / r.std() * math.sqrt(TRADING_DAYS))


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0, "win_rate": None, "profit_factor": None, "avg_win": None, "avg_loss": None,
                "avg_r_net": None, "avg_excess": None, "roundtrip_cost": None, "breakeven_win_rate": None}
    r = trades["r_net"].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    avg_w = float(wins.mean()) if len(wins) else None
    avg_l = float(-losses.mean()) if len(losses) else None
    cost = float((trades["r_gross"] - trades["r_net"]).mean())
    # 손익분기 승률 p* = (L̄ + c) / (W̄ + L̄): 총수익률 기준(비용 차감 전 W̄·L̄)
    g = trades["r_gross"].astype(float)
    gw, gl = g[g > 0], g[g <= 0]
    W = float(gw.mean()) if len(gw) else None
    L = float(-gl.mean()) if len(gl) else None
    be = (L + cost) / (W + L) if (W is not None and L is not None and (W + L) > 0) else None
    return {
        "n_trades": int(len(r)), "win_rate": float((r > 0).mean()),
        "profit_factor": float(pf) if pf is not None else None,
        "avg_win": avg_w, "avg_loss": avg_l, "avg_r_net": float(r.mean()),
        "avg_excess": float(trades["excess"].dropna().mean()) if trades["excess"].notna().any() else None,
        "roundtrip_cost": cost, "breakeven_win_rate": be,
    }


def summarize(res) -> dict:
    """벤치마크 2개를 항상 같이 낸다 (S5 사전 등록 §2).

    B2 `bench_*`         — 지수 100% buy&hold. "이 시스템에 돈을 넣는 게 지수를 사는 것보다 나은가"
    B1 `bench_matched_*` — 지수를 exposure만큼만 + 나머지 현금. "같은 노출에서 종목 선택이 이기는가"

    B1만 인용하지 않는다. 비중을 낮추면 B1 초과수익이 개선되는데 그것은 전략이 좋아진 것이
    아니라 지는 게임에 덜 참여한 것이다.
    """
    eq, be = res.equity, res.bench_equity
    bm = getattr(res, "bench_equity_matched", None)
    s_c, b_c = cagr(eq), cagr(be)
    m_c = cagr(bm) if bm is not None else None
    out = {
        "start": str(eq.index[0].date()), "end": str(eq.index[-1].date()), "days": int(len(eq)),
        "cagr": s_c, "bench_cagr": b_c,
        "excess_cagr": (s_c - b_c) if (s_c is not None and b_c is not None) else None,
        "bench_matched_cagr": m_c,
        "excess_cagr_matched": (s_c - m_c) if (s_c is not None and m_c is not None) else None,
        "exposure": res.params.get("exposure"), "weight": res.params.get("weight"),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "bench_total_return": float(be.iloc[-1] / be.iloc[0] - 1.0),
        "bench_matched_total_return": (float(bm.iloc[-1] / bm.iloc[0] - 1.0) if bm is not None else None),
        "mdd": max_drawdown(eq), "bench_mdd": max_drawdown(be),
        "bench_matched_mdd": (max_drawdown(bm) if bm is not None else None),
        "sharpe": sharpe(eq), "bench_sharpe": sharpe(be),
        "periods": int(len(res.periods)),
        "avg_picked": float(res.periods["n_picked"].mean()) if len(res.periods) else 0.0,
        "empty_periods": int((res.periods["n_picked"] == 0).sum()) if len(res.periods) else 0,
    }
    out.update(trade_stats(res.trades))
    return out


def fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.2f}%"
