"""
tests/test_backtest.py — Baseline 백테스트 (src/evaluation/backtest/).

  1. 벡터화 지표 == stage1a._compute_signals (무작위 시계열 60개)
  2. 엔진: 비용·동일가중·현금 슬롯·자산곡선 정합 (합성 가격)
  3. 지표: CAGR/MDD/손익분기 승률
  4. 대조군·walk-forward·국면·유의성 유틸
외부 호출 없음.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import controls as C
from src.evaluation.backtest import metrics as M
from src.evaluation.backtest.engine import CostModel, simulate
from src.evaluation.backtest.signals import SignalParams, compute_signal_frames, rule_buy, signals_from_series
from src.screening.stage1a_quant import _compute_signals


# ── 1. 지표 동일성 ─────────────────────────────────────────

@pytest.mark.parametrize("seed", range(60))
def test_vectorized_signals_match_stage1a(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(21, 130))
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    vols = rng.integers(1000, 100000, n).astype(float)
    series = [{"date": f"d{i}", "open": c, "high": c, "low": c, "close": float(c), "volume": float(v)}
              for i, (c, v) in enumerate(zip(closes, vols))]
    ref = _compute_signals(series)
    got = signals_from_series(series)
    assert ref is not None and got is not None
    for k in ("volume_signal", "ma_breakout", "golden_cross", "signal_count"):
        assert got[k] == ref[k], (k, seed)
    assert got["price_change_5d"] == pytest.approx(ref["price_change_5d"], abs=1e-4)


def test_short_series_is_none():
    series = [{"close": 1.0, "volume": 1.0}] * 20
    assert signals_from_series(series) is None


# ── 2. 엔진 ────────────────────────────────────────────────

def _synthetic(days=30, codes=("A", "B", "C")):
    idx = pd.bdate_range("2024-01-01", periods=days)
    close = pd.DataFrame({c: 100.0 + i * 10 + np.arange(days) * (1.0 + i) for i, c in enumerate(codes)}, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) + 0.5      # 전일 종가 + 0.5
    vol = pd.DataFrame(1000.0, index=idx, columns=codes)
    prices = {"adj_close": close, "close": close, "open": open_, "adj_open": open_, "volume": vol}
    bench = pd.DataFrame({"open": 1000.0 + np.arange(days), "close": 1000.5 + np.arange(days)}, index=idx)
    return idx, prices, bench


def test_engine_accounting_and_cash_slot():
    idx, prices, bench = _synthetic()
    universes = {idx[0]: ["A", "B", "C"]}
    # 신호: A와 B만 항상 BUY, C는 절대 아님 → N=3이면 1슬롯 현금
    buy = pd.DataFrame(False, index=idx, columns=["A", "B", "C"]); buy[["A", "B"]] = True
    frames = {"signal_count": pd.DataFrame(2.0, index=idx, columns=["A", "B", "C"]),
              "price_change_5d": pd.DataFrame(0.01, index=idx, columns=["A", "B", "C"])}
    costs = CostModel(commission=0.001, exchange_fee=0.0, slippage=0.0, tax_by_year={2024: 0.002})
    res = simulate(prices, bench, universes, buy, frames, str(idx[0].date()), str(idx[-1].date()),
                   n_picks=3, horizon=5, costs=costs)

    assert set(res.trades["code"]) == {"A", "B"}
    assert (res.periods["n_picked"] == 2).all()
    t = res.trades.iloc[0]
    exp_net = t["exit"] * (1 - 0.001 - 0.002) / (t["entry"] * (1 + 0.001)) - 1
    assert t["r_net"] == pytest.approx(exp_net)
    assert t["r_gross"] == pytest.approx(t["exit"] / t["entry"] - 1)
    # 구간 수익 = 1/3 × (A + B), 현금 슬롯 0
    p0 = res.periods.iloc[0]["ret"]
    two = res.trades[res.trades["signal_date"] == res.periods.iloc[0]["signal_date"]]["r_net"]
    assert p0 == pytest.approx(two.sum() / 3)
    # 자산곡선 끝 = 구간 수익 복리
    assert res.equity.iloc[-1] == pytest.approx(np.prod(1 + res.periods["ret"]), rel=1e-9)
    # 리밸런싱 간격 = H, 포지션 겹침 없음
    sig = list(res.periods["signal_date"])
    assert all((idx.get_loc(sig[i + 1]) - idx.get_loc(sig[i])) == 5 for i in range(len(sig) - 1))


def test_engine_random_pick_uses_universe_only():
    idx, prices, bench = _synthetic()
    universes = {idx[0]: ["A", "C"]}
    buy = pd.DataFrame(False, index=idx, columns=["A", "B", "C"])
    frames = {"signal_count": pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"]),
              "price_change_5d": pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"])}
    res = simulate(prices, bench, universes, buy, frames, str(idx[0].date()), str(idx[-1].date()),
                   n_picks=2, horizon=5, random_pick=True, seed=1)
    assert set(res.trades["code"]) <= {"A", "C"} and len(res.trades) > 0
    res2 = simulate(prices, bench, universes, buy, frames, str(idx[0].date()), str(idx[-1].date()),
                    n_picks=2, horizon=5, random_pick=True, seed=1)
    assert res.trades.equals(res2.trades), "같은 seed면 재현"


def test_tax_by_year():
    c = CostModel()
    assert c.sell_cost(pd.Timestamp("2024-03-01")) == pytest.approx(c.commission + c.exchange_fee + c.slippage + 0.0018)
    assert c.sell_cost(pd.Timestamp("2026-03-01")) == pytest.approx(c.commission + c.exchange_fee + c.slippage + 0.0020)
    assert c.with_slippage_mult(3).slippage == pytest.approx(0.003)


# ── 3. 지표 ────────────────────────────────────────────────

def test_metrics_basic():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2, 1.3], index=pd.bdate_range("2024-01-01", periods=5))
    assert M.max_drawdown(eq) == pytest.approx(0.99 / 1.1 - 1)
    assert M.cagr(eq) == pytest.approx(1.3 ** (252 / 5) - 1)
    trades = pd.DataFrame({"r_gross": [0.05, -0.03, 0.02, -0.01], "r_net": [0.045, -0.035, 0.015, -0.015],
                           "excess": [0.01, -0.02, 0.0, -0.01]})
    s = M.trade_stats(trades)
    assert s["n_trades"] == 4 and s["win_rate"] == 0.5
    W, L, c = (0.05 + 0.02) / 2, (0.03 + 0.01) / 2, 0.005
    assert s["breakeven_win_rate"] == pytest.approx((L + c) / (W + L))


# ── 4. 검사 유틸 ───────────────────────────────────────────

def test_walk_forward_windows():
    w = C.walk_forward_windows("2023-10-04", "2026-09-11")
    assert w[0] == ("2023-10-04", "2024-04-03")
    assert all(pd.Timestamp(e) <= pd.Timestamp("2026-09-11") for _, e in w)
    assert len(w) >= 9


def test_regime_labels_and_split():
    idx = pd.bdate_range("2024-01-01", periods=200)
    close = pd.Series(np.linspace(100, 130, 200), index=idx)     # 꾸준한 상승 → 60일 후 강세
    lab = C.regime_labels(close)
    assert lab.iloc[-1] == "강세" and lab.iloc[0] == "n/a"
    trades = pd.DataFrame({"signal_date": [idx[100], idx[150]], "excess": [0.02, -0.01], "r_net": [0.03, -0.02]})
    out = C.regime_split(trades, lab)
    assert out["강세"]["n"] == 2 and out["강세"]["win_rate"] == 0.5


def test_significance_small_sample_flags_insufficient():
    t = pd.DataFrame({"excess": np.random.default_rng(0).normal(0.01, 0.02, 30)})
    s = C.significance(t, n_boot=200)
    assert s["n"] == 30 and "표본 부족" in s["verdict"]
    t2 = pd.DataFrame({"excess": np.random.default_rng(1).normal(0.02, 0.01, 200)})
    assert "유의" in C.significance(t2, n_boot=200)["verdict"]


def test_plateau_or_peak():
    rows = [{"value": v, "excess_cagr": e} for v, e in [(1, 0.01), (2, 0.02), (3, 0.12), (4, 0.02), (5, 0.01)]]
    assert "봉우리" in C.plateau_or_peak(rows, 3)
    rows2 = [{"value": v, "excess_cagr": e} for v, e in [(1, 0.01), (2, 0.02), (3, 0.03), (4, 0.02), (5, 0.01)]]
    assert C.plateau_or_peak(rows2, 3) == "고원"


def test_random_control_percentile():
    class R:
        def __init__(self, v): self.v = v
    import src.evaluation.backtest.controls as cc
    cc_summ = M.summarize
    try:
        M.summarize = lambda res: {"excess_cagr": res.v}
        cc.M = M
        out = C.random_control(lambda s: R(s / 100.0), strategy_excess=0.95, n_runs=100)
        assert out["n"] == 100 and out["strategy_percentile"] == 95.0 and out["pass_top5pct"] is True
    finally:
        M.summarize = cc_summ


# ── 5. S5 — 비중 분리와 노출 일치 벤치마크 (사전 등록 doc/2026-09-15_s5-prereg.md) ──

def _s5_setup(days=30):
    idx, prices, bench = _synthetic(days=days)
    universes = {idx[0]: ["A", "B", "C"]}
    buy = pd.DataFrame(False, index=idx, columns=["A", "B", "C"]); buy[["A", "B"]] = True
    frames = {"signal_count": pd.DataFrame(2.0, index=idx, columns=["A", "B", "C"]),
              "price_change_5d": pd.DataFrame(0.01, index=idx, columns=["A", "B", "C"])}
    return prices, bench, universes, buy, frames


def test_weight_none_reproduces_full_investment():
    """R0 회귀 — weight를 주지 않으면 기존 동작(1/n_picks)과 완전히 같아야 한다.

    이 테스트가 깨지면 비중 리팩터가 기존 백테스트 결과를 바꾼 것이다.
    S5는 기간 확장과 묶여 있어, 여기서 막지 않으면 10년 결과가 달라졌을 때
    기간 탓인지 코드 탓인지 구분할 수 없다.
    """
    prices, bench, universes, buy, frames = _s5_setup()
    kw = dict(start="2024-01-01", end="2024-02-09", n_picks=5, horizon=5, costs=CostModel())
    a = simulate(prices, bench, universes, buy, frames, **kw)
    b = simulate(prices, bench, universes, buy, frames, weight=1.0 / 5, **kw)
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.periods, b.periods)
    assert a.params["weight"] == 0.2 and a.params["exposure"] == 1.0


def test_weight_scales_period_return_linearly():
    """period_ret = Σ w·r_net 이므로 비중에 선형이다.

    이 선형성이 S5 판단의 근거였다 — 대응표본 t는 차이와 표준오차가 같은 배수로
    스케일돼 약분되므로 8-A·S4·N2 판정이 비중과 무관하게 불변이다.
    """
    prices, bench, universes, buy, frames = _s5_setup()
    kw = dict(start="2024-01-01", end="2024-02-09", n_picks=5, horizon=5, costs=CostModel())
    full = simulate(prices, bench, universes, buy, frames, weight=0.20, **kw)
    ops = simulate(prices, bench, universes, buy, frames, weight=0.05, **kw)
    assert (ops.periods["ret"] / full.periods["ret"]).round(10).eq(0.25).all()
    # 개별 거래 수익률은 비중과 무관하다
    pd.testing.assert_series_equal(full.trades["r_net"], ops.trades["r_net"])


def test_matched_benchmark_tracks_exposure():
    """B1 노출 일치 벤치마크 = 1 + exposure × (지수배수 − 1). B2는 그대로 100%."""
    prices, bench, universes, buy, frames = _s5_setup()
    kw = dict(start="2024-01-01", end="2024-02-09", n_picks=5, horizon=5, costs=CostModel())
    r = simulate(prices, bench, universes, buy, frames, weight=0.05, **kw)
    assert r.params["exposure"] == pytest.approx(0.25)
    expected = 1.0 + 0.25 * (r.bench_equity - 1.0)
    pd.testing.assert_series_equal(r.bench_equity_matched, expected)
    # 지수가 오르는 합성 데이터이므로 B1은 B2보다 덜 오른다
    assert r.bench_equity_matched.iloc[-1] < r.bench_equity.iloc[-1]

    full = simulate(prices, bench, universes, buy, frames, **kw)
    pd.testing.assert_series_equal(full.bench_equity_matched, full.bench_equity,
                                   check_names=False)  # exposure=1.0이면 B1 == B2


def test_summarize_reports_both_benchmarks():
    """사전 등록 §2 — 모든 표에 B1·B2 두 열이 나와야 한다."""
    prices, bench, universes, buy, frames = _s5_setup()
    r = simulate(prices, bench, universes, buy, frames, start="2024-01-01", end="2024-02-09",
                 n_picks=5, horizon=5, costs=CostModel(), weight=0.05)
    s = M.summarize(r)
    for k in ("excess_cagr", "excess_cagr_matched", "bench_cagr", "bench_matched_cagr",
              "bench_matched_mdd", "exposure", "weight"):
        assert k in s, f"{k}가 summarize 출력에 없다"
    assert s["exposure"] == pytest.approx(0.25) and s["weight"] == pytest.approx(0.05)
