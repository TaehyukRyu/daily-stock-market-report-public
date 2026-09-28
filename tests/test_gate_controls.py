"""날짜 단위 판정 — 같은 날 5종목을 독립 표본으로 세지 않는다. 스펙 §2-2·§2-4."""
import inspect

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import controls as C


def _trades(dates, excess_per_date, per_date=1):
    rows = []
    for d, e in zip(dates, excess_per_date):
        for _ in range(per_date):
            rows.append({"signal_date": pd.Timestamp(d), "excess": e})
    return pd.DataFrame(rows)


def test_significance_by_date_hand_computed():
    tr = _trades(["2026-01-02", "2026-01-16", "2026-01-30"], [0.01, 0.02, 0.03])
    s = C.significance_by_date(tr, n_boot=200)
    assert s["n_dates"] == 3 and s["n_trades"] == 3
    assert s["mean"] == pytest.approx(0.02)
    assert s["t"] == pytest.approx(0.02 / (0.01 / np.sqrt(3)))


def test_same_day_duplicates_do_not_inflate_t():
    """Review Focus 1 — 같은 거래를 하루에 5번 넣어도 t가 같아야 한다."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=80, freq="10B")
    ex = rng.normal(0.003, 0.03, len(dates))
    one = C.significance_by_date(_trades(dates, ex, per_date=1), n_boot=200)
    five = C.significance_by_date(_trades(dates, ex, per_date=5), n_boot=200)
    assert five["t"] == pytest.approx(one["t"]) and five["n_trades"] == 5 * one["n_trades"]


def test_significance_by_date_empty():
    assert C.significance_by_date(pd.DataFrame(columns=["signal_date", "excess"])) == {"n_dates": 0, "n_trades": 0}


def test_walk_forward_from_trades_counts_positive_windows():
    dates = pd.bdate_range("2020-01-01", "2022-12-31", freq="10B")
    ex = [0.01 if d < pd.Timestamp("2021-07-01") else -0.01 for d in dates]
    out = C.walk_forward_from_trades(_trades(dates, ex), "2020-01-01", "2022-12-31")
    assert out["n"] >= 5
    assert out["positive"] + out["negative"] == out["n"]
    assert 0.3 < out["positive_ratio"] < 0.7
    assert all("mean_excess" in w and "n_dates" in w for w in out["windows"])


def test_walk_forward_from_trades_uses_only_excess():
    """현금 비율·시장 방향은 입력에 아예 없다 — 초과 CAGR 오염이 구조적으로 불가능 (스펙 §2-1)."""
    dates = pd.bdate_range("2020-01-01", "2021-12-31", freq="10B")
    out = C.walk_forward_from_trades(_trades(dates, [0.0] * len(dates)), "2020-01-01", "2021-12-31")
    assert out["positive"] == 0 and all(w["mean_excess"] == 0.0 for w in out["windows"])


def test_date_mean_matches_significance():
    tr = _trades(["2026-01-02", "2026-01-16"], [0.01, 0.03], per_date=3)
    assert C.date_mean(tr) == pytest.approx(C.significance_by_date(tr, n_boot=10)["mean"])
    assert np.isnan(C.date_mean(pd.DataFrame(columns=["signal_date", "excess"])))


def test_control_percentile():
    vals = list(np.linspace(-0.01, 0.01, 200))
    out = C.control_percentile(vals, strategy_value=0.02)
    assert out["n"] == 200 and out["strategy_percentile"] == 100.0
    assert C.control_percentile(vals, 0.0)["strategy_percentile"] == pytest.approx(50.0)
    assert C.control_percentile([float("nan")], 0.0) == {"n": 0}


def test_legacy_functions_untouched():
    """과거 R1 재현용 — 옛 함수의 시그니처가 그대로다."""
    assert list(inspect.signature(C.walk_forward).parameters) == ["run", "start", "end"]
    assert "strategy_excess" in inspect.signature(C.random_control).parameters
