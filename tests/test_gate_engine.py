"""simulate에 선정 함수를 주입. 기존 경로(selector=None)는 그대로."""
import numpy as np
import pandas as pd

from src.evaluation.backtest.engine import simulate


def _prices(n=60, tickers=("A", "B", "C", "D", "E", "F")):
    idx = pd.bdate_range("2026-01-01", periods=n)
    rng = np.random.default_rng(1)
    close = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for t in tickers}, index=idx)
    return {"adj_close": close, "adj_open": close.shift(1).bfill(), "open": close, "close": close,
            "volume": close * 0 + 1000}


def _bench(pr):
    return pd.DataFrame({"open": pr["adj_close"]["A"], "close": pr["adj_close"]["A"]})


def test_selector_is_used_and_capped_at_n_picks():
    pr = _prices()
    universes = {pr["adj_close"].index[0]: list(pr["adj_close"].columns)}
    seen = []
    def sel(t, universe, tradable):
        seen.append(t)
        return ["F", "E", "D", "C"]
    res = simulate(pr, _bench(pr), universes, buy_mask=None, frames=None,
                   start="2026-01-01", end="2026-03-20", n_picks=2, horizon=5, selector=sel)
    assert seen and set(res.trades["code"]) == {"F", "E"}


def test_selector_none_keeps_legacy_path(monkeypatch):
    from src.evaluation.backtest import engine as E
    called = {"n": 0}
    def fake_rank(t, universe, buy_mask, frames, tradable):
        called["n"] += 1
        return ["A"]
    monkeypatch.setattr(E, "rank_candidates", fake_rank)
    pr = _prices()
    universes = {pr["adj_close"].index[0]: list(pr["adj_close"].columns)}
    simulate(pr, _bench(pr), universes, buy_mask=pd.DataFrame(), frames={},
             start="2026-01-01", end="2026-03-20", n_picks=1, horizon=5)
    assert called["n"] > 0


def test_random_pick_wins_over_selector():
    pr = _prices()
    universes = {pr["adj_close"].index[0]: list(pr["adj_close"].columns)}
    def sel(t, universe, tradable):
        raise AssertionError("random_pick이면 selector를 부르지 않는다")
    simulate(pr, _bench(pr), universes, buy_mask=None, frames=None, start="2026-01-01", end="2026-03-20",
             n_picks=2, horizon=5, random_pick=True, seed=3, selector=sel)
