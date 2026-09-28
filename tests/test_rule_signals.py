"""신호 함수 단일성·필터·candidate·ranking. 스펙 §1-2. 외부 호출 없음."""
import numpy as np
import pandas as pd
import pytest

from tests.test_rule_spec import BASE


def _frames(n_days=40, tickers=("A", "B", "C", "D"), seed=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n_days)
    close = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))) for t in tickers}, index=idx)
    vol = pd.DataFrame({t: rng.integers(1000, 5000, n_days) for t in tickers}, index=idx, dtype=float)
    return {"adj_close": close, "open": close * 0.99, "close": close, "volume": vol}


def test_ret_20d_matches_hand_computed():
    from src.screening.signals import SIGNALS
    fr = _frames()
    d = fr["adj_close"].index[-1]
    out = SIGNALS["ret_20d"](d, ["A", "B"], fr, {})
    assert out["A"] == pytest.approx(fr["adj_close"]["A"].iloc[-1] / fr["adj_close"]["A"].iloc[-21] - 1)
    assert set(out) == {"A", "B"}


def test_signal_excludes_short_history():
    from src.screening.signals import SIGNALS
    fr = _frames(n_days=10)
    assert SIGNALS["ret_20d"](fr["adj_close"].index[-1], ["A"], fr, {}) == {}


def test_signal_never_sees_the_future():
    from src.screening.signals import SIGNALS
    fr = _frames(n_days=60)
    d = fr["adj_close"].index[39]
    full = SIGNALS["ret_20d"](d, ["A"], fr, {})
    cut = {k: v.iloc[:40] for k, v in fr.items()}
    assert SIGNALS["ret_20d"](d, ["A"], cut, {}) == full


def test_vol_20d_is_std_of_log_returns():
    from src.screening.signals import SIGNALS
    fr = _frames()
    d = fr["adj_close"].index[-1]
    lr = np.log(fr["adj_close"]["A"]).diff().iloc[-20:]
    assert SIGNALS["vol_20d"](d, ["A"], fr, {})["A"] == pytest.approx(float(lr.std(ddof=1)))


def test_random_score_reproducible_with_seed_and_advances_with_rng():
    from src.screening.signals import SIGNALS
    fr = _frames()
    d = fr["adj_close"].index[-1]
    a = SIGNALS["random_score"](d, ["A", "B", "C"], fr, {"seed": 7})
    assert a == SIGNALS["random_score"](d, ["A", "B", "C"], fr, {"seed": 7})
    rng = np.random.default_rng(7)
    x = SIGNALS["random_score"](d, ["A", "B", "C"], fr, {"rng": rng})
    y = SIGNALS["random_score"](d, ["A", "B", "C"], fr, {"rng": rng})
    assert x != y, "같은 생성기를 이어 쓰면 날짜마다 다른 값"


def _flow_frames(n_days=30, tickers=("A", "B")):
    idx = pd.bdate_range("2026-01-01", periods=n_days)
    fn = pd.DataFrame({"A": 100.0, "B": -50.0}, index=idx)
    fv = pd.DataFrame({"A": 1000.0, "B": 1000.0}, index=idx)
    return {"foreign_net": fn, "flow_volume": fv}


def test_foreign_net_5d_is_sum5_over_mean20_volume():
    from src.screening.signals import SIGNALS
    fr = {**_frames(n_days=30, tickers=("A", "B")), **_flow_frames()}
    d = fr["adj_close"].index[-1]
    out = SIGNALS["foreign_net_5d"](d, ["A", "B"], fr, {})
    assert out["A"] == pytest.approx(500 / 1000) and out["B"] == pytest.approx(-250 / 1000)


def test_foreign_net_5d_is_split_invariant():
    """네이버 수급은 미조정 주식 수다. 분할로 주식 수가 5배가 돼도 비율은 같아야 한다 (Review Focus 2).
    분할이 20일 창 '안'에 있으면 분모가 섞여 최대 20일 왜곡된다 — 드물어서 받아들인다(스펙 §4)."""
    from src.screening.signals import SIGNALS
    fr = {**_frames(n_days=30, tickers=("A", "B")), **_flow_frames()}
    d = fr["adj_close"].index[-1]
    base = SIGNALS["foreign_net_5d"](d, ["A"], fr, {})["A"]
    split = fr["foreign_net"].index[5]              # 마지막 20일 창(10~29행)보다 앞
    fr["foreign_net"].loc[split:, "A"] *= 5
    fr["flow_volume"].loc[split:, "A"] *= 5
    assert SIGNALS["foreign_net_5d"](d, ["A"], fr, {})["A"] == pytest.approx(base)


def test_signal_needs_declares_flow_keys():
    from src.screening.signals import SIGNAL_NEEDS
    assert SIGNAL_NEEDS["foreign_net_5d"] == {"foreign_net", "flow_volume"}
    assert SIGNAL_NEEDS.get("ret_20d", set()) == set()


def test_filter_ret_5d_max_keeps_big_drops():
    """반전 규칙의 필터는 많이 오른 것만 뺀다. 많이 떨어진 것은 신호다."""
    from src.screening.signals import FILTERS
    fr = _frames()
    last = fr["adj_close"].index[-1]
    fr["adj_close"].loc[last, "A"] = fr["adj_close"]["A"].iloc[-6] * 0.85
    fr["adj_close"].loc[last, "B"] = fr["adj_close"]["B"].iloc[-6] * 1.15
    keep = FILTERS["ret_5d_max"](last, ["A", "B", "C"], fr, {"max": 0.10})
    assert "A" in keep and "B" not in keep and "C" in keep


def test_apply_rule_percentile_max_and_ascending_rank():
    from src.screening.rules.spec import parse_spec
    from src.screening.signals import SIGNALS, apply_rule
    fr = _frames(n_days=40, tickers=tuple(f"T{i}" for i in range(20)))
    d = fr["adj_close"].index[-1]
    spec = parse_spec({**BASE, "filters": [], "candidate": {"percentile_max": 0.25}, "n_picks": 3})
    picks = apply_rule(spec, d, list(fr["adj_close"].columns), fr)
    scores = [s for _, s in picks]
    assert len(picks) == 3 and scores == sorted(scores)
    cutoff = np.quantile(list(SIGNALS["ret_20d"](d, list(fr["adj_close"].columns), fr, {}).values()), 0.25)
    assert all(s <= cutoff for s in scores)


def test_apply_rule_percentile_min_and_descending_rank():
    from src.screening.rules.spec import parse_spec
    from src.screening.signals import apply_rule
    fr = _frames(n_days=40, tickers=tuple(f"T{i}" for i in range(20)))
    d = fr["adj_close"].index[-1]
    spec = parse_spec({**BASE, "filters": [], "candidate": {"percentile_min": 0.80},
                       "ranking": "signal desc", "n_picks": 2})
    scores = [s for _, s in apply_rule(spec, d, list(fr["adj_close"].columns), fr)]
    assert len(scores) == 2 and scores == sorted(scores, reverse=True)


def test_apply_rule_empty_when_no_history():
    from src.screening.rules.spec import parse_spec
    from src.screening.signals import apply_rule
    fr = _frames(n_days=10)
    assert apply_rule(parse_spec(BASE), fr["adj_close"].index[-1], ["A", "B"], fr) == []


def test_frames_from_cache_shape(monkeypatch):
    from src.screening import signals as S
    rows = [{"date": f"2026-02-{i + 1:02d}", "open": 10 + i, "high": 11 + i, "low": 9 + i,
             "close": 10 + i, "volume": 100 + i} for i in range(25)]
    monkeypatch.setattr(S, "get_ohlcv_series", lambda t, days=65, as_of=None: rows)
    fr = S.frames_from_cache(["A", "B"], days=30)
    assert set(fr) == {"adj_close", "open", "close", "volume"}
    assert list(fr["adj_close"].columns) == ["A", "B"] and len(fr["adj_close"]) == 25
    assert S.SIGNALS["ret_20d"](fr["adj_close"].index[-1], ["A"], fr, {})["A"] == pytest.approx(34 / 14 - 1)
