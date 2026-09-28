"""run_gate 끝까지 — 합성 데이터로 두 벤치마크·대조군·walk-forward를 실제로 돌려 전이까지."""
import json

import numpy as np
import pandas as pd
import pytest

from tests.test_rule_spec import BASE

UNIV = {"kospi_rank": [1, 40], "min_turnover_krw": 0, "exclude": []}
KW = dict(start="2023-01-02", end="2026-01-30", oos_start="2025-03-01", n_random=40)


@pytest.fixture
def db(tmp_path, monkeypatch):
    from src.screening.rules import registry as R
    path = tmp_path / "m.db"
    monkeypatch.setattr(R, "DB_PATH", path)
    R.init_rule_tables()
    return path


def _synthetic(monkeypatch, edge=0.0, n_days=800, n=40):
    """40종목·800일. edge>0이면 20일 하락 하위 10% 종목이 다음 날부터 하루 edge/10씩 더 오른다."""
    from src.screening import gate as G
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"K{i:03d}" for i in range(1, n + 1)]
    rets = pd.DataFrame(rng.normal(0.0003, 0.02, (n_days, n)), index=idx, columns=codes)
    if edge > 0:
        close0 = (1 + rets).cumprod()
        low = close0.pct_change(20).rank(axis=1, pct=True) <= 0.10
        rets = rets + low.shift(1, fill_value=False).astype(float) * (edge / 10)
    close = 100 * (1 + rets).cumprod()
    vol = pd.DataFrame(1_000_000.0, index=idx, columns=codes)
    prices = {"adj_close": close, "adj_open": close, "open": close, "close": close, "volume": vol}
    pool = pd.DataFrame({"code": codes, "name": codes, "market": "KOSPI", "shares": 1000.0})
    bench = pd.DataFrame({"open": close.mean(axis=1), "close": close.mean(axis=1)})
    flow = {"foreign_net": pd.DataFrame(rng.normal(0, 100, (n_days, n)), index=idx, columns=codes),
            "flow_volume": pd.DataFrame(1000.0, index=idx, columns=codes)}

    def fake_inputs(spec, data_start, end, cache_dir):
        from src.screening.gate_data import monthly_snapshots, monthly_universes_for
        from src.screening.signals import SIGNAL_NEEDS
        unis = monthly_universes_for(spec.universe, monthly_snapshots(pool, prices))
        extra = flow if SIGNAL_NEEDS.get(spec.signal["name"]) else {}
        return {"pool": pool, "prices": prices, "bench_index": bench, "universes": unis, "extra": extra}
    monkeypatch.setattr(G, "_build_inputs", fake_inputs)


def _register(rule_id="example_rule", **over):
    from src.screening.rules import registry as R
    from src.screening.rules.spec import parse_spec
    R.register(parse_spec({**BASE, "rule_id": rule_id, "universe": UNIV, "filters": [], **over}), quarter="2026Q4")


def test_no_edge_is_rejected(db, monkeypatch, tmp_path):
    from src.screening import gate as G
    from src.screening.rules import registry as R
    _synthetic(monkeypatch, edge=0.0)
    _register()
    out = G.run_gate("example_rule", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    assert out["passed"] is False and R.get_rule("example_rule")["status"] == "rejected"
    f = tmp_path / f"gate_example_rule_{out['spec_hash']}.json"
    assert json.loads(f.read_text(encoding="utf-8"))["rule_id"] == "example_rule"


def test_real_edge_is_gated(db, monkeypatch, tmp_path):
    from src.screening import gate as G
    from src.screening.rules import registry as R
    _synthetic(monkeypatch, edge=0.04)
    _register()
    out = G.run_gate("example_rule", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    assert out["passed"] is True, out["failed"]
    r = R.get_rule("example_rule")
    assert r["status"] == "gated" and r["gate_result"]["passed"] is True


def test_both_benchmarks_reported_and_differ(db, monkeypatch, tmp_path):
    from src.screening import gate as G
    _synthetic(monkeypatch, edge=0.04)
    _register()
    out = G.run_gate("example_rule", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    s = out["summary"]
    assert s["index"]["n_dates"] == s["uni"]["n_dates"] > 60
    assert s["index"]["mean_excess"] != s["uni"]["mean_excess"]


def test_first_gate_fixes_quarter_m(db, monkeypatch, tmp_path):
    from src.screening import gate as G
    from src.screening.rules import registry as R
    _synthetic(monkeypatch)
    _register("a")
    _register("b", n_picks=4)
    out = G.run_gate("a", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    assert out["m"] == 2 and R.fix_quarter_m("2026Q4") == 2
    _register("c", n_picks=3)
    assert R.get_rule("c")["quarter"] == "2027Q1"


def test_control_distribution_is_shared_by_key(db, monkeypatch, tmp_path):
    """같은 유니버스·H·N이면 대조군 캐시 1개를 같이 쓴다. H가 다르면 다른 키."""
    from src.screening import gate as G
    from src.screening.rules import registry as R
    from src.screening.rules.spec import parse_spec
    _synthetic(monkeypatch)
    _register("a")
    _register("b", candidate={"percentile_max": 0.20})
    G.run_gate("a", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    G.run_gate("b", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    assert len(list(tmp_path.glob("control_*.json"))) == 1
    sa = parse_spec(R.get_rule("a")["spec"])
    inp = G._build_inputs(sa, "2023-01-02", "2026-01-30", None)
    sh = parse_spec({**R.get_rule("a")["spec"], "horizon": 20})
    assert G._control_key(sa, inp, "2023-01-02", "2026-01-30", 40) != G._control_key(sh, inp, "2023-01-02", "2026-01-30", 40)


def test_null_rule_uses_m1_and_varies_picks(db, monkeypatch, tmp_path):
    """Review Focus 5 — null_random이 매 날짜 같은 종목만 고르면 대조군이 아니다."""
    from src.screening import gate as G
    from src.screening.rules import registry as R
    from src.screening.rules.spec import parse_spec
    _synthetic(monkeypatch)
    _register("null_random", signal={"name": "random_score", "params": {"seed": 1}},
              candidate={"percentile_max": 1.0}, ranking="signal desc")
    assert G.run_gate("null_random", result_dir=tmp_path, cache_dir=tmp_path, **KW)["m"] == 1
    spec = parse_spec(R.get_rule("null_random")["spec"])
    inp = G._build_inputs(spec, "2023-01-02", "2026-01-30", None)
    sel = G._selector_for(spec, G._frames(inp))
    days, uni = inp["prices"]["adj_close"].index, list(inp["prices"]["adj_close"].columns)
    assert len({tuple(sel(days[i], uni, set(uni))) for i in range(30, 80, 10)}) > 1


def test_flow_rule_gets_flow_frames(db, monkeypatch, tmp_path):
    from src.screening import gate as G
    _synthetic(monkeypatch)
    _register("foreign5", signal={"name": "foreign_net_5d", "params": {}},
              candidate={"percentile_min": 0.90}, ranking="signal desc")
    out = G.run_gate("foreign5", result_dir=tmp_path, cache_dir=tmp_path, **KW)
    assert out["summary"]["index"]["n_trades"] > 0


def test_non_draft_rule_is_refused(db, monkeypatch, tmp_path):
    from src.screening import gate as G
    from src.screening.rules import registry as R
    _synthetic(monkeypatch)
    _register()
    R.transition("example_rule", "rejected", reason="x")
    with pytest.raises(ValueError, match="draft"):
        G.run_gate("example_rule", result_dir=tmp_path, cache_dir=tmp_path, **KW)
