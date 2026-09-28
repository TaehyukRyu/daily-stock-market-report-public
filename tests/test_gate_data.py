"""풀·가격 캐시 키, 월초 순위표, 유니버스 EW 벤치마크. 외부 호출 없음."""
import json

import numpy as np
import pandas as pd
import pytest


def _pool_prices():
    codes = [f"K{i:02d}" for i in range(1, 9)] + [f"Q{i:02d}" for i in range(1, 5)]
    pool = pd.DataFrame({"code": codes, "name": codes, "market": ["KOSPI"] * 8 + ["KOSDAQ"] * 4,
                         "shares": [100.0 - i for i in range(12)]})
    idx = pd.bdate_range("2026-01-01", periods=70)
    rng = np.random.default_rng(3)
    close = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 70))) for c in codes}, index=idx)
    vol = pd.DataFrame({c: 1000.0 for c in codes}, index=idx)
    return pool, {"adj_close": close, "adj_open": close, "open": close, "close": close, "volume": vol}


def test_fetch_pool_cache_name_depends_on_pages(tmp_path):
    """Review Focus 3 — 페이지를 바꿨는데 옛 pool.json을 조용히 읽으면 안 된다."""
    from src.evaluation.backtest import data as D
    (tmp_path / "pool.json").write_text(json.dumps([{"code": "OLD", "name": "o", "market": "KOSPI", "shares": 1}]), encoding="utf-8")
    (tmp_path / "pool_k8_q4.json").write_text(json.dumps([{"code": "NEW", "name": "n", "market": "KOSPI", "shares": 1}]), encoding="utf-8")
    assert D.fetch_pool(tmp_path)["code"].tolist() == ["OLD"]
    assert D.fetch_pool(tmp_path, kospi_pages=8, kosdaq_pages=4)["code"].tolist() == ["NEW"]


def test_fetch_prices_tag_selects_cache_file(tmp_path):
    from src.evaluation.backtest import data as D
    idx = pd.bdate_range("2026-01-01", periods=3)
    def wide(v):
        return pd.DataFrame({f"{f}|000001.KS": [v] * 3 for f in ("Adj Close", "Open", "Close", "Volume")}, index=idx)
    wide(1.0).to_pickle(tmp_path / "prices_2026-01-01_2026-01-10.pkl")
    wide(2.0).to_pickle(tmp_path / "prices_2026-01-01_2026-01-10_abc12345.pkl")
    pool = pd.DataFrame({"code": ["000001"], "market": ["KOSPI"]})
    assert D.fetch_prices(pool, "2026-01-01", "2026-01-10", tmp_path)["close"].iloc[0, 0] == 1.0
    assert D.fetch_prices(pool, "2026-01-01", "2026-01-10", tmp_path, tag="abc12345")["close"].iloc[0, 0] == 2.0


def test_pool_hash_depends_on_codes_only():
    from src.screening.gate_data import pool_hash
    a = pd.DataFrame({"code": ["A", "B"], "name": ["x", "y"]})
    b = pd.DataFrame({"code": ["B", "A"], "name": ["q", "r"]})
    c = pd.DataFrame({"code": ["A", "C"], "name": ["x", "y"]})
    assert pool_hash(a) == pool_hash(b) != pool_hash(c) and len(pool_hash(a)) == 8


def test_monthly_snapshots_have_rank_and_turnover():
    from src.screening.gate_data import monthly_snapshots
    pool, prices = _pool_prices()
    snaps = monthly_snapshots(pool, prices)
    assert len(snaps) >= 3
    last = list(snaps.values())[-1]
    assert set(last.columns) >= {"code", "name", "market", "rank_in_market", "turnover_20d"}
    assert sorted(last[last.market == "KOSPI"]["rank_in_market"]) == list(range(1, 9))
    assert (last["turnover_20d"] > 0).all()


def test_monthly_universes_for_slices_each_month():
    from src.screening.gate_data import monthly_snapshots, monthly_universes_for
    pool, prices = _pool_prices()
    unis = monthly_universes_for({"kospi_rank": [3, 5], "kosdaq_rank": [1, 2], "min_turnover_krw": 0, "exclude": []},
                                 monthly_snapshots(pool, prices))
    assert unis and all(len(m) == 5 for m in unis.values())


def _period(idx, s, e, x):
    return pd.DataFrame({"signal_date": [idx[s]], "entry_date": [idx[e]], "exit_date": [idx[x]]})


def test_universe_ew_bench_r_is_same_window_buy_and_hold():
    """벤치마크 = 같은 창(t+1 시가 → t+H 종가)에 유니버스 전부를 산 동일가중 수익."""
    from src.screening.gate_data import universe_ew_bench_r
    _, prices = _pool_prices()
    idx, ac, ao = prices["adj_close"].index, prices["adj_close"], prices["adj_open"]
    r = universe_ew_bench_r(prices, {idx[0]: ["K01", "K02"]}, _period(idx, 30, 31, 40))
    expect = np.mean([ac.at[idx[40], c] / ao.at[idx[31], c] - 1 for c in ("K01", "K02")])
    assert r.loc[idx[30]] == pytest.approx(expect)


def test_universe_ew_skips_untradable_members():
    """엔진과 같은 거래 가능 조건 — 진입 시가가 없는 종목은 벤치마크에서도 뺀다."""
    from src.screening.gate_data import universe_ew_bench_r
    _, prices = _pool_prices()
    idx, ac, ao = prices["adj_close"].index, prices["adj_close"], prices["adj_open"]
    prices["adj_open"] = ao.copy()
    prices["adj_open"].loc[idx[31], "K02"] = np.nan
    r = universe_ew_bench_r(prices, {idx[0]: ["K01", "K02"]}, _period(idx, 30, 31, 40))
    assert r.loc[idx[30]] == pytest.approx(ac.at[idx[40], "K01"] / ao.at[idx[31], "K01"] - 1)


def test_with_bench_replaces_excess_only():
    from src.screening.gate_data import with_bench
    d = pd.Timestamp("2026-01-02")
    tr = pd.DataFrame({"signal_date": [d, d], "code": ["A", "B"], "r_net": [0.03, -0.01],
                       "bench_r": [0.0, 0.0], "excess": [0.03, -0.01]})
    out = with_bench(tr, pd.Series({d: 0.01}))
    assert out["excess"].tolist() == pytest.approx([0.02, -0.02])
    assert tr["excess"].tolist() == [0.03, -0.01], "원본은 그대로"


def test_fetch_pool_stocks_only_drops_etf_and_uses_own_cache(tmp_path, monkeypatch):
    """네이버 시총 목록에는 ETF·ETN이 섞여 있다(코스피 상위 800 중 주식 480, 2026-09-23 실측)."""
    from src.evaluation.backtest import data as D
    page = {"stocks": [
        {"itemCode": "005930", "stockName": "삼성전자", "stockEndType": "stock", "closePrice": "70,000", "marketValue": "4,000,000"},
        {"itemCode": "069500", "stockName": "KODEX 200", "stockEndType": "etf", "closePrice": "40,000", "marketValue": "8,000,000"},
        {"itemCode": "530031", "stockName": "어떤 ETN", "stockEndType": "etn", "closePrice": "10,000", "marketValue": "100"},
    ]}

    class Resp:
        def json(self):
            return page
    monkeypatch.setattr(D.requests, "get", lambda *a, **k: Resp())
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    out = D.fetch_pool(tmp_path, kospi_pages=1, kosdaq_pages=0, stocks_only=True)
    assert out["code"].tolist() == ["005930"]
    assert (tmp_path / "pool_k1_q0_stocks.json").exists()


def test_gate_uses_stock_only_pool_with_enough_pages(monkeypatch):
    """게이트 풀은 주식만, 코스피 16페이지(주식 약 800) + 코스닥 4페이지(ETF 없음, 400)."""
    from src.evaluation.backtest import data as D
    from src.screening import gate as G
    from src.screening import gate_data as GD
    from src.screening.rules.spec import parse_spec
    from tests.test_rule_spec import BASE
    seen = {}

    def fake_pool(cache_dir, **kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(D, "fetch_pool", fake_pool)
    with pytest.raises(RuntimeError, match="stop"):
        G._build_inputs(parse_spec(BASE), "2015-09-01", "2026-09-23", None)
    assert seen == {"kospi_pages": 16, "kosdaq_pages": 4, "stocks_only": True}
    assert (GD.GATE_KOSPI_PAGES, GD.GATE_KOSDAQ_PAGES) == (16, 4)
