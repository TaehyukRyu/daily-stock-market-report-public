"""월초 순위 구간·거래대금 하한·제외. 스펙 §4."""
import pandas as pd


def _snap():
    """코드 끝자리는 0 — 5·7·9·K는 기존 우선주 필터가 코드만 보고 뺀다(filters.py)."""
    rows = [{"code": f"K{i:03d}0", "name": f"코스피{i}", "market": "KOSPI", "rank_in_market": i, "turnover_20d": 5e9}
            for i in range(1, 31)]
    rows += [{"code": f"Q{i:03d}0", "name": f"코스닥{i}", "market": "KOSDAQ", "rank_in_market": i, "turnover_20d": 5e9}
             for i in range(1, 16)]
    df = pd.DataFrame(rows)
    df.loc[df.code == "K0050", "turnover_20d"] = 1e9
    df.loc[df.code == "K0060", "name"] = "코스피6우"
    df.loc[df.code == "Q0030", "name"] = "셀트리온바이오"
    return df


def test_rank_range_is_per_market():
    from src.screening.rule_universe import slice_universe
    out = slice_universe({"kospi_rank": [3, 8], "kosdaq_rank": [2, 4], "min_turnover_krw": 0, "exclude": []}, _snap())
    assert out == ["K0030", "K0040", "K0050", "K0060", "K0070", "K0080", "Q0020", "Q0030", "Q0040"]


def test_turnover_floor_and_exclusions():
    from src.screening.rule_universe import slice_universe
    out = slice_universe({"kospi_rank": [3, 8], "kosdaq_rank": [2, 4], "min_turnover_krw": 3e9,
                          "exclude": ["preferred", "bio"]}, _snap())
    assert out == ["K0030", "K0040", "K0070", "K0080", "Q0020", "Q0040"]


def test_missing_market_key_means_none_from_that_market():
    from src.screening.rule_universe import slice_universe
    assert slice_universe({"kospi_rank": [1, 2], "min_turnover_krw": 0, "exclude": []}, _snap()) == ["K0010", "K0020"]


def test_rank_snapshot_orders_within_market():
    from src.screening.rule_universe import rank_snapshot_from_cap
    cap = pd.Series({"A": 300.0, "B": 100.0, "C": 200.0, "D": 50.0})
    mk = pd.Series({"A": "KOSPI", "B": "KOSPI", "C": "KOSDAQ", "D": "KOSDAQ"})
    nm = pd.Series({"A": "a", "B": "b", "C": "c", "D": "d"})
    tv = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0})
    s = rank_snapshot_from_cap(cap, mk, nm, tv).set_index("code")
    assert (s.loc["A", "rank_in_market"], s.loc["B", "rank_in_market"]) == (1, 2)
    assert (s.loc["C", "rank_in_market"], s.loc["D", "rank_in_market"]) == (1, 2)
