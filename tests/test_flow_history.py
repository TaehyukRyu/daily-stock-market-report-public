"""수급 이력 로더 — 페이징·부호 파싱·상장 전·캐시 이어받기. 네트워크 없음 (스펙 §4)."""
import pandas as pd
import pytest


def _fake_api(days):
    """bizdate '이전' 60개를 최신순으로 돌려주는 가짜 API (실제 동작과 같은 규약)."""
    days = sorted(days, reverse=True)
    calls = []
    def fetch(code, bizdate):
        calls.append((code, bizdate))
        older = [d for d in days if d < bizdate][:60]
        return [{"bizdate": d, "foreignerPureBuyQuant": "+1,000", "organPureBuyQuant": "-500",
                 "accumulatedTradingVolume": "10,000"} for d in older]
    return fetch, calls


def _days(n, start="2024-01-01"):
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start, periods=n)]


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    from src.data import flow_history as F
    monkeypatch.setattr(F, "PAUSE", 0.0)


def test_paging_covers_range_and_stops_at_start():
    from src.data import flow_history as F
    fetch, calls = _fake_api(_days(200))
    df = F.fetch_ticker_flow("000001", "2024-03-01", "2024-09-30", fetch=fetch)
    assert df.index.min() >= pd.Timestamp("2024-03-01") and df.index.max() <= pd.Timestamp("2024-09-30")
    assert len(df) == len(pd.bdate_range("2024-03-01", "2024-09-30").intersection(pd.to_datetime(_days(200))))
    assert len(calls) <= 4, "시작일을 지나면 멈춘다"


def test_parses_signs_and_commas():
    from src.data import flow_history as F
    fetch, _ = _fake_api(_days(10))
    df = F.fetch_ticker_flow("000001", "2024-01-01", "2024-01-31", fetch=fetch)
    row = df.iloc[0]
    assert (row["foreign_net"], row["organ_net"], row["flow_volume"]) == (1000, -500, 10000)


def test_pre_listing_is_empty_not_error():
    from src.data import flow_history as F
    df = F.fetch_ticker_flow("000001", "2015-01-01", "2015-12-31", fetch=lambda c, b: [])
    assert df.empty and list(df.columns) == ["foreign_net", "organ_net", "flow_volume"]


def test_cache_reuse_and_failed_ticker_reported(tmp_path):
    from src.data import flow_history as F
    fetch, calls = _fake_api(_days(100))
    def flaky(code, bizdate):
        if code == "BAD":
            raise RuntimeError("down")
        return fetch(code, bizdate)
    out = F.load_flow_history(["A", "BAD"], "2024-01-01", "2024-05-31", cache_dir=tmp_path, workers=2, fetch=flaky)
    assert out["failed"] == ["BAD"] and "A" in out["foreign_net"].columns
    n = len(calls)
    F.load_flow_history(["A"], "2024-01-01", "2024-05-31", cache_dir=tmp_path, workers=1, fetch=flaky)
    assert len(calls) == n, "두 번째 실행은 캐시에서 읽는다"
