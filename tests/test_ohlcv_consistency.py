"""분할 미조정 행을 잡아 재백필. 스펙 §1-2."""
import pytest


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from src.data import ohlcv_cache as oc
    monkeypatch.setattr(oc, "DB_PATH", tmp_path / "m.db")
    oc.init_ohlcv_cache()
    return oc


def _rows(t, closes):
    return [{"ticker": t, "date": f"2026-09-{i + 1:02d}", "open": c, "high": c, "low": c, "close": c, "volume": 1}
            for i, c in enumerate(closes)]


def test_split_like_gap_is_flagged(cache):
    cache.upsert_ohlcv_rows(_rows("A", [100, 101, 102, 20, 21]))
    cache.upsert_ohlcv_rows(_rows("B", [100, 101, 102, 103, 104]))
    assert cache.find_suspect_tickers(threshold=0.35, days=5) == ["A"]


def test_gap_outside_window_is_ignored(cache):
    cache.upsert_ohlcv_rows(_rows("A", [100, 20, 21, 22, 23, 24, 25, 26, 27]))
    assert cache.find_suspect_tickers(threshold=0.35, days=5) == []


def test_stage1a_backfills_suspects(monkeypatch):
    from src.screening import stage1a_quant as S
    monkeypatch.setattr(S, "find_suspect_tickers", lambda **kw: ["X", "Z"])
    monkeypatch.setattr(S, "get_ticker_count", lambda t: 999)          # 행 부족 아님
    monkeypatch.setattr(S, "_stale_days", lambda t, today: 0)         # 날짜 정체 아님
    calls = []
    monkeypatch.setattr(S, "_backfill_one", lambda t, days=S.BACKFILL_DAYS: calls.append(t) or 1)
    S._backfill_missing(["X", "Y"])
    assert calls == ["X"], "요청 종목 중 의심 종목만 (Z는 요청에 없다)"


def test_stage1a_survives_consistency_check_failure(monkeypatch):
    from src.screening import stage1a_quant as S
    def boom(**kw):
        raise RuntimeError("db locked")
    monkeypatch.setattr(S, "find_suspect_tickers", boom)
    monkeypatch.setattr(S, "get_ticker_count", lambda t: 999)
    monkeypatch.setattr(S, "_stale_days", lambda t, today: 0)
    assert S._backfill_missing(["X"]) == {}
