"""
tests/test_date_window.py

G2 — point-in-time 규율. TradingAgents date_window 규칙(반개구간·UTC·날짜미상 처리),
as-of OHLCV, 현재값 제공 거부, 모델 컷오프 assert.
외부 호출 없음.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.utils import date_window as dw

KST = dw.KST


# ── in_window ──────────────────────────────────────────────

def test_window_is_half_open_end_exclusive_next_midnight():
    start, end = dw.day_bounds("2026-09-10")
    assert dw.in_window(datetime(2026, 9, 10, 23, 59, tzinfo=KST), start, end)
    assert dw.in_window(datetime(2026, 9, 10, 0, 0, tzinfo=KST), start, end)
    assert not dw.in_window(datetime(2026, 9, 11, 0, 0, tzinfo=KST), start, end)   # 다음날 자정 정각은 밖
    assert not dw.in_window(datetime(2026, 9, 9, 23, 59, tzinfo=KST), start, end)


def test_naive_timestamp_is_treated_as_utc():
    start, end = dw.day_bounds("2026-09-10")   # KST 00:00 = UTC 09-09 15:00
    assert dw.in_window(datetime(2026, 9, 9, 16, 0), start, end)       # naive → UTC → KST 01:00
    assert not dw.in_window(datetime(2026, 9, 9, 14, 0), start, end)


def test_undated_kept_only_when_window_reaches_present():
    start, end = dw.day_bounds("2026-09-10")
    now_live = datetime(2026, 9, 10, 12, 0, tzinfo=KST)
    now_backtest = datetime(2026, 10, 1, 12, 0, tzinfo=KST)
    assert dw.in_window(None, start, end, now=now_live)
    assert not dw.in_window(None, start, end, now=now_backtest)


# ── 기사 필터 ──────────────────────────────────────────────

def test_filter_articles_as_of_drops_future_and_undated_in_backtest():
    arts = [
        {"title": "ok",     "pubDate": "Thu, 10 Sep 2026 09:00:00 +0900"},
        {"title": "future", "pubDate": "Fri, 11 Sep 2026 00:37:00 +0900"},
        {"title": "old",    "pubDate": "Tue, 08 Sep 2026 09:00:00 +0900"},
        {"title": "nodate", "pubDate": ""},
    ]
    now_backtest = datetime(2026, 10, 1, tzinfo=KST)
    kept = dw.filter_articles_as_of(arts, "2026-09-10", now=now_backtest)
    assert [a["title"] for a in kept] == ["ok"]
    kept2 = dw.filter_articles_as_of(arts, "2026-09-10", lookback_days=2, now=now_backtest)
    assert [a["title"] for a in kept2] == ["ok", "old"]


def test_parse_pubdate_formats():
    assert dw.parse_pubdate("Fri, 11 Sep 2026 00:37:00 +0900").hour == 0
    assert dw.parse_pubdate("2026-09-11T00:37:00+09:00").minute == 37
    assert dw.parse_pubdate("garbage") is None and dw.parse_pubdate(None) is None


# ── 현재값 제공 거부 ────────────────────────────────────────

def test_withhold_live_value_only_for_past_as_of():
    assert dw.withhold_live_value(None, "x", today="2026-09-11") is None
    assert dw.withhold_live_value("2026-09-11", "x", today="2026-09-11") is None
    w = dw.withhold_live_value("2026-09-01", "get_financials", today="2026-09-11")
    assert w["withheld"] is True and "error" in w and "2026-09-01" in w["error"]


# ── 모델 컷오프 ────────────────────────────────────────────

def test_assert_after_cutoff():
    dw.assert_after_cutoff("gpt-4o-mini", "2024-01-02")
    with pytest.raises(ValueError, match="look-ahead"):
        dw.assert_after_cutoff("gpt-4o-mini", "2023-06-01")
    with pytest.raises(ValueError, match="미등록"):
        dw.assert_after_cutoff("some-unknown-model", "2026-01-01")
    assert dw.knowledge_cutoff("gpt-5.6-luna-2026") == "2026-02-16"


# ── ohlcv_cache as_of ──────────────────────────────────────

@pytest.fixture
def ohlcv_db(tmp_path, monkeypatch):
    from src.data import ohlcv_cache as oc
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(oc, "DB_PATH", path)
    oc.init_ohlcv_cache()
    oc.upsert_ohlcv_rows([
        {"ticker": "005930", "date": d, "open": 1, "high": 1, "low": 1, "close": c, "volume": 1}
        for d, c in [("2026-09-08", 100), ("2026-09-09", 101), ("2026-09-10", 102)]
    ])
    return oc, path


def test_get_ohlcv_series_respects_as_of(ohlcv_db):
    oc, _ = ohlcv_db
    assert [r["close"] for r in oc.get_ohlcv_series("005930", 65)] == [100, 101, 102]
    assert [r["close"] for r in oc.get_ohlcv_series("005930", 65, as_of="2026-09-09")] == [100, 101]


def test_ohlcv_fetched_at_is_recorded_and_migrated(ohlcv_db):
    oc, path = ohlcv_db
    with sqlite3.connect(path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(ohlcv_cache)")]
        assert "fetched_at" in cols
        fa = conn.execute("SELECT fetched_at FROM ohlcv_cache LIMIT 1").fetchone()[0]
    assert fa and fa[:4] == "2026" or fa[:2] == "20"

    # 옛 스키마(fetched_at 없음)에도 init이 컬럼을 추가한다
    old = str(path) + ".old"
    with sqlite3.connect(old) as conn:
        conn.execute("CREATE TABLE ohlcv_cache (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY(ticker,date))")
    import src.data.ohlcv_cache as oc2
    oc2.DB_PATH = old
    oc2.init_ohlcv_cache()
    with sqlite3.connect(old) as conn:
        assert "fetched_at" in [r[1] for r in conn.execute("PRAGMA table_info(ohlcv_cache)")]


# ── mention_tracker as_of ──────────────────────────────────

@pytest.mark.asyncio
async def test_crawl_ticker_filters_by_as_of(monkeypatch, tmp_path):
    from src.data import mention_tracker as mt
    saved = {}

    async def fake_fetch(query, display=50, session=None):
        return [   # 제목에 회사명 — 관련도 필터(headline_relevance)가 이름 없는 제목을 버린다
            {"title": "삼성전자 오늘", "pubDate": "Thu, 10 Sep 2026 09:00:00 +0900"},
            {"title": "삼성전자 미래", "pubDate": "Fri, 11 Sep 2026 09:00:00 +0900"},
        ]

    async def fake_classify(titles):
        return [{"index": i, "sentiment": "neutral", "score": 0.0} for i in range(len(titles))]

    monkeypatch.setattr(mt, "_fetch_naver_news", fake_fetch)
    monkeypatch.setattr(mt, "classify_headlines", fake_classify)
    monkeypatch.setattr(mt, "_resolve_ticker_name", lambda t: "삼성전자")
    monkeypatch.setattr(mt, "is_already_crawled", lambda t, d: False)
    monkeypatch.setattr(mt, "insert_mentions_batch", lambda rows: saved.setdefault("rows", rows))
    monkeypatch.setattr(mt, "existing_titles", lambda t: set())          # v2: 중복 제거 심
    monkeypatch.setattr(mt, "recount_daily_stats", lambda t, dates: None)  # v2: 재집계 심 (실DB 차단)
    monkeypatch.setattr(mt, "_now_utc", lambda: datetime(2026, 10, 1, tzinfo=timezone.utc))   # 백테스트 시점

    n = await mt.crawl_ticker("005930", today="2026-09-10", as_of="2026-09-10")
    assert n == 1
    assert [r["title"] for r in saved["rows"]] == ["삼성전자 오늘"]

    saved.clear()
    n = await mt.crawl_ticker("005930", today="2026-09-10")     # as_of 없음 = 라이브, 필터 없음
    assert n == 2
