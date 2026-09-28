"""
tests/test_mention_pubdate.py

스크리닝 v2 — 기사를 발행일(pubDate) 기준으로 저장·집계한다.
유료 API 호출 없음 (NAVER·분류기는 목).
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data import mention_db as mdb


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(mdb, "DB_PATH", path)
    monkeypatch.setenv("MENTIONS_DB_PATH", path)     # mention_tracker._get_cached_stats는 env로 경로를 읽는다
    mdb.init_db()
    return path


def _row(ticker, d, title, sentiment="neutral"):
    return {"ticker": ticker, "date": d, "title": title, "sentiment": sentiment,
            "sentiment_score": 0.5, "source": "naver_news"}


TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()


# ── 1. mention_db ─────────────────────────────────────────

def test_existing_titles_returns_titles_for_ticker_only(db):
    mdb.insert_mentions_batch([_row("A", "2026-09-20", "a1"), _row("B", "2026-09-20", "b1")])
    assert mdb.existing_titles("A") == {"a1"}
    assert mdb.existing_titles("Z") == set()


def test_is_already_crawled_uses_crawled_at_not_pubdate(db):
    # 발행일은 어제, 크롤은 지금(오늘) → "오늘 크롤함"이 True, 어제 기준은 False
    mdb.insert_mentions_batch([_row("A", YESTERDAY, "a1")])
    assert mdb.is_already_crawled("A", TODAY) is True
    assert mdb.is_already_crawled("A", YESTERDAY) is False


def test_recount_daily_stats_counts_by_pubdate(db):
    mdb.insert_mentions_batch([
        _row("A", "2026-09-19", "x", "positive"),
        _row("A", "2026-09-19", "y", "negative"),
        _row("A", "2026-09-20", "z", "neutral"),
    ])
    mdb.recount_daily_stats("A", ["2026-09-19", "2026-09-20", "2026-09-21"])   # 21일은 기사 없음 → 행 안 만듦
    assert mdb.get_daily_counts("A", "2026-09-18", "2026-09-21") == {"2026-09-19": 2, "2026-09-20": 1}
    stats = {s["date"]: s for s in mdb.get_daily_stats("A", days=5)}
    assert stats["2026-09-19"]["net_sentiment"] == 0.0        # (1 - 1) / 2
    assert stats["2026-09-20"]["positive_ratio"] == 0.0


def test_get_daily_counts_omits_days_without_rows(db):
    mdb.upsert_daily_stats("A", "2026-09-10", 3, 1, 1, 1)
    assert mdb.get_daily_counts("A", "2026-09-09", "2026-09-11") == {"2026-09-10": 3}
    assert mdb.get_daily_counts("A", "2026-09-11", "2026-09-12") == {}


def test_net_sentiment_series_is_ascending_and_capped(db):
    for i, d in enumerate(["2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14",
                           "2026-09-15", "2026-09-16", "2026-09-17"]):
        mdb.upsert_daily_stats("A", d, 10, i, 10 - i, 0)      # net = i/10
    series = mdb.get_net_sentiment_series("A", n=6)
    assert len(series) == 6
    assert series == sorted(series)                            # 오름차순(오래된→최신)
    assert series[-1] == pytest.approx(0.6)


# ── 2. mention_tracker.crawl_ticker ───────────────────────

@pytest.fixture
def tracker(monkeypatch):
    from src.data import mention_tracker as mt
    monkeypatch.setattr(mt, "_resolve_ticker_name", lambda t: "테스트사")

    async def fake_classify(titles):
        return [{"index": i, "sentiment": "neutral", "score": 0.5} for i in range(len(titles))]
    monkeypatch.setattr(mt, "classify_headlines", fake_classify)
    return mt


def _with_articles(monkeypatch, mt, articles):
    async def fake_fetch(query, display=50, session=None):
        return articles
    monkeypatch.setattr(mt, "_fetch_naver_news", fake_fetch)


@pytest.mark.asyncio
async def test_crawl_assigns_pubdate_and_skips_duplicates(db, tracker, monkeypatch):
    mt = tracker
    # 제목에 회사명을 넣는다 — 관련도 필터(headline_relevance)가 이름 없는 제목을 버린다
    # 어제 이미 저장된 기사 (crawled_at을 과거로 둬서 "오늘 크롤함" 판정을 피한다)
    mdb.insert_mention("A", "2026-09-10", "테스트사 오래된 기사", "neutral", 0.5, crawled_at="2026-09-10T09:00:00")
    _with_articles(monkeypatch, mt, [
        {"title": "테스트사 새 기사",     "pubDate": "Sat, 19 Sep 2026 23:30:00 +0900"},
        {"title": "테스트사 오래된 기사", "pubDate": "Thu, 10 Sep 2026 09:00:00 +0900"},   # 중복 → skip
        {"title": "테스트사 날짜없음"},                                                    # pubDate 없음 → 크롤 날짜
    ])
    n = await mt.crawl_ticker("A", today=TODAY, session=None)
    assert n == 2
    counts = mdb.get_daily_counts("A", "2026-09-01", TODAY)
    assert counts.get("2026-09-19") == 1
    assert counts.get(TODAY) == 1
    assert "2026-09-10" not in counts          # 옛 기사는 다시 세지 않는다 (재집계는 새 기사 날짜만)
    assert mdb.existing_titles("A") == {"테스트사 오래된 기사", "테스트사 새 기사", "테스트사 날짜없음"}


@pytest.mark.asyncio
async def test_crawl_converts_pubdate_to_kst_date(db, tracker, monkeypatch):
    mt = tracker
    # UTC 15:30 = KST 다음날 00:30 → KST 날짜로 저장
    _with_articles(monkeypatch, mt, [{"title": "테스트사 자정 기사", "pubDate": "Sat, 19 Sep 2026 15:30:00 +0000"}])
    await mt.crawl_ticker("A", today=TODAY, session=None)
    assert mdb.get_daily_counts("A", "2026-09-19", "2026-09-20") == {"2026-09-20": 1}


@pytest.mark.asyncio
async def test_crawl_failure_does_not_write_synthetic_stats(db, tracker, monkeypatch):
    mt = tracker
    _with_articles(monkeypatch, mt, [])
    mdb.upsert_daily_stats("A", "2026-09-10", 5, 1, 3, 1)
    n = await mt.crawl_ticker("A", today=TODAY, session=None)
    assert n == -5                                            # 캐시 존재를 알리기만 한다
    assert mdb.get_daily_counts("A", TODAY, TODAY) == {}      # 오늘 날짜에 가짜 행을 만들지 않는다


@pytest.mark.asyncio
async def test_crawl_skips_when_already_crawled_today(db, tracker, monkeypatch):
    mt = tracker
    mdb.insert_mentions_batch([_row("A", "2026-09-19", "이미")])   # crawled_at = 지금
    calls = []

    async def fake_fetch(query, display=50, session=None):
        calls.append(query)
        return [{"title": "안 불려야 함"}]
    monkeypatch.setattr(mt, "_fetch_naver_news", fake_fetch)
    assert await mt.crawl_ticker("A", today=TODAY, session=None) == 0
    assert calls == []
