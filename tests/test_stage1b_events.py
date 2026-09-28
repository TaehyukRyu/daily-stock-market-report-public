"""
tests/test_stage1b_events.py — 스크리닝 v2 팩터 원값. DB·HTTP 없음(전부 목).
"""
from __future__ import annotations

from datetime import date

import pytest

from src.screening import stage1b_events as ev


def test_recent_window_monday_covers_weekend():
    assert ev.recent_window(date(2026, 9, 21)) == ["2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21"]


def test_recent_window_weekday_is_two_days():
    assert ev.recent_window(date(2026, 9, 22)) == ["2026-09-21", "2026-09-22"]


def test_baseline_window_is_14_days_before_recent_start():
    assert ev.baseline_window(date(2026, 9, 22)) == ("2026-09-07", "2026-09-20")
    assert ev.baseline_window(date(2026, 9, 21)) == ("2026-09-04", "2026-09-17")


def test_news_burst_is_ratio_of_daily_averages():
    recent = ["2026-09-21", "2026-09-22"]
    counts = {"2026-09-21": 4, "2026-09-22": 2}
    counts.update({f"2026-09-{d:02d}": 1 for d in range(7, 21)})       # baseline 14일 × 1건
    burst, n = ev.compute_news_burst(counts, recent, "2026-09-07", "2026-09-20")
    assert burst == pytest.approx(3.0) and n == 6


def test_news_burst_with_empty_baseline_uses_floor():
    burst, n = ev.compute_news_burst({"2026-09-22": 1}, ["2026-09-21", "2026-09-22"], "2026-09-07", "2026-09-20")
    assert burst == pytest.approx(0.5 / (1 / 14)) and n == 1          # 0.5건/일 ÷ (1/14)건/일 = 7.0


def test_news_burst_zero_when_no_recent_articles():
    burst, n = ev.compute_news_burst({"2026-09-10": 3}, ["2026-09-21", "2026-09-22"], "2026-09-07", "2026-09-20")
    assert burst == 0.0 and n == 0


def test_sentiment_delta_needs_six_points():
    assert ev.sentiment_delta([0.1, 0.2, 0.3]) is None
    assert ev.sentiment_delta([0.0, 0.0, 0.0, 0.3, 0.3, 0.3]) == pytest.approx(0.3)


def test_news_burst_ready_after_16_days(monkeypatch):
    monkeypatch.setattr(ev, "method_applied_date", lambda key: "2026-09-01")
    assert ev.news_burst_ready(date(2026, 9, 16)) is False
    assert ev.news_burst_ready(date(2026, 9, 17)) is True
    monkeypatch.setattr(ev, "method_applied_date", lambda key: None)
    assert ev.news_burst_ready(date(2026, 12, 1)) is False


def test_run_event_screen_assembles_factors(monkeypatch):
    monkeypatch.setattr(ev, "_init_mention_db", lambda: None)
    monkeypatch.setattr(ev, "collect_dart_events",
                        lambda universe, today: {"B": [{"date": "20260921", "title": "유상증자결정", "category": "capital"}]})
    counts = {"A": {"2026-09-21": 6, "2026-09-22": 6, **{f"2026-09-{d:02d}": 1 for d in range(7, 21)}}, "B": {}}
    monkeypatch.setattr(ev, "get_daily_counts", lambda t, s, e: counts.get(t, {}))
    monkeypatch.setattr(ev, "get_net_sentiment_series", lambda t, n=6: [0, 0, 0, 0.2, 0.2, 0.2] if t == "A" else [])

    out = ev.run_event_screen(["A", "B"], today=date(2026, 9, 22), news_ready=True)
    assert out["A"] == {"news_burst": 6.0, "news_recent": 12, "dart_event": 0, "dart_titles": [], "sent_delta": 0.2}
    assert out["B"]["dart_event"] == 1 and out["B"]["dart_titles"] == ["유상증자결정"]
    assert out["B"]["news_burst"] == 0.0 and out["B"]["sent_delta"] is None


def test_run_event_screen_warmup_leaves_news_burst_none(monkeypatch):
    monkeypatch.setattr(ev, "_init_mention_db", lambda: None)
    monkeypatch.setattr(ev, "collect_dart_events", lambda universe, today: {})
    monkeypatch.setattr(ev, "get_daily_counts", lambda t, s, e: {"2026-09-22": 99})
    monkeypatch.setattr(ev, "get_net_sentiment_series", lambda t, n=6: [])
    out = ev.run_event_screen(["A"], today=date(2026, 9, 22), news_ready=False)
    assert out["A"]["news_burst"] is None and out["A"]["news_recent"] == 0


def test_run_event_screen_isolates_per_ticker_failure(monkeypatch):
    monkeypatch.setattr(ev, "_init_mention_db", lambda: None)
    monkeypatch.setattr(ev, "collect_dart_events", lambda universe, today: {})
    def counts(t, s, e):
        if t == "BAD":
            raise RuntimeError("db")
        return {}
    monkeypatch.setattr(ev, "get_daily_counts", counts)
    monkeypatch.setattr(ev, "get_net_sentiment_series", lambda t, n=6: [])
    out = ev.run_event_screen(["BAD", "OK"], today=date(2026, 9, 22))
    assert set(out) == {"BAD", "OK"} and out["OK"]["news_burst"] == 0.0
