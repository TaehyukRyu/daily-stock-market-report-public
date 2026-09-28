"""
tests/test_screen_factors.py — 팩터별 초과수익 집계(스크리닝 v2 §6). 합성 데이터, 외부 호출 없음.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from src.evaluation import outcomes as oc
from src.evaluation import screen_factors as sf


def _trading_days(n, start=date(2026, 9, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


@pytest.fixture
def db(tmp_path):
    """3종목 × 40거래일. UP은 +1.3%/+0.7% 교대(평균 +1%, 날짜별 차이가 달라야 t가 정의된다), FLAT 0%, DOWN −1%."""
    p = tmp_path / "f.db"
    days = _trading_days(40)
    with sqlite3.connect(p) as c:
        c.execute("CREATE TABLE ohlcv_cache (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, "
                  "close REAL, volume INTEGER, fetched_at TEXT)")
        for t in ("UP", "FLAT", "DOWN"):
            px = 100.0
            for k, d in enumerate(days):
                c.execute("INSERT INTO ohlcv_cache VALUES (?,?,?,?,?,?,?,?)", (t, d, px, px, px, px, 1, "x"))
                step = {"UP": 1.013 if k % 2 == 0 else 1.007, "FLAT": 1.0, "DOWN": 0.99}[t]
                px *= step
    return p, days


def _rows(cand_up=True):
    return [
        {"ticker": "UP",   "is_candidate": 1 if cand_up else 0, "rank": 1, "score": 1.0, "news_burst": 3.0,
         "news_burst_pct": 1.0, "dart_event": 1, "news_match": 1, "sent_delta": 0.3, "ret5": 0.0, "filtered_reason": None},
        {"ticker": "FLAT", "is_candidate": 0, "rank": None, "score": 0.2, "news_burst": 1.0,
         "news_burst_pct": 0.5, "dart_event": 0, "news_match": 0, "sent_delta": 0.0, "ret5": 0.0, "filtered_reason": None},
        {"ticker": "DOWN", "is_candidate": 0, "rank": None, "score": 0.0, "news_burst": 0.5,
         "news_burst_pct": 0.0, "dart_event": 0, "news_match": 0, "sent_delta": -0.3, "ret5": 0.0, "filtered_reason": None},
    ]


def test_has_evidence_gate():
    assert sf.has_evidence(30, 2.0) and sf.has_evidence(30, -2.0)
    assert not sf.has_evidence(29, 3.0) and not sf.has_evidence(30, 1.5) and not sf.has_evidence(30, None)


def test_forward_excess_uses_trading_days_and_skips_incomplete(db):
    p, days = db
    trading, closes = sf._closes(p)
    rows = [{"run_date": days[0], "ticker": "UP"}, {"run_date": days[0], "ticker": "FLAT"},
            {"run_date": days[-3], "ticker": "UP"}]                       # D+5 창이 안 참
    out = sf.forward_excess(rows, 5, trading, closes)
    assert {r["ticker"] for r in out} == {"UP", "FLAT"} and all(r["run_date"] == days[0] for r in out)
    up = next(r for r in out if r["ticker"] == "UP")
    expected = 1.013 ** 3 * 1.007 ** 2 - 1                                  # k=0..4: 1.013,1.007,1.013,1.007,1.013
    assert up["ret"] == pytest.approx(expected)
    assert up["excess"] == pytest.approx(expected / 2)                      # 유니버스 평균 = (ret_up + 0)/2


def test_run_date_on_weekend_enters_next_trading_day(db):
    p, days = db
    trading, closes = sf._closes(p)
    sat = "2026-09-05"                                                      # 토요일 → 진입 09-07(월)
    out = sf.forward_excess([{"run_date": sat, "ticker": "FLAT"}], 5, trading, closes)
    assert out and out[0]["ret"] == 0.0


def test_summary_selected_vs_rest_positive_when_selected_rises(db):
    p, days = db
    for d in days[:32]:
        oc.record_screen_candidates(d, _rows(), analyzed=["UP"], db_path=p)
    s = sf.summary(p)
    sel5 = s[5]["selected"]
    assert sel5["n_dates"] >= 30 and sel5["mean_diff"] > 0 and sel5["t"] > 1.96 and sel5["evidence"] is True
    assert s[5]["dart_event"]["mean_diff"] > 0 and s[5]["news_burst"]["mean_diff"] > 0
    assert s[20]["selected"]["n_dates"] < 30 and s[20]["selected"]["evidence"] is False   # D+20은 창이 덜 참


def test_summary_ignores_legacy_rows_without_factors(db):
    p, days = db
    with sqlite3.connect(p) as c:
        oc.init_outcome_tables(p)
        c.execute("INSERT INTO screen_candidates (run_date,ticker,analyzed,created_at) VALUES (?,?,1,'x')", (days[0], "UP"))
    s = sf.summary(p)
    assert s[5]["selected"]["n_dates"] == 0 and s[5]["selected"]["t"] is None


def test_format_summary_has_table(db):
    p, days = db
    for d in days[:5]:
        oc.record_screen_candidates(d, _rows(), analyzed=["UP"], db_path=p)
    text = sf.format_summary(p)
    assert "D+5" in text and "selected" in text and "게이트" in text
