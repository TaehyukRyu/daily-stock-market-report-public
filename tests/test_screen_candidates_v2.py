"""
tests/test_screen_candidates_v2.py — screen_candidates 확장(스크리닝 v2 §4). 임시 DB, 외부 호출 없음.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.evaluation import outcomes as oc


@pytest.fixture
def db(tmp_path):
    return tmp_path / "o.db"


def _row(ticker, cand=1, rank=None, score=0.5, **kw):
    base = {"ticker": ticker, "is_candidate": cand, "rank": rank, "score": score, "news_burst": 2.0,
            "news_burst_pct": 0.9, "dart_event": 0, "news_match": 0, "sent_delta": None, "ret5": 0.01,
            "filtered_reason": None}
    base.update(kw)
    return base


def _cols(db):
    with sqlite3.connect(db) as c:
        return {r[1] for r in c.execute("PRAGMA table_info(screen_candidates)")}


def test_migration_adds_columns_and_is_idempotent(db):
    oc.init_outcome_tables(db)
    oc.init_outcome_tables(db)
    assert {"is_candidate", "cand_rank", "score", "news_burst", "news_burst_pct", "dart_event",
            "news_match", "sent_delta", "ret5", "filtered_reason"} <= _cols(db)


def test_migration_on_legacy_table(db):
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE screen_candidates (run_date TEXT NOT NULL, ticker TEXT NOT NULL, "
                  "analyzed INTEGER NOT NULL, created_at TEXT NOT NULL, UNIQUE (run_date, ticker))")
        c.execute("INSERT INTO screen_candidates VALUES ('2026-09-18','000660',1,'x')")
    oc.init_outcome_tables(db)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT is_candidate, cand_rank FROM screen_candidates").fetchone() == (None, None)


def test_record_writes_all_rows_with_factors(db):
    rows = [_row("A", 1, 1, 0.9), _row("B", 1, 2, 0.6), _row("C", 0, None, 0.1, filtered_reason="already_moved")]
    assert oc.record_screen_candidates("2026-09-21", rows, analyzed=["A"], db_path=db) == 3
    with sqlite3.connect(db) as c:
        got = c.execute("SELECT ticker, analyzed, is_candidate, cand_rank, score, filtered_reason "
                        "FROM screen_candidates ORDER BY ticker").fetchall()
    assert got == [("A", 1, 1, 1, 0.9, None), ("B", 0, 1, 2, 0.6, None), ("C", 0, 0, None, 0.1, "already_moved")]


def test_record_same_day_twice_keeps_first(db):
    oc.record_screen_candidates("2026-09-21", [_row("A", score=0.9)], ["A"], db_path=db)
    oc.record_screen_candidates("2026-09-21", [_row("A", score=0.1)], [], db_path=db)
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT analyzed, score FROM screen_candidates").fetchall() == [(1, 0.9)]


def test_record_empty_rows_is_noop(db):
    assert oc.record_screen_candidates("2026-09-21", [], [], db_path=db) == 0


def test_register_pending_only_candidates_and_analyzed(db):
    oc.record_screen_candidates("2026-09-21",
        [_row("A", 1, 1), _row("B", 0, None), _row("C", 0, None)], analyzed=["C"], db_path=db)
    with sqlite3.connect(db) as c:      # 레거시 행 (is_candidate NULL) 도 등록돼야 한다
        c.execute("INSERT INTO screen_candidates (run_date,ticker,analyzed,created_at) VALUES ('2026-09-18','L',0,'x')")
    n = oc.register_pending(db)
    with sqlite3.connect(db) as c:
        tickers = {r[0] for r in c.execute("SELECT DISTINCT ticker FROM prediction_outcomes WHERE source='candidate'")}
    assert tickers == {"A", "C", "L"} and n == 3 * len(oc.HORIZONS)


def test_method_applied_date_first_write_wins(db):
    assert oc.method_applied_date("screen_v2", db) is None
    oc.record_method_change("screen_v2", "v2", applied_date="2026-09-22", db_path=db)
    oc.record_method_change("screen_v2", "v2 again", applied_date="2026-10-01", db_path=db)
    assert oc.method_applied_date("screen_v2", db) == "2026-09-22"
    assert oc.SCREEN_V2_KEY == "screen_v2"
