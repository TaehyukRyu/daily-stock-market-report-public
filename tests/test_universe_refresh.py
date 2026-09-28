"""
tests/test_universe_refresh.py — CR-1 / 결정 3-4b.

  진실의 원천을 universe_snapshots(source='build')로 옮기고 분기 갱신을 붙였다.
  load_universe(): DB build 판 → universe_config.json → 폴백 5개
외부 호출 없음 (pykrx 미호출).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from src.evaluation import outcomes as oc
from src.evaluation import random_control as rc
from src.universe import universe_builder as ub


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(oc, "DB_PATH", path)
    rc.init_tables(path)
    return path


def test_snapshot_source_defaults_to_daily(db):
    rc.record_universe_snapshot("2026-09-14", ["A", "B"], db_path=db)
    with sqlite3.connect(db) as conn:
        srcs = {r[0] for r in conn.execute("SELECT source FROM universe_snapshots")}
    assert srcs == {"daily"}


def test_latest_built_universe_ignores_daily_rows(db):
    rc.record_universe_snapshot("2026-09-14", ["X", "Y", "Z"], db_path=db, source="daily")
    assert rc.latest_built_universe(db) == (None, [])

    rc.record_universe_snapshot("2026-10-01", ["A", "B"], db_path=db, source="build")
    rc.record_universe_snapshot("2026-10-02", ["A", "B"], db_path=db, source="daily")
    run_date, tickers = rc.latest_built_universe(db)
    assert run_date == "2026-10-01" and tickers == ["A", "B"]


def test_latest_built_universe_takes_newest_build(db):
    rc.record_universe_snapshot("2026-07-01", ["OLD"], db_path=db, source="build")
    rc.record_universe_snapshot("2026-10-01", ["NEW1", "NEW2"], db_path=db, source="build")
    assert rc.latest_built_universe(db)[1] == ["NEW1", "NEW2"]
    assert rc.build_dates(db) == ["2026-07-01", "2026-10-01"]


def test_load_universe_prefers_db(db, monkeypatch, tmp_path):
    cfg = tmp_path / "universe_config.json"
    cfg.write_text('{"tickers": ["FROM_JSON"]}', encoding="utf-8")
    monkeypatch.setattr(ub, "CONFIG_PATH", cfg)

    rc.record_universe_snapshot("2026-10-01", ["FROM_DB"], db_path=db, source="build")
    monkeypatch.setattr(oc, "DB_PATH", db)
    assert ub.load_universe() == ["FROM_DB"]


def test_load_universe_falls_back_to_json_then_default(db, monkeypatch, tmp_path):
    monkeypatch.setattr(oc, "DB_PATH", db)          # build 행 없음
    cfg = tmp_path / "universe_config.json"
    cfg.write_text('{"tickers": ["FROM_JSON"]}', encoding="utf-8")
    monkeypatch.setattr(ub, "CONFIG_PATH", cfg)
    assert ub.load_universe() == ["FROM_JSON"]

    monkeypatch.setattr(ub, "CONFIG_PATH", tmp_path / "없는파일.json")
    assert ub.load_universe() == ub._FALLBACK_TICKERS


# ── 갱신 시점 판정 ──────────────────────────────────────────

def test_refresh_not_due_outside_quarter_months(db, monkeypatch):
    monkeypatch.setattr(oc, "DB_PATH", db)
    assert ub.universe_refresh_due(datetime(2026, 9, 14)) is False   # 9월
    assert ub.universe_refresh_due(datetime(2026, 11, 1)) is False   # 11월


def test_refresh_due_when_no_build_exists(db, monkeypatch):
    monkeypatch.setattr(oc, "DB_PATH", db)
    assert ub.universe_refresh_due(datetime(2026, 10, 1)) is True


def test_refresh_not_due_twice_in_same_month(db, monkeypatch):
    monkeypatch.setattr(oc, "DB_PATH", db)
    rc.record_universe_snapshot("2026-10-01", ["A"], db_path=db, source="build")
    # 같은 달에 다시 돌리지 않는다 (cron 지연·휴장으로 1일에 못 돌아도 그 달 안에 1회)
    assert ub.universe_refresh_due(datetime(2026, 10, 1)) is False
    assert ub.universe_refresh_due(datetime(2026, 10, 20)) is False
    # 다음 분기에는 다시 돈다
    assert ub.universe_refresh_due(datetime(2027, 1, 5)) is True


def test_quarter_months_are_the_decided_ones():
    assert ub.QUARTER_MONTHS == (1, 4, 7, 10)


def test_scheduler_no_longer_hardcodes_samsung():
    """CR-12: Docker 경로가 운영과 다른 흐름을 돌면 안 된다."""
    import inspect
    from src import scheduler
    src  = inspect.getsource(scheduler.job_run_pipeline)
    body = src.split('"""')[-1]          # 독스트링은 옛 동작을 설명하므로 본문만 본다
    assert "run_daily" in body
    assert "005930" not in body
