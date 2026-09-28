"""
tests/test_performance.py

자동 청산(auto_close_triggered_positions) + 성과 집계(performance) 회귀 테스트.

[배경]
  close_position()의 호출자가 CLI 하나뿐이라 포지션이 닫히지 않았다.
  2026-09-08 실측: positions 테이블 0행, 실현 손익 기록 0건.
  → 승률·수익률을 계산할 대상 자체가 없었다.
"""

import sqlite3
import sys
import types
from pathlib import Path

import pytest

from src.data import position_tracker as pt
from src.evaluation import performance as perf


# ─────────────────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(pt, "DB_PATH", path)
    monkeypatch.setattr(perf, "DB_PATH", path)
    pt.init_position_table()
    return path


def _fake_price(monkeypatch, price: float):
    """update_current_prices()가 함수 안에서 import하는 pykrx를 가로챈다."""
    class _S:
        iloc = [price]

    class _DF:
        empty = False
        columns = ["종가"]
        def __getitem__(self, k):
            return _S()

    stock = types.SimpleNamespace(get_market_ohlcv_by_date=lambda *a, **k: _DF())
    mod = types.ModuleType("pykrx")
    mod.stock = stock
    monkeypatch.setitem(sys.modules, "pykrx", mod)
    monkeypatch.setitem(sys.modules, "pykrx.stock", stock)


def _status(db_path, ticker):
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        r = c.execute(
            "SELECT status, close_reason, realized_pnl_pct FROM positions WHERE ticker=?",
            (ticker,),
        ).fetchone()
    return dict(r) if r else {}


# ─────────────────────────────────────────────────────────
# 1. 자동 청산
# ─────────────────────────────────────────────────────────

def test_stop_loss_triggers_auto_close(db, monkeypatch):
    pt.add_position(ticker="005930", entry_price=70_000, quantity=0,
                    stop_loss=67_200, target_price=75_600, allocation_pct=10.0)
    _fake_price(monkeypatch, 67_000)          # 손절가 아래

    closed = pt.auto_close_triggered_positions()

    assert len(closed) == 1
    row = _status(db, "005930")
    assert row["status"] == "closed"
    assert row["close_reason"] == "stop_loss"
    assert row["realized_pnl_pct"] == pytest.approx(-4.29, abs=0.05)


def test_target_triggers_auto_close(db, monkeypatch):
    pt.add_position(ticker="000660", entry_price=70_000, quantity=0,
                    stop_loss=67_200, target_price=75_600, allocation_pct=10.0)
    _fake_price(monkeypatch, 76_000)          # 목표가 위

    closed = pt.auto_close_triggered_positions()

    assert len(closed) == 1
    assert _status(db, "000660")["close_reason"] == "target"


def test_position_between_bounds_stays_open(db, monkeypatch):
    """손절과 목표 사이면 건드리지 않는다 — 과잉 청산 방지."""
    pt.add_position(ticker="005380", entry_price=70_000, quantity=0,
                    stop_loss=67_200, target_price=75_600)
    _fake_price(monkeypatch, 71_000)

    assert pt.auto_close_triggered_positions() == []
    assert _status(db, "005380")["status"] == "open"


def test_auto_close_uses_propagated_stop_not_default(db, monkeypatch):
    """
    0순위 수정과의 연결 — 청산 판정이 전달된 손절가(-4%)를 쓰는지 확인.
    옛 기본값 -7%(65,100)를 보고 있으면 67,000원에서는 청산되지 않는다.
    """
    pt.add_position(ticker="051910", entry_price=70_000, quantity=0,
                    stop_loss=67_200, target_price=75_600)
    _fake_price(monkeypatch, 67_000)

    assert len(pt.auto_close_triggered_positions()) == 1


def test_no_open_positions_is_noop(db, monkeypatch):
    _fake_price(monkeypatch, 70_000)
    assert pt.auto_close_triggered_positions() == []


# ─────────────────────────────────────────────────────────
# 2. 성과 집계
# ─────────────────────────────────────────────────────────

def _close_at(monkeypatch, ticker, entry, price, alloc):
    pt.add_position(ticker=ticker, entry_price=entry, quantity=0,
                    stop_loss=entry * 0.96, target_price=entry * 1.08,
                    allocation_pct=alloc)
    _fake_price(monkeypatch, price)
    pt.auto_close_triggered_positions()


def test_empty_summary_is_explicit(db):
    s = perf.get_performance_summary()
    assert s["closed_count"] == 0
    assert s["win_rate_pct"] is None
    assert "닫힌 포지션이 없어" in s["sample_warning"]


def test_win_rate_suppressed_below_min_samples(db, monkeypatch):
    """3승 1패를 '75% 승률'로 부르지 않는다."""
    for i, (entry, price) in enumerate([(100, 108), (100, 108), (100, 108), (100, 96)]):
        _close_at(monkeypatch, f"00000{i}", entry, price, 10.0)

    s = perf.get_performance_summary()
    assert s["closed_count"] == 4
    assert s["win_count"] == 3
    assert s["win_rate_pct"] is None
    assert "표본 4건" in s["sample_warning"]


def test_win_rate_reported_at_min_samples(db, monkeypatch):
    for i in range(perf.MIN_SAMPLES_FOR_WINRATE):
        price = 108 if i < 6 else 96
        _close_at(monkeypatch, f"1{i:05d}", 100, price, 10.0)

    s = perf.get_performance_summary()
    assert s["closed_count"] == perf.MIN_SAMPLES_FOR_WINRATE
    assert s["win_rate_pct"] == 60.0


def test_cumulative_seed_pct_is_allocation_weighted(db, monkeypatch):
    """배분 비중을 곱해야 실제 포트폴리오 손익에 가깝다."""
    _close_at(monkeypatch, "005930", 100, 110, 20.0)   # +10% × 20% = +2.0
    _close_at(monkeypatch, "000660", 100, 90,  10.0)   # -10% × 10% = -1.0

    s = perf.get_performance_summary()
    assert s["cumulative_seed_pct"] == pytest.approx(1.0, abs=0.01)
    assert s["avg_pnl_pct"] == pytest.approx(0.0, abs=0.01)   # 단순평균은 0


def test_best_and_worst_are_identified(db, monkeypatch):
    _close_at(monkeypatch, "005930", 100, 115, 10.0)
    _close_at(monkeypatch, "000660", 100, 92,  10.0)

    s = perf.get_performance_summary()
    assert s["best"]["ticker"] == "005930"
    assert s["worst"]["ticker"] == "000660"


def test_kospi_failure_does_not_break_summary(db, monkeypatch):
    """지수 조회가 실패해도 나머지 집계는 나와야 한다."""
    monkeypatch.setattr(perf, "_kospi_return_pct", lambda *a: None)
    _close_at(monkeypatch, "005930", 100, 110, 20.0)

    s = perf.get_performance_summary()
    assert s["kospi_return_pct"] is None
    assert s["excess_vs_kospi_pct"] is None
    assert s["cumulative_seed_pct"] == pytest.approx(2.0, abs=0.01)


def test_excess_vs_kospi_is_difference(db, monkeypatch):
    monkeypatch.setattr(perf, "_kospi_return_pct", lambda *a: 1.5)
    _close_at(monkeypatch, "005930", 100, 110, 20.0)   # 시드 대비 +2.0%

    s = perf.get_performance_summary()
    assert s["excess_vs_kospi_pct"] == pytest.approx(0.5, abs=0.01)


def test_markdown_section_warns_about_fill_price(db, monkeypatch):
    """리포트 문구에 '체결가가 아니다' 경고가 반드시 남아야 한다."""
    _close_at(monkeypatch, "005930", 100, 110, 20.0)
    md = perf.format_performance_section()
    assert "실현 성과" in md
    assert "체결가가 아니다" in md
