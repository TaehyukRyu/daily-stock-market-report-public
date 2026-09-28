"""
tests/test_outcomes.py

G1 — 예측 → 실현 수익 귀속 (D+5/D+20, KOSPI 알파, 후보 vs 분석).
pending은 창이 다 찰 때까지 확정하지 않는다 (TradingAgents #1169 패턴).
외부 호출 없음 (KOSPI는 가짜).
"""

from __future__ import annotations

import sqlite3

import pytest

from src.evaluation import outcomes as oc


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(oc, "DB_PATH", path)
    from src.data import prediction_logger as pl
    monkeypatch.setattr(pl, "DB_PATH", path)
    pl.init_feedback_tables()
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE ohlcv_cache (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER);
        """)
        # 005930: pred_date 09-01 종가 100, 이후 5거래일 101..105, 그 뒤 없음 (D+20 미확정)
        days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"]
        closes = [100, 101, 102, 103, 104, 105]
        conn.executemany("INSERT INTO ohlcv_cache VALUES (?,?,?,?,?,?,?)",
                         [("005930", d, c, c, c, c, 1000) for d, c in zip(days, closes)])
        conn.executemany(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,confidence,regime,"
            "price_at_pred,evaluated,eval_score,created_at) VALUES (?,?,?,?,?,?,?,?,0,NULL,'t')",
            [("2026-09-01", "005930", "삼성전자", "quant_analyst", "BUY", 0.8, "Bull", 100.0),
             ("2026-09-01", "005930", "삼성전자", "technical_analyst", "SELL", 0.7, "Bull", 100.0),
             ("2026-09-01", "", "", "macro_economist", "HOLD", 0.6, "Bull", None)],   # 레거시, 무시
        )
        conn.commit()
    # KOSPI: 09-01 1000 → 09-08 1020 (+2%)
    monkeypatch.setattr(oc, "_kospi_closes", lambda s, e: {"2026-09-01": 1000.0, "2026-09-08": 1020.0})
    return path


def test_register_and_resolve_only_when_window_full(db):
    n = oc.register_pending(db)
    assert n == len(oc.HORIZONS) == 3   # (09-01,005930)×3지평(5·10·20). 레거시 빈 ticker는 제외

    r = oc.resolve_pending(db)
    assert r == {"resolved": 1, "still_pending": 2}   # D+5만 확정, D+10·D+20은 봉 부족

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        d5 = dict(conn.execute("SELECT * FROM prediction_outcomes WHERE horizon_days=5").fetchone())
        d20 = dict(conn.execute("SELECT * FROM prediction_outcomes WHERE horizon_days=20").fetchone())
    assert d5["status"] == "resolved"
    assert d5["price_at_pred"] == 100 and d5["price_at_res"] == 105
    assert d5["raw_return"] == pytest.approx(0.05)
    assert d5["kospi_return"] == pytest.approx(0.02)
    assert d5["alpha_vs_kospi"] == pytest.approx(0.03)
    assert d5["resolution_date"] == "2026-09-08"
    assert d20["status"] == "pending" and d20["raw_return"] is None


def test_resolve_is_idempotent(db):
    oc.register_pending(db)
    oc.resolve_pending(db)
    assert oc.resolve_pending(db) == {"resolved": 0, "still_pending": 2}
    assert oc.register_pending(db) == 0


def test_summary_by_agent_and_candidates(db):
    oc.record_screen_candidates("2026-09-01", [{"ticker": "005930", "is_candidate": 1}, {"ticker": "000660", "is_candidate": 1}], analyzed=["005930"], db_path=db)
    # 000660: 후보였지만 잘림. 09-01 200 → D+5 190 (-5%)
    with sqlite3.connect(db) as conn:
        days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"]
        closes = [200, 198, 196, 194, 192, 190]
        conn.executemany("INSERT INTO ohlcv_cache VALUES (?,?,?,?,?,?,?)",
                         [("000660", d, c, c, c, c, 1000) for d, c in zip(days, closes)])
        conn.commit()
    oc.register_pending(db)
    oc.resolve_pending(db)

    s = oc.summary(5, db)
    # CR-8: 표본 단위는 하루. 두 종목 모두 2026-09-01자라 n_days=1, n_predictions=2
    assert s["n_days"] == 1 and s["n_predictions"] == 2
    assert s["avg_raw"] == pytest.approx(0.0)          # (+5% + -5%) / 2 = 하루 평균 0%
    assert s["by_agent"]["quant_analyst"]["hit_rate"] == 1.0          # BUY, +5%
    assert s["by_agent"]["quant_analyst"]["buy_alpha_pos_rate"] == 1.0
    assert s["by_agent"]["technical_analyst"]["hit_rate"] == 0.0      # SELL, +5%
    assert s["candidates"]["analyzed"] == {"n": 1, "avg_raw": pytest.approx(0.05)}
    assert s["candidates"]["cut"]      == {"n": 1, "avg_raw": pytest.approx(-0.05)}

    text = oc.format_summary(db)
    assert "D+5: 1일 (2건)" in text and "상한으로 잘림 n=1" in text
    assert "069500.KS" in text


def test_export_csv(db, tmp_path):
    oc.register_pending(db)
    oc.resolve_pending(db)
    files = oc.export_csv(db, out_dir=tmp_path / "ledger")
    names = {p.name for p in files}
    assert names == {"outcomes.csv", "predictions.csv"}
    head = (tmp_path / "ledger" / "outcomes.csv").read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("pred_date,ticker,horizon_days,source,price_at_pred,price_at_res,raw_return")


def test_missing_kospi_keeps_raw(db, monkeypatch):
    monkeypatch.setattr(oc, "_kospi_closes", lambda s, e: {})
    oc.register_pending(db)
    oc.resolve_pending(db)
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT raw_return, alpha_vs_kospi FROM prediction_outcomes WHERE horizon_days=5").fetchone()
    assert row[0] == pytest.approx(0.05) and row[1] is None
