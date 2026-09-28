"""
tests/test_llm_contribution.py — 결정 3-5 (C) "LLM 기여" 집계.

규칙 투표자(quant_rule_agent)와 최종 종합(chief_strategist)이 같은 날 같은 종목에
낸 판단을 짝지어, 갈렸을 때 누가 맞았는지를 센다. 외부 호출 없음.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.evaluation import llm_contribution as lc
from src.evaluation import outcomes as oc


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(oc, "DB_PATH", path)
    monkeypatch.setattr(lc, "DB_PATH", path)
    from src.data import prediction_logger as pl
    monkeypatch.setattr(pl, "DB_PATH", path)
    pl.init_feedback_tables()
    oc.init_outcome_tables(path)
    return path


def _seed(db, cases):
    """cases: [(pred_date, ticker, rule_rec, chief_rec, raw_return)]"""
    with sqlite3.connect(db) as conn:
        for d, t, rule, chief, ret in cases:
            for agent, rec in (("quant_rule_agent", rule), ("chief_strategist", chief)):
                conn.execute(
                    "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
                    "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
                    "VALUES (?,?,'n',?,?,0.8,'Bull',100,0,NULL,'t')", (d, t, agent, rec))
            conn.execute(
                "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,raw_return,"
                "alpha_vs_kospi,status,created_at) VALUES (?,?,10,'prediction',?,?,'resolved','t')",
                (d, t, ret, ret - 0.01))
        conn.commit()


def test_empty_db_returns_zeros(db):
    s = lc.summary(10, db)
    assert s["n_pairs"] == 0 and s["disagree_rate"] is None
    assert "짝지을 표본 없음" in lc.format_summary(db)


def test_counts_agreement_and_disagreement(db):
    _seed(db, [
        ("2026-09-01", "A", "BUY",  "BUY",  0.05),   # 일치
        ("2026-09-01", "B", "SELL", "BUY",  0.05),   # 불일치 — 종합이 맞음
        ("2026-09-02", "C", "BUY",  "HOLD", 0.05),   # 불일치 — 규칙이 맞음
    ])
    s = lc.summary(10, db)
    assert s["n_pairs"] == 3
    assert s["n_days"] == 2, "표본 단위는 하루"
    assert s["agree_n"] == 1 and s["disagree_n"] == 2
    assert s["disagree_rate"] == pytest.approx(2 / 3)


def test_hit_rate_on_disagreements(db):
    """갈린 건에서 누가 맞았나 — 이것이 기여의 핵심 수치."""
    _seed(db, [
        ("2026-09-01", "B", "SELL", "BUY", 0.05),    # +5% → BUY 적중, SELL 오답
        ("2026-09-02", "C", "SELL", "BUY", 0.04),    # +4% → 같은 방향
    ])
    s = lc.summary(10, db)
    assert s["disagree_chief_hit_rate"] == 1.0
    assert s["disagree_rule_hit_rate"] == 0.0
    text = lc.format_summary(db)
    assert "갈린 건만" in text and "100%" in text


def test_hold_hit_uses_threshold_band(db):
    """HOLD는 |수익률| ≤ 문턱일 때 적중 (outcomes와 같은 규칙)."""
    thr = oc.HIT_THRESHOLD[10]
    _seed(db, [("2026-09-01", "A", "HOLD", "HOLD", thr / 2),
               ("2026-09-02", "B", "HOLD", "HOLD", thr * 3)])
    s = lc.summary(10, db)
    assert s["rule_hit_rate"] == pytest.approx(0.5)


def test_alpha_split_by_agreement(db):
    _seed(db, [
        ("2026-09-01", "A", "BUY", "BUY",  0.05),    # 일치 → alpha 0.04
        ("2026-09-02", "B", "BUY", "SELL", 0.11),    # 불일치 → alpha 0.10
    ])
    s = lc.summary(10, db)
    assert s["avg_alpha_agree"] == pytest.approx(0.04)
    assert s["avg_alpha_disagree"] == pytest.approx(0.10)


def test_rows_without_both_agents_are_excluded(db):
    """규칙만 있고 종합이 없으면 짝이 아니다."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
            "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
            "VALUES ('2026-09-01','Z','n','quant_rule_agent','BUY',0.8,'Bull',100,0,NULL,'t')")
        conn.execute(
            "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,raw_return,"
            "status,created_at) VALUES ('2026-09-01','Z',10,'prediction',0.05,'resolved','t')")
        conn.commit()
    assert lc.summary(10, db)["n_pairs"] == 0


def test_summary_notes_dependency_in_output(db):
    """chief가 규칙을 입력으로 포함한다는 한계가 출력에 남아야 한다."""
    assert "독립 비교가 아니라" in lc.format_summary(db)
