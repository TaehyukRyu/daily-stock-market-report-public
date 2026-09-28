"""
tests/test_scoring_horizon.py

단계 5 — 채점을 상품에 맞춘다 (doc/2026-09-21_agent-audit.md §2-4).

[왜 두 가지를 같이 고치나]

① 지평이 안 맞았다.
   표방 포지션은 1~4주인데 채점은 D+1 방향이었다. 하루 등락으로 매긴 점수가
   EMA 가중치를 움직였다.

② HOLD만 부분점수가 없었다.
   BUY/SELL은 ±1% 안에 머물면 0.5를 받는데 HOLD는 ±1%를 벗어나면 바로 0.0이었다.
   일 변동성 6%인 종목군에서 HOLD가 1.0을 받을 확률은 낮다. 실측 결과:
     macro_economist  (통과 22표 전부 SELL)  D+1 평균 0.841  ← 1위
     technical        (HOLD 68%)             D+1 평균 0.263
   "항상 SELL"이 보상받는 채점표였다. 이대로 표본이 차면 EMA가 그것을 학습한다.

유료 API 호출 없음.
"""

from __future__ import annotations

import sqlite3

import pytest


# ── 점수표 대칭 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("rec, change, expected", [
    # 맞음 / 애매 / 틀림이 세 방향 모두 같은 모양이어야 한다
    ("BUY",   0.05, 1.0), ("BUY",   0.000, 0.5), ("BUY",  -0.05, 0.0),
    ("SELL", -0.05, 1.0), ("SELL",  0.000, 0.5), ("SELL",  0.05, 0.0),
    ("HOLD",  0.00, 1.0), ("HOLD",  0.015, 0.5), ("HOLD",  0.05, 0.0),
    ("HOLD", -0.015, 0.5), ("HOLD", -0.05, 0.0),
])
def test_score_is_symmetric_across_three_directions(rec, change, expected):
    from src.data.feedback_evaluator import _score_prediction

    assert _score_prediction(rec, 100.0, 100.0 * (1 + change)) == pytest.approx(expected)


def test_hold_gets_partial_credit_for_near_miss():
    """종전에는 HOLD만 부분점수가 없어 '항상 SELL'이 1등이 됐다."""
    from src.data.feedback_evaluator import _score_prediction

    near_miss = _score_prediction("HOLD", 100.0, 101.5)     # +1.5%, 임계 1%를 살짝 넘김
    far_miss  = _score_prediction("HOLD", 100.0, 105.0)     # +5%

    assert near_miss == 0.5
    assert far_miss == 0.0


def test_missing_entry_price_is_neutral():
    from src.data.feedback_evaluator import _score_prediction

    assert _score_prediction("BUY", None, 100.0) == 0.5
    assert _score_prediction("BUY", 0.0, 100.0) == 0.5


# ── D+5 채점 ────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    from src.data import prediction_logger as pl
    from src.data import feedback_evaluator as fe
    from src.evaluation import outcomes as oc

    monkeypatch.setattr(pl, "DB_PATH", path)
    monkeypatch.setattr(fe, "DB_PATH", path)
    monkeypatch.setattr(oc, "DB_PATH", path)
    pl.init_feedback_tables()
    pl.seed_agent_weights()
    oc.init_outcome_tables(path)
    return path


def _pred(db, pred_date, ticker, agent, rec, price=100.0, regime="Bull"):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
            "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
            "VALUES (?,?,'n',?,?,0.7,?,?,0,NULL,'t')", (pred_date, ticker, agent, rec, regime, price))
        conn.commit()


def _outcome(db, pred_date, ticker, horizon, raw_return, status="resolved"):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,price_at_pred,"
            "raw_return,status,created_at) VALUES (?,?,?,'prediction',100.0,?,?,'t')",
            (pred_date, ticker, horizon, raw_return, status))
        conn.commit()


def test_scores_from_d5_outcome_not_next_day_close(db):
    """채점의 근거는 prediction_outcomes의 D+5 실현 수익이다."""
    from src.data import feedback_evaluator as fe

    _pred(db, "2026-09-01", "005930", "kr_market_specialist", "BUY")
    _outcome(db, "2026-09-01", "005930", 5, 0.08)       # D+5 +8% → BUY 적중
    _outcome(db, "2026-09-01", "005930", 10, -0.20)     # 다른 지평은 쓰지 않는다

    result = fe.evaluate_predictions()

    assert result["evaluated_count"] == 1
    with sqlite3.connect(db) as conn:
        score = conn.execute("SELECT eval_score FROM prediction_log").fetchone()[0]
    assert score == 1.0


def test_unresolved_predictions_are_left_alone(db):
    """D+5가 아직 안 찼으면 채점하지 않는다 — 조기 확정 금지."""
    from src.data import feedback_evaluator as fe

    _pred(db, "2026-09-20", "005930", "kr_market_specialist", "BUY")
    _outcome(db, "2026-09-20", "005930", 5, None, status="pending")

    result = fe.evaluate_predictions()

    assert result["evaluated_count"] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT evaluated FROM prediction_log").fetchone()[0] == 0


def test_scoring_is_idempotent(db):
    from src.data import feedback_evaluator as fe

    _pred(db, "2026-09-01", "005930", "kr_market_specialist", "BUY")
    _outcome(db, "2026-09-01", "005930", 5, 0.08)

    assert fe.evaluate_predictions()["evaluated_count"] == 1
    assert fe.evaluate_predictions()["evaluated_count"] == 0, "이미 채점된 행을 다시 세지 않는다"


def test_chief_is_scored_but_gets_no_weight(db):
    """단계 1에서 만든 경계가 D+5 채점에서도 지켜진다."""
    from src.data import feedback_evaluator as fe
    from src.data.prediction_logger import CHIEF_AGENT

    _pred(db, "2026-09-01", "005930", "kr_market_specialist", "BUY")
    _pred(db, "2026-09-01", "005930", CHIEF_AGENT, "BUY")
    _outcome(db, "2026-09-01", "005930", 5, 0.08)

    result = fe.evaluate_predictions()

    assert result["evaluated_count"] == 2
    assert CHIEF_AGENT in result["agent_scores"]
    assert CHIEF_AGENT not in result["weight_changes"]


def test_target_date_filters_to_one_day(db):
    from src.data import feedback_evaluator as fe

    for d in ("2026-09-01", "2026-09-02"):
        _pred(db, d, "005930", "kr_market_specialist", "BUY")
        _outcome(db, d, "005930", 5, 0.08)

    assert fe.evaluate_predictions("2026-09-01")["evaluated_count"] == 1


def test_horizon_constant_matches_outcomes_table():
    """채점 지평이 prediction_outcomes가 실제로 확정하는 지평 중 하나여야 한다."""
    from src.data.feedback_evaluator import SCORING_HORIZON_DAYS
    from src.evaluation.outcomes import HORIZONS

    assert SCORING_HORIZON_DAYS in HORIZONS
    assert SCORING_HORIZON_DAYS == 5


# ── 옛 D+1 점수와 섞이지 않는다 ─────────────────────────────────────────

def test_legacy_d1_scores_are_excluded_from_evidence(db):
    """D+1로 매긴 점수와 D+5 점수를 같은 표본에 넣으면 안 된다."""
    from src.data import prediction_logger as pl

    with sqlite3.connect(db) as conn:
        for i in range(40):
            conn.execute(
                "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
                "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
                f"VALUES ('2026-08-{i%28+1:02d}','005930','n','kr_market_specialist','SELL',0.7,'Bull',100,"
                f"{pl.EVALUATED_D1_LEGACY},1.0,'t')")
        conn.commit()

    ok, why = pl.has_weight_evidence("kr_market_specialist", "Bull")
    assert ok is False, "D+1 점수 40건이 있어도 증거로 치지 않는다"
    assert "0/30" in why


def test_retire_d1_scores_marks_existing_rows(db):
    from src.data import prediction_logger as pl

    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
            "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
            "VALUES ('2026-09-01','005930','n','kr_market_specialist','BUY',0.7,'Bull',100,1,1.0,'t')")
        conn.commit()

    assert pl.retire_d1_scores() == 1
    assert pl.retire_d1_scores() == 0, "멱등"

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT evaluated, eval_score FROM prediction_log").fetchone()
    assert row[0] == pl.EVALUATED_D1_LEGACY
    assert row[1] == 1.0, "점수는 이력으로 남긴다"


def test_d1_retirement_runs_only_once(db):
    """매 실행 돌면 D+5로 새로 매긴 점수까지 도로 이력으로 바꿔버린다."""
    from src.data import feedback_evaluator as fe

    _pred(db, "2026-09-01", "005930", "kr_market_specialist", "BUY")
    _outcome(db, "2026-09-01", "005930", 5, 0.08)

    assert fe.evaluate_predictions()["evaluated_count"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT evaluated FROM prediction_log").fetchone()[0] == 1

    fe.evaluate_predictions()          # 두 번째 실행

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT evaluated FROM prediction_log").fetchone()[0] == 1, \
            "이미 D+5로 채점된 행을 되돌리면 안 된다"


def test_old_d1_rows_are_rescored_with_d5(db):
    """D+1로 매겼던 행도 D+5가 확정되면 같은 자로 다시 매긴다."""
    from src.data import feedback_evaluator as fe
    from src.data import prediction_logger as pl

    _pred(db, "2026-09-01", "005930", "kr_market_specialist", "BUY")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE prediction_log SET evaluated=?, eval_score=0.0",
                     (pl.EVALUATED_D1_LEGACY,))
        conn.commit()
    _outcome(db, "2026-09-01", "005930", 5, 0.08)      # D+5로는 적중

    assert fe.evaluate_predictions()["evaluated_count"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT eval_score FROM prediction_log").fetchone()[0] == 1.0
