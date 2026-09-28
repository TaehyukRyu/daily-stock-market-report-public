"""
tests/test_chief_ledger.py

단계 1 — chief_strategist 최종 판단을 예측 원장에 남긴다 (doc/2026-09-21_agent-audit.md §2-6).

[왜]
  chief는 이 시스템의 **출력**인데 prediction_log에 한 행도 없었다. SCORED_AGENTS(투표자 8명)가
  저장 필터를 겸하고 있었기 때문이다. 그 결과:
    - "시스템 BUY를 따르면 버는가"를 잴 표본이 0
    - llm_contribution이 chief 행을 조인하는데 매칭 0건 → 집계가 영원히 빈 값
    - calibration의 chief 코호트가 영원히 비어 포지션이 5% 고정

[경계]
  chief는 **기록만** 한다. 투표자가 아니므로 agent_weights(EMA 가중치)에는 들어가지 않는다.
  INITIAL_WEIGHT = 1/8도 그대로다. 유료 API 호출 없음.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.schemas.agent_output import AnalysisReport


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    from src.data import prediction_logger as pl
    from src.data import feedback_evaluator as fe
    from src.evaluation import outcomes as oc
    from src.evaluation import llm_contribution as lc

    monkeypatch.setattr(pl, "DB_PATH", path)
    monkeypatch.setattr(fe, "DB_PATH", path)
    monkeypatch.setattr(oc, "DB_PATH", path)
    monkeypatch.setattr(lc, "DB_PATH", path)
    pl.init_feedback_tables()
    pl.seed_agent_weights()
    oc.init_outcome_tables(path)
    return path


def _report(agent: str, rec: str = "BUY", conf: float = 0.7, ticker: str = "005930") -> AnalysisReport:
    return AnalysisReport(
        agent_name=agent, recommendation=rec, confidence=conf, ticker=ticker, ticker_name="종목",
        reasoning=["a", "b", "c"], data_sources=["x", "y"],
        prediction_basis=["p1", "p2"], risk_factors=["r"],
    )


def _rows(path, sql, *params):
    with sqlite3.connect(path) as conn:
        return conn.execute(sql, params).fetchall()


# ── 저장 ────────────────────────────────────────────────────────────────

def test_chief_report_is_saved_to_prediction_log(db):
    """chief 리포트가 원장에 들어간다 — 이 테스트가 단계 1의 본체다."""
    from src.data.prediction_logger import log_agent_predictions

    n = log_agent_predictions(
        [_report("technical_analyst"), _report("chief_strategist", "BUY", 0.62)],
        regime="Bull", pred_date="2026-09-21", price_at_pred=100.0,
    )
    assert n == 2
    saved = _rows(db, "SELECT agent_name, recommendation, confidence FROM prediction_log ORDER BY agent_name")
    assert ("chief_strategist", "BUY", 0.62) in saved


def test_unknown_agent_is_still_filtered_out(db):
    """필터를 넓힌 것이지 없앤 것이 아니다."""
    from src.data.prediction_logger import log_agent_predictions

    n = log_agent_predictions([_report("made_up_agent")], regime="Bull", pred_date="2026-09-21")
    assert n == 0


# ── chief는 투표자가 아니다 ─────────────────────────────────────────────

def test_chief_has_no_weight_row_and_initial_weight_unchanged(db):
    """가중치는 투표자 8명만. chief가 끼면 1/9이 되고 정규화가 흔들린다."""
    from src.data.prediction_logger import (
        INITIAL_WEIGHT, SCORED_AGENTS, CHIEF_AGENT, get_agent_weights,
    )

    assert CHIEF_AGENT not in SCORED_AGENTS
    assert len(SCORED_AGENTS) == 8
    assert INITIAL_WEIGHT == pytest.approx(1 / 8)
    assert _rows(db, "SELECT COUNT(*) FROM agent_weights WHERE agent_name=?", CHIEF_AGENT)[0][0] == 0
    assert CHIEF_AGENT not in get_agent_weights("Bull")


def test_scoring_scores_chief_but_does_not_touch_weights(db, monkeypatch):
    """D+1 채점은 chief도 매긴다. 다만 EMA 가중치는 건드리지 않는다."""
    from src.data import feedback_evaluator as fe
    from src.data.prediction_logger import log_agent_predictions, CHIEF_AGENT

    log_agent_predictions(
        [_report("technical_analyst", "BUY", 0.8), _report(CHIEF_AGENT, "BUY", 0.62)],
        regime="Bull", pred_date="2026-09-20", price_at_pred=100.0,
    )
    # [2026-09-21] D+5 확정 수익 +5% → BUY 적중 (채점 근거가 prediction_outcomes로 바뀌었다)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,price_at_pred,"
            "raw_return,status,created_at) VALUES ('2026-09-20','005930',5,'prediction',100.0,"
            "0.05,'resolved','t')")
        conn.commit()

    result = fe.evaluate_predictions("2026-09-20")

    assert result["evaluated_count"] == 2
    scored = dict(_rows(db, "SELECT agent_name, eval_score FROM prediction_log WHERE evaluated=1"))
    assert scored[CHIEF_AGENT] == 1.0, "chief도 채점된다"
    assert CHIEF_AGENT in result["agent_scores"]
    assert CHIEF_AGENT not in result["weight_changes"]
    assert _rows(db, "SELECT COUNT(*) FROM agent_weights WHERE agent_name=?", CHIEF_AGENT)[0][0] == 0


# ── 하류 소비자 ─────────────────────────────────────────────────────────

def test_llm_contribution_can_pair_rule_and_chief(db):
    """llm_contribution은 chief 행을 조인한다. 원장에 chief가 없으면 표본이 영원히 0이었다."""
    from src.data.prediction_logger import log_agent_predictions
    from src.evaluation import llm_contribution as lc

    log_agent_predictions(
        [_report("quant_rule_agent", "HOLD", 0.6), _report("chief_strategist", "BUY", 0.7)],
        regime="Bull", pred_date="2026-09-01", price_at_pred=100.0,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,raw_return,"
            "alpha_vs_kospi,status,created_at) VALUES ('2026-09-01','005930',10,'prediction',0.05,0.04,'resolved','t')"
        )
        conn.commit()

    assert lc.summary(10, db)["n_pairs"] == 1


def test_register_pending_marks_chief_ticker(db):
    """chief가 판단한 종목은 D+5/10/20 대기열에 올라간다."""
    from src.data.prediction_logger import log_agent_predictions
    from src.evaluation import outcomes as oc

    log_agent_predictions([_report("chief_strategist")], regime="Bull",
                          pred_date="2026-09-21", price_at_pred=100.0)
    oc.register_pending(db)

    horizons = sorted(h for h, in _rows(
        db, "SELECT horizon_days FROM prediction_outcomes WHERE ticker='005930' AND source='prediction'"))
    assert horizons == [5, 10, 20]


# ── 과거분 이관 ─────────────────────────────────────────────────────────

def test_backfill_from_chief_decisions_is_idempotent(db):
    """decision_memory에만 있던 과거 chief 판단을 원장으로 옮긴다. 두 번 돌려도 중복 없음."""
    from src.data.prediction_logger import backfill_chief_from_decisions

    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chief_decisions ("
            "id INTEGER PRIMARY KEY, pred_date TEXT, ticker TEXT, ticker_name TEXT, recommendation TEXT,"
            "confidence REAL, regime TEXT, reasoning TEXT, price_at_pred REAL, outcome_date TEXT,"
            "actual_price REAL, return_pct REAL, score REAL, lesson TEXT, resolved INTEGER, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO chief_decisions (pred_date,ticker,ticker_name,recommendation,confidence,regime,"
            "reasoning,price_at_pred,resolved,created_at) "
            "VALUES ('2026-09-20','028260','삼성물산','HOLD',0.55,'Sideways','r',353500.0,1,'t')"
        )
        conn.commit()

    assert backfill_chief_from_decisions() == 1
    assert backfill_chief_from_decisions() == 0

    rows = _rows(db, "SELECT pred_date, ticker, recommendation, confidence, regime, price_at_pred "
                     "FROM prediction_log WHERE agent_name='chief_strategist'")
    assert rows == [("2026-09-20", "028260", "HOLD", 0.55, "Sideways", 353500.0)]


# ── 파이프라인 연결 ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_predictions_node_includes_chief(db, monkeypatch):
    """quality_gate는 chief 실행 전에 돌아 chief를 모른다. 노드가 직접 붙여야 한다."""
    from src.graph import pipeline as pp
    from src.schemas.graph_state import GraphState

    captured: dict = {}

    def _fake_log(qualified_reports, regime, price_at_pred=None):
        captured["agents"] = [r.agent_name for r in qualified_reports]
        return len(qualified_reports)

    monkeypatch.setattr(pp, "log_agent_predictions", _fake_log)

    voter = _report("technical_analyst", "HOLD", 0.8)
    chief = _report("chief_strategist", "BUY", 0.62)
    state = GraphState(
        ticker="005930",
        market_data={"stock": {"latest_close": 100.0}},
        analysis_reports=[voter, chief],
        qualified_reports=[voter],          # quality_gate 산출물 — chief 없음
        current_regime="Bull",
    )

    await pp.log_predictions_node(state)

    assert captured["agents"] == ["technical_analyst", "chief_strategist"]


@pytest.mark.asyncio
async def test_log_predictions_node_skips_review_chief(db, monkeypatch):
    """confidence 0.0은 판단이 아니라 '판단 없음'(REVIEW) — 원장에 남기지 않는다."""
    from src.graph import pipeline as pp
    from src.schemas.graph_state import GraphState

    captured: dict = {}
    monkeypatch.setattr(pp, "log_agent_predictions",
                        lambda qualified_reports, regime, price_at_pred=None:
                        captured.setdefault("agents", [r.agent_name for r in qualified_reports]) and 0)

    voter = _report("technical_analyst", "HOLD", 0.8)
    chief = _report("chief_strategist", "HOLD", 0.0)
    state = GraphState(ticker="005930", market_data={"stock": {"latest_close": 100.0}},
                       analysis_reports=[voter, chief], qualified_reports=[voter],
                       current_regime="Bull")

    await pp.log_predictions_node(state)

    assert captured["agents"] == ["technical_analyst"]
