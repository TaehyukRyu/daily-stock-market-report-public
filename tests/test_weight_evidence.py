"""
가중치 증거 게이트 (2026-09-17).

에이전트 가중치가 균등(INITIAL_WEIGHT)에서 벗어나려면
  ① 채점 표본 ≥ WARMUP_SAMPLE_COUNT(30)
  ② 평균 점수가 0.5(동전 던지기)와 통계적으로 다름 (|z| ≥ 1.96)
둘 다 필요하다. 이전에는 ①이 10개뿐이었고 ②가 없어서, 동전 10번 중 7번 앞면(17% 확률)
수준의 운으로 가중치가 움직였다.

외부 호출 없음 — sqlite 임시 파일만 쓴다.
"""
import sqlite3
from datetime import datetime

import pytest

from src.data import feedback_evaluator as fe
from src.data import prediction_logger as pl

AGENT  = "quant_analyst"
OTHER  = "technical_analyst"
REGIME = "Bull"


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(pl, "DB_PATH", path)
    monkeypatch.setattr(fe, "DB_PATH", path)
    pl.init_feedback_tables()
    pl.seed_agent_weights()
    return path


def _add_scores(db, agent, scores, regime=REGIME):
    """채점 완료(evaluated=1) 예측 행을 점수 목록만큼 넣는다."""
    now = datetime.now().isoformat()
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
            "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(f"2026-09-{i % 28 + 1:02d}", "005930", "삼성전자", agent, "BUY", 0.7,
              regime, 70000.0, 1, s, now) for i, s in enumerate(scores)],
        )
        conn.commit()


def _set_weight(db, agent, weight, regime=REGIME):
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE agent_weights SET weight=? WHERE agent_name=? AND regime=?",
                     (weight, agent, regime))
        conn.commit()


def _weight_row(db, agent, regime=REGIME):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT weight, sample_count FROM agent_weights WHERE agent_name=? AND regime=?",
                            (agent, regime)).fetchone()


# ── has_weight_evidence ─────────────────────────────────────────────────────

def test_29_samples_is_not_evidence_even_if_all_hits(db):
    _add_scores(db, AGENT, [1.0] * 29)
    ok, reason = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is False
    assert "29/30" in reason


def test_30_samples_at_coin_flip_is_not_evidence(db):
    _add_scores(db, AGENT, [1.0, 0.0] * 15)          # 평균 정확히 0.5
    ok, _ = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is False


def test_30_samples_18_hits_is_not_evidence(db):
    # 18/30 = 60%. 예전 규칙(10개 중 7개)의 확대판 — z ≈ 1.1 < 1.96 → 운으로 설명된다
    _add_scores(db, AGENT, [1.0] * 18 + [0.0] * 12)
    ok, reason = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is False, reason


def test_30_samples_24_hits_is_evidence(db):
    # 24/30 = 80%. z ≈ 4.0 → 운이 아니다
    _add_scores(db, AGENT, [1.0] * 24 + [0.0] * 6)
    ok, reason = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is True, reason


def test_30_samples_all_hits_is_evidence_despite_zero_variance(db):
    _add_scores(db, AGENT, [1.0] * 30)
    ok, reason = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is True
    assert "분산 0" in reason


def test_consistently_wrong_agent_is_also_evidence(db):
    # 나쁜 쪽도 증거다 — 가중치를 내릴 근거가 된다 (양측 검정)
    _add_scores(db, AGENT, [0.0] * 26 + [1.0] * 4)
    ok, _ = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is True


def test_other_regime_samples_do_not_count(db):
    _add_scores(db, AGENT, [1.0] * 30, regime="Bear")
    ok, _ = pl.has_weight_evidence(AGENT, REGIME)
    assert ok is False


# ── get_agent_weights (읽기 쪽) ────────────────────────────────────────────────

def test_get_agent_weights_ignores_stored_weight_without_evidence(db):
    _set_weight(db, AGENT, 0.5)                       # DB에는 튀는 값이 있지만
    _add_scores(db, AGENT, [1.0] * 29)                # 증거는 부족
    w = pl.get_agent_weights(REGIME)
    assert len(set(w.values())) == 1                  # 전원 균등


def test_get_agent_weights_uses_stored_weight_with_evidence(db):
    _set_weight(db, AGENT, 0.5)
    _add_scores(db, AGENT, [1.0] * 30)
    w = pl.get_agent_weights(REGIME)
    assert w[AGENT] > w[OTHER]


# ── _update_one_weight (쓰기 쪽) ───────────────────────────────────────────────

def test_update_resets_to_initial_without_evidence(db):
    _set_weight(db, AGENT, 0.5)                       # 예전 규칙으로 움직였던 잔재
    _add_scores(db, AGENT, [1.0] * 29)
    result = {"weight_changes": {}}
    fe._update_one_weight(AGENT, REGIME, 1.0, REGIME, result)
    weight, cnt = _weight_row(db, AGENT)
    assert weight == pytest.approx(pl.INITIAL_WEIGHT)
    assert cnt == 1                                   # 표본은 계속 누적
    assert result["weight_changes"] == {}


def test_update_applies_ema_with_evidence(db):
    _add_scores(db, AGENT, [1.0] * 30)
    result = {"weight_changes": {}}
    fe._update_one_weight(AGENT, REGIME, 1.0, REGIME, result)
    weight, cnt = _weight_row(db, AGENT)
    assert weight > pl.INITIAL_WEIGHT
    assert cnt == 1
    change = result["weight_changes"][f"{AGENT}/{REGIME}"]
    assert change["old"] == pytest.approx(pl.INITIAL_WEIGHT, abs=1e-4)
    assert "evidence" in change


# ── 계층 폴백: 레짐별 → All → 균등 ──────────────────────────────────────────

def test_all_regime_rows_are_seeded(db):
    assert _weight_row(db, AGENT, regime=pl.ALL_REGIME) == (pytest.approx(pl.INITIAL_WEIGHT), 0)


def test_all_regime_evidence_pools_across_regimes(db):
    # Bull 15 + Bear 15 = 레짐별로는 각각 부족, 합치면 30
    _add_scores(db, AGENT, [1.0] * 15, regime="Bull")
    _add_scores(db, AGENT, [1.0] * 15, regime="Bear")
    assert pl.has_weight_evidence(AGENT, "Bull")[0] is False
    assert pl.has_weight_evidence(AGENT, pl.ALL_REGIME)[0] is True


def test_get_agent_weights_falls_back_to_all_weight(db):
    _set_weight(db, AGENT, 0.5, regime=pl.ALL_REGIME)
    _add_scores(db, AGENT, [1.0] * 15, regime="Bull")
    _add_scores(db, AGENT, [1.0] * 15, regime="Bear")
    w = pl.get_agent_weights("Bull")
    assert w[AGENT] > w[OTHER]                        # Bull 증거는 없지만 All 증거로 반영


def test_get_agent_weights_prefers_regime_weight_over_all(db):
    _set_weight(db, AGENT, 0.3, regime="Bull")
    _set_weight(db, AGENT, 0.6, regime=pl.ALL_REGIME)
    _add_scores(db, AGENT, [1.0] * 30, regime="Bull")  # 둘 다 증거 있음
    w = pl.get_agent_weights("Bull")
    # 정규화 전 비율로 판별: 레짐 값(0.3)을 썼다면 AGENT/OTHER = 0.3/INITIAL
    assert w[AGENT] / w[OTHER] == pytest.approx(0.3 / pl.INITIAL_WEIGHT, rel=1e-3)


def test_evaluate_predictions_updates_regime_and_all_rows(db, monkeypatch):
    # [2026-09-21] 채점 지평이 D+1 → D+5로 바뀌었다. 근거는 prediction_outcomes의
    # 확정 행이고 종가를 다시 조회하지 않는다 (feedback_evaluator._resolved_unscored).
    now = datetime.now().isoformat()
    from src.evaluation import outcomes as oc
    monkeypatch.setattr(oc, "DB_PATH", db)
    oc.init_outcome_tables(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
            "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-09-16", "005930", "삼성전자", AGENT, "BUY", 0.7, "Bull", 70000.0, 0, None, now),
        )
        conn.execute(
            "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,price_at_pred,"
            "raw_return,status,created_at) VALUES ('2026-09-16','005930',5,'prediction',70000.0,"
            "0.0286,'resolved',?)", (now,))
        conn.commit()

    result = fe.evaluate_predictions("2026-09-16")

    assert result["evaluated_count"] == 1
    assert _weight_row(db, AGENT, regime="Bull")[1] == 1
    assert _weight_row(db, AGENT, regime=pl.ALL_REGIME)[1] == 1
    assert _weight_row(db, OTHER, regime=pl.ALL_REGIME)[1] == 0   # 예측 없던 에이전트는 그대로
