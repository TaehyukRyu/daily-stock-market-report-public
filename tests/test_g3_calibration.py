"""
tests/test_g3_calibration.py

G3 — confidence 보정: abstain(0.0) 투표 제외, 보정 표, 표본 부족 시 포지션 5% 고정.
유료 API 호출 없음.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from src.evaluation import calibration as cal
from src.schemas.agent_output import AnalysisReport


# ── 보정 표 ────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    monkeypatch.setattr(cal, "DB_PATH", path)
    from src.data import prediction_logger as pl
    monkeypatch.setattr(pl, "DB_PATH", path)
    pl.init_feedback_tables()
    return path


def _insert(path, rows):
    """rows = [(agent, confidence, score)] — score 1.0=적중(+5%), 0.0=오답(-5%), 0.5=횡보(0%, BUY 기준 미적중).
    D+10 outcomes 기준(2026-09-11): 예측마다 고유 (pred_date, ticker)와 확정된 outcome 행을 만든다."""
    from src.evaluation import outcomes as oc
    oc.init_outcome_tables(path)
    raw_of = {1.0: 0.05, 0.0: -0.05, 0.5: 0.0}
    with sqlite3.connect(path) as conn:
        base = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
        for i, (agent, conf, score) in enumerate(rows, start=base + 1):
            d, t = f"2026-01-{(i % 28) + 1:02d}", f"{i:06d}"
            conn.execute(
                "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,confidence,regime,"
                "price_at_pred,evaluated,eval_score,created_at) VALUES (?,?,'',?,'BUY',?,'Bull',1.0,0,NULL,'t')",
                (d, t, agent, conf),
            )
            conn.execute(
                "INSERT OR IGNORE INTO prediction_outcomes (pred_date,ticker,horizon_days,source,price_at_pred,"
                "price_at_res,raw_return,status,created_at) VALUES (?,?,?,'prediction',1.0,1.0,?,'resolved','t')",
                (d, t, cal.HORIZON_DAYS, raw_of[score]),
            )
        conn.commit()


def test_empty_table_gives_fixed_5pct(db):
    size, why = cal.position_size_for(0.85, "quant_analyst")
    assert size == 5.0 and "표본 부족" in why
    assert "비어 있음" in cal.format_table() and "D+10" in cal.format_table()


def test_below_min_samples_still_fixed(db):
    _insert(db, [("quant_analyst", 0.85, 1.0)] * 29)
    size, _ = cal.position_size_for(0.85, "quant_analyst")
    assert size == 5.0


def test_agent_cohort_used_when_enough(db):
    _insert(db, [("quant_analyst", 0.85, 1.0)] * 20 + [("quant_analyst", 0.82, 0.0)] * 10)   # 적중 0.667
    hit, tier = cal.calibrated_hit_rate(0.85, "quant_analyst")
    assert tier == "agent" and hit == pytest.approx(20 / 30)
    size, why = cal.position_size_for(0.85, "quant_analyst")
    assert size == pytest.approx(6.7) and "agent 코호트" in why


def test_falls_back_to_all_cohort(db):
    _insert(db, [("quant_analyst", 0.75, 1.0)] * 15 + [("technical_analyst", 0.72, 0.5)] * 15)
    hit, tier = cal.calibrated_hit_rate(0.75, "quant_analyst")
    assert tier == "all" and hit == pytest.approx(15 / 30)      # 횡보(0%)는 BUY 미적중
    size, _ = cal.position_size_for(0.75, "sentiment_analyst")
    assert size == pytest.approx(5.0)


def test_position_bounds(db):
    _insert(db, [("a", 0.9, 0.0)] * 30)      # 적중 0 → 하한 2%
    assert cal.position_size_for(0.9, "a")[0] == cal.MIN_POSITION_PCT
    _insert(db, [("b", 0.65, 1.0)] * 30)     # 적중 1.0 → 10% (상한 15 안)
    assert cal.position_size_for(0.65, "b")[0] == 10.0


def test_bin_of():
    assert cal.bin_of(0.59) == (0.0, 0.6)
    assert cal.bin_of(0.6) == (0.6, 0.7)
    assert cal.bin_of(1.0) == (0.8, 1.01)


# ── Quality Gate: abstain 제외 ─────────────────────────────

def _r(name, conf, rec="HOLD"):
    return AnalysisReport(agent_name=name, confidence=conf, recommendation=rec,
                          reasoning=["a", "b", "c"], data_sources=["x", "y"],
                          prediction_basis=["p", "q"], risk_factors=["r"])


@pytest.mark.asyncio
async def test_quality_gate_fallback_excludes_abstained():
    from src.graph.quality_gate import quality_gate_node
    from src.schemas.graph_state import GraphState
    reports = [_r("a", 0.0), _r("b", 0.0), _r("c", 0.4, "BUY")]     # 통과 0 → 폴백
    state = GraphState(ticker="005930", analysis_reports=reports)
    out = await quality_gate_node(state)
    names = [r.agent_name for r in out["qualified_reports"]]
    assert names == ["c"], "confidence 0.0(abstain)은 폴백에서도 투표에 들어가면 안 된다"
    assert any("abstain" in log for log in out["error_log"])


@pytest.mark.asyncio
async def test_quality_gate_fallback_all_abstained_gives_empty():
    from src.graph.quality_gate import quality_gate_node
    from src.schemas.graph_state import GraphState
    state = GraphState(ticker="005930", analysis_reports=[_r("a", 0.0), _r("b", 0.0)])
    out = await quality_gate_node(state)
    assert out["qualified_reports"] == []


# ── chief: 포지션 크기 보정값으로 덮어쓰기 ────────────────────

@pytest.mark.asyncio
async def test_chief_overrides_position_size_with_calibration(monkeypatch, db):
    from src.agents import chief_strategist as cs

    class _Resp:
        content = [SimpleNamespace(type="tool_use", name="submit_final_strategy", input={
            "recommendation": "BUY", "confidence": 0.8, "reasoning": ["a", "b", "c"],
            "data_sources": ["x", "y"], "prediction_basis": ["p", "q"], "risk_factors": ["r"],
            "entry_price": 100.0, "stop_loss": 96.0, "stop_loss_pct": -4.0,
            "take_profit_1": 108.0, "take_profit_2": 112.0, "rr_ratio": 2.0,
            "position_size_pct": 8.0,    # LLM은 confidence×10 = 8%를 냈다
        })]
        stop_reason = "tool_use"
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)

    class _FakeAsyncAnthropic:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=self._create)
        async def _create(self, **kw):
            return _Resp()

    monkeypatch.setattr(cs, "AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setattr(cs, "get_context_for_agent", lambda **kw: "")
    monkeypatch.setattr(cs, "inject_context_into_prompt", lambda p, c: p)

    # [2026-09-21] quant_analyst는 투표자가 아니다 → 참여 0 → HOLD. 투표자로 바꾼다.
    final = await cs.run_chief_strategist([_r("kr_market_specialist", 0.8, "BUY")],
                                          regime="bull", current_price=100.0)
    assert final.recommendation == "BUY"
    assert final.position_size_pct == 5.0, "표본 부족 → 고정 5%가 LLM의 8%를 덮어써야 한다"
    # 손절 -4%(ATR 없음) → 리스크 한도 12.5% > 보정 5% → 5% 유지
    assert any("포지션" in s for s in final.reasoning)
