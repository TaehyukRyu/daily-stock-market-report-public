"""
tests/test_option_a.py

옵션 A — CHIEF_MODE: 파이썬 투표·공식, sonnet-5 문장, legacy 모드에서 파라미터 섀도 비교.
유료 API 호출 없음.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from src.agents import chief_python as cp
from src.schemas.agent_output import AnalysisReport
from src.utils import shadow_log as sl


def _r(name, conf, rec):
    return AnalysisReport(agent_name=name, confidence=conf, recommendation=rec,
                          reasoning=["a", "b", "c"], data_sources=["x", "y"],
                          prediction_basis=["p", "q"], risk_factors=["r"])


# ── 투표 ───────────────────────────────────────────────────

def test_vote_rules_match_prompt():
    assert cp.compute_vote([_r("a", 0.8, "BUY"), _r("b", 0.5, "SELL")])[0] == "BUY"        # 0.8 > 0.5+0.2
    assert cp.compute_vote([_r("a", 0.6, "BUY"), _r("b", 0.5, "SELL")])[0] == "HOLD"       # 차이 0.1
    assert cp.compute_vote([_r("a", 0.3, "BUY"), _r("b", 0.9, "SELL")])[0] == "SELL"
    rec, conf, w = cp.compute_vote([_r("a", 0.0, "BUY"), _r("b", 0.0, "SELL")])
    assert rec == "HOLD" and conf == 0.0, "abstain만 있으면 표가 없다"


def test_vote_confidence_capped_and_ignores_abstain():
    rec, conf, w = cp.compute_vote([_r("a", 0.9, "BUY"), _r("b", 0.9, "BUY"), _r("c", 0.0, "SELL")])
    assert rec == "BUY" and conf == cp.MAX_CONFIDENCE and w["sell"] == 0.0


# ── 공식 ───────────────────────────────────────────────────

def test_trade_params_identities():
    p = cp.compute_trade_params(100.0, -4.0)
    assert p["stop_loss"] == 96.0 and p["take_profit_1"] == 108.0 and p["take_profit_2"] == 112.0
    assert p["rr_ratio"] == 2.0, "R:R은 정의상 항등식 2.0 (S1 4-2)"
    # [2026-09-21] 범위가 -3~-6 → -3~-15로 넓어졌다. ATR 손절(분석 종목 중앙값 2×7.3%
    # = -14.6%)이 옛 범위에서는 전부 -6%로 잘려 고정 손절과 다를 바 없었다.
    assert cp.compute_trade_params(100.0, -9.0)["stop_loss_pct"] == -9.0      # 이제 통과
    assert cp.compute_trade_params(100.0, -99.0)["stop_loss_pct"] == -15.0    # 상한 클램프
    assert cp.compute_trade_params(100.0, -1.0)["stop_loss_pct"] == -3.0      # 하한 클램프


def test_chief_mode_default_legacy(monkeypatch):
    monkeypatch.delenv("CHIEF_MODE", raising=False)
    assert cp.chief_mode() == "legacy"
    monkeypatch.setenv("CHIEF_MODE", "python_sonnet")
    assert cp.chief_mode() == "python_sonnet"
    monkeypatch.setenv("CHIEF_MODE", "wat")
    assert cp.chief_mode() == "legacy"


# ── python_sonnet 모드 ─────────────────────────────────────

class _Narr:
    content = [SimpleNamespace(type="tool_use", name="submit_narrative", input={
        "reasoning": ["r1", "r2", "r3"], "data_sources": ["quant_analyst", "technical_analyst"],
        "prediction_basis": ["p1", "p2"], "risk_factors": ["k"], "selection_rationale": "s",
        "stop_loss_pct": -5.0, "holding_period_weeks": 2, "entry_strategy": "분할매수",
    })]
    stop_reason = "tool_use"
    usage = SimpleNamespace(input_tokens=500, output_tokens=200)


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
    async def _create(self, **kw):
        self.calls.append(kw)
        return _Narr()


@pytest.mark.asyncio
async def test_python_sonnet_mode_buy(tmp_path, monkeypatch):
    from src.evaluation import calibration as cal
    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")   # 표 없음 → 5%
    client = _FakeClient()
    # [2026-09-21] 투표자는 4명뿐이다 (chief_python.VOTING_AGENTS). macro는 표를 안 던진다.
    reports = [_r("kr_market_specialist", 0.8, "BUY"), _r("fundamental_analyst", 0.7, "BUY"),
               _r("macro_economist", 0.6, "SELL")]
    final = await cp.run_python_sonnet(reports, "PROMPT", current_price=200.0, client=client)

    assert client.calls[0]["model"] == "claude-sonnet-5"
    assert "[파이썬 계산 결과" in client.calls[0]["messages"][0]["content"]
    assert final.recommendation == "BUY"
    # [2026-09-21] sonnet이 제안한 stop_loss_pct(-5%)는 쓰이지 않는다. 손절은 ATR 기준이고
    # ticker가 없으면 DEFAULT_STOP_PCT(-4%)로 떨어진다 → 200×0.96 = 192.
    assert final.stop_loss == 192.0 and final.take_profit_1 == 216.0 and final.rr_ratio == 2.0
    # 보정 표본 없음 → base 5%. 손절 4%면 리스크 한도(0.5/4×100 = 12.5%)가 더 커서 5% 유지.
    assert final.position_size_pct == 5.0
    assert any("[손절" in x for x in final.reasoning)
    assert final.entry_strategy == "분할매수" and final.holding_period_weeks == 2
    assert any("[결정]" in s for s in final.reasoning)


@pytest.mark.asyncio
async def test_python_sonnet_mode_hold_has_no_params():
    client = _FakeClient()
    final = await cp.run_python_sonnet([_r("kr_market_specialist", 0.6, "BUY"),
                                       _r("sentiment_analyst", 0.6, "SELL")], "P",
                                      current_price=100.0, client=client)
    assert final.recommendation == "HOLD" and final.entry_price is None and final.entry_strategy is None


# ── chief 통합: 모드 분기 + legacy 섀도 ─────────────────────

class _OpusResp:
    content = [SimpleNamespace(type="tool_use", name="submit_final_strategy", input={
        "recommendation": "BUY", "confidence": 0.8, "reasoning": ["a", "b", "c"],
        "data_sources": ["x", "y"], "prediction_basis": ["p", "q"], "risk_factors": ["r"],
        "entry_price": 100.0, "stop_loss": 96.0, "stop_loss_pct": -4.0,
        "take_profit_1": 108.0, "take_profit_2": 113.0,      # tp2를 일부러 공식과 다르게 (113 ≠ 112)
        "rr_ratio": 2.0, "position_size_pct": 8.0,
    })]
    stop_reason = "tool_use"
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)


def _patch_chief(monkeypatch, resp):
    from src.agents import chief_strategist as cs

    class _FakeAsync:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=self._create)
        async def _create(self, **kw):
            return resp
    monkeypatch.setattr(cs, "AsyncAnthropic", _FakeAsync)
    monkeypatch.setattr(cs, "get_context_for_agent", lambda **kw: "")
    monkeypatch.setattr(cs, "inject_context_into_prompt", lambda p, c: p)
    return cs


@pytest.mark.asyncio
async def test_legacy_mode_logs_param_shadow(tmp_path, monkeypatch):
    monkeypatch.delenv("CHIEF_MODE", raising=False)
    monkeypatch.setenv("SHADOW_MODE", "shadow")
    db = tmp_path / "m.db"
    monkeypatch.setattr(sl, "DB_PATH", db)
    from src.evaluation import calibration as cal
    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")
    cs = _patch_chief(monkeypatch, _OpusResp())
    monkeypatch.setattr(cs, "shadow_active", lambda name, today=None: True)

    final = await cs.run_chief_strategist([_r("kr_market_specialist", 0.8, "BUY")], regime="bull", current_price=100.0)
    assert final.recommendation == "BUY"

    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT variant, model, cost_usd FROM llm_ab_log WHERE node='chief_python_sonnet' ORDER BY variant").fetchall()
    assert [r[0] for r in rows] == ["baseline", "candidate"]
    assert rows[1][1] == "python-formula" and rows[1][2] == 0.0, "candidate는 LLM 호출 없이 비용 0"

    res = sl.compare("chief_python_sonnet", db_path=db)
    assert res["pairs"] == 1
    assert res["param_agreement"] == pytest.approx(5 / 6)       # tp2 하나만 불일치


@pytest.mark.asyncio
async def test_python_sonnet_mode_via_chief(monkeypatch, tmp_path):
    monkeypatch.setenv("CHIEF_MODE", "python_sonnet")
    monkeypatch.setenv("SHADOW_MODE", "off")
    from src.evaluation import calibration as cal
    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")
    cs = _patch_chief(monkeypatch, _Narr())
    monkeypatch.setattr(cp, "AsyncAnthropic", cs.AsyncAnthropic)

    final = await cs.run_chief_strategist([_r("kr_market_specialist", 0.8, "BUY")], regime="bull", current_price=100.0)
    # [2026-09-21] 손절은 ATR 기준. ticker 미지정 → DEFAULT -4% → 위험 4 → tp1 108
    assert final.recommendation == "BUY" and final.take_profit_1 == 108.0
