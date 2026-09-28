"""
tests/test_g4_diversity.py

G4 — 앙상블 다양성: LLM 없는 규칙 투표자, 종목 무관 에이전트 하루 1회 캐시, 파이프라인 등록.
유료 API 호출 없음.
"""

from __future__ import annotations

import pytest

from src.agents import quant_rule_agent as qr
from src.schemas.agent_output import AnalysisReport
from src.utils import daily_cache as dc


# ── 규칙 투표자 ────────────────────────────────────────────

def _sig(pc=0.0, gc=False, mb=False, vol=False):
    cnt = sum([abs(pc) >= 0.03, vol, mb, gc])
    return {"price_change_5d": pc, "golden_cross": gc, "ma_breakout": mb,
            "volume_signal": vol, "signal_count": cnt}


@pytest.mark.parametrize("sig, rec", [
    (_sig(pc=0.01, gc=True), "BUY"),
    (_sig(pc=0.0, mb=True), "BUY"),
    (_sig(pc=-0.01, gc=True), "HOLD"),          # 전환 신호 있어도 하락 중이면 관망
    (_sig(pc=-0.04), "SELL"),
    (_sig(pc=-0.04, mb=True), "HOLD"),          # 급락이지만 돌파 신호 → 판단 유보
    (_sig(pc=0.05, vol=True), "HOLD"),          # 상승+거래량만으로는 BUY 아님
])
def test_rule_recommendation(sig, rec):
    r = qr.build_report("005930", sig)
    assert r.recommendation == rec
    assert r.agent_name == "quant_rule_agent"
    assert 0.6 <= r.confidence <= 0.9
    assert len(r.reasoning) >= 3 and len(r.prediction_basis) >= 2


def test_confidence_scales_with_signal_count():
    lo = qr.build_report("A", _sig(pc=0.0)).confidence
    hi = qr.build_report("A", _sig(pc=0.05, gc=True, mb=True, vol=True)).confidence
    assert lo == pytest.approx(0.6) and hi == pytest.approx(0.9)


def test_insufficient_data_abstains():
    r = qr.build_report("A", None)
    assert r.confidence == 0.0 and r.data_sufficient is False and r.recommendation == "HOLD"


@pytest.mark.asyncio
async def test_run_agent_uses_ohlcv_cache(monkeypatch):
    series = [{"date": f"2026-08-{i:02d}", "open": 1, "high": 1, "low": 1, "close": 100 + i, "volume": 100}
              for i in range(1, 31)]
    monkeypatch.setattr(qr, "get_ohlcv_series", lambda t, days=65: series)
    r = await qr.run_quant_rule_agent("005930")
    assert r.ticker == "005930"
    assert r.recommendation in ("BUY", "SELL", "HOLD")


@pytest.mark.asyncio
async def test_run_agent_without_ticker_abstains():
    r = await qr.run_quant_rule_agent(None)
    assert r.confidence == 0.0


# ── 하루 1회 캐시 ──────────────────────────────────────────

def _rep(conf):
    return AnalysisReport(agent_name="macro_economist", confidence=conf, recommendation="HOLD",
                          reasoning=["a", "b", "c"], data_sources=["x", "y"],
                          prediction_basis=["p", "q"], risk_factors=["r"])


@pytest.mark.asyncio
async def test_daily_cache_calls_factory_once_and_returns_copies():
    dc.clear()
    calls = []

    async def factory():
        calls.append(1)
        return _rep(0.8)

    a = await dc.daily_cached("macro_economist", factory)
    b = await dc.daily_cached("macro_economist", factory)
    assert len(calls) == 1
    a.ticker = "005930"
    assert b.ticker is None, "복사본이어야 종목별 ticker 덮어쓰기가 서로 안 섞인다"


@pytest.mark.asyncio
async def test_daily_cache_skips_fallback_results():
    dc.clear()
    calls = []

    async def factory():
        calls.append(1)
        return _rep(0.0 if len(calls) == 1 else 0.7)

    await dc.daily_cached("us_market_specialist", factory)
    r = await dc.daily_cached("us_market_specialist", factory)
    assert len(calls) == 2 and r.confidence == 0.7


# ── 파이프라인 등록 ────────────────────────────────────────

def test_pipeline_registers_rule_agent_and_daily_cache():
    import pathlib
    src = pathlib.Path("src/graph/pipeline.py").read_text(encoding="utf-8")
    assert '"quant_rule_agent"' in src and "run_quant_rule_agent(" in src
    for f in ("macro_economist", "us_market_specialist"):
        s = pathlib.Path(f"src/agents/{f}.py").read_text(encoding="utf-8")
        assert "daily_cached(" in s, f
    from src.data.prediction_logger import SCORED_AGENTS
    assert "quant_rule_agent" in SCORED_AGENTS
