"""
tests/test_g5_no_signal.py

G5 — "신호 없음"을 출력할 수 있는가.
  1. Stage 2: tests/test_stage2_v2.py로 이동 (v2)
  2. daily_runner: 확정 0개면 optional로 메우지 않는다
  3. chief: 파싱 실패 → REVIEW (HOLD 날조 금지), 장중이면 '즉시 매수' 금지 주의
  4. 리포트: REVIEW 표시
유료 API 호출 없음.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from src.schemas.agent_output import AnalysisReport
from src.utils import market_session as ms


# ── 2. daily_runner ───────────────────────────────────────

def test_select_tickers_returns_empty_without_confirmed():
    from src.daily_runner import _select_tickers
    picked, mode = _select_tickers({"confirmed": [], "optional": ["A", "B", "C"]})
    assert picked == []
    assert "신호 없음" in mode


def test_select_tickers_keeps_confirmed_cap():
    from src.daily_runner import _select_tickers, MAX_CONFIRMED_TICKERS
    confirmed = [f"T{i}" for i in range(MAX_CONFIRMED_TICKERS + 3)]
    picked, _ = _select_tickers({"confirmed": confirmed, "optional": ["Z"]})
    assert picked == confirmed[:MAX_CONFIRMED_TICKERS]


def test_select_tickers_respects_regime_cap():
    from src.daily_runner import _select_tickers, MAX_CONFIRMED_TICKERS
    confirmed = [f"T{i}" for i in range(MAX_CONFIRMED_TICKERS)]
    picked, mode = _select_tickers({"confirmed": confirmed, "optional": [], "cap": 5})
    assert picked == confirmed[:5] and "상한" in mode
    picked, _ = _select_tickers({"confirmed": confirmed, "optional": [], "cap": 50})   # 코드 상한이 더 작으면 그것
    assert picked == confirmed[:MAX_CONFIRMED_TICKERS]


# ── 3. chief ──────────────────────────────────────────────

def _agent_report(rec="BUY", conf=0.8):
    return AnalysisReport(agent_name="quant_analyst", confidence=conf, recommendation=rec,
                          reasoning=["a", "b", "c"], data_sources=["x", "y"],
                          prediction_basis=["p", "q"], risk_factors=["r"])


@pytest.mark.asyncio
async def test_chief_parse_failure_yields_review(monkeypatch):
    from src.agents import chief_strategist as cs

    class _Resp:
        content = [SimpleNamespace(type="text", text="죄송합니다, 판단할 수 없습니다.")]
        stop_reason = "end_turn"
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)

    class _FakeAsyncAnthropic:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=self._create)
        async def _create(self, **kw):
            return _Resp()

    monkeypatch.setattr(cs, "AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setattr(cs, "get_context_for_agent", lambda **kw: "")
    monkeypatch.setattr(cs, "inject_context_into_prompt", lambda p, c: p)

    final = await cs.run_chief_strategist([_agent_report()], regime="bull")
    assert final.needs_review is True
    assert final.recommendation == "HOLD" and final.confidence == 0.0
    assert final.reasoning[0].startswith("[REVIEW]")


def test_chief_prompt_does_not_force_direction():
    """[2026-09-21] 방향 선택은 프롬프트 문구가 아니라 decide()가 보장한다.

    종전에는 "HOLD는 실패가 아니라 정당한 결론"이라고 써 두고 opus가 지키기를 기대했다.
    운영 40건 재현에서 프롬프트 규칙과 실제 출력이 달랐다(BUY 15 vs 3).
    이제 방향은 파이썬이 정하므로, 강요할 문구도 지킬 문구도 없다.
    """
    from src.agents.chief_strategist import CHIEF_SYSTEM_PROMPT
    from src.agents.chief_python import decide

    assert "방향은 당신이 정하지 않습니다" in CHIEF_SYSTEM_PROMPT

    # 신호가 상충하면 구조적으로 HOLD다 — 모델 선택에 맡기지 않는다
    from src.schemas.agent_output import AnalysisReport

    def _vote(agent, rec):
        return AnalysisReport(agent_name=agent, recommendation=rec, confidence=0.9,
                              reasoning=["a", "b", "c"], data_sources=["x", "y"],
                              prediction_basis=["p", "q"], risk_factors=["r"])

    assert decide([_vote("kr_market_specialist", "BUY"),
                   _vote("sentiment_analyst", "SELL")])[0] == "HOLD"


def test_market_session_note_only_during_hours():
    open_time  = datetime(2026, 9, 11, 10, 30, tzinfo=ms.KST)   # 금요일 장중
    close_time = datetime(2026, 9, 11, 4, 10, tzinfo=ms.KST)    # 새벽 (정상 스케줄)
    weekend    = datetime(2026, 9, 12, 10, 30, tzinfo=ms.KST)   # 토요일
    assert ms.is_market_hours_kst(open_time)
    assert not ms.is_market_hours_kst(close_time)
    assert not ms.is_market_hours_kst(weekend)
    assert "시장가" in ms.market_session_note(open_time)
    assert ms.market_session_note(close_time) == ""


@pytest.mark.asyncio
async def test_chief_appends_session_note_when_market_open(monkeypatch):
    from src.agents import chief_strategist as cs
    captured = {}

    class _Resp:
        content = [SimpleNamespace(type="tool_use", name="submit_final_strategy", input={
            "recommendation": "HOLD", "confidence": 0.5, "reasoning": ["a", "b", "c"],
            "data_sources": ["x", "y"], "prediction_basis": ["p", "q"], "risk_factors": ["r"],
        })]
        stop_reason = "tool_use"
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)

    class _FakeAsyncAnthropic:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=self._create)
        async def _create(self, **kw):
            captured["messages"] = kw["messages"]
            return _Resp()

    monkeypatch.setattr(cs, "AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setattr(cs, "get_context_for_agent", lambda **kw: "")
    monkeypatch.setattr(cs, "inject_context_into_prompt", lambda p, c: p)
    monkeypatch.setattr(cs, "market_session_note", lambda: "\n[장중 실행 주의] TEST\n")

    final = await cs.run_chief_strategist([_agent_report()], regime="bull")
    assert final.needs_review is False
    assert "[장중 실행 주의]" in captured["messages"][0]["content"]


# ── 4. 리포트 ─────────────────────────────────────────────

def test_report_shows_review_badge():
    from src.graph import report_formatter as rf
    final = AnalysisReport(agent_name="chief_strategist", confidence=0.0, recommendation="HOLD",
                           needs_review=True, reasoning=["[REVIEW] x", "y", "z"],
                           data_sources=["a", "b"], prediction_basis=["p", "q"], risk_factors=["r"])
    text = "\n".join(rf._agent_summary_section([_agent_report()], final))
    assert "REVIEW" in text and "사람 확인 필요" in text
