"""
tests/test_decision_rule.py

단계 4 — 최종 방향을 파이썬 규칙이 정한다 (doc/2026-09-21_agent-audit.md §1-10, §2-1, §2-3).

[왜]
  chief 프롬프트에는 "BUY 가중합 > SELL 가중합 + 0.2 → BUY"라고 적혀 있었다.
  그런데 운영 40건(2026-09-08~09-20)을 재현하면 그 규칙의 답은 BUY 15 / SELL 20 / HOLD 5인데,
  opus가 실제로 낸 것은 BUY 3 / SELL 0 / HOLD 37이었다. 프롬프트 규칙이 지켜지지 않았다.

  게다가 8명 가중합에는 상수 투표자가 섞여 있었다 — macro는 통과한 22표가 전부 SELL,
  us는 종목과 무관하게 하루 한 방향. 종목 간 차이를 못 만들고 HOLD/SELL 질량만 키웠다.

[새 규칙]
  투표자 4명(수급·재무공시·이벤트·규칙)만 센다. 반대가 없으면 그 방향, 상충하면 HOLD.
  운영 40건 재현: BUY 10 / SELL 8 / HOLD 22.

유료 API 호출 없음.
"""

from __future__ import annotations

import pytest

from src.schemas.agent_output import AnalysisReport


def _r(agent: str, rec: str = "HOLD", conf: float = 0.7) -> AnalysisReport:
    return AnalysisReport(
        agent_name=agent, recommendation=rec, confidence=conf,
        reasoning=["a", "b", "c"], data_sources=["x", "y"],
        prediction_basis=["p1", "p2"], risk_factors=["r"],
    )


# ── 투표자 명단 ─────────────────────────────────────────────────────────

def test_voting_agents_are_the_four_independent_sources():
    """서로 다른 데이터를 보는 넷만 표를 던진다."""
    from src.agents.chief_python import VOTING_AGENTS

    assert VOTING_AGENTS == (
        "kr_market_specialist",    # 수급
        "fundamental_analyst",     # 재무·공시
        "sentiment_analyst",       # 종목 이벤트
        "quant_rule_agent",        # 일봉 규칙 (LLM 없음)
    )


def test_context_agents_do_not_vote():
    """macro·us는 종목을 모르고, technical·quant는 다른 투표자와 같은 OHLCV를 본다."""
    from src.agents.chief_python import VOTING_AGENTS, decide

    reports = [
        _r("macro_economist", "SELL", 0.6),
        _r("us_market_specialist", "BUY", 0.8),
        _r("technical_analyst", "BUY", 0.85),
        _r("quant_analyst", "SELL", 0.6),
    ]
    for r in reports:
        assert r.agent_name not in VOTING_AGENTS

    rec, conf, detail = decide(reports)
    assert rec == "HOLD" and conf == 0.0
    assert detail["participants"] == 0


# ── 반대 0 규칙 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("votes, expected", [
    # (kr, fund, sent, rule) → 결정
    (("BUY",  "HOLD", "HOLD", "HOLD"), "BUY"),    # 반대 없음 → 그 방향
    (("BUY",  "BUY",  "HOLD", "HOLD"), "BUY"),
    (("BUY",  "BUY",  "BUY",  "BUY"),  "BUY"),
    (("SELL", "HOLD", "HOLD", "HOLD"), "SELL"),
    (("BUY",  "SELL", "HOLD", "HOLD"), "HOLD"),   # 상충 → 관망
    (("BUY",  "BUY",  "BUY",  "SELL"), "HOLD"),   # 3:1이어도 반대가 있으면 HOLD
    (("HOLD", "HOLD", "HOLD", "HOLD"), "HOLD"),
])
def test_direction_requires_no_opposition(votes, expected):
    from src.agents.chief_python import VOTING_AGENTS, decide

    reports = [_r(a, v) for a, v in zip(VOTING_AGENTS, votes)]
    assert decide(reports)[0] == expected


def test_abstain_is_not_a_hold_vote():
    """confidence 0.0은 '관망 의견'이 아니라 '의견 없음'이다. 참여자에서 뺀다."""
    from src.agents.chief_python import decide

    reports = [
        _r("kr_market_specialist", "BUY", 0.8),
        _r("fundamental_analyst", "HOLD", 0.0),   # abstain
        _r("sentiment_analyst", "HOLD", 0.0),     # abstain
    ]
    rec, conf, detail = decide(reports)

    assert rec == "BUY", "abstain이 반대표 노릇을 하면 안 된다"
    assert detail["participants"] == 1
    assert conf == pytest.approx(1.0 * 0.85), "1/1 동의지만 상한 0.85"


def test_no_participants_yields_no_opinion():
    from src.agents.chief_python import decide

    rec, conf, detail = decide([])
    assert rec == "HOLD" and conf == 0.0
    assert detail["participants"] == 0


# ── 확신도 ──────────────────────────────────────────────────────────────

def test_confidence_is_agreement_ratio_not_self_report():
    """자기보고 confidence는 부풀려져 있고 보정이 없다. 동의 비율을 쓴다."""
    from src.agents.chief_python import VOTING_AGENTS, decide, MAX_CONFIDENCE

    one_of_four = [_r(VOTING_AGENTS[0], "BUY", 0.99)] + [_r(a, "HOLD", 0.99) for a in VOTING_AGENTS[1:]]
    three_of_four = [_r(a, "BUY", 0.61) for a in VOTING_AGENTS[:3]] + [_r(VOTING_AGENTS[3], "HOLD", 0.61)]

    assert decide(one_of_four)[1] == pytest.approx(0.25 * MAX_CONFIDENCE)
    assert decide(three_of_four)[1] == pytest.approx(0.75 * MAX_CONFIDENCE)


def test_hold_with_participants_is_not_zero_confidence():
    """상충해서 HOLD인 것과 판단할 근거가 없는 것은 다르다."""
    from src.agents.chief_python import decide, VOTING_AGENTS

    conflicted = decide([_r(VOTING_AGENTS[0], "BUY"), _r(VOTING_AGENTS[1], "SELL")])
    assert conflicted[0] == "HOLD" and conflicted[1] > 0.0

    empty = decide([])
    assert empty[0] == "HOLD" and empty[1] == 0.0


# ── 설명용 상세 ─────────────────────────────────────────────────────────

def test_detail_lists_each_side_for_the_prompt():
    from src.agents.chief_python import decide, VOTING_AGENTS, format_decision

    reports = [
        _r(VOTING_AGENTS[0], "BUY", 0.8),
        _r(VOTING_AGENTS[1], "HOLD", 0.6),
        _r(VOTING_AGENTS[2], "BUY", 0.7),
        _r("macro_economist", "SELL", 0.6),       # 투표자 아님
    ]
    rec, conf, detail = decide(reports)

    assert detail["buy"] == [VOTING_AGENTS[0], VOTING_AGENTS[2]]
    assert detail["hold"] == [VOTING_AGENTS[1]]
    assert detail["sell"] == []
    assert VOTING_AGENTS[3] in detail["absent"], "표를 안 낸 투표자도 보여준다"

    line = format_decision(rec, conf, detail)
    assert "BUY" in line and "반대" in line


# ── LLM이 방향을 못 바꾼다 ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_python_sonnet_uses_decide_not_weighted_sum(monkeypatch, tmp_path):
    """가중합으로는 BUY인데 반대표가 있어 HOLD가 되는 표를 넣는다."""
    from types import SimpleNamespace
    from src.agents import chief_python as cp
    from src.evaluation import calibration as cal

    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")

    class _Resp:
        content = [SimpleNamespace(type="tool_use", name="submit_narrative", input={
            "reasoning": ["a", "b", "c"], "data_sources": ["x", "y"],
            "prediction_basis": ["p", "q"], "risk_factors": ["r"],
        })]
        stop_reason = "tool_use"
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)

    class _Client:
        def __init__(self): self.calls = []
        @property
        def messages(self):
            return SimpleNamespace(create=self._create)
        async def _create(self, **kw):
            self.calls.append(kw)
            return _Resp()

    client = _Client()
    reports = [
        _r("kr_market_specialist", "BUY", 0.9),
        _r("fundamental_analyst", "BUY", 0.9),
        _r("sentiment_analyst", "SELL", 0.2),      # 가중합 1.8 vs 0.2 → 옛 규칙이면 BUY
    ]
    final = await cp.run_python_sonnet(reports, "PROMPT", current_price=100.0, client=client)

    assert final.recommendation == "HOLD", "반대표가 하나라도 있으면 HOLD"
    assert "[결정]" in client.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_legacy_opus_cannot_override_direction(monkeypatch, tmp_path):
    """opus가 BUY라고 답해도 파이썬 규칙이 HOLD면 HOLD다."""
    from types import SimpleNamespace
    from src.agents import chief_strategist as cs
    from src.evaluation import calibration as cal

    monkeypatch.delenv("CHIEF_MODE", raising=False)
    monkeypatch.setenv("SHADOW_MODE", "off")
    monkeypatch.setattr(cal, "DB_PATH", tmp_path / "none.db")

    class _Resp:
        content = [SimpleNamespace(type="tool_use", name="submit_final_strategy", input={
            "recommendation": "BUY", "confidence": 0.95,      # opus의 주장
            "reasoning": ["a", "b", "c"], "data_sources": ["x", "y"],
            "prediction_basis": ["p", "q"], "risk_factors": ["r"],
            "selection_rationale": "s",
        })]
        stop_reason = "tool_use"
        usage = SimpleNamespace(input_tokens=1, output_tokens=1)

    class _FakeAsync:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=self._create)
        async def _create(self, **kw):
            return _Resp()

    monkeypatch.setattr(cs, "AsyncAnthropic", _FakeAsync)
    monkeypatch.setattr(cs, "get_context_for_agent", lambda **kw: "")
    monkeypatch.setattr(cs, "inject_context_into_prompt", lambda p, c: p)

    reports = [
        _r("kr_market_specialist", "BUY", 0.9),
        _r("sentiment_analyst", "SELL", 0.9),     # 상충
    ]
    final = await cs.run_chief_strategist(reports, regime="bull", current_price=100.0, ticker="005930")

    assert final.recommendation == "HOLD", "방향은 파이썬이 정한다"
    assert final.confidence < 0.95
    assert final.entry_price is None, "HOLD면 거래 파라미터가 없다"
