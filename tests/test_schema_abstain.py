"""
tests/test_schema_abstain.py

스키마 위반 처리 (2026-09-12):
  1. data_sufficient=false면 prediction_basis·data_sources 0개 허용, true면 각 2개 필수 (조건부)
  2. 7개 에이전트 시스템 프롬프트 끝에 ABSTAIN_RULE 주입
  3. 스키마 위반 시 같은 프롬프트 반복이 아니라 위반 내용을 피드백으로 넣어 1회만 재요청,
     재요청도 실패하면 폴백(abstain)
유료 API 호출 없음.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agents import base_agent as ba
from src.schemas.agent_output import ABSTAIN_RULE, AnalysisReport


# ── 1. 조건부 검증 ─────────────────────────────────────────

def _kw(**over):
    base = dict(agent_name="quant_analyst", confidence=0.7, recommendation="BUY",
                reasoning=["a", "b", "c"], data_sources=["s1", "s2"],
                prediction_basis=["p1", "p2"], risk_factors=["r"])
    base.update(over)
    return base


def test_insufficient_allows_empty_lists():
    r = AnalysisReport(**_kw(data_sufficient=False, confidence=0.1, recommendation="HOLD",
                             reasoning=["데이터 없음"], data_sources=[], prediction_basis=[]))
    assert r.data_sources == [] and r.prediction_basis == []


def test_sufficient_requires_two_sources_and_two_basis():
    with pytest.raises(ValidationError, match="data_sources는 2개 이상"):
        AnalysisReport(**_kw(data_sources=["only-one"]))
    with pytest.raises(ValidationError, match="prediction_basis는 2개 이상"):
        AnalysisReport(**_kw(prediction_basis=["only-one"]))
    AnalysisReport(**_kw())   # 2·2면 통과


def test_fallback_report_is_marked_insufficient():
    chain = ba.ResilientChain(chain=None, breaker=ba.openai_breaker, task_name="t")
    fb = chain._make_fallback("x")
    assert fb.data_sufficient is False and fb.confidence == 0.0


# ── 2. 규칙 주입 ───────────────────────────────────────────

class _Chain:
    """가짜 구조화 체인. behaviors: 호출 순서대로 예외 또는 결과."""
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls: list[list] = []

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def _ok():
    return AnalysisReport(**_kw())


def _validation_error():
    try:
        AnalysisReport(**_kw(data_sources=["only-one"]))
    except ValidationError as e:
        return e
    raise AssertionError


@pytest.fixture(autouse=True)
def fresh_breaker_and_counters(monkeypatch):
    import pybreaker
    monkeypatch.setattr(ba, "openai_breaker", pybreaker.CircuitBreaker(fail_max=50, reset_timeout=1))
    monkeypatch.setattr(ba, "SCHEMA_RETRY_COUNT", 0)
    monkeypatch.setattr(ba, "SCHEMA_ABSTAIN_COUNT", 0)


def _resilient(chain):
    return ba.ResilientChain(chain=chain, breaker=ba.openai_breaker, timeout_seconds=5,
                             task_name="structured_agent(test)", model_name="gpt-4o-mini",
                             system_suffix=ABSTAIN_RULE, schema_feedback=True)


@pytest.mark.asyncio
async def test_rule_appended_to_system_message_once():
    chain = _Chain([_ok()])
    await _resilient(chain).ainvoke([SystemMessage(content="SYS"), HumanMessage(content="H")])
    sys_msg = chain.calls[0][0]
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content.startswith("SYS") and sys_msg.content.count("[출력 규칙") == 1
    assert "data_sufficient=false로 답하라" in sys_msg.content


def test_create_structured_agent_injects_rule(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    agent = ba.create_structured_agent()
    assert agent._system_suffix == ABSTAIN_RULE and agent._schema_feedback is True


# ── 3. 피드백 재요청 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_violation_triggers_one_feedback_retry_then_success():
    chain = _Chain([_validation_error(), _ok()])
    r = await _resilient(chain).ainvoke([SystemMessage(content="SYS"), HumanMessage(content="H")])
    assert r.confidence == 0.7
    assert len(chain.calls) == 2
    fb = chain.calls[1][-1]
    assert isinstance(fb, HumanMessage) and "스키마를 위반" in fb.content and "data_sources는 2개 이상" in fb.content
    assert "Field required" in fb.content and "risk_factors" in fb.content, "누락 필드도 피드백에 명시 (검증 실행에서 risk_factors 누락이 남은 유형)"
    assert ba.SCHEMA_RETRY_COUNT == 1 and ba.SCHEMA_ABSTAIN_COUNT == 0


@pytest.mark.asyncio
async def test_second_violation_falls_back_to_abstain():
    chain = _Chain([_validation_error(), _validation_error(), _ok()])
    r = await _resilient(chain).ainvoke([SystemMessage(content="SYS"), HumanMessage(content="H")])
    assert r.confidence == 0.0 and r.data_sufficient is False, "재요청도 실패하면 폴백"
    assert len(chain.calls) == 2, "같은 프롬프트 3회 반복은 없어야 한다"
    assert ba.SCHEMA_RETRY_COUNT == 1 and ba.SCHEMA_ABSTAIN_COUNT == 1


@pytest.mark.asyncio
async def test_non_schema_error_has_no_feedback_retry():
    chain = _Chain([RuntimeError("boom"), _ok()])
    r = await _resilient(chain).ainvoke([SystemMessage(content="SYS")])
    assert r.confidence == 0.0 and len(chain.calls) == 1
    assert ba.SCHEMA_RETRY_COUNT == 0


def test_transient_errors_are_openai_network_types():
    import openai
    assert openai.RateLimitError in ba.TRANSIENT_ERRORS
    assert ValidationError not in ba.TRANSIENT_ERRORS
