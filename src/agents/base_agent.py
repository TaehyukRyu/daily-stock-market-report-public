"""
src/agents/base_agent.py

에이전트 팩토리 모음.
변경 이력:
  - v1.0: LangChain with_structured_output 기반 팩토리
  - v2.0: [DEP-01] langchain_anthropic 제거 → anthropic SDK 직접 사용
           Circuit Breaker + Timeout 추가 (ResilientChain 래퍼)
"""

import asyncio
import json
import logging
import re
from typing import Any

import pybreaker
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from src.schemas.agent_output import AnalysisReport, ABSTAIN_RULE
from src.utils.llm_budget import make_budget_callback
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

# 스키마 위반 재요청 횟수 (프로세스 누적). 실행 로그 요약용.
SCHEMA_RETRY_COUNT = 0     # 피드백 재요청을 보낸 횟수
SCHEMA_ABSTAIN_COUNT = 0   # 재요청도 실패해 폴백(abstain)으로 끝난 횟수

# 일시 오류만 같은 프롬프트로 재시도한다. 스키마 위반(ValidationError)은 같은 프롬프트를
# 반복해도 같은 답이 나오므로 제외하고, 아래 _invoke_with_schema_feedback이 1회만 피드백을 넣는다.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = tuple(
    getattr(__import__("openai"), n) for n in
    ("APIConnectionError", "RateLimitError", "APITimeoutError", "InternalServerError")
)
from src.utils.resilience import (
    openai_breaker,
    anthropic_breaker,
    with_timeout,
    get_breaker_status,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# ResilientChain: LangChain 체인 + Circuit Breaker + Timeout
# ──────────────────────────────────────────────────────────

class ResilientChain:
    """
    LangChain 체인을 감싸서 Circuit Breaker + Timeout을 추가하는 래퍼.

    기존 에이전트 호출 방식 그대로 유지:
        agent = create_structured_agent()
        report = await agent.ainvoke([SystemMessage(...), HumanMessage(...)])

    실패 처리 계층:
        1. Circuit Breaker OPEN  → 즉시 fallback (서비스 완전 다운)
        2. Timeout               → fallback (응답 없음)
        3. LangChain with_retry 소진 → fallback (일시적 오류 3회 초과)
    """

    def __init__(
        self,
        chain: Any,
        breaker: pybreaker.CircuitBreaker,
        timeout_seconds: float = 60.0,
        task_name: str = "unknown_agent",
        model_name: str = "unknown",
        system_suffix: str = "",
        schema_feedback: bool = False,
    ):
        self._chain = chain
        self._breaker = breaker
        self._timeout_seconds = timeout_seconds
        self._task_name = task_name
        self._model_name = model_name
        self._system_suffix = system_suffix        # 첫 SystemMessage 끝에 붙일 공통 규칙
        self._schema_feedback = schema_feedback    # 스키마 위반 시 1회 피드백 재요청

    def _with_suffix(self, messages: Any) -> Any:
        if not self._system_suffix or not isinstance(messages, list) or not messages:
            return messages
        first = messages[0]
        if isinstance(first, SystemMessage) and self._system_suffix not in str(first.content):
            return [SystemMessage(content=str(first.content) + self._system_suffix), *messages[1:]]
        return messages

    async def _call_chain(self, messages: Any) -> Any:
        # 토큰 집계 콜백 — with_structured_output(include_raw=False)라
        # 반환값에 usage가 없어 콜백으로 가로챈다 (src/utils/llm_budget.py).
        cb = make_budget_callback(self._model_name)
        config = {"callbacks": [cb]} if cb is not None else None
        return await with_timeout(
            self._chain.ainvoke(messages, config=config) if config
            else self._chain.ainvoke(messages),
            self._timeout_seconds,
            self._task_name,
        )

    async def _invoke_with_schema_feedback(self, messages: Any) -> Any:
        """1차 호출 → 스키마 위반이면 위반 내용을 피드백으로 넣어 **1회만** 재요청.

        같은 프롬프트 3회 반복(옛 with_retry)은 같은 답을 3번 받는 것이었다.
        재요청도 실패하면 예외를 그대로 올려 폴백(abstain)으로 간다. 서킷 브레이커는
        이 함수 바깥에서 최종 결과만 보므로 재요청이 실패 횟수를 두 배로 늘리지 않는다.
        """
        global SCHEMA_RETRY_COUNT, SCHEMA_ABSTAIN_COUNT
        try:
            return await self._call_chain(messages)
        except (ValidationError, OutputParserException) as e:
            if not self._schema_feedback:
                raise
            detail = str(e).splitlines()
            summary = " / ".join(d.strip() for d in detail[:4])[:600]
            SCHEMA_RETRY_COUNT += 1
            logger.warning(f"[SchemaRetry] {self._task_name} — 스키마 위반, 피드백 재요청 1회: {summary[:200]}")
            feedback = HumanMessage(content=(
                "직전 응답이 출력 스키마를 위반해 거부됐다:\n" + summary +
                "\n\n위 항목을 고쳐 다시 답하라. 'Field required'로 표시된 필수 필드(예: risk_factors, "
                "최소 1개)는 반드시 채워라. 근거·출처를 각각 2개 이상 댈 수 없으면 data_sufficient=false로 "
                "두고 그 두 목록만 비워라. 없는 수치를 지어내지 마라."
            ))
            try:
                return await self._call_chain(list(messages) + [feedback])
            except (ValidationError, OutputParserException) as e2:
                SCHEMA_ABSTAIN_COUNT += 1
                logger.error(f"[SchemaAbstain] {self._task_name} — 재요청도 스키마 위반, 폴백: {str(e2)[:200]}")
                raise

    async def ainvoke(self, messages: Any) -> Any:
        messages = self._with_suffix(messages)

        async def _invoke():
            return await self._invoke_with_schema_feedback(messages)

        try:
            return await self._breaker.call_async(_invoke)

        except pybreaker.CircuitBreakerError:
            state_info = get_breaker_status().get(self._breaker.name, {})
            logger.warning(
                f"[CB:OPEN] {self._task_name} — "
                f"서비스 차단 중 (fail_count={state_info.get('fail_count', '?')})"
            )
            return self._make_fallback(reason="circuit_breaker_open")

        except asyncio.TimeoutError:
            logger.error(
                f"[Timeout] {self._task_name} — {self._timeout_seconds}초 초과"
            )
            return self._make_fallback(reason="timeout")

        except Exception as e:
            logger.error(f"[Error] {self._task_name}: {type(e).__name__}: {e}")
            return self._make_fallback(reason=f"{type(e).__name__}")

    def _make_fallback(self, reason: str) -> AnalysisReport:
        """
        모든 시도 실패 시 반환할 기본 AnalysisReport.
        confidence=0.0 → Quality Gate(threshold 0.6)에서 자동 필터링됨.
        """
        return AnalysisReport(
            agent_name=self._task_name,
            confidence=0.0,
            recommendation="HOLD",
            data_sufficient=False,
            reasoning=[
                f"[폴백:{reason}] {self._task_name} 분석 실패",
                "Circuit Breaker, Timeout, 또는 예외로 인해 에이전트를 실행할 수 없음",
                "이 보고서는 신뢰할 수 없으므로 Quality Gate에서 자동 제외됨",
            ],
            data_sources=["error_fallback", "fallback"],
            selection_rationale=f"[폴백:{reason}] 에이전트 실패로 종목 선정 불가",
            prediction_basis=["오류로 인한 대체값", "fallback"],
            risk_factors=["에이전트 오류로 분석 불가"],
        )


# ──────────────────────────────────────────────────────────
# 팩토리 함수
# ──────────────────────────────────────────────────────────

def create_structured_agent(
    model: str = "gpt-4o-mini",
    timeout_seconds: float = 60.0,
) -> ResilientChain:
    """
    OpenAI 모델 기반 구조화 출력 에이전트.
    사용처: 7개 전문가 에이전트 (macro, kr_market, us_market, quant, technical, fundamental, sentiment)

    호출 방식 (기존과 동일):
        agent = create_structured_agent()
        report = await agent.ainvoke([SystemMessage(...), HumanMessage(...)])

    [why method="function_calling"]
      langchain-openai 기본값 "json_schema"는 strict 모드라 Optional 필드가 많은
      AnalysisReport(거래 파라미터 9개 포함) 스키마에서 큰 프롬프트(예: kr_market,
      fundamental의 10종목 × 3 MCP 도구)일 때 LLM이 텍스트만 반환하고 구조화 출력을
      누락 → "outputSchema defined but no structured output returned" 오류 발생.
      function_calling(tool-calling API)은 일반 도구 호출 메커니즘을 사용해 더 안정적.
    """
    llm = ChatOpenAI(model=model, temperature=0)
    # with_retry는 일시 오류(연결·429·5xx)만. 스키마 위반은 ResilientChain이 1회 피드백 재요청.
    chain_with_retry = llm.with_structured_output(
        AnalysisReport,
        method="function_calling",
        include_raw=False,
    ).with_retry(
        retry_if_exception_type=TRANSIENT_ERRORS,
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    return ResilientChain(
        chain=chain_with_retry,
        breaker=openai_breaker,
        timeout_seconds=timeout_seconds,
        task_name=f"structured_agent({model})",
        model_name=model,
        system_suffix=ABSTAIN_RULE,
        schema_feedback=True,
    )


def create_anthropic_agent(
    model: str = "claude-sonnet-4-6",
    timeout_seconds: float = 90.0,
) -> ResilientChain:
    """
    Anthropic 모델 기반 구조화 출력 에이전트.
    사용처: chief_strategist (claude-opus-4-6)

    [DEP-01 수정] langchain_anthropic.ChatAnthropic → anthropic.AsyncAnthropic 직접 사용

    호출 방식 (기존과 동일):
        agent = create_anthropic_agent(model="claude-opus-4-6")
        report = await agent.ainvoke([SystemMessage(...), HumanMessage(...)])

    왜 timeout=90초인가:
        Opus는 긴 추론을 하므로 60초로는 부족할 수 있음
    """
    chain = _AnthropicStructuredChain(model=model)
    return ResilientChain(
        chain=chain,
        breaker=anthropic_breaker,
        timeout_seconds=timeout_seconds,
        task_name=f"anthropic_agent({model})",
        model_name=model,
    )


def create_anthropic_text_agent(
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 500,
) -> ResilientChain:
    """
    Anthropic 모델 기반 자유 텍스트 생성 에이전트.
    사용처: debate.py의 Bull/Bear 논거 생성

    [DEP-01 수정] langchain_anthropic.ChatAnthropic → anthropic.AsyncAnthropic 직접 사용

    호출 방식 (기존과 동일):
        agent = create_anthropic_text_agent()
        text = await agent.ainvoke([SystemMessage(...), HumanMessage(...)])

    왜 temperature=0.3인가:
        토론 논거는 설득력 있는 언어가 중요 → 약간의 창의성 허용
        0.0이면 Bull/Bear 모두 동일한 패턴 문장 생성 가능
    """
    chain = _AnthropicTextChain(model=model, max_tokens=max_tokens)
    return ResilientChain(
        chain=chain,
        breaker=anthropic_breaker,
        timeout_seconds=60.0,
        task_name=f"anthropic_text_agent({model})",
        model_name=model,
    )


# ──────────────────────────────────────────────────────────
# 내부 헬퍼: Anthropic SDK 직접 사용 체인
# LangChain 메시지 리스트를 받아 Anthropic API 형식으로 변환
# ──────────────────────────────────────────────────────────

def _convert_messages(messages: list) -> tuple[str, list[dict]]:
    """
    LangChain 메시지 리스트 → Anthropic API 형식 변환.

    quant_analyst.py 호출 예시:
        [SystemMessage(content="..."), HumanMessage(content="...")]
    변환 결과:
        system = "..."
        messages = [{"role": "user", "content": "..."}]

    반환:
        (system_prompt, anthropic_messages)
    """
    system_prompt = ""
    anthropic_messages = []

    for msg in messages:
        # LangChain 메시지 객체 or 딕셔너리 모두 처리
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "type", "human")
            content = getattr(msg, "content", "")

        if role in ("system",):
            system_prompt = content
        elif role in ("human", "user"):
            anthropic_messages.append({"role": "user", "content": content})
        elif role in ("ai", "assistant"):
            anthropic_messages.append({"role": "assistant", "content": content})

    return system_prompt, anthropic_messages


class _AnthropicStructuredChain:
    """
    Anthropic SDK로 AnalysisReport를 직접 생성하는 내부 체인.
    LangChain 메시지 리스트를 받아 Anthropic API 형식으로 변환 후 호출.
    """

    def __init__(self, model: str):
        self._client = AsyncAnthropic()
        self._model = model

    async def ainvoke(self, messages: list) -> AnalysisReport:
        system_prompt, anthropic_messages = _convert_messages(messages)

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=anthropic_messages,
        )

        content = response.content[0].text
        return self._parse_to_report(content)

    def _parse_to_report(self, text: str) -> AnalysisReport:
        """
        텍스트 응답을 AnalysisReport 스키마로 변환.
        파싱 실패 시 HOLD/confidence=0.0 반환 → Quality Gate 자동 필터링.
        """
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AnalysisReport(**data)
        except Exception as e:
            logger.warning(f"[파싱 실패] AnalysisReport 변환 오류: {e}")

        return AnalysisReport(
            agent_name=self._model,
            confidence=0.0,
            recommendation="HOLD",
            reasoning=[
                "응답 파싱 실패",
                "JSON 형식으로 변환할 수 없는 응답이 반환됨",
                "이 보고서는 신뢰할 수 없으므로 Quality Gate에서 자동 제외됨",
            ],
            data_sources=["parse_error", "fallback"],
            prediction_basis=["파싱 오류로 인한 대체값", "fallback"],
            risk_factors=["응답 파싱 오류"],
        )


class _AnthropicTextChain:
    """
    Anthropic SDK로 자유 텍스트를 생성하는 내부 체인.
    debate.py의 Bull/Bear 논거 생성용.
    """

    def __init__(self, model: str, max_tokens: int):
        self._client = AsyncAnthropic()
        self._model = model
        self._max_tokens = max_tokens

    async def ainvoke(self, messages: list) -> str:
        system_prompt, anthropic_messages = _convert_messages(messages)

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.3,
            system=system_prompt,
            messages=anthropic_messages,
        )
        return response.content[0].text