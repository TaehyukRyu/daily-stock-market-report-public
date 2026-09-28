"""
src/utils/llm_budget.py

LLM 호출 토큰·비용 집계와 상한.

[왜 필요한가]
  이 코드베이스에는 토큰 집계도, 누적 비용도, 상한도, 킬 스위치도 없었다
  (grep token|cost|usage|budget → RAG 프롬프트 예산과 max_tokens 파라미터뿐).
  유일한 제동장치가 GitHub Actions의 timeout-minutes(120분)라서
  비용이 아니라 벽시계로만 막혀 있었다.

  2026-09-09 실행 실측: 확정 26종목 × (에이전트 7 + 토론 + chief 1)
  = 하루 200~280콜. 스크리닝이 더 많이 뽑는 날이면 그만큼 늘어난다.

[설계]
  - 프로세스 1회 실행 = 예산 1개. 파일이나 DB에 남기지 않는다.
    (일 단위 누적 관리는 이 시스템의 실행 모델과 맞지 않는다 —
     Actions 러너는 매번 새 환경이고, 실행은 하루 한 번이다.)
  - 상한을 넘으면 예외를 던진다. 폴백으로 조용히 계속하면 상한이 아니다.
  - 가격표는 근사치다. 정확한 청구액이 아니라 "폭주 감지"가 목적이다.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """예산 상한 초과. 호출자가 잡아서 실행을 중단시킨다."""


# ── 가격표 (USD / 1M 토큰) ────────────────────────────────────────────────
# 근사치다. 목적은 청구액 예측이 아니라 "평소의 N배가 나가는 중"을 잡는 것.
# 모르는 모델은 _DEFAULT_PRICE로 계산하고 로그를 남긴다.
#
# 2026-09-11 공식 단가로 갱신. 이전 표는 opus 15/75, haiku 0.80/4.00이라
# opus 비용을 3배 부풀려 집계했다 (doc/2026-09-10_benchmark-analysis.md F-1).
#   OpenAI    https://developers.openai.com/api/docs/pricing
#   Anthropic https://platform.claude.com/docs/en/about-claude/pricing
# prefix 매칭이므로 긴 이름을 짧은 이름보다 앞에 둔다 (gpt-4o-mini → gpt-4o).
_PRICES: dict[str, tuple[float, float]] = {
    # model prefix              (input, output)
    "gpt-4o-mini":              (0.15,  0.60),
    "gpt-4o":                   (2.50, 10.00),
    "gpt-4.1-nano":             (0.10,  0.40),
    "gpt-5.6-luna":             (0.20,  1.20),
    "text-embedding-3-small":   (0.02,  0.00),
    "claude-opus-5":            (5.00, 25.00),
    "claude-opus-4":            (5.00, 25.00),   # 4.5~4.8 동일
    "claude-sonnet-5":          (2.00, 10.00),
    "claude-sonnet-4":          (3.00, 15.00),
    "claude-haiku-4":           (1.00,  5.00),
}
_DEFAULT_PRICE = (1.00, 5.00)

# 실행 1회 기본 상한(USD). 환경변수 LLM_BUDGET_USD로 덮어쓸 수 있다.
DEFAULT_BUDGET_USD = 5.0


def _price_for(model: str) -> tuple[float, float]:
    for prefix, price in _PRICES.items():
        if model.startswith(prefix):
            return price
    logger.warning(f"[budget] 가격표에 없는 모델 '{model}' — 기본 단가 적용")
    return _DEFAULT_PRICE


@dataclass
class _Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMBudget:
    """
    프로세스 1회 실행의 토큰·비용 누적기.

    스레드 안전 — 에이전트 7개가 동시에 돌므로 락이 필요하다.
    """

    def __init__(self, limit_usd: float | None = None):
        if limit_usd is None:
            limit_usd = float(os.getenv("LLM_BUDGET_USD", DEFAULT_BUDGET_USD))
        self.limit_usd = limit_usd
        self._lock = threading.Lock()
        self._by_model: dict[str, _Usage] = {}
        self._total = _Usage()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        호출 1건을 기록하고 누적 비용을 반환한다. **예외를 던지지 않는다.**

        [왜 여기서 안 던지는가]
          기록 지점은 LLM 응답 직후 — LangChain 콜백 안이거나 SDK 호출 직후다.
          거기서 예외를 던지면 ResilientChain의 except Exception이 잡아
          confidence=0.0 폴백으로 바뀐다. 상한에 걸린 것이 "분석 실패"로
          위장되어 조용히 흘러가는 것이다.
          → 던지는 것은 check()가 하고, 호출자가 안전한 지점에서 부른다.
        """
        pin, pout = _price_for(model)
        cost = (input_tokens / 1e6) * pin + (output_tokens / 1e6) * pout

        with self._lock:
            u = self._by_model.setdefault(model, _Usage())
            for target in (u, self._total):
                target.calls += 1
                target.input_tokens += input_tokens
                target.output_tokens += output_tokens
                target.cost_usd += cost
            return self._total.cost_usd

    def check(self) -> None:
        """
        상한 초과 시 예외. 안전한 경계(종목 하나를 끝낸 직후 등)에서 호출한다.

        Raises:
            BudgetExceeded
        """
        with self._lock:
            total, calls = self._total.cost_usd, self._total.calls
        if total > self.limit_usd:
            raise BudgetExceeded(
                f"LLM 예산 초과: ${total:.2f} > ${self.limit_usd:.2f} ({calls}콜). "
                f"무한 재시도나 종목 폭주를 의심할 것. "
                f"의도한 증가라면 LLM_BUDGET_USD 환경변수를 올려라."
            )

    def over_limit(self) -> bool:
        with self._lock:
            return self._total.cost_usd > self.limit_usd

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limit_usd":     self.limit_usd,
                "total_calls":   self._total.calls,
                "input_tokens":  self._total.input_tokens,
                "output_tokens": self._total.output_tokens,
                "cost_usd":      round(self._total.cost_usd, 4),
                "by_model": {
                    m: {"calls": u.calls, "cost_usd": round(u.cost_usd, 4)}
                    for m, u in sorted(self._by_model.items())
                },
            }

    def format_summary(self) -> str:
        s = self.snapshot()
        if s["total_calls"] == 0:
            return "[budget] LLM 호출 없음"
        lines = [
            f"[budget] LLM {s['total_calls']}콜 / "
            f"입력 {s['input_tokens']:,} + 출력 {s['output_tokens']:,} 토큰 / "
            f"약 ${s['cost_usd']:.3f} (상한 ${s['limit_usd']:.2f})"
        ]
        for m, d in s["by_model"].items():
            lines.append(f"           {m:32} {d['calls']:>3}콜  ${d['cost_usd']:.3f}")
        return "\n".join(lines)


# ── 프로세스 전역 인스턴스 ────────────────────────────────────────────────
# 파이프라인 곳곳(에이전트/토론/chief/스크리너)에서 같은 예산을 공유해야
# "이번 실행 전체"를 볼 수 있다.
_budget = LLMBudget()


def get_budget() -> LLMBudget:
    return _budget


def reset_budget(limit_usd: float | None = None) -> LLMBudget:
    """테스트/재실행용. 새 예산으로 교체한다."""
    global _budget
    _budget = LLMBudget(limit_usd)
    return _budget


def record_usage(model: str, usage) -> float:
    """
    SDK 응답의 usage 객체를 받아 기록한다.

    OpenAI:    usage.prompt_tokens / usage.completion_tokens
    Anthropic: usage.input_tokens  / usage.output_tokens
    둘 다 아니면 조용히 무시한다 (집계 실패가 파이프라인을 멈추면 안 된다).
    """
    if usage is None:
        return _budget.snapshot()["cost_usd"]

    inp = getattr(usage, "input_tokens", None)
    out = getattr(usage, "output_tokens", None)
    if inp is None:
        inp = getattr(usage, "prompt_tokens", None)
        out = getattr(usage, "completion_tokens", None)

    if inp is None or out is None:
        logger.debug(f"[budget] usage 형식을 모름: {type(usage).__name__}")
        return _budget.snapshot()["cost_usd"]

    return _budget.record(model, int(inp), int(out))


# ── LangChain 콜백 ────────────────────────────────────────────────────────
# 7개 에이전트는 with_structured_output(include_raw=False) 를 쓰기 때문에
# 반환값에 usage가 없다. include_raw=True로 바꾸면 반환 형태가 달라져
# 7개 에이전트 전부를 고쳐야 하므로, 콜백으로 가로챈다.

def make_budget_callback(model: str):
    """ResilientChain.ainvoke에 붙일 LangChain 콜백 핸들러. 실패해도 무해하다."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:          # langchain 미설치 환경(테스트 등)
        return None

    class _BudgetCallback(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs) -> None:
            try:
                # ① 신형: generations[0][0].message.usage_metadata
                for gen_list in getattr(response, "generations", []) or []:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        um = getattr(msg, "usage_metadata", None) if msg else None
                        if um:
                            _budget.record(model,
                                           int(um.get("input_tokens", 0)),
                                           int(um.get("output_tokens", 0)))
                            return
                # ② 구형: llm_output["token_usage"]
                tu = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
                if tu:
                    _budget.record(model,
                                   int(tu.get("prompt_tokens", 0)),
                                   int(tu.get("completion_tokens", 0)))
            except Exception as e:
                # 집계 실패가 분석을 멈추면 안 된다.
                logger.debug(f"[budget] 콜백 집계 실패: {type(e).__name__}: {e}")

    return _BudgetCallback()
