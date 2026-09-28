"""
tests/test_llm_budget.py

LLM 예산 집계·상한 + confirmed 종목 수 상한 회귀 테스트.

[배경]
  토큰 집계도, 누적 비용도, 상한도, 킬 스위치도 없었다.
  유일한 제동장치가 GitHub Actions의 timeout-minutes(120분)라서
  비용이 아니라 벽시계로만 막혀 있었다.

  2026-09-09 실행(#99) 실측: 확정 26종목 전부 심층 분석 → 36분,
  종목당 LLM 8~11콜이므로 하루 200~280콜.
  signal_reconciliation은 하루 5종목만 채택하므로 21종목은 버려졌다.
"""

import pytest

from src.utils import llm_budget as lb


@pytest.fixture(autouse=True)
def fresh_budget():
    lb.reset_budget(limit_usd=1.0)
    yield
    lb.reset_budget()


# ─────────────────────────────────────────────────────────
# 1. 집계
# ─────────────────────────────────────────────────────────

def test_records_tokens_and_cost():
    b = lb.get_budget()
    b.record("gpt-4o-mini", 1_000_000, 1_000_000)   # $0.15 + $0.60

    s = b.snapshot()
    assert s["total_calls"] == 1
    assert s["input_tokens"] == 1_000_000
    assert s["cost_usd"] == pytest.approx(0.75, abs=0.001)


def test_accumulates_per_model():
    b = lb.get_budget()
    b.record("gpt-4o-mini", 100_000, 100_000)
    b.record("gpt-4o-mini", 100_000, 100_000)
    b.record("claude-opus-4-6", 10_000, 10_000)

    s = b.snapshot()
    assert s["total_calls"] == 3
    assert s["by_model"]["gpt-4o-mini"]["calls"] == 2
    assert s["by_model"]["claude-opus-4-6"]["calls"] == 1


def test_unknown_model_uses_default_price():
    b = lb.get_budget()
    b.record("some-future-model", 1_000_000, 0)     # 기본 단가 $1.00
    assert b.snapshot()["cost_usd"] == pytest.approx(1.00, abs=0.001)


def test_opus_is_priced_higher_than_mini():
    """모델 배치 비용 감각이 숫자로 드러나야 한다."""
    b = lb.get_budget()
    b.record("gpt-4o-mini", 100_000, 100_000)
    mini = b.snapshot()["by_model"]["gpt-4o-mini"]["cost_usd"]

    lb.reset_budget(limit_usd=1000.0)
    b2 = lb.get_budget()
    b2.record("claude-opus-4-6", 100_000, 100_000)
    opus = b2.snapshot()["by_model"]["claude-opus-4-6"]["cost_usd"]

    # 공식 단가(2026-09-11) opus 5/25 vs mini 0.15/0.60 → 같은 토큰이면 40배.
    # 이전 표(15/75)로는 100배였다. 단가표 갱신에 따라 기준을 30배로 조정.
    assert opus > mini * 30, "opus 단가가 mini보다 훨씬 높아야 한다"


# ─────────────────────────────────────────────────────────
# 2. 상한 — record는 안 던지고 check가 던진다
# ─────────────────────────────────────────────────────────

def test_record_never_raises_even_over_limit():
    """
    핵심 설계 — 응답 직후에 던지면 ResilientChain의 except Exception이 잡아
    confidence=0.0 폴백으로 위장된다. 상한 초과가 '분석 실패'로 둔갑하면 안 된다.
    """
    b = lb.get_budget()          # limit 1.0
    b.record("gpt-4o-mini", 10_000_000, 10_000_000)   # $7.50 — 상한 초과
    assert b.over_limit() is True                      # 초과는 맞지만


def test_check_raises_when_over_limit():
    b = lb.get_budget()
    b.record("gpt-4o-mini", 10_000_000, 10_000_000)
    with pytest.raises(lb.BudgetExceeded, match="예산 초과"):
        b.check()


def test_check_silent_when_under_limit():
    b = lb.get_budget()
    b.record("gpt-4o-mini", 100_000, 100_000)          # $0.075
    b.check()                                          # 예외 없음
    assert b.over_limit() is False


def test_limit_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BUDGET_USD", "42.5")
    assert lb.reset_budget().limit_usd == 42.5


def test_summary_mentions_limit():
    b = lb.get_budget()
    b.record("gpt-4o-mini", 100_000, 100_000)
    out = b.format_summary()
    assert "gpt-4o-mini" in out and "상한" in out


def test_summary_when_no_calls():
    assert "호출 없음" in lb.get_budget().format_summary()


# ─────────────────────────────────────────────────────────
# 3. record_usage — SDK별 usage 형태 흡수
# ─────────────────────────────────────────────────────────

class _AnthropicUsage:
    input_tokens = 1000
    output_tokens = 500


class _OpenAIUsage:
    prompt_tokens = 1000
    completion_tokens = 500


def test_record_usage_handles_anthropic_shape():
    lb.record_usage("claude-opus-4-6", _AnthropicUsage())
    assert lb.get_budget().snapshot()["input_tokens"] == 1000


def test_record_usage_handles_openai_shape():
    lb.record_usage("gpt-4o-mini", _OpenAIUsage())
    assert lb.get_budget().snapshot()["input_tokens"] == 1000


def test_record_usage_tolerates_none_and_garbage():
    """집계 실패가 파이프라인을 멈추면 안 된다."""
    lb.record_usage("gpt-4o-mini", None)
    lb.record_usage("gpt-4o-mini", object())
    assert lb.get_budget().snapshot()["total_calls"] == 0


# ─────────────────────────────────────────────────────────
# 4. confirmed 종목 수 상한
# ─────────────────────────────────────────────────────────

def test_confirmed_is_capped():
    """26종목 전부 분석하던 것을 상한으로 자른다 (#99 재현)."""
    from src.daily_runner import _select_tickers, MAX_CONFIRMED_TICKERS

    confirmed = [f"{i:06d}" for i in range(26)]
    picked, mode = _select_tickers({"confirmed": confirmed, "optional": []})

    assert len(picked) == MAX_CONFIRMED_TICKERS
    assert picked == confirmed[:MAX_CONFIRMED_TICKERS], "상위 점수 순서가 유지돼야 한다"
    assert "상한 적용" in mode


def test_confirmed_under_cap_is_unchanged():
    from src.daily_runner import _select_tickers

    confirmed = ["005930", "000660"]
    picked, mode = _select_tickers({"confirmed": confirmed, "optional": []})
    assert picked == confirmed
    assert mode == "confirmed"


def test_cap_is_above_max_buy_signals():
    """상한이 채택 한도보다 작으면 뽑을 수 있는 신호를 놓친다."""
    from src.daily_runner import MAX_CONFIRMED_TICKERS
    from src.graph.signal_reconciliation import MAX_BUY_SIGNALS

    assert MAX_CONFIRMED_TICKERS >= MAX_BUY_SIGNALS
