"""
tests/test_prompt_order_bias.py

2026-09-20 TradingAgents 비교에서 가져온 두 가지 프롬프트 보강을 고정한다.
LLM 호출 없음 (문자열 검사만).

  1. 토론 순서 편향 — Bull이 2번 말하고 마지막 발언까지 갖는 구조(Bull→Bear→Bull)라
     심판·chief에게 "발언 순서·횟수와 무관하게 판단하라"를 명시한다.
  2. chief 역할 설명이 "7명"이었는데 실제 투표자는 quant_rule_agent 포함 8명이다.
"""

from src.agents.chief_strategist import CHIEF_SYSTEM_PROMPT
from src.agents.debate import HOLD_VERDICT_SYSTEM_PROMPT


def test_chief_prompt_lists_all_eight_voters():
    assert "8명" in CHIEF_SYSTEM_PROMPT
    assert "quant_rule_agent" in CHIEF_SYSTEM_PROMPT
    assert "7명의 전문가 애널리스트" not in CHIEF_SYSTEM_PROMPT   # 옛 문구 잔존 방지


def test_chief_prompt_ignores_speaking_order():
    assert "발언 순서" in CHIEF_SYSTEM_PROMPT


def test_hold_verdict_prompt_ignores_speaking_order():
    assert "발언 순서" in HOLD_VERDICT_SYSTEM_PROMPT
