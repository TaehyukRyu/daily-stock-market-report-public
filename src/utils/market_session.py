"""
src/utils/market_session.py

KST 장중 여부 판정 — 리포트가 "즉시 매수"를 쓰면 안 되는 시각인지.

[왜 필요한가]
  cron 지연(실측 102~491분)으로 장중에 파이프라인이 돌면 get_stock_price가
  확정 종가가 아니라 체결 중인 현재가를 준다 (daily_runner._market_session_warning).
  daily_stock_analysis의 phase_decision_guardrail은 같은 상황에서 "立即买入/buy now"
  문구를 사후 차단한다. 우리는 chief 프롬프트에 사전 주의를 넣는다 (G5).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

KST = timezone(timedelta(hours=9))

OPEN_MIN  = 9 * 60          # 09:00
CLOSE_MIN = 15 * 60 + 30    # 15:30


def is_market_hours_kst(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(KST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=KST)
    now = now.astimezone(KST)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return OPEN_MIN <= minutes <= CLOSE_MIN


def market_session_note(now: Optional[datetime] = None) -> str:
    """장중이면 chief 프롬프트에 붙일 주의 문구, 아니면 빈 문자열."""
    if not is_market_hours_kst(now):
        return ""
    return (
        "\n[장중 실행 주의] 지금은 장중이라 '현재가'는 확정 종가가 아니라 체결 중인 가격입니다. "
        "entry_strategy에 '시장가'를 쓰지 말고 '지정가대기' 또는 '분할매수'로 두고, "
        "reasoning·selection_rationale에 '즉시 매수'·'지금 매수' 같은 표현을 쓰지 마십시오.\n"
    )
