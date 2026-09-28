"""
src/utils/daily_cache.py

종목 무관 에이전트(macro_economist, us_market_specialist)를 하루 1회만 돌린다 (G4).

[왜 필요한가]
  S1 5-1: 두 에이전트는 ticker 인자를 받지 않는다. 하루 8종목이면 같은 입력으로 8번 호출해
  (거의) 같은 답을 내고, 투표에서는 종목별 판단과 같은 1표로 센다. 정보량 0인 콜 14개 +
  상관 오류. 첫 호출 결과를 프로세스 안에서 재사용한다.

[설계]
  - 키 = 이름 + 날짜(KST). 실행은 하루 한 프로세스라 메모리 dict로 충분하다.
  - 반환은 복사본. pipeline이 r.ticker를 종목별로 덮어쓰므로 원본을 나누면 안 된다.
  - confidence 0.0(폴백) 결과는 캐시하지 않는다 — 일시 장애가 하루 종일 굳으면 안 된다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from src.schemas.agent_output import AnalysisReport

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

_CACHE: dict[str, AnalysisReport] = {}


def _key(name: str) -> str:
    return f"{name}:{datetime.now(KST).date().isoformat()}"


async def daily_cached(name: str, factory: Callable[[], Awaitable[AnalysisReport]]) -> AnalysisReport:
    key = _key(name)
    hit = _CACHE.get(key)
    if hit is not None:
        logger.info(f"[daily_cache] {name} — 오늘 결과 재사용 (LLM 호출 없음)")
        return hit.model_copy(deep=True)
    report = await factory()
    if report is not None and report.confidence > 0.0:
        _CACHE[key] = report.model_copy(deep=True)
    return report


def clear() -> None:
    _CACHE.clear()
