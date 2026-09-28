"""
src/utils/atr.py

ATR(Average True Range) — 손절폭을 종목 변동성에 맞추기 위한 계산.
감사: doc/2026-09-21_agent-audit.md §2-5

[왜 필요한가]
  손절이 진입가 −4% 고정이었다. 그런데 스크리닝이 고르는 이벤트 종목의 ATR14
  중앙값이 7.3%다 — 손절폭이 하루 평균 변동폭의 절반이었다.
  ohlcv_cache 1,305 구간 실측(2026-09-21):

    고정 -4%         10거래일 내 손절 터치 84.9%
    2×ATR 상한 10%   62.1%
    2×ATR 상한 15%   46.4%   ← 채택
    2×ATR 무상한     31.6%  (손절폭 중앙값 17.9% — 사람이 감당하기 어렵다)

  000500이 그 사례다. D+1에 −7.09%로 손절됐고 D+5에는 +28.7%였다.
  손절이 추세가 아니라 노이즈에 걸렸다.

[한계 — 반드시 같이 읽을 것]
  손절 규칙을 어떻게 잡아도 같은 구간의 시드 기준 평균 손익은 음수였다.
  손절 없이 10거래일 보유만 해도 평균 −2.02%, 승률 39%다(표본 1,305).
  즉 이 변경은 "손절이 노이즈에 걸리는 문제"를 고치는 것이지
  "수익이 나게 만드는" 것이 아니다. 후자는 종목 선택의 몫이다.

[데이터 원천]
  ohlcv_cache (stage1a가 매일 유니버스 110종목을 채운다). 외부 호출 없음.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.ohlcv_cache import get_ohlcv_series

logger = logging.getLogger(__name__)

ATR_PERIOD    = 14
ATR_MULTIPLE  = 2.0    # 실측에서 터치율 46.4% (목표 50% 미만)
MIN_STOP_PCT  = -3.0   # 이보다 좁으면 호가 노이즈에 걸린다
MAX_STOP_PCT  = -15.0  # 이보다 넓으면 사람이 들고 있기 어렵다
SERIES_DAYS   = 60     # ATR14 + 여유


def atr(bars: list[dict], period: int = ATR_PERIOD) -> Optional[float]:
    """Wilder ATR. bars는 오래된순 [{'high','low','close'}, ...]. 봉이 모자라면 None.

    True Range = max(고가−저가, |고가−전일종가|, |저가−전일종가|)
    전일 종가를 쓰므로 갭(시가 급변)이 변동폭에 반영된다.
    """
    if not bars or len(bars) < period + 1:
        return None
    try:
        trs = []
        for i in range(1, len(bars)):
            high, low = float(bars[i]["high"]), float(bars[i]["low"])
            prev_close = float(bars[i - 1]["close"])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"[atr] 봉 데이터 이상 — ATR 계산 불가: {e}")
        return None

    if len(trs) < period:
        return None
    value = sum(trs[:period]) / period
    for tr in trs[period:]:          # Wilder 평활
        value = (value * (period - 1) + tr) / period
    return value


def atr_pct(bars: list[dict], period: int = ATR_PERIOD) -> Optional[float]:
    """ATR을 최근 종가 대비 %로. 종목 간 비교가 가능해진다."""
    value = atr(bars, period)
    if value is None:
        return None
    try:
        close = float(bars[-1]["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if close <= 0:
        return None
    return value / close * 100


def stop_pct_from_atr(atr_percent: Optional[float]) -> float:
    """ATR%(종가 대비) → 손절 비율(음수). None이면 종전 고정값으로 떨어진다."""
    from src.agents.chief_python import DEFAULT_STOP_PCT

    if atr_percent is None or atr_percent <= 0:
        return DEFAULT_STOP_PCT
    raw = -(ATR_MULTIPLE * float(atr_percent))
    return max(MAX_STOP_PCT, min(MIN_STOP_PCT, raw))


def stop_pct_for(ticker: str) -> float:
    """종목의 손절 비율(음수). ohlcv_cache를 읽는다. 실패하면 종전 고정값.

    손절은 리포트에 그대로 실리고 position_tracker가 감시하는 값이라
    예외를 올리지 않는다 — 계산이 안 되면 종전 동작으로 돌아간다.
    """
    from src.agents.chief_python import DEFAULT_STOP_PCT

    if not ticker:
        return DEFAULT_STOP_PCT
    try:
        bars = get_ohlcv_series(ticker, days=SERIES_DAYS)
    except Exception as e:
        logger.warning(f"[atr] {ticker} ohlcv_cache 조회 실패 — 고정 손절 사용: {e}")
        return DEFAULT_STOP_PCT

    percent = atr_pct(bars)
    if percent is None:
        logger.info(f"[atr] {ticker} 봉 부족({len(bars or [])}개) — 고정 손절 {DEFAULT_STOP_PCT}% 사용")
        return DEFAULT_STOP_PCT
    return stop_pct_from_atr(percent)
