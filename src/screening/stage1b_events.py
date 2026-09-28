"""
src/screening/stage1b_events.py

Stage 1-B (v2): 정보 이벤트 팩터의 원값.
스펙: docs/superpowers/specs/2026-09-21-screening-v2-design.md §3

  news_burst  최근 창 일평균 기사 수 ÷ baseline 일평균 (발행일 기준, mention_db)
  dart_event  창 안 주요 공시 유무 0/1 (dart_events)
  sent_delta  최근 3일 net_sentiment 평균 − 직전 3일 평균 (기록 전용)

창 규약 (dart_events.last_weekday_before와 같은 자):
  최근 창    = 직전 평일 ~ 오늘  (월요일: 금·토·일·월 4일, 평일: 어제·오늘 2일)
  baseline   = 최근 창 시작 하루 전에서 14일

백분위·후보·필터는 stage2_scorer가 한다. 여기는 종목별 숫자만 만든다.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from src.data.mention_db import (
    get_daily_counts,
    get_net_sentiment_series,
    init_db as _init_mention_db,
)
from src.data.dart_events import collect_dart_events, last_weekday_before
from src.evaluation.outcomes import method_applied_date, SCREEN_V2_KEY

logger = logging.getLogger(__name__)

BASELINE_DAYS          = 14
BASELINE_FLOOR_PER_DAY = 1 / BASELINE_DAYS   # baseline 0건이면 "14일에 1건"으로 본다 (0 나눗셈·무한대 방지)
WARMUP_DAYS            = 16                  # 적용일 + 14일 baseline + 2일 최근 창


def recent_window(today: date) -> list[str]:
    start = last_weekday_before(today)
    return [(start + timedelta(days=i)).isoformat() for i in range((today - start).days + 1)]


def baseline_window(today: date) -> tuple[str, str]:
    end   = last_weekday_before(today) - timedelta(days=1)
    start = end - timedelta(days=BASELINE_DAYS - 1)
    return start.isoformat(), end.isoformat()


def compute_news_burst(counts: dict[str, int], recent: list[str],
                       base_start: str, base_end: str) -> tuple[float, int]:
    recent_n = sum(counts.get(d, 0) for d in recent)
    base_n   = sum(v for d, v in counts.items() if base_start <= d <= base_end)
    recent_per_day = recent_n / len(recent)
    base_per_day   = max(base_n / BASELINE_DAYS, BASELINE_FLOOR_PER_DAY)
    return round(recent_per_day / base_per_day, 4), recent_n


def sentiment_delta(series: list[float]) -> float | None:
    if len(series) < 6:
        return None
    return round(sum(series[-3:]) / 3 - sum(series[-6:-3]) / 3, 4)


def news_burst_ready(today: date) -> bool:
    """발행일 기준 데이터가 baseline만큼 쌓였는가 — v2 적용일(eval_method_log)로 판단."""
    applied = method_applied_date(SCREEN_V2_KEY)
    if not applied:
        return False
    return (today - date.fromisoformat(applied)).days >= WARMUP_DAYS


def _empty() -> dict:
    return {"news_burst": None, "news_recent": 0, "dart_event": 0, "dart_titles": [], "sent_delta": None}


def run_event_screen(tickers: list[str], today: date | None = None,
                     news_ready: bool = True) -> dict[str, dict]:
    if not tickers:
        return {}
    today = today or date.today()
    try:
        _init_mention_db()
    except Exception as e:
        logger.warning(f"[stage1b] mention_db init 실패 (계속): {e}")

    recent = recent_window(today)
    base_start, base_end = baseline_window(today)
    dart = collect_dart_events(list(tickers), today)

    out: dict[str, dict] = {}
    for t in tickers:
        row = _empty()
        try:
            if news_ready:
                counts = get_daily_counts(t, base_start, recent[-1])
                row["news_burst"], row["news_recent"] = compute_news_burst(counts, recent, base_start, base_end)
            events = dart.get(t) or []
            row["dart_event"]  = 1 if events else 0
            row["dart_titles"] = [e["title"] for e in events]
            row["sent_delta"]  = sentiment_delta(get_net_sentiment_series(t, 6))
        except Exception as e:
            logger.debug(f"[stage1b] {t} 팩터 계산 실패: {e}")
        out[t] = row
    return out
