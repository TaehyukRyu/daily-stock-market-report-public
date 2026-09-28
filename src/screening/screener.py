"""
src/screening/screener.py

스크리닝 통합 진입점.

  유니버스 110개
    → [Stage 1-A] ohlcv_cache 갱신 + 5일 수익률 (필터용)
    → [Stage 1-B] 정보 이벤트: 발행일 기준 뉴스 급증 + DART 주요 공시 (stage1b_events)
    → [Stage 1-C] 매크로 뉴스 LLM 스크린 (태그 전용)
    → [Stage 2]   백분위 → 후보 → 필터 → 레짐 상한
    → run_screening() 반환

각 stage는 독립적으로 실패해도 전체가 멈추지 않도록 try/except 격리.
Stage 1-C 실패하면 1-A + 1-B 결과만으로 Stage 2 진행.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

# pykrx의 webio.py가 module load 시점에 build_krx_session()을 호출하며
# os.getenv("KRX_ID"/"KRX_PW")의 기본값을 그때 평가한다. .env가 그 이후에
# 로드되면 "KRX_ID 또는 KRX_PW 환경 변수가 설정되지 않았습니다." 경고가
# 출력되므로, pykrx를 import하는 모듈보다 먼저 dotenv를 로드한다.
from dotenv import load_dotenv
load_dotenv()

from src.universe.universe_builder import load_universe

# 모든 SQLite 스키마는 screener 진입 시 한 번에 보장.
# CI 환경에서 mentions.db가 빈 상태로 시작해도 stage1b_events 등이 깨지지 않도록.
from src.data.mention_db   import init_db          as _init_mention_db
from src.data.ohlcv_cache  import init_ohlcv_cache as _init_ohlcv_cache

from src.screening.stage1a_quant   import run_quant_screen
from src.screening.stage1b_events  import run_event_screen, news_burst_ready
from src.screening.stage1c_news    import run_news_screen
from src.screening.stage2_scorer   import run_scoring
from src.screening import stage1c_news as _s1c

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


def _stage_log(stage: str, n: int) -> float:
    """stage 시작 로그 + 타이머 시작."""
    print(f"[Screener] {stage} 시작... ({n}개 종목)")
    return time.monotonic()


def _stage_done(stage: str, started_at: float, signaled: int) -> None:
    elapsed = time.monotonic() - started_at
    print(f"[Screener] {stage} 완료: {signaled}개 신호 발생 ({elapsed:.1f}초)")


def _count_quant_signaled(quant: dict) -> int:
    return sum(1 for m in quant.values() if m.get("signal_count", 0) >= 1)


def _count_event_signaled(event: dict) -> int:
    return sum(1 for m in event.values() if m.get("dart_event") or (m.get("news_burst") or 0.0) >= 2.0)


def _count_news_signaled(news: dict) -> int:
    return sum(1 for n in news.values() if n.get("news_match"))


def run_screening(tickers: Optional[list[str]] = None, regime: Optional[str] = None) -> dict:
    """
    유니버스에 4단계 스크리닝을 적용하고 결과 dict 반환.

    Args:
        tickers: 명시하면 그 ticker만 사용. None이면 load_universe().
        regime:  레짐 라벨(bull/bear/…). Stage 2의 후보 상한(bear·volatile 5, 그 외 8)을 고른다.

    Returns (v2):
        {
          "confirmed":      [ticker, ...],    # 후보 전부, 순위순 (상한 적용 전)
          "optional":       [],               # 항상 빈 리스트
          "scores":         {ticker: {score, news_burst, news_burst_pct, dart_event, news_match,
                                      sent_delta, ret5, is_candidate, rank, filtered_reason, dart_titles}},
          "summary":        "후보 N개 / 필터 제외 K개 / 상한 C (레짐 R[, 워밍업])",
          "headline_count": int | None,       # 건강도 판정용
          "cap":            int,
          "warmup":         bool,
          "stage_results":  {"news_burst": {...}, "dart": {...}, "news": {...}, "universe": [...]},
        }
    """
    universe = list(tickers) if tickers is not None else list(load_universe())
    n        = len(universe)
    if n == 0:
        print("[Screener] 유니버스가 비어 있음 — 스크리닝 스킵")
        return {
            "confirmed": [], "optional": [], "scores": {},
            "summary": "유니버스 비어 있음", "headline_count": None, "cap": 8, "warmup": True,
            "stage_results": {"news_burst": {}, "dart": {}, "news": {}, "universe": []},
        }

    print("=" * 60)
    print(f"[Screener] 유니버스 {n}개 스크리닝 시작")
    print("=" * 60)

    # ── DB 스키마 보장 (CI 첫 실행 시 빈 mentions.db 대비) ──
    try:
        _init_mention_db()
        _init_ohlcv_cache()
    except Exception as e:
        logger.warning(f"[Screener] DB init 일부 실패 (계속 진행): {e}")

    # ── Stage 1-A ────────────────────────────────────────────
    t0 = _stage_log("Stage 1-A (pykrx 정량)", n)
    try:
        quant_results = run_quant_screen(universe)
    except Exception as e:
        logger.error(f"[Screener] Stage 1-A 실패: {e}")
        quant_results = {}
    _stage_done("Stage 1-A", t0, _count_quant_signaled(quant_results))

    # ── Stage 1-B ────────────────────────────────────────────
    today = datetime.now(KST).date()
    t0 = _stage_log("Stage 1-B (뉴스 급증·공시)", n)
    try:
        ready = news_burst_ready(today)
        if not ready:
            print("  [stage1b] news_burst 워밍업 중 — 발행일 기준 데이터 16일 필요. 공시만으로 후보 선정")
        event_results = run_event_screen(universe, today=today, news_ready=ready)
    except Exception as e:
        logger.error(f"[Screener] Stage 1-B 실패: {e}")
        event_results = {}
    _stage_done("Stage 1-B", t0, _count_event_signaled(event_results))

    # ── Stage 1-C ────────────────────────────────────────────
    t0 = _stage_log("Stage 1-C (뉴스 LLM)", n)
    try:
        news_results = run_news_screen(universe)
    except Exception as e:
        logger.error(f"[Screener] Stage 1-C 실패: {e}")
        news_results = {t: {"news_match": False, "reason": ""} for t in universe}
    _stage_done("Stage 1-C", t0, _count_news_signaled(news_results))

    # ── Stage 2 ──────────────────────────────────────────────
    t0 = _stage_log("Stage 2 (스코어링)", n)
    try:
        scoring = run_scoring(universe, quant_results, event_results, news_results, regime=regime)
    except Exception as e:
        logger.error(f"[Screener] Stage 2 실패: {e}")
        scoring = {
            "confirmed": [], "optional": [], "scores": {},
            "summary":   "후보 0개 (스코어링 실패)", "cap": 8, "warmup": True, "regime": regime,
        }
    print(
        f"[Screener] Stage 2 완료: 후보 {len(scoring['confirmed'])}개 / {scoring['summary']} "
        f"({time.monotonic() - t0:.1f}초)"
    )
    if scoring["confirmed"]:
        print(f"  후보(순위순): {', '.join(scoring['confirmed'])}")

    return {
        **scoring,
        "headline_count": _s1c.LAST_HEADLINE_COUNT,   # 건강도 판정용 (0이면 뉴스 수집 실패)
        "stage_results": {
            "news_burst": event_results,
            "dart":       {t: r["dart_titles"] for t, r in event_results.items() if r.get("dart_titles")},
            "news":       news_results,
            "universe":   universe,
        },
    }


# ─────────────────────────────────────────────────────────────
# 단독 실행 (전체 파이프라인 통합 테스트)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_screening()

    print("\n" + "=" * 60)
    print("후보 상세")
    print("=" * 60)
    for t in result["confirmed"]:
        d = result["scores"][t]
        print(f"  {d['rank']:2d}. {t} : {d['score']:.2f}  burst={d['news_burst']}  공시={d['dart_titles'][:1]}  1-C={d['news_match']}")

    print(f"\n요약: {result['summary']}")
