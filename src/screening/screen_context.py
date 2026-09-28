"""
src/screening/screen_context.py

"오늘 이 종목이 왜 올라왔나"를 스크리닝 산출물에서 뽑아 분석 단계로 넘긴다.
감사: doc/2026-09-21_agent-audit.md §2-2

[왜 이 모듈이 필요한가]
  스크리닝 v2는 뉴스 급증·공시로 종목을 고르는데, run_pipeline에는 ticker 문자열
  하나만 넘어갔다. 그래서 8명 중 누구도 "무슨 일이 났는지"를 읽지 못했고,
  리포트를 읽는 사람도 선정 사유를 알 수 없었다.

  공개 가격·거래량 지표에는 우위가 없다는 것이 이미 확인됐다
  (doc/2026-09-20_screening-audit.md §3-1, doc/2026-09-15_verified-returns.md).
  남은 가설은 "이벤트를 일관되게 해석한다"뿐이고, 이 모듈이 그 입력을 만든다.

[경계]
  여기서는 **스크리닝이 이미 계산한 값만** 옮긴다. 외부 호출도, 새 판단도 없다.
  종목 헤드라인 본문은 에이전트가 직접 읽는다 (mention_db.get_recent_mentions).
"""

from __future__ import annotations

from typing import Optional

# 후보 판정선. stage2_scorer.NEWS_BURST_PCT_MIN과 같은 값을 쓰되, 이 모듈이
# 스코어러에 의존하지 않도록 import 실패 시 같은 기본값으로 떨어진다.
try:
    from src.screening.stage2_scorer import NEWS_BURST_PCT_MIN
except Exception:  # pragma: no cover - 스코어러 import 실패는 운영에서 일어나지 않는다
    NEWS_BURST_PCT_MIN = 0.8

MAX_DART_TITLES_IN_LINE = 2


def build_screen_context(ticker: str, screen: Optional[dict]) -> dict:
    """스크리닝 결과에서 이 종목 몫만 추린다. 없으면 빈 값으로 채운다.

    screen: screener.run_screening()의 반환 dict (None 허용 — 단독 디버그 실행).
    """
    screen = screen or {}
    detail = (screen.get("scores") or {}).get(ticker) or {}
    news   = ((screen.get("stage_results") or {}).get("news") or {}).get(ticker) or {}

    return {
        "ticker":         ticker,
        "rank":           detail.get("rank"),
        "score":          detail.get("score"),
        "news_burst":     detail.get("news_burst"),
        "news_burst_pct": detail.get("news_burst_pct"),
        "dart_event":     int(detail.get("dart_event") or 0),
        "dart_titles":    [t.strip() for t in (detail.get("dart_titles") or [])],
        "news_match":     int(detail.get("news_match") or 0),
        "news_reason":    news.get("reason") or "",
        "sent_delta":     detail.get("sent_delta"),
        "ret5":           detail.get("ret5"),
    }


def selection_reason(ctx: Optional[dict]) -> str:
    """사람이 읽는 한 줄. 리포트와 프롬프트가 같은 문장을 쓴다."""
    ctx = ctx or {}
    parts: list[str] = []

    titles = ctx.get("dart_titles") or []
    if ctx.get("dart_event") and titles:
        head = " / ".join(titles[:MAX_DART_TITLES_IN_LINE])
        more = len(titles) - MAX_DART_TITLES_IN_LINE
        parts.append(f"공시 {head}" + (f" 외 {more}건" if more > 0 else ""))
    elif ctx.get("dart_event"):
        parts.append("공시 발생")

    burst = ctx.get("news_burst")
    pct   = ctx.get("news_burst_pct")
    if burst is not None:
        # 백분위는 "상위 몇 %"로 뒤집어 쓴다. 0.93 → 상위 7%.
        rank_note = f", 유니버스 상위 {round((1 - float(pct)) * 100)}%" if pct is not None else ""
        parts.append(f"뉴스 {float(burst):.1f}배 급증{rank_note}")

    if not parts:
        return "선정 사유 없음 (스크리닝 정보 없음 — 단독 실행이거나 사유가 기록되지 않음)"

    line = " · ".join(parts)
    reason = (ctx.get("news_reason") or "").strip()
    if reason:
        line += f" · 헤드라인 해석: {reason}"
    return line


def is_candidate_by_event(ctx: Optional[dict]) -> bool:
    """이벤트(공시 또는 뉴스 급증)로 뽑힌 종목인가. 워밍업 중에는 공시만 참이 된다."""
    ctx = ctx or {}
    pct = ctx.get("news_burst_pct")
    return bool(ctx.get("dart_event")) or (pct is not None and float(pct) >= NEWS_BURST_PCT_MIN)
