"""
src/utils/date_window.py

point-in-time 규율 (G2) — "as-of 날짜 이후 정보는 보지 않는다"를 한 곳에 둔다.

[가져온 것] TradingAgents (Apache-2.0) tradingagents/dataflows/date_window.py 원문 규칙:
  - 모든 타임스탬프를 UTC로 정규화한다
  - 창은 반개구간 [start, end + 1일) — end 자정 정각에 찍힌 항목이 새지 않게
  - 날짜를 모르는 항목은 창이 "현재"에 닿을 때만(라이브 실행) 유지한다.
    백테스트에서는 미래가 아님을 증명할 수 없으므로 버린다 (#1126, #1220)
  - vintage(과거 시점 값)가 없는 소스는 과거 as-of 실행에 제공 자체를 거부하고
    이유를 문장으로 돌려준다 (#1300, withhold_live_profile)

[추가한 것 — Look-Ahead-Bench(arXiv 2601.13770) 함의]
  LLM은 사전학습 코퍼스로 과거 시세를 "본" 적이 있다. DB 레벨 PIT만으로는 부족하므로
  백테스트 시작일이 모델 지식 컷오프 이후인지 assert한다. 컷오프는 제공자 공식 모델 페이지
  (developers.openai.com/api/docs/models/*, 2026-09-11 확인).

[적용 범위]
  - mention_tracker.crawl_ticker(as_of=) 기사 pubDate 필터
  - ohlcv_cache.get_ohlcv_series(as_of=) 봉 필터
  - krx_market get_financials/get_consensus_estimates(as_of=) 현재값 제공 거부 (서버는
    서브프로세스라 이 모듈을 import하지 않고 같은 규칙을 인라인으로 둔다)
  - Stage 1-C 헤드라인은 리스트 페이지에 날짜가 없어 라이브 전용이다 (백테스트 불가 — 한계)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

KST = timezone(timedelta(hours=9))


def to_utc(dt: datetime) -> datetime:
    """naive면 UTC로 간주, aware면 UTC로 변환."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def in_window(pub_dt: Optional[datetime], start_dt: datetime, end_dt: datetime,
              now: Optional[datetime] = None) -> bool:
    """항목이 반개구간 [start, end + 1일) 안에 있는가.

    pub_dt가 None(날짜 미상)이면 창이 현재에 닿을 때만 True.
    """
    end = to_utc(end_dt)
    if pub_dt is not None:
        return to_utc(start_dt) <= to_utc(pub_dt) < end + timedelta(days=1)
    now = to_utc(now) if now else datetime.now(timezone.utc)
    return end >= now - timedelta(days=1)


def day_bounds(day: str | date, tz: timezone = KST) -> tuple[datetime, datetime]:
    """'YYYY-MM-DD' → (그날 00:00, 그날 00:00) — in_window가 end+1일을 더한다."""
    d = date.fromisoformat(day) if isinstance(day, str) else day
    start = datetime(d.year, d.month, d.day, tzinfo=tz)
    return start, start


def parse_pubdate(raw: str | None) -> Optional[datetime]:
    """NAVER 뉴스 pubDate(RFC 2822: 'Fri, 11 Sep 2026 00:37:00 +0900') 또는 ISO. 못 읽으면 None."""
    if not raw:
        return None
    text = str(raw).strip()
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_articles_as_of(articles: list[dict], as_of: str, date_key: str = "pubDate",
                          lookback_days: int = 0, now: Optional[datetime] = None) -> list[dict]:
    """as_of 날짜 창 [as_of - lookback, as_of] 안의 기사만. 날짜 미상은 라이브일 때만 유지."""
    start, end = day_bounds(as_of)
    start = start - timedelta(days=lookback_days)
    return [a for a in articles if in_window(parse_pubdate(a.get(date_key)), start, end, now=now)]


def withhold_live_value(as_of: Optional[str], label: str, today: Optional[str] = None) -> Optional[dict]:
    """과거 as_of에 '현재값만 있는' 소스를 제공하지 않는다. 제공해도 되면 None.

    TradingAgents withhold_live_profile과 같은 규칙. 반환 dict는 MCP 오류 관례({"error":…})를 따르되
    "withheld" 키로 구분해 호출자가 오류와 구별할 수 있게 한다.
    """
    if not as_of:
        return None
    today = today or date.today().isoformat()
    if as_of >= today:
        return None
    return {
        "withheld": True,
        "as_of": as_of,
        "error": (
            f"{label}: point-in-time 값이 없는 소스(현재 페이지 기준)라 {as_of} 시점 분석에 제공하지 않는다. "
            f"오늘({today}) 값을 과거 실행에 넣으면 사후 정보가 섞인다."
        ),
    }


# ── LLM 지식 컷오프 (백테스트 시작일 하한) ───────────────────────────────
# 출처: 제공자 공식 모델 페이지, 2026-09-11 확인.
MODEL_KNOWLEDGE_CUTOFF: dict[str, str] = {
    "gpt-4o-mini":               "2023-10-01",   # developers.openai.com/api/docs/models/gpt-4o-mini
    "gpt-5.6-luna":              "2026-02-16",   # developers.openai.com/api/docs/models/gpt-5.6-luna
    "gpt-5.6-terra":             "2026-02-16",
    "gpt-5.6-sol":               "2026-02-16",
    "gpt-6-astra":               "2026-04-30",   # developers.openai.com/api/docs/models (2026-09-11)
}


def knowledge_cutoff(model: str) -> Optional[str]:
    for prefix, cutoff in MODEL_KNOWLEDGE_CUTOFF.items():
        if model.startswith(prefix):
            return cutoff
    return None


def assert_after_cutoff(model: str, backtest_start: str) -> None:
    """백테스트 시작일이 모델 컷오프 이전이면 ValueError. 컷오프를 모르는 모델도 ValueError
    (모르면 통과시키는 것이 아니라 확인하고 표에 추가해야 한다)."""
    cutoff = knowledge_cutoff(model)
    if cutoff is None:
        raise ValueError(f"{model}: 지식 컷오프 미등록 — MODEL_KNOWLEDGE_CUTOFF에 공식 값을 추가하라")
    if backtest_start <= cutoff:
        raise ValueError(
            f"{model}: 백테스트 시작 {backtest_start} ≤ 지식 컷오프 {cutoff}. "
            f"코퍼스 암기(look-ahead)가 섞이므로 {cutoff} 이후만 평가하라 (Look-Ahead-Bench)."
        )
