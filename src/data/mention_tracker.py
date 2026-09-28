"""
src/data/mention_tracker.py

NAVER 뉴스 검색 API 기반 종목 언급량 크롤러

[변경사항 v2.1 — 복원력 추가]
  - _fetch_naver_news: Retry 3회 + 지수 백오프 추가
  - crawl_ticker: 크롤 실패 시 DB 캐시(전날 데이터) Fallback
  - naver_api_breaker: 연속 5회 실패 시 30초 Circuit Breaker
  - crawl_tickers: 개별 종목 실패가 전체 중단으로 이어지지 않도록 격리

[동작]
  1. 종목 코드 → 회사명 변환
  2. 네이버 뉴스 검색 API 호출 (회사명 쿼리)
  3. 헤드라인 배치 감성 분류
  4. mentions 테이블에 저장
  5. daily_mention_stats 집계 갱신

[API 제한]
  - NAVER 검색 API: 25,000건/일 (충분)
  - display 파라미터: 최대 100건/요청
  - 종목당 최대 50건 크롤링 (비용/속도 균형)
"""

import asyncio
import os
import re
import logging
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import aiohttp
import pybreaker
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.data.mention_db import (
    init_db,
    insert_mentions_batch,
    is_already_crawled,
    existing_titles,
    recount_daily_stats,
)
from src.data.sentiment_classifier import classify_headlines
from src.utils.date_window import filter_articles_as_of, parse_pubdate

KST = timezone(timedelta(hours=9))


def _now_utc() -> datetime:
    """테스트에서 '지금'을 고정하기 위한 이음새."""
    return datetime.now(timezone.utc)
from src.utils.resilience import with_timeout

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# 종목 코드 → 회사명 매핑
# ─────────────────────────────────────────────────

TICKER_TO_NAME: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "402340": "SK스퀘어",
    "005380": "현대차",
    "373220": "LG에너지솔루션",
    "034020": "두산에너빌리티",
    "329180": "HD현대중공업",
    "028260": "삼성물산",
    "009150": "삼성전기",
    "012450": "한화에어로스페이스",
    "000270": "기아",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "207940": "삼성바이오로직스",
    "068270": "셀트리온",
    "035420": "NAVER",
    "035720": "카카오",
    "105560": "KB금융",
    "055550": "신한지주",
    "086790": "하나금융지주",
    "005490": "POSCO홀딩스",
    "096770": "SK이노베이션",
    "017670": "SK텔레콤",
    "030200": "KT",
    "042660": "한화오션",
    "009540": "HD한국조선해양",
    "010140": "삼성중공업",
    "064350": "현대로템",
    "047810": "한국항공우주",
    "316140": "우리금융지주",
}

MAX_ARTICLES_PER_TICKER = 100   # API 최대. 발행일별 집계라 창이 넓을수록 baseline이 정확하다 (v2)
NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"


# ─────────────────────────────────────────────────
# ticker → 회사명 해석 (동적, lru_cache)
#
# 우선순위:
#   1. TICKER_TO_NAME 하드코딩 (즉시 hit)
#   2. pykrx.get_market_ticker_name (네트워크 1회, 이후 lru_cache hit)
#   3. ticker 코드 자체 (둘 다 실패 시)
#
# 유니버스 110종목 전수 mention 적재가 가능해진다.
# ─────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _resolve_ticker_name(ticker: str) -> str:
    cached = TICKER_TO_NAME.get(ticker)
    if cached:
        return cached
    try:
        from pykrx import stock as _pykrx_stock
        name = _pykrx_stock.get_market_ticker_name(ticker)
        if name:
            return name
    except Exception as e:
        logger.debug(f"[name] {ticker} pykrx 조회 실패: {e}")
    return ticker


# ─────────────────────────────────────────────────
# Circuit Breaker — NAVER API 전용
#
# 왜 openai_breaker와 분리하는가:
#   NAVER API가 다운돼도 LLM은 살아있을 수 있음.
#   서비스별로 독립 인스턴스를 써야 한쪽 장애가
#   다른 서비스 차단으로 이어지지 않음.
# ─────────────────────────────────────────────────
naver_api_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
    name="naver_api",
)


# ─────────────────────────────────────────────────
# NAVER 뉴스 검색 API 호출 (Retry + Circuit Breaker)
# ─────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _fetch_naver_news_inner(
    query: str,
    display: int,
    session: aiohttp.ClientSession,
) -> list[dict]:
    """
    실제 API 호출. with_retry로 감싸인 내부 함수.
    네트워크 오류, 타임아웃 시 자동 재시도.
    """
    headers = {
        "X-NCP-APIGW-API-KEY-ID": os.getenv("NAVER_CLIENT_ID", ""),
        "X-NCP-APIGW-API-KEY":    os.getenv("NAVER_CLIENT_SECRET", ""),
    }
    params = {
        "query":   query,
        "display": min(display, 100),
        "sort":    "date",
    }

    async with session.get(
        NAVER_NEWS_URL,
        headers=headers,
        params=params,
        timeout=aiohttp.ClientTimeout(total=10),  # 요청당 10초 제한
    ) as resp:
        if resp.status == 429:
            # Rate limit: 재시도해봤자 소용없음 → 즉시 포기
            logger.warning(f"[NAVER] Rate limit (429): {query}")
            return []
        if resp.status != 200:
            raise aiohttp.ClientError(f"HTTP {resp.status}: {query}")
        # API HUB는 JSON을 text/plain 으로 내려보내므로 mimetype 검사를 끈다
        data = await resp.json(content_type=None)
        return data.get("items", [])


async def _fetch_naver_news(
    query: str,
    display: int = 50,
    session: aiohttp.ClientSession = None,
) -> list[dict]:
    """
    Circuit Breaker + Retry가 적용된 NAVER 뉴스 검색.

    Circuit Breaker가 OPEN이면 즉시 [] 반환.
    네트워크 오류면 최대 3회 재시도 후 [] 반환.
    """
    should_close = session is None
    if should_close:
        session = aiohttp.ClientSession()

    try:
        async def _call():
            return await _fetch_naver_news_inner(query, display, session)

        return await naver_api_breaker.call_async(_call)

    except pybreaker.CircuitBreakerError:
        logger.warning(f"[CB:OPEN] naver_api — {query} 조회 스킵")
        return []

    except Exception as e:
        logger.error(f"[NAVER] {query} 최종 실패: {type(e).__name__}: {e}")
        return []

    finally:
        if should_close:
            await session.close()


def _clean_html_tags(text: str) -> str:
    """네이버 API 응답의 HTML 태그 제거 (<b>, </b> 등)."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _filter_relevant(fresh: list[tuple[str, str]], company_name: str) -> list[tuple[str, str]]:
    """제목에 종목명·약칭이 없는 기사는 저장하지 않는다 (판정서 §3-1). 이름을 모르면 거르지 않는다."""
    from src.data.headline_relevance import is_relevant
    if not company_name:
        return fresh
    return [(t, d) for t, d in fresh if is_relevant(t, company_name)]


# ─────────────────────────────────────────────────
# DB 캐시 Fallback 헬퍼
# ─────────────────────────────────────────────────

def _get_cached_stats(ticker: str) -> dict | None:
    """
    어제 또는 최근 7일 내 daily_mention_stats 데이터를 캐시로 반환.

    크롤링 실패 시 이전 데이터를 재사용해 파이프라인이 멈추지 않도록 함.
    sentiment_analyst는 캐시 데이터에 ⚠️ 표시를 보고 신뢰도를 낮출 수 있음.

    Returns:
        {"mention_count": 45, "positive": 20, ...} 또는 None
    """
    import sqlite3
    db_path = os.getenv("MENTIONS_DB_PATH", "data/mentions.db")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # 최근 7일 내 가장 최신 데이터 조회
        row = conn.execute("""
            SELECT mention_count, positive_count, neutral_count, negative_count, date
            FROM daily_mention_stats
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 1
        """, (ticker,)).fetchone()
        conn.close()

        if row:
            days_old = (date.today() - date.fromisoformat(row["date"])).days
            return {
                "mention_count":  row["mention_count"],
                "positive_count": row["positive_count"],
                "neutral_count":  row["neutral_count"],
                "negative_count": row["negative_count"],
                "cache_date":     row["date"],
                "days_old":       days_old,
            }
    except Exception as e:
        logger.debug(f"[캐시 조회 실패] {ticker}: {e}")

    return None


# ─────────────────────────────────────────────────
# 단일 종목 크롤링 (Fallback 포함)
# ─────────────────────────────────────────────────

async def crawl_ticker(
    ticker: str,
    today: str | None = None,
    session: aiohttp.ClientSession = None,
    as_of: str | None = None,
) -> int:
    """
    단일 종목의 오늘자 뉴스를 크롤링하고 DB에 저장.

    실패 시 동작:
      1. API 호출 실패 → Retry (최대 3회, _fetch_naver_news 내부 처리)
      2. 빈 결과 → 캐시 존재만 로그 (stats는 쓰지 않는다, v2)
      3. 캐시도 없음 → 0 반환 (파이프라인 계속 진행)

    Args:
        ticker: 종목 코드
        today:  날짜 (None이면 오늘)
        as_of:  주면 pubDate가 그날 창 밖인 기사를 버린다 (G2, 백테스트용).
                날짜 미상 기사는 창이 현재에 닿을 때만 남긴다. None이면 필터 없음(라이브 동작 유지).

    Returns:
        저장된 기사 수 (캐시 재사용 시 음수: -캐시_기사수)
    """
    target_date  = today or date.today().isoformat()
    company_name = _resolve_ticker_name(ticker)

    if is_already_crawled(ticker, target_date):
        print(f"  ⏭️ {ticker}({company_name}) — 이미 크롤링됨 ({target_date})")
        return 0

    # ── 크롤링 시도 ─────────────────────────────────
    articles = await _fetch_naver_news(
        query=company_name,
        display=MAX_ARTICLES_PER_TICKER,
        session=session,
    )

    if articles and as_of:
        before = len(articles)
        articles = filter_articles_as_of(articles, as_of, now=_now_utc())
        if len(articles) != before:
            logger.info(f"[date_window] {ticker} as_of={as_of}: {before}→{len(articles)}건 (창 밖 기사 제외)")

    # ── 크롤링 성공 경로 ─────────────────────────────
    # [v2] 기사는 발행일(pubDate, KST)로 저장한다. 최신 100건에는 며칠 전 기사가 섞여 있고
    # 다음날에도 같은 기사가 다시 오므로 제목으로 중복을 거른다. 분류는 새 기사에만 돈다.
    if articles:
        known = existing_titles(ticker)
        fresh: list[tuple[str, str]] = []          # (title, 발행일 ISO)
        seen: set[str] = set()
        for a in articles:
            title = _clean_html_tags(a.get("title", ""))
            if not title or title in known or title in seen:
                continue
            seen.add(title)
            pub = parse_pubdate(a.get("pubDate"))
            d_iso = pub.astimezone(KST).date().isoformat() if pub else target_date
            fresh.append((title, d_iso))
        fresh = _filter_relevant(fresh, company_name)

        if not fresh:
            print(f"  ⏭️ {ticker}({company_name}) — 새 기사 없음 ({len(articles)}건 — 이미 저장됐거나 종목과 무관)")
            return 0

        classifications = await classify_headlines([t for t, _ in fresh])

        rows = []
        for clf in classifications:
            idx = clf["index"]
            if idx < len(fresh):
                title, d_iso = fresh[idx]
                rows.append({
                    "ticker":          ticker,
                    "date":            d_iso,
                    "source":          "naver_news",
                    "title":           title,
                    "sentiment":       clf["sentiment"],
                    "sentiment_score": clf["score"],
                })

        if rows:
            insert_mentions_batch(rows)
            recount_daily_stats(ticker, sorted({r["date"] for r in rows}))

        positive_count = sum(1 for r in rows if r["sentiment"] == "positive")
        negative_count = sum(1 for r in rows if r["sentiment"] == "negative")
        print(
            f"  ✅ {ticker}({company_name}) — 새 기사 {len(rows)}건 저장 "
            f"(긍정:{positive_count} 부정:{negative_count}, 발행일 {len({r['date'] for r in rows})}일)"
        )
        return len(rows)

    # ── 크롤링 실패: 캐시가 있으면 알리기만 한다 ──────
    # [v2] 예전에는 옛 통계를 오늘 날짜로 다시 써 넣었다. 발행일 집계에서는 그 행이
    # "오늘 기사 N건"으로 읽혀 news_burst를 오염시키므로 쓰지 않는다.
    cached = _get_cached_stats(ticker)
    if cached:
        logger.warning(
            f"[Fallback] {ticker}({company_name}) — 크롤링 실패, "
            f"최신 통계는 {cached['days_old']}일 전 ({cached['cache_date']})"
        )
        print(f"  ⚠️ {ticker}({company_name}) — 크롤링 실패, {cached['days_old']}일 전 통계까지만 있음")
        return -cached["mention_count"]

    print(f"  ⚠️ {ticker}({company_name}) — 기사 없음 (캐시도 없음)")
    return 0


# ─────────────────────────────────────────────────
# 여러 종목 병렬 크롤링
# ─────────────────────────────────────────────────

async def crawl_tickers(
    tickers: list[str],
    today: str | None = None,
    max_concurrent: int = 3,
) -> dict:
    """
    여러 종목을 병렬로 크롤링.

    개별 종목 실패는 해당 종목만 0 처리하고 나머지 계속 진행.
    (return_exceptions 패턴 — 한 종목 오류가 전체 중단으로 이어지지 않음)

    Returns:
        {"005930": 45, "000660": -38, ...}
        양수 = 신규 크롤링, 음수 = 캐시 재사용, 0 = 데이터 없음
    """
    init_db()

    semaphore = asyncio.Semaphore(max_concurrent)
    results   = {}

    async with aiohttp.ClientSession() as session:

        async def crawl_with_semaphore(ticker: str):
            async with semaphore:
                try:
                    count = await crawl_ticker(ticker, today=today, session=session)
                    results[ticker] = count
                except Exception as e:
                    # 여기까지 오면 예상치 못한 오류 — 개별 격리
                    logger.error(f"[crawl_tickers] {ticker} 예외: {type(e).__name__}: {e}")
                    results[ticker] = 0

        await asyncio.gather(*[crawl_with_semaphore(t) for t in tickers])

    fresh   = sum(v for v in results.values() if v > 0)
    cached  = sum(1 for v in results.values() if v < 0)
    failed  = sum(1 for v in results.values() if v == 0)

    print(
        f"\n  → {len(tickers)}개 종목 완료: "
        f"신규 {fresh}건 | 캐시 {cached}개 종목 | 데이터 없음 {failed}개"
    )
    return results


# ─────────────────────────────────────────────────
# 단독 실행 (테스트용)
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_tickers = ["005930", "000660", "005380"]
    print(f"테스트 크롤링: {test_tickers}")
    result = asyncio.run(crawl_tickers(test_tickers))
    print(f"결과: {result}")