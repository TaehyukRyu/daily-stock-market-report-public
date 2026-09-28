"""
News & Economy MCP Server
담당: 뉴스, 환율, 한국 기준금리, 정책 동향
"""

# ── stdio 서버 가드 (2026-09-11) ─────────────────────────────────────────
# 이 파일이 서브프로세스로 실행될 때 stdout은 JSON-RPC 채널이다. pykrx는 **import 시점**에
# "KRX 로그인 시도..."를 print()로 stdout에 찍으므로(2차 CI 실행에서도 80건 오염), 어떤 import보다
# 먼저 print를 stderr로 돌린다. 모듈로 import될 때(테스트)는 손대지 않는다.
if __name__ == "__main__":
    import builtins as _b, functools as _ft, sys as _sys
    _b.print = _ft.partial(_b.print, file=_sys.stderr)

import json
from openai import OpenAI

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date
from urllib.parse import quote

import requests
import time
import random
from requests.exceptions import HTTPError, ConnectionError, Timeout

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# ── 환경변수 ──────────────────────────────────────────────────────
FRED_API_KEY      = os.getenv("FRED_API_KEY")
BOK_ECOS_API_KEY  = os.getenv("BOK_ECOS_API_KEY")
NAVER_CLIENT_ID   = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# ── [v5.4] 카테고리 → Google News RSS 쿼리 매핑 ─────────────────
CATEGORY_QUERIES = {
    "economy":       ("경제 금융 증시",             "google_ko"),
    "finance":       ("환율 금리 채권 통화정책",      "google_ko"),
    "politics":      ("경제정책 규제 법안 정부",      "google_ko"),
    "international": ("Fed FOMC tariff trade GDP",   "google_en"),
    "industry":      ("반도체 자동차 배터리 산업동향", "google_ko"),
}

# ── 정책 뉴스 수집용 멀티 쿼리 ────────────────────────────────────
POLICY_QUERIES = {
    "naver": [
        "경제정책", "금융정책", "산업정책", "규제",
        "기준금리", "환율", "무역",
    ],
    "google_en": [
        "Fed policy rate", "US trade tariff",
        "semiconductor export control", "economic sanctions",
        "IMF World Bank policy",
    ],
}


# 직전 nano 호출의 토큰 사용량. 이 서버는 stdio 서브프로세스라 파이프라인의
# LLMBudget(프로세스 전역)에 닿지 못한다. 결과 dict의 "llm_usage"로 실어 보내고
# 호출자(sentiment_analyst)가 기록한다.
_LAST_NANO_USAGE: dict | None = None


def _remember_nano_usage(response) -> None:
    global _LAST_NANO_USAGE
    u = getattr(response, "usage", None)
    if u is None:
        return
    _LAST_NANO_USAGE = {
        "model":         "gpt-4.1-nano",
        "input_tokens":  int(getattr(u, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
    }


def _score_relevance_batch(articles: list[dict], min_score: float) -> list[dict]:
    """
    GPT-4.1-nano로 기사 목록 전체를 한 번에 스코어링합니다.

    실패 모드(절대 raise 금지 — MCP 도구 전체 중단 방지):
      - OPENAI_API_KEY 누락           → 원본 articles 반환 (scored=False)
      - OpenAI 클라이언트 초기화 실패  → 동일
      - API 호출/JSON 파싱 실패        → 동일
    """
    if not articles:
        return articles

    # ── 환경변수 사전 점검 (CI에서 secret 누락 시 명확한 로그) ──
    if not os.getenv("OPENAI_API_KEY"):
        print("[스코어링 Fallback] OPENAI_API_KEY 미설정 — scored=False로 통과")
        for a in articles:
            a.setdefault("relevance_score", None)
            a["scored"] = False
        return articles

    try:
        client = OpenAI()                          # ← try 안으로 이동

        titles_text = "\n".join(
            f"{i+1}. {a['title']}" for i, a in enumerate(articles)
        )

        prompt = f"""다음 뉴스 기사 제목들 각각에 대해, 한국 주식시장 투자 판단에 영향을 줄 가능성을 0.0~1.0으로 평가하라.

점수 기준:
0.7~1.0: 직접적 영향 (금리/환율/기업실적/정책결정/무역/규제)
0.4~0.7: 간접적 영향 (산업동향/글로벌이슈/경기지표)
0.0~0.4: 무관 (사건사고/연예/스포츠/생활정보)

규칙: 반드시 숫자 배열 JSON만 반환하라. 설명 없음. 예시: [0.9, 0.2, 0.7]
기사 수: {len(articles)}개

{titles_text}"""

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        _remember_nano_usage(response)
        scores = json.loads(raw)

        if len(scores) != len(articles):
            raise ValueError(f"점수 수 불일치: 기사 {len(articles)}건 vs 점수 {len(scores)}개")

        scored_articles = []
        for article, score in zip(articles, scores):
            article["relevance_score"] = round(float(score), 2)
            article["scored"] = True
            if article["relevance_score"] >= min_score:
                scored_articles.append(article)

        return scored_articles

    except Exception as e:
        print(f"[스코어링 Fallback] {type(e).__name__}: {e}")
        # 폴백: 점수 없이 원본 반환. min_score 필터링은 건너뜀.
        for a in articles:
            a.setdefault("relevance_score", None)
            a["scored"] = False
        return articles


def _fetch_with_retry(url: str, headers: dict, max_retries: int = 3) -> requests.Response:
    """지수 백오프 + jitter로 HTTP GET을 재시도합니다."""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code in (429, 503):
                raise HTTPError(response=response)

            response.raise_for_status()
            return response

        except HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status not in (429, 503, 0):
                raise
            last_exception = e

        except (ConnectionError, Timeout) as e:
            last_exception = e

        if attempt == max_retries:
            break

        base_delay = 2 ** attempt
        jitter     = random.uniform(0, 1)
        wait_time  = base_delay + jitter
        print(f"[Retry] {attempt+1}/{max_retries} — {wait_time:.1f}초 대기")
        time.sleep(wait_time)

    raise last_exception


def _fetch_category_rss(category: str, max_items: int) -> list[dict]:
    """Google News RSS에서 카테고리별 기사를 수집합니다."""
    mapping = CATEGORY_QUERIES.get(category)
    if not mapping:
        return []

    query, source = mapping
    encoded_query = quote(query)

    if source == "google_ko":
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    else:
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = _fetch_with_retry(url, headers)
        response.encoding = "utf-8"

        root    = ET.fromstring(response.text)
        channel = root.find("channel")
        if channel is None:
            return []

        articles = []
        for item in channel.findall("item")[:max_items]:
            title       = re.sub(r"<[^>]+>", "", item.findtext("title",       "") or "").strip()
            link        = (item.findtext("link") or "").strip()
            description = re.sub(r"<[^>]+>", "", item.findtext("description", "") or "")[:300].strip()
            pub_date    = (item.findtext("pubDate") or "").strip()

            articles.append({
                "title":           title,
                "link":            link,
                "description":     description,
                "pub_date":        pub_date,
                "category":        category,
                "relevance_score": None,
                "scored":          False,
            })

        return articles

    except Exception as e:
        return [{"_fetch_error": str(e), "category": category}]


def _search_by_category(
    categories: list[str],
    max_per_category: int,
    min_score: float = 0.5,
) -> dict:
    """여러 카테고리에서 기사를 병렬 수집하고 URL 기준 중복 제거 후 nano 스코어링합니다."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    target = categories
    raw_results: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=len(target)) as executor:
        future_to_category = {
            executor.submit(_fetch_category_rss, category, max_per_category): category
            for category in target
        }
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                raw_results[category] = future.result()
            except Exception as e:
                raw_results[category] = [{"_fetch_error": str(e), "category": category}]

    all_articles = []
    seen_links   = set()
    errors       = []

    for category in target:
        for item in raw_results.get(category, []):
            if "_fetch_error" in item:
                errors.append({"category": category, "error": item["_fetch_error"]})
                continue

            link = item.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                all_articles.append(item)

    global _LAST_NANO_USAGE
    _LAST_NANO_USAGE = None
    scored_articles = _score_relevance_batch(all_articles, min_score=min_score)
    actually_scored = scored_articles[0]["scored"] if scored_articles else False

    return {
        "articles":  scored_articles,
        "count":     len(scored_articles),
        "mode":      "category",
        "scored":    actually_scored,
        "llm_usage": _LAST_NANO_USAGE,
        "phase":    "3 완료 — 카테고리 병렬 수집 + 중복 제거 + nano 관련성 스코어링",
        "errors":   errors,
    }


# ── FastMCP 인스턴스 ──────────────────────────────────────────────
mcp = FastMCP("news-economy-server")


# ─────────────────────────────────────────────────────────────────
# Tool 1: search_news
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def search_news(
    categories: list[str] | None = None,
    max_per_category: int = 30,
    min_relevance_score: float = 0.5,
    query: str | None = None,
    max_results: int = 20,
    source: str = "naver",
) -> dict:
    """
    뉴스를 수집합니다.

    [카테고리 모드 - v5.4] categories 파라미터 사용 시:
        Google News RSS에서 카테고리 기반으로 넓게 수집합니다.
        categories: economy / finance / politics / international / industry

    [키워드 모드 - 기존] query 파라미터 사용 시:
        query: 검색 키워드 (예: "삼성전자", "FOMC")
        source: naver / google_ko / google_en
    """
    if categories is not None:
        return _search_by_category(categories, max_per_category, min_score=min_relevance_score)

    if query is None:
        return {"error": "categories 또는 query 중 하나는 반드시 입력해야 합니다."}

    encoded_query = quote(query)

    if source == "naver":
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            return {
                "query": query, "source": source,
                "articles": [], "count": 0,
                "error": "NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET이 설정되지 않았습니다.",
            }

        try:
            resp = requests.get(
                "https://naverapihub.apigw.ntruss.com/search/v1/news",
                headers={
                    "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
                    "X-NCP-APIGW-API-KEY":    NAVER_CLIENT_SECRET,
                },
                params={"query": query, "display": max_results, "sort": "date"},
                timeout=10,
            )
            resp.raise_for_status()
            articles = [
                {
                    "title":       re.sub(r"<[^>]+>", "", item.get("title", "")).strip(),
                    "link":        item.get("originallink") or item.get("link", ""),
                    "description": re.sub(r"<[^>]+>", "", item.get("description", "")).strip(),
                    "pub_date":    item.get("pubDate", ""),
                }
                for item in resp.json().get("items", [])
            ]
            return {"query": query, "source": source, "articles": articles, "count": len(articles)}

        except Exception as e:
            return {"query": query, "source": source, "articles": [], "count": 0, "error": str(e)}

    elif source == "google_ko":
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    elif source == "google_en":
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
    else:
        return {
            "query": query, "articles": [], "count": 0,
            "error": f"지원하지 않는 source: {source}",
        }

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10,
        )
        response.raise_for_status()
        response.encoding = "utf-8"

        channel = ET.fromstring(response.text).find("channel")
        if channel is None:
            return {"query": query, "source": source, "articles": [], "count": 0, "error": "채널 없음"}

        articles = [
            {
                "title":       re.sub(r"<[^>]+>", "", item.findtext("title",       "")).strip(),
                "link":        (item.findtext("link") or "").strip(),
                "description": re.sub(r"<[^>]+>", "", item.findtext("description", "")).strip(),
                "pub_date":    (item.findtext("pubDate") or "").strip(),
            }
            for item in channel.findall("item")[:max_results]
        ]
        return {"query": query, "source": source, "articles": articles, "count": len(articles)}

    except Exception as e:
        return {"query": query, "source": source, "articles": [], "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# Tool 2: search_policy_news
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def search_policy_news(max_results_per_query: int = 10) -> dict:
    """국내외 정책 관련 뉴스를 멀티 쿼리로 수집합니다."""
    all_articles = []
    seen_links   = set()

    for query in POLICY_QUERIES["naver"]:
        for article in search_news(query=query, max_results=max_results_per_query, source="naver").get("articles", []):
            link = article.get("link", "")
            if link and link not in seen_links:
                article["source_query"] = query
                article["source_lang"]  = "ko"
                seen_links.add(link)
                all_articles.append(article)

    for query in POLICY_QUERIES["google_en"]:
        for article in search_news(query=query, max_results=max_results_per_query, source="google_en").get("articles", []):
            link = article.get("link", "")
            if link and link not in seen_links:
                article["source_query"] = query
                article["source_lang"]  = "en"
                seen_links.add(link)
                all_articles.append(article)

    return {
        "articles":    all_articles,
        "count":       len(all_articles),
        "queries_used": POLICY_QUERIES,
        "note":        "키워드 필터 없음 — 에이전트가 직접 관련성 판단",
    }


# ─────────────────────────────────────────────────────────────────
# Tool 3: get_exchange_rate
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_exchange_rate(days: int = 30) -> dict:
    """FRED API로 USD/KRW 환율을 가져옵니다."""
    if not FRED_API_KEY:
        return {"error": "FRED_API_KEY가 설정되지 않았습니다."}

    try:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days + 10)

        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id":         "DEXKOUS",
                "api_key":           FRED_API_KEY,
                "file_type":         "json",
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end":   end_date.strftime("%Y-%m-%d"),
                "sort_order":        "desc",
                "limit":             days,
            },
            timeout=10,
        )
        response.raise_for_status()

        valid = [
            {"date": o["date"], "rate": float(o["value"])}
            for o in response.json().get("observations", [])
            if o["value"] != "."
        ]

        if not valid:
            return {"error": "유효한 환율 데이터 없음"}

        latest     = valid[0]
        prev       = valid[1] if len(valid) > 1 else None
        change_pct = round((latest["rate"] - prev["rate"]) / prev["rate"] * 100, 4) if prev else None

        return {
            "series_id":   "DEXKOUS",
            "description": "USD/KRW 환율 (1달러 = ?원)",
            "latest_date": latest["date"],
            "latest_rate": latest["rate"],
            "prev_rate":   prev["rate"] if prev else None,
            "change_pct":  change_pct,
            "history":     valid,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# Tool 4: get_interest_rate
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_interest_rate(periods: int = 365) -> dict:
    """BOK ECOS API로 한국은행 기준금리를 가져옵니다."""
    if not BOK_ECOS_API_KEY:
        return {"error": "BOK_ECOS_API_KEY가 설정되지 않았습니다."}

    try:
        today        = datetime.today()
        end_period   = today.strftime("%Y%m%d")
        start_period = (today - timedelta(days=periods + 30)).strftime("%Y%m%d")

        response = requests.get(
            f"https://ecos.bok.or.kr/api/StatisticSearch"
            f"/{BOK_ECOS_API_KEY}/json/kr/1/{periods + 30}"
            f"/722Y001/D/{start_period}/{end_period}/0101000",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if "RESULT" in data:
            return {"error": f"ECOS API 오류: {data['RESULT'].get('MESSAGE')}"}

        rows = data.get("StatisticSearch", {}).get("row", [])
        if not rows:
            return {"error": "BOK ECOS 데이터 없음"}

        history = sorted(
            [{"date": r["TIME"], "rate": float(r["DATA_VALUE"])}
             for r in rows if r.get("DATA_VALUE") not in (None, "", " ")],
            key=lambda x: x["date"], reverse=True,
        )

        unique_history = []
        for h in history:
            if not unique_history or h["rate"] != unique_history[-1]["rate"]:
                unique_history.append(h)
            if len(unique_history) >= periods:
                break

        latest = unique_history[0] if unique_history else None
        prev   = unique_history[1] if len(unique_history) > 1 else None

        return {
            "stat_code":   "722Y001",
            "description": "한국은행 기준금리 (%)",
            "latest_date": latest["date"] if latest else None,
            "latest_rate": latest["rate"] if latest else None,
            "prev_rate":   prev["rate"]   if prev   else None,
            "history":     unique_history,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# Tool 5: get_dxy  [v5.2 신규]
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_dxy(days: int = 30) -> dict:
    """
    FRED API로 미국 달러 인덱스(DXY 근사치)를 가져옵니다.
    시리즈: DTWEXBGS — Nominal Broad U.S. Dollar Index (Goods)

    달러 강세(DXY↑) → 원화 약세, 신흥국 자금 이탈, 원자재 하락 압력
    달러 약세(DXY↓) → 원화 강세, 신흥국 자금 유입, 원자재 상승 압력
    """
    if not FRED_API_KEY:
        return {"error": "FRED_API_KEY가 설정되지 않았습니다."}

    try:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=days + 10)

        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id":         "DTWEXBGS",
                "api_key":           FRED_API_KEY,
                "file_type":         "json",
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end":   end_date.strftime("%Y-%m-%d"),
                "sort_order":        "desc",
                "limit":             days,
            },
            timeout=10,
        )
        response.raise_for_status()

        valid = [
            {"date": o["date"], "value": round(float(o["value"]), 2)}
            for o in response.json().get("observations", [])
            if o["value"] != "."
        ]

        if not valid:
            return {"error": "유효한 DXY 데이터 없음"}

        latest = valid[0]
        prev   = valid[1] if len(valid) > 1 else None
        change_pct = round((latest["value"] - prev["value"]) / prev["value"] * 100, 4) if prev else None

        if change_pct is None:
            direction = "데이터 부족"
        elif change_pct > 0.3:
            direction = "달러 강세 — 원화 약세 압력, 신흥국 자금 이탈 우려"
        elif change_pct < -0.3:
            direction = "달러 약세 — 원화 강세 전환, 위험자산 선호 환경"
        else:
            direction = "달러 보합 — 방향성 불명확"

        return {
            "series_id":    "DTWEXBGS",
            "description":  "미국 달러 인덱스 (Nominal Broad, FRED)",
            "latest_date":  latest["date"],
            "latest_value": latest["value"],
            "prev_value":   prev["value"] if prev else None,
            "change_pct":   change_pct,
            "direction":    direction,
            "history":      valid,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# Tool 6: get_bok_schedule  [v5.3 신규]
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_bok_schedule(days_ahead: int = 90) -> dict:
    """
    한국은행 금융통화위원회(금통위) 일정 조회.

    데이터 소스:
      - 2026년 금통위 예정일: 한국은행 공식 발표 기준 (연초 공표)
      - 최근 기준금리 결정: BOK ECOS API (get_interest_rate와 동일 소스)

    반환값:
      upcoming_meetings: days_ahead일 이내 예정된 금통위 일정
      recent_decision: 가장 최근 금리 결정 날짜 및 금리
      all_2026: 2026년 전체 금통위 일정
    """
    # 2026년 금통위 통화정책방향 결정회의 예정일
    # 출처: 한국은행 (https://www.bok.or.kr) 연초 공표
    BOK_2026 = [
        "2026-01-16", "2026-02-25", "2026-04-17",
        "2026-05-29", "2026-07-17", "2026-08-28",
        "2026-10-16", "2026-11-27",
    ]

    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    upcoming = [
        {
            "date": d,
            "days_until": (date.fromisoformat(d) - today).days,
            "label": "금통위 통화정책방향 결정회의",
        }
        for d in BOK_2026
        if today <= date.fromisoformat(d) <= cutoff
    ]

    # 최근 금리 결정 (BOK ECOS에서 가져옴)
    recent_decision = None
    if BOK_ECOS_API_KEY:
        try:
            rate_data = get_interest_rate(periods=2)
            if "latest_date" in rate_data:
                recent_decision = {
                    "date": rate_data["latest_date"],
                    "rate": rate_data["latest_rate"],
                    "prev_rate": rate_data["prev_rate"],
                    "direction": (
                        "인상" if rate_data["latest_rate"] > rate_data["prev_rate"]
                        else "인하" if rate_data["latest_rate"] < rate_data["prev_rate"]
                        else "동결"
                    ) if rate_data.get("prev_rate") else "불명",
                }
        except Exception:
            pass

    return {
        "upcoming_meetings": upcoming,
        "upcoming_count": len(upcoming),
        "recent_decision": recent_decision,
        "all_2026": BOK_2026,
        "note": "2026년 일정은 한국은행 연초 공표 기준. 임시회의는 미포함.",
    }


# ─────────────────────────────────────────────────────────────────
# Tool 7: get_legislation_schedule  [v5.3 신규]
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_legislation_schedule(days: int = 30, keyword: str = "") -> dict:
    """
    법제처 입법예고 일정 조회.
    경제·금융·산업 관련 법안의 입법예고 기간을 확인합니다.

    데이터 소스: 법제처 국가법령정보 OpenAPI (www.law.go.kr)
    API 키 불필요 (공개 API).

    Args:
        days: 최근 며칠 이내 입법예고 (기본 30일)
        keyword: 검색 키워드 (예: "반도체", "금융", "" = 전체)

    반환값:
      items: 입법예고 목록 (법령명, 예고기간, 소관부처)
      투자 관련 필터: 경제/금융/산업/세금/무역 키워드 포함 건만 반환
    """
    # 투자 관련 법령 필터 키워드
    INVEST_KEYWORDS = [
        "금융", "증권", "자본", "투자", "세금", "세제",
        "산업", "반도체", "에너지", "무역", "수출", "수입",
        "기업", "회사", "상장", "공정거래", "독점",
        "노동", "고용", "임금", "부동산", "건설",
    ]

    today = date.today()
    start = (today - timedelta(days=days)).strftime("%Y%m%d")
    end   = today.strftime("%Y%m%d")

    try:
        # 법제처 입법예고 OpenAPI
        params = {
            "OC": "test",
            "target": "lsInfoP",
            "type": "JSON",
            "display": 100,
            "page": 1,
            "efYd": end,
            "stYd": start,
        }
        if keyword:
            params["query"] = keyword

        resp = requests.get(
            "https://www.law.go.kr/DRF/lawSearch.do",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_items = data.get("LawSearch", {}).get("law", [])
        if isinstance(raw_items, dict):
            raw_items = [raw_items]

        # 투자 관련 법령만 필터링
        filtered = []
        for item in raw_items:
            name = item.get("법령명한글", "") or item.get("법령명", "")
            dept = item.get("소관부처명", "")

            if keyword:
                # 키워드 검색 시 전체 반환
                filtered.append({
                    "name": name,
                    "department": dept,
                    "notice_start": item.get("공고일자", ""),
                    "notice_end": item.get("입법예고기간", ""),
                    "law_id": item.get("법령ID", ""),
                    "url": f"https://www.law.go.kr/lsInfoP.do?lsiSeq={item.get('법령ID', '')}",
                })
            else:
                # 키워드 없을 시 투자 관련만 필터
                if any(kw in name or kw in dept for kw in INVEST_KEYWORDS):
                    filtered.append({
                        "name": name,
                        "department": dept,
                        "notice_start": item.get("공고일자", ""),
                        "notice_end": item.get("입법예고기간", ""),
                        "law_id": item.get("법령ID", ""),
                        "url": f"https://www.law.go.kr/lsInfoP.do?lsiSeq={item.get('법령ID', '')}",
                    })

        return {
            "period": f"{start}~{end}",
            "keyword": keyword or "전체 (투자 관련 필터 적용)",
            "count": len(filtered),
            "items": filtered[:20],
            "source": "법제처 국가법령정보 OpenAPI",
        }

    except Exception as e:
        return {
            "period": f"{start}~{end}",
            "count": 0,
            "items": [],
            "error": str(e),
            "note": "법제처 API 호출 실패. 직접 확인: https://opinion.lawmaking.go.kr",
        }




# ─────────────────────────────────────────────────────────────────
# Tool 8: search_naver_stock_board  [v5.3 신규 - 커뮤니티]
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def search_naver_stock_board(ticker: str, pages: int = 2) -> dict:
    """
    네이버 종목 토론방 게시글 수집.
    finance.naver.com/item/board.naver 크롤링 (EUC-KR 인코딩).

    뉴스와 달리 실제 개인 투자자의 감성을 반영합니다.
    agree/disagree 비율로 긍정/부정 심리를 파악할 수 있습니다.

    Args:
        ticker: 종목 코드 (예: "005930")
        pages: 수집할 페이지 수 (기본 2 = 약 40~60건)

    Returns:
        posts: 게시글 목록 (제목, 날짜, 조회수, 공감/비공감)
        sentiment_hint: 공감/비공감 비율 기반 간이 심리 추정
    """
    try:
        from bs4 import BeautifulSoup
        import requests as _req
    except ImportError:
        return {"error": "beautifulsoup4 또는 requests가 설치되지 않았습니다."}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.naver.com",
    }
    all_posts = []
    errors = []

    for page in range(1, pages + 1):
        url = f"https://finance.naver.com/item/board.naver?code={ticker}&page={page}"
        try:
            resp = _req.get(url, headers=headers, timeout=10)
            resp.encoding = "euc-kr"  # 네이버는 EUC-KR 인코딩
            soup = BeautifulSoup(resp.text, "html.parser")

            table = soup.find("table", class_="type2")
            if not table:
                break

            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 5:
                    continue

                title_tag = cells[1].find("a") if len(cells) > 1 else None
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                if not title or len(title) < 2:
                    continue

                link = "https://finance.naver.com" + title_tag.get("href", "")

                def safe_int(text):
                    try:
                        return int(text.replace(",", "").strip())
                    except Exception:
                        return 0

                post = {
                    "title":    title,
                    "date":     cells[3].get_text(strip=True) if len(cells) > 3 else "",
                    "views":    safe_int(cells[4].get_text()) if len(cells) > 4 else 0,
                    "agree":    safe_int(cells[5].get_text()) if len(cells) > 5 else 0,
                    "disagree": safe_int(cells[6].get_text()) if len(cells) > 6 else 0,
                    "link":     link,
                    "source":   "naver_stock_board",
                }
                all_posts.append(post)

        except Exception as e:
            errors.append(f"page {page}: {str(e)}")
            break

    # 간이 감성 추정 (공감/비공감 합산)
    total_agree    = sum(p["agree"]    for p in all_posts)
    total_disagree = sum(p["disagree"] for p in all_posts)
    total_votes    = total_agree + total_disagree

    if total_votes > 0:
        agree_ratio = round(total_agree / total_votes, 2)
        if agree_ratio >= 0.65:
            sentiment_hint = f"긍정 우세 ({agree_ratio*100:.0f}% 공감)"
        elif agree_ratio <= 0.35:
            sentiment_hint = f"부정 우세 ({(1-agree_ratio)*100:.0f}% 비공감)"
        else:
            sentiment_hint = f"중립 (공감 {agree_ratio*100:.0f}%)"
    else:
        sentiment_hint = "투표 데이터 없음"

    return {
        "ticker":         ticker,
        "count":          len(all_posts),
        "posts":          all_posts[:50],
        "sentiment_hint": sentiment_hint,
        "total_agree":    total_agree,
        "total_disagree": total_disagree,
        "source":         "네이버 종목 토론방",
        "url":            f"https://finance.naver.com/item/board.naver?code={ticker}",
        "errors":         errors,
        "note": (
            "개인 투자자 심리 데이터입니다. "
            "단기 노이즈 포함 가능하므로 다른 지표와 함께 참고하세요."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Tool 9: search_daum_stock_board  [v5.3 신규 - 커뮤니티]
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def search_daum_stock_board(ticker: str, size: int = 30) -> dict:
    """
    다음 종목 토론방 게시글 수집.
    Daum Finance 내부 API 엔드포인트를 사용합니다.

    네이버 토론방과 함께 사용하면 커뮤니티 감성 데이터의 다양성을 높입니다.

    Args:
        ticker: 종목 코드 (예: "005930")
        size: 수집할 게시글 수 (기본 30, 최대 100)

    Returns:
        posts: 게시글 목록 (제목, 작성일, 조회수, 공감/비공감)
    """
    import requests as _req

    # Daum Finance 내부 API (비공개이지만 공개 접근 가능)
    url = f"https://finance.daum.net/api/discussions/A{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer":    f"https://finance.daum.net/quotes/A{ticker}/discussions",
        "Accept":     "application/json, text/plain, */*",
    }
    params = {
        "size":          min(size, 100),
        "page":          1,
        "commentCounts": "true",
    }

    try:
        resp = _req.get(url, headers=headers, params=params, timeout=10)

        # API가 200이 아닌 경우 (접근 차단 등)
        if resp.status_code != 200:
            return {
                "ticker":  ticker,
                "count":   0,
                "posts":   [],
                "error":   f"Daum API HTTP {resp.status_code}",
                "note":    "Daum Finance 내부 API가 차단되었습니다. 네이버 토론방(search_naver_stock_board)을 대신 사용하세요.",
                "url":     f"https://finance.daum.net/quotes/A{ticker}/discussions",
            }

        data = resp.json()
        raw_posts = data.get("data", [])

        posts = []
        for item in raw_posts:
            posts.append({
                "title":    item.get("title", ""),
                "date":     item.get("createdAt", "")[:10] if item.get("createdAt") else "",
                "views":    item.get("viewCount", 0),
                "agree":    item.get("likeCount", 0),
                "disagree": item.get("dislikeCount", 0),
                "comments": item.get("commentCount", 0),
                "source":   "daum_stock_board",
            })

        total_agree    = sum(p["agree"]    for p in posts)
        total_disagree = sum(p["disagree"] for p in posts)
        total_votes    = total_agree + total_disagree

        if total_votes > 0:
            agree_ratio = round(total_agree / total_votes, 2)
            if agree_ratio >= 0.65:
                sentiment_hint = f"긍정 우세 ({agree_ratio*100:.0f}% 공감)"
            elif agree_ratio <= 0.35:
                sentiment_hint = f"부정 우세 ({(1-agree_ratio)*100:.0f}% 비공감)"
            else:
                sentiment_hint = f"중립 (공감 {agree_ratio*100:.0f}%)"
        else:
            sentiment_hint = "투표 데이터 없음"

        return {
            "ticker":         ticker,
            "count":          len(posts),
            "posts":          posts,
            "sentiment_hint": sentiment_hint,
            "source":         "다음 종목 토론방",
            "url":            f"https://finance.daum.net/quotes/A{ticker}/discussions",
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "count":  0,
            "posts":  [],
            "error":  str(e),
            "note":   "Daum Finance 접근 실패. 네이버 토론방(search_naver_stock_board)을 대신 사용하세요.",
        }




# ─────────────────────────────────────────────────────────────────
# Tool 10: get_oecd_cli  [v5.3 신규]
# ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_oecd_cli(countries: list[str] = None) -> dict:
    """
    OECD 경기선행지수(CLI: Composite Leading Indicator) 조회.
    OECD SDMX 공개 API 사용 (API 키 불필요).

    CLI > 100 + 상승: 경기 확장 국면
    CLI < 100 + 하락: 경기 수축 국면
    방향 전환 시점이 실제 경기 전환 2~6개월 선행.

    한국 주식시장과의 관계:
    - 한국(KOR) CLI 상승 → 코스피 선행 상승 경향
    - 미국(USA) CLI 하락 → 글로벌 위험자산 회피 → 코스피 하락 압력
    - 중국(CHN) CLI 회복 → 한국 수출주 수혜

    Args:
        countries: 조회할 국가 코드 (기본: ["KOR", "USA", "CHN", "DEU"])
                   OECD 3자리 코드 사용

    Returns:
        각 국가별 최근 CLI 값 + 방향성 해석
    """
    if countries is None:
        countries = ["KOR", "USA", "CHN", "DEU"]

    results = {}
    errors  = []

    for country in countries:
        try:
            # OECD SDMX API (공개, 무료)
            url = (
                f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,4.0/"
                f"{country}.M.LI....AA.H/all"
                f"?dimensionAtObservation=AllDimensions&format=jsondata&startPeriod=2024-01"
            )
            resp = requests.get(url, timeout=15)

            if resp.status_code != 200:
                errors.append(f"{country}: HTTP {resp.status_code}")
                continue

            data = resp.json()

            # OECD SDMX JSON 파싱
            obs = data.get("dataSets", [{}])[0].get("observations", {})
            series_info = data.get("structure", {}).get("dimensions", {}).get("observation", [])

            # 시간 축 찾기
            time_dim = None
            for dim in series_info:
                if dim.get("id") == "TIME_PERIOD":
                    time_dim = dim
                    break

            if not time_dim or not obs:
                errors.append(f"{country}: 데이터 파싱 실패")
                continue

            time_values = [v["id"] for v in time_dim.get("values", [])]

            # 관측값 추출 (키는 차원 인덱스 조합)
            series_data = []
            for key, val_list in obs.items():
                try:
                    # 마지막 인덱스가 시간 인덱스
                    time_idx = int(key.split(":")[-1])
                    period   = time_values[time_idx]
                    value    = val_list[0]
                    if value is not None:
                        series_data.append({"period": period, "value": round(float(value), 2)})
                except Exception:
                    continue

            series_data.sort(key=lambda x: x["period"])

            if not series_data:
                errors.append(f"{country}: 유효한 관측값 없음")
                continue

            latest = series_data[-1]
            prev   = series_data[-2] if len(series_data) >= 2 else None
            trend  = "상승" if prev and latest["value"] > prev["value"] else "하락" if prev else "불명"

            # 경기 국면 해석
            if latest["value"] > 100 and trend == "상승":
                phase = "경기 확장 강화"
            elif latest["value"] > 100 and trend == "하락":
                phase = "경기 정점 통과 가능성"
            elif latest["value"] <= 100 and trend == "상승":
                phase = "경기 수축 완화 (회복 조짐)"
            else:
                phase = "경기 수축 진행"

            results[country] = {
                "latest_period": latest["period"],
                "cli_value":     latest["value"],
                "prev_value":    prev["value"] if prev else None,
                "trend":         trend,
                "phase":         phase,
                "history":       series_data[-12:],  # 최근 12개월
            }

        except Exception as e:
            errors.append(f"{country}: {str(e)}")

    return {
        "countries": results,
        "errors":    errors,
        "source":    "OECD SDMX Public API",
        "note":      "CLI는 약 2~6개월 선행 지표. 방향성(추세)이 절대값보다 중요.",
    }




# ─────────────────────────────────────────────────────────────────
# 서버 실행
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()
