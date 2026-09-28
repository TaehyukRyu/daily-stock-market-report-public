"""
src/screening/stage1c_news.py

Stage 1-C: 매크로 뉴스 LLM 스크린 (유료, LLM 1회 호출).

1. 네이버 금융 뉴스 헤드라인 최대 30건 수집 (RSS/리스트 페이지)
2. NEWS_SCREEN_MODEL(현재 claude-haiku-4-5)에 ticker 후보 + 헤드라인을 1회 호출
   (섀도: SHADOW_POLICIES['stage1c_model']이 활성이면 candidate 모델도 같은 입력으로 호출해 기록만)
3. JSON 응답으로 영향받을 가능성 있는 종목 최대 15개 + 이유 추출

실패 모드:
  - RSS 수집 양쪽 모두 실패 → 모든 종목 news_match=False
  - LLM 호출/JSON 파싱 실패 → 모든 종목 news_match=False
  - 어떤 경우에도 raise하지 않는다 (스크리너가 다음 stage로 계속)
"""

from __future__ import annotations

import json
import logging
from datetime import date
import os
import re

import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic

from src.utils.llm_budget import record_usage
from src.utils.shadow_log import (
    SELF_VARIANT, SHADOW_POLICIES, has_variant_on, input_hash, log_shadow, shadow_active,
)

logger = logging.getLogger(__name__)

# 프로젝트 표준 모델 (CLAUDE.md DEP-01: AsyncAnthropic 직접 사용 패턴)
NEWS_SCREEN_MODEL = "claude-haiku-4-5-20251001"

# 2026-09-11 복구: finance.naver.com/news/news_list.naver 가 Npay 증권(SPA)으로 개편되어
# HTML 셀렉터(dl.newsList 등)가 전부 0건이 됐다 (3회 연속 헤드라인 0건). 같은 데이터를
# 모바일 JSON API가 내려주므로 그쪽으로 옮긴다. 항목 키: tit(제목), dt(YYYYMMDDHHMMSS), ohnm(언론사).
NAVER_NEWS_API = "https://m.stock.naver.com/api/news/list?category={category}&pageSize={size}"
NAVER_NEWS_CATEGORIES = ("mainnews", "flashnews")     # 주요뉴스 → 실시간 속보 순으로 폴백

# 옛 HTML 페이지 (참고용. 더 이상 뉴스 리스트를 담지 않는다)
NAVER_FINANCE_PRIMARY  = "https://finance.naver.com/news/mainnews"
NAVER_FINANCE_FALLBACK = "https://finance.naver.com/news/flashnews"

# 직전 실행에서 수집한 헤드라인 수. 건강도 판정(daily_runner)이 읽는다.
LAST_HEADLINE_COUNT: int | None = None
# 직전 실행의 헤드라인 원본 메타 [{title, published_at, source, category}]. news_archive(백테스트용)가 읽는다.
# 판단에는 쓰지 않는다 — LLM에는 여전히 제목 리스트만 간다.
LAST_HEADLINE_ITEMS: list[dict] = []

MAX_HEADLINES         = 30
MAX_SELECTED_TICKERS  = 15
REQUEST_TIMEOUT_SEC   = 10

# temperature를 거부하는 모델 (400 invalid_request_error: "`temperature` is deprecated for this model").
# 2026-09-13 실행 34748924455에서 섀도 candidate(claude-sonnet-5) 호출이 전부 이 오류로 죽어
# llm_ab_log에 candidate 행이 한 건도 안 쌓였다 (review_by 2026-09-26).
# baseline(claude-haiku-4-5)은 temperature=0으로 정상 동작하므로 그대로 둔다 —
# 실험 도중에 baseline 호출 조건을 바꾸면 이미 기록된 52행과 비교가 안 된다.
NO_TEMPERATURE_PREFIXES = ("claude-sonnet-5", "claude-opus-5", "claude-fable-5")

# 확장 사고를 기본으로 켜는 모델의 max_tokens.
#
# [왜 따로 두나]
#   사고 토큰도 max_tokens에 포함된다. run 34761364921에서 sonnet-5가 2048을
#   사고로 거의 다 쓰고 JSON을 끝맺지 못해 "응답에서 JSON을 찾지 못함"으로 죽었다
#   (출력 약 1,900토큰 = 상한 소진). baseline(haiku-4-5)은 사고가 없어 2048로 충분하다.
MAX_TOKENS          = 2048
MAX_TOKENS_THINKING = 8192


def _supports_temperature(model: str) -> bool:
    return not model.startswith(NO_TEMPERATURE_PREFIXES)


def _max_tokens_for(model: str) -> int:
    """사고 모델은 사고 토큰만큼 여유를 더 준다."""
    return MAX_TOKENS_THINKING if model.startswith(NO_TEMPERATURE_PREFIXES) else MAX_TOKENS


def _first_text(content) -> str:
    """응답 블록 목록에서 첫 text 블록의 문자열.

    [왜 content[0].text가 아닌가]
      claude-sonnet-5는 확장 사고가 기본이라 content[0]이 ThinkingBlock으로 오고
      .text 속성이 없다. 2026-09-13 실행 34756880693에서
      "AttributeError: 'ThinkingBlock' object has no attribute 'text'"로 candidate가 또 죽었다.
      baseline(haiku-4-5)은 text 블록 하나만 오므로 동작이 같다.
    """
    for block in content or []:
        if getattr(block, "type", None) == "text" or hasattr(block, "text"):
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return text
    return ""


# ─────────────────────────────────────────────────────────
# 1. 헤드라인 수집
# ─────────────────────────────────────────────────────────

def _fetch_headlines_from(url: str) -> list[str]:
    """단일 URL에서 헤드라인을 긁어온다. 실패 시 []."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        # 네이버 금융은 EUC-KR. requests의 자동 감지 신뢰
        resp.encoding = resp.apparent_encoding or "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 뉴스 리스트는 <dl class="newsList"> 또는 <ul class="realtimeNewsList"> 안 <a>
        anchors = soup.select("dl.newsList a, ul.realtimeNewsList a, .articleSubject a")
        titles: list[str] = []
        seen: set[str] = set()
        for a in anchors:
            text = a.get_text(strip=True)
            if not text or len(text) < 5:
                continue
            if text in seen:
                continue
            seen.add(text)
            titles.append(text)
            if len(titles) >= MAX_HEADLINES:
                break
        return titles
    except Exception as e:
        logger.debug(f"[stage1c] {url} 수집 실패: {e}")
        return []


def _parse_news_items(items: list) -> list[str]:
    """API 응답(list[dict]) → 제목 리스트. 중복·빈 제목 제거, MAX_HEADLINES 상한."""
    titles: list[str] = []
    seen: set[str] = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        text = str(it.get("tit") or it.get("title") or "").strip()
        if len(text) < 5 or text in seen:
            continue
        seen.add(text)
        titles.append(text)
        if len(titles) >= MAX_HEADLINES:
            break
    return titles


def _items_meta(items: list, category: str) -> list[dict]:
    """API 응답 → 보관용 메타. _parse_news_items와 같은 필터(5자 미만·중복 제거)."""
    out: list[dict] = []
    seen: set[str] = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        text = str(it.get("tit") or it.get("title") or "").strip()
        if len(text) < 5 or text in seen:
            continue
        seen.add(text)
        out.append({"title": text, "published_at": it.get("dt"), "source": it.get("ohnm"), "category": category})
    return out


def _fetch_headlines_api(category: str) -> list[str]:
    """모바일 JSON API 한 카테고리. 실패 시 []."""
    global LAST_HEADLINE_ITEMS
    url = NAVER_NEWS_API.format(category=category, size=MAX_HEADLINES)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
        items = resp.json()
        LAST_HEADLINE_ITEMS = _items_meta(items, category)
        return _parse_news_items(items)
    except Exception as e:
        logger.warning(f"[stage1c] 뉴스 API 실패 ({category}): {type(e).__name__}: {e}")
        return []


def fetch_market_headlines() -> list[str]:
    """mainnews → flashnews 순으로 헤드라인 수집. 수집 수를 LAST_HEADLINE_COUNT에 남긴다."""
    global LAST_HEADLINE_COUNT, LAST_HEADLINE_ITEMS
    LAST_HEADLINE_ITEMS = []
    titles: list[str] = []
    for cat in NAVER_NEWS_CATEGORIES:
        titles = _fetch_headlines_api(cat)
        if titles:
            break
    LAST_HEADLINE_COUNT = len(titles)
    return titles


# ─────────────────────────────────────────────────────────
# 2. ticker → 종목명 매핑
# ─────────────────────────────────────────────────────────

def _ticker_name_map(tickers: list[str]) -> dict[str, str]:
    """pykrx로 ticker → 종목명 매핑. 실패는 ticker 코드 자체로 폴백."""
    from pykrx import stock as pykrx_stock
    out: dict[str, str] = {}
    for t in tickers:
        try:
            name = pykrx_stock.get_market_ticker_name(t)
            out[t] = name or t
        except Exception:
            out[t] = t
    return out


# ─────────────────────────────────────────────────────────
# 3. LLM 1회 호출
# ─────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = "[REDACTED] Proprietary prompt engineering"


def _call_llm(headlines: list[str], ticker_names: dict[str, str]) -> dict | None:
    """Anthropic 호출 → JSON dict. 실패 시 None. (기존 시그니처 유지)"""
    parsed, _meta = _call_llm_with_meta(headlines, ticker_names, NEWS_SCREEN_MODEL)
    return parsed


def _call_llm_with_meta(
    headlines: list[str], ticker_names: dict[str, str], model: str,
) -> tuple[dict | None, dict]:
    """모델을 지정해 호출하고 (parsed, meta)를 돌려준다.

    meta = {"input_tokens", "output_tokens", "latency_ms"} — 섀도 로그용.
    """
    import time as _time
    meta = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("[stage1c] ANTHROPIC_API_KEY 없음 — LLM 호출 스킵")
        return None, meta

    headlines_block   = "\n".join(f"- {h}" for h in headlines)
    ticker_name_block = "\n".join(f"- {t}: {n}" for t, n in ticker_names.items())
    prompt = _PROMPT_TEMPLATE.format(
        headlines_block   = headlines_block,
        ticker_name_block = ticker_name_block,
        max_select        = MAX_SELECTED_TICKERS,
    )

    try:
        # SDK 레벨 60초 timeout — node_with_timeout(asyncio.timeout)은 worker
        # thread를 끊지 못하므로 외부 라이브러리 timeout으로 추가 보호.
        client = Anthropic(api_key=api_key, timeout=60.0)
        t0 = _time.monotonic()
        kwargs: dict = {
            "model":      model,
            "max_tokens": _max_tokens_for(model),
            "messages":   [{"role": "user", "content": prompt}],
        }
        if _supports_temperature(model):
            kwargs["temperature"] = 0
        resp = client.messages.create(**kwargs)
        meta["latency_ms"] = int((_time.monotonic() - t0) * 1000)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            meta["input_tokens"]  = int(getattr(usage, "input_tokens", 0) or 0)
            meta["output_tokens"] = int(getattr(usage, "output_tokens", 0) or 0)
        record_usage(model, usage)
        text = _first_text(resp.content)

        # JSON만 추출 (가장 바깥 {..})
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            # stop_reason을 같이 남긴다 — max_tokens 소진인지 다른 이유인지 한 줄로 구분된다.
            logger.warning(
                f"[stage1c] LLM 응답에서 JSON을 찾지 못함 ({model}) "
                f"stop_reason={getattr(resp, 'stop_reason', '?')} "
                f"출력 {meta['output_tokens']}/{kwargs['max_tokens']}토큰 텍스트 {len(text)}자"
            )
            return None, meta
        return json.loads(match.group(0)), meta
    except json.JSONDecodeError as e:
        logger.warning(f"[stage1c] JSON 파싱 실패 ({model}): {e}")
        return None, meta
    except Exception as e:
        logger.warning(f"[stage1c] LLM 호출 실패 ({model}): {type(e).__name__}: {e}")
        return None, meta


# ─────────────────────────────────────────────────────────
# 3-b. 섀도 A/B (F-6) — 리포트에는 baseline만 쓴다
# ─────────────────────────────────────────────────────────

SHADOW_NAME = "stage1c_model"


def _direct_ratio(parsed: dict | None, headlines: list[str], ticker_names: dict[str, str]) -> float | None:
    """선정 종목 중 종목명이 헤드라인에 직접 등장하는 비율.

    S1 3-3이 예상한 퇴화 양상("헤드라인에 이름이 나온 종목만 고르는 표층 매칭")을
    숫자로 잡기 위한 지표. 1.0에 가까울수록 표층 매칭.
    """
    if not parsed:
        return None
    selected = [str(t) for t in (parsed.get("selected_tickers") or [])]
    if not selected:
        return None
    blob = "\n".join(headlines)
    hits = sum(1 for t in selected if ticker_names.get(t, t) and ticker_names.get(t, t) in blob)
    return hits / len(selected)


def _run_shadow(
    headlines: list[str], ticker_names: dict[str, str],
    baseline_parsed: dict | None, baseline_meta: dict,
) -> None:
    """활성 상태면 candidate 모델을 같은 입력으로 호출하고 두 결과를 기록한다.

    어떤 실패도 스크리닝을 막지 않는다. baseline 결과는 이미 확정된 뒤다.
    """
    if not shadow_active(SHADOW_NAME):
        return
    policy = SHADOW_POLICIES[SHADOW_NAME]
    try:
        h = input_hash(NEWS_SCREEN_MODEL, *headlines, *sorted(ticker_names))
        log_shadow(SHADOW_NAME, "baseline", NEWS_SCREEN_MODEL, h,
                   {"selected_tickers": (baseline_parsed or {}).get("selected_tickers"),
                    "direct_ratio": _direct_ratio(baseline_parsed, headlines, ticker_names)},
                   baseline_meta["input_tokens"], baseline_meta["output_tokens"], baseline_meta["latency_ms"])

        cand_parsed, cand_meta = _call_llm_with_meta(headlines, ticker_names, policy["candidate"])
        log_shadow(SHADOW_NAME, "candidate", policy["candidate"], h,
                   {"selected_tickers": (cand_parsed or {}).get("selected_tickers"),
                    "direct_ratio": _direct_ratio(cand_parsed, headlines, ticker_names)},
                   cand_meta["input_tokens"], cand_meta["output_tokens"], cand_meta["latency_ms"])
        logger.info(f"[stage1c] 섀도 기록: baseline={NEWS_SCREEN_MODEL} candidate={policy['candidate']}")
        _run_self_baseline(headlines, ticker_names, h)
    except Exception as e:
        logger.warning(f"[stage1c] 섀도 실패(무시): {type(e).__name__}: {e}")


def _run_self_baseline(headlines: list[str], ticker_names: dict[str, str], h: str) -> None:
    """baseline 모델을 **같은 입력으로 한 번 더** 불러 자기 겹침률 기준선을 만든다.

    [왜]
      baseline vs candidate 겹침률이 0.22~0.26으로 낮은데, 그것이 "모델이 달라서"인지
      "Stage 1-C가 원래 흔들려서"인지 구분할 대조군이 없었다. 같은 모델을 두 번 부르면
      그 기준선이 생긴다.

    하루 1회만 부른다. 같은 날 재실행(검증 dispatch 등)에서 중복 호출하면
    비용만 늘고 표본은 CR-15 규칙상 하루 하나로 접히기 때문이다.
    비용은 haiku 1콜 약 $0.006.
    """
    run_date = date.today().isoformat()
    if has_variant_on(SHADOW_NAME, SELF_VARIANT, run_date):
        logger.info(f"[stage1c] 자기 겹침률 — {run_date} 이미 기록됨, 재호출 생략")
        return
    parsed, meta = _call_llm_with_meta(headlines, ticker_names, NEWS_SCREEN_MODEL)
    log_shadow(SHADOW_NAME, SELF_VARIANT, NEWS_SCREEN_MODEL, h,
               {"selected_tickers": (parsed or {}).get("selected_tickers"),
                "direct_ratio": _direct_ratio(parsed, headlines, ticker_names)},
               meta["input_tokens"], meta["output_tokens"], meta["latency_ms"])
    logger.info(f"[stage1c] 자기 겹침률 기준선 기록: {NEWS_SCREEN_MODEL} 재호출")


# ─────────────────────────────────────────────────────────
# 4. 공개 API
# ─────────────────────────────────────────────────────────

def _all_false(tickers: list[str]) -> dict[str, dict]:
    return {t: {"news_match": False, "reason": ""} for t in tickers}


def run_news_screen(tickers: list[str]) -> dict[str, dict]:
    """
    각 ticker에 대해 {"news_match": bool, "reason": str} 반환.

    어떤 단계든 실패하면 모든 종목 news_match=False (raise 금지).
    """
    if not tickers:
        return {}

    headlines = fetch_market_headlines()
    if not headlines:
        logger.warning("[stage1c] 헤드라인 수집 실패 — 전 종목 news_match=False")
        return _all_false(tickers)

    ticker_names = _ticker_name_map(tickers)
    parsed, meta = _call_llm_with_meta(headlines, ticker_names, NEWS_SCREEN_MODEL)
    _run_shadow(headlines, ticker_names, parsed, meta)   # 기록만. 아래는 baseline만 쓴다
    if not parsed:
        return _all_false(tickers)

    selected = parsed.get("selected_tickers") or []
    reasons  = parsed.get("reasons") or {}
    if not isinstance(selected, list):
        logger.warning("[stage1c] selected_tickers 형식 오류 — 폴백")
        return _all_false(tickers)

    # LLM이 16개 이상 반환했어도 안전하게 잘라낸다
    selected_set = set(str(t) for t in selected[:MAX_SELECTED_TICKERS])

    out: dict[str, dict] = {}
    for t in tickers:
        if t in selected_set:
            out[t] = {"news_match": True, "reason": str(reasons.get(t, ""))}
        else:
            out[t] = {"news_match": False, "reason": ""}
    return out


# ─────────────────────────────────────────────────────────
# 단독 실행 (단위 테스트)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_tickers = ["005930", "000660", "005380", "035420"]

    print("[stage1c] 헤드라인 수집 테스트")
    titles = fetch_market_headlines()
    print(f"  → {len(titles)}건 수집")
    for t in titles[:5]:
        print(f"   - {t}")
    print()

    print("[stage1c] LLM 스크린 테스트")
    res = run_news_screen(test_tickers)
    for t, info in res.items():
        flag = "MATCH" if info["news_match"] else "  -  "
        print(f"  {flag} {t}: {info['reason']}")
