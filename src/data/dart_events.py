"""
src/data/dart_events.py

DART 공시 일괄 수집 — 스크리닝 v2 팩터 `dart_event`의 원천.
스펙: docs/superpowers/specs/2026-09-21-screening-v2-design.md §2-b

[왜 종목별 호출이 아닌가]
  `krx_market.get_dart_disclosure`는 종목당 1회 호출이라 유니버스 110개면 110회다.
  `list.json`은 `stock_code` 없이 부르면 그 날짜 창의 전체 공시를 주므로(100건/페이지)
  한 번에 받아 응답의 `stock_code`로 유니버스만 거른다. 무료.

[창]
  직전 평일 ~ 오늘 (YYYYMMDD). 새벽 실행이라 "오늘" 공시는 아직 거의 없고,
  실제 정보는 직전 거래일 장 마감 후 공시다. 월요일이면 금~월.

실패 규약: 어떤 예외도 밖으로 내지 않는다. DB에 이미 있던 창 안 공시를 돌려준다.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import requests

from src.data.mention_db import get_conn

logger = logging.getLogger(__name__)

DART_LIST_URL   = "https://opendart.fss.or.kr/api/list.json"
PAGE_COUNT      = 100     # API 최대
MAX_PAGES       = 30      # 하루 전체 공시는 보통 500~2,000건 → 5~20페이지
REQUEST_TIMEOUT = 15

# 제목 키워드 → 카테고리. 첫 매칭이 이긴다. 정기·형식 공시(사업보고서, 소유상황보고 등)는 None.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": ("잠정실적", "영업실적", "매출액또는손익", "영업(잠정)실적"),
    "contract": ("단일판매", "공급계약"),
    "capital":  ("자기주식", "유상증자", "전환사채", "신주인수권"),
    "control":  ("합병", "분할", "최대주주변경", "최대주주 변경"),
}


def classify(title: str) -> str | None:
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in title for k in kws):
            return cat
    return None


def init_dart_table() -> None:
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dart_events (
                ticker     TEXT NOT NULL,
                rcept_dt   TEXT NOT NULL,          -- YYYYMMDD
                rcept_no   TEXT NOT NULL UNIQUE,   -- 접수번호 = 공시 1건의 키
                title      TEXT NOT NULL,
                category   TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_dart_ticker_date ON dart_events(ticker, rcept_dt);
        """)


# ── 날짜 창 ────────────────────────────────────────────────

def last_weekday_before(today: date) -> date:
    """오늘 이전의 마지막 평일(월~금). KRX 휴장일은 보지 않는다 — 창이 하루 넓어질 뿐이다."""
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def event_window(today: date) -> tuple[str, str]:
    return last_weekday_before(today).strftime("%Y%m%d"), today.strftime("%Y%m%d")


# ── 수집 ──────────────────────────────────────────────────

def fetch_disclosures(bgn_de: str, end_de: str, api_key: str) -> list[dict]:
    """창 안 전체 공시(원본 item dict). status '013'(데이터 없음)이면 []."""
    items: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        resp = requests.get(
            DART_LIST_URL,
            params={"crtfc_key": api_key, "bgn_de": bgn_de, "end_de": end_de,
                    "page_no": page, "page_count": PAGE_COUNT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        status = str(data.get("status") or "")
        if status == "013":                      # 조회 결과 없음 — 정상
            break
        if status != "000":
            # 키 미등록(010)·IP 불일치(012)·한도 초과(020)·점검(800)·키 만료(900) 등.
            # "공시 0건"으로 위장하면 워밍업 3주간 후보 0개가 조용히 반복된다 → 예외로 올려 경고를 남긴다.
            raise RuntimeError(f"DART status {status}: {data.get('message', '')}")
        items.extend(data.get("list") or [])
        if page >= int(data.get("total_page") or 1):
            break
    return items


def filter_universe(items: list[dict], universe: set[str]) -> list[dict]:
    out: list[dict] = []
    for it in items:
        code = str(it.get("stock_code") or "").strip()
        if code not in universe:
            continue
        title = str(it.get("report_nm") or "")
        cat = classify(title)
        if cat is None:
            continue
        out.append({
            "ticker":   code,
            "rcept_dt": str(it.get("rcept_dt") or ""),
            "rcept_no": str(it.get("rcept_no") or ""),
            "title":    title,
            "category": cat,
        })
    return out


def store_events(rows: list[dict]) -> int:
    """INSERT OR IGNORE (rcept_no UNIQUE). 새로 들어간 행 수."""
    if not rows:
        return 0
    now = datetime.now().isoformat()
    with get_conn() as conn:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO dart_events (ticker, rcept_dt, rcept_no, title, category, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [(r["ticker"], r["rcept_dt"], r["rcept_no"], r["title"], r["category"], now) for r in rows],
        )
        return cur.rowcount


def get_events(tickers: list[str], bgn_de: str, end_de: str) -> dict[str, list[dict]]:
    want = set(tickers)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT ticker, rcept_dt, title, category FROM dart_events "
            "WHERE rcept_dt BETWEEN ? AND ? ORDER BY rcept_dt, rcept_no",
            (bgn_de, end_de),
        ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        if r["ticker"] in want:
            out.setdefault(r["ticker"], []).append(
                {"date": r["rcept_dt"], "title": r["title"], "category": r["category"]})
    return out


def collect_dart_events(universe: list[str], today: date | None = None) -> dict[str, list[dict]]:
    """수집 → 유니버스 필터 → 저장 → 창 안 공시 반환. raise 없음."""
    today = today or date.today()
    try:
        init_dart_table()
    except Exception as e:
        logger.warning(f"[dart_events] 테이블 초기화 실패: {e}")
        return {}
    bgn, end = event_window(today)
    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        logger.warning("[dart_events] DART_API_KEY 없음 — 공시 이벤트 0건")
        return {}
    try:
        items = fetch_disclosures(bgn, end, api_key)
        rows  = filter_universe(items, set(universe))
        n     = store_events(rows)
        print(f"  [dart_events] {bgn}~{end} 전체 {len(items)}건 → 유니버스 주요 공시 {len(rows)}건 (신규 {n})")
    except Exception as e:
        logger.warning(f"[dart_events] 수집 실패(계속, DB에 있던 것만 사용): {type(e).__name__}: {e}")
        print(f"  ⚠️ [dart_events] 수집 실패: {type(e).__name__}")
    try:
        return get_events(universe, bgn, end)
    except Exception as e:
        logger.warning(f"[dart_events] 조회 실패: {e}")
        return {}
