"""
src/data/ohlcv_cache.py

OHLCV 캐시 (data/mentions.db에 ohlcv_cache 테이블).

screener Stage 1-A가 pykrx를 매일 110회 호출하다가 타임아웃하던 문제를
해결하기 위해 도입된 캐시 레이어.

  - 첫 실행: 종목별 get_market_ohlcv()로 65일치 초기 적재 (110회 1회만)
  - 이후 매일: get_market_ohlcv_by_ticker(today, market) 2회로 오늘만 추가
  - 지표 계산은 캐시에서 직접 읽음 (네트워크 호출 0)

PRIMARY KEY (ticker, date)로 중복 적재가 그대로 upsert가 되도록 설계.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import date as date_cls, timedelta
from typing import Iterable

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("MENTIONS_DB_PATH", "data/mentions.db")


# ─────────────────────────────────────────────────────────
# 연결 헬퍼
# ─────────────────────────────────────────────────────────

@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


# ─────────────────────────────────────────────────────────
# 스키마
# ─────────────────────────────────────────────────────────

def init_ohlcv_cache() -> None:
    """ohlcv_cache 테이블 + 인덱스를 생성. 이미 있으면 무시."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                ticker  TEXT    NOT NULL,
                date    TEXT    NOT NULL,
                open    REAL    NOT NULL,
                high    REAL    NOT NULL,
                low     REAL    NOT NULL,
                close   REAL    NOT NULL,
                volume  INTEGER NOT NULL,
                PRIMARY KEY (ticker, date)
            );

            CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date
                ON ohlcv_cache (ticker, date DESC);
        """)
        # G2: upsert가 덮어쓰기라 "언제 받은 값인지"가 사라졌다(S2 A-6). 받은 시각을 남긴다.
        cols = [r[1] for r in c.execute("PRAGMA table_info(ohlcv_cache)")]
        if "fetched_at" not in cols:
            c.execute("ALTER TABLE ohlcv_cache ADD COLUMN fetched_at TEXT")


# ─────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────

def upsert_ohlcv_rows(rows: Iterable[dict]) -> int:
    """
    rows: [{'ticker','date','open','high','low','close','volume'}, ...]
    PRIMARY KEY 충돌 시 값 갱신. 빈 입력 시 0 반환.
    """
    from datetime import datetime as _dt
    rows = [{**r, "fetched_at": r.get("fetched_at") or _dt.now().isoformat()} for r in rows]
    if not rows:
        return 0
    with _conn() as c:
        c.executemany(
            """
            INSERT INTO ohlcv_cache (ticker, date, open, high, low, close, volume, fetched_at)
            VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :fetched_at)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open   = excluded.open,
                high   = excluded.high,
                low    = excluded.low,
                close  = excluded.close,
                fetched_at = excluded.fetched_at,
                volume = excluded.volume
            """,
            rows,
        )
    return len(rows)


# ─────────────────────────────────────────────────────────
# 조회
# ─────────────────────────────────────────────────────────

def get_ohlcv_series(ticker: str, days: int = 65, as_of: str | None = None) -> list[dict]:
    """
    한 종목의 최근 N일 시계열을 오래된순으로 반환.
    각 dict: {'date','open','high','low','close','volume'}

    as_of: 'YYYY-MM-DD'. 주면 그날까지의 봉만 — 캐시는 넓게 두고 읽을 때 자른다
           (TradingAgents load_ohlcv: "Rows after curr_date are filtered out so backtests
           never see future prices"). None이면 전체(라이브).
    """
    with _conn() as c:
        rows = c.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM ohlcv_cache
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, as_of or "9999-12-31", days),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def cache_has_date(date_str: str) -> bool:
    """해당 날짜에 어떤 ticker라도 캐시에 있으면 True."""
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM ohlcv_cache WHERE date = ? LIMIT 1",
            (date_str,),
        ).fetchone()
    return row is not None


def get_ticker_count(ticker: str) -> int:
    """ticker의 캐시 행 수."""
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM ohlcv_cache WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    return int(row["n"]) if row else 0


def get_latest_cached_date(ticker: str | None = None) -> str | None:
    """
    캐시의 가장 최근 날짜 (YYYY-MM-DD).
    ticker가 주어지면 해당 종목 기준, 없으면 전체 기준.
    """
    with _conn() as c:
        if ticker:
            row = c.execute(
                "SELECT MAX(date) AS d FROM ohlcv_cache WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        else:
            row = c.execute("SELECT MAX(date) AS d FROM ohlcv_cache").fetchone()
    return row["d"] if row and row["d"] else None


def count_rows() -> int:
    """전체 캐시 행 수 (모니터링용)."""
    with _conn() as c:
        return int(c.execute("SELECT COUNT(*) FROM ohlcv_cache").fetchone()[0])


# ─────────────────────────────────────────────────────────
# 단독 실행 (DDL 적용 + 상태 확인)
# ─────────────────────────────────────────────────────────

def find_suspect_tickers(threshold: float = 0.35, days: int = 5) -> list[str]:
    """최근 days봉 안에 전일 대비 |종가 변화|가 threshold를 넘는 종목.
    pykrx 스냅샷 캐시는 액면분할 시 옛 행이 조정되지 않는다 — 그 종목은 재백필한다 (스펙 §1-2)."""
    with _conn() as c:
        rows = c.execute("SELECT ticker, close FROM ohlcv_cache ORDER BY ticker, date DESC").fetchall()
    recent: dict[str, list[float]] = {}
    for r in rows:
        lst = recent.setdefault(r["ticker"], [])
        if len(lst) < days + 1:
            lst.append(float(r["close"]))
    out = []
    for t, closes in recent.items():                      # 최신 → 과거
        if any(b > 0 and abs(a / b - 1.0) > threshold for a, b in zip(closes, closes[1:])):
            out.append(t)
    return sorted(out)


if __name__ == "__main__":
    init_ohlcv_cache()
    print(f"[ohlcv_cache] DB: {DB_PATH}")
    print(f"[ohlcv_cache] 행 수: {count_rows()}")
    print(f"[ohlcv_cache] 최신 날짜: {get_latest_cached_date()}")
