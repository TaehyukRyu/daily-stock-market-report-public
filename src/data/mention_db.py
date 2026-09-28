"""
src/data/mention_db.py

SQLite DB 초기화 및 CRUD 레이어

[테이블 2개]
  mentions            — 원시 데이터 (기사 1건 = 1행)
  daily_mention_stats — 일별 집계 (종목 × 날짜 = 1행)

[DB 경로]
  data/mentions.db (루트 data/ 폴더에 저장 — chroma_db와 동일 위치)
"""

import sqlite3
import os
from contextlib import contextmanager

DB_PATH = "data/mentions.db"


# ─────────────────────────────────────────────────
# DB 연결 컨텍스트 매니저
# ─────────────────────────────────────────────────

@contextmanager
def get_conn():
    """SQLite 연결을 with 블록으로 안전하게 사용."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # 결과를 dict처럼 접근 가능
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────
# 테이블 초기화
# ─────────────────────────────────────────────────

def init_db() -> None:
    """DB와 테이블을 초기화합니다. 이미 존재하면 무시.

    net_sentiment 컬럼이 없는 구버전 DB는 ALTER TABLE로 안전 마이그레이션.
    """
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS mentions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT    NOT NULL,
                date             TEXT    NOT NULL,
                source           TEXT    NOT NULL DEFAULT 'naver_news',
                title            TEXT    NOT NULL,
                sentiment        TEXT    NOT NULL DEFAULT 'neutral',
                sentiment_score  REAL    NOT NULL DEFAULT 0.5,
                crawled_at       TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_mentions_ticker_date
                ON mentions(ticker, date);

            CREATE TABLE IF NOT EXISTS daily_mention_stats (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT    NOT NULL,
                date             TEXT    NOT NULL,
                mention_count    INTEGER NOT NULL DEFAULT 0,
                positive_count   INTEGER NOT NULL DEFAULT 0,
                neutral_count    INTEGER NOT NULL DEFAULT 0,
                negative_count   INTEGER NOT NULL DEFAULT 0,
                positive_ratio   REAL    NOT NULL DEFAULT 0.0,
                neutral_ratio    REAL    NOT NULL DEFAULT 0.0,
                negative_ratio   REAL    NOT NULL DEFAULT 0.0,
                net_sentiment    REAL    NOT NULL DEFAULT 0.0,
                UNIQUE(ticker, date)
            );

            CREATE INDEX IF NOT EXISTS idx_stats_ticker_date
                ON daily_mention_stats(ticker, date);
        """)

        # 구버전 DB 마이그레이션: net_sentiment 컬럼이 없으면 추가
        try:
            conn.execute(
                "ALTER TABLE daily_mention_stats "
                "ADD COLUMN net_sentiment REAL NOT NULL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass  # 이미 존재
    print(f"[mention_db] DB 초기화 완료: {DB_PATH}")


# ─────────────────────────────────────────────────
# mentions CRUD
# ─────────────────────────────────────────────────

def insert_mention(
    ticker: str,
    date: str,
    title: str,
    sentiment: str,
    sentiment_score: float,
    source: str = "naver_news",
    crawled_at: str = "",
) -> None:
    """기사 1건을 mentions 테이블에 삽입."""
    from datetime import datetime
    if not crawled_at:
        crawled_at = datetime.now().isoformat()

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO mentions (ticker, date, source, title, sentiment, sentiment_score, crawled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, date, source, title, sentiment, sentiment_score, crawled_at),
        )


def insert_mentions_batch(rows: list[dict]) -> int:
    """
    기사 여러 건을 한 번에 삽입 (성능 최적화).

    rows 형식:
    [{"ticker": "005930", "date": "2026-05-11", "title": "...",
      "sentiment": "positive", "sentiment_score": 0.8, "source": "naver_news"}, ...]
    """
    from datetime import datetime
    crawled_at = datetime.now().isoformat()

    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO mentions (ticker, date, source, title, sentiment, sentiment_score, crawled_at)
            VALUES (:ticker, :date, :source, :title, :sentiment, :sentiment_score, :crawled_at)
            """,
            [{**row, "crawled_at": crawled_at} for row in rows],
        )
    return len(rows)


def get_mentions(ticker: str, date: str) -> list[dict]:
    """특정 종목의 특정 날짜 언급 목록 조회."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM mentions WHERE ticker = ? AND date = ? ORDER BY crawled_at",
            (ticker, date),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_mentions(ticker: str, start: str, end: str, limit: int = 30) -> list[dict]:
    """종목의 [start, end] 발행일 구간 헤드라인을 최신순으로.

    sentiment_analyst가 "오늘 이 종목에 무슨 일이 났나"를 읽는 입력이다
    (doc/2026-09-21_agent-audit.md §1-6). `date`는 v2부터 기사 **발행일**이다.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date, title, sentiment, sentiment_score, source FROM mentions "
            "WHERE ticker = ? AND date BETWEEN ? AND ? "
            "ORDER BY date DESC, id DESC LIMIT ?",
            (ticker, start, end, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def is_already_crawled(ticker: str, date: str) -> bool:
    """오늘 이미 크롤했는가 — `crawled_at`의 날짜로 판단한다.

    [v2] `date` 컬럼은 이제 기사 **발행일**이라 "오늘 크롤했나"의 근거로 쓸 수 없다.
    """
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM mentions WHERE ticker = ? AND substr(crawled_at, 1, 10) = ?",
            (ticker, date),
        ).fetchone()[0]
    return count > 0


def existing_titles(ticker: str) -> set[str]:
    """이 종목으로 이미 저장된 기사 제목 집합 — 다음날 크롤에 다시 오는 기사를 거른다."""
    with get_conn() as conn:
        return {r[0] for r in conn.execute(
            "SELECT DISTINCT title FROM mentions WHERE ticker = ?", (ticker,))}


def recount_daily_stats(ticker: str, dates: list[str]) -> None:
    """`mentions`를 발행일별로 다시 세어 `daily_mention_stats`에 upsert. 기사 0건인 날은 행을 만들지 않는다."""
    for d in dates:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), "
                "SUM(sentiment = 'positive'), SUM(sentiment = 'neutral'), SUM(sentiment = 'negative') "
                "FROM mentions WHERE ticker = ? AND date = ?",
                (ticker, d),
            ).fetchone()
        total = int(row[0] or 0)
        if total == 0:
            continue
        upsert_daily_stats(ticker, d, total, int(row[1] or 0), int(row[2] or 0), int(row[3] or 0))


def get_daily_counts(ticker: str, start: str, end: str) -> dict[str, int]:
    """[start, end] 안의 {발행일: 기사 수}. 행이 없는 날은 키가 없다(호출자가 0으로 본다)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date, mention_count FROM daily_mention_stats "
            "WHERE ticker = ? AND date BETWEEN ? AND ?",
            (ticker, start, end),
        ).fetchall()
    return {r["date"]: int(r["mention_count"]) for r in rows}


def get_net_sentiment_series(ticker: str, n: int = 6) -> list[float]:
    """최근 n일 net_sentiment, 날짜 오름차순(오래된→최신)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT net_sentiment FROM daily_mention_stats WHERE ticker = ? ORDER BY date DESC LIMIT ?",
            (ticker, n),
        ).fetchall()
    return [float(r["net_sentiment"]) for r in reversed(rows)]


# ─────────────────────────────────────────────────
# daily_mention_stats CRUD
# ─────────────────────────────────────────────────

def upsert_daily_stats(
    ticker: str,
    date: str,
    mention_count: int,
    positive_count: int,
    neutral_count: int,
    negative_count: int,
) -> None:
    """
    일별 통계를 삽입하거나 갱신 (INSERT OR REPLACE).
    비율은 자동 계산.
    """
    total = mention_count or 1  # 0 나눗셈 방지
    positive_ratio = round(positive_count / total, 4)
    neutral_ratio  = round(neutral_count  / total, 4)
    negative_ratio = round(negative_count / total, 4)
    net_sentiment  = round((positive_count - negative_count) / max(mention_count, 1), 4)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO daily_mention_stats
                (ticker, date, mention_count, positive_count, neutral_count, negative_count,
                 positive_ratio, neutral_ratio, negative_ratio, net_sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                mention_count  = excluded.mention_count,
                positive_count = excluded.positive_count,
                neutral_count  = excluded.neutral_count,
                negative_count = excluded.negative_count,
                positive_ratio = excluded.positive_ratio,
                neutral_ratio  = excluded.neutral_ratio,
                negative_ratio = excluded.negative_ratio,
                net_sentiment  = excluded.net_sentiment
            """,
            (ticker, date, mention_count, positive_count, neutral_count, negative_count,
             positive_ratio, neutral_ratio, negative_ratio, net_sentiment),
        )


def get_daily_stats(ticker: str, days: int = 7) -> list[dict]:
    """
    특정 종목의 최근 N일 통계 조회 (날짜 오름차순).

    반환 예시:
    [{"ticker": "005930", "date": "2026-05-05", "mention_count": 120, ...}, ...]
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM daily_mention_stats
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, days),
        ).fetchall()
    # 날짜 오름차순으로 뒤집기 (최신이 마지막)
    return [dict(r) for r in reversed(rows)]


def get_mention_window_stats(
    ticker: str,
    recent_days: int = 3,
    baseline_days: int = 14,
) -> dict | None:
    """
    스크리닝용 윈도우 통계.

    가장 최근 recent_days를 'recent', 그 이전 baseline_days를 'baseline'으로 분리하여
    spike_ratio (recent_avg / baseline_avg)와 일별 시계열을 반환한다.

    Returns:
        {
            "recent_avg":            float,
            "baseline_avg":          float,
            "spike_ratio":           float | None,   # baseline_avg == 0 → None
            "total_days":            int,
            "mention_count_series":  [int, ...],     # 오름차순 (오래된→최신)
            "net_sentiment_series":  [float, ...],
            "negative_ratio_series": [float, ...],
            "neutral_ratio_series":  [float, ...],
        }
        데이터가 (recent_days + 2)일 미만이면 None.
    """
    total_window = recent_days + baseline_days
    min_required = recent_days + 2

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT date, mention_count, net_sentiment, negative_ratio, neutral_ratio
            FROM daily_mention_stats
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, total_window),
        ).fetchall()

    if len(rows) < min_required:
        return None

    # 최신이 먼저였으니 오름차순으로 뒤집어 시계열로 만든다
    asc = list(reversed([dict(r) for r in rows]))

    mention_count_series  = [int(r["mention_count"])     for r in asc]
    net_sentiment_series  = [float(r["net_sentiment"])   for r in asc]
    negative_ratio_series = [float(r["negative_ratio"])  for r in asc]
    neutral_ratio_series  = [float(r["neutral_ratio"])   for r in asc]

    # recent 구간 = 마지막 recent_days, baseline 구간 = 그 이전
    recent_slice   = mention_count_series[-recent_days:]
    baseline_slice = mention_count_series[:-recent_days]  # 이전 전부 (DB에 baseline_days보다 적게 있어도 OK)

    recent_avg   = sum(recent_slice)   / len(recent_slice)   if recent_slice   else 0.0
    baseline_avg = sum(baseline_slice) / len(baseline_slice) if baseline_slice else 0.0

    spike_ratio = round(recent_avg / baseline_avg, 4) if baseline_avg > 0 else None

    return {
        "recent_avg":            round(recent_avg, 2),
        "baseline_avg":          round(baseline_avg, 2),
        "spike_ratio":           spike_ratio,
        "total_days":            len(asc),
        "mention_count_series":  mention_count_series,
        "net_sentiment_series":  net_sentiment_series,
        "negative_ratio_series": negative_ratio_series,
        "neutral_ratio_series":  neutral_ratio_series,
    }


def get_mention_change_rate(ticker: str, compare_days: int = 5) -> float | None:
    """
    오늘 언급량 vs N일 전 언급량 변화율 계산.

    반환값: 변화율 (예: 1.83 = +183%), None이면 데이터 부족
    """
    stats = get_daily_stats(ticker, days=compare_days + 1)
    if len(stats) < 2:
        return None

    today_count = stats[-1]["mention_count"]
    past_count  = stats[0]["mention_count"]

    if past_count == 0:
        return None

    return round((today_count - past_count) / past_count, 4)
