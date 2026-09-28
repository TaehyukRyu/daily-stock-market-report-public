"""
src/evaluation/news_archive.py

Stage 1-C 헤드라인 일별 보관 (백테스트 전용).

왜: LLM 포함 백테스트의 첫 조건은 "발행시각이 있는 뉴스 원문 아카이브"인데(doc/2026-09-13_backtest-and-ops.md A-1),
지금은 헤드라인을 어디에도 저장하지 않는다. 오늘부터 매일 제목·수집시각·출처·기사시각을 남긴다.
mention_db 본체(mentions·daily_mention_stats, mention_tracker)와는 무관하며 파일만 같이 쓴다.
판단 로직에는 관여하지 않는다(기록 전용).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from src.evaluation import outcomes as oc


def init_table(db_path: Path | None = None) -> None:
    with sqlite3.connect(db_path or oc.DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS news_archive (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date     TEXT NOT NULL,        -- 실행일 (KST)
                fetched_at   TEXT NOT NULL,        -- 수집 시각 (ISO)
                category     TEXT,                 -- mainnews | flashnews
                title        TEXT NOT NULL,
                source       TEXT,                 -- 언론사 (ohnm)
                published_at TEXT,                 -- 기사 시각 (ISO, 있으면)
                UNIQUE (title, published_at)
            );
            CREATE INDEX IF NOT EXISTS idx_news_archive_run_date ON news_archive (run_date);
        """)
        conn.commit()


def _iso(dt: str | None) -> str | None:
    """네이버 dt 'YYYYMMDDHHMMSS' → ISO. 형식이 다르면 원문 그대로."""
    if not dt:
        return None
    s = str(dt)
    if len(s) == 14 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}:{s[12:14]}"
    return s


def archive_headlines(run_date: str, items: list[dict], fetched_at: str | None = None,
                      db_path: Path | None = None) -> int:
    """items: stage1c_news.LAST_HEADLINE_ITEMS 형식 [{title, published_at, source, category}].
    반환: 새로 들어간 행 수 (같은 제목·시각은 무시)."""
    if not items:
        return 0
    init_table(db_path)
    fetched_at = fetched_at or datetime.now().isoformat()
    rows = [(run_date, fetched_at, it.get("category"), it["title"], it.get("source"), _iso(it.get("published_at")))
            for it in items if it.get("title")]
    with sqlite3.connect(db_path or oc.DB_PATH) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO news_archive (run_date, fetched_at, category, title, source, published_at) "
            "VALUES (?,?,?,?,?,?)", rows)
        conn.commit()
        return conn.total_changes - before
