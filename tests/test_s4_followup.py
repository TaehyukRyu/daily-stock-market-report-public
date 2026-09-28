"""
tests/test_s4_followup.py — S4 후속: forward 무작위 대조군 + 헤드라인 보관.

  1. random_control: 스냅샷 멱등, 100×5×3 등록, 날짜 시드 재현, outcomes와 같은 자로 확정, 백분위
  2. news_archive: 제목·시각 중복 무시, dt → ISO
  3. stage1c: 원본 메타(LAST_HEADLINE_ITEMS) 보존, 제목 리스트(판단 입력)는 그대로
외부 호출 없음.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from src.evaluation import outcomes as oc
from src.evaluation import random_control as rc
from src.evaluation import news_archive as na


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(oc, "DB_PATH", path)
    from src.data import prediction_logger as pl
    monkeypatch.setattr(pl, "DB_PATH", path)
    pl.init_feedback_tables()
    oc.init_outcome_tables(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ohlcv_cache (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)")
        days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"]
        # 6종목: A~F. 5거래일 뒤 수익률 A +5%, B +4%, C +3%, D +2%, E +1%, F 0%
        for i, t in enumerate(["A", "B", "C", "D", "E", "F"]):
            closes = [100 + (5 - i) * k / 5 for k in range(6)]
            conn.executemany("INSERT INTO ohlcv_cache VALUES (?,?,?,?,?,?,?)",
                             [(t, d, c, c, c, c, 1000) for d, c in zip(days, closes)])
        conn.commit()
    monkeypatch.setattr(oc, "_kospi_closes", lambda s, e: {"2026-09-01": 1000.0, "2026-09-08": 1020.0})
    return path


# ── 1. random_control ──────────────────────────────────────

def test_snapshot_idempotent(db):
    assert rc.record_universe_snapshot("2026-09-01", ["B", "A", "A"], db_path=db) == 2
    assert rc.record_universe_snapshot("2026-09-01", ["A", "B"], db_path=db) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM universe_snapshots").fetchone()[0] == 2


def test_register_count_and_seed_reproducible(db):
    pool = list("ABCDEF")
    n = rc.register_random_control("2026-09-01", pool, n_samples=100, n_picks=5, db_path=db)
    assert n == 100 * 5 * 3
    assert rc.register_random_control("2026-09-01", pool, n_samples=100, n_picks=5, db_path=db) == 0, "멱등"
    with sqlite3.connect(db) as conn:
        picks = conn.execute("SELECT sample_id, ticker FROM random_control_outcomes WHERE horizon_days=5 ORDER BY sample_id, ticker").fetchall()
        assert len({s for s, _ in picks}) == 100
        assert all(len([t for s2, t in picks if s2 == s]) == 5 for s in range(100))
    # 같은 날짜 → 같은 표본 (다른 DB에서도)
    import random
    a = random.Random(20260901).sample(sorted(pool), 5)
    assert sorted(a) == sorted(t for s, t in picks if s == 0)


def test_register_skips_small_universe(db):
    assert rc.register_random_control("2026-09-01", ["A", "B"], n_picks=5, db_path=db) == 0


def test_resolve_uses_same_ruler_as_outcomes(db):
    rc.register_random_control("2026-09-01", list("ABCDEF"), n_samples=3, n_picks=5, db_path=db)
    r = rc.resolve_pending(db)
    assert r["resolved"] == 3 * 5            # D+5만 확정 (봉 6개), D+10·20은 pending
    assert r["still_pending"] == 3 * 5 * 2
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT raw_return, kospi_return, alpha_vs_kospi, resolution_date FROM random_control_outcomes "
                           "WHERE status='resolved' AND ticker='A' LIMIT 1").fetchone()
    assert row is not None
    assert row[0] == pytest.approx(0.05) and row[1] == pytest.approx(0.02)
    assert row[2] == pytest.approx(0.03) and row[3] == "2026-09-08"


def test_summary_percentile_against_strategy(db):
    # 실제 예측: A(+5%, 알파 +3%) 하나 → 무작위 5종목 평균(최대 (5+4+3+2+1)/5=3% → 알파 +1%)보다 항상 위
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,confidence,regime,"
                     "price_at_pred,evaluated,eval_score,created_at) VALUES ('2026-09-01','A','a','quant_analyst','BUY',0.8,'Bull',100,0,NULL,'t')")
        conn.commit()
    oc.register_pending(db)
    oc.resolve_pending(db)
    rc.register_random_control("2026-09-01", list("ABCDEF"), n_samples=20, n_picks=5, db_path=db)
    rc.resolve_pending(db)
    s = rc.summary(5, db)
    assert s["n_days"] == 1 and s["n_samples"] == 20
    assert s["strategy_alpha"] == pytest.approx(0.03)
    assert s["strategy_percentile"] == 100.0
    assert s["random_p95"] <= 0.01 + 1e-9
    assert rc.summary(20, db)["n_days"] == 0
    assert "백분위 100%" in rc.format_summary(db)


# ── 2. news_archive ────────────────────────────────────────

def test_archive_dedup_and_iso(db):
    items = [
        {"title": "삼성전자 실적 상향", "published_at": "20260911150000", "source": "연합", "category": "mainnews"},
        {"title": "삼성전자 실적 상향", "published_at": "20260911150000", "source": "연합", "category": "mainnews"},  # 같은 제목·시각
        {"title": "삼성전자 실적 상향", "published_at": "20260911160000", "source": "연합", "category": "mainnews"},  # 시각 다름 → 별개
        {"title": "", "published_at": None},
    ]
    assert na.archive_headlines("2026-09-11", items, fetched_at="2026-09-11T04:10:00", db_path=db) == 2
    assert na.archive_headlines("2026-09-12", items, db_path=db) == 0
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT run_date, fetched_at, category, title, source, published_at FROM news_archive ORDER BY id").fetchall()
    assert rows[0] == ("2026-09-11", "2026-09-11T04:10:00", "mainnews", "삼성전자 실적 상향", "연합", "2026-09-11T15:00:00")
    assert na._iso("2026") == "2026" and na._iso(None) is None


# ── 3. stage1c 원본 메타 ───────────────────────────────────

NEWS_JSON = [
    {"tit": "외국인 순매수 1위 반전", "dt": "20260911154712", "ohnm": "한국경제"},
    {"tit": "삼성전자, 3분기 실적 컨센서스 상향", "dt": "20260911150000", "ohnm": "연합"},
    {"tit": "삼성전자, 3분기 실적 컨센서스 상향", "dt": "20260911150001", "ohnm": "연합"},   # 중복 제목
    {"tit": "짧음", "dt": "20260911"},
    "garbage",
]


def test_fetch_headlines_keeps_items_meta(monkeypatch):
    from src.screening import stage1c_news as st

    def fake_get(url, **kw):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: NEWS_JSON if "mainnews" in url else [])
    monkeypatch.setattr(st.requests, "get", fake_get)

    titles = st.fetch_market_headlines()
    assert titles == ["외국인 순매수 1위 반전", "삼성전자, 3분기 실적 컨센서스 상향"], "LLM 입력(제목)은 변화 없음"
    assert st.LAST_HEADLINE_COUNT == 2
    assert st.LAST_HEADLINE_ITEMS == [
        {"title": "외국인 순매수 1위 반전", "published_at": "20260911154712", "source": "한국경제", "category": "mainnews"},
        {"title": "삼성전자, 3분기 실적 컨센서스 상향", "published_at": "20260911150000", "source": "연합", "category": "mainnews"},
    ]


def test_fetch_headlines_failure_clears_items(monkeypatch):
    from src.screening import stage1c_news as st
    st.LAST_HEADLINE_ITEMS = [{"title": "stale"}]
    monkeypatch.setattr(st, "_fetch_headlines_api", lambda cat: [])
    assert st.fetch_market_headlines() == []
    assert st.LAST_HEADLINE_ITEMS == []
