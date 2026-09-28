"""
tests/test_stage0_fixes.py

이식 1회차 0단계 선행 수정 회귀 테스트 (doc/2026-09-10_benchmark-analysis.md 기준).

  1. 레거시 예측(ticker='')이 채점 대상에서 빠지고 evaluated=-1로 표시된다
  2. NAVER 뉴스 호출이 API HUB 엔드포인트·헤더를 쓴다
     (커밋 3205be2가 main에 병합되지 않아 두 파일이 옛 엔드포인트로 남아 있었다)
  3. llm_budget 단가표가 공식 단가다 (opus 15/75 → 5/25)
  4. nano 관련도 스코어링·임베딩 토큰이 예산에 기록된다

유료 API를 호출하지 않는다.
"""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from src.utils import llm_budget as lb


# ─────────────────────────────────────────────────────────
# 1. 레거시 예측 제외
# ─────────────────────────────────────────────────────────

@pytest.fixture
def pred_db(tmp_path, monkeypatch):
    from src.data import prediction_logger as pl

    monkeypatch.setattr(pl, "DB_PATH", tmp_path / "mentions.db")
    pl.init_feedback_tables()

    with sqlite3.connect(pl.DB_PATH) as conn:
        rows = [
            # (pred_date, ticker, agent, rec)
            ("2026-09-07", "",       "quant_analyst",     "BUY"),   # 레거시
            ("2026-09-07", "",       "technical_analyst", "HOLD"),  # 레거시
            ("2026-09-07", "005930", "quant_analyst",     "BUY"),   # 정상
        ]
        conn.executemany(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,"
            "confidence,regime,price_at_pred,evaluated,eval_score,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,0,NULL,'t')",
            [(d, t, "", a, r, 0.7, "Bull", 70000.0) for d, t, a, r in rows],
        )
        conn.commit()
    return pl


def test_unevaluated_excludes_empty_ticker(pred_db):
    rows = pred_db.get_unevaluated_predictions("2026-09-07")
    assert [r["ticker"] for r in rows] == ["005930"]


def test_retire_legacy_marks_minus_one_and_is_idempotent(pred_db):
    assert pred_db.retire_legacy_predictions() == 2
    assert pred_db.retire_legacy_predictions() == 0

    with sqlite3.connect(pred_db.DB_PATH) as conn:
        legacy = conn.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE ticker='' AND evaluated=?",
            (pred_db.EVALUATED_LEGACY,),
        ).fetchone()[0]
        scored = conn.execute("SELECT COUNT(*) FROM prediction_log WHERE evaluated=1").fetchone()[0]
    assert legacy == 2
    assert scored == 0, "레거시 행이 '채점됨(1)'으로 오인되면 안 된다"


def test_setup_feedback_system_retires_legacy(pred_db):
    pred_db.setup_feedback_system()
    with sqlite3.connect(pred_db.DB_PATH) as conn:
        pending = conn.execute("SELECT COUNT(*) FROM prediction_log WHERE evaluated=0").fetchone()[0]
    assert pending == 1


# ─────────────────────────────────────────────────────────
# 2. NAVER API HUB 엔드포인트
# ─────────────────────────────────────────────────────────

API_HUB_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"


def test_mention_tracker_uses_api_hub():
    from src.data import mention_tracker as mt

    assert mt.NAVER_NEWS_URL == API_HUB_URL
    src = inspect.getsource(mt._fetch_naver_news_inner)
    assert "X-NCP-APIGW-API-KEY-ID" in src and "X-NCP-APIGW-API-KEY" in src
    assert "X-Naver-Client-Id" not in src
    # API HUB는 JSON을 text/plain으로 내려보낸다 → mimetype 검사 해제가 없으면 파싱 실패
    assert "content_type=None" in src


def test_news_server_uses_api_hub():
    path = Path("src/mcp_servers/news_economy/server.py").read_text(encoding="utf-8")
    assert API_HUB_URL in path
    assert "openapi.naver.com" not in path
    assert "X-NCP-APIGW-API-KEY-ID" in path


# ─────────────────────────────────────────────────────────
# 3. 단가표
# ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_budget():
    lb.reset_budget(limit_usd=100.0)
    yield
    lb.reset_budget()


@pytest.mark.parametrize("model, price", [
    ("claude-opus-4-6",           (5.00, 25.00)),
    ("claude-opus-5",             (5.00, 25.00)),
    ("claude-sonnet-5",           (2.00, 10.00)),
    ("claude-sonnet-4-6",         (3.00, 15.00)),
    ("claude-haiku-4-5-20251001", (1.00,  5.00)),
    ("gpt-4o-mini",               (0.15,  0.60)),
    ("gpt-4.1-nano",              (0.10,  0.40)),
    ("gpt-5.6-luna",              (0.20,  1.20)),
    ("text-embedding-3-small",    (0.02,  0.00)),
])
def test_official_prices(model, price):
    assert lb._price_for(model) == price


def test_prefix_order_mini_before_4o():
    """'gpt-4o-mini'가 'gpt-4o'에 먼저 매칭돼 비싼 단가로 잡히면 안 된다."""
    assert lb._price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)


# ─────────────────────────────────────────────────────────
# 4. nano · 임베딩 usage 배선
# ─────────────────────────────────────────────────────────

def test_search_by_category_carries_llm_usage(monkeypatch):
    from src.mcp_servers.news_economy import server as ns

    monkeypatch.setattr(ns, "_fetch_category_rss",
                        lambda cat, n: [{"title": "t", "link": f"http://x/{cat}", "description": "", "pub_date": ""}])

    def fake_score(articles, min_score):
        ns._LAST_NANO_USAGE = {"model": "gpt-4.1-nano", "input_tokens": 120, "output_tokens": 8}
        for a in articles:
            a["relevance_score"] = 0.9
            a["scored"] = True
        return articles

    monkeypatch.setattr(ns, "_score_relevance_batch", fake_score)
    out = ns._search_by_category(["economy"], 5, min_score=0.5)
    assert out["llm_usage"] == {"model": "gpt-4.1-nano", "input_tokens": 120, "output_tokens": 8}


def test_search_by_category_usage_none_when_scoring_skipped(monkeypatch):
    from src.mcp_servers.news_economy import server as ns

    ns._LAST_NANO_USAGE = {"model": "gpt-4.1-nano", "input_tokens": 1, "output_tokens": 1}  # 이전 호출 잔재
    monkeypatch.setattr(ns, "_fetch_category_rss", lambda cat, n: [])
    monkeypatch.setattr(ns, "_score_relevance_batch", lambda articles, min_score: articles)
    out = ns._search_by_category(["economy"], 5)
    assert out["llm_usage"] is None, "이번 호출에서 nano가 안 돌았으면 잔재를 실어 보내면 안 된다"


def test_sentiment_no_longer_calls_news_mcp():
    """sentiment_analyst v3는 시장 뉴스 MCP를 부르지 않는다 (2026-09-21).

    [무엇이 바뀌었나] doc/2026-09-21_agent-audit.md §1-6
      v2는 `search_news(categories=...)`로 시장 전체 헤드라인 30건을 읽었고, 그 호출이
      MCP 서브프로세스에서 gpt-4.1-nano 관련성 스코어링을 돌렸다. 그래서 서버가
      결과에 llm_usage를 실어 보내고 에이전트가 예산에 기록했다(옛 테스트의 주제).

      v3는 종목 헤드라인을 mentions 테이블에서 직접 읽는다. 이 경로가 통째로 사라져
      nano 비용도 함께 사라졌다 — 집계에서 새는 것이 아니라 지출 자체가 없다.
      `search_news`를 부르는 에이전트는 이제 하나도 없다.

      서버가 llm_usage를 싣는 동작 자체는 위 test_search_by_category_*가 계속 지킨다.
    """
    import pathlib

    for p in pathlib.Path("src/agents").glob("*.py"):
        body = p.read_text(encoding="utf-8")
        # 주석·docstring에 이름이 남는 것은 허용하되 실제 호출은 없어야 한다
        assert 'call_tool("search_news"' not in body, f"{p.name}이 시장 뉴스 MCP를 호출한다"

    sentiment = pathlib.Path("src/agents/sentiment_analyst.py").read_text(encoding="utf-8")
    assert "mcp_client(" not in sentiment, "v3는 MCP 서브프로세스를 띄우지 않는다"
    assert "get_recent_mentions" in sentiment, "종목 헤드라인은 mentions 테이블에서 읽는다"


def test_embeddings_record_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from src.rag import chroma_store as cs

    monkeypatch.setattr(cs.OpenAIEmbeddings, "embed_query", lambda self, text, *a, **k: [0.0])
    monkeypatch.setattr(cs.OpenAIEmbeddings, "embed_documents", lambda self, texts, *a, **k: [[0.0] for _ in texts])

    emb = cs.get_embeddings()
    emb.embed_query("삼성전자 반도체 실적")
    emb.embed_documents(["a", "b"])

    s = lb.get_budget().snapshot()
    assert s["by_model"]["text-embedding-3-small"]["calls"] == 2
    assert s["input_tokens"] > 0
    assert s["output_tokens"] == 0
