"""
tests/test_screen_context.py

단계 2 — 스크리닝 사유를 분석 단계로 전달한다 (doc/2026-09-21_agent-audit.md §2-2).

[왜]
  스크리닝 v2는 "오늘 새 정보(뉴스 급증·공시)가 생긴 종목"을 고른다. 그런데
  run_pipeline에는 ticker 문자열 하나만 넘어갔다. 8명 중 누구도 "무슨 일이 났는지"를
  읽지 못했고, 리포트를 읽는 사람도 왜 이 종목이 올라왔는지 알 수 없었다.

  공개 가격·거래량 지표에 우위가 없다는 것은 이미 두 문서가 결론냈다
  (2026-09-20_screening-audit §3-1, 2026-09-15_verified-returns).
  남은 가설은 "이벤트를 일관되게 해석한다"인데 그 해석을 하는 코드가 없었다.

유료 API 호출 없음.
"""

from __future__ import annotations

import sqlite3

import pytest


# ── 스크리닝 산출물 → 종목별 사유 ───────────────────────────────────────

def _screen(**over):
    """screener.run_screening()이 내는 모양의 최소 사본."""
    base = {
        "confirmed": ["028260"],
        "scores": {
            "028260": {
                "score": 1.0, "news_burst": 2.4, "news_burst_pct": 0.93, "dart_event": 1,
                "dart_titles": ["단일판매ㆍ공급계약체결", "[기재정정]단일판매ㆍ공급계약체결"],
                "news_match": 1, "sent_delta": 0.12, "ret5": -0.0436,
                "is_candidate": 1, "rank": 1, "filtered_reason": None,
            },
            "047040": {
                "score": 0.5, "news_burst": None, "news_burst_pct": None, "dart_event": 1,
                "dart_titles": ["단일판매ㆍ공급계약체결"], "news_match": 0, "sent_delta": None,
                "ret5": 0.01, "is_candidate": 1, "rank": 2, "filtered_reason": None,
            },
        },
        "stage_results": {"news": {"028260": {"news_match": True, "reason": "건설 수주 확대 기대"}}},
    }
    base.update(over)
    return base


def test_build_context_pulls_factors_and_llm_reason():
    from src.screening.screen_context import build_screen_context

    ctx = build_screen_context("028260", _screen())

    assert ctx["ticker"] == "028260"
    assert ctx["rank"] == 1
    assert ctx["dart_event"] == 1
    assert ctx["dart_titles"][0] == "단일판매ㆍ공급계약체결"
    assert ctx["news_burst"] == 2.4
    assert ctx["news_reason"] == "건설 수주 확대 기대"


def test_build_context_of_unknown_ticker_is_empty_not_error():
    """단독 디버그 실행(스크리닝 없음)에서도 파이프라인은 돌아야 한다."""
    from src.screening.screen_context import build_screen_context

    ctx = build_screen_context("005930", _screen())
    assert ctx["dart_event"] == 0 and ctx["dart_titles"] == [] and ctx["news_burst"] is None

    assert build_screen_context("005930", None)["ticker"] == "005930"


# ── 사람이 읽는 한 줄 ───────────────────────────────────────────────────

def test_selection_reason_names_the_event():
    from src.screening.screen_context import build_screen_context, selection_reason

    line = selection_reason(build_screen_context("028260", _screen()))
    assert "공시" in line and "단일판매" in line
    assert "뉴스" in line and "2.4배" in line


def test_selection_reason_with_only_disclosure():
    from src.screening.screen_context import build_screen_context, selection_reason

    line = selection_reason(build_screen_context("047040", _screen()))
    assert "공시" in line
    assert "배" not in line, "news_burst가 None이면 급증 문구를 쓰지 않는다"


def test_selection_reason_without_screening_says_so():
    from src.screening.screen_context import selection_reason

    assert "없음" in selection_reason({})
    assert "없음" in selection_reason(None)


# ── 그래프 상태 전달 ────────────────────────────────────────────────────

def test_graph_state_carries_screen_context():
    from src.schemas.graph_state import GraphState

    s = GraphState(ticker="028260", screen_context={"dart_event": 1})
    assert s.screen_context["dart_event"] == 1
    assert GraphState().screen_context == {}, "기본값은 빈 dict — 단독 실행도 동작"


@pytest.mark.asyncio
async def test_run_pipeline_passes_screen_context_into_state(monkeypatch):
    """run_pipeline(screen_context=...)가 초기 state에 실린다."""
    from src.graph import pipeline as pp

    captured: dict = {}

    class _FakeGraph:
        async def ainvoke(self, state):
            captured.update(state)
            return {}

    monkeypatch.setattr(pp, "build_pipeline", lambda publish=True: _FakeGraph())

    await pp.run_pipeline("028260", publish=False, screen_context={"dart_event": 1, "rank": 1})

    assert captured["screen_context"] == {"dart_event": 1, "rank": 1}


# ── 종목 헤드라인 조회 ──────────────────────────────────────────────────

@pytest.fixture
def mention_db(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    from src.data import mention_db as md
    monkeypatch.setattr(md, "DB_PATH", path)
    md.init_db()
    rows = [
        ("028260", "2026-09-18", "삼성물산 3조 수주", "positive", 0.9),
        ("028260", "2026-09-19", "건설 업황 회복", "positive", 0.8),
        ("028260", "2026-09-10", "오래된 기사", "neutral", 0.5),   # 창 밖
        ("047040", "2026-09-19", "다른 종목", "neutral", 0.5),
    ]
    with sqlite3.connect(path) as conn:
        for t, d, title, s, sc in rows:
            conn.execute(
                "INSERT INTO mentions (ticker,date,source,title,sentiment,sentiment_score,crawled_at) "
                "VALUES (?,?,'naver_news',?,?,?,'t')", (t, d, title, s, sc))
        conn.commit()
    return path


def test_get_recent_mentions_filters_by_window_and_ticker(mention_db):
    from src.data.mention_db import get_recent_mentions

    rows = get_recent_mentions("028260", "2026-09-17", "2026-09-19")

    titles = [r["title"] for r in rows]
    assert "삼성물산 3조 수주" in titles and "건설 업황 회복" in titles
    assert "오래된 기사" not in titles and "다른 종목" not in titles
    assert rows[0]["date"] >= rows[-1]["date"], "최신 기사가 먼저"
    assert rows[0]["sentiment"] in ("positive", "neutral", "negative")


def test_get_recent_mentions_respects_limit(mention_db):
    from src.data.mention_db import get_recent_mentions

    assert len(get_recent_mentions("028260", "2026-09-01", "2026-09-30", limit=1)) == 1


def test_get_recent_mentions_empty_is_not_error(mention_db):
    from src.data.mention_db import get_recent_mentions

    assert get_recent_mentions("999999", "2026-09-01", "2026-09-30") == []


# ── 에이전트 입력 (LLM 호출 없음 — 프롬프트 조립만 검사) ────────────────

def _ctx(**over):
    from src.screening.screen_context import build_screen_context
    ctx = build_screen_context("028260", _screen())
    ctx.update(over)
    return ctx


def test_agent_prompt_carries_event_and_price_context():
    from src.agents.sentiment_analyst import _format_prompt

    data = {"headlines": [
        {"date": "2026-09-19", "title": "삼성물산 3조 수주", "sentiment": "positive", "sentiment_score": 0.9},
        {"date": "2026-09-18", "title": "건설 업황 회복", "sentiment": "neutral", "sentiment_score": 0.5},
    ], "window": "2026-09-18~2026-09-19"}

    p = _format_prompt("028260", data, _ctx())

    assert "선정된 사유" in p and "단일판매" in p
    assert "삼성물산 3조 수주" in p, "종목 헤드라인 원문이 들어간다"
    assert "-4.36%" in p, "5일 수익률이 '이미 반영됐는가' 대조값으로 들어간다"
    assert "긍정 1건" in p and "중립 1건" in p


def test_agent_prompt_with_no_headlines_forbids_fabrication():
    from src.agents.sentiment_analyst import _format_prompt

    p = _format_prompt("005930", {"headlines": [], "window": "x"}, _ctx(dart_titles=[], dart_event=0,
                                                                        news_burst=None, news_burst_pct=None))
    assert "0건" in p
    assert "지어내지" in p and "data_sufficient=false" in p


def test_agent_prompt_never_mentions_market_wide_categories():
    """v2가 읽던 시장 전체 헤드라인(economy/finance/industry)은 더 이상 입력이 아니다."""
    from src.agents import sentiment_analyst as sa

    src = sa.SENTIMENT_SYSTEM_PROMPT
    assert "종목" in src
    assert "search_news" not in src
    assert not hasattr(sa, "NEWS_CATEGORIES"), "시장 카테고리 수집 경로가 남아 있으면 안 된다"


# ── chief 프롬프트 주입 ─────────────────────────────────────────────────

def test_chief_prompt_shows_selection_reason():
    from src.agents.chief_strategist import _format_reports_as_prompt
    from src.schemas.agent_output import AnalysisReport

    r = AnalysisReport(agent_name="technical_analyst", recommendation="HOLD", confidence=0.7,
                       reasoning=["a", "b", "c"], data_sources=["x", "y"],
                       prediction_basis=["p1", "p2"], risk_factors=["r"])

    with_ctx = _format_reports_as_prompt([r], "bull", "", screen_context=_ctx())
    without  = _format_reports_as_prompt([r], "bull", "")

    assert "선정된 사유" in with_ctx and "단일판매" in with_ctx
    assert "선정된 사유" not in without, "스크리닝 정보가 없으면 섹션 자체가 빠진다"
