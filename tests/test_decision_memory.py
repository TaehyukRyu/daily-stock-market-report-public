"""
chief 판단 문장 교훈 메모리 (2026-09-18).

TradingAgents memory.py/reflection.py 방식. chief의 판단을 기록하고 D+1 결과로 교훈을 만들어
다음 실행 chief 프롬프트에 주입한다. 외부 호출 없음 — sqlite 임시 파일 + 가짜 LLM.
"""
import asyncio
import sqlite3

import pytest

from src.data import decision_memory as dm


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(dm, "DB_PATH", path)
    dm.init_decision_table()
    return path


def _resolve(db, pred_date, ticker, outcome_date, score, lesson):
    """테스트용: 해당 행을 복기 완료 상태로 만든다."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE chief_decisions SET resolved=1, outcome_date=?, actual_price=71000.0, "
            "return_pct=1.4, score=?, lesson=? WHERE pred_date=? AND ticker=?",
            (outcome_date, score, lesson, pred_date, ticker),
        )


def test_store_decision_overwrites_until_resolved(db):
    """같은 (pred_date, ticker) 재실행 시 미채점 상태면 최신 판단으로 덮어쓴다(F3).
    채점(resolved=1) 후에는 발행된 리포트와 어긋나지 않도록 더는 덮어쓰지 않는다."""
    ok = dm.store_decision("2026-09-17", "005930", "삼성전자", "BUY", 0.72, "Bull",
                           ["근거1", "근거2"], 70000.0)
    assert ok is True

    overwritten = dm.store_decision("2026-09-17", "005930", "삼성전자", "SELL", 0.65, "Bull",
                                    ["근거3"], 70000.0)
    assert overwritten is True
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT recommendation, resolved FROM chief_decisions").fetchone()
    assert row == ("SELL", 0)

    _resolve(db, "2026-09-17", "005930", "2026-09-18", 1.0, "교훈")
    blocked = dm.store_decision("2026-09-17", "005930", "삼성전자", "HOLD", 0.5, "Bull",
                                ["근거4"], 70000.0)
    assert blocked is False
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT recommendation FROM chief_decisions").fetchone()
    assert row == ("SELL",)


def test_store_decision_rejects_no_opinion(db):
    """confidence<=0(의견 없음)은 기록 자체를 생략한다(F4)."""
    assert dm.store_decision("2026-09-17", "005930", "삼성전자", "HOLD", 0.0, "Bull",
                             ["의견 없음"], 70000.0) is False
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chief_decisions").fetchone()[0] == 0


def test_past_context_empty_when_nothing_resolved(db):
    dm.store_decision("2026-09-17", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], 70000.0)
    assert dm.get_past_context("005930", as_of="2026-09-18") == ""


def test_past_context_same_and_cross_ticker(db):
    dm.store_decision("2026-09-15", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], 70000.0)
    dm.store_decision("2026-09-16", "000660", "SK하이닉스", "SELL", 0.65, "Bull", ["근거"], 200000.0)
    _resolve(db, "2026-09-15", "005930", "2026-09-16", 0.0, "삼성 교훈")
    _resolve(db, "2026-09-16", "000660", "2026-09-17", 1.0, "하이닉스 교훈")

    ctx = dm.get_past_context("005930", as_of="2026-09-18")

    assert "[과거 판단 교훈" in ctx
    assert "같은 종목(005930)" in ctx
    assert "삼성 교훈" in ctx
    assert "오답" in ctx
    assert "다른 종목" in ctx
    assert "000660" in ctx and "하이닉스 교훈" in ctx and "적중" in ctx


def test_past_context_respects_as_of(db):
    """백테스트 시점 차단: outcome_date가 as_of보다 뒤면 안 보인다."""
    dm.store_decision("2026-09-15", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], 70000.0)
    _resolve(db, "2026-09-15", "005930", "2026-09-16", 1.0, "미래 교훈")
    assert dm.get_past_context("005930", as_of="2026-09-15") == ""
    assert "미래 교훈" in dm.get_past_context("005930", as_of="2026-09-16")


def test_past_context_limits_count(db):
    for i in range(5):
        d = f"2026-09-{10 + i:02d}"
        dm.store_decision(d, "005930", "삼성전자", "BUY", 0.7, "Bull", ["근거"], 70000.0)
        _resolve(db, d, "005930", f"2026-09-{11 + i:02d}", 1.0, f"교훈{i}")
    ctx = dm.get_past_context("005930", as_of="2026-09-30", same_n=2)
    assert "교훈4" in ctx and "교훈3" in ctx   # 최신 2개
    assert "교훈2" not in ctx


def test_resolve_pending_scores_and_stores_lesson(db):
    dm.store_decision("2026-09-17", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], 70000.0)
    dm.store_decision("2026-09-17", "000660", "SK하이닉스", "SELL", 0.60, "Bull", ["근거"], 200000.0)

    calls = []

    async def fake_reflect(row, return_pct, score):
        calls.append((row["ticker"], round(return_pct, 2), score))
        return f"{row['ticker']} 교훈"

    def fake_prices(tickers, yyyymmdd):
        return {"005930": 71400.0, "000660": 204000.0}   # +2.0% / +2.0%

    result = asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=fake_reflect, price_fn=fake_prices))

    assert result["resolved_count"] == 2
    assert sorted(calls) == [("000660", 2.0, 0.0), ("005930", 2.0, 1.0)]
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT ticker, resolved, score, lesson, outcome_date FROM chief_decisions ORDER BY ticker"
        ).fetchall()
    assert rows[0][:4] == ("000660", 1, 0.0, "000660 교훈")
    assert rows[1][:4] == ("005930", 1, 1.0, "005930 교훈")
    assert rows[0][4] is not None


def test_resolve_pending_skips_missing_price_and_keeps_pending(db):
    dm.store_decision("2026-09-17", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], 70000.0)

    async def fake_reflect(row, return_pct, score):
        raise AssertionError("종가 없는 종목은 복기 호출이 없어야 한다")

    result = asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=fake_reflect, price_fn=lambda t, d: {}))

    assert result["resolved_count"] == 0
    assert result["errors"] == ["005930: D+1 종가 없음"]
    assert dm.get_pending("2026-09-17")[0]["ticker"] == "005930"


def test_resolve_pending_reflect_failure_isolated(db):
    """LLM이 죽어도 채점 값(return_pct/score)은 저장한다. resolved는 "교훈 작성됨"을 뜻하므로
    실패 시 0으로 남아 다음 실행 때 재시도 대상이 된다(F2)."""
    dm.store_decision("2026-09-17", "005930", "삼성전자", "HOLD", 0.5, "Bull", ["근거"], 70000.0)

    async def broken_reflect(row, return_pct, score):
        raise RuntimeError("openai down")

    result = asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=broken_reflect,
                                            price_fn=lambda t, d: {"005930": 70100.0}))
    assert result["resolved_count"] == 0
    assert result["errors"] == ["005930: 복기 실패 — openai down"]
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT resolved, score, lesson FROM chief_decisions").fetchone()
    assert row == (0, 1.0, "")
    assert dm.get_pending("2026-09-17")[0]["ticker"] == "005930"
    # lesson이 빈 행은 프롬프트에 안 들어간다
    assert dm.get_past_context("005930", as_of="2026-09-30") == ""


def test_resolve_pending_backfills_missing_pred_price(db):
    """price_at_pred가 None이면 예측일 종가를 조회해 채우고 DB에도 반영한다.
    "+0.00%" 날조 없이 실제 채점값을 만든다(F1)."""
    dm.store_decision("2026-09-17", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], None)

    seen = []

    async def fake_reflect(row, return_pct, score):
        seen.append((round(return_pct, 2), score))
        return "교훈"

    def fake_prices(tickers, yyyymmdd):
        return {"005930": 70000.0} if yyyymmdd == "20260917" else {"005930": 71400.0}

    result = asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=fake_reflect, price_fn=fake_prices))
    assert result["resolved_count"] == 1
    assert seen == [(2.0, 1.0)]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT price_at_pred FROM chief_decisions").fetchone()[0] == 70000.0


def test_resolve_pending_skips_when_pred_price_unrecoverable(db):
    """예측일 종가를 끝내 못 구하면 복기 LLM을 부르지 않고(날조 방지) pending으로 남긴다(F1)."""
    dm.store_decision("2026-09-17", "005930", "삼성전자", "BUY", 0.72, "Bull", ["근거"], None)

    async def fake_reflect(row, return_pct, score):
        raise AssertionError("예측일 종가가 없으면 복기하면 안 된다")

    def fake_prices(tickers, yyyymmdd):
        return {} if yyyymmdd == "20260917" else {"005930": 71400.0}

    result = asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=fake_reflect, price_fn=fake_prices))
    assert result["resolved_count"] == 0
    assert result["errors"] == ["005930: 예측일 종가 없음 — 복기 생략"]
    assert dm.get_pending("2026-09-17")[0]["ticker"] == "005930"


def test_resolve_pending_retries_reflect_without_rescoring(db):
    """resolved=0(복기만 실패)으로 남은 행을 재실행하면, 이미 채점된 return_pct/score는 재사용하고
    가격 조회는 다시 하지 않는다(F2) — 이미 확정된 채점을 종가 재조회로 흔들지 않기 위함."""
    dm.store_decision("2026-09-17", "005930", "삼성전자", "HOLD", 0.5, "Bull", ["근거"], 70000.0)

    async def broken(row, return_pct, score):
        raise RuntimeError("down")

    asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=broken, price_fn=lambda t, d: {"005930": 70100.0}))

    seen = []

    async def ok(row, return_pct, score):
        seen.append((round(return_pct, 2), score))
        return "교훈"

    def no_prices(tickers, yyyymmdd):
        raise AssertionError("이미 채점된 행은 종가를 다시 조회하면 안 된다")

    result = asyncio.run(dm.resolve_pending("2026-09-17", reflect_fn=ok, price_fn=no_prices))
    assert result["resolved_count"] == 1
    assert seen == [(0.14, 1.0)]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT resolved, lesson FROM chief_decisions").fetchone() == (1, "교훈")


from src.agents.chief_strategist import _format_reports_as_prompt
from src.schemas.agent_output import AnalysisReport


def _report(name="quant_analyst"):
    return AnalysisReport(
        agent_name=name, recommendation="BUY", confidence=0.7,
        reasoning=["a", "b", "c"], data_sources=["x", "y"],
        prediction_basis=["p1", "p2"], risk_factors=["r"],
    )


def test_format_prompt_includes_past_context_before_votes():
    text = _format_reports_as_prompt([_report()], "bull", "", past_context="[과거 판단 교훈 — X]")
    assert "[과거 판단 교훈 — X]" in text
    assert text.index("[과거 판단 교훈") < text.index("[투표 집계")


def test_format_prompt_omits_section_when_empty():
    text = _format_reports_as_prompt([_report()], "bull", "", past_context="")
    assert "과거 판단 교훈" not in text
