"""
tests/test_dart_events.py

스크리닝 v2 — DART 공시 일괄 수집. HTTP는 목. 유료 호출 없음.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.data import mention_db as mdb
from src.data import dart_events as de


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(mdb, "DB_PATH", path)
    mdb.init_db()
    monkeypatch.setenv("DART_API_KEY", "test-key")
    return path


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self):
        return self._p


def _item(code, name, no, dt="20260921"):
    return {"stock_code": code, "corp_name": "X", "report_nm": name, "rcept_no": no, "rcept_dt": dt}


# ── 날짜 창 ────────────────────────────────────────────────

def test_event_window_monday_starts_friday():
    assert de.last_weekday_before(date(2026, 9, 21)) == date(2026, 9, 18)      # 월 → 금
    assert de.event_window(date(2026, 9, 21)) == ("20260918", "20260921")
    assert de.event_window(date(2026, 9, 22)) == ("20260921", "20260922")      # 화 → 월


# ── 분류 ──────────────────────────────────────────────────

@pytest.mark.parametrize("title, cat", [
    ("연결재무제표기준영업(잠정)실적(공정공시)", "earnings"),
    ("단일판매ㆍ공급계약체결", "contract"),
    ("주요사항보고서(자기주식취득결정)", "capital"),
    ("유상증자결정", "capital"),
    ("최대주주변경", "control"),
    ("회사합병결정", "control"),
    ("임원ㆍ주요주주특정증권등소유상황보고서", None),
    ("사업보고서 (2025.12)", None),
])
def test_classify(title, cat):
    assert de.classify(title) == cat


# ── 수집·필터·저장 ─────────────────────────────────────────

def test_fetch_paginates_until_total_page(monkeypatch):
    pages = {
        1: {"status": "000", "total_page": 2, "list": [_item("005930", "유상증자결정", "r1")]},
        2: {"status": "000", "total_page": 2, "list": [_item("000660", "단일판매ㆍ공급계약체결", "r2")]},
    }
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(params["page_no"]); return _Resp(pages[params["page_no"]])
    monkeypatch.setattr(de.requests, "get", fake_get)
    items = de.fetch_disclosures("20260918", "20260921", "k")
    assert [i["rcept_no"] for i in items] == ["r1", "r2"] and calls == [1, 2]


def test_fetch_stops_on_no_data_status(monkeypatch):
    monkeypatch.setattr(de.requests, "get", lambda url, params=None, timeout=None: _Resp({"status": "013", "message": "조회된 데이타가 없습니다."}))
    assert de.fetch_disclosures("20260918", "20260921", "k") == []


def test_filter_universe_keeps_classified_universe_items_only():
    items = [_item("005930", "유상증자결정", "r1"), _item("999999", "유상증자결정", "r2"),
             _item("005930", "사업보고서 (2025.12)", "r3"), {"stock_code": "", "report_nm": "유상증자결정", "rcept_no": "r4", "rcept_dt": "20260921"}]
    rows = de.filter_universe(items, {"005930", "000660"})
    assert [(r["ticker"], r["category"], r["rcept_no"]) for r in rows] == [("005930", "capital", "r1")]


def test_store_ignores_duplicate_rcept_no(db):
    de.init_dart_table()
    row = {"ticker": "005930", "rcept_dt": "20260921", "rcept_no": "r1", "title": "유상증자결정", "category": "capital"}
    assert de.store_events([row]) == 1
    assert de.store_events([row]) == 0
    assert de.get_events(["005930"], "20260918", "20260921") == {
        "005930": [{"date": "20260921", "title": "유상증자결정", "category": "capital"}]}


def test_collect_fetches_filters_stores_and_returns_window(db, monkeypatch):
    monkeypatch.setattr(de.requests, "get", lambda url, params=None, timeout=None: _Resp({
        "status": "000", "total_page": 1,
        "list": [_item("005930", "유상증자결정", "r1"), _item("000660", "사업보고서", "r2"),
                 _item("005380", "회사합병결정", "r3", dt="20260901")],           # 창 밖
    }))
    out = de.collect_dart_events(["005930", "000660", "005380"], today=date(2026, 9, 21))
    assert set(out) == {"005930"}
    assert out["005930"][0]["category"] == "capital"


def test_collect_returns_stored_events_when_fetch_fails(db, monkeypatch):
    de.init_dart_table()
    de.store_events([{"ticker": "005930", "rcept_dt": "20260921", "rcept_no": "r0", "title": "유상증자결정", "category": "capital"}])
    def boom(url, params=None, timeout=None): raise ConnectionError("down")
    monkeypatch.setattr(de.requests, "get", boom)
    out = de.collect_dart_events(["005930"], today=date(2026, 9, 21))
    assert out == {"005930": [{"date": "20260921", "title": "유상증자결정", "category": "capital"}]}


def test_collect_without_api_key_returns_empty(db, monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)
    assert de.collect_dart_events(["005930"], today=date(2026, 9, 21)) == {}


def test_fetch_raises_on_error_status(monkeypatch):
    """013(데이터 없음)만 정상 종료. 키 오류·한도 초과 등은 '공시 0건'으로 위장하지 않고 예외."""
    monkeypatch.setattr(de.requests, "get", lambda url, params=None, timeout=None: _Resp({"status": "020", "message": "요청 제한을 초과하였습니다."}))
    with pytest.raises(RuntimeError, match="020"):
        de.fetch_disclosures("20260918", "20260921", "k")


def test_collect_warns_on_error_status_and_returns_stored(db, monkeypatch, caplog):
    de.init_dart_table()
    de.store_events([{"ticker": "005930", "rcept_dt": "20260921", "rcept_no": "r0", "title": "유상증자결정", "category": "capital"}])
    monkeypatch.setattr(de.requests, "get", lambda url, params=None, timeout=None: _Resp({"status": "010", "message": "등록되지 않은 키입니다."}))
    with caplog.at_level("WARNING"):
        out = de.collect_dart_events(["005930"], today=date(2026, 9, 21))
    assert out == {"005930": [{"date": "20260921", "title": "유상증자결정", "category": "capital"}]}
    assert any("010" in rec.message for rec in caplog.records)
