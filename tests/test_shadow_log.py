"""
tests/test_shadow_log.py

F-6 섀도 로그 — 승격 규칙(마감 자동 OFF), 기록, 비교, Stage 1-C 연결.
유료 API 호출 없음 (Anthropic 클라이언트는 가짜로 대체).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import shadow_log as sl


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "mentions.db"
    monkeypatch.setattr(sl, "DB_PATH", path)
    sl.init_shadow_table(path)
    return path


# ── 승격 규칙 ─────────────────────────────────────────────

def test_policies_have_required_fields():
    for name, p in sl.SHADOW_POLICIES.items():
        for k in ("started", "review_by", "min_samples", "baseline", "candidate", "promotion"):
            assert k in p, f"{name}.{k}"
        assert date.fromisoformat(p["review_by"]) > date.fromisoformat(p["started"])


def test_shadow_active_respects_deadline(monkeypatch):
    monkeypatch.setenv("SHADOW_MODE", "shadow")
    p = sl.SHADOW_POLICIES["stage1c_model"]
    assert sl.shadow_active("stage1c_model", today=date.fromisoformat(p["review_by"]))
    assert not sl.shadow_active("stage1c_model", today=date.fromisoformat(p["review_by"]).replace(day=27))


def test_shadow_active_off_mode_and_unknown(monkeypatch):
    monkeypatch.setenv("SHADOW_MODE", "off")
    assert not sl.shadow_active("stage1c_model", today=date(2026, 9, 12))
    monkeypatch.setenv("SHADOW_MODE", "shadow")
    assert not sl.shadow_active("no_such_experiment", today=date(2026, 9, 12))
    monkeypatch.setenv("SHADOW_MODE", "garbage")
    assert sl.shadow_mode() == "off", "모르는 값은 비용 안전 쪽(off)"


# ── 기록·비교 ─────────────────────────────────────────────

def _pair(db, run_date, base_sel, cand_sel, h="h1"):
    sl.log_shadow("stage1c_model", "baseline", "claude-haiku-4-5-20251001", h,
                  {"selected_tickers": base_sel, "direct_ratio": 1.0}, 3000, 200, 800, run_date=run_date, db_path=db)
    sl.log_shadow("stage1c_model", "candidate", "claude-sonnet-5", h,
                  {"selected_tickers": cand_sel, "direct_ratio": 0.5}, 3000, 300, 2500, run_date=run_date, db_path=db)


def test_log_and_compare(db):
    _pair(db, "2026-09-11", ["A", "B", "C"], ["A", "B", "D"])   # 자카드 2/4
    _pair(db, "2026-09-12", ["A"], ["A"], h="h2")               # 자카드 1

    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM llm_ab_log").fetchone()[0]
    assert n == 4

    res = sl.compare("stage1c_model", db_path=db)
    assert res["pairs"] == 2
    assert res["agreement"] == pytest.approx((0.5 + 1.0) / 2)
    assert res["direct_ratio"]["baseline"] == 1.0 and res["direct_ratio"]["candidate"] == 0.5
    # 비용: haiku 1/5, sonnet-5 2/10 (공식 단가)
    assert res["cost"]["baseline"]  == pytest.approx(2 * (3000e-6 * 1 + 200e-6 * 5), abs=1e-6)
    assert res["cost"]["candidate"] == pytest.approx(2 * (3000e-6 * 2 + 300e-6 * 10), abs=1e-6)
    assert res["enough"] is False   # min_samples 10
    out = sl.format_compare(res)
    assert "표본 2일" in out and "표본 부족" in out


def test_compare_ignores_unpaired_rows(db):
    sl.log_shadow("stage1c_model", "baseline", "claude-haiku-4-5-20251001", "only-base",
                  {"selected_tickers": ["A"]}, 1, 1, 1, run_date="2026-09-11", db_path=db)
    assert sl.compare("stage1c_model", db_path=db)["pairs"] == 0


def test_log_shadow_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "DB_PATH", tmp_path / "nonexistent_dir" / "x.db")
    sl.log_shadow("stage1c_model", "baseline", "m", "h", {}, 0, 0, 0)   # 디렉토리 없음 → 경고만


# ── Stage 1-C 연결 ────────────────────────────────────────

class _FakeMessages:
    def __init__(self, calls):
        self._calls = calls

    def create(self, **kw):
        self._calls.append(kw["model"])
        sel = ["005930"] if "haiku" in kw["model"] else ["005930", "000660"]
        text = json.dumps({"selected_tickers": sel, "reasons": {t: "r" for t in sel}})
        return SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
        )


def _install_fake_anthropic(monkeypatch, calls):
    from src.screening import stage1c_news as st

    class _FakeAnthropic:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages(calls)

    monkeypatch.setattr(st, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return st


def test_stage1c_shadow_logs_both_and_uses_baseline(db, monkeypatch):
    monkeypatch.setenv("SHADOW_MODE", "shadow")
    calls: list[str] = []
    st = _install_fake_anthropic(monkeypatch, calls)
    monkeypatch.setattr(st, "fetch_market_headlines", lambda: ["삼성전자 실적 호조", "SK하이닉스 신고가"])
    monkeypatch.setattr(st, "_ticker_name_map", lambda ts: {"005930": "삼성전자", "000660": "SK하이닉스"})
    monkeypatch.setattr(st, "shadow_active", lambda name, today=None: True)

    out = st.run_news_screen(["005930", "000660"])

    # 2026-09-14: baseline → candidate → self_baseline(같은 모델 재호출) 순으로 3콜.
    assert calls == [st.NEWS_SCREEN_MODEL, "claude-sonnet-5", st.NEWS_SCREEN_MODEL]
    assert out["005930"]["news_match"] is True
    assert out["000660"]["news_match"] is False, "리포트는 baseline(haiku) 결과만 써야 한다"

    res = sl.compare("stage1c_model", db_path=db)
    assert res["pairs"] == 1
    assert res["agreement"] == pytest.approx(0.5)
    assert res["direct_ratio"]["baseline"] == 1.0 and res["direct_ratio"]["candidate"] == 1.0
    # 가짜 클라이언트는 모델별로 고정 응답을 주므로 self는 baseline과 같은 답 → 겹침 1.0
    assert res["self_agreement"] == pytest.approx(1.0)


def test_stage1c_no_candidate_call_when_inactive(db, monkeypatch):
    calls: list[str] = []
    st = _install_fake_anthropic(monkeypatch, calls)
    monkeypatch.setattr(st, "fetch_market_headlines", lambda: ["삼성전자 실적 호조"])
    monkeypatch.setattr(st, "_ticker_name_map", lambda ts: {"005930": "삼성전자"})
    monkeypatch.setattr(st, "shadow_active", lambda name, today=None: False)

    st.run_news_screen(["005930"])
    assert calls == [st.NEWS_SCREEN_MODEL]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM llm_ab_log").fetchone()[0] == 0


# ── 2026-09-14: 겹침률 관찰 (판단은 사람이 한다) ─────────────

def _seed_pair(db, node, run_date, ihash, base_sel, cand_sel):
    import sqlite3
    import json as _json
    with sqlite3.connect(db) as conn:
        for variant, model, sel in (("baseline", "m-base", base_sel), ("candidate", "m-cand", cand_sel)):
            conn.execute(
                "INSERT INTO llm_ab_log (run_date,node,variant,model,input_hash,output_json,"
                "input_tokens,output_tokens,latency_ms,cost_usd,created_at) VALUES (?,?,?,?,?,?,1,1,10,0.001,'t')",
                (run_date, node, variant, model, ihash, _json.dumps({"selected_tickers": sel})))
        conn.commit()


def test_agreement_by_day_is_returned(db):
    _seed_pair(db, "stage1c_model", "2026-09-13", "h1", ["A", "B"], ["A", "C"])
    _seed_pair(db, "stage1c_model", "2026-09-14", "h2", ["A", "B"], ["A", "B"])
    res = sl.compare("stage1c_model", db)
    days = dict(res["agreement_by_day"])
    assert days["2026-09-13"] == pytest.approx(1 / 3)     # {A}/{A,B,C}
    assert days["2026-09-14"] == pytest.approx(1.0)


def test_low_agreement_flag_only_when_persistent(db):
    """전 구간이 임계 이하일 때만 표시한다. 한 날만 낮으면 표시하지 않는다."""
    _seed_pair(db, "stage1c_model", "2026-09-13", "h1", ["A", "B", "C"], ["D", "E", "F"])
    _seed_pair(db, "stage1c_model", "2026-09-14", "h2", ["A", "B", "C"], ["D", "E", "G"])
    text = sl.format_compare(sl.compare("stage1c_model", db))
    assert "모델 의존성 높음" in text
    assert "일별 겹침률" in text and "2026-09-13" in text

    _seed_pair(db, "stage1c_model", "2026-09-15", "h3", ["A", "B"], ["A", "B"])
    text2 = sl.format_compare(sl.compare("stage1c_model", db))
    assert "모델 의존성 높음" not in text2, "한 날이라도 높으면 표시하지 않는다"


def test_low_agreement_needs_at_least_two_days(db):
    _seed_pair(db, "stage1c_model", "2026-09-13", "h1", ["A"], ["B"])
    assert "모델 의존성 높음" not in sl.format_compare(sl.compare("stage1c_model", db))


def test_threshold_matches_observation():
    assert sl.LOW_AGREEMENT_THRESHOLD == 0.3, "2026-09-14 관측 0.250·0.200 기준"


# ── CR-15: 표본은 거래일 단위 (2026-09-14) ─────────────────

def _seed(db, node, run_date, ihash, variant, sel, cost=0.001):
    import json as _json
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO llm_ab_log (run_date,node,variant,model,input_hash,output_json,"
            "input_tokens,output_tokens,latency_ms,cost_usd,created_at) VALUES (?,?,?,?,?,?,1,1,10,?,'t')",
            (run_date, node, variant, f"m-{variant}", ihash, _json.dumps({"selected_tickers": sel}), cost))
        conn.commit()


def test_same_day_multiple_runs_count_as_one(db):
    """같은 날 dispatch를 두 번 돌려도 표본은 1일이다."""
    _seed(db, "stage1c_model", "2026-09-13", "h1", "baseline", ["A", "B"])
    _seed(db, "stage1c_model", "2026-09-13", "h1", "candidate", ["C", "D"])
    _seed(db, "stage1c_model", "2026-09-13", "h2", "baseline", ["A", "B"])
    _seed(db, "stage1c_model", "2026-09-13", "h2", "candidate", ["A", "B"])
    res = sl.compare("stage1c_model", db)
    assert res["pairs"] == 1, "거래일 단위"
    assert res["days"] == ["2026-09-13"]


def test_same_day_keeps_last_run(db):
    """같은 날 여러 쌍이면 **마지막 실행**(id 최대)을 쓴다."""
    _seed(db, "stage1c_model", "2026-09-13", "h1", "baseline", ["A", "B"])
    _seed(db, "stage1c_model", "2026-09-13", "h1", "candidate", ["C", "D"])   # 겹침 0
    _seed(db, "stage1c_model", "2026-09-13", "h2", "baseline", ["A", "B"])
    _seed(db, "stage1c_model", "2026-09-13", "h2", "candidate", ["A", "B"])   # 겹침 1 (나중)
    assert sl.compare("stage1c_model", db)["agreement"] == pytest.approx(1.0)


def test_distinct_days_still_counted(db):
    _seed(db, "stage1c_model", "2026-09-13", "h1", "baseline", ["A"])
    _seed(db, "stage1c_model", "2026-09-13", "h1", "candidate", ["A"])
    _seed(db, "stage1c_model", "2026-09-14", "h2", "baseline", ["A"])
    _seed(db, "stage1c_model", "2026-09-14", "h2", "candidate", ["B"])
    res = sl.compare("stage1c_model", db)
    assert res["pairs"] == 2 and res["days"] == ["2026-09-13", "2026-09-14"]


def test_enough_uses_trading_days(db, monkeypatch):
    """min_samples가 실행 횟수가 아니라 거래일로 판정된다 (CR-15의 핵심)."""
    monkeypatch.setitem(sl.SHADOW_POLICIES["stage1c_model"], "min_samples", 2)
    for i in range(5):                                  # 같은 날 5번 실행
        _seed(db, "stage1c_model", "2026-09-13", f"h{i}", "baseline", ["A"])
        _seed(db, "stage1c_model", "2026-09-13", f"h{i}", "candidate", ["A"])
    assert sl.compare("stage1c_model", db)["enough"] is False, "5회 실행해도 1일"
    _seed(db, "stage1c_model", "2026-09-14", "x", "baseline", ["A"])
    _seed(db, "stage1c_model", "2026-09-14", "x", "candidate", ["A"])
    assert sl.compare("stage1c_model", db)["enough"] is True


# ── 자기 겹침률 기준선 ──────────────────────────────────────

def test_self_agreement_measured_against_baseline(db):
    """같은 모델 재호출과의 자카드가 기준선으로 기록된다."""
    _seed(db, "stage1c_model", "2026-09-13", "h1", "baseline", ["A", "B", "C", "D"])
    _seed(db, "stage1c_model", "2026-09-13", "h1", "candidate", ["A", "E", "F", "G"])
    _seed(db, "stage1c_model", "2026-09-13", "h1", sl.SELF_VARIANT, ["A", "B", "C", "E"], cost=0.006)
    res = sl.compare("stage1c_model", db)
    assert res["agreement"] == pytest.approx(1 / 7)           # {A} / 7개 합집합
    assert res["self_agreement"] == pytest.approx(3 / 5)      # {A,B,C} / {A,B,C,D,E}
    assert res["cost"][sl.SELF_VARIANT] == pytest.approx(0.006)
    assert res["self_agreement_by_day"] == [("2026-09-13", pytest.approx(3 / 5))]
    text = sl.format_compare(res)
    assert "자기 겹침률" in text and "비교 기준선" in text


def test_self_agreement_absent_when_not_recorded(db):
    _seed(db, "stage1c_model", "2026-09-13", "h1", "baseline", ["A"])
    _seed(db, "stage1c_model", "2026-09-13", "h1", "candidate", ["A"])
    res = sl.compare("stage1c_model", db)
    assert res["self_agreement"] is None
    assert "아직 기록 없음" in sl.format_compare(res)


def test_has_variant_on_guards_same_day_recall(db):
    """하루 1회 가드 — 같은 날 두 번째 호출은 건너뛴다."""
    assert sl.has_variant_on("stage1c_model", sl.SELF_VARIANT, "2026-09-13", db) is False
    _seed(db, "stage1c_model", "2026-09-13", "h1", sl.SELF_VARIANT, ["A"])
    assert sl.has_variant_on("stage1c_model", sl.SELF_VARIANT, "2026-09-13", db) is True
    assert sl.has_variant_on("stage1c_model", sl.SELF_VARIANT, "2026-09-14", db) is False
