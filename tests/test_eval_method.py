"""
tests/test_eval_method.py — 2026-09-14 결정 반영분.

  CR-7  벤치마크를 069500.KS로 통일, 행마다 benchmark 열로 자기 기술
  CR-8  summary 표본 단위 = 하루 (결정 Q6-a)
  Q4    적중률 + 손익분기 승률 병기
  Q7    net_return (비용 차감) 열
  Q8    확정 0개인 날은 표본에서 빼고 일수만 따로 센다
  방법 변경 지점을 eval_method_log에 남긴다
외부 호출 없음.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.evaluation import outcomes as oc
from src.evaluation import random_control as rc


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
        # 이틀에 걸쳐 2종목씩. 하루 단위 집계가 종목 단위와 달라지도록 구성한다.
        #   09-01: A +10%, B -10%  → 하루 평균 0%
        #   09-02: C +2%           → 하루 평균 +2%
        # 종목 단위로 평균 내면 (+10 -10 +2)/3 = +0.67%, 하루 단위면 (0 + 2)/2 = +1%
        rows = [("A", 100, 110), ("B", 100, 90), ("C", 100, 102)]
        days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09"]
        # D+5 확정일: pred 09-01 → 09-08(days[5]), pred 09-02 → 09-09(days[6]).
        # 두 자리 모두 end가 되도록 앞 5개만 start로 둔다.
        for t, start, end in rows:
            closes = [start] * 5 + [end, end]
            conn.executemany("INSERT INTO ohlcv_cache VALUES (?,?,?,?,?,?,?)",
                             [(t, d, c, c, c, c, 1000) for d, c in zip(days, closes)])
        conn.executemany(
            "INSERT INTO prediction_log (pred_date,ticker,ticker_name,agent_name,recommendation,confidence,regime,"
            "price_at_pred,evaluated,eval_score,created_at) VALUES (?,?,?,?,'BUY',0.8,'Bull',100,0,NULL,'t')",
            [("2026-09-01", "A", "a", "quant_analyst"),
             ("2026-09-01", "B", "b", "quant_analyst"),
             ("2026-09-02", "C", "c", "quant_analyst")],
        )
        conn.commit()
    monkeypatch.setattr(oc, "_kospi_closes", lambda s, e: {})     # 알파 없이 raw만
    return path


def _resolve(db):
    oc.register_pending(db)
    oc.resolve_pending(db)


# ── CR-8 / Q6: 표본 단위는 하루 ────────────────────────────

def test_summary_sample_unit_is_day(db):
    _resolve(db)
    s = oc.summary(5, db)
    assert s["n_days"] == 2, "09-01, 09-02 두 날"
    assert s["n_predictions"] == 3, "행은 3개"
    # 하루 평균의 평균: (0% + 2%) / 2 = +1%. 종목 단위였다면 +0.67%
    assert s["avg_raw"] == pytest.approx(0.01, abs=1e-6)


def test_day_mean_helper():
    rows = [{"pred_date": "d1", "raw_return": 0.10}, {"pred_date": "d1", "raw_return": -0.10},
            {"pred_date": "d2", "raw_return": 0.02}, {"pred_date": "d2", "raw_return": None}]
    mean, n = oc._day_mean(rows, "raw_return")
    assert n == 2 and mean == pytest.approx(0.01)
    assert oc._day_mean([], "raw_return") == (None, 0)


# ── Q4: 적중률 + 손익분기 승률 ──────────────────────────────

def test_hit_rate_and_breakeven(db):
    _resolve(db)
    s = oc.summary(5, db)
    # D+5 문턱 1%: A +10% 적중, B -10% 패, C +2% 적중 → 2/3
    assert s["hit_rate"] == pytest.approx(2 / 3)
    # 평균 이익 (10+2)/2 = 6%, 평균 손실 10%, 왕복 비용 c
    # 손익분기 = (L̄ + c) / (W̄ + L̄)  — 백테스트 metrics.trade_stats와 같은 식
    c = oc.roundtrip_cost()
    assert s["breakeven_win_rate"] == pytest.approx((0.10 + c) / 0.16)


def test_breakeven_none_when_one_sided(db):
    """전부 이기거나 전부 지면 손익분기를 계산할 수 없다."""
    _resolve(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE prediction_outcomes SET raw_return=0.05 WHERE horizon_days=5")
        conn.commit()
    assert oc.summary(5, db)["breakeven_win_rate"] is None


# ── CR-7: 벤치마크 ──────────────────────────────────────────

def test_benchmark_recorded_per_row(db):
    _resolve(db)
    with sqlite3.connect(db) as conn:
        marks = {r[0] for r in conn.execute(
            "SELECT benchmark FROM prediction_outcomes WHERE status='resolved'")}
    assert marks == {"069500.KS"}
    assert oc.BENCHMARK == "069500.KS" and oc.LEGACY_BENCHMARK == "^KS11"


def test_legacy_rows_get_legacy_benchmark(db):
    """전환 이전에 확정된 행은 ^KS11로 표시된다 (마이그레이션)."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prediction_outcomes (pred_date,ticker,horizon_days,source,raw_return,"
            "status,created_at) VALUES ('2026-08-01','Z',5,'prediction',0.03,'resolved','t')")
        conn.execute("UPDATE prediction_outcomes SET benchmark=NULL WHERE ticker='Z'")
        conn.commit()
    oc.init_outcome_tables(db)                       # 재진입 시 마이그레이션이 돈다
    with sqlite3.connect(db) as conn:
        got = conn.execute("SELECT benchmark FROM prediction_outcomes WHERE ticker='Z'").fetchone()[0]
    assert got == "^KS11"


def test_random_control_uses_same_benchmark(db):
    rc.register_random_control("2026-09-01", list("ABC") + ["D", "E", "F"], n_samples=2, n_picks=3, db_path=db)
    rc.resolve_pending(db)
    with sqlite3.connect(db) as conn:
        marks = {r[0] for r in conn.execute(
            "SELECT benchmark FROM random_control_outcomes WHERE status='resolved'")}
    assert marks <= {"069500.KS"}


# ── Q8: 확정 0개인 날 ───────────────────────────────────────

def test_no_pick_days_counted_separately(db):
    rc.record_universe_snapshot("2026-09-01", ["A", "B", "C"], db_path=db)
    rc.record_universe_snapshot("2026-09-02", ["A", "B", "C"], db_path=db)
    rc.record_universe_snapshot("2026-09-03", ["A", "B", "C"], db_path=db)   # 확정 0개인 날
    oc.record_screen_candidates("2026-09-01", [{"ticker": "A", "is_candidate": 1}], analyzed=["A"], db_path=db)
    oc.record_screen_candidates("2026-09-02", [{"ticker": "C", "is_candidate": 1}], analyzed=["C"], db_path=db)
    assert oc.no_pick_days(db) == 1
    assert oc.summary(5, db)["no_pick_days"] == 1


# ── 방법 변경 기록 ──────────────────────────────────────────

def test_method_change_log_keeps_first_applied_date(db):
    oc.record_method_change("benchmark", "^KS11 → 069500.KS", applied_date="2026-09-14", db_path=db)
    oc.record_method_change("benchmark", "덮어쓰기 시도", applied_date="2026-10-01", db_path=db)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT applied_date, detail FROM eval_method_log WHERE change_key='benchmark'").fetchone()
    assert row[0] == "2026-09-14", "적용 시작일은 최초 1회만 남는다"
    assert "069500.KS" in row[1]


# ── Q7: 비용 차감 net_return ────────────────────────────────

def test_net_of_costs_formula():
    buy  = oc.COMMISSION + oc.EXCHANGE_FEE + oc.SLIPPAGE
    sell = buy + oc.TAX_BY_YEAR[2026]
    assert oc.net_of_costs(0.10, "2026-09-08") == pytest.approx(1.10 * (1 - sell) / (1 + buy) - 1)
    # 매도 연도별 거래세가 다르다 (2025 0.15% vs 2026 0.20%)
    assert oc.net_of_costs(0.0, "2025-06-01") > oc.net_of_costs(0.0, "2026-06-01")
    assert oc.net_of_costs(None, "2026-01-01") is None
    # 연도 미상이면 기본 세율
    assert oc.net_of_costs(0.0, None) == pytest.approx(oc.net_of_costs(0.0, "2026-01-01"))


def test_net_return_always_below_raw(db):
    _resolve(db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT raw_return, net_return FROM prediction_outcomes "
            "WHERE status='resolved' AND raw_return IS NOT NULL").fetchall()
    assert rows
    for raw, net in rows:
        assert net is not None and net < raw, "비용을 뺐으므로 항상 raw보다 작다"
    s = oc.summary(5, db)
    assert s["avg_net"] is not None and s["avg_net"] < s["avg_raw"]


def test_roundtrip_cost_matches_backtest_costmodel():
    """상수를 복제했으므로 백테스트 CostModel과 어긋나면 실패시킨다."""
    import pandas as pd
    from src.evaluation.backtest.engine import CostModel
    cm = CostModel()
    assert oc.COMMISSION   == cm.commission
    assert oc.EXCHANGE_FEE == cm.exchange_fee
    assert oc.SLIPPAGE     == cm.slippage
    assert oc.TAX_BY_YEAR  == cm.tax_by_year
    expected = cm.buy_cost() + cm.sell_cost(pd.Timestamp("2026-06-01"))
    assert oc.roundtrip_cost(2026) == pytest.approx(expected)


def test_csv_export_includes_new_columns(db, tmp_path):
    _resolve(db)
    oc.export_csv(db, out_dir=tmp_path / "ledger")
    head = (tmp_path / "ledger" / "outcomes.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "net_return" in head and "benchmark" in head


# ── CR-6 / CR-5: 손절 기본값 일치 · 보유 경과 알림 ──────────

def test_default_stop_and_target_match_chief_rule():
    """CR-6: position_tracker 기본값이 chief 규칙(-4%, R:R 2.0 → +8%)과 같아야 한다."""
    from src.data import position_tracker as pt
    from src.agents.chief_python import DEFAULT_STOP_PCT, compute_trade_params

    assert pt.STOP_LOSS_RATIO == pytest.approx(abs(DEFAULT_STOP_PCT) / 100)   # 0.04
    params = compute_trade_params(100_000.0)
    # chief 규칙으로 계산한 손절·1차 익절이 기본 비율과 같은 자리에 온다
    assert params["stop_loss"] == pytest.approx(100_000 * (1 - pt.STOP_LOSS_RATIO))
    assert params["take_profit_1"] == pytest.approx(100_000 * (1 + pt.DEFAULT_TARGET_RATIO))


def test_overdue_flag_does_not_trigger_auto_close():
    """CR-5: 경과 알림은 overdue 필드로만 나가고 alert(청산 판정)는 건드리지 않는다."""
    from src.data import position_tracker as pt
    import inspect

    src = inspect.getsource(pt.get_portfolio_pct_summary)
    assert '"overdue"' in src
    # 자동 청산 판정은 여전히 stop_loss / target 두 가지뿐
    close_src = inspect.getsource(pt.auto_close_triggered_positions)
    assert '("stop_loss", "target")' in close_src
    assert "overdue" not in close_src


def test_overdue_threshold_matches_d10():
    from src.data import position_tracker as pt
    assert pt.HOLDING_ALERT_DAYS == 14, "D+10 거래일 ≈ 14 달력일"


def test_report_marks_overdue_position():
    from src.graph.report_formatter import _portfolio_section

    def _pos(days):
        return {"ticker": "005930", "ticker_name": "삼성전자", "entry_date": "2026-09-01",
                "entry_price": 100.0, "current_price": 101.0, "allocation_pct": 5.0,
                "position_pnl_pct": 1.0, "stop_loss_price": 96.0, "take_profit_1": 108.0,
                "holding_days": days, "holding_period_weeks": 2, "alert": None,
                "overdue": days >= 14}

    old = "\n".join(_portfolio_section({"positions": [_pos(20)], "invested_pct": 5.0,
                                        "available_pct": 95.0, "cumulative_pnl_seed_pct": 0.0}))
    new = "\n".join(_portfolio_section({"positions": [_pos(3)], "invested_pct": 5.0,
                                        "available_pct": 95.0, "cumulative_pnl_seed_pct": 0.0}))
    assert "20일 ⏰" in old
    assert "⏰" not in new


# ── CR-10 ①⑦⑩⑪: 결정 무관 리포트 항목 ─────────────────────

def test_score_breakdown_lists_active_signals():
    from src.graph.report_formatter import score_breakdown
    text = score_breakdown({"total_score": 4, "mention_spike": True, "news_match": True,
                            "quant_score": 2, "quant_signal_count": 3, "price_change_5d": 0.052})
    assert text.startswith("4점 = ")
    assert "언급량 급증 +1" in text and "뉴스 매치 +1" in text and "정량신호 3개 +2" in text
    assert "5일 +5.2%" in text
    # 꺼진 항목은 안 나온다
    assert "감성 가속" not in text
    assert score_breakdown(None) == "점수 분해 없음"
    assert "가점 없음" in score_breakdown({"total_score": 0})


def test_version_line_has_code_and_model_and_mode():
    from src.graph.report_formatter import version_line
    v = version_line()
    assert "CHIEF_MODE=" in v and "chief=" in v and "섀도" in v


def test_upcoming_report_deadline_window():
    from datetime import date
    from src.graph.report_formatter import upcoming_report_deadline
    assert "3분기 보고서" in upcoming_report_deadline(date(2026, 11, 5))
    assert upcoming_report_deadline(date(2026, 9, 14)) is None      # 14일 내 마감 없음
    assert upcoming_report_deadline(date(2026, 11, 14)).startswith("2026-11-14")   # 당일 포함
    # 연말에는 다음 해 3월 마감을 보지 않는다 (14일 창 밖)
    assert upcoming_report_deadline(date(2026, 12, 20)) is None


def test_no_pick_reason_explains_with_numbers():
    from src.graph.report_formatter import no_pick_reason
    text = no_pick_reason(
        {"scores": {"A": {"total_score": 1}, "B": {"total_score": 1}, "C": {"total_score": 0}},
         "confirmed_min": 2}, "bear")
    assert "확정 기준 2점" in text and "최고 점수 1점" in text
    assert "1점 2개" in text and "0점 1개" in text
    # 점수 자체가 없으면 스크리닝 고장을 의심하라고 말한다
    assert "스크리닝이 정상 동작했는지" in no_pick_reason({}, "bull")


def test_combined_blocks_include_score_breakdown():
    from src.graph.notion_publisher import build_v4_blocks_combined
    blocks = build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": None,
                             "qualified_reports": [], "all_reports": []}],
        regime="neutral", strategy="HOLD",
        screen={"scores": {"005930": {"total_score": 3, "news_match": True,
                                      "quant_score": 2, "quant_signal_count": 3}}},
    )
    texts = []
    for b in blocks:
        for kind in ("paragraph", "callout"):
            if b.get("type") == kind:
                texts.append("".join(rt.get("text", {}).get("content", "")
                                     for rt in b[kind].get("rich_text", [])))
    # [2026-09-15] 종목 상세가 toggle 안으로 들어가 children까지 훑어야 한다
    for b in blocks:
        for child in (b.get(b.get("type")) or {}).get("children") or []:
            kind = child.get("type")
            node = child.get(kind) or {}
            if "rich_text" in node:
                texts.append("".join(rt.get("text", {}).get("content", "") for rt in node["rich_text"]))
    assert any("선정 근거 — 3점 =" in t for t in texts)


def test_disclaimer_callout_carries_version():
    """⑩ 버전 표기는 면책 블록에 들어간다 (publish_combined_to_notion이 덧붙인다)."""
    from src.graph.notion_publisher import _disclaimer_callout
    text = "".join(rt.get("text", {}).get("content", "")
                   for rt in _disclaimer_callout()["callout"]["rich_text"])
    assert "CHIEF_MODE=" in text and "투자 조언 아님" in text


def test_market_only_report_has_reason_and_version():
    """확정 0개인 날의 리포트에도 ① 사유와 ⑩ 버전이 들어간다."""
    from src.daily_runner import _format_market_only_report
    md = _format_market_only_report(
        regime="bear",
        market={"kospi": 2500.0},
        screen={"scores": {"A": {"total_score": 1}}, "confirmed_min": 3,
                "confirmed": [], "optional": []},
        portfolio=None,
        health_note="✅ 실행 정상 — 오늘 확정 종목 없음",
    )
    assert "신호 없음 사유 —" in md
    assert "확정 기준 3점" in md and "최고 점수 1점" in md
    assert "버전: 코드" in md and "CHIEF_MODE=" in md
    assert "실행 정상" in md


# ── CR-4: 되돌림 (2026-09-14) ───────────────────────────────

@pytest.mark.parametrize("module_name,const", [
    ("macro_economist", "MACRO_SYSTEM_PROMPT"),
    ("kr_market_specialist", "KR_MARKET_SYSTEM_PROMPT"),
    ("us_market_specialist", "US_MARKET_SYSTEM_PROMPT"),
    ("quant_analyst", "QUANT_SYSTEM_PROMPT"),
    ("technical_analyst", "TECHNICAL_SYSTEM_PROMPT"),
    ("sentiment_analyst", "SENTIMENT_SYSTEM_PROMPT"),
    ("fundamental_analyst", "FUNDAMENTAL_SYSTEM_PROMPT"),
])
def test_risk_factors_present_in_every_output_spec(module_name, const):
    """risk_factors는 7개 전부의 출력 명세에 있어야 한다.

    quant/technical/us_market에는 원래 **아예 없었다**. 그것은 위치와 무관한
    별개 결함이라 되돌리지 않고 남긴다.
    """
    import importlib
    prompt = getattr(importlib.import_module(f"src.agents.{module_name}"), const)
    assert "risk_factors" in prompt.split("[출력")[-1]


@pytest.mark.parametrize("module_name,const", [
    ("macro_economist", "MACRO_SYSTEM_PROMPT"),
    ("kr_market_specialist", "KR_MARKET_SYSTEM_PROMPT"),
    ("us_market_specialist", "US_MARKET_SYSTEM_PROMPT"),
    ("quant_analyst", "QUANT_SYSTEM_PROMPT"),
    ("technical_analyst", "TECHNICAL_SYSTEM_PROMPT"),
    ("sentiment_analyst", "SENTIMENT_SYSTEM_PROMPT"),
    ("fundamental_analyst", "FUNDAMENTAL_SYSTEM_PROMPT"),
])
def test_risk_factors_is_not_hoisted_to_front(module_name, const):
    """위치 이동은 되돌렸다 — 종목당 재요청이 1.54 → 3.67로 늘었기 때문.

    reasoning보다 뒤에 있어야 원래 순서다.
    """
    import importlib
    spec = getattr(importlib.import_module(f"src.agents.{module_name}"), const).split("[출력")[-1]
    assert "1. risk_factors" not in spec, "목록 1번으로 올린 변경이 남아 있다"
    assert spec.index("risk_factors") > spec.index("reasoning"), "reasoning 뒤에 와야 한다"


def test_schema_min_lengths_unchanged():
    """CR-4는 프롬프트만 건드렸다. 스키마 제약은 시종 그대로다."""
    from src.schemas.agent_output import AnalysisReport
    fields = AnalysisReport.model_fields
    assert fields["risk_factors"].metadata[0].min_length == 1
    assert fields["reasoning"].metadata[0].min_length == 1
    assert fields["selection_rationale"].default is None, "여전히 Optional"


# ── CR-10 ②③④⑧ (2026-09-14, Q2·Q4 결정 반영) ──────────────

def test_recent_performance_shows_shortfall_not_numbers(db, monkeypatch):
    """③ 표본이 모자라면 수치 대신 '표본 N일 (최소 M)'만 보여준다."""
    from src.graph import report_formatter as rf
    monkeypatch.setattr(oc, "DB_PATH", db)
    _resolve(db)                                   # 2일치만 확정된다
    lines = rf.recent_performance_lines()
    assert any("표본 2일 (최소 30일)" in ln for ln in lines)
    assert all("적중" not in ln for ln in lines), "표본 부족이면 적중률을 내보이지 않는다"


def test_recent_performance_shows_numbers_when_enough(monkeypatch):
    """③ 표본이 차면 평균·알파·비용차감·적중·손익분기를 낸다."""
    from src.graph import report_formatter as rf
    monkeypatch.setattr(oc, "MIN_DAYS_FOR_TREND", 1)
    monkeypatch.setattr(oc, "summary", lambda h, db_path=None: {
        "n_days": 40, "avg_raw": 0.012, "avg_alpha": 0.003, "avg_net": 0.008,
        "hit_rate": 0.48, "breakeven_win_rate": 0.47})
    monkeypatch.setattr(oc, "no_pick_days", lambda db_path=None: 3)
    lines = rf.recent_performance_lines()
    assert any("40일" in ln and "적중 48%" in ln and "손익분기 47%" in ln for ln in lines)
    assert any("확정 0개였던 날 **3일**" in ln for ln in lines), "Q8-a: 일수는 따로 센다"


def test_min_days_for_trend_is_distinct_from_promotion_bar():
    """리포트 표시선(30일)과 승격 판정선(175거래일)은 다른 값이다."""
    assert oc.MIN_DAYS_FOR_TREND == 30


def test_calibrated_note_marks_uncalibrated(db, monkeypatch):
    """④ 표본 미달이면 '미보정'이라고 못박는다 — 5% 고정이 확신도 탓으로 읽히면 안 된다."""
    from src.graph import report_formatter as rf
    from src.evaluation import calibration as cal
    monkeypatch.setattr(cal, "DB_PATH", db, raising=False)
    monkeypatch.setattr(cal, "calibrated_hit_rate",
                        lambda c, agent=None, table=None, db_path=None: (None, "insufficient"))
    note = rf.calibrated_confidence_note(0.72, "chief_strategist")
    assert "72%" in note and "미보정" in note and "자기보고" in note

    monkeypatch.setattr(cal, "calibrated_hit_rate",
                        lambda c, agent=None, table=None, db_path=None: (0.41, "all"))
    note2 = rf.calibrated_confidence_note(0.72, "chief_strategist")
    assert "실제 적중 41%" in note2 and "전체 기준" in note2
    assert rf.calibrated_confidence_note(None) == "확신도 없음"


def test_human_decides_note_includes_breakeven(monkeypatch):
    """⑧ 사람이 정한다 + 손익분기 승률 병기."""
    from src.graph import report_formatter as rf
    monkeypatch.setattr(oc, "summary", lambda h, db_path=None: {
        "n_days": 40, "breakeven_win_rate": 0.47})
    note = rf.human_decides_note(10)
    # [2026-09-15] "사람이 정한다"는 stop_loss_note로 옮겼다 (중복 제거)
    assert "손익분기 승률은 **47%**" in note

    monkeypatch.setattr(oc, "summary", lambda h, db_path=None: {"n_days": 0, "breakeven_win_rate": None})
    assert "표본이 쌓이면" in rf.human_decides_note(10)


def test_benchmark_line_computes_three_windows(monkeypatch):
    """② 5/10/20거래일 지수 수익률. 실패하면 None이라 리포트에서 줄이 빠진다."""
    import sys
    import types
    from src.graph import report_formatter as rf

    class _Hist:
        def __getitem__(self, k):
            class _S:
                def dropna(self_inner):
                    return [100.0 + i for i in range(30)]     # 단조 상승
            return _S()

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda sym: types.SimpleNamespace(history=lambda **kw: _Hist())
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    line = rf.benchmark_context_line()
    assert "069500.KS" in line and "5일 +" in line and "10일 +" in line and "20일 +" in line

    fake.Ticker = lambda sym: (_ for _ in ()).throw(RuntimeError("net down"))
    assert rf.benchmark_context_line() is None, "실패는 리포트를 막지 않는다"


def test_report_sections_carry_new_items(monkeypatch):
    """마크다운 리포트에 ④⑧이 실제로 들어간다."""
    from src.graph import report_formatter as rf
    from src.schemas.agent_output import AnalysisReport

    monkeypatch.setattr(rf, "human_decides_note", lambda horizon=10: "사람이 정한다 문구")
    monkeypatch.setattr(rf, "calibrated_confidence_note", lambda c, a=None: "미보정 문구")
    final = AnalysisReport(agent_name="chief_strategist", confidence=0.72, recommendation="BUY",
                           reasoning=["a", "b", "c"], data_sources=["x", "y"],
                           prediction_basis=["p", "q"], risk_factors=["r"],
                           entry_price=100.0, stop_loss=96.0, stop_loss_pct=-4.0,
                           take_profit_1=108.0, position_size_pct=5.0)
    plan = "\n".join(rf._action_plan_section("005930", final, None))
    assert "사람이 정한다 문구" in plan
    summary = "\n".join(rf._agent_summary_section([], final))
    assert "미보정 문구" in summary
