"""
tests/test_report_honesty.py — 2026-09-15 리포트 정직성·가독성 정리.

  1. 거래 파라미터(배분·익절·R:R)가 본문에서 빠졌는가. **DB 기록은 유지되는가**
  2. CR-10 나머지 ①⑤⑨⑪
  3. 아침 워크플로 순서대로 재배치됐는가 + "오늘 한 줄"
  4. 지평 비교 출력
외부 호출 없음 (yfinance·DART는 monkeypatch 또는 실패 허용).
"""

from __future__ import annotations


import pytest

from src.graph import report_formatter as rf
from src.schemas.agent_output import AnalysisReport


def _report(name: str, conf: float, rec: str, ds: bool = True,
            reasoning: list[str] | None = None) -> AnalysisReport:
    return AnalysisReport(
        agent_name=name, confidence=conf, recommendation=rec, data_sufficient=ds,
        reasoning=reasoning or ["근거1", "근거2", "근거3"],
        data_sources=["출처A", "출처B"], prediction_basis=["수치1", "수치2"],
        risk_factors=[f"리스크 {name}"],
    )


@pytest.fixture
def chief() -> AnalysisReport:
    c = _report("chief_strategist", 0.72, "BUY")
    c.entry_price, c.stop_loss, c.stop_loss_pct = 254_000.0, 243_840.0, -4.0
    c.take_profit_1, c.take_profit_2 = 274_320.0, 284_480.0
    c.position_size_pct, c.rr_ratio, c.holding_period_weeks = 5.0, 2.0, 2
    return c


@pytest.fixture
def agents() -> list[AnalysisReport]:
    return [
        _report("quant_analyst", 0.0, "HOLD", ds=False,
                reasoning=["[폴백:ValidationError] 분석 실패"]),
        _report("technical_analyst", 0.85, "BUY"),
        _report("macro_economist", 0.60, "SELL"),
    ]


# ── 1. 거래 파라미터 제거 ──────────────────────────────────

def test_markdown_report_drops_trade_params(chief, agents, monkeypatch):
    """배분·익절·R:R이 본문에 없어야 한다 (S5 D-2 경로3)."""
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    md = rf.format_report_v4(
        ticker="000500", regime="neutral", strategy="BUY", final=chief, agents=agents,
        qualified_reports=agents[1:], portfolio_summary=None, market_data={"kospi": 6910},
    )
    for banned in ("시드의", "R:R", "1차 익절", "2차 익절", "목표가1"):
        assert banned not in md, f"본문에 {banned}가 남아 있다"
    # 남아야 하는 것
    assert "진입가" in md and "제안 손절가" in md
    assert "254,000원" in md and "243,840원" in md


def test_suppressed_fields_still_on_the_model(chief):
    """DB·섀도가 쓰는 값은 모델에 그대로 있어야 한다. 리포트에서만 뺀 것이다."""
    for field in rf.SUPPRESSED_TRADE_FIELDS:
        assert getattr(chief, field) is not None, f"{field}가 모델에서 사라졌다"


def test_stop_loss_note_marks_it_as_suggestion(chief):
    note = rf.stop_loss_note(chief)
    assert "243,840" in note and "참고용 제안" in note and "사람이 정한다" in note
    assert "참고용 제안" in rf.stop_loss_note(None), "손절가가 없어도 문구는 남는다"


def test_human_decides_note_no_longer_duplicates_stop_loss(monkeypatch):
    """⚖️ 한 줄에 같은 말이 두 번 나오면 둘 다 흘려 읽는다."""
    from src.evaluation import outcomes as oc
    monkeypatch.setattr(oc, "summary", lambda h, db_path=None: {"n_days": 0, "breakeven_win_rate": None})
    note = rf.human_decides_note()
    assert "사람이 정한다" not in note, "그 문구는 stop_loss_note가 맡는다"
    assert "손익분기" in note


# ── 2. CR-10 나머지 ①⑤⑨⑪ ─────────────────────────────────

def test_abstain_summary_splits_reasons(agents):
    """⑤ 몇 명이 왜 유보했나. 사유를 두 갈래로."""
    text = rf.abstain_summary(agents)
    assert "3명 중" in text and "1명 유보" in text
    assert "데이터 없음 1" in text
    assert "실제 투표는 2명" in text
    assert rf.abstain_summary([_report("a", 0.8, "BUY")]) is None, "유보가 없으면 줄을 안 낸다"
    assert rf.abstain_summary([]) is None


def test_debate_status_distinguishes_why_skipped():
    """⑨ '한쪽 우세로 생략'과 '열려야 했는데 안 열림'을 구분한다."""
    dominant = [_report("a", 0.9, "BUY"), _report("b", 0.9, "BUY"), _report("c", 0.2, "SELL")]
    assert "한쪽 우세" in rf.debate_status(dominant)

    close = [_report("a", 0.8, "BUY"), _report("c", 0.7, "SELL")]
    text = rf.debate_status(close)
    assert "한쪽 우세" not in text
    assert "열리지 않았다" in text and "반대 논거가 약할 수 있다" in text

    assert "진행됨" in rf.debate_status(close, "토론 요약 있음")


def test_no_pick_reason_used_for_zero_ticker_day():
    """① 0종목인 날의 사유가 수치로 나온다."""
    text = rf.no_pick_reason(
        {"scores": {"A": {"total_score": 1}, "B": {"total_score": 0}}, "confirmed_min": 2}, "bear")
    assert "확정 기준 2점" in text and "최고 점수 1점" in text


def test_earnings_window_note_survives_dart_failure(monkeypatch):
    """⑪ DART가 끊겨도(CR-3) 리포트 발행을 막지 않는다."""
    import sys
    import types
    fake = types.ModuleType("src.mcp_servers.krx_market.server")
    fake.get_earnings_calendar = lambda **kw: (_ for _ in ()).throw(RuntimeError("DART down"))
    monkeypatch.setitem(sys.modules, "src.mcp_servers.krx_market.server", fake)
    assert rf.earnings_window_note(["005930"]) is None

    fake.get_earnings_calendar = lambda **kw: {"error": "DART_API_KEY가 설정되지 않았습니다."}
    assert rf.earnings_window_note(["005930"]) is None

    fake.get_earnings_calendar = lambda **kw: {
        "upcoming_deadlines": [{"date": "2026-11-14", "description": "3분기 보고서 제출 마감"}],
        "recent_disclosures": [{"stock_code": "005930"}]}
    got = rf.earnings_window_note(["005930"])
    assert "2026-11-14" in got and "실적 공시 1건" in got


# ── 3. 오늘 한 줄 + 순서 ───────────────────────────────────

def test_today_line_covers_new_held_and_health(chief):
    line = rf.today_line(
        [{"ticker": "000500", "chief_report": chief}],
        {"positions": [{"ticker": "A", "alert": "stop_loss"},
                       {"ticker": "B", "overdue": True},
                       {"ticker": "C"}]},
        healthy=True)
    assert line.startswith("오늘: ")
    assert "신규 1종목" in line and "보유 3종목" in line
    assert "1개 손절·목표선 도달" in line and "1개 D+10 경과" in line
    assert "시스템 정상" in line


def test_today_line_flags_degraded_run():
    line = rf.today_line([], None, healthy=False)
    assert "신규 추천 없음" in line and "보유 없음" in line
    assert "시스템 이상" in line


def test_today_line_excludes_review_from_new_count(chief):
    """REVIEW는 판단이 없는 것이므로 신규 추천으로 세지 않는다."""
    chief.needs_review = True
    assert "신규 추천 없음" in rf.today_line([{"ticker": "A", "chief_report": chief}], None)


def test_markdown_section_order_matches_morning_workflow(chief, agents, monkeypatch):
    """건강도 → 판단 → 액션 → 리스크 → 상세 → 시장 → 포트폴리오."""
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    md = rf.format_report_v4(
        ticker="000500", regime="neutral", strategy="BUY", final=chief, agents=agents,
        qualified_reports=agents[1:], portfolio_summary=None, market_data={"kospi": 6910})
    order = [md.index(h) for h in (
        "AI 분석단 의견", "오늘의 액션 플랜", "오늘의 주요 리스크",
        "에이전트별 상세 분석", "시장 지표", "내 포트폴리오 현황")]
    assert order == sorted(order), "섹션 순서가 아침 워크플로와 다르다"
    assert md.index("📌 오늘:") < order[0], "오늘 한 줄이 맨 위에 있어야 한다"


# ── 4. Notion 블록 ────────────────────────────────────────

def _texts(blocks: list[dict]) -> list[str]:
    out = []
    for b in blocks:
        kind = b.get("type")
        node = b.get(kind) or {}
        if "rich_text" in node:
            out.append("".join(rt.get("text", {}).get("content", "") for rt in node["rich_text"]))
        for child in node.get("children") or []:
            out += _texts([child])
    return out


def test_notion_blocks_lead_with_today_and_health(chief, agents, monkeypatch):
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)

    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "000500", "chief_report": chief,
                             "qualified_reports": agents[1:], "all_reports": agents + [chief]}],
        regime="neutral", strategy="BUY",
        health_note="✅ 실행 정상 — 에이전트 24/24 응답",
        screen={"scores": {"000500": {"total_score": 3, "news_match": True,
                                      "quant_score": 2, "quant_signal_count": 3}}},
    )
    texts = _texts(blocks)
    assert texts[0].startswith("오늘: "), "첫 블록이 오늘 한 줄"
    assert "실행 정상" in texts[1], "둘째가 건강도"
    assert blocks[1]["callout"]["color"] == "green_background"
    # 종목 상세는 toggle 안으로 접힌다
    assert any(b["type"] == "toggle" and "000500" in _texts([b])[0] for b in blocks)
    # 뺀 자리에 선정 근거가 들어갔다
    assert any("선정 근거 — 3점 =" in t for t in texts)
    assert any("투표 —" in t for t in texts)


def test_notion_blocks_drop_trade_params_from_action_plan(chief, agents, monkeypatch):
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "000500", "chief_report": chief,
                             "qualified_reports": agents[1:], "all_reports": agents + [chief]}],
        regime="neutral", strategy="BUY")
    joined = "\n".join(_texts(blocks))
    assert "참고용 제안" in joined, "손절가는 제안이라고 밝힌다"
    assert "R:R" not in joined
    # [2026-09-15] 처음엔 "R:R" 한 줄만 봤다가 통합 경로의 세 군데(액션 플랜 표,
    #   toggle 안 최종 판단 줄, 포트폴리오 bullet)에 익절가·배분이 남은 것을 놓쳤다.
    #   _texts()가 children까지 훑으므로 값 자체로 검사한다.
    assert "274,320" not in joined, "1차 익절가가 남아 있다"
    assert "284,480" not in joined, "2차 익절가가 남아 있다"
    assert "목표가" not in joined
    assert "시드의" not in joined, "배분(position_size_pct)이 남아 있다"
    # 모델·DB에는 그대로 있어야 한다 — (a) 전환 시 연속성과 옵션 A 섀도가 쓴다
    assert chief.take_profit_1 == 274_320.0
    assert chief.position_size_pct == 5.0 and chief.rr_ratio == 2.0


def test_notion_zero_ticker_day_explains_why(monkeypatch):
    """① 0종목인 날 Notion에도 사유가 들어간다."""
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[], regime="bear", strategy="NO_PICK",
        screen={"scores": {"A": {"total_score": 1}}, "confirmed_min": 3},
        health_note="✅ 실행 정상 — 오늘 확정 종목 없음")
    joined = "\n".join(_texts(blocks))
    assert "확정 기준 3점" in joined and "최고 점수 1점" in joined
    assert "신규 추천 없음" in joined


# ── 5. 지평 비교 ──────────────────────────────────────────

def test_horizon_comparison_reports_shortfall(tmp_path, monkeypatch):
    from src.evaluation import outcomes as oc
    db = tmp_path / "mentions.db"
    monkeypatch.setattr(oc, "DB_PATH", db)
    oc.init_outcome_tables(db)
    res = oc.horizon_comparison(db)
    assert set(res["by_horizon"]) == set(oc.HORIZONS)
    assert res["trend"] == "표본 부족"
    text = oc.format_horizon_comparison(db)
    assert "D+5" in text and "D+20" in text and "최소 30일 미달" in text


@pytest.mark.parametrize("alphas,expected", [
    ({5: -0.03, 10: -0.02, 20: -0.01}, "H가 길수록 개선"),
    ({5: -0.01, 10: -0.02, 20: -0.03}, "H가 길수록 악화"),
    ({5: -0.02, 10: -0.03, 20: -0.02}, "수익원이 문제"),
    ({5: 0.01, 10: -0.03, 20: 0.02}, "일관된 방향 없음"),
])
def test_horizon_trend_verdicts(alphas, expected, monkeypatch):
    """S5 D-4: 지평 추세가 구조 판단의 입력이다."""
    from src.evaluation import outcomes as oc
    monkeypatch.setattr(oc, "summary", lambda h, db_path=None: {
        "n_days": 40, "avg_raw": alphas[h], "avg_alpha": alphas[h], "avg_net": alphas[h],
        "hit_rate": 0.5, "breakeven_win_rate": 0.47})
    assert expected in oc.horizon_comparison()["trend"]


# ─────────────────────────────────────────────────────────────────────────────
# 회귀: 표 헤더와 데이터 행의 열 수가 어긋나면 값이 통째로 밀린다
#
# 2026-09-15에 실제로 났던 버그. 포트폴리오 표 헤더에서 3개 열을 뺐는데
# 행 생성부를 안 고쳐서, 헤더 7열 / 행 10열이 되어 수익률 자리에 배분이,
# 손절가 자리에 수익률이 찍혔다. 마크다운은 조용히 렌더링하므로 눈으로만
# 잡을 수 있었다. 열 수를 세는 것이 가장 싼 방어다.
# ─────────────────────────────────────────────────────────────────────────────

def _table_widths(lines: list[str]) -> list[list[int]]:
    """연속한 표 블록마다 각 행의 열 수를 센다."""
    blocks, cur = [], []
    for ln in lines:
        t = ln.strip()
        if t.startswith("|") and t.endswith("|"):
            if set(t) <= set("|-: "):          # 구분선은 세지 않는다
                continue
            cur.append(len(t.strip("|").split("|")))
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def test_portfolio_table_columns_align():
    """보유 포지션 표: 헤더와 모든 데이터 행의 열 수가 같아야 한다."""
    portfolio = {
        "invested_pct": 42.0, "available_pct": 58.0, "cumulative_pnl_seed_pct": -1.8,
        "positions": [{
            "ticker": "000500", "entry_date": "2026-09-01", "entry_price": 12000,
            "current_price": 11300, "allocation_pct": 4.0, "position_pnl_pct": -5.8,
            "stop_loss_price": 11520, "take_profit_1": 12960,
            "holding_days": 13, "holding_period_weeks": 2, "overdue": True,
        }],
        "closed_recent": [],
    }
    for block in _table_widths(rf._portfolio_section(portfolio)):
        assert len(set(block)) == 1, f"열 수가 어긋났다: {block}"


def test_action_plan_table_columns_align(chief):
    """액션 플랜 표도 같은 검사. 거래 파라미터를 빼면서 열을 줄였다."""
    portfolio = {"invested_pct": 0, "available_pct": 100,
                 "cumulative_pnl_seed_pct": 0, "positions": [], "closed_recent": []}
    for block in _table_widths(rf._action_plan_section("005930", chief, portfolio)):
        assert len(set(block)) == 1, f"열 수가 어긋났다: {block}"


# ─────────────────────────────────────────────────────────────────────────────
# 실제 발행본을 보고 잡은 것 (2026-09-15)
#
# 로컬 덤프로는 "형태"만 보이고 "무엇이 채워지는가"는 안 보인다. 실제 Notion
# 페이지에서야 맨 위 최종 판단 줄이 "[투표 집계] BUY 가중합 0.00, SELL 가중합
# 1.40..."으로 시작해 문장 중간에서 잘리는 것이 보였다.
# ─────────────────────────────────────────────────────────────────────────────

def _chief_with_tally() -> AnalysisReport:
    c = _report("chief_strategist", 0.55, "HOLD")
    c.reasoning = [
        "[투표 집계] BUY 가중합 0.00, SELL 가중합 1.40, HOLD 가중합 2.20. "
        "SELL이 BUY보다 1.40 높지만, HOLD가 가장 높은 가중합을 기록. SELL 가중합(1.40)이 "
        "BUY 가중합(0.00) + 0.2를 초과하나, HOLD 가중합(2.20)이 SELL을 압도하여 방향성 확신 부족.",
        "[핵심 쟁점] 기관 순매도 203,789주 vs 외국인 순매수 117,791주로 수급 혼재.",
        "[결론] 정배열 유지·RSI 중립·VIX 안정 등 상반된 신호가 공존. "
        "방향성이 명확하지 않으므로 HOLD가 가장 합리적인 판단.",
    ]
    return c


def test_headline_reason_skips_vote_tally():
    """요약 줄에는 가중합이 아니라 결론이 와야 한다."""
    line = rf.headline_reason(_chief_with_tally())
    assert "가중합" not in line, "집계가 요약 줄에 실렸다"
    assert "[결론]" not in line, "말머리는 떼고 내용만"
    assert "방향성이 명확하지 않으므로" in line


def test_headline_reason_cuts_at_sentence_boundary():
    """길면 문장 경계에서 끊는다. 중간에서 끊기면 읽다가 멈춘다."""
    c = _report("chief_strategist", 0.6, "HOLD")
    c.reasoning = ["첫 문장은 짧다. " + "두 번째 문장은 아주 길어서 잘릴 것이다" * 5 + "."]
    line = rf.headline_reason(c, 40)
    assert line.endswith((".", "…")), f"경계에서 안 끊겼다: {line!r}"


def test_headline_reason_falls_back_when_all_tally():
    """집계뿐이면 그거라도 보여준다 — 빈 줄보다는 낫다."""
    c = _report("chief_strategist", 0.6, "HOLD")
    c.reasoning = ["투표 집계: BUY 1.0 vs SELL 2.0", "투표 집계 재확인", "가중합 정리"]
    assert rf.headline_reason(c), "전부 집계여도 빈 문자열이면 안 된다"


def test_notion_summary_shows_company_name_and_conclusion(agents, monkeypatch):
    """맨 위 최종 판단 줄: 종목명 + 집계가 아닌 근거."""
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    chief = _chief_with_tally()
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": chief,
                             "qualified_reports": agents[1:], "all_reports": agents + [chief]}],
        regime="neutral", strategy="HOLD")
    summary = next(t for t in _texts(blocks) if "🟡" in t or "관망" in t)
    assert "삼성전자" in summary, "종목코드만 있으면 어느 회사인지 알 수 없다"
    assert "가중합" not in summary


def test_notion_says_no_new_pick_when_zero_buys(agents, monkeypatch):
    """신규 매수 0건인 날 액션 플랜이 빈 것처럼 보이면 안 된다."""
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    chief = _report("chief_strategist", 0.55, "HOLD")
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": chief,
                             "qualified_reports": agents[1:], "all_reports": agents + [chief]}],
        regime="neutral", strategy="HOLD")
    joined = "\n".join(_texts(blocks))
    assert "신규 매수 추천" in joined, "heading이 사라지면 스크롤하다 놓친다"
    assert "오늘 신규 진입 추천 없음" in joined


# ─────────────────────────────────────────────────────────────────────────────
# 2026-09-15 발행본 점검에서 나온 것들
#
# 공통점: 로컬 픽스처로는 안 잡혔다. 실제 데이터가 들어가야 드러나는 종류다.
#   - 000500이 손절선 아래인데 액션 플랜은 "모니터링"만 (판정이 두 곳에 따로 있었다)
#   - 보유 -3%/-5%인데 "누적 수익 +0.00%" (실현만 세고 있었다)
#   - PER 1161.57배가 고평가 근거로 (숫자가 있으니 판단처럼 보였다)
#   - macro/us_market 문단이 4종목에 4번 반복
# ─────────────────────────────────────────────────────────────────────────────

def _pos(ticker="000500", cur=241_000.0, sl=241_300.0, tp=None,
         entry=254_000.0, days=1, overdue=False, alert=None):
    return {"ticker": ticker, "ticker_name": "", "entry_date": "2026-09-13",
            "entry_price": entry, "current_price": cur, "allocation_pct": 5.0,
            "position_pnl_pct": round((cur / entry - 1) * 100, 2),
            "seed_pnl_pct": round(5.0 * (cur / entry - 1) * 100 / 100, 2),
            "stop_loss_price": sl, "target_price": tp, "take_profit_1": None,
            "holding_days": days, "holding_period_weeks": 2, "overdue": overdue,
            "alert": alert}


# ── 1. 손절·목표선 판정이 한 곳에서 나온다 ───────────────────────────────

def test_alert_reached_when_price_below_stop():
    """발행본의 실제 값: 현재가 241,000 / 손절 241,300."""
    hit = rf.position_alert(_pos(alert="stop_loss"))
    assert hit is not None
    grade, note = hit
    assert grade == "reached"
    assert "손절선 도달" in note and "241,000" in note


def test_alert_near_within_threshold():
    """손절선 위 2% 이내는 '근접'. 3%는 아무것도 아니다."""
    assert rf.position_alert(_pos(cur=101_500.0, sl=100_000.0))[0] == "near"
    assert rf.position_alert(_pos(cur=103_000.0, sl=100_000.0)) is None


def test_alert_reached_beats_overdue():
    """도달과 경과가 겹치면 도달이 이긴다. 둘 다 붙이면 뭐가 급한지 안 보인다."""
    assert rf.position_alert(_pos(alert="stop_loss", overdue=True))[0] == "reached"


def test_today_line_and_action_plan_agree(monkeypatch):
    """한 줄 요약과 액션 플랜이 어긋나면 안 된다 — 같은 함수를 쓴다."""
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    portfolio = {"invested_pct": 5.0, "available_pct": 95.0,
                 "cumulative_pnl_seed_pct": 0.0, "unrealized_pnl_seed_pct": -0.26,
                 "priced_positions": 1, "positions": [_pos(alert="stop_loss")],
                 "closed_recent": []}
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[], regime="neutral", strategy="NO_PICK",
        portfolio_summary=portfolio, screen={"scores": {}, "confirmed_min": 3})
    texts = _texts(blocks)
    assert "손절·목표선 도달" in texts[0], "오늘 한 줄에 나와야 한다"
    joined = "\n".join(texts)
    plan = joined[joined.index("기존 포지션 처리"):]
    assert "손절선 도달" in plan, "결정하는 자리에도 나와야 한다"


# ── 2. 실현 / 미실현 분리 ────────────────────────────────────────────────

def test_seed_pnl_note_separates_realized_and_unrealized():
    note = rf.seed_pnl_note({"cumulative_pnl_seed_pct": 0.0,
                             "unrealized_pnl_seed_pct": -0.44,
                             "priced_positions": 2, "positions": [1, 2]})
    assert "실현 손익: +0.00%" in note
    assert "미실현: -0.44%" in note, "보유 손실이 0으로 보이면 안 된다"


def test_seed_pnl_note_flags_missing_prices():
    """현재가를 못 받은 종목이 있으면 미실현이 과소 계상된다 — 밝힌다."""
    note = rf.seed_pnl_note({"cumulative_pnl_seed_pct": 1.0,
                             "unrealized_pnl_seed_pct": -0.2,
                             "priced_positions": 1, "positions": [1, 2, 3]})
    assert "미수신 2종목" in note


# ── 3. 극단값 ────────────────────────────────────────────────────────────

def test_extreme_per_is_dropped_from_reasons():
    """발행본의 실제 값: 삼성SDI PER 1161.57배."""
    items = ["삼성SDI의 PER이 1161.57배로 매우 높은 수준이며, 이는 고평가를 나타냅니다.",
             "PBR이 1.72배로 청산가치 이하로 평가됩니다."]
    kept, dropped = rf.filter_extreme_values(items)
    assert len(kept) == 1 and "PBR" in kept[0]
    assert dropped == ["PER 1161.57배"]
    assert "근거에서 제외" in rf.extreme_value_note(dropped)


def test_normal_valuations_survive():
    """정상 범위는 하나도 건드리지 않는다."""
    items = ["PER은 14.02배로 중립적이다.", "PBR 0.59배로 청산가치 이하.",
             "배당수익률 3.2%로 양호하다."]
    kept, dropped = rf.filter_extreme_values(items)
    assert kept == items and dropped == []


def test_negative_per_is_extreme():
    """적자면 PER 정의가 성립하지 않는다."""
    _, dropped = rf.filter_extreme_values(["PER이 -8.3배로 나타났다."])
    assert dropped == ["PER -8.3배"]


@pytest.mark.parametrize("raw, survives", [
    ("199", True),    # 상한 바로 아래
    ("200", True),    # 상한 그 자체 — 판정이 low < val <= high 라 포함된다
    ("201", False),   # 상한 초과
    ("0",   False),   # 하한은 배제(<) — 0이면 적자라 PER이 성립하지 않는다
])
def test_per_boundary_values(raw, survives):
    """VALUATION_RANGES["PER"] = (0.0, 200.0) 의 경계를 고정한다.

    범위를 손볼 때 어느 쪽 끝이 열리고 닫히는지 눈에 보이게 하려는 것이다.
    """
    items = [f"PER이 {raw}배로 나타났다."]
    kept, dropped = rf.filter_extreme_values(items)
    if survives:
        assert kept == items and dropped == []
    else:
        assert kept == [] and dropped == [f"PER {raw}배"]


def test_non_numeric_per_passes_through():
    """숫자가 아니면 _METRIC_RE가 잡지 못해 그대로 나간다 — 현재 필터의 한계.

    막으려면 포맷 단계가 아니라 fundamental_analyst 쪽에서 처리해야 한다.
    여기서는 "통과한다"는 사실만 고정해 둔다.
    """
    items = ["PER: N/A배", "PER은 알 수 없다"]
    kept, dropped = rf.filter_extreme_values(items)
    assert kept == items and dropped == []


# ── 4. 시장 공통 에이전트 ────────────────────────────────────────────────

def test_market_background_dedupes():
    """4종목이면 4번 나오던 것을 에이전트당 1건으로."""
    per_ticker = []
    for _ in range(4):                       # 같은 보고서가 종목 수만큼 반복된다
        per_ticker += [_report(n, 0.6, "HOLD") for n in rf.MARKET_WIDE_AGENTS]
        per_ticker.append(_report("quant_analyst", 0.7, "BUY"))
    bg = rf.market_background(per_ticker)
    assert [r.agent_name for r in bg] == list(rf.MARKET_WIDE_AGENTS)


def test_market_agents_move_to_background_but_stay_in_vote(agents, chief, monkeypatch):
    """상세에서는 빠지고, 투표 집계에는 남는다."""
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": chief,
                             "qualified_reports": agents, "all_reports": agents + [chief]}],
        regime="neutral", strategy="BUY")
    joined = "\n".join(_texts(blocks))
    assert "오늘의 시장 배경" in joined
    detail = joined[joined.index("에이전트별 상세 분석"):]
    assert "macro_economist" not in detail, "상세에서 빠져야 한다"
    vote = joined[joined.index("AI 분석단 의견"):joined.index("에이전트별 상세 분석")]
    assert "매크로 분석" in vote, "투표 집계에는 남아야 한다"


# ── 5. 자잘한 것 ─────────────────────────────────────────────────────────

def test_health_callout_has_no_duplicate_emoji(monkeypatch):
    """✅가 두 번 찍히지 않게 본문 앞 기호를 뗀다."""
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[], regime="neutral", strategy="NO_PICK",
        health_note="✅ 실행 정상 — 에이전트 31/32 응답",
        screen={"scores": {}, "confirmed_min": 3})
    assert blocks[1]["callout"]["icon"]["emoji"] == "✅"
    assert not _texts(blocks)[1].startswith("✅"), "본문에 기호가 또 있으면 중복이다"


def test_debate_status_summary_and_per_ticker(agents, chief, monkeypatch):
    """리스크 섹션엔 몇 종목에서 열렸는지만, 종목별 사유는 그 종목 toggle 안에.

    발행본의 "토론 생략 — BUY 2.50 vs SELL 3.90"은 어느 종목 얘긴지 없어서
    읽을 수 없었다. 종목마다 그 긴 문장을 반복하면 리스크 목록이 아래로 밀린다.
    """
    from src.graph import notion_publisher as np
    monkeypatch.setattr(rf, "benchmark_context_line", lambda: None)
    monkeypatch.setattr(rf, "recent_performance_lines", lambda: [])
    monkeypatch.setattr(rf, "earnings_window_note", lambda t, days_ahead=14: None)
    blocks = np.build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": chief,
                             "qualified_reports": agents, "all_reports": agents + [chief]}],
        regime="neutral", strategy="BUY")
    texts = _texts(blocks)
    assert any("토론 열림 0/1종목" in t for t in texts), "리스크 섹션엔 요약만"
    tog = next(b for b in blocks
               if b["type"] == "toggle" and "005930" in _texts([b])[0])
    assert any("토론 생략" in t for t in _texts([tog])), "종목 toggle에 그 종목 사유"


# ── 2026-09-14 두 번째 발행본에서 추가로 본 것 ──────────────────────────

def test_headline_reason_strips_bold_marker():
    """chief는 "[결론]"만 쓰지 않는다. "**결론:**"도 쓴다 — 발행본에서 그대로 실렸다."""
    c = _report("chief_strategist", 0.45, "HOLD")
    c.reasoning = ["**투표 집계:** BUY 1.60 vs SELL 1.30 vs HOLD 1.50.",
                   "**토론 핵심 쟁점 — 동일 지표의 이중 해석:** 외국인 순매수 해석이 갈린다.",
                   "**결론:** 기술적 분석과 외국인 수급은 약한 BUY 신호를 준다."]
    line = rf.headline_reason(c)
    assert not line.startswith("**"), f"말머리가 남았다: {line!r}"
    assert line.startswith("기술적 분석")


def test_fallback_risks_excluded_from_risk_list():
    """"에이전트 오류로 분석 불가"가 오늘의 주요 리스크 5위에 올라왔었다."""
    ok = _report("quant_analyst", 0.6, "SELL")
    ok.risk_factors = ["유가 하락 시 실적 악화"]
    dead = _report("macro_economist", 0.0, "HOLD")
    dead.risk_factors = ["에이전트 오류로 분석 불가"]
    risks = rf.real_risk_factors([ok, dead])
    assert risks == ["유가 하락 시 실적 악화"]


def test_real_risk_factors_keeps_order_and_dedupes():
    a = _report("quant_analyst", 0.6, "SELL")
    a.risk_factors = ["A", "B"]
    b = _report("technical_analyst", 0.8, "HOLD")
    b.risk_factors = ["B", "C"]
    assert rf.real_risk_factors([a, b]) == ["A", "B", "C"]
