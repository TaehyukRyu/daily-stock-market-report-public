"""
tests/test_report_formatter.py

report_formatter v4.1 단위 테스트 — LLM / API 호출 없음.

7섹션 구조:
  1. 💼 내 포트폴리오 현황
  2. 🎯 오늘의 액션 플랜
  3. 📊 시장 지표
  4. 🤖 AI 분석단 의견
  5. 📋 에이전트별 상세 분석
  6. ⚠️ 오늘의 주요 리스크
  7. 면책 / 생성 정보
"""

import pytest
from src.graph.report_formatter import (
    format_report_v4,
    _header_block,
    _portfolio_section,
    _action_plan_section,
    _market_data_section,
    _agent_summary_section,
    _details_section,
    _risk_section,
    _footer_section,
    REGIME_KR,
    AGENT_NAME_KR,
    SIGNAL_KR,
    STRATEGY_KR,
)
from src.schemas.agent_output import AnalysisReport


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

def _make_agent(name: str, rec: str, conf: float) -> AnalysisReport:
    return AnalysisReport(
        agent_name=name,
        recommendation=rec,
        confidence=conf,
        reasoning=["근거1", "근거2", "근거3"],
        data_sources=["소스A", "소스B"],
        prediction_basis=["정량1", "정량2"],
        risk_factors=[f"리스크-{name}"],
    )


@pytest.fixture
def agents() -> list[AnalysisReport]:
    return [
        _make_agent("macro_economist",     "BUY",  0.75),
        _make_agent("kr_market_specialist","HOLD", 0.60),
        _make_agent("us_market_specialist","SELL", 0.55),
        _make_agent("quant_analyst",       "BUY",  0.70),
    ]


@pytest.fixture
def chief_buy() -> AnalysisReport:
    return AnalysisReport(
        agent_name="chief_strategist",
        recommendation="BUY",
        confidence=0.78,
        reasoning=["진입 근거 한줄", "근거2", "근거3"],
        data_sources=["소스X", "소스Y"],
        prediction_basis=["숫자1", "숫자2"],
        risk_factors=["리스크-chief"],
        entry_price=300_000.0,
        stop_loss=288_000.0,
        stop_loss_pct=-4.0,
        take_profit_1=324_000.0,
        take_profit_2=336_000.0,
        rr_ratio=2.0,
        position_size_pct=20.0,
        holding_period_weeks=2,
        entry_strategy="분할매수",
    )


@pytest.fixture
def chief_hold() -> AnalysisReport:
    return AnalysisReport(
        agent_name="chief_strategist",
        recommendation="HOLD",
        confidence=0.60,
        reasoning=["방향성 합의 부재", "관망 권장", "단기 변동성 주의"],
        data_sources=["소스X", "소스Y"],
        prediction_basis=["숫자1", "숫자2"],
        risk_factors=["리스크-chief-hold"],
    )


@pytest.fixture
def chief_sell() -> AnalysisReport:
    return AnalysisReport(
        agent_name="chief_strategist",
        recommendation="SELL",
        confidence=0.70,
        reasoning=["하락 추세 가속", "외국인 이탈", "기술적 지지선 이탈"],
        data_sources=["소스X", "소스Y"],
        prediction_basis=["숫자1", "숫자2"],
        risk_factors=["리스크-chief-sell"],
    )


@pytest.fixture
def qualified(agents) -> list[AnalysisReport]:
    return agents[:2]


@pytest.fixture
def portfolio_with_positions() -> dict:
    return {
        "invested_pct": 70.0,
        "available_pct": 30.0,
        "cumulative_pnl_seed_pct": 2.3,
        "positions": [
            {
                "ticker": "005930", "ticker_name": "삼성전자",
                "entry_date": "2026-05-10", "entry_price": 265000, "current_price": 270500,
                "allocation_pct": 20, "position_pnl_pct": 2.07, "seed_pnl_pct": 0.41,
                "stop_loss_price": 252000, "take_profit_1": 294300, "take_profit_2": 310500,
                "holding_days": 6, "holding_period_weeks": 3, "rr_ratio": 2.0, "alert": None,
            },
            {
                "ticker": "000660", "ticker_name": "SK하이닉스",
                "entry_date": "2026-05-08", "entry_price": 1_750_000, "current_price": 1_819_000,
                "allocation_pct": 30, "position_pnl_pct": 3.94, "seed_pnl_pct": 1.18,
                "stop_loss_price": 1_627_500, "take_profit_1": 1_925_000, "take_profit_2": None,
                "holding_days": 8, "holding_period_weeks": 3, "rr_ratio": 1.8, "alert": None,
            },
        ],
        "closed_recent": [
            {
                "ticker": "035720", "ticker_name": "카카오",
                "entry_price": 52000, "close_price": 55000, "allocation_pct": 10,
                "position_pnl_pct": 5.77, "seed_pnl_pct": 0.58, "holding_days": 9,
                "close_reason": "manual",
            },
        ],
    }


@pytest.fixture
def market_data_basic() -> dict:
    return {
        "kospi": 3000.50,
        "vix": 18.43,
        "us_10y_yield": 4.59,
        "usd_krw": 1382.0,
        "wti": 105.42,
    }


@pytest.fixture
def report_buy(agents, chief_buy, qualified, portfolio_with_positions, market_data_basic) -> str:
    return format_report_v4(
        ticker="005930", regime="bull", strategy="BUY",
        final=chief_buy, agents=agents, qualified_reports=qualified,
        debate_summary="Bull: 강세. Bear: 과열 우려.",
        portfolio_summary=portfolio_with_positions,
        market_data=market_data_basic,
    )


@pytest.fixture
def report_hold(agents, chief_hold, qualified) -> str:
    return format_report_v4(
        ticker="005930", regime="VOLATILE", strategy="HOLD",
        final=chief_hold, agents=agents, qualified_reports=qualified,
    )


@pytest.fixture
def report_no_portfolio(agents, chief_hold, qualified) -> str:
    return format_report_v4(
        ticker="005930", regime="bull", strategy="HOLD",
        final=chief_hold, agents=agents, qualified_reports=qualified,
        portfolio_summary=None, market_data=None,
    )


# ─────────────────────────────────────────────────────────
# 기본 구조 검증
# ─────────────────────────────────────────────────────────

def test_report_is_string(report_buy):
    assert isinstance(report_buy, str)


def test_report_nonempty(report_buy):
    assert len(report_buy) > 500


def test_report_title_format(report_buy):
    assert "Daily stock market report_" in report_buy
    assert "_BUY" in report_buy


def test_report_title_includes_date(report_buy):
    import re
    assert re.search(r"Daily stock market report_\d{8}_\d{2}:\d{2}_", report_buy)


# ─────────────────────────────────────────────────────────
# 상수 매핑 — KR 라벨
# ─────────────────────────────────────────────────────────

def test_regime_kr_has_all_main_keys():
    for k in ("VOLATILE", "BULL", "BEAR", "NEUTRAL", "UNKNOWN"):
        assert k in REGIME_KR


def test_agent_name_kr_has_chief():
    assert AGENT_NAME_KR["chief_strategist"] == "종합 판단"


def test_signal_kr_three_signals():
    assert "BUY" in SIGNAL_KR and "HOLD" in SIGNAL_KR and "SELL" in SIGNAL_KR


def test_strategy_kr_mapping():
    assert STRATEGY_KR["BUY"] == "매수"
    assert STRATEGY_KR["HOLD"] == "관망"
    assert STRATEGY_KR["SELL"] == "매도"


# ─────────────────────────────────────────────────────────
# validate_report() 호환 — 최종 판단 키워드 존재
# ─────────────────────────────────────────────────────────

def test_validate_compat_final_section(report_buy):
    """'최종 판단' 문자열이 포함돼야 validate_report() 통과."""
    assert "최종 판단" in report_buy


def test_validate_compat_hold_report(report_hold):
    assert "최종 판단" in report_hold


def test_validate_compat_no_portfolio_report(report_no_portfolio):
    assert "최종 판단" in report_no_portfolio


# ─────────────────────────────────────────────────────────
# 헤더 블록
# ─────────────────────────────────────────────────────────

def test_header_block_returns_list():
    lines = _header_block("VOLATILE")
    assert isinstance(lines, list)


def test_header_block_renders_regime_kr():
    lines = _header_block("VOLATILE")
    combined = "\n".join(lines)
    assert "변동성 장세" in combined


def test_header_block_includes_weekday():
    lines = _header_block("BULL")
    combined = "\n".join(lines)
    assert any(w in combined for w in ["월", "화", "수", "목", "금", "토", "일"])


def test_header_block_falls_back_unknown_regime():
    lines = _header_block("xyz_garbage")
    combined = "\n".join(lines)
    # 알 수 없는 레짐은 원문 그대로 렌더
    assert "xyz_garbage" in combined


# ─────────────────────────────────────────────────────────
# 섹션 1: 포트폴리오
# ─────────────────────────────────────────────────────────

def test_portfolio_section_empty():
    lines = _portfolio_section(None)
    combined = "\n".join(lines)
    assert "보유 포지션 없음" in combined


def test_portfolio_section_empty_positions():
    lines = _portfolio_section({"positions": [], "available_pct": 100, "invested_pct": 0})
    combined = "\n".join(lines)
    assert "보유 포지션 없음" in combined


def test_portfolio_section_invested_pct(portfolio_with_positions):
    lines = _portfolio_section(portfolio_with_positions)
    combined = "\n".join(lines)
    assert "투자중: 70" in combined
    assert "가용: 30" in combined


def test_portfolio_section_cumulative_pnl(portfolio_with_positions):
    lines = _portfolio_section(portfolio_with_positions)
    combined = "\n".join(lines)
    assert "+2.30% of seed" in combined


def test_portfolio_section_position_row(portfolio_with_positions):
    lines = _portfolio_section(portfolio_with_positions)
    combined = "\n".join(lines)
    assert "삼성전자" in combined
    assert "265,000" in combined
    assert "270,500" in combined
    assert "+2.07%" in combined


def test_portfolio_section_closed_history(portfolio_with_positions):
    lines = _portfolio_section(portfolio_with_positions)
    combined = "\n".join(lines)
    assert "청산 이력" in combined
    assert "카카오" in combined or "035720" in combined


def test_portfolio_section_alert_stop_loss():
    summary = {
        "invested_pct": 20, "available_pct": 80, "cumulative_pnl_seed_pct": 0,
        "positions": [{
            "ticker": "005930", "ticker_name": "삼성전자",
            "entry_date": "2026-05-10", "entry_price": 100000, "current_price": 90000,
            "allocation_pct": 20, "position_pnl_pct": -10.0, "seed_pnl_pct": -2.0,
            "stop_loss_price": 93000, "take_profit_1": 110000, "take_profit_2": None,
            "holding_days": 3, "holding_period_weeks": 2, "rr_ratio": 1.5,
            "alert": "stop_loss",
        }],
        "closed_recent": [],
    }
    lines = _portfolio_section(summary)
    combined = "\n".join(lines)
    assert "⚠️" in combined


def test_portfolio_section_alert_target():
    summary = {
        "invested_pct": 20, "available_pct": 80, "cumulative_pnl_seed_pct": 0,
        "positions": [{
            "ticker": "005930", "ticker_name": "삼성전자",
            "entry_date": "2026-05-10", "entry_price": 100000, "current_price": 115000,
            "allocation_pct": 20, "position_pnl_pct": 15.0, "seed_pnl_pct": 3.0,
            "stop_loss_price": 93000, "take_profit_1": 110000, "take_profit_2": None,
            "holding_days": 3, "holding_period_weeks": 2, "rr_ratio": 1.5,
            "alert": "target",
        }],
        "closed_recent": [],
    }
    lines = _portfolio_section(summary)
    combined = "\n".join(lines)
    assert "🎯" in combined


# ─────────────────────────────────────────────────────────
# 섹션 2: 액션 플랜
# ─────────────────────────────────────────────────────────

def test_action_plan_no_positions_no_buy(agents, chief_hold):
    lines = _action_plan_section("005930", chief_hold, None)
    combined = "\n".join(lines)
    assert "기존 포지션 처리" in combined
    assert "보유 포지션 없음" in combined
    assert "오늘 신규 진입 추천 없음" in combined


def test_action_plan_buy_recommendation(chief_buy, portfolio_with_positions):
    lines = _action_plan_section("005930", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "신규 매수 추천" in combined
    assert "300,000" in combined  # entry price
    assert "005930" in combined


def test_action_plan_buy_includes_stop_loss(chief_buy, portfolio_with_positions):
    lines = _action_plan_section("005930", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "288,000" in combined
    assert "-4.0%" in combined


def test_action_plan_buy_omits_take_profit(chief_buy, portfolio_with_positions):
    """[2026-09-15] 익절가를 본문에서 뺐다 (S5 D-2 경로3).

    R:R 2.0 항등식에서 기계적으로 나온 값이라 정보가 없는데 숫자라서 정보처럼 읽힌다.
    모델·DB에는 그대로 남는다 — 아래 두 assert가 그것을 확인한다.
    """
    lines = _action_plan_section("005930", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "324,000" not in combined
    assert "336,000" not in combined
    assert chief_buy.take_profit_1 == 324000, "모델에는 남아 있어야 한다"
    assert chief_buy.take_profit_2 == 336000


def test_action_plan_buy_omits_rr_and_weeks(chief_buy, portfolio_with_positions):
    """[2026-09-15] R:R은 항등식 2.0이라 신호가 아니다. 본문에서 뺐다."""
    lines = _action_plan_section("005930", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "R:R" not in combined
    assert "예상보유" not in combined
    assert chief_buy.rr_ratio == 2.0, "모델에는 남아 있어야 한다"


def test_action_plan_buy_omits_allocation(chief_buy, portfolio_with_positions):
    """[2026-09-15] 배분은 표본 부족이면 전부 5% 고정이라 판단 정보가 아니다."""
    lines = _action_plan_section("005930", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "시드의" not in combined
    assert chief_buy.position_size_pct == 20.0, "모델에는 남아 있어야 한다"


def test_action_plan_no_cash_warning_after_allocation_removed():
    """[2026-09-15] 배분을 본문에서 빼면서 "가용 현금 초과" 경고도 사라졌다.

    경고가 배분 수치를 전제로 하기 때문이다. 배분은 여전히 DB에 쌓이므로
    필요하면 포트폴리오 쪽에서 따로 계산해 되살릴 수 있다.
    """
    chief = AnalysisReport(
        agent_name="chief_strategist", recommendation="BUY", confidence=0.7,
        reasoning=["a", "b", "c"], data_sources=["s1", "s2"],
        prediction_basis=["p1", "p2"], risk_factors=["r1"],
        entry_price=100000, stop_loss=95000, stop_loss_pct=-5,
        take_profit_1=110000, take_profit_2=120000, rr_ratio=2.0,
        position_size_pct=50.0, holding_period_weeks=2,
    )
    portfolio = {
        "invested_pct": 70, "available_pct": 30,
        "cumulative_pnl_seed_pct": 0, "positions": [], "closed_recent": [],
    }
    combined = "\n".join(_action_plan_section("005930", chief, portfolio))
    assert "가용 현금" not in combined
    assert "진입가" in combined and "제안 손절가" in combined

def test_action_plan_no_warning_when_within_budget(chief_buy, portfolio_with_positions):
    # chief_buy: position_size_pct=20.0, portfolio available_pct=30 → 경고 없어야 함
    lines = _action_plan_section("005930", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "초과합니다" not in combined


def test_action_plan_hold_existing_position(chief_hold, portfolio_with_positions):
    lines = _action_plan_section("005930", chief_hold, portfolio_with_positions)
    combined = "\n".join(lines)
    # 005930은 분석 대상이므로 'HOLD' → '🟡 유지'
    assert "유지" in combined


def test_action_plan_sell_existing_position(chief_sell, portfolio_with_positions):
    lines = _action_plan_section("005930", chief_sell, portfolio_with_positions)
    combined = "\n".join(lines)
    assert "매도 검토" in combined


def test_action_plan_other_ticker_monitoring(chief_buy, portfolio_with_positions):
    """분석 ticker가 아닌 보유 종목은 '모니터링' 표시."""
    lines = _action_plan_section("000660", chief_buy, portfolio_with_positions)
    combined = "\n".join(lines)
    # 005930은 분석 대상 아니므로 모니터링
    assert "모니터링" in combined


# ─────────────────────────────────────────────────────────
# 섹션 3: 시장 지표
# ─────────────────────────────────────────────────────────

def test_market_data_none():
    lines = _market_data_section(None)
    combined = "\n".join(lines)
    assert "시장 지표 데이터 없음" in combined


def test_market_data_empty():
    lines = _market_data_section({})
    combined = "\n".join(lines)
    assert "시장 지표 데이터 없음" in combined


def test_market_data_basic_indicators(market_data_basic):
    lines = _market_data_section(market_data_basic)
    combined = "\n".join(lines)
    assert "KOSPI" in combined
    assert "공포지수(VIX)" in combined
    assert "미 10년 금리" in combined
    assert "WTI" in combined


def test_market_data_vix_interpretation():
    lines = _market_data_section({"vix": 35.0})
    combined = "\n".join(lines)
    assert "공포" in combined  # VIX > 30 → 공포


def test_market_data_vix_stable():
    lines = _market_data_section({"vix": 18.0})
    combined = "\n".join(lines)
    assert "안정적" in combined


def test_market_data_high_yield_warning():
    lines = _market_data_section({"us_10y_yield": 5.0})
    combined = "\n".join(lines)
    assert "주식 부담" in combined


def test_market_data_scalar_extraction_from_dict():
    """vix 키가 dict로 들어와도 추출 가능."""
    lines = _market_data_section({"vix": {"value": 22.5}})
    combined = "\n".join(lines)
    assert "주의" in combined  # 22.5 → 주의


# ─────────────────────────────────────────────────────────
# 섹션 4: AI 분석단 의견
# ─────────────────────────────────────────────────────────

def test_agent_summary_includes_all_agents(agents, chief_buy):
    lines = _agent_summary_section(agents, chief_buy)
    combined = "\n".join(lines)
    for r in agents:
        kr_name = AGENT_NAME_KR.get(r.agent_name, r.agent_name)
        assert kr_name in combined


def test_agent_summary_includes_chief_label(agents, chief_buy):
    lines = _agent_summary_section(agents, chief_buy)
    combined = "\n".join(lines)
    assert "최종 판단" in combined


def test_agent_summary_sort_order(agents, chief_buy):
    """SELL → HOLD → BUY 순으로 정렬되어야 함."""
    lines = _agent_summary_section(agents, chief_buy)
    combined = "\n".join(lines)
    sell_pos = combined.find("🔴 매도")
    hold_pos = combined.find("🟡 관망")
    buy_pos  = combined.find("🟢 매수")
    assert sell_pos < hold_pos < buy_pos


def test_agent_summary_low_confidence_note():
    """confidence < 0.6 → ⚠️ 참고용 비고."""
    low_conf = _make_agent("macro_economist", "HOLD", 0.40)
    lines = _agent_summary_section([low_conf], None)
    combined = "\n".join(lines)
    assert "참고용" in combined


def test_agent_summary_zero_confidence_note():
    """confidence == 0 → ⚠️ 데이터없음 비고."""
    zero_conf = _make_agent("macro_economist", "HOLD", 0.0)
    lines = _agent_summary_section([zero_conf], None)
    combined = "\n".join(lines)
    assert "데이터없음" in combined


# ─────────────────────────────────────────────────────────
# 섹션 5: 에이전트별 상세 분석 (접기)
# ─────────────────────────────────────────────────────────

def test_details_section_uses_collapsible(agents, chief_buy):
    qualified_names = {"macro_economist"}
    lines = _details_section(agents, chief_buy, qualified_names)
    combined = "\n".join(lines)
    assert "<details>" in combined
    assert "</details>" in combined
    assert "<summary>" in combined


def test_details_section_includes_reasoning(agents, chief_buy):
    qualified_names = set()
    lines = _details_section(agents, chief_buy, qualified_names)
    combined = "\n".join(lines)
    assert "근거1" in combined


def test_details_section_qg_marks(agents, chief_buy):
    """[2026-09-15] macro_economist는 상세에서 빠졌으므로 종목 고유 에이전트로 검사한다.

    macro·us_market은 하루 1회 캐시라 종목마다 같은 문단이 반복된다.
    상단 "오늘의 시장 배경"으로 옮겼다 — 투표 집계에는 그대로 남는다.
    """
    qualified_names = {"quant_analyst"}
    lines = _details_section(agents, chief_buy, qualified_names)
    combined = "\n".join(lines)
    assert "✅" in combined
    assert "QG 폴백" in combined
    assert "macro_economist" not in combined, "시장 공통 에이전트는 상세에 없어야 한다"
    assert "us_market_specialist" not in combined


# ─────────────────────────────────────────────────────────
# 섹션 6: 리스크 (top 5)
# ─────────────────────────────────────────────────────────

def test_risk_section_lists_risks(agents, chief_buy):
    lines = _risk_section(agents, chief_buy)
    combined = "\n".join(lines)
    assert "리스크-macro_economist" in combined


def test_risk_section_limits_to_five():
    """5개 초과 리스크 → top 5만."""
    many_agents = [
        AnalysisReport(
            agent_name=f"agent_{i}", confidence=0.8, recommendation="HOLD",
            reasoning=["a", "b", "c"], data_sources=["s1", "s2"],
            prediction_basis=["p1", "p2"],
            risk_factors=[f"unique-risk-{i}"],
        )
        for i in range(10)
    ]
    lines = _risk_section(many_agents, None)
    combined = "\n".join(lines)
    found = sum(1 for i in range(10) if f"unique-risk-{i}" in combined)
    assert found == 5


def test_risk_section_dedup():
    """중복 리스크는 한 번만."""
    a1 = _make_agent("macro_economist", "BUY", 0.7)
    a2 = _make_agent("quant_analyst", "BUY", 0.7)
    # 둘 다 risk_factors=["리스크-..."] 이므로 다름 — 중복 만들기
    a1.risk_factors = ["공통리스크"]
    a2.risk_factors = ["공통리스크"]
    lines = _risk_section([a1, a2], None)
    combined = "\n".join(lines)
    assert combined.count("공통리스크") == 1


def test_risk_section_no_risks():
    lines = _risk_section([], None)
    combined = "\n".join(lines)
    assert "주요 리스크 없음" in combined


# ─────────────────────────────────────────────────────────
# 섹션 7: 푸터
# ─────────────────────────────────────────────────────────

def test_footer_section_has_kst():
    lines = _footer_section()
    combined = "\n".join(lines)
    assert "KST" in combined


def test_footer_section_has_disclaimer():
    lines = _footer_section()
    combined = "\n".join(lines)
    assert "투자 조언 아님" in combined


# ─────────────────────────────────────────────────────────
# 통합 — format_report_v4
# ─────────────────────────────────────────────────────────

def test_format_full_report_sections_order(report_buy):
    """[2026-09-15] 순서를 아침 워크플로(morning-workflow 1절)에 맞췄다.

    판단 → 액션 → 리스크 → 상세 → 시장 → 포트폴리오.
    예전에는 포트폴리오·시장 지표가 앞에 있어 결정에 필요한 것이 뒤로 밀렸다.
    """
    ai_idx  = report_buy.find("🤖 AI 분석단 의견")
    act_idx = report_buy.find("🎯 오늘의 액션 플랜")
    rsk_idx = report_buy.find("⚠️ 오늘의 주요 리스크")
    det_idx = report_buy.find("📋 에이전트별 상세 분석")
    mkt_idx = report_buy.find("📊 시장 지표")
    pos_idx = report_buy.find("💼 내 포트폴리오 현황")
    assert 0 < ai_idx < act_idx < rsk_idx < det_idx < mkt_idx < pos_idx
    assert report_buy.find("📌 오늘:") < ai_idx, "오늘 한 줄이 맨 위"


def test_format_report_no_portfolio_safe(report_no_portfolio):
    """portfolio_summary=None이어도 정상 출력."""
    assert "보유 포지션 없음" in report_no_portfolio
    assert "최종 판단" in report_no_portfolio


def test_format_report_no_market_data_safe(agents, chief_hold, qualified):
    """market_data=None → '데이터 없음' 처리."""
    r = format_report_v4(
        ticker="005930", regime="bull", strategy="HOLD",
        final=chief_hold, agents=agents, qualified_reports=qualified,
        market_data=None,
    )
    assert "시장 지표 데이터 없음" in r


def test_format_report_none_chief_safe(agents, qualified, portfolio_with_positions):
    """final=None → 액션 플랜에서 안전 처리."""
    r = format_report_v4(
        ticker="005930", regime="bull", strategy="HOLD",
        final=None, agents=agents, qualified_reports=qualified,
        portfolio_summary=portfolio_with_positions,
    )
    assert "오늘 신규 진입 추천 없음" in r


def test_format_report_buy_appears_in_title(report_buy):
    assert "_BUY" in report_buy


def test_format_report_filename_safe_title_chars(report_buy):
    """제목에 경로/와일드카드 문자가 없어야 함. 콜론(`:`)은 HH:MM 표기를 위해 허용."""
    title_line = report_buy.split("\n", 1)[0]
    for bad in ["/", "\\", "*", "?", '"', "<", ">", "|"]:
        assert bad not in title_line, f"제목에 '{bad}' 문자 발견: {title_line}"
