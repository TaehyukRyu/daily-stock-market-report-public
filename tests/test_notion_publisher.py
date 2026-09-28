"""
tests/test_notion_publisher.py

notion_publisher v4.1 단위 테스트 — Notion API 호출 없음.
build_v4_blocks()와 _markdown_to_blocks() 블록 구조 검증.

7섹션 구조:
  헤더 callout → 💼 포트폴리오 → 🎯 액션플랜 → 📊 시장지표
  → 🤖 AI 분석단 → 📋 상세분석 toggle → ⚠️ 리스크 numbered_list
"""

import pytest
from src.graph.notion_publisher import (
    build_v4_blocks,
    _markdown_to_blocks,
    _build_portfolio_blocks,
    _build_action_plan_blocks,
    _build_market_data_blocks,
    _build_agent_summary_blocks,
    _build_rationale_toggle,
    _build_risk_blocks,
    _rich_text,
    _parse_bold,
    _block,
    _divider,
    _callout,
    _toggle,
    _heading,
    _bullet,
    _numbered,
    _paragraph,
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
        _make_agent("macro_economist", "BUY",  0.75),
        _make_agent("quant_analyst",   "HOLD", 0.60),
        _make_agent("sentiment_analyst","SELL", 0.55),
    ]


@pytest.fixture
def chief_buy() -> AnalysisReport:
    return AnalysisReport(
        agent_name="chief_strategist",
        recommendation="BUY",
        confidence=0.78,
        reasoning=["진입 근거 한줄", "이유2", "이유3"],
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
        reasoning=["방향성 합의 부재", "이유2", "이유3"],
        data_sources=["소스X", "소스Y"],
        prediction_basis=["숫자1", "숫자2"],
        risk_factors=["리스크-chief-hold"],
    )


@pytest.fixture
def qualified(agents) -> list[AnalysisReport]:
    return agents[:2]


@pytest.fixture
def portfolio() -> dict:
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
def market_data() -> dict:
    return {
        "kospi": 3000.50,
        "vix": 18.43,
        "us_10y_yield": 4.59,
        "usd_krw": 1382.0,
        "wti": 105.42,
    }


@pytest.fixture
def blocks_buy(agents, chief_buy, qualified, portfolio, market_data) -> list[dict]:
    return build_v4_blocks(
        ticker="005930",
        regime="VOLATILE",
        strategy="BUY",
        chief_report=chief_buy,
        qualified_reports=qualified,
        all_reports=agents + [chief_buy],
        debate_summary="Bull: 강세. Bear: 과열.",
        error_log=[],
        portfolio_summary=portfolio,
        market_data=market_data,
    )


@pytest.fixture
def blocks_hold(agents, chief_hold, qualified) -> list[dict]:
    return build_v4_blocks(
        ticker="005930",
        regime="VOLATILE",
        strategy="HOLD",
        chief_report=chief_hold,
        qualified_reports=qualified,
        all_reports=agents + [chief_hold],
        debate_summary="",
        error_log=[],
    )


# ─────────────────────────────────────────────────────────
# 헬퍼: 블록 내 텍스트 추출
# ─────────────────────────────────────────────────────────

def _block_text(block: dict) -> str:
    """블록의 rich_text content를 평탄화해 단일 문자열로 반환."""
    t = block["type"]
    if t == "callout":
        chunks = block["callout"]["rich_text"]
    elif t == "toggle":
        chunks = block["toggle"]["rich_text"]
    elif t.startswith("heading_"):
        chunks = block[t]["rich_text"]
    elif t == "bulleted_list_item":
        chunks = block["bulleted_list_item"]["rich_text"]
    elif t == "numbered_list_item":
        chunks = block["numbered_list_item"]["rich_text"]
    elif t == "paragraph":
        chunks = block["paragraph"]["rich_text"]
    else:
        return ""
    return "".join(c["text"]["content"] for c in chunks)


# ─────────────────────────────────────────────────────────
# _rich_text 유틸 테스트
# ─────────────────────────────────────────────────────────

def test_rich_text_basic():
    rt = _rich_text("hello")
    assert isinstance(rt, list)
    assert rt[0]["text"]["content"] == "hello"


def test_rich_text_bold():
    rt = _rich_text("bold", bold=True)
    assert rt[0]["annotations"]["bold"] is True


def test_rich_text_chunking():
    long_text = "x" * 4500
    rt = _rich_text(long_text)
    assert len(rt) == 3
    for chunk in rt:
        assert len(chunk["text"]["content"]) <= 2000


def test_rich_text_empty():
    rt = _rich_text("")
    assert len(rt) == 1
    assert rt[0]["text"]["content"] == ""


# ─────────────────────────────────────────────────────────
# _parse_bold 유틸 테스트
# ─────────────────────────────────────────────────────────

def test_parse_bold_mixed():
    rt = _parse_bold("일반 **굵게** 일반2")
    texts = [chunk["text"]["content"] for chunk in rt]
    bolds = [chunk["annotations"]["bold"] for chunk in rt]
    assert "굵게" in texts
    bold_idx = texts.index("굵게")
    assert bolds[bold_idx] is True


def test_parse_bold_no_bold():
    rt = _parse_bold("그냥 텍스트")
    assert all(not c["annotations"]["bold"] for c in rt)


# ─────────────────────────────────────────────────────────
# Block 유틸 테스트
# ─────────────────────────────────────────────────────────

def test_divider_structure():
    d = _divider()
    assert d["type"] == "divider"
    assert d["object"] == "block"


def test_callout_structure():
    c = _callout("경고 텍스트", emoji="⚠️")
    assert c["type"] == "callout"
    assert c["callout"]["icon"]["emoji"] == "⚠️"
    assert any("경고 텍스트" in chunk["text"]["content"]
               for chunk in c["callout"]["rich_text"])


def test_toggle_structure():
    children = [_divider()]
    t = _toggle("제목", children)
    assert t["type"] == "toggle"
    assert t["toggle"]["children"] == children
    assert t["toggle"]["rich_text"][0]["text"]["content"] == "제목"


def test_heading_levels():
    for level in (1, 2, 3):
        h = _heading(level, "테스트")
        assert h["type"] == f"heading_{level}"


def test_bullet_structure():
    b = _bullet("항목")
    assert b["type"] == "bulleted_list_item"


def test_numbered_structure():
    n = _numbered("1번 항목")
    assert n["type"] == "numbered_list_item"


def test_paragraph_structure():
    p = _paragraph("문단 텍스트")
    assert p["type"] == "paragraph"


# ─────────────────────────────────────────────────────────
# _build_portfolio_blocks
# ─────────────────────────────────────────────────────────

def test_portfolio_blocks_empty():
    blocks = _build_portfolio_blocks(None)
    assert any("보유 포지션 없음" in _block_text(b) for b in blocks)


def test_portfolio_blocks_empty_positions():
    blocks = _build_portfolio_blocks({"positions": []})
    assert any("보유 포지션 없음" in _block_text(b) for b in blocks)


def test_portfolio_blocks_header_present(portfolio):
    blocks = _build_portfolio_blocks(portfolio)
    headings = [b for b in blocks if b["type"] == "heading_2"]
    assert any("내 포트폴리오 현황" in _block_text(h) for h in headings)


def test_portfolio_blocks_callout_color_green(portfolio):
    """누적 수익 양수 → green_background callout."""
    blocks = _build_portfolio_blocks(portfolio)
    callouts = [b for b in blocks if b["type"] == "callout"]
    assert any(c["callout"]["color"] == "green_background" for c in callouts)


def test_portfolio_blocks_callout_color_red():
    summary = {
        "invested_pct": 50, "available_pct": 50,
        "cumulative_pnl_seed_pct": -3.5,
        "positions": [{
            "ticker": "005930", "ticker_name": "삼성전자",
            "entry_date": "2026-05-10", "entry_price": 100000, "current_price": 90000,
            "allocation_pct": 50, "position_pnl_pct": -10.0, "seed_pnl_pct": -5.0,
            "stop_loss_price": 93000, "take_profit_1": None, "take_profit_2": None,
            "holding_days": 3, "holding_period_weeks": 2, "rr_ratio": 1.5, "alert": None,
        }],
        "closed_recent": [],
    }
    blocks = _build_portfolio_blocks(summary)
    callouts = [b for b in blocks if b["type"] == "callout"]
    assert any(c["callout"]["color"] == "red_background" for c in callouts)


def test_portfolio_blocks_position_bullets(portfolio):
    blocks = _build_portfolio_blocks(portfolio)
    bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
    texts = [_block_text(b) for b in bullets]
    assert any("삼성전자" in t for t in texts)
    assert any("265,000" in t for t in texts)


def test_portfolio_blocks_alert_stop_loss_prefix():
    summary = {
        "invested_pct": 20, "available_pct": 80, "cumulative_pnl_seed_pct": 0,
        "positions": [{
            "ticker": "005930", "ticker_name": "삼성전자",
            "entry_date": "2026-05-10", "entry_price": 100000, "current_price": 90000,
            "allocation_pct": 20, "position_pnl_pct": -10.0, "seed_pnl_pct": -2.0,
            "stop_loss_price": 93000, "take_profit_1": None, "take_profit_2": None,
            "holding_days": 3, "holding_period_weeks": 2, "rr_ratio": 1.5,
            "alert": "stop_loss",
        }],
        "closed_recent": [],
    }
    blocks = _build_portfolio_blocks(summary)
    bullets = [_block_text(b) for b in blocks if b["type"] == "bulleted_list_item"]
    assert any("⚠️" in t for t in bullets)


def test_portfolio_blocks_alert_target_prefix():
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
    blocks = _build_portfolio_blocks(summary)
    bullets = [_block_text(b) for b in blocks if b["type"] == "bulleted_list_item"]
    assert any("🎯" in t for t in bullets)


def test_portfolio_blocks_closed_recent_heading(portfolio):
    blocks = _build_portfolio_blocks(portfolio)
    headings = [b for b in blocks if b["type"] == "heading_3"]
    assert any("청산 이력" in _block_text(h) for h in headings)


# ─────────────────────────────────────────────────────────
# _build_action_plan_blocks
# ─────────────────────────────────────────────────────────

def test_action_plan_blocks_buy_callout(chief_buy, portfolio):
    blocks = _build_action_plan_blocks(chief_buy, portfolio, "005930")
    callouts = [b for b in blocks if b["type"] == "callout"]
    green = [c for c in callouts if c["callout"]["color"] == "green_background"]
    assert green


def test_action_plan_blocks_buy_includes_entry_price(chief_buy, portfolio):
    blocks = _build_action_plan_blocks(chief_buy, portfolio, "005930")
    texts = [_block_text(b) for b in blocks]
    combined = "\n".join(texts)
    assert "300,000" in combined


def test_action_plan_blocks_buy_includes_allocation(chief_buy, portfolio):
    blocks = _build_action_plan_blocks(chief_buy, portfolio, "005930")
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "시드의 20%" in combined


def test_action_plan_blocks_buy_includes_rr_weeks(chief_buy, portfolio):
    blocks = _build_action_plan_blocks(chief_buy, portfolio, "005930")
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "2주" in combined
    assert "2.0" in combined


def test_action_plan_blocks_cash_shortage_warning():
    chief = AnalysisReport(
        agent_name="chief_strategist", recommendation="BUY", confidence=0.7,
        reasoning=["a", "b", "c"], data_sources=["s1", "s2"],
        prediction_basis=["p1", "p2"], risk_factors=["r1"],
        entry_price=100000, stop_loss=95000, stop_loss_pct=-5,
        take_profit_1=110000, take_profit_2=120000, rr_ratio=2.0,
        position_size_pct=50.0,
        holding_period_weeks=2,
    )
    portfolio_short = {
        "invested_pct": 70, "available_pct": 30,
        "cumulative_pnl_seed_pct": 0, "positions": [], "closed_recent": [],
    }
    blocks = _build_action_plan_blocks(chief, portfolio_short, "005930")
    callouts = [b for b in blocks if b["type"] == "callout"]
    red = [c for c in callouts if c["callout"]["color"] == "red_background"]
    assert red
    assert any("초과" in _block_text(c) for c in red)


def test_action_plan_blocks_no_buy_recommendation(chief_hold, portfolio):
    blocks = _build_action_plan_blocks(chief_hold, portfolio, "005930")
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "오늘 신규 진입 추천 없음" in combined


def test_action_plan_blocks_other_ticker_monitoring(chief_buy, portfolio):
    """분석 대상 ticker가 보유 종목과 다를 때 모니터링 표시."""
    blocks = _build_action_plan_blocks(chief_buy, portfolio, "000660")
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "모니터링" in combined


def test_action_plan_blocks_existing_position_hold(chief_hold, portfolio):
    blocks = _build_action_plan_blocks(chief_hold, portfolio, "005930")
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "유지" in combined


# ─────────────────────────────────────────────────────────
# _build_market_data_blocks
# ─────────────────────────────────────────────────────────

def test_market_data_blocks_none():
    blocks = _build_market_data_blocks(None)
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "시장 지표 데이터 없음" in combined


def test_market_data_blocks_empty():
    blocks = _build_market_data_blocks({})
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "시장 지표 데이터 없음" in combined


def test_market_data_blocks_indicators(market_data):
    blocks = _build_market_data_blocks(market_data)
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "KOSPI" in combined
    assert "공포지수(VIX)" in combined
    assert "미 10년 금리" in combined


def test_market_data_blocks_vix_high_interp():
    blocks = _build_market_data_blocks({"vix": 35.0})
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "공포" in combined


def test_market_data_blocks_yield_high_interp():
    blocks = _build_market_data_blocks({"us_10y_yield": 5.0})
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "주식 부담" in combined


# ─────────────────────────────────────────────────────────
# _build_agent_summary_blocks
# ─────────────────────────────────────────────────────────

def test_agent_summary_blocks_agents(agents, chief_buy):
    blocks = _build_agent_summary_blocks(agents, chief_buy, {"macro_economist"})
    combined = "\n".join(_block_text(b) for b in blocks)
    # 한글 이름으로 렌더되는지
    assert "매크로 분석" in combined or "macro_economist" in combined


def test_agent_summary_blocks_final_label(agents, chief_buy):
    blocks = _build_agent_summary_blocks(agents, chief_buy, set())
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "최종 판단" in combined


def test_agent_summary_blocks_low_conf_note():
    low = _make_agent("macro_economist", "HOLD", 0.40)
    blocks = _build_agent_summary_blocks([low], None, set())
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "참고용" in combined


def test_agent_summary_blocks_zero_conf_note():
    zero = _make_agent("macro_economist", "HOLD", 0.0)
    blocks = _build_agent_summary_blocks([zero], None, set())
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "데이터없음" in combined


# ─────────────────────────────────────────────────────────
# _build_rationale_toggle (섹션 5)
# ─────────────────────────────────────────────────────────

def test_rationale_toggle_returns_toggle(agents, chief_buy):
    t = _build_rationale_toggle(agents, chief_buy, set())
    assert t["type"] == "toggle"
    assert "에이전트별 상세 분석" in _block_text(t)


def test_rationale_toggle_has_children(agents, chief_buy):
    t = _build_rationale_toggle(agents, chief_buy, set())
    assert len(t["toggle"]["children"]) > 0


# ─────────────────────────────────────────────────────────
# _build_risk_blocks
# ─────────────────────────────────────────────────────────

def test_risk_blocks_numbered_list(agents, chief_buy):
    blocks = _build_risk_blocks(agents, chief_buy)
    numbered = [b for b in blocks if b["type"] == "numbered_list_item"]
    assert len(numbered) > 0


def test_risk_blocks_limit_to_five():
    many = [
        AnalysisReport(
            agent_name=f"agent_{i}", confidence=0.8, recommendation="HOLD",
            reasoning=["a", "b", "c"], data_sources=["s1", "s2"],
            prediction_basis=["p1", "p2"],
            risk_factors=[f"unique-risk-{i}"],
        )
        for i in range(10)
    ]
    blocks = _build_risk_blocks(many, None)
    numbered = [b for b in blocks if b["type"] == "numbered_list_item"]
    assert len(numbered) == 5


def test_risk_blocks_empty():
    blocks = _build_risk_blocks([], None)
    combined = "\n".join(_block_text(b) for b in blocks)
    assert "주요 리스크 없음" in combined


# ─────────────────────────────────────────────────────────
# build_v4_blocks 구조 검증
# ─────────────────────────────────────────────────────────

def test_build_returns_list(blocks_buy):
    assert isinstance(blocks_buy, list)


def test_build_nonempty(blocks_buy):
    assert len(blocks_buy) > 0


def test_build_all_have_type(blocks_buy):
    for b in blocks_buy:
        assert "type" in b


def test_build_header_callout(blocks_buy):
    """첫 번째 블록은 헤더 callout이며 'Daily stock market report' 포함."""
    first = blocks_buy[0]
    assert first["type"] == "callout"
    text = _block_text(first)
    assert "Daily stock market report" in text


def test_build_header_callout_includes_regime_kr(blocks_buy):
    first = blocks_buy[0]
    text = _block_text(first)
    assert "변동성 장세" in text


def test_build_header_callout_includes_strategy_kr(blocks_buy):
    first = blocks_buy[0]
    text = _block_text(first)
    assert "매수" in text


def test_build_has_section_dividers(blocks_buy):
    dividers = [b for b in blocks_buy if b["type"] == "divider"]
    # 헤더 + 6개 섹션 사이 = 최소 5개
    assert len(dividers) >= 5


def test_build_has_seven_section_headings(blocks_buy):
    """heading_2가 5개 이상 (포트폴리오/액션/시장/AI/리스크)."""
    h2 = [b for b in blocks_buy if b["type"] == "heading_2"]
    assert len(h2) >= 5


def test_build_section_order(blocks_buy):
    """heading_2의 순서가 섹션 순서를 따른다."""
    h2_texts = [_block_text(b) for b in blocks_buy if b["type"] == "heading_2"]
    expected_keywords = ["포트폴리오", "액션 플랜", "시장 지표", "AI 분석단", "리스크"]
    j = 0
    for h in h2_texts:
        if j < len(expected_keywords) and expected_keywords[j] in h:
            j += 1
    assert j == len(expected_keywords)


def test_build_buy_has_green_action_callout(blocks_buy):
    callouts = [b for b in blocks_buy if b["type"] == "callout"]
    green = [c for c in callouts if c["callout"]["color"] == "green_background"]
    assert len(green) >= 1


def test_build_buy_entry_price_in_callout(blocks_buy):
    callouts = [b for b in blocks_buy if b["type"] == "callout"]
    green = [c for c in callouts if c["callout"]["color"] == "green_background"]
    found = False
    for c in green:
        if "300,000" in _block_text(c):
            found = True
            break
    assert found


def test_build_has_rationale_toggle(blocks_buy):
    toggles = [b for b in blocks_buy if b["type"] == "toggle"]
    assert any("상세 분석" in _block_text(t) for t in toggles)


def test_build_risk_section_numbered_list(blocks_buy):
    numbered = [b for b in blocks_buy if b["type"] == "numbered_list_item"]
    assert len(numbered) >= 1


def test_build_hold_no_green_action_callout(blocks_hold):
    """HOLD에선 신규 매수 green callout 없어야 함."""
    callouts = [b for b in blocks_hold if b["type"] == "callout"]
    green = [c for c in callouts if c["callout"]["color"] == "green_background"]
    # 헤더는 blue. 포트폴리오 없으면 신규 매수 callout 없음.
    assert len(green) == 0


def test_build_no_chief_in_agents_section(agents, chief_buy, qualified):
    """all_reports에 chief가 섞여도 분리되어 헤더 행 1개만 등장."""
    blocks = build_v4_blocks(
        ticker="005930", regime="bull", strategy="BUY",
        chief_report=chief_buy,
        qualified_reports=qualified,
        all_reports=agents + [chief_buy],
    )
    bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
    chief_lines = [b for b in bullets if "최종 판단" in _block_text(b)]
    assert len(chief_lines) == 1


def test_build_without_structured_data_no_crash():
    """chief_report=None, all_reports=[] → 빈 블록 목록이어도 안전해야 함."""
    blocks = build_v4_blocks(
        ticker="005930", regime="neutral", strategy="HOLD",
        chief_report=None,
        all_reports=[],
    )
    assert isinstance(blocks, list)
    assert len(blocks) > 0


def test_build_with_portfolio_renders_position_bullet(blocks_buy):
    bullets = [b for b in blocks_buy if b["type"] == "bulleted_list_item"]
    texts = [_block_text(b) for b in bullets]
    assert any("삼성전자" in t for t in texts)


def test_build_with_market_data_renders_indicators(blocks_buy):
    combined = "\n".join(_block_text(b) for b in blocks_buy)
    assert "KOSPI" in combined
    assert "공포지수(VIX)" in combined


# ─────────────────────────────────────────────────────────
# _markdown_to_blocks fallback 테스트
# ─────────────────────────────────────────────────────────

def test_markdown_heading1():
    blocks = _markdown_to_blocks("# 제목1")
    h1 = [b for b in blocks if b["type"] == "heading_1"]
    assert len(h1) == 1
    assert h1[0]["heading_1"]["rich_text"][0]["text"]["content"] == "제목1"


def test_markdown_heading2():
    blocks = _markdown_to_blocks("## 섹션")
    h2 = [b for b in blocks if b["type"] == "heading_2"]
    assert len(h2) == 1


def test_markdown_heading3():
    blocks = _markdown_to_blocks("### 소섹션")
    h3 = [b for b in blocks if b["type"] == "heading_3"]
    assert len(h3) == 1


def test_markdown_bullet():
    blocks = _markdown_to_blocks("- 항목1\n- 항목2")
    bullets = [b for b in blocks if b["type"] == "bulleted_list_item"]
    assert len(bullets) == 2


def test_markdown_divider():
    blocks = _markdown_to_blocks("---")
    dividers = [b for b in blocks if b["type"] == "divider"]
    assert len(dividers) == 1


def test_markdown_paragraph():
    blocks = _markdown_to_blocks("일반 텍스트 입니다.")
    paras = [b for b in blocks if b["type"] == "paragraph"]
    assert len(paras) >= 1


def test_markdown_bold_inline():
    blocks = _markdown_to_blocks("**굵게** 일반")
    paras = [b for b in blocks if b["type"] == "paragraph"]
    assert paras
    rich = paras[0]["paragraph"]["rich_text"]
    bold_chunks = [c for c in rich if c["annotations"]["bold"]]
    assert bold_chunks


def test_markdown_callout_warning():
    blocks = _markdown_to_blocks("⚠️ 주의 사항")
    callouts = [b for b in blocks if b["type"] == "callout"]
    assert len(callouts) == 1


def test_markdown_empty_string():
    blocks = _markdown_to_blocks("")
    assert isinstance(blocks, list)


def test_markdown_mixed():
    md = "# 제목\n\n## 섹션\n\n- 항목\n\n---\n\n일반"
    blocks = _markdown_to_blocks(md)
    types = [b["type"] for b in blocks]
    assert "heading_1" in types
    assert "heading_2" in types
    assert "bulleted_list_item" in types
    assert "divider" in types
    assert "paragraph" in types
