"""
tests/test_mcp_result_and_schema.py

2순위 회귀 테스트 — 무음 실패 제거.

[배경]
  ① MCP 서버 3개가 실패를 `return {"error": ...}` 로 돌려주는데(35곳),
     호출부에서 error 키를 검사하는 곳은 regime_detector.py 한 군데뿐이었다.
     오류 메시지가 로그에도 안 남고 "데이터 부족"으로 치환되어 프롬프트에 들어갔다.

  ② AnalysisReport가 reasoning≥3 / prediction_basis≥2를 무조건 강제해서,
     입력이 비었을 때도 LLM이 "정량 근거" 2개를 만들어야 스키마를 통과할 수 있었다.
     프롬프트로는 "지어내지 마"라고 하면서 스키마로는 "채워라"라고 한 셈이다.
"""

import json
import logging

import pytest
from pydantic import ValidationError

from src.schemas.agent_output import AnalysisReport
from src.utils import mcp_result as mr


# ─────────────────────────────────────────────────────────
# 1. MCP 결과 파싱 — 오류가 로그에 남는가
# ─────────────────────────────────────────────────────────

class _Structured:
    def __init__(self, payload):
        self.structured_content = payload
        self.content = None


@pytest.fixture(autouse=True)
def clean_counts():
    mr.reset_error_counts()
    yield
    mr.reset_error_counts()


def test_error_dict_is_logged_not_swallowed(caplog):
    """MCP가 error를 돌려주면 반드시 error 레벨 로그가 남아야 한다."""
    raw = _Structured({"error": "ticker 005930에 대한 데이터가 없습니다."})

    with caplog.at_level(logging.ERROR):
        out = mr.parse_mcp_result(raw, tool="get_stock_price", context="005930")

    assert mr.has_error(out)
    assert any("get_stock_price(005930)" in r.message for r in caplog.records)
    assert any("데이터가 없습니다" in r.message for r in caplog.records)


def test_error_level_not_warning(caplog):
    """warning이면 CI 로그에서 묻힌다 — 46일간 그랬다. error여야 한다."""
    with caplog.at_level(logging.WARNING):
        mr.parse_mcp_result(_Structured({"error": "x"}), tool="t")
    assert [r.levelno for r in caplog.records] == [logging.ERROR]


def test_exception_input_becomes_error_dict(caplog):
    """gather(return_exceptions=True)가 넘긴 예외도 오류로 기록된다."""
    with caplog.at_level(logging.ERROR):
        out = mr.parse_mcp_result(TimeoutError("30s 초과"), tool="get_vix")
    assert mr.has_error(out)
    assert "TimeoutError" in out["error"]
    assert any("get_vix" in r.message for r in caplog.records)


def test_success_passes_through_untouched(caplog):
    payload = {"ticker": "005930", "latest_close": 70000, "ohlcv": {"2026-09-08": {}}}
    with caplog.at_level(logging.ERROR):
        out = mr.parse_mcp_result(_Structured(payload), tool="get_stock_price")
    assert out == payload
    assert not mr.has_error(out)
    assert caplog.records == []


def test_error_counts_accumulate():
    mr.parse_mcp_result(_Structured({"error": "a"}), tool="get_stock_price")
    mr.parse_mcp_result(_Structured({"error": "b"}), tool="get_stock_price")
    mr.parse_mcp_result(_Structured({"error": "c"}), tool="get_vix")
    assert mr.get_error_counts() == {"get_stock_price": 2, "get_vix": 1}


def test_failure_note_is_explicit():
    note = mr.failure_note({"error": "KRX 로그인 실패"}, what="주가")
    assert note is not None
    assert "존재하지 않는다" in note
    assert mr.failure_note({"ok": 1}, what="주가") is None


# ─────────────────────────────────────────────────────────
# 2. 스키마 — 날조 강제가 풀렸는가
# ─────────────────────────────────────────────────────────

def _report(**kw):
    base = dict(
        agent_name="quant_analyst", confidence=0.7, recommendation="HOLD",
        reasoning=["a", "b", "c"], data_sources=["x", "y"],
        prediction_basis=["p", "q"], risk_factors=["z"],
    )
    base.update(kw)
    return AnalysisReport(**base)


def test_normal_report_still_requires_depth():
    """정상 분석의 품질 기준은 그대로여야 한다 — 완화가 아니라 조건부."""
    with pytest.raises(ValidationError, match="reasoning은 3단계 이상"):
        _report(reasoning=["하나만"])

    with pytest.raises(ValidationError, match="prediction_basis는 2개 이상"):
        _report(prediction_basis=["하나만"])


def test_data_insufficient_allows_minimal_report():
    """
    핵심 — 데이터가 없을 때 날조 없이도 스키마를 통과할 수 있어야 한다.
    이전에는 이 조합이 ValidationError였고, 재시도 루프가 LLM에게
    '정량 근거 2개'를 만들어내도록 압박했다.
    """
    r = _report(
        data_sufficient=False,
        confidence=0.0,
        reasoning=["MCP 수집 실패로 모멘텀·괴리율 수치가 없음"],
        prediction_basis=["데이터 수집 실패로 정량 근거 없음"],
    )
    assert r.data_sufficient is False
    assert len(r.reasoning) == 1
    assert len(r.prediction_basis) == 1


def test_data_sufficient_defaults_to_true():
    """기존 코드가 이 필드를 모르고 만들어도 종전과 동일하게 동작한다."""
    assert _report().data_sufficient is True


def test_empty_reasoning_still_rejected():
    """탈출구가 '아무것도 안 써도 된다'는 뜻은 아니다."""
    with pytest.raises(ValidationError):
        _report(data_sufficient=False, reasoning=[], prediction_basis=["x"])


# ─────────────────────────────────────────────────────────
# 3. quant 프롬프트 — 실패가 명시되는가
# ─────────────────────────────────────────────────────────

def test_prompt_marks_collection_failure_explicitly():
    from src.agents.quant_analyst import _format_prompt

    prompt = _format_prompt({
        "005930": {
            "name": "삼성전자",
            "momentum": {"error": "데이터 부족"},
            "analyst": {"target_price": None, "upside_pct": None, "report_count": 0},
        }
    })

    assert "수집 실패" in prompt
    assert "존재하지 않는다" in prompt
    assert "data_sufficient=false" in prompt
    # 실패를 정상 데이터처럼 json으로 흘려보내면 안 된다
    assert '"error": "데이터 부족"' not in prompt


def test_prompt_normal_path_unchanged():
    from src.agents.quant_analyst import _format_prompt

    prompt = _format_prompt({
        "005930": {
            "name": "삼성전자",
            "momentum": {"current_price": 70000, "momentum_20d_pct": 2.79,
                         "momentum_5d_pct": 1.1, "volume_ratio": 1.2},
            "analyst": {"target_price": 85000, "upside_pct": 21.4, "report_count": 5},
        }
    })

    assert "수집 실패" not in prompt
    assert "momentum_20d_pct" in prompt
    assert "정량적 투자 매력도를 평가" in prompt


# ─────────────────────────────────────────────────────────
# 4. 나머지 에이전트 프롬프트 — 수집 실패가 '없음'과 구분되는가
#    (TradingAgents v0.5.0 "a vendor failure is reported as a vendor failure"와 같은 문제.
#     failure_note는 만들어져 있었지만 호출처가 0곳이었다 — 2026-09-25)
# ─────────────────────────────────────────────────────────

def test_kr_prompt_marks_each_failure():
    """이전: 실패한 항목은 아무 말 없이 줄이 빠졌다."""
    from src.agents.kr_market_specialist import _format_prompt

    prompt = _format_prompt({"005930": {
        "price": {"error": "E-주가"}, "trends": {"error": "E-수급"}, "reports": {"error": "E-리포트"},
    }})

    for what, reason in (("주가", "E-주가"), ("수급", "E-수급"), ("애널리스트 리포트", "E-리포트")):
        assert f"{what} 수집 실패: {reason}" in prompt


def test_fundamental_failure_not_shown_as_absence():
    """이전: 컨센서스 실패는 '데이터 없음', 공시 실패는 '공시: 없음'으로 나갔다."""
    from src.agents.fundamental_analyst import _format_prompt

    ok_fin = {"per": 10, "pbr": 1.2, "eps": 5000, "div_yield": 2.0}
    failed = _format_prompt({"005930": {
        "financials": ok_fin, "price": {},
        "consensus": {"error": "E-컨센서스"}, "disclosure": {"error": "E-DART"},
    }})
    assert "컨센서스 수집 실패: E-컨센서스" in failed
    assert "컨센서스: 데이터 없음" not in failed
    assert "공시 수집 실패: E-DART" in failed
    assert "공시: 없음" not in failed

    # 진짜로 공시가 0건인 경우는 그대로 '없음'이어야 한다
    empty = _format_prompt({"005930": {
        "financials": ok_fin, "price": {},
        "consensus": {"revision_direction": "상향", "latest_tp": 90000, "sample_count": 3},
        "disclosure": {"disclosures": []},
    }})
    assert "공시: 없음" in empty
    assert "수집 실패" not in empty


def test_macro_prompt_survives_whole_section_failure():
    """이전: 국채·원자재 결과 자체가 오류면 'N/A':+.2f / str.get 에서 예외가 났다."""
    from src.agents.macro_economist import _format_prompt

    prompt = _format_prompt({k: {"error": f"E-{k}"} for k in (
        "treasury_yields", "vix", "commodities", "exchange_rate", "interest_rate", "policy_news")})

    for what in ("미 국채 수익률", "VIX", "원자재", "환율", "기준금리", "정책 뉴스"):
        assert f"{what} 수집 실패" in prompt


def test_us_prompt_does_not_dump_error_json():
    """이전: {"error": ...}를 json.dumps로 그대로 넣어 LLM이 데이터로 읽었다."""
    from src.agents.us_market_specialist import _format_prompt

    prompt = _format_prompt({
        "sp500": {"error": "E-sp500"}, "vix": {"close": 15.0}, "treasury": {"error": "E-tsy"},
        "bigtech": {"AAPL": {"error": "E-aapl"}, "MSFT": {"price": 400}},
    })
    assert "S&P500 수집 실패: E-sp500" in prompt
    assert "미국 국채금리 수집 실패: E-tsy" in prompt
    assert "AAPL 수집 실패: E-aapl" in prompt
    assert '"error"' not in prompt
    assert '"close": 15.0' in prompt          # 정상 항목은 그대로


def test_technical_prompt_does_not_dump_error_json():
    from src.agents.technical_analyst import _format_prompt

    prompt = _format_prompt({"005930": {"error": "데이터 부족 (10일, 최소 52일 필요)"}})
    assert "기술적 지표 수집 실패: 데이터 부족" in prompt
    assert '"error"' not in prompt


# ─────────────────────────────────────────────────────────
# 5. 수집 단계 — 예외가 로그 없이 {} 로 사라지지 않는가
# ─────────────────────────────────────────────────────────

class _FailingClient:
    """모든 도구 호출이 예외를 던지는 가짜 MCP 클라이언트."""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, args):
        raise RuntimeError(f"{name} 다운")


@pytest.mark.asyncio
@pytest.mark.parametrize("module, collect", [
    ("src.agents.us_market_specialist", "_collect_us_market_data"),
    ("src.agents.quant_analyst", "_collect_quant_data"),
    ("src.agents.technical_analyst", "_collect_technical_data"),
])
async def test_collect_exceptions_are_logged(monkeypatch, module, collect):
    """이전: `parse(x) if not isinstance(x, Exception) else {}` — 예외가 로그 없이 빈 dict가 됐다."""
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "mcp_client", lambda *a, **k: _FailingClient())

    fn = getattr(mod, collect)
    if collect == "_collect_us_market_data":
        data = await fn()
        assert mr.has_error(data["sp500"])
    else:
        await fn(target_ticker="005930")

    assert sum(mr.get_error_counts().values()) > 0
