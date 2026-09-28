"""
tests/test_ops_recovery.py

2026-09-11 운영 복구 회귀 테스트.
  1. 네이버 뉴스 JSON API 파싱 (헤드라인) + LAST_HEADLINE_COUNT
  2. 네이버 증권 integration/trend API 파싱 (재무·수급)
  3. 건강도: 크레딧 400 / 헤드라인 0건 / MCP 키 누락
외부 호출 없음.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest


# ── 1. 헤드라인 ────────────────────────────────────────────

NEWS_JSON = [
    {"tit": "\"주가 반토막 났는데 쓸어담았다\"…외국인 순매수 1위 '반전'", "dt": "20260911154712", "ohnm": "한국경제"},
    {"tit": "삼성전자, 3분기 실적 컨센서스 상향", "dt": "20260911150000", "ohnm": "연합"},
    {"tit": "삼성전자, 3분기 실적 컨센서스 상향", "dt": "20260911150001", "ohnm": "연합"},   # 중복
    {"tit": "짧음", "dt": "20260911"},                                                   # 5자 미만
    "garbage",
]


def test_parse_news_items_dedups_and_filters():
    from src.screening import stage1c_news as st
    titles = st._parse_news_items(NEWS_JSON)
    assert titles == [NEWS_JSON[0]["tit"], NEWS_JSON[1]["tit"]]


def test_fetch_market_headlines_uses_api_and_records_count(monkeypatch):
    from src.screening import stage1c_news as st
    calls = []

    class _Resp:
        def __init__(self, data): self._d = data
        def raise_for_status(self): ...
        def json(self): return self._d

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _Resp([] if "mainnews" in url else NEWS_JSON)      # mainnews 비면 flashnews 폴백

    monkeypatch.setattr(st.requests, "get", fake_get)
    titles = st.fetch_market_headlines()
    assert len(titles) == 2 and st.LAST_HEADLINE_COUNT == 2
    assert "category=mainnews" in calls[0] and "category=flashnews" in calls[1]


def test_fetch_market_headlines_zero_sets_count_zero(monkeypatch):
    from src.screening import stage1c_news as st
    monkeypatch.setattr(st, "_fetch_headlines_api", lambda cat: [])
    assert st.fetch_market_headlines() == [] and st.LAST_HEADLINE_COUNT == 0


# ── 2. 재무·수급 ───────────────────────────────────────────

@pytest.fixture(scope="module")
def krx():
    spec = importlib.util.spec_from_file_location("krx_server_for_test", "src/mcp_servers/krx_market/server.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


INTEGRATION = {
    "stockName": "삼성전자",
    "totalInfos": [
        {"code": "per", "key": "PER", "value": "11.64배"},
        {"code": "eps", "key": "EPS", "value": "22,292원"},
        {"code": "cnsPer", "key": "추정PER", "value": "5.38배"},
        {"code": "cnsEps", "key": "추정EPS", "value": "48,239원"},
        {"code": "pbr", "key": "PBR", "value": "3.02배"},
        {"code": "bps", "key": "BPS", "value": "86,052원"},
        {"code": "dividendYieldRatio", "key": "배당수익률", "value": "0.64%"},
        {"code": "dividend", "key": "주당배당금", "value": "1,668원"},
    ],
    "consensusInfo": {"recommMean": "4.05", "priceTargetMean": "488,409"},
}


def test_parse_integration_maps_old_schema(krx):
    f = krx._parse_integration("005930", INTEGRATION)
    assert f["name"] == "삼성전자" and f["source"] == "naver_stock_api"
    assert f["per"] == 11.64 and f["pbr"] == 3.02 and f["eps"] == 22292
    assert f["bps"] == 86052 and f["dps"] == 1668 and f["div_yield"] == 0.64
    assert f["cns_per"] == 5.38 and f["cns_eps"] == 48239
    assert f["target_price_mean"] == 488409.0 and f["recomm_mean"] == 4.05
    for k in ("ticker", "date", "per", "pbr", "eps", "bps", "dps", "div_yield", "cns_per", "cns_eps"):
        assert k in f, k


def test_parse_integration_errors(krx):
    assert "error" in krx._parse_integration("X", {})
    assert "error" in krx._parse_integration("X", {"stockName": "a", "totalInfos": []})


def test_parse_trend_rows(krx):
    rows = [
        {"bizdate": "20260910", "foreignerPureBuyQuant": "-5,769,453", "organPureBuyQuant": "+4,266,985", "closePrice": "269,000"},
        {"bizdate": "20260909", "foreignerPureBuyQuant": "1,000", "organPureBuyQuant": "-2,000", "closePrice": "269,500"},
        {"bizdate": "20260908", "foreignerPureBuyQuant": None, "organPureBuyQuant": "1", "closePrice": "1"},   # 결측 → 제외
    ]
    recs = krx._parse_trend_rows(rows, days=10)
    assert recs[0] == {"close": 269000, "foreign_net": -5769453, "institution_net": 4266985, "date": "20260910"}
    assert len(recs) == 2
    assert len(krx._parse_trend_rows(rows, days=1)) == 1


# ── 3. 건강도 ──────────────────────────────────────────────

def _payload(reports, errors=()):
    return {"all_reports": reports, "error_log": list(errors)}


def _rep(conf):
    return SimpleNamespace(agent_name="quant_analyst", confidence=conf)


def test_health_flags_credit_error(monkeypatch):
    from src import daily_runner as dr
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: 0)
    h = dr._assess_run_health(
        [_payload([_rep(0.8)], errors=["[NodeError] chief_strategist: BadRequestError: Error code: 400 - "
                                       "Your credit balance is too low to access the Anthropic API."])],
        ["005930"], {"headline_count": 30},
    )
    assert h["healthy"] is False
    assert any("크레딧" in r for r in h["reasons"])


def test_health_flags_zero_headlines(monkeypatch):
    from src import daily_runner as dr
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: 0)
    h = dr._assess_run_health([_payload([_rep(0.8)])], ["005930"], {"headline_count": 0})
    assert h["healthy"] is False and any("헤드라인 0건" in r for r in h["reasons"])
    ok = dr._assess_run_health([_payload([_rep(0.8)])], ["005930"], {"headline_count": 25})
    assert ok["healthy"] is True


def test_health_flags_mcp_key_missing(monkeypatch):
    from src import daily_runner as dr
    from src.utils import mcp_result as mr
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: 0)
    monkeypatch.setattr(mr, "_RECENT_MESSAGES", ["krx_market(005930): DART_API_KEY가 설정되지 않았습니다."])
    h = dr._assess_run_health([_payload([_rep(0.8)])], ["005930"], {"headline_count": 30})
    assert h["healthy"] is False and any("API 키를 못 읽음" in r for r in h["reasons"])


def test_health_without_screen_is_backward_compatible(monkeypatch):
    from src import daily_runner as dr
    monkeypatch.setattr(dr, "_ohlcv_staleness_days", lambda: 0)
    from src.utils import mcp_result as mr
    monkeypatch.setattr(mr, "_RECENT_MESSAGES", [])
    assert dr._assess_run_health([_payload([_rep(0.8)])], ["005930"])["healthy"] is True


# ── 4. CR-2 / CR-9 (2026-09-14) ────────────────────────────

def test_temperature_omitted_for_claude5_models():
    """claude-sonnet-5 계열은 temperature를 거부한다 (400). baseline haiku는 유지."""
    from src.screening import stage1c_news as st
    assert st._supports_temperature("claude-haiku-4-5-20251001") is True
    assert st._supports_temperature("claude-opus-4-6") is True
    assert st._supports_temperature("claude-sonnet-5") is False
    assert st._supports_temperature("claude-opus-5") is False
    assert st._supports_temperature("claude-fable-5-1") is False


def test_stage1c_call_omits_temperature_for_sonnet5(monkeypatch):
    """실제 호출 kwargs에 sonnet-5면 temperature가 없고 haiku면 있다."""
    from src.screening import stage1c_news as st

    captured: list[dict] = []

    class _FakeMessages:
        def create(self, **kwargs):
            captured.append(kwargs)
            class _R:
                content = [SimpleNamespace(text='{"selected_tickers": [], "reasons": {}}')]
                usage   = SimpleNamespace(input_tokens=10, output_tokens=5)
            return _R()

    class _FakeAnthropic:
        def __init__(self, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(st, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    st._call_llm_with_meta(["헤드라인 하나"], {"005930": "삼성전자"}, "claude-sonnet-5")
    st._call_llm_with_meta(["헤드라인 하나"], {"005930": "삼성전자"}, "claude-haiku-4-5-20251001")

    assert "temperature" not in captured[0], "sonnet-5에 temperature를 보내면 400"
    assert captured[1]["temperature"] == 0, "baseline은 실험 도중 조건이 바뀌면 안 된다"
    assert captured[0]["model"] == "claude-sonnet-5"


def test_health_note_healthy_and_degraded():
    """리포트 한 줄: 정상/실패/0종목/경고."""
    import importlib.util
    if importlib.util.find_spec("src.daily_runner") is None:      # pragma: no cover
        pytest.skip("daily_runner 로드 불가")
    from src.daily_runner import _health_note

    ok = _health_note({"healthy": True, "live_agent_reports": 27,
                       "total_agent_reports": 32, "ohlcv_stale_days": None, "reasons": [], "warnings": []})
    assert ok.startswith("✅") and "27/32" in ok

    nopick = _health_note({"healthy": True, "live_agent_reports": 0,
                           "total_agent_reports": 0, "ohlcv_stale_days": None, "reasons": [], "warnings": []})
    assert "확정 종목 없음" in nopick

    bad = _health_note({"healthy": False, "live_agent_reports": 0, "total_agent_reports": 32,
                        "ohlcv_stale_days": 9, "reasons": ["에이전트 전부 실패"], "warnings": ["장중 실행"]})
    assert bad.startswith("⛔") and "에이전트 전부 실패" in bad
    assert "OHLCV 9일 지연" in bad and "장중 실행" in bad


def _first_text_of(block: dict) -> str:
    node = block.get(block.get("type")) or {}
    return "".join(rt.get("text", {}).get("content", "") for rt in node.get("rich_text") or [])


def test_combined_blocks_include_health_note():
    """건강도 한 줄이 헤더 바로 다음 블록으로 들어간다."""
    from src.graph.notion_publisher import build_v4_blocks_combined

    blocks = build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": None,
                             "qualified_reports": [], "all_reports": []}],
        regime="neutral", strategy="HOLD",
        health_note="⛔ 이 리포트는 정상 분석으로 볼 수 없다 — 에이전트 전부 실패",
    )
    texts = [
        "".join(rt.get("text", {}).get("content", "") for rt in b.get("callout", {}).get("rich_text", []))
        for b in blocks if b.get("type") == "callout"
    ]
    assert any("정상 분석으로 볼 수 없다" in t for t in texts)
    # [2026-09-15] 재배치: 0번이 "오늘 한 줄", 1번이 건강도
    assert blocks[0]["type"] == "callout" and _first_text_of(blocks[0]).startswith("오늘: ")
    assert blocks[1]["type"] == "callout"
    assert blocks[1]["callout"]["color"] == "red_background"

    clean = build_v4_blocks_combined(
        per_ticker_results=[{"ticker": "005930", "chief_report": None,
                             "qualified_reports": [], "all_reports": []}],
        regime="neutral", strategy="HOLD",
    )
    assert clean[1]["type"] == "paragraph", "health_note 없으면 건강도 callout이 빠진다"


def test_first_text_skips_thinking_block():
    """sonnet-5는 content[0]이 ThinkingBlock이라 .text가 없다 (run 34756880693)."""
    from src.screening import stage1c_news as st

    thinking = SimpleNamespace(type="thinking", thinking="내부 추론")
    text     = SimpleNamespace(type="text", text='{"selected_tickers": ["005930"]}')
    assert st._first_text([thinking, text]) == '{"selected_tickers": ["005930"]}'
    assert st._first_text([text]) == '{"selected_tickers": ["005930"]}'
    assert st._first_text([thinking]) == ""
    assert st._first_text([]) == ""
    assert st._first_text(None) == ""


def test_stage1c_parses_response_with_thinking_block(monkeypatch):
    """thinking 블록이 앞에 와도 candidate 결과가 파싱된다."""
    from src.screening import stage1c_news as st

    class _FakeMessages:
        def create(self, **kwargs):
            class _R:
                content = [
                    SimpleNamespace(type="thinking", thinking="어떤 종목이 영향받을까"),
                    SimpleNamespace(type="text", text='{"selected_tickers": ["005930"], "reasons": {"005930": "반도체"}}'),
                ]
                usage = SimpleNamespace(input_tokens=100, output_tokens=50)
            return _R()

    class _FakeAnthropic:
        def __init__(self, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(st, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    parsed, meta = st._call_llm_with_meta(["헤드라인"], {"005930": "삼성전자"}, "claude-sonnet-5")
    assert parsed == {"selected_tickers": ["005930"], "reasons": {"005930": "반도체"}}
    assert meta["input_tokens"] == 100 and meta["output_tokens"] == 50


def test_thinking_models_get_larger_max_tokens():
    """사고 토큰이 max_tokens를 먹어 JSON이 잘렸다 (run 34761364921)."""
    from src.screening import stage1c_news as st
    assert st._max_tokens_for("claude-haiku-4-5-20251001") == st.MAX_TOKENS
    assert st._max_tokens_for("claude-sonnet-5") == st.MAX_TOKENS_THINKING
    assert st.MAX_TOKENS_THINKING > st.MAX_TOKENS


def test_stage1c_sends_model_specific_max_tokens(monkeypatch):
    from src.screening import stage1c_news as st
    captured: list[dict] = []

    class _FakeMessages:
        def create(self, **kwargs):
            captured.append(kwargs)
            class _R:
                content = [SimpleNamespace(type="text", text='{"selected_tickers": []}')]
                usage   = SimpleNamespace(input_tokens=1, output_tokens=1)
                stop_reason = "end_turn"
            return _R()

    class _FakeAnthropic:
        def __init__(self, **kw):
            self.messages = _FakeMessages()

    monkeypatch.setattr(st, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    st._call_llm_with_meta(["h"], {"005930": "삼성전자"}, "claude-sonnet-5")
    st._call_llm_with_meta(["h"], {"005930": "삼성전자"}, "claude-haiku-4-5-20251001")
    assert captured[0]["max_tokens"] == st.MAX_TOKENS_THINKING
    assert captured[1]["max_tokens"] == st.MAX_TOKENS


def test_stage1c_self_baseline_called_once_per_day(monkeypatch):
    """자기 겹침률 재호출은 하루 1회. 같은 날 재실행이면 건너뛴다."""
    from src.screening import stage1c_news as st

    calls: list[str] = []
    monkeypatch.setattr(st, "_call_llm_with_meta",
                        lambda h, t, model: (calls.append(model),
                                             ({"selected_tickers": ["005930"]},
                                              {"input_tokens": 1, "output_tokens": 1, "latency_ms": 1}))[1])
    logged: list[tuple] = []
    monkeypatch.setattr(st, "log_shadow", lambda *a, **k: logged.append(a[1]))

    monkeypatch.setattr(st, "has_variant_on", lambda *a, **k: False)
    st._run_self_baseline(["h"], {"005930": "삼성전자"}, "hash1")
    assert calls == [st.NEWS_SCREEN_MODEL], "baseline 모델을 다시 부른다"
    assert logged == [st.SELF_VARIANT]

    calls.clear()
    logged.clear()
    monkeypatch.setattr(st, "has_variant_on", lambda *a, **k: True)
    st._run_self_baseline(["h"], {"005930": "삼성전자"}, "hash1")
    assert calls == [] and logged == [], "이미 기록됐으면 호출하지 않는다"
