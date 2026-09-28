"""
tests/conftest.py

테스트를 비용/속도 기준으로 나누는 마커 자동 부착.

[왜 필요한가]
  이 스위트의 테스트 상당수가 **실제 유료 API를 호출한다** (목이 아니다).
  2026-09-08 실측:
    - 전체 스위트: 78분, OpenAI 호출 82건 (매 실행 과금)
    - OpenAI 키가 죽어 있던 날에는 79 errors 로 무더기 실패

  CI에 pytest를 넣으려면 이 구분이 먼저다. 아무 생각 없이 `pytest tests/`를
  붙이면 푸시할 때마다 과금되고 78분을 기다려야 한다.

[사용]
  pytest -m "not costly"   # 기본 — 252건 / 약 36초 / 과금 0
  pytest                   # 전체 (유료 API 호출)
  pytest -m costly         # 유료만

[분류 근거]
  아래 COSTLY_MODULES는 추측이 아니라 관측이다.
  OPENAI_API_KEY가 만료됐던 2026-09-07~08 전체 실행에서 실제로
  인증 오류로 무너진 모듈들 + 파이프라인 통합 테스트(18분)를 모았다.
"""

import pytest

# 유료 API(LLM/외부 인증)를 실제로 호출하는 모듈.
# 새 테스트를 추가할 때 LLM을 호출하면 여기에 이름을 넣어야 한다.
COSTLY_MODULES = {
    # 7개 전문가 에이전트 — OpenAI 호출
    "test_macro_economist",
    "test_kr_market_specialist",
    "test_us_market_specialist",
    "test_quant_analyst",
    "test_technical_analyst",
    "test_sentiment_analyst",
    "test_fundamental_analyst",
    # Anthropic 호출
    "test_chief_strategist",
    "test_debate",
    "test_consensus",
    # OpenAI 임베딩 / 구조화 출력
    "test_rag",
    "test_structured_output",
    # 파이프라인 전체 (2026-09-08 실측 18분)
    "test_pipeline",
    # 외부 인증이 필요한 MCP (NAVER / KRX / FRED / BOK)
    "test_news_economy_mcp",
    "test_krx_mcp",
    "test_us_market_mcp",
}

# 공개본 전용 — 프롬프트와 _format_prompt를 [REDACTED]로 가려서 내용 검증이 불가능한 테스트.
# 비공개 원본에서는 전부 통과한다 (2026-09-28 확인).
REDACTED_PROMPT_TESTS = {
    "test_eval_method": {"test_risk_factors_present_in_every_output_spec",
                         "test_risk_factors_is_not_hoisted_to_front"},
    "test_g5_no_signal": {"test_chief_prompt_does_not_force_direction"},
    "test_mcp_result_and_schema": {"test_prompt_marks_collection_failure_explicitly",
                                   "test_prompt_normal_path_unchanged",
                                   "test_kr_prompt_marks_each_failure",
                                   "test_fundamental_failure_not_shown_as_absence",
                                   "test_macro_prompt_survives_whole_section_failure",
                                   "test_us_prompt_does_not_dump_error_json",
                                   "test_technical_prompt_does_not_dump_error_json"},
    "test_prompt_order_bias": {"test_chief_prompt_lists_all_eight_voters",
                               "test_chief_prompt_ignores_speaking_order",
                               "test_hold_verdict_prompt_ignores_speaking_order"},
    "test_screen_context": {"test_agent_prompt_carries_event_and_price_context",
                            "test_agent_prompt_with_no_headlines_forbids_fabrication",
                            "test_agent_prompt_never_mentions_market_wide_categories"},
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "costly: 실제 유료 API를 호출한다. CI 기본 실행에서는 제외된다.",
    )


def pytest_collection_modifyitems(items):
    """모듈 이름을 보고 costly 마커를 자동으로 붙인다.

    테스트 파일 82개에 일일이 데코레이터를 다는 대신 한 곳에서 관리한다.
    """
    for item in items:
        mod = item.module.__name__.rsplit(".", 1)[-1]
        if mod in COSTLY_MODULES:
            item.add_marker(pytest.mark.costly)
        if item.originalname in REDACTED_PROMPT_TESTS.get(mod, ()):
            item.add_marker(pytest.mark.skip(reason="공개본: 프롬프트 [REDACTED] — 내용 검증 불가"))
