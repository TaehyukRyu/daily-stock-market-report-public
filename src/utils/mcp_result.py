"""
src/utils/mcp_result.py

MCP 도구 결과를 파싱하면서 오류를 드러내는 공통 헬퍼.

[왜 필요한가]
  MCP 서버 3개는 실패를 `return {"error": str(e)}` 로 돌려준다 (총 35곳).
  파이프라인 중단을 막으려는 의도였지만, 오류가 **성공 응답과 같은 타입**이라
  호출부가 구분할 이유가 없어졌다. 실제로 35곳 중 호출부가 error 키를 검사하는
  곳은 regime_detector.py:119 **한 군데뿐**이었다.

  나머지 34곳은 이렇게 흘렀다:
      {"error": "..."} → parse() 통과 → .get("ohlcv", {}) → {} → "데이터 부족"
  즉 MCP가 준 실제 오류 메시지가 로그에도 남지 않고 사라지고,
  전혀 다른 오류("데이터 부족")로 치환되어 LLM 프롬프트에 들어갔다.

  실제 피해: OHLCV 캐시가 2026-07-24부터 46일간 멈췄는데
  (_bulk_snapshot이 KRX 로그인 실패로 None 반환) 아무도 알아채지 못했다.

[방침]
  오류를 예외로 바꾸지 않는다 — 파이프라인 중단 방지라는 원래 목표는 옳다.
  대신 (1) 반드시 로그에 남기고 (2) 호출부가 물어볼 수 있게 만들고
  (3) 프롬프트에 "데이터 없음"을 명시할 수 있게 한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 이번 프로세스에서 관측된 MCP 오류 (tool → 건수). 건강도 리포트용.
_ERROR_COUNTS: dict[str, int] = {}


def parse_mcp_result(raw: Any, *, tool: str, context: str = "") -> dict:
    """
    MCP 호출 결과를 dict로 변환한다. 오류면 로그에 남기되 dict는 그대로 반환한다.

    Args:
        raw: fastmcp Client.call_tool() 반환값, 또는 gather()가 준 Exception
        tool: 도구 이름 (로그 식별용). 예: "get_stock_price"
        context: 부가 식별자. 예: ticker

    Returns:
        파싱된 dict. 실패 시 {"error": "..."} 형태를 유지한다
        (하위 로직이 .get()으로 계속 동작하도록).
    """
    label = f"{tool}({context})" if context else tool

    # gather(return_exceptions=True)가 넘긴 예외
    if isinstance(raw, BaseException):
        msg = f"{type(raw).__name__}: {raw}"
        _record(tool, label, msg)
        return {"error": msg}

    data: dict = {}
    try:
        if hasattr(raw, "structured_content") and raw.structured_content:
            data = raw.structured_content
        elif hasattr(raw, "content") and raw.content:
            data = json.loads(raw.content[0].text)
        elif isinstance(raw, dict):
            data = raw
    except Exception as e:
        msg = f"파싱 실패 {type(e).__name__}: {e}"
        _record(tool, label, msg)
        return {"error": msg}

    if not isinstance(data, dict):
        msg = f"예상치 못한 반환 타입 {type(data).__name__}"
        _record(tool, label, msg)
        return {"error": msg}

    if "error" in data:
        _record(tool, label, str(data["error"]))

    return data


# 최근 오류 메시지 (건강도 판정이 "키 누락" 같은 원인을 문자열로 찾는다). 상한 200.
_RECENT_MESSAGES: list[str] = []
_RECENT_MAX = 200


def recent_errors() -> list[str]:
    return list(_RECENT_MESSAGES)


def _record(tool: str, label: str, message: str) -> None:
    _ERROR_COUNTS[tool] = _ERROR_COUNTS.get(tool, 0) + 1
    _RECENT_MESSAGES.append(f"{label}: {message}")
    if len(_RECENT_MESSAGES) > _RECENT_MAX:
        del _RECENT_MESSAGES[0]
    # error 레벨 — 이게 warning이면 CI 로그에서 묻힌다 (46일간 그랬다).
    logger.error(f"[MCP] {label} 실패: {message}")


def has_error(data: Any) -> bool:
    """MCP 결과가 오류인지."""
    return isinstance(data, dict) and "error" in data


def failure_note(data: Any, *, what: str) -> str | None:
    """
    프롬프트에 넣을 실패 문구. 정상이면 None.

    LLM에게 "이 데이터는 수집에 실패했다"를 명시해야, 없는 수치를 지어내는 대신
    data_sufficient=False로 답할 수 있다.
    """
    if not has_error(data):
        return None
    return f"⚠️ {what} 수집 실패: {data['error']} — 이 항목의 수치는 존재하지 않는다."


def get_error_counts() -> dict[str, int]:
    """이번 프로세스에서 누적된 MCP 오류 건수 (tool별)."""
    return dict(_ERROR_COUNTS)


def reset_error_counts() -> None:
    _ERROR_COUNTS.clear()
