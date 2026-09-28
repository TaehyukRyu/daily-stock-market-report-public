"""
tests/test_mcp_client.py

MCP 서브프로세스가 부모 환경변수를 받는지 — 실제 서버를 띄워 확인한다.
(유료 API 호출 없음. 로컬 파이썬 프로세스 1개만 뜬다.)
"""

from __future__ import annotations

import os
import textwrap

import pytest

from src.utils import mcp_client as mc


def test_transport_carries_parent_env(monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "dart-test-123")
    tr = mc.build_transport("src/mcp_servers/krx_market/server.py")
    assert tr.env["DART_API_KEY"] == "dart-test-123"
    assert "PATH" in tr.env, "화이트리스트(PATH 등)는 유지돼야 한다"


def test_all_agents_use_helper():
    """MCP를 여는 곳은 전부 헬퍼를 쓴다 — 하나라도 옛 방식이면 CI에서 키가 사라진다.

    [2026-09-21] 7곳. sentiment_analyst가 v3에서 시장 뉴스 MCP 수집을 버리고
    mentions 테이블(로컬 DB)을 읽게 되면서 한 곳이 줄었다
    (doc/2026-09-21_agent-audit.md §1-6). 중요한 불변식은 old == 0이다.
    """
    import pathlib
    old = new = 0
    for p in pathlib.Path("src/agents").glob("*.py"):
        s = p.read_text(encoding="utf-8")
        old += s.count('Client("src/mcp_servers/')
        new += s.count('mcp_client("src/mcp_servers/')
    assert old == 0
    assert new == 7


@pytest.mark.asyncio
async def test_subprocess_sees_secret(tmp_path, monkeypatch):
    """진짜 stdio 서버를 띄워 서브프로세스가 부모 env를 읽는지 end-to-end로 확인."""
    server = tmp_path / "echo_env_server.py"
    server.write_text(textwrap.dedent("""
        import os
        from fastmcp import FastMCP
        mcp = FastMCP("echo-env")

        @mcp.tool()
        def read_env(name: str) -> dict:
            return {"value": os.getenv(name, "")}

        if __name__ == "__main__":
            mcp.run()
    """), encoding="utf-8")

    monkeypatch.setenv("PROBE_SECRET", "seen-by-child")
    async with mc.mcp_client(str(server)) as client:
        raw = await client.call_tool("read_env", {"name": "PROBE_SECRET"})
    data = raw.structured_content if getattr(raw, "structured_content", None) else None
    assert data and data["value"] == "seen-by-child"
