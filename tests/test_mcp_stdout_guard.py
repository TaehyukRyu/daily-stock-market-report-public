"""
tests/test_mcp_stdout_guard.py

MCP stdio 서버 안의 print()가 JSON-RPC 채널(stdout)을 오염시키지 않는지.
2026-09-11 CI 실행: KRX 자격증명이 서브프로세스에 전달되자 pykrx가 **import 시점에** stdout에
"KRX 로그인 시도..."를 찍어 "Failed to parse JSONRPC message" 80건이 났다.
가드는 어떤 import보다 먼저 와야 한다 (2차 실행에서 __main__ 하단 가드로는 안 잡혔다).
"""

from __future__ import annotations

import logging
import pathlib
import textwrap

import pytest

from src.utils import mcp_client as mc

GUARD = "_b.print = _ft.partial(_b.print, file=_sys.stderr)"


def test_all_servers_redirect_print_before_any_import():
    for f in ("krx_market", "news_economy", "us_market"):
        s = pathlib.Path(f"src/mcp_servers/{f}/server.py").read_text(encoding="utf-8")
        guard_at = s.index(GUARD)
        first_import = min(i for i in (s.find("\nimport "), s.find("\nfrom ")) if i >= 0)
        assert guard_at < first_import, f"{f}: 가드가 첫 import보다 앞이어야 한다 (pykrx는 import 시점에 print)"


@pytest.mark.asyncio
async def test_import_time_and_tool_time_prints_do_not_corrupt_channel(tmp_path, caplog):
    server = tmp_path / "noisy_server.py"
    server.write_text(textwrap.dedent(f"""
        if __name__ == "__main__":
            import builtins as _b, functools as _ft, sys as _sys
            {GUARD}

        print("KRX 로그인 시도...")          # import 시점 print (pykrx 흉내)
        from fastmcp import FastMCP
        mcp = FastMCP("noisy")

        @mcp.tool()
        def noisy(x: int) -> dict:
            print("  만료 시간: 2026-09-11")  # 도구 실행 중 print
            return {{"value": x * 2}}

        if __name__ == "__main__":
            mcp.run()
    """), encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        async with mc.mcp_client(str(server)) as client:
            raw = await client.call_tool("noisy", {"x": 21})
    assert raw.structured_content["value"] == 42
    assert not any("JSONRPC" in r.getMessage() for r in caplog.records)
