"""
src/utils/mcp_client.py

MCP 서버 클라이언트 생성 — 환경변수를 서브프로세스에 명시적으로 넘긴다.

[왜 필요한가]
  에이전트 8곳이 `Client("src/mcp_servers/…/server.py")`로 서버를 띄운다.
  파일 경로를 주면 fastmcp는 PythonStdioTransport(서브프로세스)를 쓰는데,
  MCP 라이브러리는 부모 환경변수 중 화이트리스트(HOME·PATH 등 6개)만 넘긴다
  (mcp.client.stdio.DEFAULT_INHERITED_ENV_VARS,
   https://gofastmcp.com/clients/transports "STDIO servers inherit only a small
   allowlist of environment variables").
  로컬은 서버 쪽 load_dotenv()가 .env로 채워 증상이 안 보였고, CI는 .env가 없어
  DART·FRED·BOK 키가 서버 안에서 빈 값이었다 (2026-09-09 실행 로그 24건).

[설계]
  env=dict(os.environ) — 부모의 환경 전부를 merge한다. 화이트리스트 위에 덮어쓰는
  방식이라 PATH 등은 유지된다. 시크릿은 같은 러너 안에서만 오간다.
"""

from __future__ import annotations

import os

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


def build_transport(script_path: str) -> PythonStdioTransport:
    """테스트에서 env 전달을 검증할 수 있도록 transport 생성을 분리한다."""
    return PythonStdioTransport(script_path, env=dict(os.environ))


def mcp_client(script_path: str) -> Client:
    """`Client("경로")`의 대체. 부모 환경변수를 서브프로세스에 그대로 넘긴다."""
    return Client(build_transport(script_path))
