# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Connecting to an MCP server the operator runs, rather than spawning one."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from agent6.config import Config
from agent6.tools.mcp_client import MCPManager, MCPServerSpec
from agent6.tools.mcp_http import MAX_BODY_BYTES, HttpTransport, MCPHttpError


def _serve(reply: Any, *, sse: bool = False, status: int = 200, body: bytes | None = None):
    """A one-connection MCP server on loopback. Returns (url, seen_headers)."""
    seen: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.update({k.lower(): v for k, v in self.headers.items()})
            request = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(status)
            self.send_header("content-type", "text/event-stream" if sse else "application/json")
            self.end_headers()
            if body is not None:
                self.wfile.write(body)
                return
            answer = reply(request) if callable(reply) else reply
            raw = json.dumps(answer).encode()
            self.wfile.write(b"event: message\ndata: " + raw + b"\n\n" if sse else raw)

        def log_message(self, format: str, *args: Any) -> None:
            return  # a test server must not print to stderr

    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_port}/mcp", seen, httpd


def _mcp_reply(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    if method == "initialize":
        result: Any = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif method == "tools/list":
        result = {"tools": [{"name": "ping", "description": "d", "inputSchema": {}}]}
    else:
        result = {"content": [{"type": "text", "text": "pong"}]}
    return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}


def test_agent6_connects_instead_of_spawning() -> None:
    """A server that wants a browser or a device is the operator's to run, in
    whatever container they chose; agent6 owning its lifetime is the wrong
    owner. The handshake and a call go over one POST each."""
    url, _seen, httpd = _serve(_mcp_reply)
    try:
        mgr = MCPManager.start(
            [
                MCPServerSpec(
                    name="remote",
                    command=(),
                    startup_timeout_s=10.0,
                    call_timeout_s=10.0,
                    http=HttpTransport(name="remote", url=url),
                )
            ]
        )
        try:
            assert [(d.server_name, d.tool_name) for d in mgr.descriptors()] == [("remote", "ping")]
            assert mgr.call("mcp__remote__ping", {}) == {
                "content": [{"type": "text", "text": "pong"}]
            }
        finally:
            mgr.close()
    finally:
        httpd.shutdown()


def test_a_streamed_answer_is_read_like_any_other() -> None:
    """Streamable HTTP lets a server answer one request with an SSE frame.
    Reading only a bare body made every such server look like it sent
    garbage."""
    url, _seen, httpd = _serve(_mcp_reply, sse=True)
    try:
        got = HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
        assert got is not None and got["id"] == 1
    finally:
        httpd.shutdown()


def test_the_token_is_read_from_the_environment_never_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret written into a config file is a secret in a backup."""
    monkeypatch.setenv("MCP_TEST_TOKEN", "s3cr3t")
    url, seen, httpd = _serve(_mcp_reply)
    try:
        HttpTransport(name="s", url=url, token_env="MCP_TEST_TOKEN").send(
            {"jsonrpc": "2.0", "id": 1}, timeout_s=5.0
        )
        assert seen["authorization"] == "Bearer s3cr3t"
    finally:
        httpd.shutdown()


def test_an_oversized_body_is_refused_rather_than_buffered() -> None:
    """The same bound the stdio reader applies: a runaway server must not be
    able to buffer an unbounded body into the agent."""
    url, _seen, httpd = _serve(None, body=b"x" * (MAX_BODY_BYTES + 64))
    try:
        with pytest.raises(MCPHttpError, match="more than"):
            HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=10.0)
    finally:
        httpd.shutdown()


def test_an_http_failure_is_a_clean_tool_error() -> None:
    url, _seen, httpd = _serve(_mcp_reply, status=503)
    try:
        with pytest.raises(MCPHttpError, match="HTTP 503"):
            HttpTransport(name="s", url=url).send({"jsonrpc": "2.0", "id": 1}, timeout_s=5.0)
    finally:
        httpd.shutdown()


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({}, "exactly one"),
        ({"command": ["x"], "url": "https://h/mcp"}, "exactly one"),
        ({"url": "ftp://h/mcp"}, "http"),
        ({"command": ["x"], "token_env": "T"}, "pass_env"),
    ],
)
def test_a_server_names_one_transport(entry: dict[str, Any], message: str) -> None:
    """Both or neither is a config error, not a guess."""
    with pytest.raises(ValueError, match=message):
        Config.model_validate({"mcp": {"enabled": True, "servers": {"s": entry}}})
