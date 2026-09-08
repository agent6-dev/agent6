# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for two MCP client fixes.

* isError=true tool results (a SUCCESSFUL JSON-RPC result, no top-level
  "error") must surface as an MCPError so the dispatcher reports ok=False.
* The reader must NOT treat server-INITIATED requests (which carry both an
  int "id" and a "method") as responses; their id can collide with one of
  ours and corrupt/orphan the real response.
"""

from __future__ import annotations

import sys
import textwrap
import time
from typing import Any

import pytest

from agent6.tools.mcp_client import (
    MCP_TOOL_PREFIX,
    MCPError,
    MCPManager,
    MCPServerSpec,
)


def _iserror_server_argv() -> tuple[str, ...]:
    """A server whose tools/call returns a tool-level failure as a normal
    JSON-RPC result with isError=true (spec-compliant), no top-level error."""
    script = textwrap.dedent(
        """
        import json, sys
        def reply(req_id, result):
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id,
                                         "result": result}) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method is None or "id" not in msg:
                continue
            if method == "initialize":
                reply(msg["id"], {"protocolVersion": "2024-11-05",
                                  "capabilities": {},
                                  "serverInfo": {"name": "fake", "version": "0"}})
                continue
            if method == "tools/list":
                reply(msg["id"], {"tools": [
                    {"name": "boom", "description": "always fails",
                     "inputSchema": {"type": "object"}},
                ]})
                continue
            if method == "tools/call":
                reply(msg["id"], {"content": [
                    {"type": "text", "text": "kaboom: disk on fire"}
                ], "isError": True})
                continue
            reply(msg["id"], {})
        """
    )
    return (sys.executable, "-c", script)


def _echoed_secret_server_argv(*, tool_level: bool = False) -> tuple[str, ...]:
    script = textwrap.dedent(
        f"""
        import json, os, sys
        TOOL_LEVEL = {tool_level!r}
        def send(obj):
            sys.stdout.write(json.dumps(obj) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            msg = json.loads(line)
            if "id" not in msg:
                continue
            method = msg.get("method")
            if method == "initialize":
                result = {{}}
            elif method == "tools/list":
                result = {{"tools": [{{"name": "fail", "inputSchema": {{}}}}]}}
            else:
                if TOOL_LEVEL:
                    result = {{"isError": True, "content": [
                        {{"type": "text", "text": os.environ["MCP_TEST_SECRET"]}}]}}
                else:
                    send({{"jsonrpc": "2.0", "id": msg["id"], "error": {{
                        "code": -1, "message": os.environ["MCP_TEST_SECRET"]}}}})
                    continue
            send({{"jsonrpc": "2.0", "id": msg["id"], "result": result}})
        """
    )
    return (sys.executable, "-c", script)


def test_a_passed_secret_echoed_in_a_protocol_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passed credential stays out of the transcript even when the server
    copies it into a JSON-RPC error rather than stderr."""
    secret = "secret-from-environment-" + "x" * 3000
    monkeypatch.setenv("MCP_TEST_SECRET", secret)
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="leaky",
                command=_echoed_secret_server_argv(),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
                pass_env=("MCP_TEST_SECRET",),
            )
        ]
    )
    try:
        with pytest.raises(MCPError) as caught:
            mgr.call(f"{MCP_TOOL_PREFIX}leaky__fail", {})
    finally:
        mgr.close()

    assert "secret-from-environment" not in str(caught.value)
    assert "<REDACTED>" in str(caught.value)


def test_a_passed_secret_echoed_in_a_tool_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_TEST_SECRET", "secret-from-environment")
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="leaky",
                command=_echoed_secret_server_argv(tool_level=True),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
                pass_env=("MCP_TEST_SECRET",),
            )
        ]
    )
    try:
        with pytest.raises(MCPError) as caught:
            mgr.call(f"{MCP_TOOL_PREFIX}leaky__fail", {})
    finally:
        mgr.close()

    assert "secret-from-environment" not in str(caught.value)
    assert "<REDACTED>" in str(caught.value)


def _server_request_collision_argv() -> tuple[str, ...]:
    """A server that, when tools/call arrives, FIRST emits its own request
    (id=1, method='roots/list') — colliding with the client's first id — and
    THEN the genuine response. The pre-fix reader stored the server request
    under id=1 and popped it as the response (no result -> non-dict failure)."""
    script = textwrap.dedent(
        """
        import json, sys
        def send(obj):
            sys.stdout.write(json.dumps(obj) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method is None or "id" not in msg:
                continue
            if method == "initialize":
                send({"jsonrpc": "2.0", "id": msg["id"],
                      "result": {"protocolVersion": "2024-11-05",
                                 "capabilities": {},
                                 "serverInfo": {"name": "fake", "version": "0"}}})
                continue
            if method == "tools/list":
                send({"jsonrpc": "2.0", "id": msg["id"],
                      "result": {"tools": [
                          {"name": "echo", "description": "echo",
                           "inputSchema": {"type": "object"}}]}})
                continue
            if method == "tools/call":
                # Server-initiated request whose id collides with the client's
                # outstanding tools/call id. Must be ignored by the reader.
                send({"jsonrpc": "2.0", "id": msg["id"], "method": "roots/list",
                      "params": {}})
                # The genuine response.
                args = msg["params"].get("arguments", {})
                send({"jsonrpc": "2.0", "id": msg["id"],
                      "result": {"content": [
                          {"type": "text", "text": str(args.get("text", ""))}]}})
                continue
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        """
    )
    return (sys.executable, "-c", script)


def test_a_json_rpc_error_cannot_flood_the_context() -> None:
    """A server controls its JSON-RPC error message, which reaches the model's
    context, so it gets the same inline bound as a tool-level error."""
    from agent6.tools.mcp_client import (
        _MAX_INLINE_TEXT_CHARS,  # pyright: ignore[reportPrivateUsage]
        _result_of,  # pyright: ignore[reportPrivateUsage]
    )

    with pytest.raises(MCPError) as caught:
        _result_of(
            {"error": {"message": "x" * (_MAX_INLINE_TEXT_CHARS * 4)}},
            name="fake",
            method="tools/call",
        )

    message = str(caught.value)
    assert len(message) < _MAX_INLINE_TEXT_CHARS + 100
    assert "[agent6: truncated]" in message


def test_iserror_tool_result_surfaces_as_error() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_iserror_server_argv(),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ]
    )
    try:
        with pytest.raises(MCPError) as ei:
            mgr.call(f"{MCP_TOOL_PREFIX}fake__boom", {})
        assert "disk on fire" in str(ei.value)
    finally:
        mgr.close()


def test_server_initiated_request_not_treated_as_response() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_server_request_collision_argv(),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ]
    )
    try:
        # Pre-fix: the colliding server request (id=N, method=roots/list) was
        # popped as the response, failing the non-dict-result check. Post-fix:
        # it's ignored and the genuine response is returned.
        out = mgr.call(f"{MCP_TOOL_PREFIX}fake__echo", {"text": "ok"})
        assert out["content"][0]["text"] == "ok"
    finally:
        mgr.close()


def test_tools_list_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every tools/list page is part of one listing; stopping at nextCursor
    silently hid every tool after the server's first page."""
    from agent6.tools.mcp_client import _MCPServer  # pyright: ignore[reportPrivateUsage]
    from agent6.tools.mcp_http import HttpTransport

    def send(
        _transport: HttpTransport, payload: dict[str, Any], *, timeout_s: float
    ) -> dict[str, Any] | None:
        del timeout_s
        method = payload["method"]
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            result: dict[str, Any] = {}
        elif payload["params"].get("cursor") == "page-2":
            result = {"tools": [{"name": "second", "description": "", "inputSchema": {}}]}
        else:
            result = {
                "tools": [{"name": "first", "description": "", "inputSchema": {}}],
                "nextCursor": "page-2",
            }
        return {"jsonrpc": "2.0", "id": payload["id"], "result": result}

    monkeypatch.setattr(HttpTransport, "send", send)
    srv = _MCPServer(  # pyright: ignore[reportPrivateUsage]
        name="pages",
        command=(),
        startup_timeout_s=5.0,
        call_timeout_s=5.0,
        http=HttpTransport(name="pages", url="https://example.invalid/mcp"),
    )

    srv.start()

    assert [tool.tool_name for tool in srv.tools] == ["first", "second"]


def test_tools_list_pagination_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server minting a fresh nextCursor on every page would otherwise hold
    the handshake forever and grow the roster without bound."""
    from agent6.tools.mcp_client import _MCPServer  # pyright: ignore[reportPrivateUsage]
    from agent6.tools.mcp_http import HttpTransport

    pages = 0

    def send(
        _transport: HttpTransport, payload: dict[str, Any], *, timeout_s: float
    ) -> dict[str, Any] | None:
        nonlocal pages
        del timeout_s
        method = payload["method"]
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            result: dict[str, Any] = {}
        else:
            pages += 1
            result = {"tools": [], "nextCursor": f"page-{pages}"}
        return {"jsonrpc": "2.0", "id": payload["id"], "result": result}

    monkeypatch.setattr(HttpTransport, "send", send)
    srv = _MCPServer(  # pyright: ignore[reportPrivateUsage]
        name="endless",
        command=(),
        startup_timeout_s=5.0,
        call_timeout_s=5.0,
        http=HttpTransport(name="endless", url="https://example.invalid/mcp"),
    )

    with pytest.raises(MCPError, match="paged past"):
        srv.start()
    assert pages == 64


def _poison_tools_server_argv() -> tuple[str, ...]:
    """A server whose tools/list advertises, besides a valid `echo`: a tool
    whose 54-char name pushes the qualified name past the 64-char provider
    bound, and a duplicate `echo` entry."""
    script = textwrap.dedent(
        """
        import json, sys
        def reply(req_id, result):
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id,
                                         "result": result}) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method is None or "id" not in msg:
                continue
            if method == "initialize":
                reply(msg["id"], {"protocolVersion": "2024-11-05",
                                  "capabilities": {},
                                  "serverInfo": {"name": "fake", "version": "0"}})
                continue
            if method == "tools/list":
                reply(msg["id"], {"tools": [
                    {"name": "echo", "description": "first",
                     "inputSchema": {"type": "object"}},
                    {"name": "a" * 54, "description": "overlong",
                     "inputSchema": {"type": "object"}},
                    {"name": "echo", "description": "duplicate",
                     "inputSchema": {"type": "object"}},
                ]})
                continue
            reply(msg["id"], {})
        """
    )
    return (sys.executable, "-c", script)


def test_registration_skips_tools_that_would_poison_the_tools_array() -> None:
    """An over-64-char qualified name or a duplicate name would 400 the WHOLE
    provider tools array every turn; both are dropped at registration (first
    occurrence wins) like the invalid-char skip, so one bad entry cannot take
    the run down."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_poison_tools_server_argv(),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ]
    )
    try:
        descs = mgr.descriptors()
        assert [d.qualified_name for d in descs] == [f"{MCP_TOOL_PREFIX}fake__echo"]
        assert descs[0].description == "first"
    finally:
        mgr.close()


def _slow_call_server_argv() -> tuple[str, ...]:
    """Handshake replies promptly; every tools/call sleeps 0.5s before
    replying, so a short-call-timeout client times out and the reply arrives
    late."""
    script = textwrap.dedent(
        """
        import json, sys, time
        def reply(req_id, result):
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id,
                                         "result": result}) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method is None or "id" not in msg:
                continue
            if method == "initialize":
                reply(msg["id"], {"protocolVersion": "2024-11-05",
                                  "capabilities": {},
                                  "serverInfo": {"name": "fake", "version": "0"}})
                continue
            if method == "tools/list":
                reply(msg["id"], {"tools": [
                    {"name": "slow", "description": "slow echo",
                     "inputSchema": {"type": "object"}},
                ]})
                continue
            if method == "tools/call":
                time.sleep(0.5)
                reply(msg["id"], {"content": [{"type": "text", "text": "late"}]})
                continue
            reply(msg["id"], {})
        """
    )
    return (sys.executable, "-c", script)


def test_timed_out_requests_leave_no_pending_residue() -> None:
    """A reply landing after its caller timed out must be dropped, not stored:
    the reader retained ANY response-shaped message forever once no _request
    was left to pop it, growing _pending (up to 8 MiB per entry) without
    bound against a slow or runaway server."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_slow_call_server_argv(),
                startup_timeout_s=5.0,
                call_timeout_s=0.15,
            )
        ]
    )
    try:
        srv = mgr._servers["fake"]  # pyright: ignore[reportPrivateUsage]
        for _ in range(2):
            with pytest.raises(MCPError, match="timed out"):
                mgr.call(f"{MCP_TOOL_PREFIX}fake__slow", {})
        # The server answers sequentially (0.5s each), so both late replies
        # have flushed well before 2s; the drop is unobservable from outside,
        # so wait past that point and then prove nothing was retained.
        time.sleep(2.0)
        with srv._pending_cv:  # pyright: ignore[reportPrivateUsage]
            pending = dict(srv._pending)  # pyright: ignore[reportPrivateUsage]
        assert pending == {}
    finally:
        mgr.close()
