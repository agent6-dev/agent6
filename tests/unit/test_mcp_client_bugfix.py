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

import pytest

from agent6.tools.mcp_client import (
    MCP_TOOL_PREFIX,
    MCPError,
    MCPManager,
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


def test_iserror_tool_result_surfaces_as_error() -> None:
    mgr = MCPManager.start([("fake", _iserror_server_argv(), 5.0, 5.0)])
    try:
        with pytest.raises(MCPError) as ei:
            mgr.call(f"{MCP_TOOL_PREFIX}fake__boom", {})
        assert "disk on fire" in str(ei.value)
    finally:
        mgr.close()


def test_server_initiated_request_not_treated_as_response() -> None:
    mgr = MCPManager.start([("fake", _server_request_collision_argv(), 5.0, 5.0)])
    try:
        # Pre-fix: the colliding server request (id=N, method=roots/list) was
        # popped as the response, failing the non-dict-result check. Post-fix:
        # it's ignored and the genuine response is returned.
        out = mgr.call(f"{MCP_TOOL_PREFIX}fake__echo", {"text": "ok"})
        assert out["content"][0]["text"] == "ok"
    finally:
        mgr.close()


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
    mgr = MCPManager.start([("fake", _poison_tools_server_argv(), 5.0, 5.0)])
    try:
        descs = mgr.descriptors()
        assert [d.qualified_name for d in descs] == [f"{MCP_TOOL_PREFIX}fake__echo"]
        assert descs[0].description == "first"
    finally:
        mgr.close()
