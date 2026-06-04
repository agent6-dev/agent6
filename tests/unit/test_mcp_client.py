# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Smoke-tests for the stdio MCP client.

Uses a tiny in-tree Python "MCP server" that talks just enough JSON-RPC
to satisfy ``initialize`` + ``tools/list`` + ``tools/call``. No external
dependency.
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


def _fake_server_argv(*, hang: bool = False, crash_after_init: bool = False) -> tuple[str, ...]:
    """Return argv that runs a tiny Python MCP server inline.

    The server speaks line-delimited JSON-RPC 2.0 over stdio:
    * ``initialize``  -> empty result
    * ``tools/list``  -> two tools: ``echo`` and ``shout``
    * ``tools/call``  -> echoes back the args under "content"

    Knobs:
    * ``hang=True``: never responds (forces client timeout).
    * ``crash_after_init=True``: exits 0 right after handshake.
    """
    script = textwrap.dedent(
        f"""
        import json, sys
        HANG = {hang!r}
        CRASH = {crash_after_init!r}
        def reply(req_id, result):
            sys.stdout.write(json.dumps({{
                "jsonrpc": "2.0", "id": req_id, "result": result,
            }}) + "\\n")
            sys.stdout.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            method = msg.get("method")
            if method is None:
                continue
            if "id" not in msg:
                continue  # notification
            if HANG:
                continue
            if method == "initialize":
                reply(msg["id"], {{"protocolVersion": "2024-11-05",
                                    "capabilities": {{}},
                                    "serverInfo": {{"name": "fake", "version": "0"}}}})
                if CRASH:
                    sys.exit(0)
                continue
            if method == "tools/list":
                reply(msg["id"], {{"tools": [
                    {{"name": "echo", "description": "echo the input",
                      "inputSchema": {{"type": "object",
                                       "properties": {{"text": {{"type": "string"}}}}}}}},
                    {{"name": "shout", "description": "upper-case echo",
                      "inputSchema": {{"type": "object",
                                       "properties": {{"text": {{"type": "string"}}}}}}}},
                ]}})
                continue
            if method == "tools/call":
                args = msg["params"].get("arguments", {{}})
                tname = msg["params"].get("name")
                if tname == "shout":
                    out = str(args.get("text", "")).upper()
                else:
                    out = str(args.get("text", ""))
                reply(msg["id"], {{"content": [
                    {{"type": "text", "text": out}}
                ]}})
                continue
            reply(msg["id"], {{}})
        """
    )
    return (sys.executable, "-c", script)


def test_manager_starts_and_discovers_tools() -> None:
    mgr = MCPManager.start(
        [("fake", _fake_server_argv(), 5.0, 5.0)],
    )
    try:
        descs = mgr.descriptors()
        names = sorted(d.qualified_name for d in descs)
        assert names == [
            f"{MCP_TOOL_PREFIX}fake__echo",
            f"{MCP_TOOL_PREFIX}fake__shout",
        ]
        for d in descs:
            assert d.input_schema.get("type") == "object"
    finally:
        mgr.close()


def test_manager_routes_calls_to_right_server_and_tool() -> None:
    mgr = MCPManager.start(
        [("fake", _fake_server_argv(), 5.0, 5.0)],
    )
    try:
        echo = mgr.call(f"{MCP_TOOL_PREFIX}fake__echo", {"text": "hi"})
        assert echo["content"][0]["text"] == "hi"
        shout = mgr.call(f"{MCP_TOOL_PREFIX}fake__shout", {"text": "hi"})
        assert shout["content"][0]["text"] == "HI"
    finally:
        mgr.close()


def test_manager_rejects_non_mcp_name() -> None:
    mgr = MCPManager.start([])
    try:
        with pytest.raises(MCPError, match="not an MCP tool name"):
            mgr.call("not_mcp", {})
    finally:
        mgr.close()


def test_manager_rejects_unknown_server() -> None:
    mgr = MCPManager.start([])
    try:
        with pytest.raises(MCPError, match="unknown MCP server"):
            mgr.call(f"{MCP_TOOL_PREFIX}nope__t", {})
    finally:
        mgr.close()


def test_manager_logs_and_skips_unstartable_server() -> None:
    logs: list[str] = []
    mgr = MCPManager.start(
        [("bogus", ("/this/binary/does/not/exist/agent6-test", "x"), 1.0, 1.0)],
        logger=logs.append,
    )
    try:
        assert mgr.descriptors() == ()
        assert any("failed to start" in m for m in logs)
    finally:
        mgr.close()


def test_manager_times_out_on_hanging_server() -> None:
    # 0.5s startup timeout; the hang server never responds, so start()
    # should log the failure and the manager should end up with zero
    # servers. We do NOT raise from MCPManager.start because the
    # design is "one bad server doesn't take the run down".
    logs: list[str] = []
    mgr = MCPManager.start(
        [("hang", _fake_server_argv(hang=True), 0.5, 0.5)],
        logger=logs.append,
    )
    try:
        assert mgr.descriptors() == ()
        assert any("timed out" in m for m in logs)
    finally:
        mgr.close()


def test_manager_close_is_idempotent() -> None:
    mgr = MCPManager.start([("fake", _fake_server_argv(), 5.0, 5.0)])
    mgr.close()
    mgr.close()  # must not raise
