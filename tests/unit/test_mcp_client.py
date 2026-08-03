# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Smoke-tests for the stdio MCP client.

Uses a tiny in-tree Python "MCP server" that talks just enough JSON-RPC
to satisfy ``initialize`` + ``tools/list`` + ``tools/call``. No external
dependency.
"""

from __future__ import annotations

import json
import sys
import textwrap
import threading

import pytest

from agent6.tools.mcp_client import (
    MCP_TOOL_PREFIX,
    MCPError,
    MCPManager,
    MCPServerSpec,
)


def _fake_server_argv(
    *, hang: bool = False, crash_after_init: bool = False, bad_tool: bool = False
) -> tuple[str, ...]:
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
        BAD_TOOL = {bad_tool!r}
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
                tools = [
                    {{"name": "echo", "description": "echo the input",
                      "inputSchema": {{"type": "object",
                                       "properties": {{"text": {{"type": "string"}}}}}}}},
                    {{"name": "shout", "description": "upper-case echo",
                      "inputSchema": {{"type": "object",
                                       "properties": {{"text": {{"type": "string"}}}}}}}},
                ]
                if BAD_TOOL:
                    tools.append({{"name": "has a space", "description": "invalid",
                                   "inputSchema": {{"type": "object"}}}})
                reply(msg["id"], {{"tools": tools}})
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
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ],
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


def test_manager_skips_tools_with_invalid_names() -> None:
    # A server-advertised tool whose name isn't [A-Za-z0-9_-] can't form a valid
    # provider tool name; it must be skipped, not poison the whole tools array.
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_fake_server_argv(bad_tool=True),
                startup_timeout_s=5.0,
                call_timeout_s=5.0,
            )
        ]
    )
    try:
        names = sorted(d.qualified_name for d in mgr.descriptors())
        assert names == [f"{MCP_TOOL_PREFIX}fake__echo", f"{MCP_TOOL_PREFIX}fake__shout"]
    finally:
        mgr.close()


def test_manager_routes_calls_to_right_server_and_tool() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ],
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
        [
            MCPServerSpec(
                name="bogus",
                command=("/this/binary/does/not/exist/agent6-test", "x"),
                startup_timeout_s=1.0,
                call_timeout_s=1.0,
            )
        ],
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
        [
            MCPServerSpec(
                name="hang",
                command=_fake_server_argv(hang=True),
                startup_timeout_s=0.5,
                call_timeout_s=0.5,
            )
        ],
        logger=logs.append,
    )
    try:
        assert mgr.descriptors() == ()
        assert any("timed out" in m for m in logs)
    finally:
        mgr.close()


def test_manager_close_is_idempotent() -> None:
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake", command=_fake_server_argv(), startup_timeout_s=5.0, call_timeout_s=5.0
            )
        ]
    )
    mgr.close()
    mgr.close()  # must not raise


def test_concurrent_calls_do_not_interleave_stdin_writes() -> None:
    """tools/call from concurrent threads (explore-review seats share one
    dispatcher across a thread pool) must serialize on the server's stdin:
    pipe writes larger than PIPE_BUF interleave across unlocked writers,
    corrupting the JSON-RPC framing -- the server read malformed JSON and
    died, failing every in-flight call."""
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="fake",
                command=_fake_server_argv(),
                startup_timeout_s=10.0,
                call_timeout_s=30.0,
            )
        ]
    )
    try:
        payloads = {i: f"p{i}-" + "x" * 300_000 for i in range(8)}
        results: dict[int, str] = {}
        errors: list[Exception] = []

        def call(i: int) -> None:
            try:
                out = mgr.call(f"{MCP_TOOL_PREFIX}fake__echo", {"text": payloads[i]})
                results[i] = out["content"][0]["text"]
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call, args=(i,)) for i in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors
        assert results == payloads
    finally:
        mgr.close()


def test_a_server_is_not_handed_the_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawn passed no `env`, so a server inherited the agent's FULL
    environment -- including the keys resolved via `[providers.*].api_key_env`.
    An MCP server is third-party code that may log or forward what it is given.
    Proved by asking the server itself what it can see."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-DECOY")
    monkeypatch.setenv("MCP_PROBE_TOKEN", "named-and-wanted")
    script = (
        "import json,os,sys\n"
        "def w(o): sys.stdout.write(json.dumps(o)+chr(10)); sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    m=json.loads(line)\n"
        "    if m.get('method')=='initialize':\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2024-11-05',"
        "'capabilities':{},'serverInfo':{'name':'p','version':'1'}}})\n"
        "    elif m.get('method')=='tools/list':\n"
        "        seen=sorted(k for k in os.environ if 'API_KEY' in k or k=='MCP_PROBE_TOKEN')\n"
        "        w({'jsonrpc':'2.0','id':m['id'],'result':{'tools':[{'name':'x',"
        "'description':json.dumps(seen),'inputSchema':{'type':'object'}}]}})\n"
    )
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="probe",
                command=(sys.executable, "-c", script),
                startup_timeout_s=10.0,
                call_timeout_s=10.0,
                pass_env=("MCP_PROBE_TOKEN",),
            )
        ]
    )
    try:
        seen = json.loads(mgr.descriptors()[0].description)
    finally:
        mgr.close()
    assert seen == ["MCP_PROBE_TOKEN"], "a server sees only what it named"
