# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""MCPServer unit tests — handler routing + JSON-RPC framing."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config, load_config
from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, write_node
from agent6.mcp_server import MCPServer, _deny_approver  # pyright: ignore[reportPrivateUsage]
from agent6.tools.dispatch import ToolError

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
profile = "auto"
network = "no"
run_commands = "no"
protect_git = true
protect_agent6 = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
commit_strategy = "per_step"
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _config(tmp_path: Path, *, run_commands: str = "no") -> Config:
    toml = _VALID_TOML.replace('run_commands = "no"', f'run_commands = "{run_commands}"')
    p = tmp_path / "agent6.toml"
    p.write_text(toml, encoding="utf-8")
    return load_config(p)


def _server(tmp_path: Path, **kwargs: Any) -> MCPServer:
    cfg = _config(tmp_path, **kwargs)
    return MCPServer(
        root=tmp_path,
        config=cfg,
        stdin=io.BytesIO(),
        stdout=io.BytesIO(),
    )


def _roundtrip(server: MCPServer, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Feed messages into the server's stdin, drive serve() to EOF,
    and parse responses from stdout."""
    payload = b"".join(json.dumps(m).encode("utf-8") + b"\n" for m in messages)
    server._stdin = io.BytesIO(payload)  # type: ignore[attr-defined]  # test-only stdin swap
    server._stdout = io.BytesIO()  # type: ignore[attr-defined]
    server.serve()
    server._stdout.seek(0)  # type: ignore[attr-defined]
    out: list[dict[str, Any]] = []
    for line in server._stdout.readlines():  # type: ignore[attr-defined]
        out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# JSON-RPC framing
# ---------------------------------------------------------------------------


def test_initialize_returns_server_info(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}],
    )
    assert len(resps) == 1
    assert resps[0]["id"] == 1
    info = resps[0]["result"]
    assert info["serverInfo"]["name"] == "agent6"
    assert info["protocolVersion"] == "2024-11-05"
    assert "tools" in info["capabilities"]


def test_tools_list_advertises_five_tools(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}],
    )
    tools = resps[0]["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {
        "run_verify",
        "run_in_sandbox",
        "apply_patch_in_sandbox",
        "query_dag",
        "list_runs",
    }
    # Every tool advertises a JSON-schema object.
    for t in tools:
        assert t["inputSchema"]["type"] == "object"


def test_unknown_method_returns_rpc_error(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "id": 3, "method": "nonsense", "params": {}}],
    )
    assert resps[0]["error"]["code"] == -32601
    assert "nonsense" in resps[0]["error"]["message"]


def test_unknown_tool_returns_rpc_error(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "no_such_tool", "arguments": {}},
            }
        ],
    )
    assert resps[0]["error"]["code"] == -32601


def test_notifications_produce_no_response(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}],
    )
    assert resps == []


def test_malformed_json_is_skipped(tmp_path: Path) -> None:
    server = _server(tmp_path)
    server._stdin = io.BytesIO(  # type: ignore[attr-defined]
        b"not json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}).encode("utf-8")
        + b"\n"
    )
    server._stdout = io.BytesIO()  # type: ignore[attr-defined]
    server.serve()
    server._stdout.seek(0)  # type: ignore[attr-defined]
    lines = server._stdout.readlines()  # type: ignore[attr-defined]
    assert len(lines) == 1  # only the valid request got a response
    assert json.loads(lines[0])["id"] == 7


# ---------------------------------------------------------------------------
# Tool handlers that don't need the jail
# ---------------------------------------------------------------------------


def test_list_runs_empty(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_runs", "arguments": {}},
            }
        ],
    )
    payload = resps[0]["result"]["structuredContent"]
    assert payload == {"runs": []}


def test_list_runs_reads_manifests(tmp_path: Path) -> None:
    runs = tmp_path / ".agent6" / "runs"
    (runs / "run-a").mkdir(parents=True)
    (runs / "run-b").mkdir(parents=True)
    (runs / "run-a" / "manifest.json").write_text(json.dumps({"task": "alpha"}), encoding="utf-8")
    # run-b has no manifest -> entry without one
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_runs", "arguments": {}},
            }
        ],
    )
    runs_out = resps[0]["result"]["structuredContent"]["runs"]
    assert [r["run_id"] for r in runs_out] == ["run-b", "run-a"]
    assert runs_out[1]["manifest"] == {"task": "alpha"}
    assert "manifest" not in runs_out[0]


def test_query_dag_missing_run_returns_tool_error(tmp_path: Path) -> None:
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "query_dag", "arguments": {}},
            }
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert "no runs" in resps[0]["result"]["content"][0]["text"]


def test_query_dag_reads_persisted_nodes(tmp_path: Path) -> None:
    layout = RunLayout(root=tmp_path, run_id="r1")
    layout.ensure()
    node = TaskNode(
        id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        parent_id=None,
        title="root task",
        status="pending",
        rationale="seed",
        acceptance="done",
        relevant_paths=(),
        created_by="planner",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    write_node(layout, {node.id: node}, node)
    server = _server(tmp_path)
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "query_dag",
                    "arguments": {"run_id": "r1"},
                },
            }
        ],
    )
    payload = resps[0]["result"]["structuredContent"]
    assert payload["run_id"] == "r1"
    assert payload["nodes"]["01ARZ3NDEKTSV4RRFFQ69G5FAV"]["title"] == "root task"


# ---------------------------------------------------------------------------
# Tool handlers that delegate to the jailed dispatcher
# ---------------------------------------------------------------------------


def test_run_verify_delegates_to_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path)
    captured: list[tuple[str, dict[str, Any]]] = []

    def fake_dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
        captured.append((name, args))
        return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.0}

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_verify", "arguments": {}},
            }
        ],
    )
    assert captured == [("run_verify", {})]
    assert resps[0]["result"]["structuredContent"]["returncode"] == 0


def test_run_in_sandbox_validates_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path, run_commands="yes")

    def fake_dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.0}

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    # Empty argv -> tool error.
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_in_sandbox", "arguments": {"argv": []}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "run_in_sandbox",
                    "arguments": {"argv": ["echo", "hi"]},
                },
            },
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert resps[1]["result"]["structuredContent"]["stdout"] == "ok"


def test_apply_patch_runs_verify_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path)
    calls: list[str] = []

    def fake_dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append(name)
        if name == "apply_patch":
            return {"path": "foo.py", "bytes_written": 5}
        if name == "run_verify":
            return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 0.1}
        raise AssertionError(name)

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "apply_patch_in_sandbox",
                    "arguments": {"path": "foo.py", "patch": "diff"},
                },
            }
        ],
    )
    assert calls == ["apply_patch", "run_verify"]
    payload = resps[0]["result"]["structuredContent"]
    assert payload["apply"]["bytes_written"] == 5
    assert payload["verify"]["returncode"] == 0


def test_apply_patch_surfaces_tool_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path)

    def fake_dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise ToolError("patch did not apply")

    monkeypatch.setattr(server._dispatcher, "dispatch", fake_dispatch)  # type: ignore[attr-defined]
    resps = _roundtrip(
        server,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "apply_patch_in_sandbox",
                    "arguments": {"path": "foo.py", "patch": "diff"},
                },
            }
        ],
    )
    assert resps[0]["result"]["isError"] is True
    assert "patch did not apply" in resps[0]["result"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Approver
# ---------------------------------------------------------------------------


def test_deny_approver_always_returns_false() -> None:
    assert _deny_approver("Allow this?") is False
