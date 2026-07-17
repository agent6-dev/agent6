# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Spawn `python -m agent6 mcp serve` as a subprocess and
round-trip line-delimited JSON-RPC over its stdio.

This is the only test that exercises the real argv-to-server pipeline
including arg parsing, config loading, and the stdio reader loop.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir


def _userns_available() -> bool:
    res = subprocess.run(["unshare", "-U", "-r", "true"], capture_output=True, check=False)
    return res.returncode == 0


_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
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
agent_network = "open"
run_commands = "no"
protect_git = true
[git]
require_clean_worktree = true
auto_stash = false
branch_per_run = true
allow_push = false
allow_force = false
allow_history_rewrite = false
[workflow]
verify_command = ["true"]
[budget]
max_input_tokens = 100000
max_output_tokens = 10000
"""


def _send_recv(
    proc: subprocess.Popen[bytes], messages: list[dict[str, object]]
) -> list[dict[str, object]]:
    assert proc.stdin is not None and proc.stdout is not None
    payload = b"".join(json.dumps(m).encode("utf-8") + b"\n" for m in messages)
    out, _err = proc.communicate(input=payload, timeout=20)
    responses: list[dict[str, object]] = []
    for raw in out.splitlines():
        if not raw.strip():
            continue
        responses.append(json.loads(raw))
    return responses


def test_mcp_serve_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(_VALID_TOML, encoding="utf-8")
    # Seed a run dir so list_runs has something to enumerate.
    (resolved_state_dir(tmp_path) / "runs" / "demo").mkdir(parents=True)
    (resolved_state_dir(tmp_path) / "runs" / "demo" / "manifest.json").write_text(
        json.dumps({"user_task": "demo-task"}), encoding="utf-8"
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from agent6.ui.cli import main; raise SystemExit(main())",
            "mcp",
            "serve",
            "--config",
            str(cfg_path),
        ],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        responses = _send_recv(
            proc,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "list_runs", "arguments": {}},
                },
            ],
        )
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0
    # 3 requests, 1 notification (no response) -> 3 responses.
    assert len(responses) == 3
    by_id = {r["id"]: r for r in responses}
    init = by_id[1]["result"]
    assert isinstance(init, dict)
    assert init["serverInfo"]["name"] == "agent6"
    tools_resp = by_id[2]["result"]
    assert isinstance(tools_resp, dict)
    tools = tools_resp["tools"]
    assert {t["name"] for t in tools} == {
        "run_verify",
        "run_in_sandbox",
        "apply_patch_in_sandbox",
        "query_dag",
        "list_runs",
    }
    list_runs_resp = by_id[3]["result"]
    assert isinstance(list_runs_resp, dict)
    runs = list_runs_resp["structuredContent"]["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "demo"
    # The manifest ships as the typed RunManifest dump (defaults filled in).
    assert runs[0]["manifest"]["user_task"] == "demo-task"
    assert runs[0]["manifest"]["version"] == 2


def test_mcp_run_verify_resolves_through_real_dispatcher(tmp_path: Path) -> None:
    """End-to-end `run_verify` through the real server + jailed dispatcher.

    Regression: the handler dispatched the dispatcher tool name `run_verify`,
    but the registered name is `run_verify_command`, so every call came back
    `{"isError": true, "text": "Unknown tool: run_verify"}`. The unit tests
    monkeypatched `dispatch` and so encoded the wrong name; only an end-to-end
    call against the real dispatcher catches it. verify_command is `["true"]`,
    which exits 0 inside the jail.
    """
    if not _userns_available():
        pytest.skip("unprivileged user namespaces not available")
    cfg_path = tmp_path / "agent6.toml"
    cfg_path.write_text(_VALID_TOML, encoding="utf-8")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from agent6.ui.cli import main; raise SystemExit(main())",
            "mcp",
            "serve",
            "--config",
            str(cfg_path),
        ],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        responses = _send_recv(
            proc,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "run_verify", "arguments": {}},
                }
            ],
        )
    finally:
        if proc.poll() is None:
            proc.kill()

    assert len(responses) == 1
    result = responses[0]["result"]
    assert isinstance(result, dict)
    # The bug surfaced as isError + "Unknown tool: run_verify"; assert neither.
    assert not result.get("isError"), result
    assert result["structuredContent"]["returncode"] == 0
