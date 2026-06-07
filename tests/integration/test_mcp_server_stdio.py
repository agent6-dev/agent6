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
agent_network = "open"
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
    (tmp_path / ".agent6" / "runs" / "demo").mkdir(parents=True)
    (tmp_path / ".agent6" / "runs" / "demo" / "manifest.json").write_text(
        json.dumps({"task": "demo-task"}), encoding="utf-8"
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from agent6.cli import main; raise SystemExit(main())",
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
    assert runs == [{"run_id": "demo", "manifest": {"task": "demo-task"}}]
