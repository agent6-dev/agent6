# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Subprocess round-trip for the four-seam machine-agent IPC.

Spawns the real `python -m agent6.ui.cli.machine_agent <request.json>
<result.json>` on a request that REFUSES (no API key / network needed), and
asserts the process exits 0 with a `result.json` that validates back as an
`AgentExecResult`. Locks the argv contract + the request/result file shapes
end-to-end, and pins that a preflight refusal (or a config-setup error) is
salvaged into a written `error` result rather than a bare traceback + a
missing result the host has to recover from.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent6.app.machine_agent import MachineAgentRequest
from agent6.machine import AgentExecResult, AgentRequest


def _round_trip(tmp_path: Path, req: MachineAgentRequest) -> tuple[int, AgentExecResult | None]:
    req_file = tmp_path / "request.json"
    out_file = tmp_path / "result.json"
    req_file.write_text(req.model_dump_json(), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "agent6.ui.cli.machine_agent", str(req_file), str(out_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    result = (
        AgentExecResult.model_validate_json(out_file.read_text(encoding="utf-8"))
        if out_file.is_file()
        else None
    )
    return proc.returncode, result


def test_network_refusal_writes_a_valid_error_result(tmp_path: Path) -> None:
    # agent_network='local' on the 'hardened' profile is an unenforceable combo:
    # check_network_profile refuses before any provider call. No key/network.
    cwd = tmp_path / "repo"
    cwd.mkdir()
    req = MachineAgentRequest(
        cwd=cwd,
        root=cwd,
        overlay={"sandbox": {"agent_network": "local"}},
        profile="hardened",
        transcript_dir=cwd / "transcripts",
        request=AgentRequest(prompt="hi", timeout_s=30.0, mode="agent"),
    )
    rc, result = _round_trip(tmp_path, req)
    assert rc == 0
    assert result is not None, "the subprocess must WRITE result.json, not crash"
    assert result.reason == "error"
    assert result.payload is None


def test_config_error_is_salvaged_not_a_traceback(tmp_path: Path) -> None:
    # A bad machine [config] overlay makes the config re-validation raise. run_one
    # must catch it and write an `error` result -- not let a pydantic traceback
    # escape with a non-zero exit and no result.json (host-side salvage only).
    cwd = tmp_path / "repo"
    cwd.mkdir()
    req = MachineAgentRequest(
        cwd=cwd,
        root=cwd,
        overlay={"budget": {"max_input_tokens": -5}},
        profile="none",
        transcript_dir=cwd / "transcripts",
        request=AgentRequest(prompt="hi", timeout_s=30.0, mode="agent"),
    )
    rc, result = _round_trip(tmp_path, req)
    assert rc == 0
    assert result is not None and result.reason == "error"
