# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A model's command is handed back, never killed for taking too long.

A wall-clock timeout has to answer a question it cannot: whether a command that
has run twenty minutes is stuck or working. `[workflow].command_checkin_s`
replaces the kill with a hand-back, so the judgement goes to the model (or the
operator), and the command keeps running either way.

The operator's gate (`run_verify_command`) is deliberately NOT part of this: the
loop needs a verdict from it, not a handle.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher


def _dispatcher(tmp_path: Path, checkin: float) -> ToolDispatcher:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "workflow": {"command_checkin_s": checkin},
        }
    )
    return ToolDispatcher(
        root=root,
        config=cfg,
        isolation="none",
        session_dir=session_dir,
        use_jail_session=True,
    )


def _run(d: ToolDispatcher, script: str) -> dict[str, Any]:
    return d.dispatch("run_command", {"argv": ["/bin/sh", "-c", script]}).to_wire()


def test_a_command_that_finishes_is_an_ordinary_result(tmp_path: Path) -> None:
    d = _dispatcher(tmp_path, checkin=30.0)
    try:
        out = _run(d, "echo fast; exit 2")
    finally:
        d.close()
    assert out["returncode"] == 2
    assert out["stdout"] == "fast\n"
    assert "still_running" not in out
    assert "background_id" not in out


def test_a_command_outliving_the_checkin_comes_back_as_a_background_job(tmp_path: Path) -> None:
    """One ExecResult shape either way: `returncode` is null and a
    `background_id` names where the command went, so nothing has to branch on
    "a result OR a handle"."""
    d = _dispatcher(tmp_path, checkin=0.5)
    try:
        started = time.monotonic()
        out = _run(d, "echo starting; sleep 30; echo never")
        elapsed = time.monotonic() - started

        assert out["returncode"] is None
        assert out["still_running"] is True
        assert out["background_id"] == "bg1"
        # The output from before the hand-back comes back with it.
        assert out["stdout"] == "starting\n"
        assert elapsed < 10, "the hand-back waited for the command instead of returning"

        # It really is still running, and it is an ordinary background job now.
        read = d.dispatch("read_background", {"id": "bg1"}).to_wire()
        assert "running" in str(read["shells"])
        assert "starting" in str(read["output"])
        stopped = d.dispatch("stop_background", {"id": "bg1"}).to_wire()
        assert "stopped" in str(stopped["shells"])
    finally:
        d.close()


def test_a_zero_checkin_waits_for_the_command(tmp_path: Path) -> None:
    """`0` disables the hand-back: correct when a human is watching and can
    interrupt, and the path a run with no background roster falls back to."""
    d = _dispatcher(tmp_path, checkin=0.0)
    try:
        out = _run(d, "sleep 1; echo waited")
    finally:
        d.close()
    assert out["returncode"] == 0
    assert out["stdout"] == "waited\n"
    assert "background_id" not in out


def test_the_verify_gate_is_never_handed_back(tmp_path: Path) -> None:
    """The operator's gate must return a verdict; a handle would leave the loop
    with nothing to decide on."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    session_dir = tmp_path / "session"
    session_dir.mkdir(exist_ok=True)
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "workflow": {
                "command_checkin_s": 0.5,
                "verify_command": ["/bin/sh", "-c", "sleep 2; echo verified"],
                "verify_timeout_s": 30.0,
            },
        }
    )
    d = ToolDispatcher(
        root=root, config=cfg, isolation="none", session_dir=session_dir, use_jail_session=True
    )
    try:
        out = d.dispatch("run_verify_command", {}).to_wire()
    finally:
        d.close()
    assert out["returncode"] == 0
    assert "verified" in out["stdout"]
    assert "background_id" not in out


@pytest.mark.parametrize("checkin", [0.5, 0.0])
def test_nothing_a_handed_back_command_started_outlives_the_run(
    tmp_path: Path, checkin: float
) -> None:
    """Teardown stops the roster, so a command that was handed back dies with
    the run exactly like one the model backgrounded itself."""
    d = _dispatcher(tmp_path, checkin=checkin)
    marker = tmp_path / "pid"
    try:
        _run(d, f"echo $$ > {marker}; sleep 0.2")
    finally:
        d.close()
    assert marker.exists()
    pid = int(marker.read_text().strip())
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        state = "gone"
    assert state in ("gone", "Z")
