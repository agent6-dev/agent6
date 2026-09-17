# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The sandbox-reachability note: the second consecutive jail exec failure of
a binary the host has draws one note; a nonzero exit, an absent binary, a
different binary or a later call draws none."""

from __future__ import annotations

from agent6.tools.results import ExecResult, RawResult
from agent6.workflows._guards import unreachable_tool
from agent6.workflows._loop_state import LoopState


def _exec(*, exec_failed: bool) -> ExecResult:
    return ExecResult(
        returncode=127, stdout="", stderr="not executable", duration_s=0.0, exec_failed=exec_failed
    )


def test_the_second_exec_failure_of_a_host_binary_draws_one_note() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    argv = {"argv": ["sh", "-c", "true"]}
    assert unreachable_tool(state, "run_command", argv, _exec(exec_failed=True)) is None
    note = unreachable_tool(state, "run_command", argv, _exec(exec_failed=True))
    assert note is not None and "`sh` is installed on this machine" in note.text
    assert note.event == "loop.sandbox_tool_unreachable" and note.fields == {"binary": "sh"}
    assert note.log == "LOOP: sandbox tool unreachable: sh exists on host, fails in jail"
    assert state.reach.warned is True
    assert unreachable_tool(state, "run_command", argv, _exec(exec_failed=True)) is None


def test_a_nonzero_exit_an_absent_binary_or_another_tool_is_no_reach_problem() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    exited = {"argv": ["sh", "-c", "exit 1"]}
    for _ in range(3):
        assert unreachable_tool(state, "run_command", exited, _exec(exec_failed=False)) is None
    missing = {"argv": ["no-such-binary-on-this-host"]}
    for _ in range(3):
        assert unreachable_tool(state, "run_command", missing, _exec(exec_failed=True)) is None
    assert unreachable_tool(state, "read_file", {"path": "x"}, RawResult({"content": ""})) is None
