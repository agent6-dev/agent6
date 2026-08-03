# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Was the gate already red before this run touched anything?

Observed for free during the run -- a verify against an unmodified tree IS the
answer -- rather than bought with a second full gate run in the teardown.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent6.tools.results import ExecResult
from agent6.workflows.loop import (
    Workflow,
    _LoopState,  # pyright: ignore[reportPrivateUsage]
    _TurnState,  # pyright: ignore[reportPrivateUsage]
)


def _wf(*, dirty: bool) -> Workflow:
    wf = Workflow.__new__(Workflow)
    object.__setattr__(wf, "_worktree_dirty", lambda: dirty)

    def _quiet(*_a: object, **_k: object) -> None:
        return None

    object.__setattr__(wf, "_emit", _quiet)
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest",))
    )
    return wf


def _verify(rc: int) -> ExecResult:
    return ExecResult(returncode=rc, stdout="", stderr="", duration_s=0.1, exec_failed=False)


@pytest.mark.parametrize("rc", [0, 1])
def test_a_verify_on_an_untouched_tree_is_the_baseline(rc: int) -> None:
    state = _LoopState(original_task="t", tool_calls=0)
    turn = _TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    _wf(dirty=False)._note_verify_result(state, turn, _verify(rc))  # pyright: ignore[reportPrivateUsage]
    assert state.baseline_ok is (rc == 0)


def test_the_worker_is_told_when_it_inherited_a_red_gate() -> None:
    """So it stops chasing failures it did not cause, DURING the run -- which
    is worth more than the same fact explained afterwards."""
    state = _LoopState(original_task="t", tool_calls=0)
    turn = _TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    _wf(dirty=False)._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert any("ALREADY failing" in str(n) for n in turn.tool_results)


@pytest.mark.parametrize(("dirty", "edited"), [(True, False), (False, True)])
def test_a_verify_over_changed_work_says_nothing_about_the_base(dirty: bool, edited: bool) -> None:
    """ "I do not know" is the honest answer, and the end-of-run block says so."""
    state = _LoopState(original_task="t", tool_calls=0)
    state.ever_edited = edited
    turn = _TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    _wf(dirty=dirty)._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    assert state.baseline_ok is None


def test_the_first_observation_wins() -> None:
    """The base commit is judged once; a later verify judges the run's work."""
    state = _LoopState(original_task="t", tool_calls=0)
    turn = _TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    wf = _wf(dirty=False)
    wf._note_verify_result(state, turn, _verify(1))  # pyright: ignore[reportPrivateUsage]
    wf._note_verify_result(state, turn, _verify(0))  # pyright: ignore[reportPrivateUsage]
    assert state.baseline_ok is False


def test_a_red_tree_still_exits_red_whoever_caused_it() -> None:
    """Attribution belongs in the word, not the exit code: a script reading 0
    would take it as a passing gate, and the tree is not green either way."""
    from agent6.app.finalize import run_exit_code
    from agent6.workflows._run_state import RunResult

    inherited = RunResult(
        completed=True,
        reason="gate_red_at_base",
        summary="s",
        iterations=1,
        tool_calls=1,
        verified="failed",
    )
    assert run_exit_code(inherited) == 4


def test_the_listing_and_the_header_agree_on_the_word() -> None:
    from agent6.viewmodel.listing import status_word

    assert status_word(finished=True, all_passed=False, end_reason="gate_red_at_base") == (
        "finished",
        "gate was already red",
    )


def test_green_is_not_demanded_of_a_run_that_inherited_a_red_gate(tmp_path: Path) -> None:
    """`require_verify_to_finish` bounces a finish until the gate goes green.
    Over a gate that was already red, that is demanding the worker repair
    whatever it inherited before it may stop."""
    wf = _wf(dirty=False)
    wf.mode = "run"
    wf.config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        workflow=SimpleNamespace(verify_command=("pytest",), require_verify_to_finish=True)
    )
    state = _LoopState(original_task="t", tool_calls=0)
    state.last_verify_ok = False
    state.baseline_ok = False
    turn = _TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    turn.finish_signal = MagicMock()
    turn.finish_kind = "finish_run"

    wf._gate_verify_green(state, turn)  # pyright: ignore[reportPrivateUsage]

    assert turn.finish_signal is not None, "the finish was bounced over an inherited failure"


def test_a_broken_gate_is_not_a_red_baseline() -> None:
    """A verify that exits instantly without running anything (runner absent)
    is a BROKEN gate, not a red one. Recorded as the baseline it would excuse
    every real failure for the rest of the run."""
    state = _LoopState(original_task="t", tool_calls=0)
    turn = _TurnState(iteration=1, resp=MagicMock(), assistant=MagicMock())
    broken = ExecResult(
        returncode=127,
        stdout="",
        stderr="pytest: command not found",
        duration_s=0.01,
        exec_failed=False,
    )
    _wf(dirty=False)._note_verify_result(state, turn, broken)  # pyright: ignore[reportPrivateUsage]
    assert state.baseline_ok is None
