# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Was the gate already red before this run touched anything?

Observed during the run: a verify against an unmodified tree is the answer, with no second gate run
in the teardown.
"""

from __future__ import annotations

import pathlib
import types
from unittest import mock

import pytest

from agent6.harness import _chain, _finish_gates, _loop_state, loop
from agent6.tools import results
from tests.unit.turn_context import turn_context

_BASE = "b" * 40


def _wf(*, head: str = _BASE, clean: bool = True) -> loop.Harness:
    wf = loop.Harness.__new__(loop.Harness)
    wf.chain = _chain.RunChain(pathlib.Path("/nonexistent"), base_sha=_BASE)
    object.__setattr__(
        wf, "_git_status", lambda: types.SimpleNamespace(is_clean=clean, head_sha=head)
    )
    object.__setattr__(wf, "_emit", _quiet)
    wf.config = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        harness=types.SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=("pytest",),
            verify_when="finish",
            verify_retries=2,
            verify_timeout_s=60.0,
            verify_infer=True,
        )
    )
    # Gate presence reads the command policy first: a gate someone may run.
    wf.dispatcher = types.SimpleNamespace(command_policy=lambda: "yes")  # pyright: ignore[reportAttributeAccessIssue]
    return wf


def _quiet(*_a: object, **_k: object) -> None:
    return None


def _patch_git(monkeypatch: pytest.MonkeyPatch, wf: loop.Harness) -> None:
    def _status(_root: object, **_kw: object) -> object:
        return wf._git_status()  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr("agent6.harness._verify_gate.git_status", _status)


def _state() -> _loop_state.LoopState:
    return _loop_state.LoopState(original_task="t", tool_calls=0)


def _verify(rc: int, *, duration_s: float = 5.0) -> results.ExecResult:
    return results.ExecResult(
        returncode=rc, stdout="", stderr="", duration_s=duration_s, exec_failed=False
    )


def _turn() -> _loop_state.TurnState:
    return _loop_state.TurnState(iteration=1, resp=mock.MagicMock(), assistant=mock.MagicMock())


@pytest.mark.parametrize("rc", [0, 1])
def test_a_verify_at_the_base_commit_is_the_baseline(
    rc: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, turn = _state(), _turn()
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf.gate.note_result(state, turn, _verify(rc))
    assert state.verify.baseline_ok is (rc == 0)


def test_the_worker_is_told_when_it_inherited_a_red_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker is told during the run when it inherited a red gate."""
    state, turn = _state(), _turn()
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf.gate.note_result(state, turn, _verify(1))
    assert any("already failing" in str(n) for n in turn.tool_results)


def test_a_execution_that_moved_past_the_base_claims_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An execution that moved past the base claims nothing.

    A resume commits the previous execution's work first, so execution two opens on a clean tree
    whose HEAD carries execution one's breakage; `/parallel` merges lane commits the same way.
    """
    state, turn = _state(), _turn()
    wf = _wf(head="c" * 40)
    _patch_git(monkeypatch, wf)
    wf.gate.note_result(state, turn, _verify(1))
    assert state.verify.baseline_ok is None


def test_a_dirty_tree_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    state, turn = _state(), _turn()
    wf = _wf(clean=False)
    _patch_git(monkeypatch, wf)
    wf.gate.note_result(state, turn, _verify(1))
    assert state.verify.baseline_ok is None


def test_an_unreadable_git_claims_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable git fails closed here.

    "assume clean" would exonerate the run's own breakage.
    """
    from agent6 import git_ops

    def _boom(_root: object, **_kw: object) -> object:
        raise git_ops.GitError("index.lock held")

    state, turn = _state(), _turn()
    monkeypatch.setattr("agent6.harness._verify_gate.git_status", _boom)
    _wf().gate.note_result(state, turn, _verify(1))
    assert state.verify.baseline_ok is None


def test_a_run_that_already_went_green_owns_its_later_red(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that already went green owns its later red, even with a red gate at the base."""
    state, turn = _state(), _turn()
    state.verify.ever_passed = True
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf.gate.note_result(state, turn, _verify(1))
    assert state.verify.baseline_ok is None


def test_a_recovered_red_baseline_does_not_exempt_a_later_regression() -> None:
    """Once this run made an inherited red gate green.

    Later red loses the inherited-failure label.
    """
    wf = _wf()
    wf.mode = "run"
    wf.dispatcher = mock.MagicMock()
    wf.dispatcher.command_policy.return_value = "ask"
    wf.config.harness.verify_when = "finish"
    wf.config.harness.verify_retries = 2
    state = _state()
    state.verify.baseline_ok = False
    state.verify.ever_passed = True
    state.verify.last_ok = False
    finish = _finish_gates.FinishCall("finish_session", "done")

    assert _finish_gates.red_gate_returns(
        wf.config.harness.verify_when,
        wf.config.harness.verify_retries,
        state.verify,
        state.gates,
        gate_present=wf.gate.present(state.verify),
    )
    assert (
        _finish_gates.finish_reason(
            finish.kind,
            stale_gate=finish.stale_gate,
            tree_green=wf.gate.tree_green(state.verify),
            verify=state.verify,
        )
        == "finish_session"
    )


@pytest.mark.parametrize(
    "result",
    [
        results.ExecResult(
            returncode=127,
            stdout="",
            stderr="pytest: command not found",
            duration_s=0.01,
            exec_failed=False,
        ),
        results.ExecResult(
            returncode=124, stdout="", stderr="", duration_s=600.0, exec_failed=False
        ),
        results.ExecResult(returncode=1, stdout="", stderr="", duration_s=1.0, exec_failed=True),
    ],
    ids=["runner-absent", "timed-out", "could-not-exec"],
)
def test_a_gate_that_never_produced_a_verdict_is_not_a_red_baseline(
    result: results.ExecResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording one would excuse every real failure for the rest of the run."""
    state, turn = _state(), _turn()
    wf = _wf()
    _patch_git(monkeypatch, wf)
    wf.gate.note_result(state, turn, result)
    assert state.verify.baseline_ok is None


def test_a_plan_pass_is_not_reported_as_a_red_gate() -> None:
    """A plan pass is never relabelled as a red gate: plan mode runs the gate but never edits."""
    wf = loop.Harness.__new__(loop.Harness)
    wf.chain = _chain.RunChain(pathlib.Path("/nonexistent"))
    wf.mode = "plan"
    wf.config = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        harness=types.SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=("pytest",),
            verify_when="finish",
            verify_retries=2,
            verify_timeout_s=60.0,
            verify_infer=True,
        )
    )
    wf.dispatcher = types.SimpleNamespace(command_policy=lambda: "yes")  # pyright: ignore[reportAttributeAccessIssue]
    state = _state()
    state.verify.baseline_ok = False
    state.verify.last_ok = False
    finish = _finish_gates.FinishCall("finish_planning", "done")
    assert (
        _finish_gates.finish_reason(
            finish.kind,
            stale_gate=finish.stale_gate,
            tree_green=wf.gate.tree_green(state.verify),
            verify=state.verify,
        )
        == "finish_planning"
    )


def test_a_red_tree_still_exits_red_whoever_caused_it() -> None:
    """A red tree exits red whoever caused it; attribution belongs in the word.

    The exit code.
    """
    from agent6.app import finalize
    from agent6.harness import _snapshot

    inherited = _snapshot.SessionResult(
        completed=True,
        reason="gate_red_at_base",
        summary="s",
        iterations=1,
        tool_calls=1,
        verified="failed",
    )
    assert finalize.session_exit_code(inherited) == 4


def test_the_listing_and_the_header_agree_on_the_word() -> None:
    from agent6.viewmodel import listing

    assert listing.status_word(finished=True, all_passed=False, end_reason="gate_red_at_base") == (
        "finished",
        "gate was already red",
    )


def test_green_is_not_demanded_of_a_run_that_inherited_a_red_gate(tmp_path: pathlib.Path) -> None:
    """A red finish is returned until the gate goes green.

    Whatever the gate looked like at start.
    """
    wf = _wf()
    wf.mode = "run"
    wf.config = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        harness=types.SimpleNamespace(
            standing_patience=-1,
            went_quiet_max_nudges=4,
            loop_guard_kill_threshold=10,
            stagnation_notice_after_s=300.0,
            verify_command=("pytest",),
            verify_when="finish",
            verify_retries=2,
            verify_timeout_s=60.0,
            verify_infer=True,
        )
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.last_ok = False
    state.verify.baseline_ok = False
    turn = _loop_state.TurnState(iteration=1, resp=mock.MagicMock(), assistant=mock.MagicMock())
    turn.finish = _finish_gates.FinishCall("finish_session", "done")

    ctx = turn_context(tree_green=lambda: False, gate_present=lambda: True)
    bounced = _finish_gates.verify_finish(turn, state, ctx) is not None
    assert not bounced, "the finish was bounced over an inherited failure"
