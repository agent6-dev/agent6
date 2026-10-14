# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A worker may say the gate is wrong instead of reverting correct work."""

from __future__ import annotations

import contextlib
import io
import pathlib
import shlex
import subprocess
from unittest import mock

import pytest

from agent6 import paths
from agent6.app import reporter
from agent6.config import Config
from agent6.harness import _chain, _conversation, _loop_state, _operator, _snapshot, loop
from agent6.tools import results
from agent6.ui import cli
from agent6.viewmodel import listing


def test_a_stale_gate_finish_reads_finished_with_its_cause() -> None:
    """A finish over a gate the worker called stale read "failed · gate stale".

    The same finish without the claim read "finished · gate red", so making the claim cost
    the run its word. A stale gate is a detail, not a new status word: `died_without_end`,
    the compare gates and the TUI colours key off the existing set.
    """
    assert listing.status_word(finished=True, all_passed=False, end_reason="gate_stale") == (
        "finished",
        "gate_stale",
    )


def test_a_green_tree_is_still_what_passes() -> None:
    """The word mapping is grounded on all_passed; the worker records a proposal, never a verdict.

    `gate_stale` never reaches a green run (see finish_reason).
    """
    assert listing.status_word(finished=True, all_passed=True, end_reason="gate_stale") == (
        "passed",
        "",
    )


def test_the_tool_result_says_nothing_changed() -> None:
    """The tool result says nothing changed, so the model cannot believe it swapped the gate."""
    wire = results.FinishSessionResult(
        summary_text="done", result=None, stale_gate="uv run pytest tests/unit"
    ).to_wire()
    assert "unchanged" in wire["stale_gate"]
    assert "does not pass" in wire["stale_gate"]
    assert (
        results.FinishSessionResult(summary_text="d", result=None).to_wire().get("stale_gate")
        is None
    )


def test_the_operator_gets_a_paste_ready_line() -> None:
    """Applying the proposal is the operator's call, so the run prints the exact command instead."""
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True,
                reason="gate_stale",
                summary="s",
                iterations=1,
                tool_calls=1,
                stale_gate="uv run pytest tests/unit",
                verified="failed",
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    text = out.getvalue()
    assert "nothing changed" in text
    # argv, as `config set` accepts it: the shell string it proposes is
    # rejected with "Input should be a valid tuple".
    assert (
        "agent6 config set --repo harness.verify_command"
        ' \'["uv", "run", "pytest", "tests/unit"]\'' in text
    )


def test_the_adopt_line_pins_the_gate_for_this_repo_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The printed adopt command writes the per-repo config and leaves the global one alone.

    A proposed gate names repo-local paths (`venv/bin/python`); in the global config it
    would judge every repository on the machine.
    """
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    finalize._print_stale_gate(
        _snapshot.SessionResult(
            completed=True,
            reason="gate_stale",
            summary="s",
            iterations=1,
            tool_calls=1,
            stale_gate="venv/bin/python -m pytest -q",
            verified="failed",
        ),
        reporter=reporter.STDIO_REPORTER,
    )
    adopt = shlex.split(capsys.readouterr().out.splitlines()[-1])
    assert adopt[0] == "agent6"

    assert cli.main(adopt[1:]) == 0

    assert not paths.global_config_path().exists()
    assert "venv/bin/python" in paths.repo_config_path(tmp_path).read_text(encoding="utf-8")


def test_a_proposal_over_a_green_gate_is_not_printed() -> None:
    """It would ask the operator to replace a gate that just passed."""
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True,
                reason="finish_session",
                summary="s",
                iterations=1,
                tool_calls=1,
                stale_gate="uv run pytest tests/unit",
                verified="passed",
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    assert out.getvalue() == ""


def test_nothing_is_printed_without_a_declaration() -> None:
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True, reason="finish_session", summary="s", iterations=1, tool_calls=1
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    assert out.getvalue() == ""


@pytest.mark.parametrize(
    ("declared", "green", "expected"),
    [
        ("uv run pytest tests/unit", False, "gate_stale"),  # red + declared
        ("uv run pytest tests/unit", True, "finish_session"),  # green: it passed, truthfully
        # None = GATELESS: no gate exists, so none can be stale. Reading this as
        # "not green" made such a run pass, exit 0 and auto-merge.
        ("uv run pytest tests/unit", None, "finish_session"),
        ("", False, "finish_session"),  # red with no declaration is an ordinary finish
    ],
)
def test_a_declaration_names_the_end_only_over_a_red_tree(
    declared: str, green: bool | None, expected: str
) -> None:
    from agent6.harness import _finish_gates, _verify_verdict

    reason = _finish_gates.finish_reason(
        "finish_session",
        stale_gate=declared,
        tree_green=green,
        verify=_verify_verdict.VerifyVerdict(),
    )
    assert reason == expected


def test_the_verify_result_names_the_command_that_judged_the_run() -> None:
    """A worker cannot tell a real failure from a stale gate without knowing which command ran.

    It never chose this one: the gate is the operator's, or inferred from the repo.
    """
    wire = results.ExecResult(
        returncode=1,
        stdout="",
        stderr="boom",
        duration_s=0.5,
        exec_failed=False,
        command=("uv", "run", "pytest", "-k", "not slow"),
    ).to_wire()
    assert wire["command"] == "uv run pytest -k 'not slow'"


def test_a_command_the_model_chose_is_not_echoed_back() -> None:
    """run_command already knows its own argv; repeating it is noise."""
    wire = results.ExecResult(
        returncode=0, stdout="", stderr="", duration_s=0.1, exec_failed=False
    ).to_wire()
    assert "command" not in wire


def _finish_call(
    wf: loop.Harness, state: _loop_state.LoopState, tool_input: dict[str, object]
) -> _loop_state.TurnState:
    """Return the turn after a finish_session went through the capture and the finish gates."""
    turn = _loop_state.TurnState(
        iteration=3, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )
    wf._capture_finish(state, turn, "finish_session", tool_input)  # pyright: ignore[reportPrivateUsage]
    ctx = wf._turn_context(state, iteration=turn.iteration, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    wf._turn_finish_gates(state, turn, ctx)  # pyright: ignore[reportPrivateUsage]
    return turn


def _red_gated_run(tmp_path: pathlib.Path) -> tuple[loop.Harness, _loop_state.LoopState]:
    dispatcher = mock.MagicMock()
    dispatcher.command_policy.return_value = "yes"
    wf = loop.Harness(
        chain=_chain.RunChain(tmp_path),
        config=Config.model_validate(
            {"harness": {"verify_command": ["true"], "verify_when": "finish", "verify_retries": 2}}
        ),
        provider=mock.MagicMock(),
        dispatcher=dispatcher,
        logger=lambda _line: None,
        mode="run",
        bridge=_operator.OperatorBridge(stop_requested=lambda: True),
    )
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.note_fail("1 failed")
    return wf, state


def test_a_proposal_on_a_returned_finish_reaches_the_end_report(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A proposal was kept on the finish call alone, so a gate that returned the call lost it.

    The tool result had told the model the proposal was recorded for the operator.
    """
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    wf, state = _red_gated_run(tmp_path)
    returned = _finish_call(wf, state, {"summary": "done", "stale_gate": "pytest tests/unit"})
    assert returned.finish is None and returned.end_returned
    state.gates.verify_retries_used = 2
    stands = _finish_call(wf, state, {"summary": "done"})
    result = wf._turn_stop_checks(state, stands, _conversation.Conversation())  # pyright: ignore[reportPrivateUsage]
    assert result is not None and result.stale_gate == "pytest tests/unit"
    # The end is named by its own call, which made no claim.
    assert (result.reason, result.verified) == ("finish_session", "failed")
    finalize._print_stale_gate(result, reporter=reporter.STDIO_REPORTER)
    assert "  it proposes: pytest tests/unit" in capsys.readouterr().out


def test_the_latest_proposal_is_the_one_reported_at_any_end(tmp_path: pathlib.Path) -> None:
    """A later proposal replaces an earlier one, and an end that is no finish still reports it."""
    wf, state = _red_gated_run(tmp_path)
    _finish_call(wf, state, {"summary": "done", "stale_gate": "pytest tests/unit"})
    _finish_call(wf, state, {"summary": "done", "stale_gate": "pytest tests/fast"})
    stopped = wf._operator_boundary(_conversation.Conversation(), 4, state)  # pyright: ignore[reportPrivateUsage]
    assert stopped is not None and stopped.reason == "steer_abort"
    assert stopped.stale_gate == "pytest tests/fast"
