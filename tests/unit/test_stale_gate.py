# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A worker may say the gate is wrong instead of reverting correct work."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
from unittest import mock

import pytest

from agent6.app import reporter
from agent6.config import Config
from agent6.harness import (
    _chain,
    _conversation,
    _loop_state,
    _loop_steer,
    _operator,
    _snapshot,
    loop,
)
from agent6.tools import results
from agent6.viewmodel import _session_status


def test_a_stale_gate_finish_reads_finished_with_its_cause() -> None:
    """A finish over a gate the worker called stale read "failed · gate stale".

    The same finish without the claim read "finished · gate red", so making the claim cost
    the run its word. A stale gate is a detail, not a new status word: `died_without_end`,
    the compare gates and the TUI colours key off the existing set.
    """
    assert _session_status.status_word(
        finished=True, all_passed=False, end_reason="gate_stale"
    ) == (
        "finished",
        "gate_stale",
    )


def test_a_green_tree_is_still_what_passes() -> None:
    """The word mapping is grounded on all_passed; the worker records a proposal, never a verdict.

    `gate_stale` never reaches a green run (see finish_reason).
    """
    assert _session_status.status_word(finished=True, all_passed=True, end_reason="gate_stale") == (
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


def test_the_proposal_is_printed_for_the_operator_to_set() -> None:
    """The proposal is shown as the worker's words, with no command that would install it.

    agent6 cannot tell a gate that judges the work from one that always passes (`true`, a
    `;` list ending in prose), so applying it stays the operator's call.
    """
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True,
                reason="gate_stale",
                summary="s",
                turns=1,
                tool_calls=1,
                stale_gate="uv run pytest tests/unit",
                verified="failed",
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    text = out.getvalue()
    assert "  it says: uv run pytest tests/unit\n" in text
    assert "nothing changed" in text
    assert "agent6 config set --repo harness.verify_command" in text
    assert json.dumps(["uv", "run", "pytest", "tests/unit"]) not in text


def test_a_long_proposal_is_clipped() -> None:
    """A paragraph is printed clipped to one line, never as an argv."""
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    prose = (
        "The configured gate runs the whole suite including the slow integration tests, which "
        "take twenty minutes and need a database; the task only touches the parser, so the "
        "gate should be narrowed to the parser tests, for example pytest tests/parser, and "
        "the integration suite left to CI where the database exists."
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True,
                reason="gate_stale",
                summary="s",
                turns=1,
                tool_calls=1,
                stale_gate=prose,
                verified="failed",
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    text = out.getvalue()
    assert "it says: The configured gate runs" in text and text.count("...") == 1
    longest = max(len(line) for line in text.splitlines())
    assert longest <= finalize.STALE_GATE_NOTE_MAX_CHARS + 20


def test_a_short_prose_proposal_is_printed_whole_with_no_clip_mark() -> None:
    """Only a clipped proposal ends in `...`.

    On a two-line note under the clip length, a `...` would read as if text had been cut.
    """
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    note = "Narrow the gate to the parser tests.\nThe database suite runs in CI."
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True,
                reason="gate_stale",
                summary="s",
                turns=1,
                tool_calls=1,
                stale_gate=note,
                verified="failed",
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    text = out.getvalue()
    assert f"  it says: {' '.join(note.split())}\n" in text and "..." not in text


def _stale_report(proposal: str) -> str:
    from agent6.app import finalize  # pyright: ignore[reportPrivateUsage]

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finalize._print_stale_gate(
            _snapshot.SessionResult(
                completed=True,
                reason="gate_stale",
                summary="s",
                turns=1,
                tool_calls=1,
                stale_gate=proposal,
                verified="failed",
            ),
            reporter=reporter.STDIO_REPORTER,
        )
    return out.getvalue()


def test_a_bare_sh_c_proposal_prints_as_it_is() -> None:
    """`sh -c` with no line after it is printed as the worker's words."""
    text = _stale_report("sh -c")
    assert "  it says: sh -c\n" in text


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
                turns=1,
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
                completed=True, reason="finish_session", summary="s", turns=1, tool_calls=1
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
        turn=3, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )
    wf._capture_finish(state, turn, "finish_session", tool_input)  # pyright: ignore[reportPrivateUsage]
    ctx = wf._turn_context(state, turn=turn.turn, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    wf._turn_finish_gates(state, turn, ctx)  # pyright: ignore[reportPrivateUsage]
    return turn


def _red_gated_run(tmp_path: pathlib.Path) -> tuple[loop.Harness, _loop_state.LoopState]:
    dispatcher = mock.MagicMock()
    dispatcher.command_policy.return_value = "allow"
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
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], jail_bin: pathlib.Path
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
    assert "  it says: pytest tests/unit" in capsys.readouterr().out


def test_the_latest_proposal_is_the_one_reported_at_any_end(tmp_path: pathlib.Path) -> None:
    """A later proposal replaces an earlier one, and an end that is no finish still reports it."""
    wf, state = _red_gated_run(tmp_path)
    _finish_call(wf, state, {"summary": "done", "stale_gate": "pytest tests/unit"})
    _finish_call(wf, state, {"summary": "done", "stale_gate": "pytest tests/fast"})
    stopped = _loop_steer.operator_boundary(wf, _conversation.Conversation(), 4, state)
    assert stopped is not None and stopped.reason == "steer_abort"
    assert stopped.stale_gate == "pytest tests/fast"
