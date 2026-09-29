# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A worker may say the gate is wrong instead of reverting correct work."""

from __future__ import annotations

import contextlib
import io

import pytest

from agent6.app import reporter
from agent6.harness import _snapshot
from agent6.tools import results
from agent6.viewmodel import listing


def test_the_reason_reads_as_a_failure_with_its_cause() -> None:
    """A stale gate is a reason, not a new status word.

    `died_without_end`, the compare gates and the TUI colours key off the existing set.
    """
    assert listing.status_word(finished=True, all_passed=False, end_reason="gate_stale") == (
        "failed",
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
        'agent6 config set harness.verify_command \'["uv", "run", "pytest", "tests/unit"]\'' in text
    )


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
