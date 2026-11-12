# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`resume` process exit codes.

docs/cli.md documents a budget-exhausted run as exit 3 (resumable: raise the cap
and `agent6 resume`), a finish over a red verify as exit 4 and an operator stop as
exit 130; everything else completed=False is exit 1, a clean or ungated finish is 0, and so
is an unverified finish when the harness runs no gate of its own.
"""

from __future__ import annotations

import pathlib
from typing import get_args

import pytest

from agent6 import kinds
from agent6.app import finalize
from agent6.harness import loop_snapshot
from agent6.viewmodel.session import status as session_status


def _result(
    *,
    completed: bool,
    reason: loop_snapshot.SessionEndReason,
    verification: kinds.Verification = "not_applicable",
) -> loop_snapshot.SessionResult:
    return loop_snapshot.SessionResult(
        completed=completed,
        reason=reason,
        summary="",
        turns=1,
        tool_calls=1,
        verification=verification,
    )


def test_exit_code_success_is_zero() -> None:
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session"), harness_gate=True
        )
        == 0
    )


def test_exit_code_budget_exhausted_is_three() -> None:
    # The documented "raise the cap and resume" signal.
    assert (
        finalize.session_exit_code(
            _result(completed=False, reason="budget_exhausted"), harness_gate=True
        )
        == 3
    )


def test_exit_code_other_failures_are_one() -> None:
    for reason in ("provider_error", "max_turns", "went_quiet", "no_progress", "crashed"):
        assert (
            finalize.session_exit_code(_result(completed=False, reason=reason), harness_gate=True)
            == 1
        )


def test_exit_code_operator_stop_is_130() -> None:
    """An operator stop exited 1, the code a crash and a provider error share.

    A script could not tell `agent6 stop` from a broken run.
    """
    for reason in ("steer_abort", "steer_exit", "interrupted"):
        assert (
            finalize.session_exit_code(_result(completed=False, reason=reason), harness_gate=True)
            == 130
        )
    # The stop at the commit hook is a completed end, and the operator's stop whatever the gate.
    for verification in get_args(kinds.Verification):
        stopped = _result(completed=True, reason="interactive_stop", verification=verification)
        assert finalize.session_exit_code(stopped, stranded=True, harness_gate=True) == 130
    # One owner: exactly the ends the listing words "stopped" exit 130.
    for reason in get_args(loop_snapshot.SessionEndReason):
        word, _detail = session_status.status_word(
            finished=True, verification="unverified", stop_reason=reason
        )
        code = finalize.session_exit_code(
            _result(completed=False, reason=reason), harness_gate=True
        )
        assert (code == 130) == (word == "stopped"), reason


def test_exit_code_finish_over_a_red_verify_is_four() -> None:
    """`completed` means the agent stopped deliberately, not that the work verified.

    A finish_session over a red or stale gate exited 0 and read as success to every script;
    its own code is distinct from a broken run (1).
    """
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session", verification="failed"),
            harness_gate=True,
        )
        == 4
    )
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="settled", verification="failed"), harness_gate=True
        )
        == 4
    )


def test_exit_code_stale_gate_finish_is_four() -> None:
    """A stale-gate finish exits 4, as a red finish does, whatever its headline reads."""
    for verification in ("failed", "unverified"):
        stale = _result(completed=True, reason="gate_stale", verification=verification)
        assert finalize.session_exit_code(stale, harness_gate=True) == 4


def test_exit_code_verified_finish_is_zero() -> None:
    # Green, and gateless (nothing to verify) -- both are exit 0.
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session", verification="passed"),
            harness_gate=True,
        )
        == 0
    )
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="settled", verification="not_applicable"),
            harness_gate=True,
        )
        == 0
    )


@pytest.mark.parametrize("reason", ["verify_settled", "silent_finish", "metric_plateau"])
@pytest.mark.parametrize(
    ("verification", "code"),
    [("passed", 0), ("not_applicable", 0), ("failed", 4), ("unverified", 4)],
)
def test_harness_end_labels_do_not_change_exit_codes(
    reason: loop_snapshot.SessionEndReason, verification: kinds.Verification, code: int
) -> None:
    """A harness end keeps exit 0 for green or absent gates and 4 for not-green gates."""
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason=reason, verification=verification), harness_gate=True
        )
        == code
    )


def test_exit_code_unverified_finish_is_four() -> None:
    """A gated finish nothing observed exits 4, like a red one.

    No verify ran this execution, or edits landed after the last green; exiting 0 would let a
    worker pass by never running the gate.
    """
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session", verification="unverified"),
            harness_gate=True,
        )
        == 4
    )


def test_auto_merge_needs_a_model_finish_and_a_vouched_for_tree() -> None:
    """Auto merge requires finish_session and a green or absent gate.

    A red or unverified finish stays on its branch; run and resume share the one predicate.
    """
    assert finalize.auto_merge_eligible(
        _result(completed=True, reason="finish_session", verification="passed")
    )
    assert finalize.auto_merge_eligible(
        _result(completed=True, reason="finish_session", verification="not_applicable")
    )
    for bad in ("failed", "unverified"):
        assert not finalize.auto_merge_eligible(
            _result(completed=True, reason="finish_session", verification=bad)  # pyright: ignore[reportArgumentType]
        )
    assert not finalize.auto_merge_eligible(
        _result(completed=False, reason="max_turns", verification="passed")
    )
    # A `run --parallel` run lands its best-ranked child by the same gate rule.
    assert finalize.auto_merge_eligible(
        _result(completed=True, reason="children_ranked", verification="passed")
    )
    assert not finalize.auto_merge_eligible(
        _result(completed=True, reason="children_ranked", verification="unverified")
    )


@pytest.mark.parametrize("reason", ["verify_settled", "settled", "silent_finish", "metric_plateau"])
@pytest.mark.parametrize("verification", ["passed", "not_applicable"])
def test_auto_merge_rejects_a_harness_end(
    reason: loop_snapshot.SessionEndReason, verification: kinds.Verification
) -> None:
    """A harness-ended run must not auto-merge even with a green or absent gate."""
    assert not finalize.auto_merge_eligible(
        _result(completed=True, reason=reason, verification=verification)
    )


def test_exit_code_stranded_edits_are_five() -> None:
    """A green finish whose promised branch never materialized exits 5, not 0.

    The edits sit uncommitted, so 0 would tell a script the deliverable landed. A red gate
    outranks 5; an unstranded finish stays 0.
    """
    ok = _result(completed=True, reason="finish_session", verification="passed")
    assert finalize.session_exit_code(ok, stranded=True, harness_gate=True) == 5
    assert finalize.session_exit_code(ok, stranded=False, harness_gate=True) == 0
    red = _result(completed=True, reason="finish_session", verification="failed")
    assert finalize.session_exit_code(red, stranded=True, harness_gate=True) == 4
    broke = _result(completed=False, reason="provider_error")
    assert finalize.session_exit_code(broke, stranded=True, harness_gate=True) == 1


def test_stranded_edits_reads_git_reality(tmp_path: pathlib.Path) -> None:
    """The predicate is true exactly when the promised branch is missing and the tree is dirty.

    A clean tree and an existing branch are both False.
    """
    import subprocess

    from agent6.sessions import layout as sessions_layout

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        check=True,
    )
    import json

    layout = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="r1")
    layout.session_dir.mkdir(parents=True)
    (layout.session_dir / "manifest.json").write_text(
        json.dumps(
            {"mode": "run", "user_task": "t", "run_branch": "agent6/r1", "base_branch": "master"}
        ),
        encoding="utf-8",
    )
    result = _result(completed=True, reason="finish_session", verification="passed")
    import os

    old = pathlib.Path.cwd()
    os.chdir(repo)
    try:
        assert finalize.stranded_edits(result, layout, repo) is False  # clean tree
        (repo / "a.txt").write_text("changed", encoding="utf-8")
        assert finalize.stranded_edits(result, layout, repo) is True  # dirty + branch missing
        subprocess.run(["git", "-C", str(repo), "branch", "agent6/r1"], check=True)
        assert finalize.stranded_edits(result, layout, repo) is False  # branch exists
    finally:
        os.chdir(old)


def test_stranded_edits_reads_the_run_record_not_its_branch(tmp_path: pathlib.Path) -> None:
    """The stranded predicate reads the commit record, not the branch name.

    Keyed on the branch name, `branch_per_run = false` exited 0 with no warning over a commit
    that never landed, and `commit_per_turn = false` exited 5 over a commit nothing attempted.
    It reads `commits_ref` (the branch, else the chain ref) and the manifest's commit stamp; a
    run with no chain to commit to (a plan, `[git].control = "model"`) never strands.
    """
    import json
    import os
    import subprocess

    from agent6.sessions import layout as sessions_layout
    from agent6.vcs.git import read as git_read

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"],
        check=True,
    )
    result = _result(completed=True, reason="finish_session", verification="passed")

    def layout_for(session_id: str, manifest: dict[str, object]) -> sessions_layout.SessionLayout:
        layout = sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id=session_id)
        layout.session_dir.mkdir(parents=True)
        (layout.session_dir / "manifest.json").write_text(
            json.dumps({"mode": "run", **manifest}), encoding="utf-8"
        )
        return layout

    old = pathlib.Path.cwd()
    os.chdir(repo)
    try:
        (repo / "a.txt").write_text("changed", encoding="utf-8")
        branchless = layout_for("b1", {"session_id": "b1", "user_task": "t", "run_branch": None})
        assert (
            finalize.stranded_edits(result, branchless, repo) is True
        )  # dirty, no chain: nothing landed
        subprocess.run(
            ["git", "-C", str(repo), "update-ref", git_read.chain_ref_for("b1"), "HEAD"], check=True
        )
        assert (
            finalize.stranded_edits(result, branchless, repo) is False
        )  # the chain holds the record
        never_commits = layout_for(
            "n1",
            {
                "session_id": "n1",
                "user_task": "t",
                "run_branch": "agent6/n1",
                "policy": {"commit_per_turn": False},
            },
        )
        assert (
            finalize.stranded_edits(result, never_commits, repo) is False
        )  # nothing commits by design
        branch_lost = layout_for(
            "c1", {"session_id": "c1", "user_task": "t", "run_branch": "agent6/c1"}
        )
        assert finalize.stranded_edits(result, branch_lost, repo) is True  # no branch, no chain
        subprocess.run(
            ["git", "-C", str(repo), "update-ref", git_read.chain_ref_for("c1"), "HEAD"], check=True
        )
        assert (
            finalize.stranded_edits(result, branch_lost, repo) is False
        )  # the chain holds the record
        for design in ({"mode": "plan"}, {"mode": "run", "git_control": "model"}):
            layout = layout_for(
                f"d-{design.get('git_control', design['mode'])}",
                {"session_id": "d", "user_task": "t", "run_branch": None, **design},
            )
            assert finalize.stranded_edits(result, layout, repo) is False  # no chain to commit to
    finally:
        os.chdir(old)


@pytest.mark.parametrize("reason", ["finish_session", "silent_finish", "settled"])
def test_an_unverified_finish_exits_four_only_under_a_harness_gate(
    reason: loop_snapshot.SessionEndReason,
) -> None:
    """Non-zero is for something wrong: with the gate off, nothing judged the tree."""
    unverified = _result(completed=True, reason=reason, verification="unverified")
    assert finalize.session_exit_code(unverified, harness_gate=True) == 4
    assert finalize.session_exit_code(unverified, harness_gate=False) == 0


def test_a_red_finish_exits_four_whatever_the_gate_policy() -> None:
    red = _result(completed=True, reason="finish_session", verification="failed")
    assert finalize.session_exit_code(red, harness_gate=False) == 4
