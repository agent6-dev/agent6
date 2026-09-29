# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`resume` process exit codes.

CONFIG.md documents a budget-exhausted run as exit 3 (resumable: raise the cap
and `agent6 resume`) and a finish over a red verify as exit 4; everything else
completed=False is exit 1, a clean or ungated finish is 0.
"""

from __future__ import annotations

import pathlib

from agent6.app import finalize
from agent6.harness import _snapshot


def _result(
    *,
    completed: bool,
    reason: _snapshot.SessionEndReason,
    verified: _snapshot.Verification = "not_applicable",
) -> _snapshot.SessionResult:
    return _snapshot.SessionResult(
        completed=completed,
        reason=reason,
        summary="",
        iterations=1,
        tool_calls=1,
        verified=verified,
    )


def test_exit_code_success_is_zero() -> None:
    assert finalize.session_exit_code(_result(completed=True, reason="finish_session")) == 0


def test_exit_code_budget_exhausted_is_three() -> None:
    # The documented "raise the cap and resume" signal.
    assert finalize.session_exit_code(_result(completed=False, reason="budget_exhausted")) == 3


def test_exit_code_other_failures_are_one() -> None:
    for reason in ("provider_error", "max_iterations", "went_quiet", "steer_abort"):
        assert finalize.session_exit_code(_result(completed=False, reason=reason)) == 1


def test_exit_code_finish_over_a_red_verify_is_four() -> None:
    """`completed` means the agent stopped deliberately, not that the work verified.

    A finish_session over a red or stale gate exited 0 and read as success to every script;
    its own code is distinct from a broken run (1).
    """
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session", verified="failed")
        )
        == 4
    )
    assert (
        finalize.session_exit_code(_result(completed=True, reason="settled", verified="failed"))
        == 4
    )


def test_exit_code_verified_finish_is_zero() -> None:
    # Green, and gateless (nothing to verify) -- both are exit 0.
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session", verified="passed")
        )
        == 0
    )
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="settled", verified="not_applicable")
        )
        == 0
    )


def test_exit_code_unverified_finish_is_four() -> None:
    """A gated finish nothing observed exits 4, like a red one.

    No verify ran this execution, or edits landed after the last green; exiting 0 would let a
    worker pass by never running the gate.
    """
    assert (
        finalize.session_exit_code(
            _result(completed=True, reason="finish_session", verified="unverified")
        )
        == 4
    )


def test_auto_merge_needs_a_vouched_for_tree() -> None:
    """auto_merge lands only work the gate vouched for, or that had no gate.

    A red or unverified finish stays on its branch; run and resume share the one predicate.
    """
    assert finalize.auto_merge_eligible(
        _result(completed=True, reason="finish_session", verified="passed")
    )
    assert finalize.auto_merge_eligible(
        _result(completed=True, reason="settled", verified="not_applicable")
    )
    for bad in ("failed", "unverified"):
        assert not finalize.auto_merge_eligible(
            _result(completed=True, reason="finish_session", verified=bad)  # pyright: ignore[reportArgumentType]
        )
    assert not finalize.auto_merge_eligible(
        _result(completed=False, reason="max_iterations", verified="passed")
    )


def test_exit_code_stranded_edits_are_five() -> None:
    """A green finish whose promised branch never materialized exits 5, not 0.

    The edits sit uncommitted, so 0 would tell a script the deliverable landed. A red gate
    outranks 5; an unstranded finish stays 0.
    """
    ok = _result(completed=True, reason="finish_session", verified="passed")
    assert finalize.session_exit_code(ok, stranded=True) == 5
    assert finalize.session_exit_code(ok, stranded=False) == 0
    red = _result(completed=True, reason="finish_session", verified="failed")
    assert finalize.session_exit_code(red, stranded=True) == 4
    broke = _result(completed=False, reason="provider_error")
    assert finalize.session_exit_code(broke, stranded=True) == 1


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
    result = _result(completed=True, reason="finish_session", verified="passed")
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
    that never landed, and `commit_per_step = false` exited 5 over a commit nothing attempted.
    It reads `commits_ref` (the branch, else the chain ref) and the manifest's commit stamp; a
    run with no chain to commit to (a plan, `[git].control = "model"`) never strands.
    """
    import json
    import os
    import subprocess

    from agent6 import git_ops
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
    result = _result(completed=True, reason="finish_session", verified="passed")

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
            ["git", "-C", str(repo), "update-ref", git_ops.chain_ref_for("b1"), "HEAD"], check=True
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
                "policy": {"commit_per_step": False},
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
            ["git", "-C", str(repo), "update-ref", git_ops.chain_ref_for("c1"), "HEAD"], check=True
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
