# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Harness-free child bookkeeping behind `/parallel` dispatch (`harness/parallel_dispatch`)."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest

from agent6 import child_records, directive, errors
from agent6.app import children
from agent6.harness import parallel_dispatch, parallel_join
from agent6.vcs.git import ops as git_ops


def _res(
    *, ok: bool, error: str = "", branch: str = "agent6/child-1", has_commits: bool = True
) -> child_records.ChildResult:
    spec = child_records.ChildSpec(
        number=1,
        session_id="child-1",
        worktree=pathlib.Path("/nowhere"),
        settings=None,
    )
    return child_records.ChildResult(
        spec=spec,
        session_dir=pathlib.Path("/nowhere"),
        branch=branch,
        ok=ok,
        error=error,
        imported_ref=branch if has_commits else "",
        has_commits=has_commits,
    )


def test_segment_children_expands_counts_and_models() -> None:
    three = parallel_dispatch.segment_children(directive.Segment(task="t", spec="3"), limit=4)
    assert [lt.model for lt in three] == [None, None, None]
    children = parallel_dispatch.segment_children(
        directive.Segment(task="t", spec="m1,m2"), limit=4
    )
    assert [lt.model for lt in children] == ["m1", "m2"]
    assert all(lt.task == "t" for lt in children)
    with pytest.raises(directive.DirectiveError):
        parallel_dispatch.segment_children(directive.Segment(task="t", spec="0"), limit=4)


def test_join_child_result_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """join_child_result never raises.

    A failed child, a conflicted merge and a GitError each become a non-"joined" ChildJoin instead
    of aborting the run.
    """
    kw: dict[str, Any] = {
        "ref": "refs/agent6/run",
        "fallback_parent": None,
        "identity": None,
        "also_branch": None,
    }
    assert parallel_join.join_child_result(
        pathlib.Path("/r"), _res(ok=False, error="died"), **kw
    ) == child_records.ChildJoin("child-1", "agent6/child-1", "failed", "", "died")

    def _conflict(*_a: Any, **_k: Any) -> str | None:
        return None

    monkeypatch.setattr(git_ops, "chain_merge", _conflict)
    assert (
        parallel_join.join_child_result(pathlib.Path("/r"), _res(ok=True), **kw).status
        == "conflict"
    )

    def _boom(*_a: Any, **_k: Any) -> str | None:
        raise errors.GitError("fetch failed")

    monkeypatch.setattr(git_ops, "chain_merge", _boom)
    j = parallel_join.join_child_result(pathlib.Path("/r"), _res(ok=True), **kw)
    assert (j.status, j.detail) == ("failed", "fetch failed")

    def _clean(*_a: Any, **_k: Any) -> str | None:
        return "a" * 40

    monkeypatch.setattr(git_ops, "chain_merge", _clean)
    j = parallel_join.join_child_result(pathlib.Path("/r"), _res(ok=True), **kw)
    assert (j.status, j.sha) == ("joined", "a" * 40)


def test_a_child_without_commits_joins_as_no_changes_without_merging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child with a result and no commit beyond its base joins as `no_changes`.

    Nothing is merged: no `chain_merge` call, no sha.
    """
    merges: list[str] = []

    def _merge(_root: pathlib.Path, branch: str, *_a: Any, **_k: Any) -> str | None:
        merges.append(branch)
        return "b" * 40

    monkeypatch.setattr(git_ops, "chain_merge", _merge)
    kw: dict[str, Any] = {
        "ref": "refs/agent6/run",
        "fallback_parent": None,
        "identity": None,
        "also_branch": None,
    }
    j = parallel_join.join_child_result(pathlib.Path("/r"), _res(ok=True, has_commits=False), **kw)
    assert j == child_records.ChildJoin("child-1", "agent6/child-1", "no_changes", "", "")
    assert merges == []
    # A child with commits merges the ref the import merged, not the branch name it was given.
    res = child_records.ChildResult(
        spec=_res(ok=True).spec,
        session_dir=pathlib.Path("/nowhere"),
        branch="agent6/child-1",
        ok=True,
        error="",
        imported_ref="refs/agent6/child-1",
        imported_tip="c" * 40,
        has_commits=True,
    )
    assert parallel_join.join_child_result(pathlib.Path("/r"), res, **kw).status == "joined"
    assert merges == ["refs/agent6/child-1"]


def _join(status: Any, session_id: str = "child-1", sha: str = "") -> child_records.ChildJoin:
    return child_records.ChildJoin(
        session_id, f"agent6/{session_id}", status, sha, "boom" if status == "failed" else ""
    )


def test_segment_stamp_reduces_children() -> None:
    # Single child keeps the old shape: passed with the join sha, or failed.
    assert parallel_join.segment_stamp([_join("joined", sha="abc123def4567")]) == (
        "passed",
        "child-1 joined at abc123def456",
        "abc123def4567",
    )
    status, note, sha = parallel_join.segment_stamp([_join("failed")])
    assert (status, sha) == ("failed", "")
    assert "child-1 failed: boom" in note
    # Multi-child: any join passes, recording the LAST joined sha; the note
    # names every child. All-conflict fails (NodeStatus has no "blocked").
    status, note, sha = parallel_join.segment_stamp(
        [_join("joined", "a", sha="1111111111111"), _join("joined", "b", sha="2222222222222")]
    )
    assert (status, sha) == ("passed", "2222222222222")
    assert "a joined at" in note and "b joined at" in note
    status, note, _sha = parallel_join.segment_stamp([_join("conflict"), _join("conflict", "b")])
    assert status == "failed"
    assert "conflicted; merge manually" in note
    # A segment whose children all ended with no changes passed: there was nothing to merge.
    assert parallel_join.segment_stamp([_join("no_changes")]) == (
        "passed",
        "child-1 made no changes",
        "",
    )
    # Mixed: the sha is the last joined child's; a no-change child adds none.
    status, note, sha = parallel_join.segment_stamp(
        [_join("joined", "a", sha="1111111111111"), _join("no_changes", "b")]
    )
    assert (status, sha) == ("passed", "1111111111111")
    assert "b made no changes" in note


def test_summary_text_names_every_outcome() -> None:
    text = parallel_join.summary_text(
        "p1",
        [
            _join("joined", "a", sha="abc123def4567"),
            _join("conflict", "b"),
            _join("failed", "c"),
            _join("no_changes", "d"),
        ],
    )
    assert "group p1 complete (4 child(ren))" in text
    assert "a (agent6/a): joined at abc123def456" in text
    assert "d (agent6/d): finished with no changes to join." in text
    assert "CONFLICT" in text and "git merge agent6/b" in text
    assert "FAILED -- boom; nothing joined." in text
    assert text.endswith("Review what merged and continue.")


def test_child_note_wordings() -> None:
    assert (
        parallel_join.child_note(_join("joined", sha="abc123def4567"))
        == "child-1 joined at abc123def456"
    )
    assert parallel_join.child_note(_join("conflict")) == "child-1 conflicted; merge manually"
    assert parallel_join.child_note(_join("failed")) == "child-1 failed: boom"
    assert parallel_join.child_note(_join("no_changes")) == "child-1 made no changes"


def test_segment_children_carry_the_operator_pins_out_of_band() -> None:
    """Segment children carry the operator pins out of band.

    `/pin` tells the worker an instruction "stays binding for the rest of the run", and a child's
    branch is merged back into the parent's, so a child that never saw the pin could violate a
    standing instruction and have that work merge. A pin folded into the task text is the child's
    manifest user_task, so every listing and the judge's CandidateBrief lead with "PINNED operator
    instructions (verbatim):" instead of the work. Pins ride the ChildTask; the task text stays the
    task.
    """
    children = parallel_dispatch.segment_children(
        directive.Segment(spec="2", task="refactor the model layer"),
        ["never touch schema files"],
        limit=4,
    )
    assert len(children) == 2
    for child in children:
        assert child.task == "refactor the model layer"
        assert child.pins == ("never touch schema files",)

    # No pins: nothing rides along.
    plain = parallel_dispatch.segment_children(directive.Segment(spec="", task="do it"), limit=4)
    assert plain[0].task == "do it" and plain[0].pins == ()


def test_the_end_notice_gives_a_childs_finish_summary_and_the_merge_gives_its_join() -> None:
    """What a child said reaches its parent when the child ends; how it joined, when it merges.

    The parent read only "finished with no changes to join." for a child sent to find
    something out, and went through the child's transcript for the answer.
    """
    said = "The suite passes as it is.\n3 tests, exit 0."
    ended = dataclasses.replace(_res(ok=True, has_commits=False), report=said)
    assert parallel_join.end_notice(ended).splitlines() == [
        "[parallel] child child-1 finished.",
        "summary: The suite passes as it is.",
        "  3 tests, exit 0.",
    ]
    stopped = dataclasses.replace(
        _res(ok=True), stopped_by="stopped at the run's budget", commits=2
    )
    assert parallel_join.end_notice(stopped) == (
        "[parallel] child child-1 stopped at the run's budget, not merged;"
        " agent6/child-1 holds 2 commit(s); conversation at last call: unknown; last ran: unknown."
    )
    assert parallel_join.end_notice(_res(ok=False, error="no result (failed)")) == (
        "[parallel] child child-1 ended without a result: no result (failed)"
    )
    text = parallel_join.summary_text(
        "p1",
        [
            dataclasses.replace(_join("no_changes", "a"), report=said),
            _join("joined", "b", sha="abc123def4567"),
            dataclasses.replace(_join("failed", "c"), report="  "),
        ],
    )
    assert text.splitlines() == [
        "[parallel] group p1 complete (3 child(ren)):",
        "  - a (agent6/a): finished with no changes to join.",
        "  - b (agent6/b): joined at abc123def456",
        "  - c (agent6/c): FAILED -- boom; nothing joined.",
        "Review what merged and continue.",
    ]


def test_started_text_tells_one_child_from_attempts() -> None:
    """The line a dispatch answers with names each child's id, the handle the tools take."""
    assert parallel_join.started_text(["run-p1-l1"]).startswith(
        "[parallel] started child run-p1-l1. "
    )
    assert parallel_join.started_text(["run-p1-l1", "run-p1-l2"]).startswith(
        "[parallel] started 2 attempts at one task: run-p1-l1, run-p1-l2. "
    )


def test_summary_text_says_a_stopped_child_did_not_merge() -> None:
    """A stopped child's line says what stopped it, that nothing merged, and what holds its work.

    A stopped child's commits were merged, its line reading "joined at <sha> (stopped by its
    parent)".
    """
    idle = dataclasses.replace(_join("stopped", "a"), detail="stopped at the run's budget")
    busy = dataclasses.replace(_join("stopped", "b"), detail="stopped by its parent", commits=3)
    text = parallel_join.summary_text("p1", [idle, busy])
    assert text.splitlines() == [
        "[parallel] group p1 complete (2 child(ren)):",
        "  - a (agent6/a): stopped at the run's budget, not merged; the branch holds 0 commit(s);"
        " conversation at last call: unknown; last ran: unknown.",
        "  - b (agent6/b): stopped by its parent, not merged; the branch holds 3 commit(s);"
        " conversation at last call: unknown; last ran: unknown.",
        "Every child stopped: nothing merged.",
    ]
    # A task whose every child stopped did not pass; one that merged a sibling did.
    assert parallel_join.segment_stamp([idle, busy]) == (
        "failed",
        "a stopped at the run's budget; not merged; b stopped by its parent; not merged",
        "",
    )
    merged = _join("joined", "c", sha="abc123def4567")
    assert parallel_join.segment_stamp([busy, merged])[0] == "passed"
    assert parallel_join.summary_text("p1", [busy, merged]).splitlines()[-1] == (
        "Review what merged and continue."
    )
    assert parallel_join.join_words(busy) == (
        "stopped by its parent, not merged; agent6/b holds 3 commit(s);"
        " conversation at last call: unknown; last ran: unknown"
    )


def test_summary_hands_a_conflicted_child_to_the_operator() -> None:
    """The summary hands a conflicted child to the operator.

    Git is agent6's in a run and `.git` is read-only in the jail, so the line names the operator and
    tells the model to continue rather than run `git merge` itself.
    """
    text = parallel_join.summary_text("g", [_join("conflict")])
    assert "(`git merge agent6/child-1` for the operator)" in text
    assert "nothing merged" in text and "continue without it" in text
    assert "resolve" not in text
    named = dataclasses.replace(_join("conflict"), detail="merge conflict in a.py, b.py")
    assert "CONFLICT -- merge conflict in a.py, b.py; nothing merged." in (
        parallel_join.summary_text("g", [named])
    )


def _ended_child(dest: pathlib.Path, *, reason: str, verification: str) -> None:
    dest.mkdir()
    (dest / "manifest.json").write_text(json.dumps({"mode": "run", "session_id": dest.name}))
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {
            "type": "tool.call",
            "name": "finish_session",
            "args_preview": {"summary": "The suite passes: 3 tests, exit 0."},
        },
        {"type": "session.stop", "reason": reason, "verification": verification},
    ]
    (dest / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_children_result_carries_its_finish_summary(tmp_path: pathlib.Path) -> None:
    """A child that finished hands its parent the summary it finished with; another end, none."""
    spec = child_records.ChildSpec(number=1, session_id="done", worktree=tmp_path, settings=None)
    done = tmp_path / "done"
    _ended_child(done, reason="finish_session", verification="not_applicable")
    res = children.recorded_result(spec, "", done, child_records.Imported(done), origin=tmp_path)
    assert res.ok and res.report == "The suite passes: 3 tests, exit 0."

    # A stopped child keeps its result, says what stopped it, and joins as stopped.
    halted = tmp_path / "halted"
    _ended_child(halted, reason="steer_abort", verification="unverified")
    res = children.recorded_result(
        spec,
        "",
        halted,
        child_records.Imported(halted),
        origin=tmp_path,
        stopped_by="stopped at the run's budget",
    )
    assert res.ok and res.stopped_by == "stopped at the run's budget"
    join = parallel_join.stopped_join(res)
    assert (join.status, join.detail, join.commits) == ("stopped", "stopped at the run's budget", 0)

    # A child the operator stopped, or a resume found stopped, reads stopped, not finished.
    res = children.recorded_result(
        spec, "", halted, child_records.Imported(halted), origin=tmp_path
    )
    assert res.ok and res.stopped_by == "stopped"
    assert parallel_join.end_notice(res).splitlines()[0] == (
        "[parallel] child done stopped, not merged; 0 commit(s); ref unknown;"
        " conversation at last call: unknown; last ran: unknown."
    )
    join = parallel_join.stopped_join(res)
    assert parallel_join.summary_text("g", [join]).splitlines()[1:] == [
        "  - done (): stopped, not merged; the branch holds 0 commit(s);"
        " conversation at last call: unknown; last ran: unknown.",
        "Every child stopped: nothing merged.",
    ]

    cut = tmp_path / "cut"
    _ended_child(cut, reason="max_turns", verification="unverified")
    res = children.recorded_result(spec, "", cut, child_records.Imported(cut), origin=tmp_path)
    assert not res.ok and res.report == ""
