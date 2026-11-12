# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Harness-free lane bookkeeping behind `/parallel` dispatch (`harness/parallel_dispatch`)."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from agent6 import directive, errors, lane_records
from agent6.app import parallel, preflight
from agent6.harness import parallel_dispatch
from agent6.vcs import git_ops, git_read


def _res(
    *, ok: bool, error: str = "", branch: str = "agent6/lane-1", has_commits: bool = True
) -> lane_records.LaneResult:
    spec = lane_records.LaneSpec(
        lane=1,
        session_id="lane-1",
        workdir=pathlib.Path("/nowhere"),
        settings=None,
    )
    return lane_records.LaneResult(
        spec=spec,
        session_dir=pathlib.Path("/nowhere"),
        branch=branch,
        ok=ok,
        error=error,
        imported_ref=branch if has_commits else "",
        has_commits=has_commits,
    )


def test_segment_lanes_expands_counts_and_models() -> None:
    three = parallel_dispatch.segment_lanes(directive.Segment(task="t", spec="3"), limit=4)
    assert [lt.model for lt in three] == [None, None, None]
    lanes = parallel_dispatch.segment_lanes(directive.Segment(task="t", spec="m1,m2"), limit=4)
    assert [lt.model for lt in lanes] == ["m1", "m2"]
    assert all(lt.task == "t" for lt in lanes)
    with pytest.raises(directive.DirectiveError):
        parallel_dispatch.segment_lanes(directive.Segment(task="t", spec="0"), limit=4)


def test_join_lane_result_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """join_lane_result never raises.

    A failed lane, a conflicted merge and a GitError each reduce to a non-"joined" LaneJoin instead
    of aborting the run.
    """
    kw: dict[str, Any] = {
        "ref": "refs/agent6/run",
        "fallback_parent": None,
        "identity": None,
        "also_branch": None,
    }
    assert parallel_dispatch.join_lane_result(
        pathlib.Path("/r"), _res(ok=False, error="died"), **kw
    ) == lane_records.LaneJoin("lane-1", "agent6/lane-1", "failed", "", "died")

    def _conflict(*_a: Any, **_k: Any) -> str | None:
        return None

    monkeypatch.setattr(git_ops, "chain_merge", _conflict)
    assert (
        parallel_dispatch.join_lane_result(pathlib.Path("/r"), _res(ok=True), **kw).status
        == "conflict"
    )

    def _boom(*_a: Any, **_k: Any) -> str | None:
        raise errors.GitError("fetch failed")

    monkeypatch.setattr(git_ops, "chain_merge", _boom)
    j = parallel_dispatch.join_lane_result(pathlib.Path("/r"), _res(ok=True), **kw)
    assert (j.status, j.detail) == ("failed", "fetch failed")

    def _clean(*_a: Any, **_k: Any) -> str | None:
        return "a" * 40

    monkeypatch.setattr(git_ops, "chain_merge", _clean)
    j = parallel_dispatch.join_lane_result(pathlib.Path("/r"), _res(ok=True), **kw)
    assert (j.status, j.sha) == ("joined", "a" * 40)


def test_a_lane_without_commits_joins_as_no_changes_without_merging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lane with a result and no commit beyond its base joins as `no_changes`.

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
    j = parallel_dispatch.join_lane_result(
        pathlib.Path("/r"), _res(ok=True, has_commits=False), **kw
    )
    assert j == lane_records.LaneJoin("lane-1", "agent6/lane-1", "no_changes", "", "")
    assert merges == []
    # A lane with commits merges the ref the import landed, not the branch name it was given.
    res = lane_records.LaneResult(
        spec=_res(ok=True).spec,
        session_dir=pathlib.Path("/nowhere"),
        branch="agent6/lane-1",
        ok=True,
        error="",
        imported_ref="refs/agent6/lane-1",
        imported_tip="c" * 40,
        has_commits=True,
    )
    assert parallel_dispatch.join_lane_result(pathlib.Path("/r"), res, **kw).status == "joined"
    assert merges == ["refs/agent6/lane-1"]


def _join(status: Any, session_id: str = "lane-1", sha: str = "") -> lane_records.LaneJoin:
    return lane_records.LaneJoin(
        session_id, f"agent6/{session_id}", status, sha, "boom" if status == "failed" else ""
    )


def test_segment_stamp_reduces_lanes() -> None:
    # Single lane keeps the old shape: passed with the join sha, or failed.
    assert parallel_dispatch.segment_stamp([_join("joined", sha="abc123def4567")]) == (
        "passed",
        "lane-1 joined at abc123def456",
        "abc123def4567",
    )
    status, note, sha = parallel_dispatch.segment_stamp([_join("failed")])
    assert (status, sha) == ("failed", "")
    assert "lane-1 failed: boom" in note
    # Multi-lane: any join passes, recording the LAST joined sha; the note
    # names every lane. All-conflict fails (NodeStatus has no "blocked").
    status, note, sha = parallel_dispatch.segment_stamp(
        [_join("joined", "a", sha="1111111111111"), _join("joined", "b", sha="2222222222222")]
    )
    assert (status, sha) == ("passed", "2222222222222")
    assert "a joined at" in note and "b joined at" in note
    status, note, _sha = parallel_dispatch.segment_stamp(
        [_join("conflict"), _join("conflict", "b")]
    )
    assert status == "failed"
    assert "conflicted; merge manually" in note
    # A segment whose lanes all ended with no changes passed: there was nothing to merge.
    assert parallel_dispatch.segment_stamp([_join("no_changes")]) == (
        "passed",
        "lane-1 made no changes",
        "",
    )
    # Mixed: the sha is the last joined lane's; a no-change lane adds none.
    status, note, sha = parallel_dispatch.segment_stamp(
        [_join("joined", "a", sha="1111111111111"), _join("no_changes", "b")]
    )
    assert (status, sha) == ("passed", "1111111111111")
    assert "b made no changes" in note


def test_summary_text_names_every_outcome() -> None:
    text = parallel_dispatch.summary_text(
        "p1",
        [
            _join("joined", "a", sha="abc123def4567"),
            _join("conflict", "b"),
            _join("failed", "c"),
            _join("no_changes", "d"),
        ],
    )
    assert "group p1 complete (4 lane(s))" in text
    assert "a (agent6/a): joined at abc123def456" in text
    assert "d (agent6/d): finished with no changes to join." in text
    assert "CONFLICT" in text and "git merge agent6/b" in text
    assert "FAILED -- boom; nothing joined." in text
    assert text.endswith("Review what landed and continue.")


def test_lane_note_wordings() -> None:
    assert (
        parallel_dispatch.lane_note(_join("joined", sha="abc123def4567"))
        == "lane-1 joined at abc123def456"
    )
    assert parallel_dispatch.lane_note(_join("conflict")) == "lane-1 conflicted; merge manually"
    assert parallel_dispatch.lane_note(_join("failed")) == "lane-1 failed: boom"
    assert parallel_dispatch.lane_note(_join("no_changes")) == "lane-1 made no changes"


def test_segment_lanes_carry_the_operator_pins_out_of_band() -> None:
    """Segment lanes carry the operator pins out of band.

    `/pin` tells the worker an instruction "stays binding for the rest of the run", and a lane's
    branch is merged back into the coordinator's, so a lane that never saw the pin could violate a
    standing instruction and have that work land. A pin folded into the task text becomes the lane's
    manifest user_task, so every listing and the judge's CandidateBrief lead with "PINNED operator
    instructions (verbatim):" instead of the work. Pins ride the LaneTask; the task text stays the
    task.
    """
    lanes = parallel_dispatch.segment_lanes(
        directive.Segment(spec="2", task="refactor the model layer"),
        ["never touch schema files"],
        limit=4,
    )
    assert len(lanes) == 2
    for lane in lanes:
        assert lane.task == "refactor the model layer"
        assert lane.pins == ("never touch schema files",)

    # No pins: nothing rides along.
    plain = parallel_dispatch.segment_lanes(directive.Segment(spec="", task="do it"), limit=4)
    assert plain[0].task == "do it" and plain[0].pins == ()


def test_a_dirty_origin_fans_out_under_stash_and_include(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dirty origin fans out under stash and include.

    One knob decides for the run path and `--parallel` alike; reading `require_clean_worktree` alone
    refuses the documented stash-without-asking setup and names a key the run path ignores.
    """
    from agent6.config import Config
    from agent6.ui.cli import parallel as cli_parallel

    def _dirty(_origin: pathlib.Path) -> list[str]:
        return ["a.py"]

    def _clear(_cfg: Config) -> None:
        return None

    monkeypatch.setattr(git_read, "modified_paths", _dirty)
    monkeypatch.setattr(preflight, "budget_preflight", _clear)
    monkeypatch.setattr(cli_parallel, "_parallel_approval_refusal", _clear)
    fanned: list[str] = []

    def _fake_run_parallel(task: str, *_a: object, **_kw: object) -> int:
        fanned.append(task)
        return 0

    monkeypatch.setattr(parallel, "run_parallel", _fake_run_parallel)
    ask = Config.model_validate({"git": {"dirty_tree": "ask"}})
    assert (
        cli_parallel.dispatch_parallel(ask, "t", "2", explicit_leaves=frozenset(), cwd=tmp_path)
        == 2
    )
    assert "dirty_tree" in capsys.readouterr().err and fanned == []
    for choice in ("stash", "include"):
        cfg = Config.model_validate({"git": {"dirty_tree": choice}})
        assert (
            cli_parallel.dispatch_parallel(cfg, "t", "2", explicit_leaves=frozenset(), cwd=tmp_path)
            == 0
        ), choice
    assert fanned == ["t", "t"]


def test_a_fan_out_prints_the_sandbox_degrades_a_run_prints(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lanes run detached, so the coordinator prints the unsandboxed warning once, up front."""
    from agent6.config import Config
    from agent6.ui.cli import parallel as cli_parallel

    def _clear(_cfg: Config) -> None:
        return None

    def _clean(_origin: pathlib.Path) -> list[str]:
        return []

    def _fake_run_parallel(_task: str, *_a: object, **_kw: object) -> int:
        return 0

    monkeypatch.setattr(preflight, "budget_preflight", _clear)
    monkeypatch.setattr(cli_parallel, "_parallel_approval_refusal", _clear)
    monkeypatch.setattr(git_read, "modified_paths", _clean)
    monkeypatch.setattr(parallel, "run_parallel", _fake_run_parallel)
    cfg = Config.model_validate({"sandbox": {"isolation": "none"}})
    assert (
        cli_parallel.dispatch_parallel(cfg, "t", "2", explicit_leaves=frozenset(), cwd=tmp_path)
        == 0
    )
    assert "running UNSANDBOXED (sandbox.isolation = none)" in capsys.readouterr().err


def test_summary_hands_a_conflicted_lane_to_the_operator() -> None:
    """The summary hands a conflicted lane to the operator.

    Git is agent6's in a run and `.git` is read-only in the jail, so the line names the operator and
    tells the model to continue rather than run `git merge` itself.
    """
    text = parallel_dispatch.summary_text("g", [_join("conflict")])
    assert "for the operator (`git merge " in text and "continue without it" in text
    assert "resolve" not in text


def test_the_coordinator_takes_sigterm_as_a_ctrl_c_stop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIGTERM reaches the coordinator as a KeyboardInterrupt, which journals the stop.

    `timeout`, CI and the kill `agent6 stop` falls back to send it; the handler is put back after.
    """
    import os
    import signal
    import time

    from agent6.config import Config
    from agent6.ui.cli import parallel as cli_parallel

    def _clean(_origin: pathlib.Path) -> list[str]:
        return []

    def _clear(_cfg: Config) -> None:
        return None

    monkeypatch.setattr(git_read, "modified_paths", _clean)
    monkeypatch.setattr(preflight, "budget_preflight", _clear)
    monkeypatch.setattr(cli_parallel, "_parallel_approval_refusal", _clear)

    def _killed_run_parallel(*_a: object, **_kw: object) -> int:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(2.0)
        return 0

    monkeypatch.setattr(parallel, "run_parallel", _killed_run_parallel)
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        cli_parallel.dispatch_parallel(
            Config.model_validate({}), "t", "2", explicit_leaves=frozenset(), cwd=tmp_path
        )
    assert signal.getsignal(signal.SIGTERM) is before
