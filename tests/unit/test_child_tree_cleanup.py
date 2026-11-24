"""Child clone deletion needs the child's working and staged files, not just its commits."""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from agent6 import child_records, paths
from agent6.app import worktrees
from agent6.config import Config
from tests.unit import test_children_background as background
from tests.unit import test_children_cli as cli


@pytest.mark.parametrize("sweep", [False, True], ids=["cleanup", "prune"])
@pytest.mark.parametrize(
    ("measured", "nested"),
    [(True, ""), (False, ""), (True, "indexed"), (True, "worktree")],
    ids=["recorded", "unmeasured", "indexed-gitlink", "worktree-gitlink"],
)
def test_cleanup_requires_both_dirty_trees(
    tmp_path: pathlib.Path, sweep: bool, measured: bool, nested: str
) -> None:
    repo = tmp_path / "repo"
    background._init_repo(repo)
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "clones")}}})
    group = worktrees.place(cfg, repo, "parent-p1")
    clone = group / "child-1"
    sid = "parent-p1-l1"
    cli._git(repo, "clone", str(repo), str(clone))
    spec = child_records.ChildSpec(1, sid, clone, None)
    state = paths.state_dir(repo)
    record = state / "sessions" / "runs" / sid
    record.mkdir(parents=True)
    (record / "manifest.json").write_text(
        json.dumps({"session_id": sid, "mode": "run", "git_control": "model"})
    )
    (clone / "draft").write_text("working")
    if nested:
        background._init_repo(clone / "nested")
        (clone / "nested" / "README.md").write_text("uncommitted nested work")
        cli._git(clone, "add", "nested")
    cli._git(clone, "add", "draft")
    work = cli._git(clone, "write-tree")
    if nested == "worktree":
        cli._git(clone, "rm", "--cached", "-f", "nested")
    (clone / "draft").write_text("staged")
    cli._git(clone, "add", "draft")
    index = cli._git(clone, "write-tree")
    (clone / "draft").write_text("working")
    stop = {"type": "session.stop", "reason": "provider_error"}
    if measured:
        stop.update(tree=work, index_tree=index)
    (record / "logs.jsonl").write_text(json.dumps(stop) + "\n")
    if sweep:
        worktrees.sweep_group_clones(repo, cfg)
    else:
        worktrees.cleanup(
            [spec],
            origin=repo,
            record_state=state,
            worktree_root=group,
            base=group.parent,
            launch_dir=tmp_path / "launch",
            cfg=cfg,
        )
    if nested:
        assert (clone / "nested" / "README.md").read_text() == "uncommitted nested work"
    elif measured:
        assert not clone.exists()
        cli._git(repo, "gc", "--prune=now")
        assert cli._git(repo, "show", f"{work}:draft") == "working"
        assert cli._git(repo, "show", f"{index}:draft") == "staged"
    else:
        assert (clone / "draft").read_text() == "working"
        assert cli._git(clone, "show", ":draft") == "staged"


def test_a_sweep_under_a_weaker_level_than_the_child_ran_under_keeps_the_clone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later sweep read a `full` child's clone with the sandbox disabled.

    The sweep took its level from its own environment, so the clone's side of the fetch ran
    on the host over a `.git` the child's commands could write.
    """
    repo = tmp_path / "repo"
    background._init_repo(repo)
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "clones")}}})
    clone = worktrees.place(cfg, repo, "parent-p1") / "child-1"
    sid = "parent-p1-l1"
    cli._git(repo, "clone", str(repo), str(clone))
    record = paths.state_dir(repo) / "sessions" / "runs" / sid
    record.mkdir(parents=True)
    (record / "manifest.json").write_text(
        json.dumps({"session_id": sid, "mode": "run", "policy": {"isolation": "full"}})
    )
    (clone / "draft").write_text("working")
    cli._git(clone, "add", "draft")
    tree = cli._git(clone, "write-tree")
    stop = {"type": "session.stop", "reason": "provider_error", "tree": tree, "index_tree": tree}
    (record / "logs.jsonl").write_text(json.dumps(stop) + "\n")
    monkeypatch.setenv("AGENT6_DISABLE_SANDBOX", "1")

    worktrees.sweep_group_clones(repo, cfg)

    assert (clone / "draft").read_text() == "working"
    arrived = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", tree], capture_output=True, check=False
    )
    assert arrived.returncode != 0, "the sweep fetched the clone's tree on the host"


@pytest.mark.parametrize("sweep", [False, True], ids=["cleanup", "prune"])
@pytest.mark.parametrize("nested", ["unchanged", "moved", "populated"])
def test_cleanup_keeps_a_clone_for_nested_work_only(
    tmp_path: pathlib.Path, sweep: bool, nested: str
) -> None:
    """A gitlink the child's base records, with no nested checkout, kept every clone."""
    lib = tmp_path / "lib"
    background._init_repo(lib)
    repo = tmp_path / "repo"
    background._init_repo(repo)
    link = cli._git(lib, "rev-parse", "HEAD")
    cli._git(repo, "update-index", "--add", "--cacheinfo", f"160000,{link},vendor")
    cli._git(repo, "commit", "-q", "-m", "track a nested repository")
    base = cli._git(repo, "rev-parse", "HEAD")
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "clones")}}})
    group = worktrees.place(cfg, repo, "parent-p1")
    clone = group / "child-1"
    sid = "parent-p1-l1"
    cli._git(repo, "clone", str(repo), str(clone))
    state = paths.state_dir(repo)
    record = state / "sessions" / "runs" / sid
    record.mkdir(parents=True)
    (record / "manifest.json").write_text(
        json.dumps({"session_id": sid, "mode": "run", "git_control": "model", "base_sha": base})
    )
    if nested == "moved":
        cli._git(clone, "update-index", "--cacheinfo", f"160000,{base},vendor")
    if nested == "populated":
        cli._git(lib, "clone", "-q", str(lib), str(clone / "vendor"))
        (clone / "vendor" / "README.md").write_text("uncommitted nested work")
    tree = cli._git(clone, "write-tree")
    stop = {"type": "session.stop", "reason": "finish_session", "tree": tree, "index_tree": tree}
    (record / "logs.jsonl").write_text(json.dumps(stop) + "\n")
    if sweep:
        worktrees.sweep_group_clones(repo, cfg)
    else:
        worktrees.cleanup(
            [child_records.ChildSpec(1, sid, clone, None)],
            origin=repo,
            record_state=state,
            worktree_root=group,
            base=group.parent,
            launch_dir=tmp_path / "launch",
            cfg=cfg,
        )
    assert clone.exists() is (nested != "unchanged")
    if nested == "populated":
        assert (clone / "vendor" / "README.md").read_text() == "uncommitted nested work"


@pytest.mark.parametrize("sweep", [False, True], ids=["cleanup", "prune"])
def test_cleanup_removes_a_reviewer_clone_that_holds_a_command_artifact(
    tmp_path: pathlib.Path, sweep: bool
) -> None:
    """A reviewer's end records no tree, so one unignored file kept its clone."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "clones")}}})
    group = worktrees.place(cfg, repo, "parent-rfinal1")
    clone = group / "child-1"
    sid = "parent-rfinal1-l1"
    cli._git(repo, "clone", str(repo), str(clone))
    state = paths.state_dir(repo)
    record = state / "sessions" / "reviewers" / sid
    record.mkdir(parents=True)
    (record / "manifest.json").write_text(json.dumps({"session_id": sid, "mode": "reviewer"}))
    rows = [
        {"type": "session.start", "mode": "reviewer"},
        {"type": "session.stop", "reason": "finish_session"},
    ]
    (record / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (clone / "coverage.xml").write_text("<coverage/>")
    if sweep:
        worktrees.sweep_group_clones(repo, cfg)
    else:
        worktrees.cleanup(
            [child_records.ChildSpec(1, sid, clone, None, mode="reviewer")],
            origin=repo,
            record_state=state,
            worktree_root=group,
            base=group.parent,
            launch_dir=tmp_path / "launch",
            cfg=cfg,
        )
    assert not clone.exists()
