from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6.vcs import git_ops, git_read, review


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.parametrize("linked", [False, True], ids=["main", "linked"])
@pytest.mark.parametrize("operation", ["review", "diff", "snapshot", "restore", "sync"])
def test_temporary_indexes_do_not_write_shared_indexes(
    tmp_path: pathlib.Path, linked: bool, operation: str
) -> None:
    """A temporary split index must not create sharedindex files in the real git directory."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked").write_text("base\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "tracked").write_text("head\n")
    _git(tmp_path, "commit", "-qam", "head")
    root = tmp_path
    if linked:
        root = tmp_path / "linked"
        _git(tmp_path, "worktree", "add", "--detach", str(root))
    _git(root, "config", "core.splitIndex", "true")
    _git(root, "update-index", "--split-index")
    # One new file over one shared entry triggers a new shared index on Git 2.53.
    (root / "new").write_text("untracked\n")
    gitdir = pathlib.Path(_git(root, "rev-parse", "--absolute-git-dir"))
    before = set(gitdir.glob("sharedindex.*"))
    assert before
    index = (gitdir / "index").read_bytes()

    if operation == "review":
        compared = review.comparison(root, base="", head="HEAD", paths=())
        assert "+untracked" in compared.diff
    elif operation == "diff":
        assert "+untracked" in git_read.diff_since(root, "HEAD")
    elif operation == "snapshot":
        tree = git_read.worktree_tree(root, "HEAD", ())
        assert _git(root, "show", f"{tree}:new") == "untracked"
    elif operation == "restore":
        tree = git_ops.tree_with_paths_from(root, "HEAD", "HEAD~1", ("tracked",))
        assert _git(root, "show", f"{tree}:tracked") == "base"
    else:
        git_ops.sync_worktree(root, "HEAD", "HEAD~1")
        assert (root / "tracked").read_text() == "base\n"

    assert (gitdir / "index").read_bytes() == index
    assert set(gitdir.glob("sharedindex.*")) == before
