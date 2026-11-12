from __future__ import annotations

import pathlib

import pytest

from agent6.vcs.git import files as git_files
from agent6.vcs.git import read as git_read
from tests.unit import test_git_history


@pytest.mark.parametrize("source", ["gitignore", "info", "global"])
def test_ignored_files_from_every_exclude_source_need_opt_in(
    tmp_path: pathlib.Path, source: str
) -> None:
    """Git's ignore rules, including matching directories, determine opt-in rows and reads."""
    test_git_history.build_history(tmp_path)
    if source == "gitignore":
        excludes = tmp_path / ".gitignore"
    elif source == "info":
        excludes = tmp_path / ".git" / "info" / "exclude"
    else:
        excludes = tmp_path / ".git" / "global-excludes"
        test_git_history._git(tmp_path, "config", "core.excludesFile", str(excludes))
    excludes.write_text("WORK_QUEUE.md\nplans/\n", encoding="utf-8")
    (tmp_path / "WORK_QUEUE.md").write_text("# Queue\n", encoding="utf-8")
    (tmp_path / "plans" / "nested").mkdir(parents=True)
    (tmp_path / "plans" / "nested" / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "plans" / "link.md").symlink_to("../../outside")
    hidden = {"WORK_QUEUE.md", "plans/nested/plan.md", "plans/link.md"}
    assert not hidden.intersection(f.path for f in git_files.worktree_tree_files(tmp_path))
    assert ("!!", "plans/") in git_read.porcelain_entries(tmp_path, ignored=True)
    rows = {f.path: f for f in git_files.worktree_tree_files(tmp_path, ignored=True)}
    assert all(rows[name].state == "ignored" for name in hidden)
    assert rows["plans/link.md"].kind == "symlink"
    assert rows["WORK_QUEUE.md"] == git_files.TreeFile("WORK_QUEUE.md", 8, "file", "ignored")
    assert rows["b.txt"].state == "tracked"
    assert git_files.worktree_blob(tmp_path, "WORK_QUEUE.md", max_bytes=100) is None
    assert git_files.worktree_blob(
        tmp_path, "WORK_QUEUE.md", max_bytes=100, ignored=True
    ) == git_files.Blob(8, b"# Queue\n")
    assert git_files.worktree_blob(
        tmp_path, "plans/nested/plan.md", max_bytes=100, ignored=True
    ) == git_files.Blob(7, b"# Plan\n")
    for name in ("plans/link.md", ".git/config", "../outside", "plans"):
        assert git_files.worktree_blob(tmp_path, name, max_bytes=100, ignored=True) is None
