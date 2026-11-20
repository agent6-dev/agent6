# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A merge onto a checked-out branch writes inside that checkout, and nowhere else.

`sessions merge` runs on the host, in a checkout a jailed command may have written to, and
lands a tree a session's commands may have built. git writes the merged files: it follows
no link a command left in the checkout and checks out no `..` or `.git` tree entry.
"""

from __future__ import annotations

import pathlib
import stat
import subprocess

import pytest

from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


def _git(root: pathlib.Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, input=stdin
    ).stdout.strip()


def _repo(root: pathlib.Path) -> pathlib.Path:
    """Make a repository on `main` holding `d/f.txt`, with a `feature` branch checked out."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "x@y.z")
    _git(root, "config", "user.name", "x")
    (root / "d").mkdir()
    (root / "d" / "f.txt").write_text("old\n", encoding="utf-8")
    (root / "keep.txt").write_text("keep\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    return root


def _commit_and_return_to_main(root: pathlib.Path) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "feature")
    _git(root, "checkout", "-q", "main")


@pytest.mark.parametrize("plant", ["leaf", "parent-add", "parent-change", "parent-delete"])
def test_a_merge_follows_no_link_out_of_the_checkout(tmp_path: pathlib.Path, plant: str) -> None:
    """A link left in the checkout sent a merged file's write, or its delete, to its target."""
    root = _repo(tmp_path / "r")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("old\n", encoding="utf-8")
    if plant == "leaf":
        (root / "new.txt").write_text("merged\n", encoding="utf-8")
    elif plant == "parent-add":
        (root / "sub").mkdir()
        (root / "sub" / "new.txt").write_text("merged\n", encoding="utf-8")
    elif plant == "parent-change":
        (root / "d" / "f.txt").write_text("merged\n", encoding="utf-8")
    else:
        (root / "d" / "f.txt").unlink()
    _commit_and_return_to_main(root)
    # Ignored, so git may replace the link: an untracked one makes it refuse the merge's files.
    (root / ".git" / "info" / "exclude").write_text("new.txt\nsub\nd\n", encoding="utf-8")
    if plant == "leaf":
        (root / "new.txt").symlink_to(outside / "new.txt")  # dangling
    elif plant == "parent-add":
        (root / "sub").symlink_to(outside, target_is_directory=True)
    else:
        (root / "d" / "f.txt").unlink()
        (root / "d").rmdir()
        (root / "d").symlink_to(outside, target_is_directory=True)

    merged = git_ops.plumb_merge(root, "main", "feature", method="merge")

    assert merged.merged_sha == git_read.branch_tip_sha(root, "main")
    assert {p.name: p.read_text(encoding="utf-8") for p in outside.iterdir()} == {
        "f.txt": "old\n"
    }, "the merge wrote or deleted through a link, outside the checkout"


@pytest.mark.parametrize("name", ["..", ".git"])
def test_a_merge_writes_no_path_git_would_not_check_out(tmp_path: pathlib.Path, name: str) -> None:
    """A tree entry named `..` or `.git` was written by path: beside the checkout, or in `.git`."""
    root = _repo(tmp_path / "r")
    _git(root, "checkout", "-q", "main")
    blob = _git(root, "hash-object", "-w", "--stdin", stdin="planted\n")
    inner = _git(root, "mktree", stdin=f"100755 blob {blob}\tplanted\n")
    entries = _git(root, "ls-tree", "HEAD")
    tree = _git(root, "mktree", stdin=f"040000 tree {inner}\t{name}\n{entries}\n")
    commit = _git(root, "commit-tree", tree, "-p", "HEAD", "-m", "planted")
    _git(root, "update-ref", "refs/heads/feature", commit)

    merged = git_ops.plumb_merge(root, "main", "feature", method="ff_only")

    assert not (root / name / "planted").exists(), f"the merge wrote {name}/planted"
    assert merged.left_behind == (f"{name}/planted",)


def test_a_merge_brings_bytes_modes_and_deletions_to_the_checkout(tmp_path: pathlib.Path) -> None:
    """The checkout holds what git checks out: a binary's bytes, an executable, a deletion."""
    root = _repo(tmp_path / "r")
    data = bytes(range(256)) * 4
    (root / "bin.dat").write_bytes(data)
    (root / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "run.sh").chmod(0o755)
    (root / "keep.txt").write_text("changed\n", encoding="utf-8")
    (root / "d" / "f.txt").unlink()
    _commit_and_return_to_main(root)

    merged = git_ops.plumb_merge(root, "main", "feature", method="squash", message="m")

    assert merged.left_behind == ()
    assert (root / "bin.dat").read_bytes() == data
    assert (root / "run.sh").stat().st_mode & stat.S_IXUSR
    assert (root / "keep.txt").read_text(encoding="utf-8") == "changed\n"
    assert not (root / "d").exists()
    assert git_read.status(root).is_clean
