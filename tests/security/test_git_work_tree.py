# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6's own git works on the checkout it runs in, whatever `core.worktree` names.

`core.worktree` in a repository's config is the directory git reads and writes in place of
the one holding `.git`. A jailed command writes that config where `.git` is writable
(`partial` `sandbox.isolation`, `protect_git = "off"`, `git.control = "model"`), and a
copied repository brings one. Each test points the key at a directory outside the checkout
and runs the agent6 operations that read or write files.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6 import errors
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    _git(root, "config", "user.email", "x@y.z")
    _git(root, "config", "user.name", "x")
    (root / "pkg").mkdir()
    (root / "pkg" / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def _outside(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return a directory outside the checkout, holding one file."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("another directory's file\n", encoding="utf-8")
    return outside


def test_a_checkpoint_stages_the_checkout_whatever_core_worktree_names(
    tmp_path: pathlib.Path,
) -> None:
    """A per-turn commit recorded the files of the directory `core.worktree` named."""
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD")
    _git(root, "config", "core.worktree", str(_outside(tmp_path)))
    (root / "new.txt").write_text("new\n", encoding="utf-8")

    assert git_read.toplevel(root / "pkg") == root
    assert git_read.untracked_paths(root) == {"new.txt"}
    # From a directory below the root too: the work tree is the checkout, never that directory.
    sha = git_ops.chain_commit(root / "pkg", "turn", ref="refs/agent6/s/head", fallback_parent=base)
    assert sha is not None
    assert git_read.tree_paths(root, sha) == {"new.txt", "pkg/f.txt"}


def test_a_tree_move_writes_the_checkout_whatever_core_worktree_names(
    tmp_path: pathlib.Path,
) -> None:
    """A chain merge's `read-tree -u` wrote and deleted files in the directory the key named."""
    root = _repo(tmp_path / "r")
    older = _git(root, "rev-parse", "HEAD")
    (root / "pkg" / "f.txt").write_text("v2\n", encoding="utf-8")
    (root / "added.txt").write_text("added\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "second")
    newer = _git(root, "rev-parse", "HEAD")
    outside = _outside(tmp_path)
    # The same content as the checkout's, so git takes the file for one it may remove.
    (outside / "added.txt").write_text("added\n", encoding="utf-8")
    _git(root, "config", "core.worktree", str(outside))

    git_ops.sync_worktree(root, newer, older)
    assert (root / "pkg" / "f.txt").read_text(encoding="utf-8") == "v1\n"
    assert not (root / "added.txt").exists()
    assert sorted(p.name for p in outside.iterdir()) == ["added.txt", "secret.txt"]


def test_a_linked_worktree_refuses_a_work_tree_that_is_not_its_own(
    tmp_path: pathlib.Path,
) -> None:
    """A linked worktree's `config.worktree` redirected its git the same way."""
    root = _repo(tmp_path / "r")
    base = _git(root, "rev-parse", "HEAD")
    linked = tmp_path / "linked"
    git_ops.add_worktree(root, linked, base)
    (linked / "new.txt").write_text("new\n", encoding="utf-8")
    sha = git_ops.chain_commit(linked, "turn", ref="refs/agent6/s/head", fallback_parent=base)
    assert sha is not None
    assert git_read.tree_paths(root, sha) == {"new.txt", "pkg/f.txt"}

    _git(root, "config", "extensions.worktreeConfig", "true")
    _git(linked, "config", "--worktree", "core.worktree", str(_outside(tmp_path)))
    with pytest.raises(errors.GitError, match=r"core\.worktree"):
        git_read.untracked_paths(linked)
    with pytest.raises(errors.GitError, match=r"core\.worktree"):
        git_ops.chain_commit(linked, "turn", ref="refs/agent6/s/head", fallback_parent=base)
