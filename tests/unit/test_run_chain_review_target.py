# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6 import errors
from agent6.harness import run_chain
from agent6.vcs.git import read as git_read


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_a_model_controlled_run_is_reviewed_from_its_base_to_the_checkout(
    tmp_path: pathlib.Path,
) -> None:
    """Without a chain the model's commits are the record: the target spans base to HEAD.

    `review_target` returned None, so a `finish` panel skipped with `no_target`.
    """
    repo, base = _repo(tmp_path)
    chain = run_chain.RunChain(repo, ref=None, fallback_parent=base, base_sha=base)
    untouched = chain.review_target()
    assert untouched is not None
    assert untouched.base_commit == base and untouched.head_commit == base
    assert git_read.diff_range(repo, untouched.base_tree, untouched.head_tree) == ""

    (repo / "seed.txt").write_text("model\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "model commit")
    head = _git(repo, "rev-parse", "HEAD")
    committed = chain.review_target()
    assert committed is not None
    assert (committed.base_commit, committed.head_commit) == (base, head)
    assert committed.source == "revisions" and committed.snapshot_commit == head
    assert "seed.txt" in git_read.diff_range(repo, committed.base_tree, committed.head_tree)

    (repo / "later.txt").write_text("uncommitted\n", encoding="utf-8")
    edited = chain.review_target()
    assert edited is not None
    assert edited.source == "worktree" and edited.head_commit is None
    assert edited.head_tree != committed.head_tree
    assert "later.txt" in git_read.diff_range(repo, edited.base_tree, edited.head_tree)


def test_unreadable_worktree_never_falls_back_to_a_committed_review(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed worktree read must not certify the older committed bytes."""
    repo, base = _repo(tmp_path)
    (repo / "seed.txt").write_text("unreadable change\n")
    chain = run_chain.RunChain(repo, fallback_parent=base, base_sha=base)

    def unreadable(*_args: object) -> str:
        raise errors.GitError("worktree unreadable")

    monkeypatch.setattr(git_read, "worktree_tree", unreadable)
    with pytest.raises(errors.GitError, match="worktree unreadable"):
        chain.review_target()


def test_unborn_uncommitted_review_has_an_exact_tree(tmp_path: pathlib.Path) -> None:
    """Disabling per-turn commits must not remove the first review target in an unborn repo."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "new.py").write_text("first work\n")
    chain = run_chain.RunChain(tmp_path, ref=git_read.chain_ref_for("new"), per_turn=False)
    target = chain.review_target()
    assert target is not None
    assert target.base_tree == git_read.EMPTY_TREE and target.head_commit is None
    assert git_read.read_file_at(tmp_path, target.head_tree, "new.py") == "first work\n"
