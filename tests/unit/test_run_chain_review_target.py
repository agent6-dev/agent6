# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import pathlib
import subprocess

from agent6.harness import run_chain
from agent6.vcs import git_read


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

    `review_target` returned None, so a `before_finish` panel skipped with `no_target`.
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
