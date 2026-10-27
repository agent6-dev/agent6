# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A `[sandbox].hide_paths` entry inside the repository stays out of every tree the chain writes."""

from __future__ import annotations

import pathlib
import subprocess

from agent6.harness import _chain
from agent6.vcs import git_read


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_the_chain_keeps_hidden_paths_out_of_its_commits_diffs_and_dirty_checks(
    tmp_path: pathlib.Path,
) -> None:
    """A hidden file written mid-run reaches no commit, diff, dirty check or object store."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    base = _git(repo, "rev-parse", "HEAD")
    hidden = git_read.hidden_in_repo(repo, (str(repo / "secrets"), str(repo / ".env")))
    chain = _chain.RunChain(
        repo, ref="refs/agent6/test/head", fallback_parent=base, base_sha=base, hidden=hidden
    )
    (repo / "secrets" / "deep").mkdir(parents=True)
    (repo / "secrets" / "deep" / "key.txt").write_text("hunter2\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    sha = chain.commit("turn 1")
    assert sha
    assert _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines() == ["a.py"]
    for rel in ("secrets/deep/key.txt", ".env"):
        blob = _git(repo, "hash-object", rel)
        missing = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", blob], check=False)
        assert missing.returncode != 0, f"{rel} reached the object store"
    assert chain.is_dirty() is False
    assert chain.dirty_note() == ""
    assert chain.name_status() == (("M", "a.py"),)
    diff = chain.diff_since_base()
    assert "x = 2" in diff
    assert "hunter2" not in diff and "TOKEN" not in diff
    assert chain.excluded == hidden
