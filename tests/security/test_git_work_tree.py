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
from typing import Any

import pytest

from agent6 import errors
from agent6.vcs.git import command as git_command
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import policy as git_policy
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


def _git_processes(trace: pathlib.Path) -> list[str]:
    """Return the git commands a trace file recorded, one per process, and empty the file."""
    lines = trace.read_text(encoding="utf-8").splitlines()
    trace.write_text("", encoding="utf-8")
    return [line.partition("trace: built-in: ")[2] for line in lines if "trace: built-in: " in line]


def test_a_call_runs_no_process_to_learn_a_work_tree_no_repository_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every git call ran one more git process, to learn a work tree the key had not moved."""
    root = _repo(tmp_path / "r")
    outside = _outside(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    # git reads `core.worktree` from the repository's own config alone: the operator's is no key.
    (home / ".gitconfig").write_text(f"[core]\n\tworktree = {outside}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    trace = tmp_path / "trace"
    monkeypatch.setenv("GIT_TRACE", str(trace))

    assert git_read.toplevel(root / "pkg") == root
    assert [argv.split()[1] for argv in _git_processes(trace)] == ["config", "rev-parse"]

    _git(root, "config", "core.worktree", str(outside))
    trace.write_text("", encoding="utf-8")
    assert git_read.toplevel(root / "pkg") == root
    assert [argv.split()[1] for argv in _git_processes(trace)] == [
        "config",
        "rev-parse",  # the checkout's own query
        "rev-parse",
    ]


@pytest.mark.parametrize("spelling", ["[CORE]\n\tWorkTree = {}\n", "[core]\n\tworktree={}\n"])
def test_the_key_is_found_however_the_config_spells_it(
    tmp_path: pathlib.Path, spelling: str
) -> None:
    """Git matches section and key names in any letter case."""
    root = _repo(tmp_path / "r")
    config = root / ".git" / "config"
    config.write_text(
        config.read_text(encoding="utf-8") + spelling.format(_outside(tmp_path)), encoding="utf-8"
    )
    (root / "new.txt").write_text("new\n", encoding="utf-8")

    assert git_read.untracked_paths(root) == {"new.txt"}


@pytest.mark.parametrize("failure", ["timed out", "killed"])
def test_a_config_git_cannot_list_gets_the_checkout_query(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A listing that fails says nothing about the key: the call still names its checkout."""
    root = _repo(tmp_path / "r")
    _git(root, "config", "core.worktree", str(_outside(tmp_path)))
    (root / "new.txt").write_text("new\n", encoding="utf-8")
    run = subprocess.run

    def no_listing(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--show-scope" not in argv:
            return run(argv, **kwargs)
        if failure == "timed out":
            raise subprocess.TimeoutExpired(argv, 10)
        return subprocess.CompletedProcess(argv, -9, "", "")

    monkeypatch.setattr(git_policy.subprocess, "run", no_listing)
    assert git_read.untracked_paths(root) == {"new.txt"}


def test_a_repository_named_apart_from_the_directory_gets_the_checkout_query(
    tmp_path: pathlib.Path,
) -> None:
    """The listing reads the repository named: it says nothing of the one git runs in."""
    root = _repo(tmp_path / "r")
    other = _repo(tmp_path / "other")
    _git(root, "config", "core.worktree", str(_outside(tmp_path)))
    (root / "new.txt").write_text("new\n", encoding="utf-8")

    listed = git_command.run(root, "ls-files", "--others", repo=other).stdout.split()
    assert listed == ["new.txt"]


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
