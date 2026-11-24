# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`sessions diff`'s dirty-worktree note reads the checkout through the `vcs` package.

Its own `git status` carried the `-c` overrides alone: a nested repository's clean filter
ran on the host, and `core.worktree` chose the directory it counted.
"""

from __future__ import annotations

import pathlib
import subprocess

from agent6.ui.cli import sessions_cmds


def _git(root: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "agent6/run", str(root)], check=True)
    _git(root, "config", "user.email", "x@y.z")
    _git(root, "config", "user.name", "x")
    (root / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_dirty_worktree_note_runs_no_program_a_repository_names(tmp_path: pathlib.Path) -> None:
    """The note's `git status` ran a nested repository's clean filter, and would an fsmonitor."""
    root = _repo(tmp_path / "r")
    marker = tmp_path / "ran"
    program = tmp_path / "program"
    program.write_text(f'#!/bin/sh\necho ran >> {marker}\ncat "$@"\n', encoding="utf-8")
    program.chmod(0o755)
    nested = _repo(root / "vendor")
    (nested / ".gitattributes").write_text("* filter=x\n", encoding="utf-8")
    _git(nested, "add", "-A")
    _git(nested, "commit", "-qm", "attributes")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "vendor")
    _git(nested, "config", "filter.x.clean", str(program))
    _git(root, "config", "core.fsmonitor", str(program))
    (nested / "f.txt").write_text("edited\n", encoding="utf-8")
    (root / "f.txt").write_text("v2\n", encoding="utf-8")
    (root / "new.txt").write_text("new\n", encoding="utf-8")

    note = sessions_cmds._dirty_worktree_note(root, "agent6/run")
    assert not marker.exists(), "a program the repository names ran on the host"
    assert "2 files modified" in note
    assert sessions_cmds._dirty_worktree_note(root, "agent6/other") == ""


def test_dirty_worktree_note_counts_the_checkout_whatever_core_worktree_names(
    tmp_path: pathlib.Path,
) -> None:
    """The note counted the files of the directory `core.worktree` named."""
    root = _repo(tmp_path / "r")
    outside = tmp_path / "outside"
    outside.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (outside / name).write_text("another directory's file\n", encoding="utf-8")
    _git(root, "config", "core.worktree", str(outside))
    (root / "f.txt").write_text("v2\n", encoding="utf-8")

    assert "1 file modified" in sessions_cmds._dirty_worktree_note(root, "agent6/run")


def test_dirty_worktree_note_is_empty_outside_a_repository(tmp_path: pathlib.Path) -> None:
    assert sessions_cmds._dirty_worktree_note(tmp_path, "agent6/run") == ""
