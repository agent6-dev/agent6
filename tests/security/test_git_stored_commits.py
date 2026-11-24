# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6's own git reads no replace ref and no graft file.

A jailed command that can write `.git` (`partial` `sandbox.isolation`, `sandbox.protect_git =
"off"`, `git.control = "model"`) can add a replace ref or a graft file. Plain git then
answers for one commit with another commit's content, or with other parents.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6.vcs.git import command as git_command
from agent6.vcs.git import read as git_read


def _git(repo: pathlib.Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, input=stdin
    ).stdout.strip()


def _commit(repo: pathlib.Path, text: str, subject: str) -> str:
    (repo / "app.py").write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", subject)
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("rigging", ["replace ref", "graft file"])
def test_a_view_shows_the_stored_commit(tmp_path: pathlib.Path, rigging: str) -> None:
    """A replace ref showed a decoy's diff for the real sha; a graft file showed no diff."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    base = _commit(repo, "print('base')\n", "base")
    real = _commit(repo, "print('base')\nprint('added by the run')\n", "the run's change")
    stored = git_read.commit_diff(repo, real)
    assert "+print('added by the run')" in stored

    if rigging == "replace ref":
        blob = _git(repo, "hash-object", "-w", "--stdin", stdin="print('decoy')\n")
        tree = _git(repo, "mktree", stdin=f"100644 blob {blob}\tapp.py\n")
        decoy = _git(repo, "commit-tree", tree, "-p", base, "-m", "the run's change")
        _git(repo, "replace", real, decoy)
    else:
        # A parent with the commit's own tree: the commit then changes nothing.
        same = _git(repo, "commit-tree", f"{real}^{{tree}}", "-m", "grafted parent")
        (repo / ".git" / "info" / "grafts").write_text(f"{real} {same}\n", encoding="utf-8")
    # The rigging is live: plain git answers with it.
    shown = _git(repo, "show", "--format=", "--no-color", real, "--", ".")
    assert "+print('added by the run')" not in shown, "the rigging changes nothing in plain git"

    assert git_read.commit_diff(repo, real) == stored


def test_the_graft_switch_keeps_git_quiet(tmp_path: pathlib.Path) -> None:
    """A graft file git could open (`/dev/null`) made it print a deprecation hint on stderr.

    A review snapshot's object check reads any stderr as a corrupt object.
    """
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    head = _commit(repo, "print('base')\n", "base")
    assert git_command.run(repo, "rev-list", "--objects", head).stderr == ""
