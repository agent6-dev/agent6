# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` collects the working-tree diff WITHOUT mutating the index.

The no-base review path registers intent-to-add entries so untracked files show
in `git diff HEAD`; it must undo them so the command stays read-only ("Never
touches the worktree").
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent6.ui.cli.review_cmds import _collect_review_diff  # pyright: ignore[reportPrivateUsage]


def _git(root: Path, *args: str) -> str:
    git = shutil.which("git")
    assert git is not None
    out = subprocess.run([git, *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_review_diff_includes_untracked_and_restores_index(repo: Path) -> None:
    git = shutil.which("git")
    assert git is not None
    (repo / "newfile.py").write_text("y = 2\n", encoding="utf-8")  # untracked
    (repo / "tracked.py").write_text("x = 99\n", encoding="utf-8")  # modified tracked

    proc = _collect_review_diff(git, repo, base="", head="HEAD", paths=())
    assert proc.returncode == 0
    # Both the modification and the new untracked file appear in the review diff.
    assert "tracked.py" in proc.stdout
    assert "newfile.py" in proc.stdout
    assert "y = 2" in proc.stdout

    # ...and the index is left as found: newfile.py is still UNTRACKED ("??"),
    # not a lingering intent-to-add entry.
    status = _git(repo, "status", "--porcelain")
    assert "?? newfile.py" in status
    assert "A  newfile.py" not in status
    assert " A newfile.py" not in status


def test_review_diff_with_base_is_plain_diff(repo: Path) -> None:
    git = shutil.which("git")
    assert git is not None
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")
    proc = _collect_review_diff(git, repo, base="HEAD~1", head="HEAD", paths=())
    assert proc.returncode == 0
    assert "x = 2" in proc.stdout
