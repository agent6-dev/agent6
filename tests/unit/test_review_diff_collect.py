# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 review` collects the working-tree diff WITHOUT mutating the index.

The no-base review path registers intent-to-add entries so untracked files show
in `git diff HEAD`; it must undo them so the command stays read-only ("Never
touches the worktree").
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

from agent6.ui.cli import review_cmds  # pyright: ignore[reportPrivateUsage]


def _git(root: pathlib.Path, *args: str) -> str:
    git = shutil.which("git")
    assert git is not None
    out = subprocess.run(
        [git, "-c", "commit.gpgsign=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_review_diff_includes_untracked_and_restores_index(repo: pathlib.Path) -> None:
    git = shutil.which("git")
    assert git is not None
    (repo / "newfile.py").write_text("y = 2\n", encoding="utf-8")  # untracked
    (repo / "tracked.py").write_text("x = 99\n", encoding="utf-8")  # modified tracked

    proc = review_cmds._collect_review_diff(git, repo, base="", head="HEAD", paths=())
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


def test_review_diff_with_base_is_plain_diff(repo: pathlib.Path) -> None:
    git = shutil.which("git")
    assert git is not None
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "second")
    proc = review_cmds._collect_review_diff(git, repo, base="HEAD~1", head="HEAD", paths=())
    assert proc.returncode == 0
    assert "x = 2" in proc.stdout


@pytest.mark.parametrize("base", ["", "base"])
def test_review_refuses_a_range_in_head(base: str, capsys: pytest.CaptureFixture[str]) -> None:
    """A range in --head cannot discard --base or select the working tree."""
    assert review_cmds._cmd_review((), base=base, head="a..b", paths=()) == 2
    assert "--head must be a ref; use --base a..b" in capsys.readouterr().err


@pytest.mark.parametrize("explicit_range", [False, True])
def test_review_branch_after_base_advanced(
    repo: pathlib.Path, explicit_range: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ref pairs exclude base-only changes; explicit two-dot ranges retain them."""
    git = shutil.which("git")
    assert git is not None
    fork = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "branch", "base")
    _git(repo, "checkout", "-qb", "topic")
    (repo / "tracked.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "topic change")
    _git(repo, "checkout", "base")
    for name in ("base_only_a.py", "base_only_b.py"):
        (repo / name).write_text("keep = True\n", encoding="utf-8")
        _git(repo, "add", name)
        _git(repo, "commit", "-qm", name)
    proc = review_cmds._collect_review_diff(
        git, repo, base="base..topic" if explicit_range else "base", head="topic", paths=()
    )
    assert proc.returncode == 0, proc.stderr
    assert "+x = 2" in proc.stdout
    assert ("base_only_a.py" in proc.stdout) is explicit_range
    assert ("base_only_b.py" in proc.stdout) is explicit_range
    monkeypatch.chdir(repo)
    reviewed = review_cmds._reviewed_diff("base..topic" if explicit_range else "base", "topic", ())
    assert not isinstance(reviewed, int)
    start = _git(repo, "rev-parse", "base").strip() if explicit_range else fork
    assert f"(from {start})" in reviewed[3]
    assert ("base_only_a.py" in reviewed[2]) is explicit_range
    assert ("base_only_b.py" in reviewed[2]) is explicit_range
