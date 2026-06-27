# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the per-run repo-map prior in `_context.py`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.git_ops import tracked_files
from agent6.workflows._context import (
    _build_repo_map,  # pyright: ignore[reportPrivateUsage]
)


def _repo_map(root: Path) -> str:
    return _build_repo_map(tracked_files(root))


def _init_repo(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)


def test_repo_map_empty_outside_git(tmp_path: Path) -> None:
    assert _repo_map(tmp_path) == ""


def test_repo_map_lists_directories_with_files(tmp_path: Path) -> None:
    _init_repo(
        tmp_path,
        {
            "README.md": "x",
            "src/a.py": "x",
            "src/b.py": "x",
            "tests/test_a.py": "x",
        },
    )
    out = _repo_map(tmp_path)
    assert "./" in out
    assert "src/" in out
    assert "tests/" in out
    assert "a.py" in out
    assert "test_a.py" in out


def test_repo_map_truncates_long_file_lists(tmp_path: Path) -> None:
    files = {f"pkg/m{i}.py": "x" for i in range(20)}
    _init_repo(tmp_path, files)
    out = _repo_map(tmp_path)
    assert "pkg/" in out
    # Only 6 names shown, rest counted.
    assert "+14 more" in out


@pytest.mark.parametrize("count", [80])
def test_repo_map_caps_total_rows(tmp_path: Path, count: int) -> None:
    files = {f"d{i:03d}/x.py": "x" for i in range(count)}
    _init_repo(tmp_path, files)
    out = _repo_map(tmp_path)
    rows = out.splitlines()
    # 60-line cap (LINES) + 1 trailing "more directories" summary line.
    assert len(rows) <= 61
    assert "more directories" in rows[-1]
