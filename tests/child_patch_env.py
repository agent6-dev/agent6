# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A child's journal in its parent's state, with the commit still in the clone."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest

from agent6.sessions import journal
from agent6.vcs.git import command as git_command
from tests import child_env


@dataclasses.dataclass(frozen=True, slots=True)
class Child:
    directory: pathlib.Path
    clone: pathlib.Path
    sha: str
    base: str


def recorded_child(repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Child:
    """Leave a journaled patch older than the fold's history and guard viewer Git calls."""
    git = child_env.git
    git(repo, "init", "-q", "-b", "main")
    git(repo, "commit", "--allow-empty", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    clone = repo / "child-1"
    git(repo, "clone", "-q", str(repo), str(clone))
    sid = "parent-p1-l1"
    branch = f"agent6/{sid}"
    git(clone, "checkout", "-qb", branch)
    (clone / "a.txt").write_text("committed text\n")
    git(clone, "add", "a.txt")
    git(clone, "commit", "-qm", "child work")
    sha = git(clone, "rev-parse", "HEAD")
    directory = child_env.create_record(repo, sid, parent="parent")
    manifest = directory / "manifest.json"
    data = json.loads(manifest.read_text())
    manifest.write_text(
        json.dumps({**data, "worktree": str(clone), "run_branch": branch, "base_sha": base})
    )
    with (directory / "logs.jsonl").open("a") as log:
        for turn in range(1, 33):
            commit = sha if turn == 1 else f"{turn:040x}"
            log.write(
                json.dumps({"type": "checkpoint.committed", "turn": turn, "sha": commit}) + "\n"
            )
            if turn < 32:
                row: dict[str, Any] = {
                    "type": "diff.updated",
                    "sha": commit,
                    "patch": f"+preview {turn}\n",
                }
                if turn in (2, 3):
                    row["patch"] = "é" * journal.DiffUpdated.MAX_CHARS
                if turn not in (2, 4):
                    row["truncated"] = turn == 1
                log.write(json.dumps(row) + "\n")

    spawn = git_command.spawn

    def parent_only(
        cwd: pathlib.Path,
        args: tuple[str, ...],
        env_extra: dict[str, str] | None,
        stdin_text: str | None,
        repo: pathlib.Path | None = None,
    ) -> tuple[tuple[str, ...], int, bytes, bytes]:
        assert not cwd.resolve().is_relative_to(clone)
        assert repo is None or not repo.resolve().is_relative_to(clone)
        assert all(str(clone) not in arg for arg in args) and "fetch" not in args
        return spawn(cwd, args, env_extra, stdin_text, repo)

    monkeypatch.setattr(git_command, "spawn", parent_only)
    return Child(directory, clone, sha, base)
