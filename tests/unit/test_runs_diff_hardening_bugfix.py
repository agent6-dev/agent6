# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`runs diff`'s dirty-worktree note shelled out `git status`/`git rev-parse`
WITHOUT the host-RCE hardening flags every other git call carries, so a poisoned
`.git/config` core.fsmonitor would fire on the host during `git status`'s index
refresh. Both probes must carry the hardening flags."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.git_ops import git_hardening_flags
from agent6.ui.cli.runs_cmds import _dirty_worktree_note  # pyright: ignore[reportPrivateUsage]


def test_dirty_worktree_note_hardens_its_git_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    flags = list(git_hardening_flags())

    class _Done:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout

    def _fake_run(argv: list[str], **_kw: object) -> _Done:
        seen.append(argv)
        # rev-parse -> current branch matches the run branch; status -> one file
        return _Done("agent6/run\n" if "rev-parse" in argv else " M a.py\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    note = _dirty_worktree_note(Path("/repo"), "agent6/run")
    assert "1 file modified" in note
    assert len(seen) == 2  # rev-parse + status
    for argv in seen:
        assert argv[0] == "git"
        # the hardening flags sit right after "git", before the subcommand
        assert argv[1 : 1 + len(flags)] == flags
