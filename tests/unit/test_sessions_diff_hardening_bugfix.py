# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`sessions diff`'s dirty-worktree probes carry the host-RCE hardening flags.

Without them a poisoned `.git/config` core.fsmonitor fired on the host during `git status`.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6 import git_ops
from agent6.ui.cli import sessions_cmds  # pyright: ignore[reportPrivateUsage]


def test_dirty_worktree_note_hardens_its_git_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    flags = list(git_ops.git_hardening_flags(pathlib.Path.cwd()))

    class _Done:
        def __init__(self, stdout: str) -> None:
            self.returncode = 0
            self.stdout = stdout

    def _fake_run(argv: list[str], **_kw: object) -> _Done:
        seen.append(argv)
        # rev-parse -> current branch matches the run branch; status -> one file
        return _Done("agent6/run\n" if "rev-parse" in argv else " M a.py\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    note = sessions_cmds._dirty_worktree_note(pathlib.Path("/repo"), "agent6/run")
    assert "1 file modified" in note
    # `git_hardening_flags` reads the repo's driver names first, so each probe
    # is preceded by that config read (an absolute-path git, not "git").
    probes = [argv for argv in seen if argv[0] == "git"]
    assert len(probes) == 2  # rev-parse + status
    for argv in probes:
        # the hardening flags sit right after "git", before the subcommand
        assert argv[1 : 1 + len(flags)] == flags
