# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 prompt show` assembles the real system prompt for the current repo."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent6.cli.prompt_cmds import _cmd_prompt_show  # pyright: ignore[reportPrivateUsage]


def _git_repo(tmp_path: Path) -> Path:
    p = tmp_path / "repo"
    p.mkdir()
    (p / "f.py").write_text("x = 1\n", encoding="utf-8")
    (p / "AGENTS.md").write_text("# conventions\n- be terse here\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@e",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@e",
        "PATH": os.environ.get("PATH", ""),
    }
    subprocess.run(["git", "init", "-q"], cwd=p, check=True)
    subprocess.run(["git", "add", "-A"], cwd=p, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=p, env=env, check=True)
    return p


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    monkeypatch.chdir(repo)
    # isolate from the developer's real global config / state
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))


def test_prompt_show_run_mode_injects_agents_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    rc = _cmd_prompt_show(None, mode="run")
    out = capsys.readouterr().out
    assert rc == 0
    # static structural blocks + the per-repo priors block
    assert "<role>" in out and "<repo-priors>" in out
    # the repo's AGENTS.md is injected verbatim into the prompt
    assert "be terse here" in out


def test_prompt_show_plan_mode_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _git_repo(tmp_path)
    _isolate(tmp_path, monkeypatch, repo)
    rc = _cmd_prompt_show(None, mode="plan")
    out = capsys.readouterr().out
    assert rc == 0 and "PLAN mode" in out
