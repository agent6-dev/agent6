# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`init` git pre-flight: nice no-repo error + the init git offer."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent6.cli import main
from agent6.cli.init_cmds import _offer_git_setup  # pyright: ignore[reportPrivateUsage]
from agent6.cli.run import (
    _require_git_repo,  # pyright: ignore[reportPrivateUsage]
    _warn_if_headless_ask,  # pyright: ignore[reportPrivateUsage]
)
from agent6.config import Config
from agent6.git_ops import init_repo, is_git_repo


def test_warn_if_headless_ask(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ask = cast(Config, SimpleNamespace(sandbox=SimpleNamespace(run_commands="ask")))
    # Headless (no TTY, no TUI) + ask -> warn (run_command would be auto-denied).
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    _warn_if_headless_ask(ask, tui_enabled=False)
    assert "headless" in capsys.readouterr().err
    # No warning when a TUI is up, or stdin is a TTY, or run_commands != ask.
    _warn_if_headless_ask(ask, tui_enabled=True)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    _warn_if_headless_ask(ask, tui_enabled=False)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    yes = cast(Config, SimpleNamespace(sandbox=SimpleNamespace(run_commands="yes")))
    _warn_if_headless_ask(yes, tui_enabled=False)
    assert capsys.readouterr().err == ""


def test_run_surfaces_git_wall_before_provider_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Non-git scratch dir with no provider configured (conftest isolates config):
    # the not-a-git-repo error surfaces first, not after the provider/key walls.
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "do a thing", "--no-tui"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a git repository" in err
    assert "No providers configured" not in err  # git wall comes first now


def test_require_git_repo_errors_outside_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _require_git_repo(tmp_path) is False
    err = capsys.readouterr().err
    assert "not a git repository" in err
    assert "agent6 init" in err  # points at the guided fix


def test_require_git_repo_ok_inside_repo(tmp_path: Path) -> None:
    init_repo(tmp_path)
    assert _require_git_repo(tmp_path) is True


def test_offer_git_setup_noninteractive_just_notes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _offer_git_setup(tmp_path, (tmp_path / "AGENTS.md",), interactive=False)
    out = capsys.readouterr().out
    assert "not a git repository" in out
    assert is_git_repo(tmp_path) is False  # did NOT create a repo non-interactively


def test_offer_git_setup_interactive_inits_and_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _yes(_prompt: str) -> str:
        return "y"

    monkeypatch.setattr("builtins.input", _yes)
    # GIT_* env supplies the commit identity without touching global config.
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(k, "Test")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(k, "t@t.t")
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    # The per-repo config lives OUT of the workspace. Passing such a path must
    # not crash _offer_git_setup (it filters to paths under the repo) and it is
    # never committed.
    out_of_repo_cfg = tmp_path.parent / "a6-state" / "config.toml"
    out_of_repo_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_of_repo_cfg.write_text("# cfg\n", encoding="utf-8")

    _offer_git_setup(
        tmp_path,
        (out_of_repo_cfg, tmp_path / "AGENTS.md", tmp_path / ".gitignore"),
        interactive=True,
    )

    assert is_git_repo(tmp_path) is True
    tracked = subprocess.run(
        ["git", "-C", str(tmp_path), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert "AGENTS.md" in tracked
    assert ".gitignore" in tracked
    assert "config.toml" not in tracked  # out-of-repo config is never committed
