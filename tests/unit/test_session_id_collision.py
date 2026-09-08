# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An explicit --session-id held by another bucket is refused up front.

Ids are one public namespace. Before this refusal, `run --session-id demo`
beside an existing plan `demo` started fine, then every read surface fell
apart: the CLI resolver refused the id as ambiguous, listings showed two
identical rows, and the web silently picked whichever bucket came first.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.paths import state_dir


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gdir = tmp_path / "cfg"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    return repo


def _load_cfg() -> Config:
    from agent6.config.layer import load_effective

    return load_effective(Path.cwd(), None).config


def _strict(*_args: object, **_kwargs: object) -> str:
    return "strict"


def test_run_refuses_an_explicit_id_held_by_another_bucket(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.app.run import run_task

    state = state_dir(repo)
    (state / "sessions" / "plans" / "demo").mkdir(parents=True)

    rc = run_task(_load_cfg(), "do a thing", frontend=MagicMock(), session_id="demo", mode="run")

    assert rc == 2
    err = capsys.readouterr().err
    assert "plans/" in err and "unique across every bucket" in err
    # Nothing was created under runs/: the refusal fired before any state.
    assert not (state / "sessions" / "runs" / "demo").exists()


def test_run_refuses_an_invalid_id_before_sandbox_and_git_preflight(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed identifier is already conclusive, so the run must not ask
    whether to proceed unconfined or report an unrelated host failure first."""
    from agent6.app import run as run_mod

    def _must_not_preflight(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("environment preflight ran for an invalid id")

    monkeypatch.setattr(run_mod, "select_isolation", _must_not_preflight)
    monkeypatch.setattr(run_mod, "git_preflight", _must_not_preflight)

    rc = run_mod.run_task(
        _load_cfg(), "do a thing", frontend=MagicMock(), session_id="bad id", mode="run"
    )

    assert rc == 2
    assert "invalid --session-id 'bad id'" in capsys.readouterr().err


def test_an_existing_finished_id_names_a_runnable_resume_command(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finished run refuses bare resume, so its collision hint must include
    the --steer that gives that run new work."""
    from agent6.app import run as run_mod

    session = state_dir(repo) / "sessions" / "runs" / "done-run"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": "done-run", "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    (session / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_mod, "select_isolation", _strict)

    rc = run_mod.run_task(
        _load_cfg(), "do a thing", frontend=MagicMock(), session_id="done-run", mode="run"
    )

    assert rc == 2
    assert 'agent6 resume done-run --steer "<what to do next>"' in capsys.readouterr().err


def test_a_damaged_existing_id_does_not_name_an_unusable_resume_command(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resume cannot load a malformed manifest, so this refusal must not send
    the operator to a command known to fail on the same record."""
    from agent6.app import run as run_mod

    session = state_dir(repo) / "sessions" / "runs" / "damaged-run"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text("{not json\n", encoding="utf-8")
    monkeypatch.setattr(run_mod, "select_isolation", _strict)

    rc = run_mod.run_task(
        _load_cfg(), "do a thing", frontend=MagicMock(), session_id="damaged-run", mode="run"
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "Choose a different --session-id" in err
    assert "agent6 resume" not in err
