# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One live run-mode worker per checkout: the repo.lock + park-and-resume flow.

Two concurrent run-mode workers share one working tree; each auto-commit is a
`git add -A` on whatever HEAD points at, so whichever run's branch was checked
out last received BOTH runs' commits. The repo-scoped flock refuses the second
worker up front, and the refused submission is PARKED (the manifest saves the
verbatim task) so the typed prompt is never dropped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.runs.layout import RunLayout
from agent6.runs.lock import (
    acquire_repo_writer,
    release_single_writer,
    repo_writer_held,
    repo_writer_holder,
)
from agent6.runs.manifest import read_manifest


def test_repo_writer_second_acquire_refused_and_holder_named(tmp_path: Path) -> None:
    fd = acquire_repo_writer(tmp_path, "run-A")
    assert fd is not None
    try:
        assert acquire_repo_writer(tmp_path, "run-B") is None
        assert repo_writer_holder(tmp_path) == "run-A"
        assert repo_writer_held(tmp_path) is True
    finally:
        release_single_writer(fd)
    # Released: the checkout is free again and the probe agrees.
    assert repo_writer_held(tmp_path) is False
    fd2 = acquire_repo_writer(tmp_path, "run-B")
    assert fd2 is not None
    release_single_writer(fd2)


def test_repo_writer_probe_does_not_hold(tmp_path: Path) -> None:
    # The advisory probe must not itself keep the lock (it acquires + releases).
    assert repo_writer_held(tmp_path) is False  # no lock file yet
    fd = acquire_repo_writer(tmp_path, "run-A")
    assert fd is not None
    release_single_writer(fd)
    assert repo_writer_held(tmp_path) is False
    fd2 = acquire_repo_writer(tmp_path, "run-C")
    assert fd2 is not None
    release_single_writer(fd2)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp git repo with isolated state + a minimal runnable global config."""
    gdir = tmp_path / "cfg"
    gdir.mkdir()
    (gdir / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(gdir))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    _init_repo(repo)
    monkeypatch.chdir(repo)
    return repo


def _load_cfg() -> Config:
    from agent6.config.layer import load_effective

    return load_effective(Path.cwd(), None).config


def test_second_run_parks_with_the_verbatim_task(repo: Path) -> None:
    """While a live worker holds the checkout, a second `run` submission is
    refused, but the exact typed prompt is saved as a parked, resumable run —
    with no tree mutation (no stash, no branch cut)."""
    from agent6.app.run import run_task

    state = resolved_state_dir(repo)
    long_task = "fix the thing " + "x" * 5000  # > the 4000-char display cap
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        rc = run_task(
            _load_cfg(),
            long_task,
            frontend=MagicMock(),
            run_id="run-PARKED",
            mode="run",
        )
    finally:
        release_single_writer(holder_fd)
    assert rc == 2
    layout = RunLayout(state_dir=state, run_id="run-PARKED")
    m = read_manifest(layout.run_dir)
    assert m.parked_task == long_task  # verbatim, not the truncated display twin
    assert m.run_branch is None
    # No branch was cut and the tree is untouched.
    branches = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "agent6/run-PARKED"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""
    # The parked dir survives (it is a saved run, not a discardable husk) and
    # the listing tells the truth about it.
    from agent6.viewmodel.listing import summarize_run_dir

    row = summarize_run_dir(layout.run_dir)
    assert row.status == "parked"
    assert "resume" in row.reason


def test_resume_starts_a_parked_run_with_the_saved_task(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 resume <parked-id>` delegates to a fresh run_task with the
    verbatim saved task under the same run id (releasing its own locks first,
    so the fresh start can take them)."""
    from agent6.app import resume as resume_mod
    from agent6.app.manifest import write_run_manifest

    state = resolved_state_dir(repo)
    layout = RunLayout(state_dir=state, run_id="run-PARKED2")
    layout.ensure()
    write_run_manifest(
        layout,
        run_id="run-PARKED2",
        user_task="do the saved thing",
        base_sha="",
        base_branch="main",
        run_branch=None,
        cfg=_load_cfg(),
        mode="run",
        parked_task="do the saved thing",
    )
    called: dict[str, Any] = {}

    def fake_run_task(cfg: Config, task: str, **kw: Any) -> int:
        called["task"] = task
        called["run_id"] = kw.get("run_id")
        called["mode"] = kw.get("mode")
        return 0

    monkeypatch.setattr(resume_mod, "run_task", fake_run_task)
    rc = resume_mod.resume_task(None, "run-PARKED2", frontend=MagicMock(), force=False)
    assert rc == 0
    assert called == {"task": "do the saved thing", "run_id": "run-PARKED2", "mode": "run"}
    # The delegation released the run-dir lock before handing off, so a real
    # run_task can re-acquire it: prove the lock is free.
    from agent6.runs.lock import acquire_single_writer

    fd = acquire_single_writer(layout.run_dir)
    assert fd is not None
    release_single_writer(fd)


def test_resume_refuses_while_another_run_drives_the_checkout(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resuming run B while run A's worker is live in the same checkout must
    refuse (a resumed worker drives the tree exactly like a fresh one)."""
    from agent6.app import resume as resume_mod

    state = resolved_state_dir(repo)
    layout = RunLayout(state_dir=state, run_id="run-B")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": "run-B",
                "mode": "run",
                "base_sha": "",
                "base_branch": "main",
                "run_branch": "agent6/run-B",
                "user_task": "t",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    holder_fd = acquire_repo_writer(state, "run-A")
    try:
        rc = resume_mod.resume_task(None, "run-B", frontend=MagicMock(), force=False)
    finally:
        release_single_writer(holder_fd)
    assert rc == 2
    err = capsys.readouterr().err
    assert "run-A" in err and "checkout" in err


def test_web_new_work_preflight_refuses_while_checkout_busy(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web hub refuses a New Work `run` submission up front (naming the live
    run) instead of spawning a detached run that parks and times out the locate;
    plan submissions are read-only and spawn freely."""
    from agent6.ui.web import actions

    spawned: list[str] = []

    def must_not_spawn(*a: object, **k: object) -> tuple[Path | None, str]:
        raise AssertionError("must not spawn")

    monkeypatch.setattr(actions, "spawn_and_locate", must_not_spawn)
    state = resolved_state_dir(repo)
    holder_fd = acquire_repo_writer(state, "run-LIVE")
    try:
        run_id, err = actions.spawn_new_work(repo, "run", "another task")
    finally:
        release_single_writer(holder_fd)
    assert run_id is None
    assert "run-LIVE" in err and "checkout" in err
    assert spawned == []
