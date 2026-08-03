# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 resume` preflight ordering: the snapshot-version refusal must land
BEFORE the egress broker is spawned (like `fork`, which refuses instantly), so a
v1-snapshot resume never spawns a broker + netns or prints the egress preamble.
"""

from __future__ import annotations

import json
import subprocess as sp
from pathlib import Path

import pytest

import agent6.app.resume as resume_mod
from agent6.ui.cli._common import _state_dir  # pyright: ignore[reportPrivateUsage]
from agent6.ui.cli.resume import _cmd_resume  # pyright: ignore[reportPrivateUsage]


def _git_repo(path: Path) -> None:
    sp.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    sp.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    sp.run(["git", "add", "seed.txt"], cwd=path, check=True)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_v1_snapshot_resume_refuses_before_starting_egress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    state_dir = _state_dir(repo)
    run_dir = state_dir / "runs" / "old-run-AAAA11"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "run_id": "old-run-AAAA11", "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    # A pre-format-change (v1) snapshot: load_run_snapshot refuses it.
    (run_dir / "loop_state.json").write_text(json.dumps({"version": 1}), encoding="utf-8")

    def _no_egress_allowed(*_a: object, **_k: object) -> object:
        pytest.fail("maybe_start_egress must not run before the snapshot refusal")

    monkeypatch.setattr(resume_mod, "maybe_start_egress", _no_egress_allowed)

    rc = _cmd_resume(None, "old-run-AAAA11", force=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert "predates a state-format change" in err
    assert "provider-only egress" not in err  # no broker preamble printed


def test_parked_resume_does_not_replay_a_config_selected_profile_as_a_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parked branch is the SECOND profile replay site: it handed the raw
    stamped name to load_effective, where _select_profile treats it as a flag
    that outranks every config layer -- so a parked submission under a
    config-selected profile started under a config its original submission
    never had. The snapshot-resume path already replays via replay_profile."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    run_dir = _state_dir(repo) / "runs" / "parked-AAAA11"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": "parked-AAAA11",
                "mode": "run",
                "user_task": "queued work",
                "parked_task": "queued work",
                "workflow": {"profile": "t", "profile_from_flag": False},
            }
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    def _capture_load_effective(*_a: object, profile: str = "", **_k: object) -> object:
        from agent6.config import ConfigError

        seen.append(profile)
        raise ConfigError("stop before run_task")  # short-circuit the branch

    monkeypatch.setattr(resume_mod, "load_effective", _capture_load_effective)
    rc = _cmd_resume(None, "parked-AAAA11", force=False)
    assert rc == 2
    # A config-selected profile re-resolves from the config files; only a
    # --profile flag is replayed (WorkflowStamp.replay_profile's contract).
    assert seen == [""]
