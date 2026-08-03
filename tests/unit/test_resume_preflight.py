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


class _StubGit:
    run_repo_hooks = False


class _StubCfg:
    git = _StubGit()

    def require_runnable(self, _role: str) -> None:
        return None


class _StubLoaded:
    config = _StubCfg()


def _park_manifest(run_dir: Path, *, profile: str, from_flag: bool) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": run_dir.name,
                "mode": "run",
                "user_task": "queued work",
                "parked_task": "queued work",
                "workflow": {"profile": profile, "profile_from_flag": from_flag},
            }
        ),
        encoding="utf-8",
    )


def _stub_start_of_run(resume: object, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Let a parked resume reach `run_task`; capture the kwargs it hands over."""

    def _load(*_a: object, **_k: object) -> _StubLoaded:
        return _StubLoaded()

    def _hook_policy(_v: object) -> None:
        return None

    captured: dict[str, object] = {}

    def _capture_run_task(*_a: object, **k: object) -> int:
        captured.update(k)
        return 0

    monkeypatch.setattr(resume, "load_effective", _load)
    monkeypatch.setattr(resume, "set_repo_hook_policy", _hook_policy)
    monkeypatch.setattr(resume, "run_task", _capture_run_task)
    return captured


def test_parked_resume_carries_the_original_flag_selected_profile_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parked leg never ran, but its manifest recorded a FLAG-selected profile.
    Restarting it must re-stamp the SAME (name, from_flag) so a later resume/fork
    replays the flag precedence; deriving the stamp from the (empty) resume
    `profile` dropped the from_flag bit and silently downgraded a flag-selected
    profile's blocking veto on the next leg."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(_state_dir(repo) / "runs" / "parked-BBBB22", profile="strict", from_flag=True)
    captured = _stub_start_of_run(resume_mod, monkeypatch)

    assert _cmd_resume(None, "parked-BBBB22", force=False) == 0
    assert captured["profile_stamp"] == ("strict", True)


def test_parked_resume_with_its_own_profile_flag_lets_run_task_derive_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume that DOES pass --profile is a fresh flag choice for this leg, so
    it must NOT pin the manifest's old stamp -- run_task derives from `profile`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(_state_dir(repo) / "runs" / "parked-CCCC33", profile="strict", from_flag=True)
    captured = _stub_start_of_run(resume_mod, monkeypatch)

    assert _cmd_resume(None, "parked-CCCC33", force=False, profile="none") == 0
    assert captured["profile_stamp"] is None
    assert captured["profile"] == "none"


def test_parked_resume_of_a_config_selected_profile_re_derives_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CONFIG-selected profile (from_flag False) re-resolves from the CURRENT
    config on restart, so pinning the manifest's OLD name would show a stale
    profile if the config changed since. Pass profile_stamp=None so run_task
    derives from the re-resolved cfg, like a fresh run -- only a FLAG-selected
    profile (whose blocking veto must survive) is pinned."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(_state_dir(repo) / "runs" / "parked-DDDD44", profile="hardened", from_flag=False)
    captured = _stub_start_of_run(resume_mod, monkeypatch)

    assert _cmd_resume(None, "parked-DDDD44", force=False) == 0
    assert captured["profile_stamp"] is None  # re-derives, not the stale manifest name
