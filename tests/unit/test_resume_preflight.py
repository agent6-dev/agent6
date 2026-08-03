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

import agent6.app._session as session_mod
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

    monkeypatch.setattr(session_mod, "maybe_start_egress", _no_egress_allowed)

    rc = _cmd_resume(None, "old-run-AAAA11", force=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert "predates a state-format change" in err
    assert "provider-only egress" not in err  # no broker preamble printed


def test_parked_resume_does_not_replay_a_config_selected_profile_as_a_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parked branch is the SECOND preset replay site: it handed the raw
    stamped name to load_effective, where _select_preset treats it as a flag
    that outranks every config layer -- so a parked submission under a
    config-selected preset started under a config its original submission
    never had. The snapshot-resume path already replays via replay_preset."""
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
                "workflow": {"preset": "t", "preset_from_flag": False},
            }
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    def _capture_load_effective(*_a: object, preset: str = "", **_k: object) -> object:
        from agent6.config import ConfigError

        seen.append(preset)
        raise ConfigError("stop before run_task")  # short-circuit the branch

    monkeypatch.setattr(resume_mod, "load_effective", _capture_load_effective)
    rc = _cmd_resume(None, "parked-AAAA11", force=False)
    assert rc == 2
    # A config-selected preset re-resolves from the config files; only a
    # --preset flag is replayed (WorkflowStamp.replay_preset's contract).
    assert seen == [""]


class _StubGit:
    run_repo_hooks = False


class _StubCfg:
    git = _StubGit()

    def require_runnable(self, _role: str) -> None:
        return None


class _StubLoaded:
    config = _StubCfg()


def _park_manifest(run_dir: Path, *, preset: str, from_flag: bool) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": run_dir.name,
                "mode": "run",
                "user_task": "queued work",
                "parked_task": "queued work",
                "workflow": {"preset": preset, "preset_from_flag": from_flag},
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
    """A parked leg never ran, but its manifest recorded a FLAG-selected preset.
    Restarting it must re-stamp the SAME (name, from_flag) so a later resume/fork
    replays the flag precedence; deriving the stamp from the (empty) resume
    `preset` dropped the from_flag bit and silently downgraded a flag-selected
    preset's blocking veto on the next leg."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(_state_dir(repo) / "runs" / "parked-BBBB22", preset="strict", from_flag=True)
    captured = _stub_start_of_run(resume_mod, monkeypatch)

    assert _cmd_resume(None, "parked-BBBB22", force=False) == 0
    assert captured["preset_stamp"] == ("strict", True)


def test_parked_resume_with_its_own_profile_flag_lets_run_task_derive_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume that DOES pass --preset is a fresh flag choice for this leg, so
    it must NOT pin the manifest's old stamp -- run_task derives from `preset`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(_state_dir(repo) / "runs" / "parked-CCCC33", preset="strict", from_flag=True)
    captured = _stub_start_of_run(resume_mod, monkeypatch)

    assert _cmd_resume(None, "parked-CCCC33", force=False, preset="none") == 0
    assert captured["preset_stamp"] is None
    assert captured["preset"] == "none"


def test_parked_resume_of_a_config_selected_profile_re_derives_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CONFIG-selected preset (from_flag False) re-resolves from the CURRENT
    config on restart, so pinning the manifest's OLD name would show a stale
    preset if the config changed since. Pass preset_stamp=None so run_task
    derives from the re-resolved cfg, like a fresh run -- only a FLAG-selected
    preset (whose blocking veto must survive) is pinned."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _park_manifest(_state_dir(repo) / "runs" / "parked-DDDD44", preset="hardened", from_flag=False)
    captured = _stub_start_of_run(resume_mod, monkeypatch)

    assert _cmd_resume(None, "parked-DDDD44", force=False) == 0
    assert captured["preset_stamp"] is None  # re-derives, not the stale manifest name


class _Stop(Exception):
    """Sentinel: the resume path reached the seam past the assertion point."""


def _plan_run_dir(repo: Path, run_id: str) -> None:
    run_dir = _state_dir(repo) / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "run_id": run_id, "mode": "plan", "user_task": "t"}),
        encoding="utf-8",
    )
    (run_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "version": 2,
                "system": "s",
                "messages": [],
                "tool_calls": 0,
                "next_iteration": 1,
                "root_task_id": None,
                "original_task": "t",
                "verify_command": [],
            }
        ),
        encoding="utf-8",
    )


_PROVIDER_TOML = """
[agent6]
config_version = 1

[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[models.reviewer]
provider = "anthropic"
model = "planner-model"
"""

_PLANNER_ONLY = (
    _PROVIDER_TOML
    + """
[models.planner]
provider = "anthropic"
model = "planner-model"
"""
)

_PLANNER_AND_WORKER = (
    _PLANNER_ONLY
    + """
[models.worker]
provider = "anthropic"
model = "worker-model"
"""
)


def _stub_load_effective(monkeypatch: pytest.MonkeyPatch, toml_body: str, tmp: Path) -> None:
    from agent6.config import load_config

    cfg_path = tmp / "cfg.toml"
    cfg_path.write_text(toml_body, encoding="utf-8")
    cfg = load_config(cfg_path)

    class _Loaded:
        config = cfg

    def _load(*_a: object, **_k: object) -> _Loaded:
        return _Loaded()

    monkeypatch.setattr(resume_mod, "load_effective", _load)


def test_plan_resume_requires_the_planner_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan run resumes under the planner role. Resume hard-coded "worker" at
    its readiness gate, so a planner-only config could START a plan (fresh
    preflight passes require_runnable("planner")) but never resume it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _plan_run_dir(repo, "plan-AAAA11")
    _stub_load_effective(monkeypatch, _PLANNER_ONLY, tmp_path)

    def _stop(*_a: object, **_k: object) -> object:
        raise _Stop()

    # Reaching detect_env means the readiness gate accepted the planner-only
    # config; the old hard-coded require_runnable("worker") returned 2 first.
    monkeypatch.setattr(session_mod, "detect_env", _stop)
    with pytest.raises(_Stop):
        _cmd_resume(None, "plan-AAAA11", force=False)


def test_plan_resume_builds_the_planner_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resumed leg's DRIVING provider is the planner route: with both roles
    configured, the old path silently switched a plan run to the worker model
    on its second leg (and stamped the transcript seat "worker")."""
    import dataclasses

    import agent6.ui.cli.resume as cli_resume_mod
    from agent6.ui.cli.run import run_frontend

    repo = tmp_path / "repo"
    repo.mkdir()
    _git_repo(repo)
    monkeypatch.chdir(repo)
    _plan_run_dir(repo, "plan-BBBB22")
    _stub_load_effective(monkeypatch, _PLANNER_AND_WORKER, tmp_path)

    def _yes(*_a: object) -> bool:
        return True

    def _frontend() -> object:
        return dataclasses.replace(run_frontend(), confirm_unconfined_autorun=_yes)

    def _none(*_a: object, **_k: object) -> None:
        return None

    def _strict(*_a: object, **_k: object) -> str:
        return "strict"

    def _strict_viable(*_a: object, **_k: object) -> tuple[str, None]:
        return ("strict", None)

    def _no_egress(*_a: object, **_k: object) -> tuple[object, None]:
        return (session_mod.EgressGuard(), None)

    monkeypatch.setattr(cli_resume_mod, "run_frontend", _frontend)
    monkeypatch.setattr(session_mod, "detect_env", object)
    monkeypatch.setattr(session_mod, "select_profile", _strict)
    monkeypatch.setattr(session_mod, "warn_sandbox_gaps", _none)
    monkeypatch.setattr(session_mod, "check_network_profile", _none)
    monkeypatch.setattr(session_mod, "resolve_strict_egress_viability", _strict_viable)
    monkeypatch.setattr(resume_mod, "check_provider_keys", _none)
    monkeypatch.setattr(session_mod, "budget_preflight", _none)
    monkeypatch.setattr(resume_mod, "verify_git_identity", _none)
    monkeypatch.setattr(session_mod, "maybe_start_egress", _no_egress)
    monkeypatch.setattr(session_mod, "maybe_apply_agent_landlock", _none)
    monkeypatch.setattr(resume_mod, "ensure_on_run_branch", _none)

    captured: list[str] = []

    def _capture_role(_cfg: object, role: str, **_k: object) -> object:
        captured.append(role)
        raise _Stop()

    monkeypatch.setattr(session_mod, "build_role_provider", _capture_role)
    with pytest.raises(_Stop):
        _cmd_resume(None, "plan-BBBB22", force=False)
    assert captured == ["planner"]
