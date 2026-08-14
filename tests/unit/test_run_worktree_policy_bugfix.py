# SPDX-License-Identifier: Apache-2.0
"""Regression test: `_cmd_run` must enforce require_clean_worktree / auto_stash.

Before the fix these config fields were dead: a dirty working tree was neither
refused nor stashed, so the agent's per-step `git add -A` auto-commits swallowed
the user's pre-existing uncommitted work. We assert the preflight now:

  * refuses (rc=2, REFUSING message) on a dirty tree with the default config
    (require_clean_worktree=True, auto_stash=False), BEFORE cutting a branch or
    spawning anything; and
  * auto-stashes the dirty work when auto_stash=True, leaving a clean tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

import agent6.app._session as session_mod
import agent6.app.preflight as preflight_mod
import agent6.app.run as app_run_mod
import agent6.ui.cli.run as run_mod
from agent6.config import (
    Config,
    GitConfig,
    ModelsConfig,
    OpenAIProviderEntry,
    RoleModel,
    SandboxConfig,
)
from agent6.git_ops import status as git_status


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial")


def _runnable_cfg(git_cfg: GitConfig) -> Config:
    return Config(
        providers={
            "openrouter": OpenAIProviderEntry(
                api_format="openai",
                base_url="https://openrouter.ai/api/v1",
            )
        },
        models=ModelsConfig(worker=RoleModel(provider="openrouter", model="kimi")),
        git=git_cfg,
        # Answerable, so the worktree policy under test is what refuses: the
        # `ask` default has nobody to answer it here and is refused first.
        sandbox=SandboxConfig(run_commands="yes"),
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    class _Loaded:
        config = cfg
        # Per-leaf provenance: the preflight uses it to tell a DEFAULT that
        # this host cannot honour (degrade with a warning) from a value the
        # operator wrote down (refuse).
        sources: ClassVar[dict[str, str]] = {}

    def _load_effective(*a: object, **k: object) -> _Loaded:
        return _Loaded()

    def _noop(*a: object, **k: object) -> None:
        return None

    # The fake worker model ("kimi") isn't in the real on-disk model cache, so
    # the new configured-model preflight would refuse it before the dirty-tree
    # logic under test. Bypass it here (its own validation is covered separately).
    def _model_ok(*a: object, **k: object) -> object:
        from agent6.models.validate import ModelValidation

        return ModelValidation(unknown=(), suggestions={}, can_validate=False)

    monkeypatch.setattr(run_mod, "load_effective", _load_effective)
    monkeypatch.setattr(preflight_mod, "apply_git_ops_policy", _noop)
    monkeypatch.setattr(run_mod, "validate_configured_model", _model_ok)
    monkeypatch.setattr(preflight_mod, "verify_git_identity", _noop)


def test_dirty_tree_refused_with_default_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    # Make the tree dirty (the user's uncommitted WIP).
    (repo / "wip.txt").write_text("user work\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    cfg = _runnable_cfg(GitConfig())  # defaults: require_clean_worktree=True
    _patch_common(monkeypatch, cfg)

    # Guard must fire BEFORE the in-process curator is built; make it loud.
    def _loud_curator(*a: object, **k: object) -> object:
        return pytest.fail("built the curator past the guard")

    monkeypatch.setattr(session_mod, "GraphCurator", _loud_curator)

    rc = run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    assert rc == 2
    assert "REFUSING: working tree is not clean" in capsys.readouterr().err
    # The user's WIP is untouched and no run branch was cut.
    assert (repo / "wip.txt").read_text(encoding="utf-8") == "user work\n"
    assert not git_status(repo).is_clean


def test_dirty_tree_auto_stashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "wip.txt").write_text("user work\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    cfg = _runnable_cfg(GitConfig(auto_stash=True))
    _patch_common(monkeypatch, cfg)

    # Stop the run at the very next step after the stash so we don't build
    # providers / spawn a curator: a successful stash leaves a clean tree,
    # which we assert from inside the stub.
    class _Stop(Exception):
        pass

    def _sink_stub(*_a: object, **_k: object) -> object:
        assert git_status(repo).is_clean, "tree should be clean after auto-stash"
        raise _Stop

    monkeypatch.setattr(app_run_mod, "TranscriptSink", _sink_stub)

    with pytest.raises(_Stop):
        run_mod._cmd_run(None, "do a thing")  # pyright: ignore[reportPrivateUsage]

    # The stash entry holds the user's WIP; the working tree is clean.
    stash_list = subprocess.run(
        ["git", "-C", str(repo), "stash", "list"], check=True, capture_output=True, text=True
    ).stdout
    assert "agent6 auto-stash before run" in stash_list
