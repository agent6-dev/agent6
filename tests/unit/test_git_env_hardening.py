# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Host-side Git strips shell overrides and provider keys while preserving authentication."""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from agent6 import child_env
from agent6.app import _setup
from agent6.config import Config
from agent6.ui.cli import _ask, review_cmds, sessions_cmds, skills_cmds
from agent6.vcs import _git_command, git_policy, git_read


@pytest.fixture(autouse=True)
def _reset_policy() -> object:  # pyright: ignore[reportUnusedFunction]
    yield
    child_env.set_provider_key_env([])  # module-level state; do not leak across tests


def _captured_git_envs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    overrides: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, value in (overrides or {}).items():
        monkeypatch.setenv(name, value)
    seen: list[dict[str, str]] = []
    real = subprocess.Popen

    def spy(*args: object, **kwargs: object):
        env = kwargs.get("env")
        seen.append(env if isinstance(env, dict) else dict(os.environ))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_git_command.subprocess, "Popen", spy)
    _git_command.run(tmp_path, "status", "--porcelain", check=False)
    return seen


def test_a_configured_provider_key_is_stripped_from_gits_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-reach-git")
    _setup.apply_git_ops_policy(
        Config.model_validate(
            {
                "providers": {
                    "anthropic": {"api_format": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}
                }
            }
        )
    )
    for env in _captured_git_envs(monkeypatch, tmp_path):
        assert "ANTHROPIC_API_KEY" not in env, "a provider key reached git's environment"
        assert "PATH" in env, "git lost an environment variable it needs"


def test_a_variable_that_is_not_a_provider_key_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -i /home/me/.ssh/id_ed25519")
    monkeypatch.setenv("SOME_OTHER_TOKEN", "not-a-configured-key")
    _setup.apply_git_ops_policy(
        Config.model_validate(
            {
                "providers": {
                    "anthropic": {"api_format": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"}
                }
            }
        )
    )
    for env in _captured_git_envs(monkeypatch, tmp_path):
        assert env.get("GIT_SSH_COMMAND") == "ssh -i /home/me/.ssh/id_ed25519"
        assert env.get("SOME_OTHER_TOKEN") == "not-a-configured-key"


def test_no_providers_configured_strips_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Keys living only in secrets.toml never enter the environment; a same-named var stays."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    _setup.apply_git_ops_policy(Config())
    for env in _captured_git_envs(monkeypatch, tmp_path):
        assert env.get("ANTHROPIC_API_KEY") == "from-the-shell"


@pytest.mark.parametrize(
    "name",
    [
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_VALUE_0",
        "GIT_DIFF_OPTS",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXTERNAL_DIFF",
        "GIT_GLOB_PATHSPECS",
        "GIT_ICASE_PATHSPECS",
        "GIT_INDEX_FILE",
        "GIT_LITERAL_PATHSPECS",
        "GIT_NAMESPACE",
        "GIT_NOGLOB_PATHSPECS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_UNKNOWN_FUTURE_SETTING",
        "GIT_WORK_TREE",
    ],
)
def test_shell_git_overrides_do_not_reach_the_command_or_driver_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, name: str
) -> None:
    """Shell Git overrides must not redirect commands or bypass their driver policy."""
    envs = _captured_git_envs(monkeypatch, tmp_path, {name: "shell-override"})
    assert len(envs) == 2
    for env in envs:
        assert name not in env


@pytest.mark.parametrize("reader", ["runner", "drivers", "ask", "review", "sessions", "skills"])
def test_shell_git_dir_cannot_redirect_host_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, reader: str
) -> None:
    """A shell GIT_DIR must not substitute a foreign repository for the requested checkout."""
    root = tmp_path / "repo"
    foreign = tmp_path / "foreign"
    for directory in (root, foreign):
        subprocess.run(["git", "init", "-qb", "main", str(directory)], check=True)
        (directory / "file").write_text(f"{directory.name}\n")
        subprocess.run(["git", "add", "."], cwd=directory, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=directory, check=True)
    head = _git_command.run(root, "rev-parse", "HEAD").stdout.strip()
    _git_command.run(root, "config", "filter.local.clean", "cat")
    if reader == "ask":
        (root / "file").write_text("changed\n")
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))

    if reader == "runner":
        assert _git_command.run(root, "rev-parse", "HEAD").stdout.strip() == head
    elif reader == "drivers":
        assert "filter.local.clean=" in git_policy.repo_driver_overrides(root)
    elif reader == "ask":
        diff = _ask._git_diff_text(root, "HEAD")[1]
        assert "-repo\n+changed" in diff
    elif reader == "review":
        assert review_cmds._is_checked_out("git", root, head)
    elif reader == "sessions":
        assert sessions_cmds._dirty_worktree_note(root, "main") == ""
    else:
        assert skills_cmds._git_clone(str(root), tmp_path / "clone") == head
    assert git_read.toplevel(root) == root


def test_explicit_git_environment_overrides_still_reach_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """agent6's explicit index wins over the shell without staging into the real index."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "file").write_text("content\n")
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "shell-index"))
    index = tmp_path / "private-index"
    _git_command.run(tmp_path, "add", "file", env_extra={"GIT_INDEX_FILE": str(index)})
    assert index.is_file()
    assert not (tmp_path / "shell-index").exists()
    assert not (tmp_path / ".git/index").exists()
