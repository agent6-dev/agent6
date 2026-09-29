# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6's own git ops run on the host and inherit the environment, minus the provider keys.

A credential helper or content driver would otherwise inherit an API key set in the operator's
shell; everything git needs (PATH, HOME) stays.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6 import child_env, git_ops
from agent6.app import _setup
from agent6.config import Config


@pytest.fixture(autouse=True)
def _reset_policy() -> object:  # pyright: ignore[reportUnusedFunction]
    yield
    child_env.set_provider_key_env([])  # module-level state; do not leak across tests


def _captured_git_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> dict[str, str]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    seen: dict[str, dict[str, str]] = {}
    real = subprocess.Popen

    def spy(*args: object, **kwargs: object):
        env = kwargs.get("env")
        if isinstance(env, dict):
            seen["env"] = env
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(git_ops.subprocess, "Popen", spy)
    git_ops.status(tmp_path)
    return seen["env"]


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
    env = _captured_git_env(monkeypatch, tmp_path)
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
    env = _captured_git_env(monkeypatch, tmp_path)
    # Only the configured key name is stripped, not every token-shaped var.
    assert env.get("GIT_SSH_COMMAND") == "ssh -i /home/me/.ssh/id_ed25519"
    assert env.get("SOME_OTHER_TOKEN") == "not-a-configured-key"


def test_no_providers_configured_strips_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Keys living only in secrets.toml never enter the environment; a same-named var stays."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
    _setup.apply_git_ops_policy(Config())
    env = _captured_git_env(monkeypatch, tmp_path)
    assert env.get("ANTHROPIC_API_KEY") == "from-the-shell"
