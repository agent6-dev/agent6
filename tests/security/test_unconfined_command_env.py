# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An unconfined command carries none of agent6's own provider keys.

At `isolation = "none"` a model-chosen command runs as the operator, so the filesystem
is already open to it, but a key named by `[providers.*].api_key_env` lives in the shell
environment and not on that disk. The jailed path builds its env from the policy and
never carries one; this path must not merge the whole `os.environ`.
"""

from __future__ import annotations

import pathlib

import pytest

from agent6 import child_env
from agent6.app import session_setup
from agent6.config import Config
from agent6.tools import dispatch

_CFG = {
    "sandbox": {"run_commands": "allow", "isolation": "none"},
    "providers": {
        "anthropic": {
            "api_format": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
    },
}


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    monkeypatch.setattr(child_env, "_provider_key_env", set())  # module-level state


def test_an_unconfined_command_does_not_see_the_provider_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRET")
    monkeypatch.setenv("A_HARMLESS_VAR", "keep-me")
    cfg = Config.model_validate(_CFG)
    session_setup.apply_git_ops_policy(cfg)  # what a real run does at startup
    dispatcher = dispatch.ToolDispatcher(root=tmp_path, config=cfg, isolation="none")

    try:
        out = dispatcher.dispatch("run_command", {"argv": ["/usr/bin/env"]}).to_wire()["stdout"]
    finally:
        dispatcher.close()

    assert "sk-ant-SECRET" not in out
    assert "A_HARMLESS_VAR=keep-me" in out, "the rest of the operator's env still resolves"
