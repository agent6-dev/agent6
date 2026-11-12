# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An unconfined command carries none of agent6's own provider keys.

At `isolation = "none"` a model-chosen command runs as the operator, so the filesystem
is already open to it, but a key named by `[providers.*].api_key_env` lives in the shell
environment and not on that disk. A command's environment is the policy's passthrough
set, built without the operator's shell environment, so the key never reaches it.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from agent6.app.session import setup as session_setup
from agent6.config import Config
from agent6.tools import dispatch

_CFG = {
    "sandbox": {"run_commands": "allow", "isolation": "none"},
    "providers": {
        "anthropic": {
            "backend": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
    },
}


def test_an_unconfined_command_does_not_see_the_provider_key(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    build_dispatcher: Callable[..., dispatch.ToolDispatcher],
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SECRET")
    monkeypatch.setenv("A_HARMLESS_VAR", "keep-me")
    cfg = Config.model_validate(_CFG)
    session_setup.apply_git_ops_policy(cfg)  # what a real run does at startup
    dispatcher = build_dispatcher(root=tmp_path, config=cfg, isolation="none")

    try:
        out = dispatcher.dispatch("run_command", {"argv": ["/usr/bin/env"]}).to_wire()["stdout"]
    finally:
        dispatcher.close()

    assert "sk-ant-SECRET" not in out
    assert "A_HARMLESS_VAR" not in out, "the operator's shell environment is not the command's"
    assert "HOME=" in out, "the passthrough set still resolves"
