# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Gateless wiring: with no verify_command, the verify tool is hidden and the
system prompt swaps the verify block for the no-verify block."""

from __future__ import annotations

from pathlib import Path

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import RepoSummary
from agent6.workflows._prompts import build_system_prompt


def _cfg(*, verify: bool) -> Config:
    data = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Config.model_validate(data)


def _repo(root: Path) -> RepoSummary:
    return RepoSummary(
        root=root,
        branch="main",
        head_sha="0" * 40,
        file_count=0,
        top_level=(),
        agents_md="",
        recent_log="",
    )


def test_verify_tool_hidden_when_command_unset(tmp_path: Path) -> None:
    with_verify = ToolDispatcher(root=tmp_path, config=_cfg(verify=True))
    gateless = ToolDispatcher(root=tmp_path, config=_cfg(verify=False))
    assert "run_verify_command" in with_verify.available_tool_names()
    assert "run_verify_command" not in gateless.available_tool_names()


def test_system_prompt_switches_verify_block(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with_verify = build_system_prompt(config=_cfg(verify=True), repo=repo, mode="run")
    gateless = build_system_prompt(config=_cfg(verify=False), repo=repo, mode="run")
    assert "<verify-command>" in with_verify and "<no-verify-command>" not in with_verify
    assert "<no-verify-command>" in gateless and "<verify-command>" not in gateless
