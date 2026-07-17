# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Gateless wiring: with no verify_command, the verify tool is hidden and the
system prompt swaps the verify block for the no-verify block."""

from __future__ import annotations

from pathlib import Path

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import RepoSummary
from agent6.workflows._prompt_blocks import build_system_prompt


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


def test_no_verify_block_wording_matches_the_mode(tmp_path: Path) -> None:
    """The gateless block must name the mode's real terminal tool and must not
    claim auto-commits in the read-only modes: plan finishes via
    `finish_planning`, ask has no terminal tool at all, and neither can edit."""
    repo = _repo(tmp_path)
    cfg = _cfg(verify=False)
    run = build_system_prompt(config=cfg, repo=repo, mode="run")
    plan = build_system_prompt(config=cfg, repo=repo, mode="plan")
    ask = build_system_prompt(config=cfg, repo=repo, mode="ask")

    def block(text: str) -> str:
        start = text.index("<no-verify-command>")
        return text[start : text.index("</no-verify-command>", start)]

    run_block, plan_block, ask_block = block(run), block(plan), block(ask)
    assert "finish_run" in run_block and "commits each editing step" in run_block
    assert "finish_planning" in plan_block
    assert "finish_run" not in plan_block and "commits" not in plan_block
    assert "finish_run" not in ask_block and "finish_planning" not in ask_block
    assert "commits" not in ask_block
    # All three still disarm stray instructions to call the absent verify tool.
    for b in (run_block, plan_block, ask_block):
        assert "Ignore any" in b and "run_verify_command" in b
