# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Workflow package: built-in deterministic state machines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher
from agent6.workflows._context import load_repo_summary
from agent6.workflows._prompt_blocks import build_system_prompt
from agent6.workflows.review import CodeReviewError, code_review

__all__ = [
    "CodeReviewError",
    "code_review",
    "system_prompt_for",
]


def system_prompt_for(
    config: Config,
    root: Path,
    mode: Literal["run", "plan", "ask", "machine", "agent"] = "run",
) -> str:
    """Assemble the exact system prompt agent6 would send for *root* + *config*
    in *mode*. Public entry point for `agent6 prompt show` and tooling. Builds a
    ToolDispatcher so the ``<repo-priors>`` block is FULLY enriched (repo map +
    AGENTS.md + recent commits + hot symbols + co-change + symbol outline) -- the
    same view the run loop sees, so prompt show matches reality."""
    dispatcher = (
        ToolDispatcher(root=root, config=config) if config.prompt.structural_priors else None
    )
    repo = load_repo_summary(root, dispatcher=dispatcher)
    return build_system_prompt(config=config, repo=repo, mode=mode)
