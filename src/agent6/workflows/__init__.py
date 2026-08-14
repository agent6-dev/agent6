# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Workflow package: built-in deterministic state machines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent6.config import Config
from agent6.memory import index_text as memory_index_text
from agent6.memory import memory_dir
from agent6.sandbox.detect import IsolationUnavailableError, detect, resolve_isolation
from agent6.skills import ResolvedSkills
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import IsolationLevel
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
    *,
    state_dir: Path | None = None,
) -> str:
    """Assemble the exact system prompt agent6 would send for *root* + *config*
    in *mode*. Public entry point for `agent6 prompt show` and tooling. Builds a
    ToolDispatcher so the `<repo-priors>` block is FULLY enriched (repo map +
    AGENTS.md + recent commits + hot symbols + co-change + symbol outline) -- the
    same view the run loop sees, so prompt show matches reality.

    The memory index and installed skills are loaded on
    the loop's own rules (none of the first two in machine/agent modes, skills
    in run mode only): omitting them would print "(none recorded yet)" for
    an operator checking what future runs actually receive. *state_dir* is
    the per-repo state dir those live under, injected by the caller exactly as
    the loop's is."""
    dispatcher = (
        ToolDispatcher(root=root, config=config) if config.prompt.structural_priors else None
    )
    repo = load_repo_summary(root, dispatcher=dispatcher)
    # Machine and agent modes assemble without repo context, so neither half of
    # per-repo recall applies: one gate, not one per block.
    recall = None if mode in ("machine", "agent") else state_dir
    return build_system_prompt(
        config=config,
        repo=repo,
        mode=mode,
        memory_index=memory_index_text(recall) if recall is not None else "",
        memory_dir_path=str(memory_dir(recall)) if recall is not None else "",
        skills=_installed_skills(root, config, mode),
        isolation=_shown_isolation(config),
    )


def _shown_isolation(config: Config) -> IsolationLevel:
    """The level a run here would resolve, for prompt display; an explicit
    setting this host cannot honor shows as "none" rather than refusing a
    read-only preview."""
    try:
        return resolve_isolation(config.sandbox.isolation, detect())
    except IsolationUnavailableError:
        return "none"


def _installed_skills(
    root: Path, config: Config, mode: Literal["run", "plan", "ask", "machine", "agent"]
) -> ResolvedSkills | None:
    """The loop's `_load_skills` rules: run mode only, and nothing installed
    renders no block."""
    if mode != "run":
        return None
    resolved = ToolDispatcher(root=root, config=config).resolved_skills()
    return resolved if (resolved.enabled or resolved.always) else None
