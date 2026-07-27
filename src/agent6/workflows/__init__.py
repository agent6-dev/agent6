# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Workflow package: built-in deterministic state machines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent6.config import Config
from agent6.memory import MemoryEntry, MemoryStoreError
from agent6.memory import list_entries as memory_list_entries
from agent6.skills import ResolvedSkills
from agent6.tools.dispatch import ToolDispatcher
from agent6.workflows._context import load_repo_summary
from agent6.workflows._prompt_blocks import (
    MEMORIES_MAX_CHARS,
    MEMORY_ENTRY_MAX_CHARS,
    build_system_prompt,
)
from agent6.workflows.review import CodeReviewError, code_review

__all__ = [
    "MEMORIES_MAX_CHARS",
    "MEMORY_ENTRY_MAX_CHARS",
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
    ToolDispatcher so the ``<repo-priors>`` block is FULLY enriched (repo map +
    AGENTS.md + recent commits + hot symbols + co-change + symbol outline) -- the
    same view the run loop sees, so prompt show matches reality.

    Recorded memories and installed skills are loaded on the loop's own rules
    (no memories in machine/agent modes, skills in run mode only): omitting them
    printed "(none recorded yet)" for an operator checking what future runs
    would actually receive. *state_dir* is the per-repo state dir the memories
    live under, injected by the caller exactly as the loop's is."""
    dispatcher = (
        ToolDispatcher(root=root, config=config) if config.prompt.structural_priors else None
    )
    repo = load_repo_summary(root, dispatcher=dispatcher)
    return build_system_prompt(
        config=config,
        repo=repo,
        mode=mode,
        memories=_active_memories(state_dir, mode),
        skills=_installed_skills(root, config, mode),
    )


def _active_memories(
    state_dir: Path | None, mode: Literal["run", "plan", "ask", "machine", "agent"]
) -> tuple[MemoryEntry, ...]:
    """The loop's ``_load_memories`` rules: none without a state dir or in
    machine/agent modes, and an unreadable store degrades to none (memory is
    context, not correctness)."""
    if state_dir is None or mode in ("machine", "agent"):
        return ()
    try:
        entries = memory_list_entries(state_dir)
    except (MemoryStoreError, OSError):
        return ()
    return tuple(e for e in entries if e.is_active)


def _installed_skills(
    root: Path, config: Config, mode: Literal["run", "plan", "ask", "machine", "agent"]
) -> ResolvedSkills | None:
    """The loop's ``_load_skills`` rules: run mode only, and nothing installed
    renders no block."""
    if mode != "run":
        return None
    resolved = ToolDispatcher(root=root, config=config).resolved_skills()
    return resolved if (resolved.enabled or resolved.always) else None
