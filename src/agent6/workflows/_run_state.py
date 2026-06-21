# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""How a run ends and how it resumes: the RunResult the workflow returns, the
ResumeError it raises, and the provider-agnostic resume snapshot written before
each LLM call (load here; the loop owns saving it)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final state of a run.

    ``reason`` values:
      finish_run       - agent called the finish_run tool explicitly.
      silent_finish    - agent emitted text but no tool_use (talking).
      went_quiet       - agent emitted neither text nor tool_use.
      budget_exhausted - BudgetTracker raised; partial progress kept.
      provider_error   - ProviderError after retry; loop aborted.
    metric_plateau   - metric run tied prior best after enough samples.
            prompt_revision_failed - revise_prompt failed before the worker loop.
      max_iterations   - hit max_iterations cap without finish.
      steer_abort      - operator typed "abort" at a steering prompt.
    """

    completed: bool
    reason: str
    summary: str
    iterations: int
    tool_calls: int
    finish_payload: dict[str, Any] | None = None


class ResumeError(Exception):
    """Raised when resume cannot proceed (missing/corrupt snapshot)."""


@dataclass(frozen=True, slots=True)
class _ResumeSnapshot:
    """provider-agnostic in-memory snapshot of loop state.

    Written before each LLM call so a crash mid-call can be resumed from
    the same point. Provider-agnostic because the OpenAI provider
    translates anthropic-shaped messages before its transcript sink runs
    - we cannot reuse provider transcripts for cross-provider resume.
    """

    system: str
    messages: list[dict[str, Any]]
    tool_calls: int
    next_iteration: int
    root_task_id: str | None


SNAPSHOT_VERSION = 1


def load_resume_snapshot(path: Path) -> _ResumeSnapshot:
    """Load and validate a resume snapshot. Raises on bad shape."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"snapshot version mismatch at {path}: {version!r} != {SNAPSHOT_VERSION}")
    return _ResumeSnapshot(
        system=raw["system"],
        messages=raw["messages"],
        tool_calls=int(raw["tool_calls"]),
        next_iteration=int(raw["next_iteration"]),
        root_task_id=raw.get("root_task_id"),
    )
