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
    # The verify command the original run resolved (possibly inferred), so resume
    # reuses it rather than re-inferring (which could flip and diverge from the
    # frozen system prompt's verify/no-verify block). `()` = the run was gateless;
    # `None` = a pre-field snapshot (resume falls back to re-inference).
    verify_command: tuple[str, ...] | None = None
    # Per-run review-panel block counter, so the anti-stall gate-disarm survives
    # resume instead of resetting to 0 (additive; absent in older snapshots = 0).
    review_rejections_total: int = 0
    # Completion-relevant scalars, so the metric / verify-settled stop logic
    # doesn't regress across a resume (all additive; absent in older snapshots
    # = the safe defaults below). We persist a compact metric *summary* (best
    # score + at-ceiling flag) rather than the full list[MetricSample]: that is
    # all `_metric_at_ceiling` and the plateau seed need.
    verify_ever_passed: bool = False
    gateless_ever_committed: bool = False
    metric_best_score: float | None = None
    metric_at_ceiling: bool = False


SNAPSHOT_VERSION = 1


def load_resume_snapshot(path: Path) -> _ResumeSnapshot:
    """Load and validate a resume snapshot. Raises on bad shape."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"snapshot version mismatch at {path}: {version!r} != {SNAPSHOT_VERSION}")
    vc = raw.get("verify_command")  # additive field; absent in older snapshots
    best = raw.get("metric_best_score")  # additive; absent in older snapshots
    return _ResumeSnapshot(
        system=raw["system"],
        messages=raw["messages"],
        tool_calls=int(raw["tool_calls"]),
        next_iteration=int(raw["next_iteration"]),
        root_task_id=raw.get("root_task_id"),
        verify_command=tuple(vc) if isinstance(vc, list) else None,
        review_rejections_total=int(raw.get("review_rejections_total", 0)),
        verify_ever_passed=bool(raw.get("verify_ever_passed", False)),
        gateless_ever_committed=bool(raw.get("gateless_ever_committed", False)),
        metric_best_score=float(best) if isinstance(best, int | float) else None,
        metric_at_ceiling=bool(raw.get("metric_at_ceiling", False)),
    )
