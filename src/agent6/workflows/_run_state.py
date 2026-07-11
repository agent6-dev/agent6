# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""How a run ends and how it resumes: the RunResult the workflow returns, the
ResumeError it raises, and the provider-agnostic resume snapshot written before
each LLM call (load here; the loop owns saving it)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Every way a run can end. The loop constructs all of these except
# "ask_repl_empty" (an interactive ask session that ended before any question
# was asked, ui/cli/_ask.py). Typed so a new outcome must be declared here
# before a RunResult can carry it.
RunReason = Literal[
    "finish_run",
    "finish_planning",
    "silent_finish",
    "went_quiet",
    "budget_exhausted",
    "provider_error",
    "metric_plateau",
    "verify_settled",
    "verify_command_unexecutable",
    "loop_guard_killed",
    "interactive_stop",
    "steer_abort",
    "detached",
    "prompt_revision_failed",
    "max_iterations",
    "ask_repl_empty",
]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final state of a run.

    ``reason`` values (each constructed in loop.py):
      finish_run        - agent called the finish_run tool explicitly.
      finish_planning   - plan-mode agent called the finish_planning tool.
      silent_finish     - agent emitted text but no tool_use (talking).
      went_quiet        - agent emitted neither text nor tool_use.
      budget_exhausted  - BudgetTracker raised; partial progress kept.
      provider_error    - ProviderError after retry; loop aborted.
      metric_plateau    - metric run tied prior best after enough samples.
      verify_settled    - verify passed and the worker stopped making changes.
      verify_command_unexecutable - operator verify/metric command cannot run
                          in the jail; the model cannot fix operator config.
      loop_guard_killed - identical tool call repeated past the kill threshold.
      interactive_stop  - operator chose "stop" at the REPL after_auto_commit hook.
      steer_abort       - operator typed "abort" at a steering prompt.
      detached          - operator chose "detach"; the CLI respawns a detached
                          `resume` to continue the run in the background.
      prompt_revision_failed - revise_prompt failed before the worker loop.
      max_iterations    - hit max_iterations cap without finish.
      ask_repl_empty    - interactive ask session ended with no question asked.
    """

    completed: bool
    reason: RunReason
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


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """One per-turn fork checkpoint: a resume snapshot payload plus the workspace
    HEAD sha and curator graph_version captured at that turn.

    ``payload`` is the exact dict the loop wrote (a superset of ``loop_state.json``);
    ``agent6 fork`` copies it verbatim as the new run's ``loop_state.json`` and seed
    checkpoint. ``head_sha`` is the workspace HEAD at the turn (``""`` if it could
    not be read), used to cut the fork's git branch; ``graph_version`` is the
    curator DAG version (0 = no curator/empty graph)."""

    turn: int
    head_sha: str
    graph_version: int
    payload: dict[str, Any]


def load_checkpoint(path: Path) -> _Checkpoint:
    """Load a per-turn checkpoint. Raises on bad shape (fail loudly)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"checkpoint version mismatch at {path}: {version!r} != {SNAPSHOT_VERSION}"
        )
    return _Checkpoint(
        turn=int(raw["next_iteration"]),
        head_sha=str(raw.get("head_sha", "")),
        graph_version=int(raw.get("graph_version", 0)),
        payload=raw,
    )


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
