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

from pydantic import BaseModel, ConfigDict, ValidationError

# Every way a run can end. The loop constructs all of these except
# "ask_repl_empty" (an interactive ask session that ended before any question
# was asked, ui/cli/_ask.py). Typed so a new outcome must be declared here
# before a RunResult can carry it.
RunReason = Literal[
    "finish_run",
    "finish_planning",
    "answered",  # ask mode: the final prose IS the answer (a normal, successful end)
    "silent_finish",
    "went_quiet",
    "budget_exhausted",
    "provider_error",
    "metric_plateau",
    "verify_settled",
    "settled",
    "no_progress",
    "tool_error_stuck",
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
      settled           - a GATELESS run's quiet finish: work committed, the
                          worker went idle, and no verify command existed to
                          gate any of it (all_passed stays False).
      no_progress       - the same verify failure survived ten consecutive
                          runs and two harness interventions; stopped to save
                          the remaining budget (resumable).
      tool_error_stuck  - the same tool call failed with the identical error
                          eight times through two interventions; stopped to
                          save the remaining budget (resumable).
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


# Bump on ANY change to the persisted shape below. An in-flight run written by an
# older agent6 then refuses to resume/fork loudly (see load_run_snapshot) rather
# than parsing into a half-populated run. Finished runs never need a snapshot, so
# they keep rendering across the bump.
SNAPSHOT_VERSION = 2


class RunSnapshot(BaseModel):
    """The persisted state of an in-flight run: what ``resume`` re-enters and what
    ``fork`` clones. The loop writes it before each LLM call and again after each
    iteration's tools land, to ``loop_state.json`` and an append-only
    ``checkpoints/<NNNN>.json`` (identical bytes), so a crash resumes from the last
    safe point. Provider-agnostic (anthropic-shaped ``messages``): the OpenAI
    provider translates per call, so its transcript can't seed a cross-provider
    resume.

    On-disk JSON crossing a process + trust boundary, so pydantic owns the shape.
    ``extra="forbid"`` plus a bumped ``version`` mean a snapshot from before a
    state-format change is refused loudly, never coerced into a partial run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = SNAPSHOT_VERSION
    system: str
    messages: list[dict[str, Any]]
    tool_calls: int
    next_iteration: int
    root_task_id: str | None
    # The exact task string the run launched with. Resume re-enters with it
    # verbatim, instead of recovering a truncated copy out of messages[0].
    # RunManifest.user_task is the DISPLAY twin (truncated [:4000]); this is
    # engine state -- never read one where the other is meant.
    original_task: str
    # The verify command the original run resolved (possibly inferred): resume
    # reuses it rather than re-inferring (which could flip and diverge from the
    # frozen system prompt's verify/no-verify block). ``()`` = the run was gateless.
    verify_command: tuple[str, ...]
    # Completion-relevant bookkeeping, so the metric / verify-settled stop logic
    # doesn't regress across a resume. A compact metric *summary* (best score +
    # at-ceiling flag), not the full history: all `_metric_at_ceiling` and the
    # plateau seed need. review_rejections_total keeps the anti-stall gate-disarm.
    review_rejections_total: int = 0
    verify_ever_passed: bool = False
    gateless_ever_committed: bool = False
    metric_best_score: float | None = None
    metric_at_ceiling: bool = False
    # Fork extras: the workspace HEAD and curator graph_version at this turn, so
    # ``fork --at-turn N`` cuts the branch at the right sha and clones the DAG as of
    # that version. Best-effort at write time: "" / 0 when git/curator was
    # unreadable. Plain resume reads head_sha (its divergence guard) only.
    head_sha: str = ""
    graph_version: int = 0


def _load_state_object(path: Path, what: str) -> dict[str, Any]:
    """Read a state JSON file and require the top-level shape to be an object.

    Valid JSON that is null, a list, or a scalar (a truncated/tampered state
    file) otherwise reached ``raw.get(...)`` / ``raw[...]`` and surfaced as an
    ``AttributeError``/``TypeError`` traceback the callers do not catch. Failing
    with a clean ``ValueError`` routes it to the same loud message as a version
    mismatch or a JSON decode error."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"malformed {what} at {path}: expected a JSON object, got {type(raw).__name__}"
        )
    return raw


def load_run_snapshot(path: Path) -> RunSnapshot:
    """Load a persisted run-state snapshot (``loop_state.json`` or a checkpoint).

    Refuses a snapshot from before the current ``SNAPSHOT_VERSION`` loudly: an
    in-flight run started before a state-format change predates this format and
    cannot be resumed or forked. Raises ``ValueError`` on any bad shape (fail
    loudly); ``resume``/``fork`` map it to a friendly refusal."""
    raw = _load_state_object(path, "run-state snapshot")
    version = raw.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"run-state snapshot at {path} is version {version!r}, not {SNAPSHOT_VERSION}: this "
            "run predates a state-format change and cannot be resumed or forked. Start a new run."
        )
    try:
        return RunSnapshot.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"malformed run-state snapshot at {path}: {exc}") from exc
