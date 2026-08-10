# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""How a session ends and how it resumes: the SessionResult the workflow returns, the
ResumeError it raises, and the provider-agnostic resume snapshot written before
each LLM call (load here; the loop owns saving it)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

# Every way a session can end -- run, plan, ask alike, which is why the
# reasons include finish_planning and answered. The loop constructs all of
# these except
# "ask_repl_empty" (an interactive ask session that ended before any question
# was asked, ui/cli/_ask.py). Typed so a new outcome must be declared here
# before a SessionResult can carry it.
SessionEndReason = Literal[
    "finish_session",
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
    "interrupted",  # KeyboardInterrupt; emitted by the app layer, not the loop
    "crashed",  # the loop escaped with a fault; also app-layer only
    "steer_abort",
    "undone",  # /undo forked back before the last message; the fork continues
    "detached",
    "prompt_revision_failed",
    "plan_unreadable",
    "max_iterations",
    "ask_repl_empty",
    "gate_stale",
    "gate_red_at_base",
]


# Whether the verify gate was green when the run ended, on its own axis: a
# deliberate finish and a verified one are different facts, and collapsing them
# into ``completed`` made a finish_session over a red verify exit 0 and auto-merge.
# ``failed`` means a red gate was OBSERVED (the last verify ran and failed);
# ``unverified`` means a gate exists but no observation covers the final tree
# (no verify ran this leg, or edits landed after the last green) -- folding that
# into ``failed`` printed "the gate is red" over a gate that never ran.
# ``not_applicable`` covers both a gateless session (no verify_command) and one
# that stopped before any verdict existed.
Verification = Literal["passed", "failed", "unverified", "not_applicable"]


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Final state of a session.

    ``reason`` values (each constructed in loop.py):
      finish_session        - agent called the finish_session tool explicitly.
      finish_planning   - plan-mode agent called the finish_planning tool.
      silent_finish     - agent emitted text but no tool_use (talking).
      went_quiet        - agent emitted neither text nor tool_use.
      budget_exhausted  - BudgetTracker raised; partial progress kept.
      provider_error    - ProviderError after retry; loop aborted.
      metric_plateau    - metric run tied prior best after enough samples.
      verify_settled    - verify passed and the worker stopped making changes.
      settled           - a GATELESS run's quiet finish: work committed, the
                          worker went idle, and no verify ever gated it (none
                          existed, or a mid-run adopted one never passed;
                          all_passed stays False).
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
      undone            - operator sent /undo; the leg ended after forking a
                          child at the state before their last message.
      detached          - operator chose "detach"; the CLI respawns a detached
                          `resume` to continue the run in the background.
      prompt_revision_failed - revise_prompt failed before the worker loop.
      plan_unreadable   - plan mode could not re-read plan.md; parked with
                          the remedy in the summary (resumable).
      max_iterations    - hit max_iterations cap without finish.
      ask_repl_empty    - interactive ask session ended with no question asked.
      gate_stale        - the worker finished over a red gate it says no longer
                          matches the task (it tests behaviour this run changed,
                          or cannot run at all) and proposed a replacement. The
                          gate is UNCHANGED and the run does not pass; the
                          operator decides.
      gate_red_at_base  - the gate is red, and it was ALREADY red before this
                          run touched anything (a verify ran against an
                          unmodified tree and failed). "Your run failed" and
                          "your change broke nothing new" are different facts.
    """

    completed: bool
    reason: SessionEndReason
    summary: str
    iterations: int
    tool_calls: int
    finish_payload: dict[str, Any] | None = None
    # The replacement gate the worker proposed, when it finished declaring the
    # configured one stale. Recorded and surfaced; never acted on.
    stale_gate: str = ""
    # The SAME fact `session.end.all_passed` carries, on the result the app layer
    # reads: `completed` means the agent stopped deliberately, never that the
    # work verified.
    verified: Verification = "not_applicable"


class ResumeError(Exception):
    """Raised when resume cannot proceed (missing/corrupt snapshot)."""


# Bump on ANY change to the persisted shape below. An in-flight run written by an
# older agent6 then refuses to resume/fork loudly (see load_session_snapshot) rather
# than parsing into a half-populated run. Finished runs never need a snapshot, so
# they keep rendering across the bump.
SNAPSHOT_VERSION = 2


class SessionSnapshot(BaseModel):
    """The persisted state of an in-flight session: what ``resume`` re-enters and what
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
    # SessionManifest.user_task is the DISPLAY twin (truncated [:4000]); this is
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
    # The last verify observation, so a resumed leg is not born amnesiac: a
    # green finish resumed and finished untouched used to read "unverified"
    # (and before that, "the gate is red"). Carried into the new leg only when
    # head_sha still matches a clean worktree (see _carry_verify_verdict);
    # baseline_ok is about the BASE commit, which resume never moves. Additive
    # defaults: an older snapshot loads as "nothing observed", exactly its truth.
    last_verify_ok: bool | None = None
    edited_since_verify: bool = False
    baseline_ok: bool | None = None
    # Standing-goal spin guard (see _LoopState.standing_tools_mark). Additive:
    # an old snapshot without it restores the never-absorbed default.
    standing_tools_mark: int = -1
    # /parallel groups dispatched so far. Run-lifetime, not leg-lifetime: lane
    # ids and their imported branches embed the group number
    # (`<run>-p<N>-l<i>`), so a resume that restarted at p1 rebuilt a prior
    # group's exact ids and collided on its clone dirs / branches.
    parallel_groups_dispatched: int = 0
    # Operator /pin instructions, re-injected verbatim after every tier-2
    # restart. Run-lifetime like the group counter above (additive default:
    # a snapshot written before pins existed loads with none).
    pins: tuple[str, ...] = ()
    # Fork extras: the workspace HEAD and curator graph_version at this turn.
    # ``fork --at-turn N`` cuts the branch at head_sha; graph_version is recorded
    # but the DAG is copied at its latest state, not rebuilt (see app/fork.py).
    # Best-effort at write time: "" / 0 when git/curator was unreadable. Plain
    # resume reads head_sha (its divergence guard) only.
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


def load_session_snapshot(path: Path) -> SessionSnapshot:
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
        return SessionSnapshot.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"malformed run-state snapshot at {path}: {exc}") from exc
