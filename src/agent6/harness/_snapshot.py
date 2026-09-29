# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold a session's end and the snapshot a resume re-enters.

`SessionResult` is what the harness returns and `SessionSnapshot` the provider-agnostic state
written before each provider call; the loop saves it and `load_session_snapshot` loads it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Any, Literal

import pydantic

# Every way a session can end; `SessionResult` says what each means.
SessionEndReason = Literal[
    "finish_session",
    "finish_planning",
    "answered",
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
    "interrupted",
    "crashed",
    "steer_abort",
    "steer_exit",
    "undone",
    "detached",
    "prompt_revision_failed",
    "plan_unreadable",
    "max_iterations",
    "ask_repl_empty",
    "gate_stale",
    "gate_red_at_base",
    "no_lane_result",
    "no_lane_passed",
]


# The gate's word at the end: `failed` is an observed red, `unverified` a tree no verdict covers.
Verification = Literal["passed", "failed", "unverified", "not_applicable"]


@dataclasses.dataclass(frozen=True, slots=True)
class SessionResult:
    """Hold the final state of a session.

    The end reasons, constructed by the loop unless a layer is named:
        finish_session: the model called finish_session.
        finish_planning: the model called finish_planning in plan mode.
        answered: the final prose of an ask session is the answer.
        interrupted: a KeyboardInterrupt (the app layer).
        crashed: the loop escaped with a fault (the app layer).
        steer_exit: /exit at the pause menu.
        silent_finish: the model sent text and no tool call.
        went_quiet: the model sent neither text nor a tool call.
        budget_exhausted: the budget tracker raised; partial progress is kept.
        provider_error: a provider error survived the retry.
        metric_plateau: a metric run tied its prior best after enough samples.
        verify_settled: the gate passed and the worker stopped changing the tree.
        settled: a quiet finish nothing verified: no gate, an adopted one that never passed,
            or edits after the last green.
        no_progress: the same verify failure outlived the no-progress ladder (resumable).
        tool_error_stuck: the same tool error outlived the tool-error ladder (resumable).
        verify_command_unexecutable: the operator's verify or metric command cannot run in the jail.
        loop_guard_killed: an identical tool call repeated past the kill threshold.
        interactive_stop: the operator chose stop at the after-commit hook.
        steer_abort: the operator stopped the run, mid-call, at a step boundary or while parked.
        undone: the operator sent /undo; a child was forked at the state before their message.
        detached: the operator chose detach; the CLI respawns a detached resume.
        prompt_revision_failed: revise_prompt failed before the worker loop.
        plan_unreadable: plan mode could not re-read plan.md (resumable).
        max_iterations: the iteration cap was hit without a finish.
        ask_repl_empty: an interactive ask session ended with no question (the CLI).
        gate_stale: the worker finished over a red gate it declares stale, with a replacement
            proposed; the gate is unchanged and the run does not pass.
        gate_red_at_base: the gate was red before the run touched anything.
        no_lane_result: no lane of a fan-out produced a rankable result (the app layer).
        no_lane_passed: no lane of a fan-out went green (the app layer).

    Attributes:
        completed: Whether the model stopped deliberately; never whether the work verified.
        reason: The end reason.
        summary: The end's summary text.
        iterations: The turns run.
        tool_calls: The tool calls made.
        finish_payload: The finish tool's payload, when any.
        stale_gate: The replacement gate the worker proposed; recorded, never acted on.
        verified: The gate's word, the same fact `session.end.all_passed` carries.
    """

    completed: bool
    reason: SessionEndReason
    summary: str
    iterations: int
    tool_calls: int
    finish_payload: dict[str, Any] | None = None
    stale_gate: str = ""
    verified: Verification = "not_applicable"


@dataclasses.dataclass(frozen=True, slots=True)
class End:
    """Record a decision to end the run, as `Harness._finish` applies it.

    The finish checkpoints a dirty worktree first, writes the `session.end` event, then returns
    the result.

    Attributes:
        reason: The end reason.
        summary: The end's summary text.
        completed: Whether the model stopped deliberately.
        verdict: What the event's `all_passed` carries: `failed`, `grounded` on the final tree's
            verify state, or `passed`.
        checkpoint: Whether to checkpoint a dirty worktree; an operator's stop keeps the choice.
        event: Whether to write the event; a detach writes none, the caller respawns the run.
        roots: Whether to pass the pending root tasks; None passes them on a clean verdict only.
        scoped: Whether a scoped gate certified the tree, for a `passed` verdict.
        finish_payload: The finish tool's payload, when any.
        stale_gate: The replacement gate the worker proposed.
        fields: Extra fields on the event.
    """

    reason: SessionEndReason
    summary: str
    completed: bool = False
    verdict: Literal["failed", "grounded", "passed"] = "failed"
    checkpoint: bool = True
    event: bool = True
    roots: bool | None = None
    scoped: bool = False
    finish_payload: dict[str, Any] | None = None
    stale_gate: str = ""
    fields: Mapping[str, object] = dataclasses.field(default_factory=dict)


class ResumeError(Exception):
    """A resume cannot proceed: the snapshot is missing or corrupt."""


# Bump on any change to the persisted shape: an older in-flight run then refuses to resume or fork.
SNAPSHOT_VERSION = 4


class SessionSnapshot(pydantic.BaseModel):
    """Hold the persisted state of an in-flight session, what a resume re-enters and a fork clones.

    The loop advances `loop_state.json` at every safe boundary (before each provider call and
    after each turn's tools land) and writes `checkpoints/<NNNN>.json` once per turn before the
    call, so `fork --at-turn` has one meaning. The messages are anthropic-shaped, so an OpenAI
    transcript cannot seed a resume. Every field added since version 1 has an additive default
    that reads as "nothing observed".

    Attributes:
        version: The snapshot format, refused when it is not `SNAPSHOT_VERSION`.
        system: The frozen system prompt.
        messages: The conversation.
        tool_calls: The tool calls made so far.
        next_iteration: The turn the snapshot's provider call runs.
        root_task_id: The task graph's root, when the run has one.
        original_task: The exact task text the run launched with; the manifest holds a
            truncated display twin.
        verify_command: The gate the run resolved, () when gateless; a resume reuses it rather
            than inferring again.
        review_rejections_total: The before-finish panel's rejection count, which disarms it.
        verify_ever_passed: Whether any verify passed.
        verify_ever_failed: Whether any verify failed.
        gateless_ever_edited: Whether a gateless run edited the tree.
        metric_best_score: The best metric score, when a metric is configured.
        metric_at_ceiling: Whether the metric reached its ceiling.
        last_verify_ok: The last verify's result, carried only onto a clean tree at `head_sha`.
        edited_since_verify: Whether the tree changed after the last verify.
        baseline_ok: Whether the gate passed on the base commit, which a resume never moves.
        verify_scoped: Whether the full gate overran its timeout; a fact about the suite, so it
            carries unconditionally.
        memory_written: Whether the worker wrote a memory fact; the nudges are once per run.
        memory_flip_nudged: Whether the memory flip nudge fired.
        memory_finish_nudged: Whether the memory finish nudge fired.
        ok_tool_calls: The executed dispatches, for the standing spin guard.
        standing_tools_mark: The dispatch count at the last standing re-entry, -1 before any.
        standing_fruitless: The fruitless standing re-entries in a row.
        parallel_groups_dispatched: The /parallel groups dispatched, run-lifetime because lane
            ids embed the group number.
        pins: The operator's /pin texts, re-injected after every context restart.
        head_sha: The chain tip at this turn, "" when git was unreadable; a fork cuts here and a
            resume checks it for divergence.
        graph_version: The task graph version a fork rebuilds by replay, 0 when unreadable.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    version: int = SNAPSHOT_VERSION
    system: str
    messages: list[dict[str, Any]]
    tool_calls: int
    next_iteration: int
    root_task_id: str | None
    original_task: str
    verify_command: tuple[str, ...]
    review_rejections_total: int = 0
    verify_ever_passed: bool = False
    verify_ever_failed: bool = False
    gateless_ever_edited: bool = False
    metric_best_score: float | None = None
    metric_at_ceiling: bool = False
    last_verify_ok: bool | None = None
    edited_since_verify: bool = False
    baseline_ok: bool | None = None
    verify_scoped: bool = False
    memory_written: bool = False
    memory_flip_nudged: bool = False
    memory_finish_nudged: bool = False
    ok_tool_calls: int = 0
    standing_tools_mark: int = -1
    standing_fruitless: int = 0
    parallel_groups_dispatched: int = 0
    pins: tuple[str, ...] = ()
    head_sha: str = ""
    graph_version: int = 0


def _load_state_object(path: pathlib.Path, what: str) -> dict[str, Any]:
    """Read a state JSON file whose top level must be an object.

    Args:
        path: The file.
        what: The file's name in errors.

    Returns:
        The parsed object.

    Raises:
        ValueError: When the file is not valid JSON or its top level is not an object.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"unreadable {what} at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"malformed {what} at {path}: expected a JSON object, got {type(raw).__name__}"
        )
    return raw


def load_session_snapshot(path: pathlib.Path) -> SessionSnapshot:
    """Load a snapshot, `loop_state.json` or a checkpoint.

    Args:
        path: The snapshot file.

    Returns:
        The snapshot.

    Raises:
        ValueError: When the snapshot predates `SNAPSHOT_VERSION` or has a bad shape; resume
            and fork turn it into a refusal.
    """
    raw = _load_state_object(path, "run-state snapshot")
    version = raw.get("version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"run-state snapshot at {path} is version {version!r}, not {SNAPSHOT_VERSION}: this "
            "run predates a state-format change and cannot be resumed or forked. Start a new run."
        )
    try:
        return SessionSnapshot.model_validate(raw)
    except pydantic.ValidationError as exc:
        raise ValueError(f"malformed run-state snapshot at {path}: {exc}") from exc


# Written before a turn's tools dispatch and deleted after the snapshot advances past them.
TURN_IN_FLIGHT_NAME = "turn_in_flight.json"


def write_turn_marker(path: pathlib.Path, iteration: int, tools: tuple[str, ...]) -> None:
    """Write the marker; a failed write never fails the turn.

    Args:
        path: The marker file.
        iteration: The turn whose tools are about to run.
        tools: The tool names of the turn.
    """
    with contextlib.suppress(OSError):
        path.write_text(
            json.dumps({"iteration": iteration, "tools": list(tools)}), encoding="utf-8"
        )


def read_turn_marker(path: pathlib.Path) -> tuple[int, tuple[str, ...]] | None:
    """Return the marker's (iteration, tool names), or None when it is absent or unreadable.

    Args:
        path: The marker file.

    Returns:
        The turn whose tools may have run and their names, or None.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    iteration = raw.get("iteration") if isinstance(raw, dict) else None
    if not isinstance(iteration, int):
        return None
    tools = raw.get("tools")
    names = tuple(t for t in tools if isinstance(t, str)) if isinstance(tools, list) else ()
    return iteration, names


def clear_turn_marker(path: pathlib.Path) -> None:
    """Delete the marker when present."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
