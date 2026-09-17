# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent loop's mutable state shapes.

`LoopState` is the per-run bookkeeping threaded through every loop phase,
`TurnState` the per-turn slice one dispatching iteration accumulates, and
`NEXT_TURN` the sentinel for a discarded turn. `restore_completion_state`
carries a resume snapshot's completion-relevant fields into fresh state.
The loop's phase methods live in `loop.py`; these are the shapes they take.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent6.providers import ProviderResponse
from agent6.workflows._conversation import AssistantTurn, Notice, ToolResultItem
from agent6.workflows._guards import (
    BudgetNudges,
    FinishGates,
    FocusGuard,
    MemoryNudges,
    MetricGuard,
    NoProgressGuard,
    QuietGuard,
    ReachabilityGuard,
    SettledGuard,
    StagnationGuard,
    StandingGoal,
    Stop,
)
from agent6.workflows._metric import MetricSample
from agent6.workflows._session_state import SessionSnapshot
from agent6.workflows._spiral_guards import SpiralGuard
from agent6.workflows._verify_verdict import VerifyVerdict


@dataclass(slots=True)
class LoopState:
    """Mutable per-run bookkeeping threaded through the agent loop: the shared
    facts every phase reads, and one guard object per heuristic (`_guards`)
    holding that heuristic's counters and its rule, so adding, tuning or
    deleting a heuristic touches its guard, its loop hook and its pin."""

    original_task: str
    tool_calls: int
    # Dispatches that EXECUTED (no ToolError): the standing-goal spin guard
    # reads this, so a refused call (a malformed edit, a retirement the
    # curator rejects) is not "work since the last re-entry".
    ok_tool_calls: int = 0
    # Rulings this leg appended to DECISIONS.md, for the finish-time check.
    decisions_recorded: list[str] = field(default_factory=list)
    # Tier-2 re-fires only after the context grew 25% past the last restart's
    # size: a restart that lands near the threshold must not summarise every
    # other iteration. Leg-local: a resumed leg rebuilds a small context anyway.
    tier2_floor_chars: int = 0
    # The verify verdict: the one object every "is the run green" consumer
    # reads (gates, review grounding, snapshot, notices).
    verify: VerifyVerdict = field(default_factory=VerifyVerdict)
    ever_edited: bool = False
    # plan mode: the plan.md text the planner was last shown. Fresh per leg, so a
    # resumed planner is always re-shown the file the operator may have edited.
    plan_injected: str = ""
    # The DAG root task id and the system prompt (set once by _drive_loop), so
    # a steer-boundary phase can parent a node or snapshot without being
    # handed them.
    root_task_id: str | None = None
    system: str = ""
    # How many `/parallel` sibling groups this run has dispatched. Names each
    # group's lanes (`<run-id>-p<seq>-l<i>`); persisted.
    parallel_groups_dispatched: int = 0
    # Operator `/pin` instructions, re-injected verbatim after every tier-2
    # restart; total chars capped at PINS_MAX_CHARS; persisted.
    pins: list[str] = field(default_factory=list)
    # The guards (`_guards`; the spiral one in `_spiral_guards`).
    spiral: SpiralGuard = field(default_factory=SpiralGuard)
    no_progress: NoProgressGuard = field(default_factory=NoProgressGuard)
    settled: SettledGuard = field(default_factory=SettledGuard)
    metric: MetricGuard = field(default_factory=MetricGuard)
    quiet: QuietGuard = field(default_factory=QuietGuard)
    stagnation: StagnationGuard = field(default_factory=StagnationGuard)
    memory: MemoryNudges = field(default_factory=MemoryNudges)
    standing: StandingGoal = field(default_factory=StandingGoal)
    reach: ReachabilityGuard = field(default_factory=ReachabilityGuard)
    focus: FocusGuard = field(default_factory=FocusGuard)
    gates: FinishGates = field(default_factory=FinishGates)
    budget_nudges: BudgetNudges = field(default_factory=BudgetNudges)


def restore_completion_state(state: LoopState, snap: SessionSnapshot) -> None:
    """Carry a resume snapshot's completion-relevant bookkeeping into fresh loop
    state, so the review gate-disarm, metric, and verify-settled stop logic don't
    regress to zero after a resume (re-rejecting a correct finish_session, re-counting
    idle). A fresh run() never calls this and keeps LoopState's defaults. Adding a
    persisted completion field is one field on SessionSnapshot plus one line here."""
    state.gates.review_total = snap.review_rejections_total
    state.verify.ever_passed = snap.verify_ever_passed
    state.verify.ever_failed = snap.verify_ever_failed
    state.verify.scoped = snap.verify_scoped
    state.settled.gateless_ever_edited = snap.gateless_ever_edited
    state.memory.written = snap.memory_written
    state.memory.flip_nudged = snap.memory_flip_nudged
    state.memory.finish_nudged = snap.memory_finish_nudged
    state.parallel_groups_dispatched = snap.parallel_groups_dispatched
    state.pins = list(snap.pins)
    if snap.metric_at_ceiling or snap.metric_best_score is not None:
        # Seed one synthetic sample so `MetricGuard.at_ceiling` and the plateau guard
        # see the prior best (the snapshot carries a compact summary, not the
        # full history). `label` marks it resume-reconstructed. Consequence:
        # `metric_plateau_summary` needs several parsed samples to fire, so a
        # resumed already-plateaued run takes a few measurements to re-arm the
        # plateau-stop (it never stops early; the ceiling-stop is immediate).
        state.metric.history.append(
            MetricSample(
                label="resumed",
                score=snap.metric_best_score,
                returncode=0,
                at_ceiling=snap.metric_at_ceiling,
            )
        )


class NextTurn:
    """Sentinel returned by `_turn_provider_call`: the turn was discarded (a
    mid-stream steer chose continue, or injected an instruction) and the loop
    should start the next iteration immediately."""


NEXT_TURN = NextTurn()


@dataclass(slots=True)
class TurnState:
    """Mutable bookkeeping for ONE assistant turn that dispatched tools.

    `_drive_loop` creates one per tool-use iteration and threads it through
    the turn phases; a field earns its place by being written in one phase and
    read in a later one, so each phase is a method taking `(state, turn)`
    rather than a slice of ~15 hand-threaded locals. Cross-iteration state
    stays on `LoopState`.
    """

    iteration: int
    # The provider response driving this turn.
    resp: ProviderResponse
    # The response's turn in the conversation; its parsed tool_uses drive the
    # dispatch (the conversation is the single source of what was called).
    assistant: AssistantTurn
    # A finish_session/finish_planning call captured this turn; the finish gates
    # may revoke it (set back to None) before the stop checks honour it.
    finish_signal: str | None = None
    finish_payload: dict[str, Any] | None = None
    # An end the harness or the model declared without finish_session (a
    # settled stop, a silent finish) that a gate handed back this turn.
    end_returned: bool = False
    # A finish that declared the configured gate stale, with the replacement it
    # proposes. Recorded and surfaced; the gate itself never moves.
    finish_stale_gate: str = ""
    finish_kind: Literal["finish_session", "finish_planning"] = "finish_session"
    # The user-turn items accumulated for this turn: tool results in dispatch
    # order, with advisory notices (review, metric, nudges) appended after
    # (or, for the broken-verify flag, between them).
    tool_results: list[ToolResultItem | Notice] = field(default_factory=list)
    verify_just_passed: bool = False
    verify_just_failed: bool = False
    # Verify went green THIS turn after the run's last verify was red; feeds
    # the one-shot memory flip advisory in _turn_notices.
    verify_flipped_green: bool = False
    # An apply_edit/apply_patch AFTER a passing verify in the same turn changes
    # the tree that verify validated, so the green no longer applies. Tracked
    # separately from verify_just_passed (which the metric path also reads) so
    # only the auto-commit gate is affected.
    edit_since_verify_pass: bool = False
    edited: bool = False
    committed: bool = False
    dag_mutated: bool = False
    metric_sampled: bool = False  # the worker ran the metric itself this turn
    metric_feedback: str | None = None
    metric_plateau_finish: str | None = None
    review_text: str | None = None
    # The before-finish panel's verdict on this turn's end (True: rejected),
    # sat once however many ends the turn declares.
    end_reviewed: bool | None = None
    plateau_should_stop: bool = False
    verify_settled_stop: bool = False
    no_progress_stop: bool = False
    # The advisors' decisions to end the run, in the order they were made;
    # the stop checks honour the first one left after a standing task's absorb.
    stops: list[Stop] = field(default_factory=list)
