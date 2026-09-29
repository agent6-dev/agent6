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

import dataclasses

from agent6.harness import (
    _advice,
    _conversation,
    _finish_gates,
    _guards,
    _metric,
    _quiet_turns,
    _snapshot,
    _spiral,
    _verify_verdict,
)
from agent6.providers import ProviderResponse


@dataclasses.dataclass(slots=True)
class LoopState:
    """Mutable per-run bookkeeping threaded through the agent loop.

    One guard object per heuristic holds its counters; the rule itself is the
    advisor function in `_guards`, so a heuristic is its guard, its advisor and
    its pin.

    Attributes:
        original_task: The task text the run started with.
        tool_calls: Dispatches so far.
        ok_tool_calls: Dispatches that executed without a ToolError; the standing
            goal's spin guard counts only these as work since the last re-entry.
        decisions_recorded: Rulings this execution appended to DECISIONS.md, for
            the finish-time check.
        tier2_floor_chars: The context size at the last tier-2 restart; the next
            fires only once the context grew 25% past it. Execution-local.
        verify: The verify verdict every "is the run green" consumer reads.
        ever_edited: True once any edit landed.
        plan_injected: The plan.md text the planner was last shown; fresh per
            execution, so a resumed planner is re-shown the operator's edits.
        root_task_id: The DAG root task id, set once by the loop.
        system: The system prompt, set once by the loop.
        parallel_groups_dispatched: How many `/parallel` sibling groups this run
            dispatched; names each group's lanes. Persisted.
        pins: Operator `/pin` instructions, re-injected verbatim after every tier-2
            restart, capped at PINS_MAX_CHARS in total. Persisted.
        spiral: The repeat-call guard (`_spiral`).
        no_progress: The no-progress ladder.
        settled: The settled-stop guard.
        metric: The metric guard.
        quiet: The empty-turn guard.
        stagnation: The stagnation guard.
        memory: The memory nudge state.
        standing: The standing goal's state.
        reach: The reachability guard.
        focus: The DAG focus guard.
        gates: The finish gates' state.
        budget_nudges: The budget nudge state.
    """

    original_task: str
    tool_calls: int
    ok_tool_calls: int = 0
    decisions_recorded: list[str] = dataclasses.field(default_factory=list)
    tier2_floor_chars: int = 0
    verify: _verify_verdict.VerifyVerdict = dataclasses.field(
        default_factory=_verify_verdict.VerifyVerdict
    )
    ever_edited: bool = False
    plan_injected: str = ""
    root_task_id: str | None = None
    system: str = ""
    parallel_groups_dispatched: int = 0
    pins: list[str] = dataclasses.field(default_factory=list)
    spiral: _spiral.SpiralGuard = dataclasses.field(default_factory=_spiral.SpiralGuard)
    no_progress: _guards.Ladder = dataclasses.field(default_factory=_guards.no_progress_ladder)
    settled: _guards.SettledGuard = dataclasses.field(default_factory=_guards.SettledGuard)
    metric: _metric.MetricGuard = dataclasses.field(default_factory=_metric.MetricGuard)
    quiet: _quiet_turns.QuietGuard = dataclasses.field(default_factory=_quiet_turns.QuietGuard)
    stagnation: _guards.StagnationGuard = dataclasses.field(default_factory=_guards.StagnationGuard)
    memory: _guards.MemoryState = dataclasses.field(default_factory=_guards.MemoryState)
    standing: _guards.StandingGoal = dataclasses.field(default_factory=_guards.StandingGoal)
    reach: _guards.ReachabilityGuard = dataclasses.field(default_factory=_guards.ReachabilityGuard)
    focus: _guards.FocusGuard = dataclasses.field(default_factory=_guards.FocusGuard)
    gates: _finish_gates.FinishGates = dataclasses.field(default_factory=_finish_gates.FinishGates)
    budget_nudges: _guards.BudgetNudges = dataclasses.field(default_factory=_guards.BudgetNudges)


def restore_completion_state(state: LoopState, snap: _snapshot.SessionSnapshot) -> None:
    """Carry a resume snapshot's completion bookkeeping into fresh loop state.

    The review gate-disarm, metric and verify-settled stop logic keep their
    counts across a resume. A fresh run never calls this. A persisted completion
    field is one field on SessionSnapshot plus one line here.

    Args:
        state: The fresh loop state.
        snap: The snapshot the run resumes from.
    """
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
        # One synthetic sample carries the prior best; the plateau stop re-arms after a
        # few measurements, the ceiling stop is immediate.
        state.metric.history.append(
            _metric.MetricSample(
                label="resumed",
                score=snap.metric_best_score,
                returncode=0,
                at_ceiling=snap.metric_at_ceiling,
            )
        )


class NextTurn:
    """The sentinel for a discarded turn: the loop starts the next iteration at once.

    A mid-stream steer that chose continue, or injected an instruction, discards
    the turn.
    """


NEXT_TURN = NextTurn()


@dataclasses.dataclass(slots=True)
class TurnState:
    """Mutable bookkeeping for one assistant turn that dispatched tools.

    The loop creates one per tool-use iteration and threads it through the turn
    phases; a field is written in one phase and read in a later one.
    Cross-iteration state stays on `LoopState`.

    Attributes:
        iteration: The loop iteration.
        resp: The provider response driving this turn.
        assistant: The response's turn in the conversation; its parsed tool_uses
            drive the dispatch.
        finish: The finish_session or finish_planning call dispatched this turn;
            a finish gate may revoke it before the stop checks honour it.
        end_returned: True when a gate handed back an end declared without
            finish_session (a settled stop, a silent finish).
        ending: The end whose gates are running: "finish_session" for the tool
            call, else the name the harness declared ("settled", "silent_finish").
        tool_results: The user-turn items in dispatch order, with advisory notices
            appended after (or, for the broken-verify flag, between them).
        verify_just_passed: Verify went green this turn.
        verify_just_failed: Verify went red this turn.
        verify_flipped_green: Verify went green this turn after the run's last
            verify was red; feeds the one-shot memory flip advisory.
        edit_since_verify_pass: An edit after a passing verify in the same turn
            changed the tree that verify validated; only the auto-commit gate
            reads it.
        edited: An edit landed this turn.
        committed: A checkpoint committed this turn.
        dag_mutated: A DAG tool ran this turn.
        metric_sampled: The worker ran the metric itself this turn.
        metric_feedback: The metric notice for the model, if any.
        metric_plateau_finish: The plateau finish text, if the metric guard ended
            the run.
        review_text: The review's findings, if a review ran.
        end_rejected: Whether the before-finish panel rejected this turn's end,
            sat once however many ends the turn declares; None until it sits.
        stops: The advisors' decisions to end the run, in the order made; the
            stop checks honour the first one left after a standing task's absorb.
    """

    iteration: int
    resp: ProviderResponse
    assistant: _conversation.AssistantTurn
    finish: _finish_gates.FinishCall | None = None
    end_returned: bool = False
    ending: str | None = None
    tool_results: list[_conversation.ToolResultItem | _conversation.Notice] = dataclasses.field(
        default_factory=list
    )
    verify_just_passed: bool = False
    verify_just_failed: bool = False
    verify_flipped_green: bool = False
    edit_since_verify_pass: bool = False
    edited: bool = False
    committed: bool = False
    dag_mutated: bool = False
    metric_sampled: bool = False
    metric_feedback: str | None = None
    metric_plateau_finish: str | None = None
    review_text: str | None = None
    end_rejected: bool | None = None
    stops: list[_advice.Stop] = dataclasses.field(default_factory=list)
