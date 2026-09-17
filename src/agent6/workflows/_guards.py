# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop's guards: each heuristic that nudges or ends a run is one
advisor function here, reading its counters from the guard object the loop
holds on `LoopState` and returning what the harness says and records
(`Nudge`) or nothing. The loop runs `AFTER_TOOLS` in order once a turn's
tools have run and applies each outcome (`Workflow._take`).

Leg-local by design, like every counter not named in `SessionSnapshot`: a
resume is operator-initiated, so a resumed leg's refreshed patience is the
operator granting another window. The completion-relevant subset persists
(`restore_completion_state`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from agent6.workflows._dag_focus import STUCK_NUDGE_MAX, STUCK_ON_TASK_AFTER
from agent6.workflows._metric import MetricSample
from agent6.workflows._nudges import (
    LOOP_GUARD_NOTICE_AFTER,
    MEMORY_FLIP_NUDGE,
    NO_PROGRESS_ESCALATE_AFTER,
    NO_PROGRESS_ESCALATION,
    NO_PROGRESS_NUDGE,
    NO_PROGRESS_NUDGE_AFTER,
    NO_PROGRESS_STOP_AFTER,
    STAGNATION_NUDGE,
    STAGNATION_NUDGE_GATELESS,
    TOOL_DENIED_NUDGE,
    TOOL_ERROR_ESCALATION,
    TOOL_ERROR_NUDGE,
    VERIFY_SETTLED_NUDGE_AFTER,
    VERIFY_SETTLED_STOP_AFTER,
    loop_guard_words,
)
from agent6.workflows._session_state import End

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState, TurnState

Rung = Literal["nudge", "escalate", "stop"]


@dataclass(frozen=True, slots=True)
class Nudge:
    """What an advisor says to the model this turn, and how the harness
    records it: the notice text, the event (with its fields) and the log
    line; "" skips the event or the line."""

    text: str
    event: str = ""
    fields: Mapping[str, object] = field(default_factory=dict)
    log: str = ""


@dataclass(frozen=True, slots=True)
class Stop:
    """An advisor's decision to end the run, honoured once the turn's results
    and snapshot are on disk (`Workflow._turn_stop_checks`): `end` is what
    `_finish` records, `log` the line written then. `soft` names the reason
    a standing task's re-entry nudge carries when it absorbs the stop ("" =
    a hard stop nothing converts); `declared` names the ending the end gates
    judge as soon as the stop is decided ("" = a fault no gate judges). The
    event is emitted when the stop is decided."""

    end: End
    soft: str = ""
    declared: str = ""
    event: str = ""
    fields: Mapping[str, object] = field(default_factory=dict)
    log: str = ""


@dataclass(frozen=True, slots=True)
class TurnContext:
    """The run facts an advisor reads, built once per turn. A fact that can
    move within the turn (a harness verify can deny or un-adopt the gate,
    an edit moves the tree) is a zero-arg callable read where it is needed."""

    mode: Literal["run", "plan", "ask", "agent"]
    iteration: int
    # The leg's first iteration: a turn allowance counts from it.
    leg_start: int
    guards: GuardSettings
    # A metric goal is configured: the plateau and ceiling rules own the
    # run's end, and the plain-run guards stand down.
    metric: bool
    # A memory store is wired, so the memory nudges apply.
    memory_wired: bool
    gate_present: Callable[[], bool]
    verify_command: Callable[[], tuple[str, ...]]
    tree_sha: Callable[[], str]
    tree_green: Callable[[], bool | None]
    budget_remaining: Callable[[], float | None]
    operator_wait_s: Callable[[], float]
    open_subtasks: Callable[[], list[tuple[str, str]]]


# An advisor: one heuristic over the turn, the run state and the context,
# answering with what to say, a decision to end the run, or nothing.
Advisor = Callable[["TurnState", "LoopState", TurnContext], Nudge | Stop | None]


@dataclass(frozen=True, slots=True)
class GuardSettings:
    """The guards' operator-facing knobs. `went_quiet_max_nudges`: an empty
    turn (no text, no tool call) is answered with a harness notice and
    re-asked up to this many times per streak, reasoning-starvation bursts
    included; 0 ends the run on the first. `loop_guard_kill_threshold`: the
    same (tool, args) call this many times in a row ends the run as
    loop_guard_killed (the notice fires from three, every other turn); 0
    leaves the notice alone. `stagnation_notice_after_s`: one notice when
    this much wall clock passes with no edit and no verify (a recall spiral
    makes few calls with long reasoning between them, below every
    call-count guard's horizon); 0 disables."""

    went_quiet_max_nudges: int = 4
    loop_guard_kill_threshold: int = 10
    stagnation_notice_after_s: float = 300.0


def climb(
    streak: int, used: int, *, nudge_after: int, escalate_after: int, stop_after: int
) -> Rung | None:
    """The rung a streak reaches on a nudge, escalate, stop ladder: the first
    nudge at `nudge_after`, the escalation at `escalate_after` once the nudge
    went out, the stop at `stop_after` once both did; None between rungs.
    `used` is how many rungs went out already."""
    if streak >= stop_after and used >= 2:
        return "stop"
    if streak >= escalate_after and used == 1:
        return "escalate"
    if streak >= nudge_after and used == 0:
        return "nudge"
    return None


@dataclass(slots=True)
class NoProgressGuard:
    """N consecutive verify failures sharing one signature (run mode, no
    metric): nudge, escalate, then stop the run as no_progress. A green
    verify or a different failure resets the streak (`VerifyVerdict`); a new
    stuck point re-arms the ladder."""

    nudges_used: int = 0

    def climb(self, streak: int) -> Rung | None:
        rung = climb(
            streak,
            self.nudges_used,
            nudge_after=NO_PROGRESS_NUDGE_AFTER,
            escalate_after=NO_PROGRESS_ESCALATE_AFTER,
            stop_after=NO_PROGRESS_STOP_AFTER,
        )
        if rung == "nudge":
            self.nudges_used = 1
        elif rung == "escalate":
            self.nudges_used = 2
        return rung


def no_progress(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | Stop | None:
    """Nudge, escalate, then stop on a plain run's streak of identical verify
    failures (`NoProgressGuard.climb`, judged on a turn whose verify failed).
    A metric run is left to its plateau, ceiling and early-finish rules: a
    failing verify while it searches for an optimisation is expected there."""
    if ctx.mode != "run" or ctx.metric or not turn.verify_just_failed:
        return None
    streak = state.verify.fail_streak
    rung = state.no_progress.climb(streak)
    if rung is None:
        return None
    if rung == "stop":
        return Stop(
            End(
                "no_progress",
                f"stopped: the same verify failure persisted through {streak} consecutive"
                " runs despite two harness interventions; resume with a new approach or"
                " a bigger budget",
            ),
            soft="no_progress",
            log=f"LOOP: no_progress stop at iter {turn.iteration} (streak {streak})",
        )
    level = 2 if rung == "escalate" else 1
    return Nudge(
        NO_PROGRESS_ESCALATION if rung == "escalate" else NO_PROGRESS_NUDGE,
        event="loop.no_progress.nudge",
        fields={"iteration": turn.iteration, "streak": streak, "level": level},
    )


@dataclass(slots=True)
class SettledGuard:
    """Verify-settled completion (run mode, no metric): once the run reached a
    good state (a green verify, or an editing step on a gateless run), count
    the idle turns (no edit, no commit, an unchanged tree; a verify run is
    neutral), nudge once at `VERIFY_SETTLED_NUDGE_AFTER`, stop at
    `VERIFY_SETTLED_STOP_AFTER`. `tree` is the last tree seen, leg-local, so
    a resumed leg re-measures on its first turn."""

    tree: str = ""
    idle: int = 0
    nudged: bool = False
    gateless_ever_edited: bool = False

    def note_turn(self, tree: str, *, progress: bool, verify_ran: bool) -> None:
        self.tree = tree
        if progress:
            self.restart()
        elif not verify_ran:
            self.idle += 1

    def restart(self) -> None:
        """A fresh idle streak: the guard counts from zero and may nudge again."""
        self.idle = 0
        self.nudged = False

    def stop_due(self) -> bool:
        return self.idle >= VERIFY_SETTLED_STOP_AFTER

    def nudge_due(self) -> bool:
        """Whether the one settled nudge goes out now (it then counts as sent)."""
        if self.idle < VERIFY_SETTLED_NUDGE_AFTER or self.nudged:
            return False
        self.nudged = True
        return True


@dataclass(slots=True)
class MetricGuard:
    """A metric run's readings and its two patience counters: `plateau_nudges_used`
    counts final-slice plateau nudges, `finish_nudges_used` early finishes
    rejected while runway remains. `tree` is the worktree the metric was
    last sampled on (one reading per state of the tree); `denied` withholds
    the automatic metric for the rest of the run after the operator's no."""

    history: list[MetricSample] = field(default_factory=list)
    tree: str = ""
    denied: bool = False
    plateau_nudges_used: int = 0
    finish_nudges_used: int = 0

    def at_ceiling(self) -> bool:
        """Whether any verified sample reached the metric's provable ceiling
        (`SCORE: 27/27`): a metric that cannot improve, so an early finish is
        honoured and the nudging stops."""
        return any(sample.at_ceiling for sample in self.history)


@dataclass(slots=True)
class QuietGuard:
    """The turns that say nothing: an empty turn draws a nudge up to the cap
    per streak (`went_quiet_nudges_used`, reset by any non-empty turn); an
    early prose turn on an untouched tree draws `SILENT_NO_WORK_PATIENCE`
    nudges; a prose turn ending on a question draws one nudge to call
    ask_user."""

    went_quiet_nudges_used: int = 0
    silent_no_work_nudges_used: int = 0
    question_nudged: bool = False


@dataclass(slots=True)
class StagnationGuard:
    """One notice when the wall clock passes `stagnation_notice_after_s` with
    no edit and no verify. Monotonic time is process-relative, so neither
    field persists: a resumed run gets a fresh window."""

    started_monotonic: float = field(default_factory=time.monotonic)
    nudged: bool = False


def tool_error_ladder(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | Stop | None:
    """Nudge, escalate, then stop on a streak of identical tool errors (a
    call that keeps failing the same way: malformed args, a bad path),
    judged after each failed call (`SpiralGuard.climb_error`). A plain run's
    guard: a metric run's own machinery owns its end. A denial streak is a
    policy outcome, so its nudge says refused, not malformed."""
    if ctx.mode != "run" or ctx.metric:
        return None
    streak = state.spiral.error_streak
    rung = state.spiral.climb_error()
    if rung is None:
        return None
    if rung == "stop":
        return Stop(
            End(
                "tool_error_stuck",
                f"stopped: the same tool call failed {streak} times with the identical"
                " error despite two harness interventions; resume with a different"
                " approach",
            ),
            log=f"LOOP: tool_error stop at iter {turn.iteration} (streak {streak})",
        )
    if state.spiral.last_error_was_denial:
        text = TOOL_DENIED_NUDGE
    else:
        text = TOOL_ERROR_ESCALATION if rung == "escalate" else TOOL_ERROR_NUDGE
    level = 2 if rung == "escalate" else 1
    return Nudge(
        text,
        event="loop.tool_error.nudge",
        fields={"iteration": turn.iteration, "streak": streak, "level": level},
    )


def loop_guard_notice(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | None:
    """The repeat notice: the same (tool, args) call `LOOP_GUARD_NOTICE_AFTER`
    times in a row (`SpiralGuard.call_streak`, reset by any other call), re-armed
    after one quiet iteration, so an unbroken streak hears it every other turn
    until the kill threshold ends the run."""
    spiral = state.spiral
    if not (
        spiral.call_streak >= LOOP_GUARD_NOTICE_AFTER
        and spiral.warned_at_iteration < turn.iteration - 1
    ):
        return None
    spiral.warned_at_iteration = turn.iteration
    tool = (spiral.last_call_sig or "").split(":", 1)[0] or "<unknown>"
    return Nudge(
        loop_guard_words(tool, spiral.call_streak),
        event="loop.loop_guard.triggered",
        fields={"iteration": turn.iteration, "tool": tool, "streak": spiral.call_streak},
        log=f"  loop-guard: {tool} called {spiral.call_streak}x in a row - injecting notice",
    )


def stagnation(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | None:
    """One notice when `stagnation_notice_after_s` of wall clock passed on a
    run with no edit and no verify yet; time blocked on the operator is not
    the model's, and a gateless run's notice names no gate."""
    guard = state.stagnation
    after = ctx.guards.stagnation_notice_after_s
    if not (
        after > 0
        and ctx.mode == "run"
        and not guard.nudged
        and not state.ever_edited
        and state.verify.last_ok is None
    ):
        return None
    elapsed = time.monotonic() - guard.started_monotonic
    if elapsed >= after:
        elapsed -= ctx.operator_wait_s()
    if elapsed < after:
        return None
    guard.nudged = True
    minutes = max(1, int(elapsed // 60))
    text = STAGNATION_NUDGE if ctx.gate_present() else STAGNATION_NUDGE_GATELESS
    return Nudge(
        text.format(minutes=minutes),
        event="loop.stagnation.nudged",
        fields={"iteration": turn.iteration, "elapsed_s": int(elapsed)},
        log=f"  stagnation: {minutes}m with no attempt - injecting notice",
    )


@dataclass(slots=True)
class MemoryNudges:
    """The two memory write nudges (run mode, a memory store wired): one flip
    advisory when verify first goes green after failing, one deferred
    finish_session as the backstop; both silent once the worker recorded
    anything. Run-lifetime: all three persist in the snapshot."""

    written: bool = False
    flip_nudged: bool = False
    finish_nudged: bool = False


def memory_flip(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | None:
    """The memory flip advisory: once per run, at the first verify that goes
    green after a red one, while the worker has recorded nothing in the
    memory store (that is the moment a hard-won root cause is in hand; see
    `_nudges` for the measurement behind it)."""
    if not (
        turn.verify_flipped_green
        and ctx.mode == "run"
        and ctx.memory_wired
        and not state.memory.written
        and not state.memory.flip_nudged
    ):
        return None
    state.memory.flip_nudged = True
    return Nudge(
        MEMORY_FLIP_NUDGE,
        event="loop.memory_flip.nudged",
        fields={"iteration": turn.iteration},
        log="  memory: verify flipped green - injecting memory advisory",
    )


@dataclass(slots=True)
class StandingGoal:
    """Standing-goal re-entry: `ok_tool_calls` at the last absorption (-1 =
    never) and the consecutive fruitless re-entries since work last landed;
    `[workflow].standing_patience` decides how many are absorbed before an
    end is honoured (-1 = never on its own). Both persist."""

    tools_mark: int = -1
    fruitless: int = 0


@dataclass(slots=True)
class ReachabilityGuard:
    """Sandbox reachability: argv[0] of a run_command the jail failed to exec
    (not a nonzero exit) and its consecutive count. Only executed commands
    feed it; a validation error or a denial never entered the jail."""

    binary: str = ""
    streak: int = 0
    warned: bool = False

    def note(self, binary: str, *, exec_failed: bool) -> bool:
        """Record one command; True when its second consecutive exec failure
        is the first worth warning about (the caller still checks the host
        has the binary)."""
        if not exec_failed or not binary:
            self.binary = ""
            self.streak = 0
            return False
        if binary == self.binary:
            self.streak += 1
        else:
            self.binary = binary
            self.streak = 1
        return self.streak >= 2 and not self.warned


@dataclass(slots=True)
class FocusGuard:
    """One task at a time: `surfaced_task_id` is the subtask last injected as
    the focus banner (re-surfaced on a focus change or after a tier-2
    restart, which resets it); the anti-grind counter holds the focus task,
    how many consecutive turns it has held (not reset by compaction, only by
    forward motion) and how many stuck nudges fired for it."""

    surfaced_task_id: str | None = None
    last_focus_id: str | None = None
    turns_on_task: int = 0
    stuck_nudges_fired: int = 0

    def clear(self) -> None:
        """The frontier is empty: nothing to grind on."""
        self.turns_on_task = 0
        self.last_focus_id = None

    def note(self, current_id: str, *, standing: bool) -> bool:
        """Count a turn on *current_id*; True when the stuck nudge fires (every
        `STUCK_ON_TASK_AFTER` turns, `STUCK_NUDGE_MAX` times per task, never
        for a standing task)."""
        if current_id != self.last_focus_id:
            self.turns_on_task = 0
            self.last_focus_id = current_id
            self.stuck_nudges_fired = 0
            return False
        self.turns_on_task += 1
        if (
            self.turns_on_task % STUCK_ON_TASK_AFTER == 0
            and self.stuck_nudges_fired < STUCK_NUDGE_MAX
            and not standing
        ):
            self.stuck_nudges_fired += 1
            return True
        return False


@dataclass(slots=True)
class BudgetNudges:
    """The one-shot finish directives a low budget or too many turns draw: a
    plan's, and a non-metric run's."""

    plan_finish: bool = False
    run_budget: bool = False


# The advisors that run once a turn's tools have run, in the order their
# notices reach the model.
AFTER_TOOLS: tuple[Advisor, ...] = (memory_flip, loop_guard_notice, stagnation, no_progress)
