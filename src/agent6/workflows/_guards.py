# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop's guards: each heuristic that nudges or ends a run owns its
counters and its threshold rule here, in one small object the loop holds on
`LoopState`. The loop does the reading and the writing (the notice, the
event, the log line, the stop); the guard decides when.

Leg-local by design, like every counter not named in `SessionSnapshot`: a
resume is operator-initiated, so a resumed leg's refreshed patience is the
operator granting another window. The completion-relevant subset persists
(`restore_completion_state`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from agent6.workflows._dag_focus import STUCK_NUDGE_MAX, STUCK_ON_TASK_AFTER
from agent6.workflows._metric import MetricSample
from agent6.workflows._nudges import (
    NO_PROGRESS_ESCALATE_AFTER,
    NO_PROGRESS_NUDGE_AFTER,
    NO_PROGRESS_STOP_AFTER,
    VERIFY_SETTLED_NUDGE_AFTER,
    VERIFY_SETTLED_STOP_AFTER,
)

Rung = Literal["nudge", "escalate", "stop"]


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


@dataclass(slots=True)
class MemoryNudges:
    """The two memory write nudges (run mode, a memory store wired): one flip
    advisory when verify first goes green after failing, one deferred
    finish_session as the backstop; both silent once the worker recorded
    anything. Run-lifetime: all three persist in the snapshot."""

    written: bool = False
    flip_nudged: bool = False
    finish_nudged: bool = False


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
class FinishGates:
    """The finish gates' counters: the open-task refusals sent
    (`TASK_FINISH_PATIENCE` caps them), the red finish certifications
    returned (`verify_retries` caps them), the before-finish panel's
    consecutive rejections (its cap lets the end through) and its run-total
    (persisted; past `max_total_rejections` the gate disarms to advisory)."""

    task_nudges_used: int = 0
    verify_retries_used: int = 0
    review_consecutive: int = 0
    review_total: int = 0


@dataclass(slots=True)
class BudgetNudges:
    """The one-shot finish directives a low budget or too many turns draw: a
    plan's, and a non-metric run's."""

    plan_finish: bool = False
    run_budget: bool = False
