# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop's guards: each heuristic that nudges or ends a run is one
advisor function here (the shapes it answers with are in `_advice`), reading
its counters from the guard object the loop holds on `LoopState`. The loop
runs `BEFORE_CALL` ahead of the provider call and `AFTER_TOOLS` once a turn's
tools have run, in order, and applies each answer.

Leg-local by design, like every counter not named in `SessionSnapshot`: a
resume is operator-initiated, so a resumed leg's refreshed patience is the
operator granting another window. The completion-relevant subset persists
(`restore_completion_state`).
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent6.graph.models import TaskNode
from agent6.graph.order import is_focusable_subtask
from agent6.tools.results import ExecResult, ToolResult
from agent6.workflows._advice import (
    Advisor,
    BeforeCallAdvisor,
    Nudge,
    Stop,
    TurnContext,
    with_open_tasks,
)
from agent6.workflows._dag_focus import (
    STUCK_NUDGE_MAX,
    STUCK_ON_TASK_AFTER,
    stuck_on_task_nudge,
)
from agent6.workflows._metric import metric_plateau
from agent6.workflows._nudges import (
    LOOP_GUARD_NOTICE_AFTER,
    MEMORY_FLIP_NUDGE,
    NO_PROGRESS_ESCALATE_AFTER,
    NO_PROGRESS_ESCALATION,
    NO_PROGRESS_NUDGE,
    NO_PROGRESS_NUDGE_AFTER,
    NO_PROGRESS_STOP_AFTER,
    PLAN_BUDGET_NUDGE,
    PLAN_BUDGET_NUDGE_BELOW,
    PLAN_NUDGE_AFTER_ITERS,
    RUN_BUDGET_NUDGE,
    RUN_BUDGET_NUDGE_BELOW,
    RUN_BUDGET_NUDGE_GATELESS,
    STAGNATION_NUDGE,
    STAGNATION_NUDGE_GATELESS,
    TOOL_DENIED_NUDGE,
    TOOL_ERROR_ESCALATION,
    TOOL_ERROR_NUDGE,
    VERIFY_SETTLED_NUDGE,
    VERIFY_SETTLED_NUDGE_AFTER,
    VERIFY_SETTLED_STOP_AFTER,
    loop_guard_words,
    unreachable_tool_notice,
)
from agent6.workflows._session_state import End

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState, TurnState

Rung = Literal["nudge", "escalate", "stop"]


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
        end = End(
            "no_progress",
            f"stopped: the same verify failure persisted through {streak} consecutive"
            " runs despite two harness interventions; resume with a new approach or"
            " a bigger budget",
        )
        return Stop(
            lambda: end,
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


def settled_reason(state: LoopState, ctx: TurnContext) -> str:
    """Why a settled end is not a pass."""
    if state.verify.last_ok is False:
        return "the worker settled, but the verify gate is still red"
    if state.verify.ever_passed:
        return "the worker settled, but edits after the last green verify were never re-verified"
    if not ctx.verify_command():
        return "the worker settled after committing work; no verify command existed to gate it"
    if not ctx.gate_present():
        return (
            "the worker settled after committing work; the verify command could not run"
            " (commands withheld, or the gate denied)"
        )
    return "the worker settled after committing work; the verify never passed"


def settled_end(state: LoopState, ctx: TurnContext) -> End:
    """The settled stop's end, grounded on the tree: `verify_settled` (a
    pass) only when a green verify covers the tree as it stands, else
    `settled` with the reason it is not a pass; both name the open subtasks."""
    if state.verify.ever_passed and ctx.tree_green() is not False:
        return End(
            "verify_settled",
            with_open_tasks(
                "verify passed and the worker stopped making changes", ctx.open_subtasks()
            ),
            completed=True,
            verdict="passed",
            scoped=state.verify.scoped,
        )
    return End(
        "settled",
        with_open_tasks(settled_reason(state, ctx), ctx.open_subtasks()),
        completed=True,
        roots=True,
    )


def verify_settled(turn: TurnState, state: LoopState, ctx: TurnContext) -> Nudge | Stop | None:
    """The settled end of a plain run (`SettledGuard`): once a green verify
    (or, gateless, an editing step) seeded it, each turn with no edit, no
    commit and an unchanged tree counts as idle (a verify run is neutral):
    one nudge, then the stop, an ending the end gates judge and a standing
    task may absorb. A finish call this turn disarms both. A metric run's
    end belongs to its plateau and ceiling rules, and its read-only turns
    are work."""
    if ctx.mode != "run" or ctx.metric:
        return None
    guard = state.settled
    if not (state.verify.ever_passed or guard.gateless_ever_edited):
        return None
    tree = ctx.tree_sha()
    guard.note_turn(
        tree,
        progress=turn.committed or turn.edited or tree != guard.tree,
        verify_ran=turn.verify_just_passed or turn.verify_just_failed,
    )
    if turn.finish_signal is not None:
        return None
    if guard.stop_due():
        return Stop(
            lambda: settled_end(state, ctx),
            soft="verify_settled",
            declared="settled",
            log=f"LOOP: verify_settled at iter {turn.iteration} (idle {guard.idle})",
        )
    if guard.nudge_due():
        return Nudge(
            VERIFY_SETTLED_NUDGE,
            event="loop.verify_settled.nudge",
            fields={"iteration": turn.iteration, "idle": guard.idle},
        )
    return None


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
        end = End(
            "tool_error_stuck",
            f"stopped: the same tool call failed {streak} times with the identical"
            " error despite two harness interventions; resume with a different"
            " approach",
        )
        return Stop(
            lambda: end, log=f"LOOP: tool_error stop at iter {turn.iteration} (streak {streak})"
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


def loop_guard_kill(turn: TurnState, state: LoopState, ctx: TurnContext) -> Stop | None:
    """The same (tool, args) call `loop_guard_kill_threshold` times in a row
    ends the run: the notice was advisory, and a worker still circling would
    spend the rest of the budget on it. 0 leaves the notice alone. Observed
    last, so a turn's other stops outrank it."""
    threshold = ctx.guards.loop_guard_kill_threshold
    streak = state.spiral.call_streak
    if not (threshold > 0 and streak >= threshold):
        return None
    tool = (state.spiral.last_call_sig or "").split(":", 1)[0] or "<unknown>"
    end = End(
        "loop_guard_killed",
        f"loop-guard killed run: `{tool}` called {streak}x in a row with identical"
        f" arguments (threshold {threshold})",
        fields={"tool": tool, "streak": streak},
    )
    return Stop(
        lambda: end,
        log=(
            f"LOOP: loop_guard_killed at iter {turn.iteration} - {tool} called {streak}x"
            f" in a row (threshold={threshold})"
        ),
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


def unreachable_tool(
    state: LoopState, name: str, tool_input: Any, result: ToolResult
) -> Nudge | None:
    """The sandbox-reachability note, judged after each executed call. The
    one true "present on the host, broken in the jail" signal is a
    run_command the jail failed to EXEC (`exec_failed`; a nonzero exit is the
    command's own result) for a binary `shutil.which` finds on the host: the
    second consecutive failure of the same binary tells the model once and
    emits the event the session's finalize reads for its operator warning."""
    if name != "run_command" or not isinstance(result, ExecResult):
        return None
    argv = tool_input.get("argv") if isinstance(tool_input, dict) else None
    binary = str(argv[0]) if isinstance(argv, list) and argv else ""
    if not state.reach.note(binary, exec_failed=result.exec_failed):
        return None
    if shutil.which(binary) is None:
        return None
    state.reach.warned = True
    return Nudge(
        unreachable_tool_notice(binary),
        event="loop.sandbox_tool_unreachable",
        fields={"binary": binary},
        log=f"LOOP: sandbox tool unreachable: {binary} exists on host, fails in jail",
    )


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

    def note(self, current_id: str, *, standing: bool, progressed: bool = True) -> bool:
        """Count a turn on *current_id*; True when the stuck nudge fires (every
        `STUCK_ON_TASK_AFTER` turns, `STUCK_NUDGE_MAX` times per task, never
        for a standing task).

        A focus change only resets the count when something was concluded.
        Switching away from a task that is still open (the worker claiming
        another with update_task) keeps counting, so a worker that moves from
        task to task without finishing one is caught like one that grinds on a
        single task."""
        if current_id != self.last_focus_id:
            self.last_focus_id = current_id
            if progressed:
                self.turns_on_task = 0
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


def stuck_on_task(
    state: LoopState, current_id: str, node: TaskNode, nodes: dict[str, TaskNode]
) -> Nudge | None:
    """The anti-grind nudge, judged each turn the focus phase names a current
    task: every `STUCK_ON_TASK_AFTER` turns with nothing concluded (a task
    marked done or decomposed resets the count; compaction and a claim of
    another open task do not), up to `STUCK_NUDGE_MAX` times per task, it
    offers to split, pass or skip the task. A standing task is exempt: it
    never concludes."""
    focus = state.focus
    # Forward motion is the previous focus ceasing to be workable: passed,
    # retired, or decomposed into children that are the work now. A previous
    # focus still sitting there ready means the worker claimed another task
    # instead of concluding this one, which the counter keeps counting.
    previous = nodes.get(focus.last_focus_id or "")
    progressed = previous is None or not is_focusable_subtask(nodes, previous)
    if not focus.note(current_id, standing=node.standing, progressed=progressed):
        return None
    return Nudge(
        stuck_on_task_nudge(current_id, node, focus.turns_on_task),
        event="loop.task.stuck_nudge",
        fields={"task_id": current_id, "turns": focus.turns_on_task, "n": focus.stuck_nudges_fired},
        log=(
            f"LOOP: stuck-on-task nudge #{focus.stuck_nudges_fired} for {current_id}"
            f" after {focus.turns_on_task} turns"
        ),
    )


@dataclass(slots=True)
class BudgetNudges:
    """The one-shot finish directives a low budget or too many turns draw: a
    plan's, and a non-metric run's."""

    plan_finish: bool = False
    run_budget: bool = False


def plan_budget_nudge(state: LoopState, ctx: TurnContext) -> Nudge | None:
    """The planner's one-shot finish directive: the budget fraction fell to
    `PLAN_BUDGET_NUDGE_BELOW`, or the leg reached `PLAN_NUDGE_AFTER_ITERS`
    turns without a plan landing (a planner takes many cheap cached turns,
    so the turn count is the lever that reaches the reads-forever case)."""
    if ctx.mode != "plan" or state.budget_nudges.plan_finish:
        return None
    remaining = ctx.budget_remaining()
    low_budget = remaining is not None and remaining <= PLAN_BUDGET_NUDGE_BELOW
    too_many_turns = ctx.iteration - ctx.leg_start + 1 >= PLAN_NUDGE_AFTER_ITERS
    if not (low_budget or too_many_turns):
        return None
    state.budget_nudges.plan_finish = True
    return Nudge(
        PLAN_BUDGET_NUDGE,
        event="loop.plan_finish.nudge",
        fields={"iteration": ctx.iteration, "budget_remaining": remaining},
        log=(
            f"LOOP: plan finish-nudge at iter {ctx.iteration}"
            f" (turns={too_many_turns}, low_budget={low_budget})"
        ),
    )


def run_budget_nudge(state: LoopState, ctx: TurnContext) -> Nudge | None:
    """A plain run's one-shot wrap-up directive once the budget fraction falls
    to `RUN_BUDGET_NUDGE_BELOW`: verify and finish before a cap ends the run
    (gateless, finish alone). A metric run has its own end-game."""
    if ctx.mode != "run" or ctx.metric or state.budget_nudges.run_budget:
        return None
    remaining = ctx.budget_remaining()
    if remaining is None or remaining > RUN_BUDGET_NUDGE_BELOW:
        return None
    state.budget_nudges.run_budget = True
    return Nudge(
        RUN_BUDGET_NUDGE if ctx.gate_present() else RUN_BUDGET_NUDGE_GATELESS,
        event="loop.run_budget.nudge",
        fields={"iteration": ctx.iteration, "budget_remaining": remaining},
        log=f"LOOP: run budget-nudge at iter {ctx.iteration}",
    )


# The advisors that run once a turn's tools have run, in the order their
# notices reach the model.
AFTER_TOOLS: tuple[Advisor, ...] = (
    memory_flip,
    loop_guard_notice,
    stagnation,
    metric_plateau,
    verify_settled,
    no_progress,
    loop_guard_kill,
)

# The advisors that speak before the provider call, in order.
BEFORE_CALL: tuple[BeforeCallAdvisor, ...] = (plan_budget_nudge, run_budget_nudge)
