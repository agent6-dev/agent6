# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop's guards: one advisor function per heuristic that nudges or ends a run.

Each reads its counters from the guard object the loop holds on `LoopState`
and answers with a shape from `advice`. The loop runs `BEFORE_CALL` ahead of
the provider call and `AFTER_TOOLS` once a turn's tools have run, in order.

Counters not named in `SessionSnapshot` are execution-local: a resume is
operator-initiated, so refreshed patience is the operator granting another
window. The completion-relevant subset persists (`restore_completion_state`).
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from typing import TYPE_CHECKING, Any, Literal

from agent6.graph import models, order
from agent6.harness import advice, dag_focus, finish_gates, loop_snapshot, metric_rules, nudges
from agent6.sessions import journal
from agent6.tools import results

if TYPE_CHECKING:
    from agent6.harness import loop_state

Rung = Literal["nudge", "escalate", "stop"]


@dataclasses.dataclass(slots=True)
class Ladder:
    """A nudge, escalate, stop ladder over a streak.

    Attributes:
        nudge_after: The streak the first nudge goes out at.
        escalate_after: The streak the escalation goes out at, once the nudge did.
        stop_after: The streak the stop comes at, once both did.
        used: The rungs that went out; a new stuck point re-arms it.
    """

    nudge_after: int
    escalate_after: int
    stop_after: int
    used: int = 0

    def climb(self, streak: int) -> Rung | None:
        """Return the rung the streak reaches now, and count it as sent.

        Args:
            streak: The current streak.

        Returns:
            The rung, or None between rungs.
        """
        if streak >= self.stop_after and self.used >= 2:
            return "stop"
        if streak >= self.escalate_after and self.used == 1:
            self.used = 2
            return "escalate"
        if streak >= self.nudge_after and self.used == 0:
            self.used = 1
            return "nudge"
        return None

    def rearm(self) -> None:
        """Start the ladder over for a new stuck point."""
        self.used = 0

    @staticmethod
    def level(rung: Rung) -> int:
        """Return the level a nudge event carries.

        Args:
            rung: The rung sent.

        Returns:
            1 for the nudge, 2 for the escalation.
        """
        return 2 if rung == "escalate" else 1


def no_progress_ladder() -> Ladder:
    """Return the ladder over a plain run's streak of identical verify failures.

    Returns:
        The ladder at the `nudges` thresholds.
    """
    return Ladder(
        nudges.NO_PROGRESS_NUDGE_AFTER,
        nudges.NO_PROGRESS_ESCALATE_AFTER,
        nudges.NO_PROGRESS_STOP_AFTER,
    )


def no_progress(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Nudge | advice.Stop | None:
    """Nudge, escalate, then stop on a plain run's streak of identical verify failures.

    Judged on a turn whose verify failed. A metric run is left to its plateau,
    ceiling and early-finish rules: a failing verify is expected while it
    searches for an optimisation.

    Args:
        turn: The turn whose tools ran.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The rung's nudge or stop, or None.
    """
    if ctx.mode != "run" or ctx.metric or not turn.verify_just_failed:
        return None
    streak = state.verify.fail_streak
    rung = state.no_progress.climb(streak)
    if rung is None:
        return None
    if rung == "stop":
        end = loop_snapshot.End(
            "no_progress",
            f"stopped: the same verify failure persisted through {streak} consecutive"
            " runs despite two harness interventions; resume with a new approach or"
            " a bigger budget",
        )
        return advice.Stop(
            lambda: end,
            soft="no_progress",
            log=f"LOOP: no_progress stop at turn {turn.number} (streak {streak})",
        )
    return advice.Nudge(
        nudges.NO_PROGRESS_ESCALATION if rung == "escalate" else nudges.NO_PROGRESS_NUDGE,
        event=journal.LoopNoProgressNudge(
            turn=turn.number, streak=streak, level=Ladder.level(rung)
        ),
    )


@dataclasses.dataclass(slots=True)
class SettledGuard:
    """The verify-settled completion's counters.

    Once the run has a green verify or an editing step, idle turns count:
    one nudge at `VERIFY_SETTLED_NUDGE_AFTER`,
    the stop at `harness.settled_stop_after` unless disabled.

    Attributes:
        tree: The last tree sha seen; execution-local, so a resumed execution
            re-measures on its first turn.
        head: The last HEAD seen; a commit the model makes under its own git control
            moves it without touching the tree.
        idle: Consecutive turns with no edit, no commit, no memory write, an unchanged
            tree and an unchanged HEAD.
        nudged: The one settled nudge went out.
        gateless_ever_edited: An editing step did not end with a green verify. Persisted.
    """

    tree: str = ""
    head: str = ""
    idle: int = 0
    nudged: bool = False
    gateless_ever_edited: bool = False

    def note_turn(self, tree: str, head: str, *, progress: bool, neutral: bool) -> None:
        """Count the turn: progress restarts the streak, a neutral turn holds it.

        Args:
            tree: The tree sha after the turn.
            head: The HEAD sha after the turn.
            progress: An edit, a commit, a memory write, a tree change or a HEAD move
                happened.
            neutral: A verify ran, or the worker waited on a command of its own still
                running.
        """
        self.tree = tree
        self.head = head
        if progress:
            self.restart()
        elif not neutral:
            self.idle += 1

    def restart(self) -> None:
        """Start a fresh idle streak; the nudge may go out again."""
        self.idle = 0
        self.nudged = False

    def nudge_due(self) -> bool:
        """Return whether the one settled nudge goes out now, and count it as sent.

        Returns:
            True once per streak, at `VERIFY_SETTLED_NUDGE_AFTER` idle turns.
        """
        if self.idle < nudges.VERIFY_SETTLED_NUDGE_AFTER or self.nudged:
            return False
        self.nudged = True
        return True


def settled_reason(state: loop_state.LoopState, ctx: advice.TurnContext) -> str:
    """Return why a settled end is not a pass.

    Args:
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The reason, in the receipt's words.
    """
    if state.verify.last_ok is False:
        return "the worker settled, but the verify gate is still red"
    if state.verify.green_and_untouched and state.verify.baseline_ok:
        return (
            "the worker settled with the tree as it was at the start; the only green was the base's"
        )
    if state.verify.ever_passed:
        return "the worker settled, but edits after the last green verify were never re-verified"
    if not ctx.verify_command():
        return "the worker settled; no verify command existed to gate it"
    if not ctx.gate_present():
        return (
            "the worker settled; the verify command could not run"
            " (commands withheld, or the gate denied)"
        )
    return "the worker settled; the verify never passed"


def settled_end(state: loop_state.LoopState, ctx: advice.TurnContext) -> loop_snapshot.End:
    """Return the settled stop's end, grounded on the tree.

    Args:
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        `verify_settled` when a green verify covers the tree as it stands;
        else `settled` with the reason none does. Both name the open subtasks.
    """
    if state.verify.ever_passed and ctx.tree_green() is not False:
        return loop_snapshot.End(
            "verify_settled",
            advice.with_open_tasks(
                "verify passed and the worker stopped making changes", ctx.open_subtasks()
            ),
            completed=True,
            verdict="passed",
            scoped=state.verify.scoped,
        )
    return loop_snapshot.End(
        "settled",
        advice.with_open_tasks(settled_reason(state, ctx), ctx.open_subtasks()),
        completed=True,
        roots=True,
    )


def verify_settled(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Nudge | advice.Stop | None:
    """Nudge once, then stop, a plain run that settled after a good state.

    The stop is an ending the end gates judge and a standing task may absorb.
    A finish call this turn disarms both. A verify run and a wait on a command
    the worker started are neutral. A metric run's end belongs to its plateau
    and ceiling rules, and its read-only turns are work.

    Args:
        turn: The turn whose tools ran.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The nudge or the stop, or None.
    """
    if ctx.mode != "run" or ctx.metric:
        return None
    guard = state.settled
    if not (state.verify.ever_passed or guard.gateless_ever_edited):
        return None
    tree, head = ctx.tree_sha(), ctx.head_sha()
    guard.note_turn(
        tree,
        head,
        progress=turn.committed
        or turn.edited
        or turn.wrote_memory
        or tree != guard.tree
        or head != guard.head,
        neutral=turn.verify_just_passed or turn.verify_just_failed or turn.waited_on_background,
    )
    if turn.finish is not None:
        return None
    stop_after = ctx.settled_stop_after
    if stop_after > 0 and guard.idle >= stop_after:
        return advice.Stop(
            lambda: settled_end(state, ctx),
            soft="verify_settled",
            declared="settled",
            log=f"LOOP: verify_settled at turn {turn.number} (idle {guard.idle})",
        )
    # The nudge promises that a finish ends the run; while a red gate that is the model's to
    # fix would send a finish back, it stays unspent for a green later in the streak.
    refused = state.verify.last_ok is False and finish_gates.red_gate_returns(
        ctx.verify_when,
        ctx.verify_retries,
        state.verify,
        state.gates.verify_retries_used,
        gate_present=ctx.gate_present(),
    )
    if not refused and guard.nudge_due():
        return advice.Nudge(
            nudges.verify_settled_nudge(stop_after),
            event=journal.LoopVerifySettledNudge(turn=turn.number, idle=guard.idle),
        )
    return None


def _now() -> float:
    """Return the monotonic clock, read through the module so a test can drive it."""
    return time.monotonic()


@dataclasses.dataclass(slots=True)
class StagnationGuard:
    """The stagnation notice's counters; a resumed run gets a fresh window.

    Attributes:
        started_monotonic: The execution's start on the monotonic clock.
        wait_at_start: The operator time the dispatcher had counted when the window opened,
            the start gate's approval included; only the time waited since is the model's.
        harness_gate_s: The seconds the harness's own gate runs took inside the window; its
            time, not the model's.
        nudged: The one notice went out.
    """

    started_monotonic: float = dataclasses.field(default_factory=_now)
    wait_at_start: float = 0.0
    harness_gate_s: float = 0.0
    nudged: bool = False


def tool_error_ladder(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Nudge | advice.Stop | None:
    """Nudge, escalate, then stop on a streak of identical tool errors.

    Judged after each failed call. A plain run's guard: a metric run's own
    machinery owns its end. A denial streak is a policy outcome, so its nudge
    says refused, not malformed.

    Args:
        turn: The turn being dispatched.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The rung's nudge or stop, or None.
    """
    if ctx.mode != "run" or ctx.metric:
        return None
    streak = state.spiral.error_streak
    rung = state.spiral.climb_error()
    if rung is None:
        return None
    if rung == "stop":
        end = loop_snapshot.End(
            "tool_error_stuck",
            f"stopped: the same tool call failed {streak} times with the identical"
            " error despite two harness interventions; resume with a different"
            " approach",
        )
        return advice.Stop(
            lambda: end, log=f"LOOP: tool_error stop at turn {turn.number} (streak {streak})"
        )
    if state.spiral.last_error_was_denial:
        text = nudges.TOOL_DENIED_NUDGE
    else:
        text = nudges.TOOL_ERROR_ESCALATION if rung == "escalate" else nudges.TOOL_ERROR_NUDGE
    return advice.Nudge(
        text,
        event=journal.LoopToolErrorNudge(turn=turn.number, streak=streak, level=Ladder.level(rung)),
    )


def loop_guard_notice(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Nudge | None:
    """Notice the same (tool, args) call `LOOP_GUARD_NOTICE_AFTER` times in a row.

    Re-armed after one quiet turn, so an unbroken streak hears it every
    other turn until the kill threshold ends the run.

    Args:
        turn: The turn whose tools ran.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The notice, or None.
    """
    spiral = state.spiral
    if not (
        spiral.call_streak >= nudges.LOOP_GUARD_NOTICE_AFTER
        and spiral.warned_at_turn < turn.number - 1
    ):
        return None
    spiral.warned_at_turn = turn.number
    tool = (spiral.last_call_sig or "").split(":", 1)[0] or "<unknown>"
    return advice.Nudge(
        nudges.loop_guard_words(tool, spiral.call_streak),
        event=journal.LoopLoopGuardTriggered(
            turn=turn.number, tool=tool, streak=spiral.call_streak
        ),
        log=(
            f"  loop guard: {tool} called {spiral.call_streak}x in a row;"
            " the model is told to change course"
        ),
    )


def loop_guard_kill(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Stop | None:
    """End the run on the same (tool, args) call `loop_guard_kill_threshold` times in a row.

    A threshold of 0 leaves the notice alone. Observed last, so a turn's other
    stops outrank it.

    Args:
        turn: The turn whose tools ran.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The stop, or None.
    """
    threshold = ctx.loop_guard_kill_threshold
    streak = state.spiral.call_streak
    if not (threshold > 0 and streak >= threshold):
        return None
    tool = (state.spiral.last_call_sig or "").split(":", 1)[0] or "<unknown>"
    end = loop_snapshot.End(
        "loop_guard_killed",
        f"loop-guard killed run: `{tool}` called {streak}x in a row with identical"
        f" arguments (threshold {threshold})",
        fields={"tool": tool, "streak": streak},
    )
    return advice.Stop(
        lambda: end,
        log=(
            f"LOOP: loop_guard_killed at turn {turn.number} - {tool} called {streak}x"
            f" in a row (threshold={threshold})"
        ),
    )


def stagnation(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Nudge | None:
    """Notice once when `stagnation_notice_after_s` passed with no edit and no verify.

    Time blocked on the operator, and time the harness's own gate runs take, is not the
    model's; a gateless run's notice names no gate. The start gate's baseline is no verify of
    the run's own.

    Args:
        turn: The turn whose tools ran.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The notice, or None.
    """
    guard = state.stagnation
    after = ctx.stagnation_notice_after_s
    if not (
        after > 0
        and ctx.mode == "run"
        and not guard.nudged
        and not state.ever_edited
        and not (state.verify.ever_passed or state.verify.ever_failed)
    ):
        return None
    elapsed = _now() - guard.started_monotonic - guard.harness_gate_s
    if elapsed >= after:
        elapsed -= ctx.operator_wait_s() - guard.wait_at_start
    if elapsed < after:
        return None
    guard.nudged = True
    minutes = max(1, int(elapsed // 60))
    text = nudges.STAGNATION_NUDGE if ctx.gate_present() else nudges.STAGNATION_NUDGE_GATELESS
    return advice.Nudge(
        text.format(minutes=minutes),
        event=journal.LoopStagnationNudged(turn=turn.number, elapsed_s=int(elapsed)),
        log=(
            f"  stagnation: {minutes}m with no edit and no verify;"
            " the model is reminded the budget is finite"
        ),
    )


@dataclasses.dataclass(slots=True)
class MemoryState:
    """The run's memory bookkeeping.

    The flip advisory goes silent once the worker recorded anything.

    Attributes:
        written: The worker edited anything under the store. Persisted.
        flip_nudged: The flip advisory went out. Persisted.
        wrote: The facts this execution wrote, for the use record.
        created: The facts an edit tool made rather than changed.
        deleted: The facts this execution deleted.
        read: Reads per fact this execution.
    """

    written: bool = False
    flip_nudged: bool = False
    wrote: list[str] = dataclasses.field(default_factory=list)
    created: list[str] = dataclasses.field(default_factory=list)
    deleted: list[str] = dataclasses.field(default_factory=list)
    read: dict[str, int] = dataclasses.field(default_factory=dict)

    def note_write(self, fact: str, op: str) -> None:
        """Fold one write of a fact.

        A delete ends the fact for this execution, its reads so far with it, so
        its entry goes at the execution's end as `memory remove` drops it; a create
        after that starts it afresh.

        Args:
            fact: The fact's name.
            op: `create`, `edit` or `delete`.
        """
        if op == "delete":
            for names in (self.wrote, self.created):
                if fact in names:
                    names.remove(fact)
            self.read.pop(fact, None)  # its old life's reads go with it
            if fact not in self.deleted:
                self.deleted.append(fact)
            return
        if fact not in self.wrote:
            self.wrote.append(fact)
        if op == "create" and fact not in self.created:
            self.created.append(fact)


def memory_flip(
    turn: loop_state.TurnState, state: loop_state.LoopState, ctx: advice.TurnContext
) -> advice.Nudge | None:
    """Advise once per run, at the first red-to-green verify with no memory write.

    That is the moment a hard-won root cause is in hand; `nudges` holds the
    measurement behind it. The red has to be one of the run's own verifies: a gate that
    was red before the first turn is the baseline, and every fix would flip it.

    Args:
        turn: The turn whose tools ran.
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The advisory, or None.
    """
    if not (
        turn.verify_flipped_green
        and state.verify.ever_failed
        and ctx.mode == "run"
        and ctx.memory_advisory
        and not state.memory.written
        and not state.memory.flip_nudged
    ):
        return None
    state.memory.flip_nudged = True
    return advice.Nudge(
        nudges.MEMORY_FLIP_NUDGE,
        event=journal.LoopMemoryFlipNudged(turn=turn.number),
        log=(
            "  memory: verify flipped green with nothing recorded;"
            " the model is asked to write the rule"
        ),
    )


@dataclasses.dataclass(slots=True)
class StandingGoal:
    """The standing goal's re-entry counters. Both persist.

    `[harness].standing_patience` decides how many fruitless re-entries are
    absorbed before an end is honoured (-1 never honours one on its own).

    Attributes:
        tools_mark: `ok_tool_calls` at the last absorption; -1 for never.
        fruitless: Consecutive re-entries since work last landed.
    """

    tools_mark: int = -1
    fruitless: int = 0


@dataclasses.dataclass(slots=True)
class ReachabilityGuard:
    """The sandbox reachability counters.

    Only executed commands feed it; a validation error or a denial never entered
    the jail.

    Attributes:
        binary: argv[0] of the run_command the jail failed to exec.
        streak: Its consecutive exec failures.
        warned: The notice went out.
    """

    binary: str = ""
    streak: int = 0
    warned: bool = False

    def note(self, binary: str, *, exec_failed: bool) -> bool:
        """Record one command.

        Args:
            binary: The command's argv[0].
            exec_failed: The jail failed to exec it, as opposed to a nonzero exit.

        Returns:
            True at the second consecutive exec failure, once; the caller still
            checks the host has the binary.
        """
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
    state: loop_state.LoopState, name: str, tool_input: Any, result: results.ToolResult
) -> advice.Nudge | None:
    """Note a binary present on the host but broken in the jail, after each call.

    The signal is a run_command the jail failed to exec (a nonzero exit is the
    command's own result) for a binary `shutil.which` finds on the host: the
    second consecutive failure tells the model once and emits the event the
    session's finalize reads for its operator warning.

    Args:
        state: The run's loop state.
        name: The tool's name.
        tool_input: The model's input to the tool.
        result: The tool's result.

    Returns:
        The notice, or None.
    """
    if name != "run_command" or not isinstance(result, results.ExecResult):
        return None
    argv = tool_input.get("argv") if isinstance(tool_input, dict) else None
    binary = str(argv[0]) if isinstance(argv, list) and argv else ""
    if not state.reach.note(binary, exec_failed=result.exec_failed):
        return None
    if shutil.which(binary) is None:
        return None
    state.reach.warned = True
    return advice.Nudge(
        nudges.unreachable_tool_notice(binary),
        event=journal.LoopSandboxToolUnreachable(binary=binary),
        log=f"LOOP: sandbox tool unreachable: {binary} exists on host, fails in jail",
    )


@dataclasses.dataclass(slots=True)
class FocusGuard:
    """The focus banner's and the stuck-on-task nudge's counters.

    Attributes:
        surfaced_task_id: The subtask last injected as the focus banner; a tier-2
            restart resets it.
        last_focus_id: The focus task the turn count is on.
        turns_on_task: Consecutive turns on it; only forward motion resets it.
        stuck_nudges_fired: Stuck nudges fired for it.
    """

    surfaced_task_id: str | None = None
    last_focus_id: str | None = None
    turns_on_task: int = 0
    stuck_nudges_fired: int = 0

    def clear(self) -> None:
        """Reset the count: the frontier is empty."""
        self.turns_on_task = 0
        self.last_focus_id = None

    def note(self, current_id: str, *, standing: bool, progressed: bool = True) -> bool:
        """Count a turn on the current task.

        A focus change only resets the count when something was concluded, so a
        worker that moves from task to task without finishing one is caught like
        one that grinds on a single task.

        Args:
            current_id: The current task's id.
            standing: The task is a standing task, which never draws the nudge.
            progressed: The previous focus was concluded.

        Returns:
            True when the stuck nudge fires: every `STUCK_ON_TASK_AFTER` turns,
            `STUCK_NUDGE_MAX` times per task.
        """
        if current_id != self.last_focus_id:
            self.last_focus_id = current_id
            if progressed:
                self.turns_on_task = 0
                self.stuck_nudges_fired = 0
                return False
        self.turns_on_task += 1
        if (
            self.turns_on_task % dag_focus.STUCK_ON_TASK_AFTER == 0
            and self.stuck_nudges_fired < dag_focus.STUCK_NUDGE_MAX
            and not standing
        ):
            self.stuck_nudges_fired += 1
            return True
        return False


def stuck_on_task(
    state: loop_state.LoopState,
    current_id: str,
    node: models.TaskNode,
    nodes: dict[str, models.TaskNode],
) -> advice.Nudge | None:
    """Offer to split, pass or skip a task the worker has ground on too long.

    Judged each turn the focus phase names a current task. A task marked done or
    decomposed resets the count; compaction and a claim of another open task do
    not. A standing task is exempt: it never concludes.

    Args:
        state: The run's loop state.
        current_id: The current task's id.
        node: The current task.
        nodes: The task graph's nodes by id.

    Returns:
        The nudge, or None.
    """
    focus = state.focus
    # A previous focus still ready means the worker claimed another task instead of
    # concluding this one.
    previous = nodes.get(focus.last_focus_id or "")
    progressed = previous is None or not order.is_focusable_subtask(nodes, previous)
    if not focus.note(current_id, standing=node.standing, progressed=progressed):
        return None
    return advice.Nudge(
        dag_focus.stuck_on_task_nudge(current_id, node, focus.turns_on_task),
        event=journal.LoopTaskStuckNudge(
            task_id=current_id, turns=focus.turns_on_task, nudges_used=focus.stuck_nudges_fired
        ),
        log=(
            f"LOOP: stuck-on-task nudge #{focus.stuck_nudges_fired} for {current_id}"
            f" after {focus.turns_on_task} turns"
        ),
    )


@dataclasses.dataclass(slots=True)
class BudgetNudges:
    """Which one-shot budget directives went out.

    Attributes:
        plan_finish: The planner's finish directive.
        run_budget: A plain run's wrap-up directive.
    """

    plan_finish: bool = False
    run_budget: bool = False


def plan_budget_nudge(state: loop_state.LoopState, ctx: advice.TurnContext) -> advice.Nudge | None:
    """Direct the planner once to finish on a low budget or too many turns.

    A planner takes many cheap cached turns, so the `PLAN_NUDGE_AFTER_TURNS`
    turn count is the lever that reaches the reads-forever case.

    Args:
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The directive, or None.
    """
    if ctx.mode != "plan" or state.budget_nudges.plan_finish:
        return None
    remaining = ctx.budget_remaining()
    low_budget = remaining is not None and remaining <= nudges.PLAN_BUDGET_NUDGE_BELOW
    too_many_turns = ctx.turn - ctx.execution_start + 1 >= nudges.PLAN_NUDGE_AFTER_TURNS
    if not (low_budget or too_many_turns):
        return None
    state.budget_nudges.plan_finish = True
    return advice.Nudge(
        nudges.PLAN_BUDGET_NUDGE,
        event=journal.LoopPlanFinishNudge(turn=ctx.turn, budget_remaining=remaining),
        log=(
            f"LOOP: plan finish-nudge at turn {ctx.turn}"
            f" (turns={too_many_turns}, low_budget={low_budget})"
        ),
    )


def run_budget_nudge(state: loop_state.LoopState, ctx: advice.TurnContext) -> advice.Nudge | None:
    """Direct a plain run once to verify and finish before a cap ends it.

    Fires once the budget fraction falls to `RUN_BUDGET_NUDGE_BELOW`; gateless,
    finish alone. A metric run has its own end-game.

    Args:
        state: The run's loop state.
        ctx: The turn's facts.

    Returns:
        The directive, or None.
    """
    if ctx.mode != "run" or ctx.metric or state.budget_nudges.run_budget:
        return None
    remaining = ctx.budget_remaining()
    if remaining is None or remaining > nudges.RUN_BUDGET_NUDGE_BELOW:
        return None
    state.budget_nudges.run_budget = True
    return advice.Nudge(
        nudges.RUN_BUDGET_NUDGE if ctx.gate_present() else nudges.RUN_BUDGET_NUDGE_GATELESS,
        event=journal.LoopRunBudgetNudge(turn=ctx.turn, budget_remaining=remaining),
        log=f"LOOP: run budget-nudge at turn {ctx.turn}",
    )


# The order is the order the notices reach the model and the precedence among the
# stops; a stop the tool-error ladder raised during dispatch outranks them all.
AFTER_TOOLS: tuple[advice.Advisor, ...] = (
    memory_flip,
    loop_guard_notice,
    stagnation,
    metric_rules.metric_plateau,
    verify_settled,
    no_progress,
    loop_guard_kill,
)

# The advisors that speak before the provider call, in order.
BEFORE_CALL: tuple[advice.BeforeCallAdvisor, ...] = (plan_budget_nudge, run_budget_nudge)
