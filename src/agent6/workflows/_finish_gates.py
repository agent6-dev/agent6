# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The finish gates: what a finish_session must satisfy before the loop
honours it, what an end is called, and the words each refusal carries. The
loop runs the gates in order over a turn that called finish_session and
applies the first `Refusal` (`Workflow._turn_finish_gates`); the ends
the harness declares (settled, plateau, a silent finish) pass the same
rules through `Workflow._end_gates`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent6.graph.models import TaskNode
from agent6.graph.order import OPEN_STATUSES
from agent6.workflows._advice import Refusal, TurnContext
from agent6.workflows._nudges import MEMORY_FINISH_NUDGE, TASK_FINISH_PATIENCE
from agent6.workflows._session_state import SessionEndReason
from agent6.workflows._verify_gate import finish_red_notice
from agent6.workflows._verify_verdict import VerifyVerdict

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState, TurnState


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


# The before-finish panel's rejection, by the ending it rejected; the
# findings follow.
REVIEW_REJECTED = {
    "finish_session": (
        "The review panel rejected your finish_session call. Address the"
        " issues below before calling finish_session again.\n\n"
    ),
    "silent_finish": (
        "The review panel rejected your silent finish (no tool_use, just"
        " text). Address the issues below and continue the task.\n\n"
    ),
    "settled": (
        "The review panel rejected the settled end. Address the issues"
        " below; the run ends when it settles again or finish_session"
        " passes.\n\n"
    ),
    "metric_plateau": (
        "The review panel rejected the end at the metric plateau. Address the"
        " issues below; the run ends when the plateau holds again or"
        " finish_session passes.\n\n"
    ),
}


def open_subtasks(nodes: Mapping[str, TaskNode]) -> list[tuple[str, str]]:
    """The worker's own subtasks still open: `(id, title)` pairs. Only
    SUBTASKS (parent_id is not None) count: the auto-root is pending until
    the run ends, so counting it would deadlock every gate. A standing task
    is not unfinished work: it gates the finish via its own re-entry, never
    via the capped nudge."""
    return [
        (nid, node.title[:120])
        for nid, node in nodes.items()
        if node.parent_id is not None and node.status in OPEN_STATUSES and not node.standing
    ]


def with_open_tasks(summary: str, open_tasks: Sequence[tuple[str, str]]) -> str:
    """*summary* with the open subtasks named, when an end went through over
    them (the gate's cap): the receipt says what was left."""
    if not open_tasks:
        return summary
    titles = ", ".join(title for _tid, title in open_tasks)
    return f"{summary} ({len(open_tasks)} open task(s): {titles})"


def task_finish_nudge(open_tasks: Sequence[tuple[str, str]], gates: FinishGates) -> str | None:
    """The nudge to re-prompt with instead of finishing while the worker's
    own subtasks are open; None lets the end through. Capped by
    `TASK_FINISH_PATIENCE`, as the review gate is: after that many refusals
    the end goes through and its receipt names the open tasks
    (`with_open_tasks`), so a worker that neither closes nor retires a task
    cannot bounce the loop for the whole budget."""
    if not open_tasks:
        return None
    if gates.task_nudges_used >= TASK_FINISH_PATIENCE:
        return None  # cap reached: the end goes through, the receipt names them
    gates.task_nudges_used += 1
    listing = "\n".join(f"- {tid}: {title}" for tid, title in open_tasks)
    return (
        f"[harness] finish_session deferred: {len(open_tasks)} task(s) are"
        f" pending or in_progress:\n{listing}\n"
        "update_task marks one skipped or obsolete; the run ends once the"
        f" list is clear, or on the {TASK_FINISH_PATIENCE + 1}th call."
    )


def red_gate_returns(
    verify_when: Literal["finish", "step", "never"],
    verify_retries: int,
    verify: VerifyVerdict,
    gates: FinishGates,
    *,
    gate_present: bool,
) -> bool:
    """Whether a red gate is the model's to fix, so an end over it goes back:
    a gate exists and is the harness's to run, was not red before the run
    touched anything (or this run has since made it green), was not denied
    or withheld by the operator, and returns are left. One answer for
    finish_session and the ends the harness declares, so neither can hand
    back a gate the model cannot run."""
    return (
        verify_when != "never"
        and gate_present
        and (verify.baseline_ok is not False or verify.ever_passed)
        and gates.verify_retries_used < verify_retries
    )


def contract_refusal(problems: Sequence[str]) -> str:
    """The notice a finish_session whose `result` violates the machine
    state's output_schema returns with."""
    return (
        "finish_session refused: "
        + "; ".join(problems)
        + ". Call finish_session again with a `result` that satisfies the schema."
    )


def finish_reason(
    kind: SessionEndReason, *, stale_gate: str, tree_green: bool | None, verify: VerifyVerdict
) -> SessionEndReason:
    """What a finish is called. `gate_stale` needs a gate that is actually
    RED: green means it passed, and `tree_green` is None on a gateless run,
    where no gate can be stale. `gate_red_at_base` outranks a plain finish
    over red: the gate was failing before this run touched anything, so a
    red end is not this run's failure; only ever from an observation, never
    a guess."""
    if kind == "finish_session" and tree_green is False:
        if stale_gate:
            return "gate_stale"
        if verify.baseline_ok is False and not verify.ever_passed:
            return "gate_red_at_base"
    return kind


def finish_contract(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """A finish_session whose `result` does not satisfy the machine state's
    output_schema returns to the model with the problems, so the retry
    happens in-leg instead of the leg ending failed over correct work.
    Unbounded on purpose: the budget and iteration backstops end a model
    that never conforms, and the engine records that truthfully."""
    if ctx.finish_validator is None:
        return None
    problems = ctx.finish_validator(turn.finish_payload)
    if not problems:
        return None
    return Refusal(
        contract_refusal(problems),
        event="loop.finish_contract.refused",
        fields={"iteration": turn.iteration, "problems": problems},
        log=f"  finish_session returned: result violates the contract at iter {turn.iteration}",
    )


def open_tasks_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """A run's finish_session waits while the worker's own subtasks are open
    (`task_finish_nudge`, capped)."""
    if ctx.mode != "run":
        return None
    nudge = task_finish_nudge(ctx.open_subtasks(), state.gates)
    if nudge is None:
        return None
    return Refusal(
        nudge,
        event="loop.task_finish.gated",
        fields={"iteration": turn.iteration, "nudges_used": state.gates.task_nudges_used},
        log=(
            f"  finish_session gated: open subtasks remain (nudge"
            f" #{state.gates.task_nudges_used}) at iter {turn.iteration}"
        ),
    )


def verify_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """A run's finish over a tree the gate did not certify returns to the
    model `verify_retries` times (`red_gate_returns`), then stands: reported
    finished, never passed. A gate that was red before the run touched
    anything is not the model's to fix, so it is never returned."""
    if not (
        ctx.mode == "run"
        and ctx.tree_green() is False
        and red_gate_returns(
            ctx.verify_when,
            ctx.verify_retries,
            state.verify,
            state.gates,
            gate_present=ctx.gate_present(),
        )
    ):
        return None
    state.gates.verify_retries_used += 1
    used = state.gates.verify_retries_used
    return Refusal(
        finish_red_notice(used=used, retries=ctx.verify_retries),
        event="loop.verify_finish.gated",
        fields={"iteration": turn.iteration, "nudges_used": used},
        log=(
            f"  finish_session returned: verify not green (return #{used} of"
            f" {ctx.verify_retries}) at iter {turn.iteration}"
        ),
    )


def memory_finish(turn: TurnState, state: LoopState, ctx: TurnContext) -> Refusal | None:
    """The memory write-side backstop: a run's first finish_session after a
    recovery from a red verify to green, with nothing recorded in the memory
    store, is deferred once; the nudge asks for the root cause or an
    immediate re-finish (see `_nudges` for the measurement behind it)."""
    if not (
        ctx.mode == "run"
        and ctx.memory_wired
        and state.verify.ever_failed
        and state.verify.last_ok is True
        and not state.memory.written
        and not state.memory.finish_nudged
    ):
        return None
    state.memory.finish_nudged = True
    return Refusal(
        MEMORY_FINISH_NUDGE,
        event="loop.memory_finish.gated",
        fields={"iteration": turn.iteration},
        log=f"  finish_session deferred once: memory backstop at iter {turn.iteration}",
    )
