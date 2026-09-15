# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The finish gates' rules: what an end must satisfy before the loop honours
it, what it is called, and the words each refusal carries. The loop runs
the gates in order (the contract, the panel, the metric, the open tasks, the
verify, the memory backstop, the standing goal) and does the I/O; each rule
here is a pure function over the state it reads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent6.config import WorkflowConfig
from agent6.graph.models import TaskNode
from agent6.graph.order import OPEN_STATUSES
from agent6.workflows._guards import FinishGates
from agent6.workflows._nudges import TASK_FINISH_PATIENCE
from agent6.workflows._session_state import SessionEndReason
from agent6.workflows._verify_verdict import VerifyVerdict

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
    workflow: WorkflowConfig, verify: VerifyVerdict, gates: FinishGates, *, gate_present: bool
) -> bool:
    """Whether a red gate is the model's to fix, so an end over it goes back:
    a gate exists and is the harness's to run, was not red before the run
    touched anything (or this run has since made it green), was not denied
    or withheld by the operator, and returns are left. One answer for
    finish_session and the ends the harness declares, so neither can hand
    back a gate the model cannot run."""
    return (
        workflow.verify_when != "never"
        and gate_present
        and (verify.baseline_ok is not False or verify.ever_passed)
        and gates.verify_retries_used < workflow.verify_retries
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
