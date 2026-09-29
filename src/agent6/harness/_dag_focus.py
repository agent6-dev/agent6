# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The DAG focus frontier and its directives.

Pure helpers over the curator's nodes dict. The loop keeps the worker on one
task at a time: each turn it computes the current task (the curator cursor
while it points at a focusable subtask, else the first dependency-satisfied
open subtask in creation order), advances the cursor to it, and injects a
focus banner when the focus first appears, changes, or was wiped by a tier-2
restart. The banner survives tier-1 elision. Only subtasks are focus
candidates, as in the finish gate: the always-pending root is the whole job.
"""

from __future__ import annotations

from agent6.graph import models, order

# After one of these runs the loop re-snapshots the graph (the graph.update event).
DAG_MUTATING_TOOLS = frozenset({"add_task", "update_task"})


# A model that stays on one task this many turns without a focus change is nudged to
# split, pass or skip it; the nudge re-fires up to STUCK_NUDGE_MAX times per task.
STUCK_ON_TASK_AFTER = 20
STUCK_NUDGE_MAX = 3


def first_ready_subtask(nodes: dict[str, models.TaskNode]) -> str | None:
    """Return the first focusable subtask in the order the task tree shows.

    Focusable is open, dependencies satisfied, no open child; the order is
    depth-first through each parent's `children` list, so a reordered child
    executes where every renderer displays it. Roots and unreachable nodes fall
    back to id order, which is creation order even on a resumed run. When
    nothing ordinary is ready, the first ready standing task is the fallback.

    Args:
        nodes: The task graph's nodes by id.

    Returns:
        The subtask's id, or None when nothing is ready.
    """
    for nid in order.tree_order(nodes):
        if order.is_focusable_subtask(nodes, nodes[nid]):
            return nid
    for nid in order.tree_order(nodes):
        node = nodes[nid]
        if node.standing and order.ready_subtask(nodes, node):
            return nid
    return None


def current_task_id(nodes: dict[str, models.TaskNode], cursor: str | None) -> str | None:
    """Return the subtask to focus on now.

    The curator cursor wins while it points at a focusable subtask (a decomposed
    parent does not qualify, its leaves do, so a split moves focus forward); else
    the first ready subtask.

    Args:
        nodes: The task graph's nodes by id.
        cursor: The curator cursor, or None.

    Returns:
        The subtask's id, or None when no subtask is focusable.
    """
    if cursor is not None:
        node = nodes.get(cursor)
        if node is not None and order.is_focusable_subtask(nodes, node):
            return cursor
    return first_ready_subtask(nodes)


def current_task_banner(task_id: str, node: models.TaskNode, *, decompose: bool = False) -> str:
    """Return the focus directive naming the current task and its acceptance.

    Args:
        task_id: The current task's id.
        node: The current task.
        decompose: True invites child subtasks under a task without children.

    Returns:
        The banner text.
    """
    title = node.title.strip() or "(untitled)"
    lines = [f"[harness focus] Current task ({task_id}): {title}"]
    if models.queued_by_operator(node):
        # The operator's whole text is the spec; the title is only its first line.
        if (queued := node.rationale.strip()) and queued != title:
            lines.append(queued)
        lines.append(
            "The operator queued this task while the run was going. Their wording is the"
            " spec: work it as written, and add_task child subtasks under it if it is large."
        )
    acceptance = node.acceptance.strip()
    if acceptance:
        lines.append(f"Acceptance: {acceptance}")
    paths = node.relevant_paths
    if paths:
        lines.append("Relevant paths: " + ", ".join(paths[:8]))
    if node.standing:
        # The curator refuses `passed` and every retirement on a standing task.
        lines.append(
            "This is a standing task: it never passes, so do not mark it passed;"
            " only the operator retires it. Work a round on it now, add_task each"
            " follow-up you find so"
            " it is worked in turn, and call finish_session when a round finds"
            " nothing left to do."
        )
    else:
        lines.append(
            "Work this ONE task to completion before anything else. When its"
            " acceptance is met, mark it passed with update_task -- you will then be"
            " moved to the next task. If you find unrelated work, add_task it"
            " instead of switching to it now."
        )
    # Decompose runs plan recursively: a task that turns out large gets a finer plan.
    if decompose and not node.children:
        lines.append(
            "If this task is itself large or multi-step, add child subtasks under"
            f" it (parent_id={task_id}) breaking it into finer steps, then do those."
        )
    return "\n".join(lines)


def stuck_on_task_nudge(task_id: str, node: models.TaskNode, turns: int) -> str:
    """Return the nudge offering the three ways to record progress on a stuck task.

    Never for a standing task: it concludes nothing by design, and two of the
    three moves are refused on it.

    Args:
        task_id: The current task's id.
        node: The current task.
        turns: The consecutive turns spent on it.

    Returns:
        The nudge text.
    """
    title = node.title.strip() or "(untitled)"
    return (
        f"[harness] You have spent {turns} turns on the current task"
        f" ({task_id}: {title}) without concluding it. Pick ONE now and record it:\n"
        "- Too big? Split it into smaller ordered subtasks with add_task and work"
        " the first one.\n"
        "- Effectively done? Mark it passed with update_task.\n"
        "- Not needed? Mark it obsolete or skipped with update_task.\n"
        "Keep the task list in step with your progress rather than working on"
        " without updating it."
    )


def initial_dag_hint(root_id: str | None, mode: str, decompose: bool) -> str:
    """Return the DAG hint appended to the first user message.

    Only run and plan expose the DAG tools (tools/schema.py); ask wires a curator
    but no `add_task`, so a hint there would name a tool the model cannot call.
    The decompose-first directive is run-mode only: it references the run-only
    `<decompose-first>` system block and tells the worker to edit.

    Args:
        root_id: The DAG root task id, or None when no DAG is wired.
        mode: The run mode.
        decompose: True asks for the plan before the first edit.

    Returns:
        The hint, "" when the mode has no DAG tools.
    """
    if root_id is None or mode not in ("run", "plan"):
        return ""
    if mode == "run" and decompose:
        return (
            "\n\nThe DAG-as-tool surface is wired (root task id"
            f" `{root_id}`). START by calling `add_task` several times to"
            " lay out your whole plan as ordered subtasks (see"
            " <decompose-first>), then work the first one. Do not edit"
            " before the plan exists."
        )
    return (
        "\n\nThe DAG-as-tool surface is wired. Root task id is"
        f" `{root_id}`. Use `add_task` to break this into trackable"
        " subtasks (or skip the DAG entirely - it's optional)."
    )
