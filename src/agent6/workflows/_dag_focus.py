# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""surface-current-task: the DAG focus frontier and its directives.

Pure helpers over the curator's nodes dict. The loop keeps a small/weak
worker on ONE task at a time: each turn it computes the current task -- the
curator cursor when it still points at an open subtask (the worker's explicit
choice wins), else the first dependency-satisfied open subtask in creation
order -- advances the cursor to it, and injects a focus banner when the focus
first appears, changes, or was wiped by a tier-2 restart. The banner survives
tier-1 elision, so the worker keeps seeing it between those events without
re-appending every turn. Only SUBTASKS are focus candidates, mirroring the
finish-gate: the always-pending auto-root is the whole job, not a unit of
work to surface.
"""

from __future__ import annotations

from typing import Any

# Tool names that mutate the task DAG; after one runs the loop re-snapshots the
# graph (graph.update event) so a live viewer can render the worker's task
# breakdown.
DAG_MUTATING_TOOLS = frozenset({"add_task", "update_task", "set_cursor"})

OPEN_STATUSES = frozenset({"pending", "in_progress"})
DEPS_SATISFIED_STATUSES = frozenset({"passed", "skipped", "obsolete"})

# Anti-grind: a weak model on a vague/oversized task can stay on one DAG task for
# many turns, reading without ever marking it done, decomposing it, or trying to
# finish -- so neither the finish-gate (fires on a finish attempt) nor went_quiet
# (it is busy) catches it (observed live: GLM ground task 1 for 200 turns, 394
# reads, 0 passed). Every this-many consecutive turns on the SAME focus task with
# no forward motion (cursor advance / mark-done / decompose, any of which changes
# the focus and resets the count), fire a nudge offering split / pass / skip. It
# re-fires periodically -- a weak model was observed ignoring a single nudge --
# but caps at STUCK_NUDGE_MAX per task so it cannot nag forever; generous so a
# model making normal progress (which changes focus well before this) never sees it.
STUCK_ON_TASK_AFTER = 20
STUCK_NUDGE_MAX = 3


def has_open_child(nodes: dict[str, Any], node: dict[str, Any]) -> bool:
    """True if any of ``node``'s children is still open. A subtask with open
    children is a container -- its children are the unit of work, not it -- so the
    frontier surfaces the children's leaves instead."""
    for cid in node.get("children", ()) or ():
        c = nodes.get(cid)
        if c is not None and c.get("status") in OPEN_STATUSES:
            return True
    return False


def is_focusable_subtask(nodes: dict[str, Any], node: dict[str, Any]) -> bool:
    """An open SUBTASK ready to work: not the auto-root, open, dependencies all
    satisfied, and no open child (a decomposed parent is not itself a unit of
    work)."""
    if node.get("parent_id") is None or node.get("status") not in OPEN_STATUSES:
        return False
    for dep in node.get("depends_on", ()) or ():
        d = nodes.get(dep)
        if d is None or d.get("status") not in DEPS_SATISFIED_STATUSES:
            return False  # a missing or not-yet-done dependency blocks the subtask
    return not has_open_child(nodes, node)


def first_ready_subtask(nodes: dict[str, Any]) -> str | None:
    """First focusable subtask (open, deps satisfied, no open child) in node
    creation order. Node ids are time-sortable ULIDs, so sorting by id restores
    creation order even on a resumed run, where the nodes dict is in filesystem
    order. Returns None when nothing is ready (no subtasks, all done, or all
    blocked / waiting on open children)."""
    for nid in sorted(nodes):
        if is_focusable_subtask(nodes, nodes[nid]):
            return nid
    return None


def current_task_id(nodes: dict[str, Any], cursor: str | None) -> str | None:
    """The subtask to focus on now: the curator cursor when it still points at a
    focusable subtask (a decomposed parent does NOT qualify -- its leaves do, so
    a split moves focus forward), else the first ready subtask. None when no
    subtask is focusable."""
    if cursor is not None:
        node = nodes.get(cursor)
        if node is not None and is_focusable_subtask(nodes, node):
            return cursor
    return first_ready_subtask(nodes)


def current_task_banner(task_id: str, node: dict[str, Any], *, decompose: bool = False) -> str:
    """The per-turn focus directive naming the current task and its acceptance."""
    title = str(node.get("title", "")).strip() or "(untitled)"
    lines = [f"[harness focus] Current task ({task_id}): {title}"]
    acceptance = str(node.get("acceptance", "")).strip()
    if acceptance:
        lines.append(f"Acceptance: {acceptance}")
    paths = node.get("relevant_paths") or ()
    if paths:
        lines.append("Relevant paths: " + ", ".join(str(p) for p in paths[:8]))
    lines.append(
        "Work this ONE task to completion before anything else. When its acceptance"
        " is met, mark it passed with update_task -- you will then be moved to the"
        " next task. If you find unrelated work, add_task it instead of switching"
        " to it now."
    )
    # Decompose runs plan recursively: invite a finer plan for a task that turns
    # out large, at the point the model has the most context to plan it.
    if decompose and not node.get("children"):
        lines.append(
            "If this task is itself large or multi-step, add child subtasks under"
            f" it (parent_id={task_id}) breaking it into finer steps, then do those."
        )
    return "\n".join(lines)


def stuck_on_task_nudge(task_id: str, node: dict[str, Any], turns: int) -> str:
    """The anti-grind directive: the model has spent ``turns`` turns on one task
    without concluding it; offer the three ways to record progress."""
    title = str(node.get("title", "")).strip() or "(untitled)"
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
    """The DAG hint appended to the first user message. The decompose-first
    directive is RUN-MODE ONLY -- it references the run-only ``<decompose-first>``
    system block and tells the worker to edit, neither of which fits plan/ask
    (which also wire a curator, so ``root_id`` is non-None there). Every other
    case gets the optional-DAG hint."""
    if root_id is None:
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
