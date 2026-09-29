# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""DAG handlers: add_task, update_task and list_tasks.

Each raises ToolError when no curator was wired, so a standalone dispatcher works.
"""

from __future__ import annotations

from typing import Any

from agent6.graph import curator as graph_curator
from agent6.graph import models, order
from agent6.tools import errors, results, schema


def add_task(
    curator: graph_curator.GraphCurator | None, run_root_node_id: str | None, raw: dict[str, Any]
) -> results.AddTaskResult:
    """Add a subtask under the given parent, or under the run's root.

    Args:
        curator: The graph curator, or None when this run has no graph.
        run_root_node_id: The parent when the call names none.
        raw: The tool call's arguments.

    Returns:
        The new node's id, parent, title and status.

    Raises:
        ToolError: No curator is wired.
    """
    if curator is None:
        raise errors.ToolError("DAG curator not available in this run")
    args = schema.DagAddTaskInput.model_validate(raw)
    parent_id = args.parent_id or run_root_node_id
    draft = models.TaskNodeDraft(
        title=args.title,
        rationale=args.rationale,
        acceptance=args.acceptance,
        relevant_paths=args.relevant_paths,
        depends_on=args.depends_on,
        created_by="worker",
    )
    intent = models.AddSubtaskIntent(parent_id=parent_id, draft=draft, after=args.after)
    node = curator.add_subtask(intent)
    return results.AddTaskResult(
        id=node.id,
        parent_id=node.parent_id,
        title=node.title,
        status=node.status,
    )


def update_task(
    curator: graph_curator.GraphCurator | None, raw: dict[str, Any]
) -> results.UpdateTaskResult:
    """Change a task's status, add dependencies, or both.

    Args:
        curator: The graph curator, or None when this run has no graph.
        raw: The tool call's arguments.

    Returns:
        The node after the change, with a claim note when it was marked in progress.

    Raises:
        ToolError: No curator is wired, a note comes without a status, the call retires a task
            the operator queued or the standing goal, or it changes nothing.
    """
    if curator is None:
        raise errors.ToolError("DAG curator not available in this run")
    args = schema.DagUpdateTaskInput.model_validate(raw)
    node = None
    if args.note and args.status is None:
        # The graph records a note on a status change only.
        raise errors.ToolError("a note rides along with a status; pass status too")
    if args.status is not None:
        if args.status in ("skipped", "obsolete"):
            current = curator.get(args.id)
            if models.queued_by_operator(current):
                # The curator stays permissive: `/retire` is the operator's own route through it.
                raise errors.ToolError(
                    f"update_task: {args.id} was queued by the operator, so it is not"
                    " yours to retire; pass it when it is done, or leave it open"
                )
            if current.standing:
                # The model retiring the standing goal would turn the fallback into an early finish.
                raise errors.ToolError(
                    f"update_task: {args.id} is the operator's standing goal;"
                    " it stays until the operator retires it. Work it when"
                    " nothing else is ready, or finish_session on a hard limit."
                )
        intent = models.UpdateStatusIntent(
            id=args.id,
            new_status=args.status,  # type: ignore[arg-type]  # pydantic validates the literal
            note=args.note,
        )
        node = curator.update_status(intent)
    # The curator rejects unknown ids and cycles; dispatch() surfaces that as a ToolError.
    for dep in args.depends_on:
        node = curator.add_dependency(models.AddDependencyIntent(id=args.id, depends_on=dep))
    if node is None:
        raise errors.ToolError("update_task: pass status and/or depends_on")
    return results.UpdateTaskResult(
        id=node.id,
        status=node.status,
        title=node.title,
        depends_on=tuple(node.depends_on),
        note=_claim_note(curator, node) if args.status == "in_progress" else "",
    )


def _claim_note(curator: graph_curator.GraphCurator, node: models.TaskNode) -> str:
    """Return what marking a task in_progress did to the focus.

    The harness works the claimed task next while it stays workable, so a claim the frontier
    cannot honour says so instead of quietly doing nothing.
    """
    if order.is_focusable_subtask(curator.nodes(), node):
        return "claimed: this is the task the harness works next"
    return (
        "not workable yet (a dependency, an open child, or a standing task), so the"
        " harness works its own next pick instead"
    )


def list_tasks(
    curator: graph_curator.GraphCurator | None, raw: dict[str, Any]
) -> results.ListTasksResult:
    """List the graph's tasks in tree order, optionally filtered by status.

    Args:
        curator: The graph curator, or None when this run has no graph.
        raw: The tool call's arguments.

    Returns:
        The tasks as the model reads them, and their count.

    Raises:
        ToolError: No curator is wired.
    """
    if curator is None:
        raise errors.ToolError("DAG curator not available in this run")
    args = schema.DagListTasksInput.model_validate(raw)
    out: list[dict[str, Any]] = []
    # Tree order is what the frontier executes; map order differs live and after a resume.
    nodes = curator.nodes()
    for node_id in order.tree_order(nodes):
        node = nodes[node_id]
        if args.status and node.status != args.status:
            continue
        out.append(
            {
                "id": node_id,
                "parent_id": node.parent_id,
                "title": node.title,
                "status": node.status,
                "acceptance": node.acceptance,
                "relevant_paths": list(node.relevant_paths),
                "depends_on": list(node.depends_on),
                # A standing task never passes and never gates a finish.
                "standing": node.standing,
            }
        )
    return results.ListTasksResult(tasks=tuple(out), count=len(out))
