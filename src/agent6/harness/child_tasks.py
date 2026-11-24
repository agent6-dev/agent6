"""Resolve the graph task a child dispatch or follow-up can take."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent6.graph import models, order
from agent6.sessions import child_dispatch
from agent6.tools import tool_errors

if TYPE_CHECKING:
    from agent6.harness import parallel_dispatch


def listed(
    dispatcher: parallel_dispatch.ParallelDispatcher, task_id: str, *, retry: bool = False
) -> models.TaskNode:
    """Return a ready listed task with no outstanding child work.

    Args:
        dispatcher: The run that owns the graph and its open dispatches.
        task_id: The listed task to claim.
        retry: A continuation or explicit merge may resolve a failed task.

    Returns:
        The node that can take the result.

    Raises:
        ToolError: The task is absent, held, or not ready.
    """
    nodes = dispatcher.curator.nodes() if dispatcher.curator else {}
    node = nodes.get(task_id)
    if node is None:
        raise tool_errors.ToolError(f"no listed task {task_id}")
    if node.held_by:
        raise tool_errors.ToolError(f"task {task_id} is held by {', '.join(node.held_by)}")
    for task in dispatcher.open.values():
        if task.record.parent_node == task_id and task.record.node_id is None:
            raise tool_errors.ToolError(f"task {task_id} has a pending text dispatch")
    candidate = (
        node.model_copy(update={"status": "pending"}) if retry and node.status == "failed" else node
    )
    if not order.is_focusable_subtask(nodes, candidate) or order.unresolved_children(nodes, node):
        raise tool_errors.ToolError(f"task {task_id} is not a ready ordinary subtask")
    return node


def followup(
    dispatcher: parallel_dispatch.ParallelDispatcher, record: child_dispatch.DispatchRecord
) -> child_dispatch.DispatchRecord:
    """Keep the task for a retry, or detach the follow-up from a final task.

    Args:
        dispatcher: The parent run.
        record: The child's completed dispatch, before its next operation.

    Returns:
        The dispatch with the node that can take its result, if any.

    Raises:
        ToolError: The unfinished task cannot take this child.
    """
    if record.node_id and dispatcher.curator:
        if dispatcher.curator.get(record.node_id).status in order.DONE_STATUSES:
            return record.model_copy(update={"node_id": None, "parent_node": None})
        listed(dispatcher, record.node_id, retry=True)
    return record
