# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The order a task graph is read in, and what "open" means in it.

One owner for every surface.
"""

from __future__ import annotations

from agent6.graph import models

# A task nobody has finished with; a parent with one is a container, its children the work.
OPEN_STATUSES = frozenset({"pending", "in_progress"})

# Any of these satisfies a dependent's wait; a parent passes only once every child is one.
DONE_STATUSES = frozenset({"passed", "skipped", "obsolete"})


def has_open_child(nodes: dict[str, models.TaskNode], node: models.TaskNode) -> bool:
    """Return whether any child of the node is still open.

    A failed child is not open, so its parent is the unit of work again until the child is
    retried or retired.

    Args:
        nodes: The graph by id.
        node: The parent.

    Returns:
        True when a child is pending or in progress.
    """
    return any(
        (c := nodes.get(cid)) is not None and c.status in OPEN_STATUSES for cid in node.children
    )


def unresolved_children(nodes: dict[str, models.TaskNode], node: models.TaskNode) -> list[str]:
    """Return the children that are open, or failed and never retired.

    `passed` on the parent would claim work no one did or that failed; the refusal names
    these.

    Args:
        nodes: The graph by id.
        node: The parent.

    Returns:
        The child ids not in a done status.
    """
    return [
        cid
        for cid in node.children
        if (c := nodes.get(cid)) is not None and c.status not in DONE_STATUSES
    ]


def ready_subtask(nodes: dict[str, models.TaskNode], node: models.TaskNode) -> bool:
    """Return whether the node is an open subtask with its dependencies done and no open child.

    Args:
        nodes: The graph by id.
        node: The candidate; the root is never ready.

    Returns:
        True when the node is a unit of work the frontier may surface.
    """
    if node.parent_id is None or node.status not in OPEN_STATUSES:
        return False
    for dep in node.depends_on:
        d = nodes.get(dep)
        if d is None or d.status not in DONE_STATUSES:
            return False
    return not has_open_child(nodes, node)


def is_focusable_subtask(nodes: dict[str, models.TaskNode], node: models.TaskNode) -> bool:
    """Return whether the node is a ready ordinary subtask.

    A standing task is the fallback, selected only when nothing ordinary is ready.

    Args:
        nodes: The graph by id.
        node: The candidate.

    Returns:
        True when the node is ready and not standing.
    """
    return not node.standing and ready_subtask(nodes, node)


def tree_order(nodes: dict[str, models.TaskNode]) -> list[str]:
    """Return every node id depth-first through `children`, roots in id order.

    The children list is the order the frontier executes, so every surface shows this order;
    the node map would give insertion order live and filesystem order after a resume. A
    child absent from the map is skipped, and a node no walk reached (a cycle, a dangling
    parent) is appended in id order, so every node is visited exactly once.

    Args:
        nodes: The graph by id.

    Returns:
        The ids in display order.
    """
    order: list[str] = []
    seen: set[str] = set()

    def walk(nid: str) -> None:
        if nid in seen or nid not in nodes:
            return
        seen.add(nid)
        order.append(nid)
        for child in nodes[nid].children:
            walk(child)

    for nid in sorted(nodes, key=id_order):
        if nodes[nid].parent_id is None:
            walk(nid)
    for nid in sorted(nodes, key=id_order):
        walk(nid)
    return order


def id_order(task_id: str) -> tuple[int, str]:
    """Return the sort key that orders ids by creation: shorter first, then lexicographic.

    Ids are zero-padded numbers, so string order is creation order at a fixed width; the
    length key keeps that true once a run outgrows the padding.

    Args:
        task_id: The id.

    Returns:
        The key.
    """
    return len(task_id), task_id
