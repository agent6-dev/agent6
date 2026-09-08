# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The order a task graph is read in, and what "open" means in it: one owner
for every surface."""

from __future__ import annotations

from agent6.graph.models import TaskNode

# A task nobody has finished with. The frontier surfaces these, the finish
# gate counts them, and a parent with one is a container rather than a unit of
# work: its children are.
OPEN_STATUSES = frozenset({"pending", "in_progress"})

# A task nobody needs to work on: finished, skipped or retired. Any of them
# satisfies a dependent's wait, a count of done tasks includes them all, and a
# parent passes only once every child is one of them.
DONE_STATUSES = frozenset({"passed", "skipped", "obsolete"})


def has_open_child(nodes: dict[str, TaskNode], node: TaskNode) -> bool:
    """True if any of `node`'s children is still open: the subtask is a
    container whose children are the unit of work, so the frontier surfaces
    their leaves instead. A failed child is not open, so its parent is the
    unit of work again until the child is retried or retired."""
    return any(
        (c := nodes.get(cid)) is not None and c.status in OPEN_STATUSES for cid in node.children
    )


def unresolved_children(nodes: dict[str, TaskNode], node: TaskNode) -> list[str]:
    """The ids of `node`'s children that are open, or failed and never retired:
    `passed` on the parent would claim work no one did or that failed, so the
    refusal names them."""
    return [
        cid
        for cid in node.children
        if (c := nodes.get(cid)) is not None and c.status not in DONE_STATUSES
    ]


def tree_order(nodes: dict[str, TaskNode]) -> list[str]:
    """Every node id, depth-first through `children`, roots in id order.

    The children list is the order the frontier executes, so this is the order
    every surface shows: the renderers, and the `list_tasks` the model reads
    its own plan back from. Iterating the node map instead would give insertion
    order live and filesystem order after a resume (a task placed second
    showing up last).

    A child named by a parent but absent from *nodes* is skipped, and a node no
    walk reached (a cycle, a dangling parent_id) is appended in id order, so
    every node is still visited exactly once.
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

    for nid in sorted(nodes):
        if nodes[nid].parent_id is None:
            walk(nid)
    for nid in sorted(nodes):
        walk(nid)
    return order
