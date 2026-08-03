# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The order a task graph is read in: one owner for every surface."""

from __future__ import annotations

from agent6.graph.models import TaskNode


def tree_order(nodes: dict[str, TaskNode]) -> list[str]:
    """Every node id, depth-first through ``children``, roots in id order.

    The children list is the order the frontier executes, so this is the order
    every surface shows -- the renderers, and the `list_tasks` the model reads
    its own plan back from. Iterating the node map instead gave insertion order
    live and filesystem order after a resume, so a task placed second showed up
    last.

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
