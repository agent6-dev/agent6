# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a task DAG as an indented text tree.

Shared by `sessions graph`, `sessions show` and the live CLI stream's plan block,
so the plan reads the same everywhere, including a headless run.
"""

from __future__ import annotations

from typing import Any

from agent6.viewmodel import task_tree_views


def task_tree_lines(nodes: dict[str, Any], cursor: str | None = None) -> list[str]:
    """Render the nodes as one line each, DFS left to right.

    The walk is the read model's, so a node its parent's `children` list does not
    name is still shown, and the focus task wears the in-progress glyph.

    Args:
        nodes: Raw node dicts, a `graph.update` event's or a persisted graph's dump.
        cursor: The focus task's id.

    Returns:
        `<id> <indent><glyph> <title>` per node, plus the note and the commit's short sha.
    """
    out: list[str] = []
    for view in task_tree_views(nodes, cursor):
        node = nodes.get(view.id)
        sha = node.get("commit_sha") if isinstance(node, dict) else None
        commit = f"  ({sha[:7]})" if isinstance(sha, str) and sha else ""
        note = f"  ({view.note})" if view.note else ""
        # The id leads the line: it is what `/retire` takes.
        out.append(
            f"{view.short_id:>3}  {'  ' * view.depth}{view.glyph} {view.title}{note}{commit}"
        )
    return out
