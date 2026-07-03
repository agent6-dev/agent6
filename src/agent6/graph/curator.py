# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Authoritative in-process graph mutator.

`GraphCurator` is the single source of truth for one run's task graph. The
production deployment runs it inside the `graph-curator` subprocess, which is a
plain subprocess (no jail of its own) that inherits the agent process's
confinement and writes the run's graph under the out-of-tree per-repo state dir.
The class itself is process-agnostic: unit tests instantiate it directly to
exercise the mutation logic without spinning up a UDS server.

Mutations are validated structurally, then applied as:

  1. mutate in-memory graph state
  2. append entry to `graph.jsonl` (the journal, append-only audit log)
  3. atomically rewrite the affected node's `.md` file
  4. (if topology changed) atomically regenerate `graph.dot`
  5. bump `graph_version`

The flock around every mutation prevents interleaved file writes from
accidental parallel curator instances (which we explicitly forbid). It does
not merge their in-memory state: each instance caches the graph at
construction, so a second live instance would still lose updates. One curator
per run is the invariant; the lock only bounds the damage if it is broken.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    ObsoleteIntent,
    RecordCommitIntent,
    ReorderChildrenIntent,
    SetCursorIntent,
    TaskNode,
    UpdateStatusIntent,
)
from agent6.graph.storage import (
    RunLayout,
    flock,
    load_graph,
    read_cursor,
    write_cursor,
    write_dot,
    write_journal,
    write_node,
)
from agent6.graph.ulid import new_ulid


class CuratorError(Exception):
    """A curator intent was rejected (validation, not I/O)."""


def _now() -> datetime:
    return datetime.now(tz=UTC)


class GraphCurator:
    """Owns one run's graph, in-memory and on-disk."""

    def __init__(self, layout: RunLayout) -> None:
        self._layout = layout
        layout.ensure()
        self._nodes: dict[str, TaskNode] = load_graph(layout)
        self._graph_version = max(
            (
                int(gv)
                for entry in self._iter_recent_journal()
                if isinstance((gv := entry.get("graph_version", 0)), int)
            ),
            default=len(self._nodes),
        )

    # ---- accessors --------------------------------------------------------

    @property
    def layout(self) -> RunLayout:
        return self._layout

    @property
    def graph_version(self) -> int:
        return self._graph_version

    def nodes(self) -> dict[str, TaskNode]:
        return dict(self._nodes)

    def get(self, node_id: str) -> TaskNode:
        if node_id not in self._nodes:
            raise CuratorError(f"unknown node: {node_id}")
        return self._nodes[node_id]

    def cursor(self) -> str | None:
        return read_cursor(self._layout)

    # ---- mutations --------------------------------------------------------

    def add_subtask(self, intent: AddSubtaskIntent) -> TaskNode:
        with flock(self._layout.lock_path):
            parent = self._nodes.get(intent.parent_id) if intent.parent_id else None
            if intent.parent_id is not None and parent is None:
                raise CuratorError(f"add_subtask: unknown parent {intent.parent_id!r}")
            for dep in intent.draft.depends_on:
                if dep not in self._nodes:
                    raise CuratorError(f"add_subtask: unknown dep {dep!r}")
            now = _now()
            node = TaskNode(
                id=new_ulid(),
                parent_id=intent.parent_id,
                title=intent.draft.title,
                rationale=intent.draft.rationale,
                acceptance=intent.draft.acceptance,
                relevant_paths=intent.draft.relevant_paths,
                depends_on=intent.draft.depends_on,
                children=(),
                status="pending",
                created_at=now,
                updated_at=now,
                created_by=intent.draft.created_by,
            )
            self._nodes[node.id] = node
            # Write the child node BEFORE the parent->child link so a crash in
            # between can at worst leave an orphan node (parent_id set, not yet
            # listed in parent.children) rather than a dangling reference to a
            # child whose .md never made it to disk.
            write_node(self._layout, self._nodes, node)
            if parent is not None:
                updated_parent = parent.model_copy(
                    update={
                        "children": (*parent.children, node.id),
                        "updated_at": now,
                    }
                )
                self._nodes[parent.id] = updated_parent
                write_node(self._layout, self._nodes, updated_parent)
            self._post_mutation(
                {
                    "op": "add_subtask",
                    "id": node.id,
                    "parent_id": intent.parent_id,
                    "by": intent.draft.created_by,
                }
            )
            return node

    def update_status(self, intent: UpdateStatusIntent) -> TaskNode:
        with flock(self._layout.lock_path):
            node = self.get(intent.id)
            if node.status == "passed" and intent.new_status not in ("obsolete",):
                raise CuratorError(
                    f"cannot transition passed node {intent.id} to {intent.new_status}"
                )
            updated = node.model_copy(
                update={
                    "status": intent.new_status,
                    "updated_at": _now(),
                    "notes": (
                        node.notes if not intent.note else (node.notes + "\n" + intent.note).strip()
                    ),
                }
            )
            self._nodes[updated.id] = updated
            write_node(self._layout, self._nodes, updated)
            self._post_mutation(
                {
                    "op": "update_status",
                    "id": updated.id,
                    "new_status": intent.new_status,
                }
            )
            return updated

    def add_dependency(self, intent: AddDependencyIntent) -> TaskNode:
        with flock(self._layout.lock_path):
            node = self.get(intent.id)
            if intent.depends_on not in self._nodes:
                raise CuratorError(f"unknown dep {intent.depends_on!r}")
            if intent.depends_on in node.depends_on:
                return node
            if self._would_introduce_cycle(intent.id, intent.depends_on):
                raise CuratorError(
                    f"add_dependency {intent.id} -> {intent.depends_on} would introduce cycle"
                )
            updated = node.model_copy(
                update={
                    "depends_on": (*node.depends_on, intent.depends_on),
                    "updated_at": _now(),
                }
            )
            self._nodes[updated.id] = updated
            write_node(self._layout, self._nodes, updated)
            self._post_mutation(
                {
                    "op": "add_dependency",
                    "id": updated.id,
                    "depends_on": intent.depends_on,
                }
            )
            return updated

    def obsolete(self, intent: ObsoleteIntent) -> TaskNode:
        with flock(self._layout.lock_path):
            node = self.get(intent.id)
            updated = node.model_copy(
                update={
                    "status": "obsolete",
                    "updated_at": _now(),
                    "notes": (
                        f"{node.notes}\n[obsolete] {intent.reason}".strip()
                        if intent.reason
                        else node.notes
                    ),
                }
            )
            self._nodes[updated.id] = updated
            write_node(self._layout, self._nodes, updated)
            self._post_mutation({"op": "obsolete", "id": updated.id, "reason": intent.reason})
            return updated

    def reorder_children(self, intent: ReorderChildrenIntent) -> TaskNode:
        with flock(self._layout.lock_path):
            parent = self.get(intent.parent_id)
            # Multiset comparison: a set check would accept a new_order with a
            # duplicated child id (set drops the dup), silently corrupting
            # parent.children into a list that visits a child twice.
            if sorted(intent.new_order) != sorted(parent.children):
                raise CuratorError(
                    f"reorder_children {intent.parent_id}: new_order must be a permutation"
                )
            updated = parent.model_copy(update={"children": intent.new_order, "updated_at": _now()})
            self._nodes[updated.id] = updated
            write_node(self._layout, self._nodes, updated)
            self._post_mutation(
                {
                    "op": "reorder_children",
                    "parent_id": intent.parent_id,
                    "new_order": list(intent.new_order),
                }
            )
            return updated

    def record_commit(self, intent: RecordCommitIntent) -> TaskNode:
        with flock(self._layout.lock_path):
            node = self.get(intent.id)
            updated = node.model_copy(update={"commit_sha": intent.sha, "updated_at": _now()})
            self._nodes[updated.id] = updated
            write_node(self._layout, self._nodes, updated)
            self._post_mutation({"op": "record_commit", "id": updated.id, "sha": intent.sha})
            return updated

    def set_cursor(self, intent: SetCursorIntent) -> None:
        with flock(self._layout.lock_path):
            if intent.id is not None and intent.id not in self._nodes:
                raise CuratorError(f"set_cursor: unknown node {intent.id!r}")
            write_cursor(self._layout, intent.id)
            self._post_mutation({"op": "set_cursor", "id": intent.id}, regen_dot=False)

    # ---- internals --------------------------------------------------------

    def _post_mutation(self, entry: dict[str, object], *, regen_dot: bool = True) -> None:
        self._graph_version += 1
        full_entry = dict(entry)
        full_entry["graph_version"] = self._graph_version
        write_journal(self._layout, full_entry)
        if regen_dot:
            write_dot(self._layout, self._nodes)

    def _iter_recent_journal(self) -> list[dict[str, object]]:
        path = self._layout.journal_path
        if not path.is_file():
            return []
        entries: list[dict[str, object]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError:
                # A crash mid-append can leave a torn final line. The node .md
                # files are the source of truth (read atomically by load_graph)
                # and graph_version is a self-healing monotonic counter, so skip
                # the corrupt line rather than crashing curator startup -- which
                # would otherwise make the whole run unresumable.
                sys.stderr.write(
                    f"graph-curator: skipping malformed journal line: {stripped[:80]!r}\n"
                )
        return entries

    def _would_introduce_cycle(self, src: str, new_dep: str) -> bool:
        """True iff adding src→new_dep would create a cycle in the dep DAG."""
        # Walk dep transitively from new_dep; if we reach src, it's a cycle.
        stack = [new_dep]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            node = self._nodes.get(cur)
            if node is None:
                continue  # dangling depends_on edge (target missing): not a cycle
            stack.extend(node.depends_on)
        return False
