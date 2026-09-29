# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one in-process mutator of a session's task graph.

A mutation is validated, applied in memory, written as the affected node files stamped
with the version it will journal, then appended to `graph.jsonl` with the version bump.
One curator per run is the invariant, upheld by the session's `worker.lock`; the flock
around each mutation only bounds the damage if it is broken, since each instance caches
the graph at construction. On a write-path fault `_mutating` reloads from disk before
re-raising, so a later read never sees a node that was never persisted.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import sys
from collections.abc import Generator
from typing import Literal

import pydantic

from agent6.graph import models, order, storage
from agent6.sessions import layout as sessions_layout


class CuratorError(Exception):
    """A curator intent was rejected by validation, before anything was applied."""


class _JournalBase(pydantic.BaseModel):
    """The base of the typed journal entries.

    The node files are the source of truth; the journal is read back for `graph_version`
    and by `graph.replay`. `storage.write_journal` adds the timestamp and sorts the keys.

    Attributes:
        graph_version: The version the mutation produced, stamped by `_post_mutation`.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    graph_version: int = 0


class AddSubtaskJournal(_JournalBase):
    """A node was added.

    Attributes:
        op: The intent name.
        id: The assigned id.
        parent_id: The parent, or None.
        by: The actor.
    """

    op: Literal["add_subtask"] = "add_subtask"
    id: str
    parent_id: str | None
    by: models.NodeActor


class UpdateStatusJournal(_JournalBase):
    """A node's status changed."""

    op: Literal["update_status"] = "update_status"
    id: str
    new_status: models.NodeStatus


class AddDependencyJournal(_JournalBase):
    """A node gained a dependency."""

    op: Literal["add_dependency"] = "add_dependency"
    id: str
    depends_on: str


class RecordCommitJournal(_JournalBase):
    """A node's commit was recorded."""

    op: Literal["record_commit"] = "record_commit"
    id: str
    sha: str


class SetCursorJournal(_JournalBase):
    """The focus moved."""

    op: Literal["set_cursor"] = "set_cursor"
    id: str | None


JournalEntry = (
    AddSubtaskJournal
    | UpdateStatusJournal
    | AddDependencyJournal
    | RecordCommitJournal
    | SetCursorJournal
)


def _place(
    children: tuple[str, ...], new_id: str, after: str | None, *, standing_at: int | None = None
) -> tuple[str, ...]:
    """Insert a child after a named sibling, else last, ahead of a standing sibling.

    The standing goal is the run's last resort and the tree keeps it at the end; a named
    position still wins.

    Args:
        children: The parent's children.
        new_id: The child to insert.
        after: The sibling to follow, or None for last.
        standing_at: The standing sibling's index, or None.

    Returns:
        The new children.
    """
    if after is not None:
        at = children.index(after) + 1
        return (*children[:at], new_id, *children[at:])
    if standing_at is None:
        return (*children, new_id)
    return (*children[:standing_at], new_id, *children[standing_at:])


# Zero-padded so a listing lines up; a run past 9999 keeps working, since id_order sorts on
# the number. The busiest recorded graph held 24 tasks.
TASK_ID_WIDTH = 4


def _next_task_id(nodes: dict[str, models.TaskNode]) -> str:
    """Return the next task id: the highest number plus one, zero-padded.

    Nothing deletes a node and the flock serialises every mutation, so the number is free
    and never reused; a graph whose ids are opaque starts the count at one.

    Args:
        nodes: The graph by id.

    Returns:
        The id.
    """
    highest = max((int(nid) for nid in nodes if nid.isdigit()), default=0)
    return f"{highest + 1:0{TASK_ID_WIDTH}d}"


def _now() -> datetime.datetime:
    """Return the current UTC time.

    Returns:
        An aware datetime.
    """
    return datetime.datetime.now(tz=datetime.UTC)


class GraphCurator:
    """One session's graph, in memory and on disk."""

    def __init__(self, layout: sessions_layout.SessionLayout) -> None:
        """Load the graph and resync the version counter to a journal that lost its tail.

        A node stamped newer than the journal's max version is a death between the node
        write and its journal append; the counter continues from the node's stamp so the
        lost number is never reused, and the change stays out of historical replay.

        Args:
            layout: The session layout, whose dirs are created.
        """
        self._layout = layout
        layout.ensure()
        self._nodes: dict[str, models.TaskNode] = storage.load_graph(layout)
        self._graph_version = self._compute_graph_version()
        node_max = max((n.graph_version for n in self._nodes.values()), default=0)
        if node_max > self._graph_version:
            sys.stderr.write(
                f"agent6: graph journal lost its tail (a node is stamped v{node_max},"
                f" the journal ends at v{self._graph_version}); continuing from"
                f" v{node_max}. The lost operation stays in the current graph but"
                " will not appear in fork --at-turn replays.\n"
            )
            self._graph_version = node_max

    def _compute_graph_version(self) -> int:
        """Return the journal's highest version, else the node count.

        Returns:
            The version.
        """
        return max(
            (
                int(gv)
                for entry in self._iter_recent_journal()
                if isinstance((gv := entry.get("graph_version", 0)), int)
            ),
            default=len(self._nodes),
        )

    @property
    def layout(self) -> sessions_layout.SessionLayout:
        """The session layout."""
        return self._layout

    @property
    def graph_version(self) -> int:
        """The version of the last mutation."""
        return self._graph_version

    def nodes(self) -> dict[str, models.TaskNode]:
        """Return a copy of the graph.

        Returns:
            The nodes by id.
        """
        return dict(self._nodes)

    def get(self, node_id: str) -> models.TaskNode:
        """Return one node.

        Args:
            node_id: The id.

        Returns:
            The node.

        Raises:
            CuratorError: No such node.
        """
        if node_id not in self._nodes:
            raise CuratorError(f"unknown node: {node_id}")
        return self._nodes[node_id]

    def cursor(self) -> str | None:
        """Read the focused node's id from disk.

        Returns:
            The id, or None.
        """
        return storage.read_cursor(self._layout)

    @contextlib.contextmanager
    def _mutating(self) -> Generator[None]:
        """Hold the graph flock for one mutation, reloading from disk on a write fault.

        A `CuratorError` rejected the intent before anything was applied and propagates
        untouched. Any other fault escapes after the in-memory graph was updated, so the
        reload, under the same flock, keeps a later read from seeing an unpersisted node.

        Yields:
            Nothing; the lock is held for the block.

        Raises:
            CuratorError: The block rejected its intent; re-raised without a reload.
        """
        with storage.flock(self._layout.lock_path):
            try:
                yield
            except CuratorError:
                raise
            except Exception:
                self._nodes = storage.load_graph(self._layout)
                self._graph_version = self._compute_graph_version()
                raise

    def _write(self, node: models.TaskNode) -> models.TaskNode:
        """Stamp a node with the version this mutation will journal, cache it and write it.

        Every write inside one mutation carries the number `_post_mutation` then records.

        Args:
            node: The node.

        Returns:
            The stamped copy.
        """
        stamped = node.model_copy(update={"graph_version": self._graph_version + 1})
        self._nodes[stamped.id] = stamped
        storage.write_node(self._layout, self._nodes, stamped)
        return stamped

    def add_subtask(self, intent: models.AddSubtaskIntent) -> models.TaskNode:
        """Add a node under a parent.

        The child is written before the parent's link, so a crash between leaves at worst an
        orphan rather than a dangling reference. Only the operator's steering may set the
        standing flag: a model asking for one keeps its task and loses the flag.

        Args:
            intent: The intent.

        Returns:
            The new node.

        Raises:
            CuratorError: The parent, the `after` sibling, or a dependency is unknown.
        """
        with self._mutating():
            parent = self._nodes.get(intent.parent_id) if intent.parent_id else None
            if intent.parent_id is not None and parent is None:
                raise CuratorError(f"add_subtask: unknown parent {intent.parent_id!r}")
            if intent.after is not None and (parent is None or intent.after not in parent.children):
                raise CuratorError(
                    f"add_subtask: after {intent.after!r} is not a child of"
                    f" {intent.parent_id!r}; a position names a sibling"
                )
            for dep in intent.draft.depends_on:
                if dep not in self._nodes:
                    raise CuratorError(f"add_subtask: unknown dep {dep!r}")
            now = _now()
            node = models.TaskNode(
                id=_next_task_id(self._nodes),
                parent_id=intent.parent_id,
                title=intent.draft.title,
                rationale=intent.draft.rationale,
                acceptance=intent.draft.acceptance,
                relevant_paths=intent.draft.relevant_paths,
                depends_on=intent.draft.depends_on,
                standing=intent.draft.standing and intent.draft.created_by == "steering",
                children=(),
                status="pending",
                created_at=now,
                updated_at=now,
                created_by=intent.draft.created_by,
            )
            node = self._write(node)
            if parent is not None:
                updated_parent = parent.model_copy(
                    update={
                        "children": _place(
                            parent.children,
                            node.id,
                            intent.after,
                            standing_at=self._standing_at(parent),
                        ),
                        "updated_at": now,
                    }
                )
                self._write(updated_parent)
            self._post_mutation(
                AddSubtaskJournal(
                    id=node.id, parent_id=intent.parent_id, by=intent.draft.created_by
                )
            )
            return node

    def _standing_at(self, parent: models.TaskNode) -> int | None:
        """Return the index of the parent's standing child, so a new sibling lands before it.

        Args:
            parent: The parent.

        Returns:
            The index, or None when the parent has no standing child.
        """
        for i, cid in enumerate(parent.children):
            child = self._nodes.get(cid)
            if child is not None and child.standing:
                return i
        return None

    def update_status(self, intent: models.UpdateStatusIntent) -> models.TaskNode:
        """Set a node's status, appending the note.

        An end is final: a passed task may only be retired, and a retired one stays retired,
        or `passed -> obsolete -> pending` would re-open work every dependent was told had
        passed. Claiming a task in progress also moves the cursor to it.

        Args:
            intent: The intent.

        Returns:
            The updated node.

        Raises:
            CuratorError: The node is unknown, the transition re-opens an end, a standing
                task would pass, or a non-root parent would pass over unresolved children.
        """
        with self._mutating():
            node = self.get(intent.id)
            if node.status == "passed" and intent.new_status != "obsolete":
                raise CuratorError(
                    f"cannot transition passed node {intent.id} to {intent.new_status}"
                )
            if node.status in ("skipped", "obsolete") and intent.new_status != node.status:
                raise CuratorError(
                    f"{intent.id} is retired ({node.status}) and stays retired;"
                    " add_task if the work is needed after all"
                )
            if node.standing and intent.new_status == "passed":
                raise CuratorError(
                    f"a standing task never passes ({intent.id}); mark it skipped or"
                    " obsolete to retire it"
                )
            if (
                intent.new_status == "passed"
                and node.parent_id is not None
                and (unresolved := order.unresolved_children(self._nodes, node))
            ):
                # The root is exempt: nothing depends on it.
                raise CuratorError(
                    f"{intent.id} has unresolved children ({', '.join(unresolved)}), so it is"
                    " not finished; mark them passed, skipped or obsolete first"
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
            updated = self._write(updated)
            if intent.new_status == "in_progress":
                # The frontier honours the cursor while it points at a focusable subtask.
                storage.write_cursor(self._layout, updated.id)
            self._post_mutation(UpdateStatusJournal(id=updated.id, new_status=intent.new_status))
            return updated

    def add_dependency(self, intent: models.AddDependencyIntent) -> models.TaskNode:
        """Make a node wait on another; a dependency already present is a no-op.

        Args:
            intent: The intent.

        Returns:
            The updated node.

        Raises:
            CuratorError: A node is unknown, or the edge would make a cycle.
        """
        with self._mutating():
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
            updated = self._write(updated)
            self._post_mutation(AddDependencyJournal(id=updated.id, depends_on=intent.depends_on))
            return updated

    def record_commit(self, intent: models.RecordCommitIntent) -> models.TaskNode:
        """Record the commit that landed a node.

        Args:
            intent: The intent.

        Returns:
            The updated node.

        Raises:
            CuratorError: The node is unknown.
        """
        with self._mutating():
            node = self.get(intent.id)
            updated = node.model_copy(update={"commit_sha": intent.sha, "updated_at": _now()})
            updated = self._write(updated)
            self._post_mutation(RecordCommitJournal(id=updated.id, sha=intent.sha))
            return updated

    def set_cursor(self, intent: models.SetCursorIntent) -> None:
        """Move the focus.

        Args:
            intent: The intent.

        Raises:
            CuratorError: The node is unknown.
        """
        with self._mutating():
            if intent.id is not None and intent.id not in self._nodes:
                raise CuratorError(f"set_cursor: unknown node {intent.id!r}")
            storage.write_cursor(self._layout, intent.id)
            self._post_mutation(SetCursorJournal(id=intent.id))

    def _post_mutation(self, entry: JournalEntry) -> None:
        """Bump the version and journal the entry stamped with it.

        Args:
            entry: The mutation's entry.
        """
        self._graph_version += 1
        stamped = entry.model_copy(update={"graph_version": self._graph_version})
        storage.write_journal(self._layout, stamped.model_dump(mode="json"))

    def _iter_recent_journal(self) -> list[dict[str, object]]:
        """Read the journal, skipping a torn line with a note on stderr.

        The node files are the source of truth and the version counter self-heals, so a
        torn final line must not make the run unresumable.

        Returns:
            The entries in file order.
        """
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
                sys.stderr.write(f"agent6: skipping malformed journal line: {stripped[:80]!r}\n")
        return entries

    def _would_introduce_cycle(self, src: str, new_dep: str) -> bool:
        """Return whether a new dependency edge would make a cycle.

        Args:
            src: The node gaining the dependency.
            new_dep: The node it would wait on.

        Returns:
            True when the walk from the dependency reaches the node.
        """
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
