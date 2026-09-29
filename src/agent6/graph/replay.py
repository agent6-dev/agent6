# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rebuild a task graph as it stood at an older `graph_version`, for `fork --at-turn`.

The journal records operations, not node content, and the content operations never touch
is immutable after creation, so the rebuild starts from the current nodes and undoes every
mutation stamped after the target version. Two display-only fields cannot be unwound and
keep their current value: `notes` and `updated_at`. A mutation whose journal entry was
lost to a crash has nothing to undo and stays visible in every replayed version.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agent6.graph.models import TaskNode


@dataclass(frozen=True, slots=True)
class ReplayedGraph:
    """A graph as of one `graph_version`.

    Attributes:
        nodes: The surviving nodes by id.
        cursor: The focused id at that version, or None.
    """

    nodes: dict[str, TaskNode]
    cursor: str | None


def _usable(journal: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the journal entries with a usable op and version, in applied order.

    A torn or foreign line is dropped rather than allowed to abort the rebuild, as
    `load_graph` drops a malformed node file.

    Args:
        journal: The journal entries.

    Returns:
        The usable entries.
    """
    return [
        e
        for e in journal
        if isinstance(e.get("op"), str) and isinstance(e.get("graph_version"), int)
    ]


def journal_prefix(journal: Iterable[dict[str, Any]], version: int) -> list[dict[str, Any]]:
    """Return the entries stamped at or before a version.

    A rebuilt graph ships with the journal it has, so its curator keeps numbering from there.

    Args:
        journal: The journal entries.
        version: The target version.

    Returns:
        The prefix.
    """
    return [e for e in _usable(journal) if int(e["graph_version"]) <= version]


def _str_field(entry: dict[str, Any], key: str) -> str | None:
    """Return an entry's string field, or None when absent or not a string.

    Args:
        entry: The journal entry.
        key: The field.

    Returns:
        The value, or None.
    """
    value = entry.get(key)
    return value if isinstance(value, str) else None


def _children_at(current: tuple[str, ...], kept: dict[str, TaskNode]) -> tuple[str, ...]:
    """Return the parent's surviving children in the order the parent holds them.

    Args:
        current: The parent's children now.
        kept: The nodes that survive at the target version.

    Returns:
        The surviving ids, in execution order.
    """
    return tuple(c for c in current if c in kept)


def graph_at_version(
    nodes: dict[str, TaskNode],
    journal: Iterable[dict[str, Any]],
    version: int,
    *,
    current_cursor: str | None = None,
) -> ReplayedGraph:
    """Rebuild the graph as of a version from the current nodes and the journal.

    A node whose `add_subtask` is stamped after the version is dropped. A node the journal
    never mentions is kept as is: an unknown creation time is not evidence of a later one,
    and dropping it would empty the fork's graph.

    Args:
        nodes: The current nodes by id.
        journal: The journal entries.
        version: The target version.
        current_cursor: The cursor file's value, used only when the journal records no
            cursor at all.

    Returns:
        The rebuilt graph.
    """
    past = _usable(journal)
    at = journal_prefix(past, version)

    born: dict[str, int] = {
        nid: int(e["graph_version"])
        for e in past
        if e["op"] == "add_subtask" and (nid := _str_field(e, "id")) is not None
    }
    kept = {nid: n for nid, n in nodes.items() if born.get(nid, version) <= version}

    # Last write wins, exactly as the curator applied them.
    status: dict[str, str] = {}
    for e in at:
        nid = _str_field(e, "id")
        if nid is None:
            continue
        if e["op"] == "obsolete":  # an older journal's op name for retirement
            status[nid] = "obsolete"
        elif e["op"] == "update_status" and (new := _str_field(e, "new_status")) is not None:
            status[nid] = new
    commit: dict[str, str] = {
        nid: sha
        for e in at
        if e["op"] == "record_commit"
        and (nid := _str_field(e, "id")) is not None
        and (sha := _str_field(e, "sha")) is not None
    }
    deps_after: dict[str, set[str]] = {}
    for e in past:
        if e["op"] != "add_dependency" or int(e["graph_version"]) <= version:
            continue
        nid, dep = _str_field(e, "id"), _str_field(e, "depends_on")
        if nid is not None and dep is not None:
            deps_after.setdefault(nid, set()).add(dep)
    # None in the prefix but some later means the run held no cursor at this version.
    cursors = [e for e in at if e["op"] == "set_cursor"]
    if cursors:
        cursor = _str_field(cursors[-1], "id")
    elif any(e["op"] == "set_cursor" for e in past):
        cursor = None
    else:
        cursor = current_cursor

    rebuilt = {
        nid: node.model_copy(
            update={
                # A node the journal never mentioned keeps what it has.
                "status": status.get(nid, "pending" if nid in born else node.status),
                "commit_sha": commit.get(nid, "" if nid in born else node.commit_sha),
                "depends_on": tuple(d for d in node.depends_on if d not in deps_after.get(nid, ())),
                "children": _children_at(node.children, kept),
                # A stamp past the version would read as a lost journal tail on every open.
                "graph_version": min(node.graph_version, version),
            }
        )
        for nid, node in kept.items()
    }
    return ReplayedGraph(nodes=rebuilt, cursor=cursor if cursor in rebuilt else None)
