# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The task graph's on-disk format.

One markdown file per node with a YAML frontmatter of the structured fields, laid out to
mirror the tree: `graph/<root>.md`, `graph/<root>/<child>.md`, and so on, beside the
append-only `graph.jsonl` journal and `cursor.json`. Every replacement write is atomic, and
the curator holds the `.lock` flock for a whole mutation. The frontmatter is a single-level
mapping of scalars and lists of strings, parsed by hand.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from agent6.graph.models import TaskNode
from agent6.paths import mkdir_for_real_user
from agent6.portable import atomic_write, fsync_dir, lock_exclusive, unlock
from agent6.sessions.layout import SessionLayout


def _append_line(path: Path, line: str) -> None:
    """Append one line durably.

    Args:
        path: The file, created with its directory when absent.
        line: The line; a newline is added when missing.

    Raises:
        OSError: A short write, rather than lost bytes.
    """
    mkdir_for_real_user(path.parent)
    payload = (line if line.endswith("\n") else line + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write appending to {path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, entry: dict[str, object]) -> None:
    """Append one JSON object as a line, durably.

    The caller supplies the whole entry, timestamp included.

    Args:
        path: The file.
        entry: The object.
    """
    _append_line(path, json.dumps(entry, sort_keys=True))


@contextmanager
def flock(path: Path) -> Generator[None]:
    """Hold an exclusive flock on a file, created when missing.

    Args:
        path: The lock file.

    Yields:
        Nothing; the lock is held for the block.
    """
    mkdir_for_real_user(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        lock_exclusive(fd, blocking=True)
        yield
    finally:
        try:
            unlock(fd)
        finally:
            os.close(fd)


def _yaml_quote(s: str) -> str:
    """Quote a scalar so it round-trips through `_yaml_unquote`.

    Args:
        s: The scalar.

    Returns:
        The double-quoted text, backslash, quote, newline and carriage return escaped.
    """
    # `\r` is escaped too: the parser splits on "\n" only, so an unescaped one would be
    # emitted literally; the other Unicode line separators survive for the same reason.
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def _yaml_unquote(s: str) -> str:
    """Unquote a scalar `_yaml_quote` wrote; an unquoted value is returned stripped.

    Args:
        s: The raw text.

    Returns:
        The scalar.
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        body = s[1:-1]
        out: list[str] = []
        i = 0
        while i < len(body):
            c = body[i]
            if c == "\\" and i + 1 < len(body):
                nxt = body[i + 1]
                if nxt == "n":
                    out.append("\n")
                elif nxt == "r":
                    out.append("\r")
                elif nxt == '"':
                    out.append('"')
                elif nxt == "\\":
                    out.append("\\")
                else:
                    out.append(nxt)
                i += 2
                continue
            out.append(c)
            i += 1
        return "".join(out)
    return s


def _dump_frontmatter(node: TaskNode) -> str:
    """Render a node as its frontmatter and notes body.

    Args:
        node: The node.

    Returns:
        The file text.
    """
    fm: list[str] = ["---"]
    fm.append(f"id: {_yaml_quote(node.id)}")
    fm.append(f"parent_id: {_yaml_quote(node.parent_id) if node.parent_id else '~'}")
    fm.append(f"title: {_yaml_quote(node.title)}")
    fm.append(f"rationale: {_yaml_quote(node.rationale)}")
    fm.append(f"acceptance: {_yaml_quote(node.acceptance)}")
    fm.append("relevant_paths:")
    for p in node.relevant_paths:
        fm.append(f"  - {_yaml_quote(p)}")
    fm.append("depends_on:")
    for d in node.depends_on:
        fm.append(f"  - {_yaml_quote(d)}")
    fm.append("children:")
    for c in node.children:
        fm.append(f"  - {_yaml_quote(c)}")
    fm.append(f"status: {_yaml_quote(node.status)}")
    fm.append(f"created_at: {_yaml_quote(node.created_at.isoformat())}")
    fm.append(f"updated_at: {_yaml_quote(node.updated_at.isoformat())}")
    fm.append(f"created_by: {_yaml_quote(node.created_by)}")
    fm.append(f"commit_sha: {_yaml_quote(node.commit_sha)}")
    fm.append(f'graph_version: "{node.graph_version}"')
    if node.standing:
        fm.append('standing: "true"')
    fm.append("---")
    fm.append("")
    fm.append(node.notes if node.notes else "")
    return "\n".join(fm) + "\n"


def _parse_frontmatter(text: str) -> TaskNode:
    """Parse a node file back into a TaskNode.

    Args:
        text: The file text.

    Returns:
        The node.

    Raises:
        ValueError: The frontmatter is malformed, a timestamp does not parse, or a field
            fails the model's validation.
    """
    # Split on "\n" only, the inverse of the dump: str.splitlines() also breaks on \r, \v,
    # \f, NEL and U+2028/2029, which a model can put in a title, and would crash the parser.
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        raise ValueError("missing leading '---'")
    fm: dict[str, str | list[str] | None] = {}
    i = 1
    current_list: list[str] | None = None
    current_list_key: str | None = None
    while i < len(lines):
        line = lines[i]
        if line.rstrip() == "---":
            i += 1
            break
        if line.startswith("  - "):
            if current_list is None or current_list_key is None:
                raise ValueError(f"list item without parent at line {i}: {line!r}")
            current_list.append(_yaml_unquote(line[4:]))
            i += 1
            continue
        if current_list is not None and current_list_key is not None:
            fm[current_list_key] = current_list
            current_list = None
            current_list_key = None
        if ":" not in line:
            raise ValueError(f"bad frontmatter line {i}: {line!r}")
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if raw == "":
            current_list_key = key
            current_list = []
        elif raw == "~":
            fm[key] = None
        else:
            fm[key] = _yaml_unquote(raw)
        i += 1
    if current_list is not None and current_list_key is not None:
        fm[current_list_key] = current_list

    notes = "\n".join(lines[i:]).strip("\n")

    def _str(k: str) -> str:
        v = fm.get(k, "")
        if isinstance(v, list) or v is None:
            return ""
        return v

    def _opt(k: str) -> str | None:
        v = fm.get(k)
        if isinstance(v, list):
            return None
        return v

    def _list(k: str) -> tuple[str, ...]:
        v = fm.get(k, ())
        if isinstance(v, list):
            return tuple(v)
        return ()

    created_at = datetime.fromisoformat(_str("created_at"))
    updated_at = datetime.fromisoformat(_str("updated_at"))
    # `status` and `created_by` are validated by pydantic on construction.
    return TaskNode(
        id=_str("id"),
        parent_id=_opt("parent_id"),
        title=_str("title"),
        rationale=_str("rationale"),
        acceptance=_str("acceptance"),
        relevant_paths=_list("relevant_paths"),
        depends_on=_list("depends_on"),
        children=_list("children"),
        status=_str("status"),  # type: ignore[arg-type]  # pydantic Literal check
        created_at=created_at,
        updated_at=updated_at,
        created_by=_str("created_by"),  # type: ignore[arg-type]
        commit_sha=_str("commit_sha"),
        notes=notes,
        standing=_str("standing") == "true",
        graph_version=int(_str("graph_version") or "0"),
    )


def _ancestor_chain(nodes: dict[str, TaskNode], node_id: str) -> list[str]:
    """Return the ids from the root down to a node, following parent pointers.

    Args:
        nodes: The graph by id.
        node_id: The node.

    Returns:
        `[root, ..., node_id]`; a missing ancestor ends the chain, so the deepest present
        node is treated as a root.

    Raises:
        ValueError: The parent chain has a cycle.
    """
    chain: list[str] = []
    cur: str | None = node_id
    seen: set[str] = set()
    while cur is not None:
        if cur not in nodes:
            break
        if cur in seen:
            raise ValueError(f"cycle in parent chain at {cur}")
        seen.add(cur)
        chain.append(cur)
        cur = nodes[cur].parent_id
    chain.reverse()
    return chain


def node_md_path(layout: SessionLayout, nodes: dict[str, TaskNode], node_id: str) -> Path:
    """Return a node's canonical file path, its ancestors as directory components.

    Args:
        layout: The session layout.
        nodes: The graph by id.
        node_id: The node.

    Returns:
        `<graph_dir>/<root>/.../<node_id>.md`.

    Raises:
        ValueError: The parent chain has a cycle.
    """
    chain = _ancestor_chain(nodes, node_id)
    rel = Path(*chain[:-1]) / f"{chain[-1]}.md"
    return layout.graph_dir / rel


def write_node(layout: SessionLayout, nodes: dict[str, TaskNode], node: TaskNode) -> None:
    """Write a node's file atomically at its canonical path, then drop any stale copy.

    The canonical path moves when `load_graph` re-roots an orphan; the new file is durable
    before the stale one is unlinked, so a crash between leaves a recoverable duplicate,
    never a missing node.

    Args:
        layout: The session layout.
        nodes: The graph by id.
        node: The node.
    """
    path = node_md_path(layout, nodes, node.id)
    mkdir_for_real_user(path.parent)
    if node.children:
        child_dir = path.with_suffix("")
        mkdir_for_real_user(child_dir)
    atomic_write(path, _dump_frontmatter(node))
    _prune_stale_node_files(layout, node.id, keep=path)


def _prune_stale_node_files(layout: SessionLayout, node_id: str, *, keep: Path) -> None:
    """Delete every other `<node_id>.md` under the graph dir.

    Args:
        layout: The session layout.
        node_id: The node.
        keep: The canonical file.
    """
    if not layout.graph_dir.is_dir():
        return
    keep_resolved = keep.resolve()
    for stale in layout.graph_dir.rglob(f"{node_id}.md"):
        if stale.resolve() == keep_resolved:
            continue
        with contextlib.suppress(OSError):
            stale.unlink()
            fsync_dir(stale.parent)


def load_graph(layout: SessionLayout) -> dict[str, TaskNode]:
    """Read every node file under the graph dir.

    A malformed or torn file is skipped with a note on stderr rather than bricking resume,
    and a child whose parent was skipped is re-rooted, so every `parent_id` resolves.

    Args:
        layout: The session layout.

    Returns:
        The nodes by id; empty without a graph dir.
    """
    nodes: dict[str, TaskNode] = {}
    if not layout.graph_dir.is_dir():
        return nodes
    for md in layout.graph_dir.rglob("*.md"):
        try:
            node = _parse_frontmatter(md.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            sys.stderr.write(f"agent6: skipping malformed node file {md}: {exc}\n")
            continue
        nodes[node.id] = node
    for node_id, node in list(nodes.items()):
        if node.parent_id is not None and node.parent_id not in nodes:
            sys.stderr.write(
                f"agent6: re-rooting orphan node {node_id} (parent {node.parent_id} missing)\n"
            )
            nodes[node_id] = node.model_copy(update={"parent_id": None})
    return nodes


def write_journal(layout: SessionLayout, entry: dict[str, object]) -> None:
    """Append one entry to the journal, stamped with the time when it carries none.

    Args:
        layout: The session layout.
        entry: The entry.
    """
    payload = dict(entry)
    payload.setdefault("ts", datetime.now(tz=UTC).isoformat())
    _append_line(layout.journal_path, json.dumps(payload, sort_keys=True))


def write_cursor(layout: SessionLayout, node_id: str | None) -> None:
    """Record the focused node.

    Args:
        layout: The session layout.
        node_id: The node, or None for no focus.
    """
    payload = json.dumps({"node_id": node_id})
    atomic_write(layout.cursor_path, payload)


def read_cursor(layout: SessionLayout) -> str | None:
    """Read the focused node's id.

    A malformed or unreadable cursor reads as none, said on stderr: a torn pointer must not
    brick resume, fork or `/undo`.

    Args:
        layout: The session layout.

    Returns:
        The id, or None when none is recorded.
    """
    if not layout.cursor_path.is_file():
        return None
    try:
        return _cursor_of(json.loads(layout.cursor_path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"agent6: ignoring malformed {layout.cursor_path}: {exc}\n")
        return None


def _cursor_of(raw: object) -> str | None:
    """Return the node id a parsed cursor file names.

    Args:
        raw: The parsed JSON.

    Returns:
        The id, or None.

    Raises:
        ValueError: The value is not an object with a string or null `node_id`.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"not an object: {raw!r}")
    if "node_id" not in raw:
        raise ValueError("no node_id")
    cursor = raw["node_id"]
    if cursor is None or isinstance(cursor, str):
        return cursor
    raise ValueError(f"node_id is {cursor!r}")


def list_checkpoint_turns(layout: SessionLayout) -> list[int]:
    """Return the recorded checkpoint turns, ascending.

    Args:
        layout: The session layout.

    Returns:
        The turn indices; empty without a checkpoints dir, which is how `agent6 fork` falls
        back to the snapshot alone.
    """
    cp_dir = layout.checkpoints_dir
    if not cp_dir.is_dir():
        return []
    turns: list[int] = []
    for p in cp_dir.glob("*.json"):
        try:
            turns.append(int(p.stem))
        except ValueError:
            continue  # a non-numeric stray file is not a checkpoint
    return sorted(turns)
