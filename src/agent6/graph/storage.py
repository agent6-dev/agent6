# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""On-disk format for the task graph.

The canonical form is one markdown file per node with a YAML frontmatter header
holding the structured fields. Files are laid out to mirror the parent→child
tree: a node with children has a sibling directory of the same id.

    <run-dir>/
      manifest.json
      graph/<root>.md
      graph/<root>/<child>.md
      graph/<root>/<child>/<grandchild>.md
      graph.jsonl          # append-only journal of every mutation
      graph.dot            # derived; rebuilt on every mutation
      cursor.json          # which node is currently in_progress; for resume
      snapshots/<node>.json

All writes go through `_atomic_write`, which writes a tmp file in the same
directory, fsyncs it, then renames into place and fsyncs the parent directory.
The curator additionally holds an fcntl flock on `.lock` for the full duration
of a mutation, so concurrent intents serialize cleanly even though we never
expect more than one curator process per run.

YAML is parsed by hand (no PyYAML dep), the frontmatter we emit is restricted
to a single-level mapping of scalars and lists-of-strings, which is trivial to
serialize and parse deterministically.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent6.graph.models import NodeSnapshot, TaskNode
from agent6.portable import fsync_dir, lock_exclusive, unlock

# ---- run layout -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Filesystem layout for one `agent6 run`.

    ``state_dir`` is the resolved run-state base
    (``$XDG_STATE_HOME/agent6/<repo-id>`` by default, or wherever
    ``[agent6].state_dir`` points). See ``agent6.paths.state_dir``.
    """

    state_dir: Path
    run_id: str
    # Top-level bucket under state_dir. "runs" for `agent6 run`/`plan`; "asks"
    # for `agent6 ask` so read-only Q&A sessions stay separate from real runs.
    subdir: str = "runs"

    @property
    def run_dir(self) -> Path:
        return self.state_dir / self.subdir / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def graph_dir(self) -> Path:
        return self.run_dir / "graph"

    @property
    def journal_path(self) -> Path:
        return self.run_dir / "graph.jsonl"

    @property
    def dot_path(self) -> Path:
        return self.run_dir / "graph.dot"

    @property
    def cursor_path(self) -> Path:
        return self.run_dir / "cursor.json"

    @property
    def lock_path(self) -> Path:
        return self.run_dir / ".lock"

    @property
    def snapshots_dir(self) -> Path:
        return self.run_dir / "snapshots"

    @property
    def checkpoints_dir(self) -> Path:
        """Append-only per-turn resume checkpoints (``<NNNN>.json``).

        Each holds the same payload as ``loop_state.json`` for that turn plus
        the workspace ``head_sha`` and curator ``graph_version`` at the turn, so
        ``agent6 fork`` can roll a run back to turn N. ``loop_state.json`` stays
        the "latest" pointer for plain ``resume``.
        """
        return self.run_dir / "checkpoints"

    @property
    def transcripts_dir(self) -> Path:
        return self.run_dir / "transcripts"

    @property
    def logs_path(self) -> Path:
        return self.run_dir / "logs.jsonl"

    @property
    def user_inputs_path(self) -> Path:
        """JSONL audit log of every interactive prompt + the operator's answer.

        Separate from logs.jsonl so the human-decision trail stays readable
        without grepping through machine telemetry.
        """
        return self.run_dir / "user_inputs.jsonl"

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    def checkpoint_path(self, turn: int) -> Path:
        """Path of the checkpoint for ``turn`` (zero-padded to 4 digits)."""
        return self.checkpoints_dir / f"{turn:04d}.json"


# ---- atomic write + flock helpers ----------------------------------------


def _atomic_write(path: Path, data: str | bytes) -> None:
    """Write data via tmp file + rename, fsyncing both file and parent dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(data, bytes):
        with tmp.open("wb") as fb:
            fb.write(data)
            fb.flush()
            os.fsync(fb.fileno())
    else:
        with tmp.open("w", encoding="utf-8") as ft:
            ft.write(data)
            ft.flush()
            os.fsync(ft.fileno())
    tmp.replace(path)
    # fsync parent dir so the rename is durable (no-op on Windows).
    fsync_dir(path.parent)


def _append_line(path: Path, line: str) -> None:
    """Append one line atomically (single write())."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (line if line.endswith("\n") else line + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, entry: dict[str, object]) -> None:
    """Append one JSON object as a line to ``path`` (durable single write).

    Public wrapper over the atomic append used for the per-repo fork
    ``lineage.jsonl``; the caller supplies a fully-formed entry (including any
    timestamp) so this stays a pure I/O helper with no clock dependency."""
    _append_line(path, json.dumps(entry, sort_keys=True))


@contextmanager
def flock(path: Path) -> Generator[None]:
    """fcntl exclusive lock on ``path``. Creates the file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        lock_exclusive(fd, blocking=True)
        yield
    finally:
        try:
            unlock(fd)
        finally:
            os.close(fd)


# ---- YAML frontmatter (handwritten, restricted dialect) ------------------


def _yaml_quote(s: str) -> str:
    """Quote a scalar so it round-trips through `_yaml_unquote`."""
    # Always double-quote to keep round-trip simple; escape backslash, quotes,
    # and BOTH newline chars. `\r` must be escaped too: the parser splits on
    # "\n" only, but an un-escaped `\r` would otherwise be emitted literally and
    # an adversarial title/notes value could smuggle one in. Other Unicode line
    # separators (U+2028/2029, \v, \f, NEL, …) survive because the parser no
    # longer treats them as line breaks (it uses str.split("\n"), not
    # str.splitlines()).
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def _yaml_unquote(s: str) -> str:
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
    """Render a node to its canonical YAML frontmatter + freeform body form."""
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
    fm.append("---")
    fm.append("")
    fm.append(node.notes if node.notes else "")
    return "\n".join(fm) + "\n"


def _parse_frontmatter(text: str) -> TaskNode:
    """Parse the YAML frontmatter back into a TaskNode. Strict."""
    # Split on "\n" only (the exact inverse of `_dump_frontmatter`'s
    # "\n".join). str.splitlines() additionally breaks on \r, \v, \f, NEL,
    # U+2028/2029, \x1c-\x1e, so a scalar containing any of those (which an
    # adversarial LLM can put in a task title via add_task) would be read back
    # as two physical lines and crash the parser, permanently bricking resume.
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
        # close any in-progress list
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
    )


# ---- node path resolution ------------------------------------------------


def _ancestor_chain(nodes: dict[str, TaskNode], node_id: str) -> list[str]:
    """Return [root, ..., node_id] following parent pointers."""
    chain: list[str] = []
    cur: str | None = node_id
    seen: set[str] = set()
    while cur is not None:
        if cur not in nodes:
            # Orphaned ancestor: its file was skipped as malformed by load_graph
            # (or a node carries a dangling parent_id). Terminate the chain here
            # and treat the deepest present node as a root, instead of KeyError-ing
            # on the missing parent.
            break
        if cur in seen:
            raise ValueError(f"cycle in parent chain at {cur}")
        seen.add(cur)
        chain.append(cur)
        cur = nodes[cur].parent_id
    chain.reverse()
    return chain


def node_md_path(layout: RunLayout, nodes: dict[str, TaskNode], node_id: str) -> Path:
    """Resolve the canonical .md path for a node based on its ancestor chain."""
    chain = _ancestor_chain(nodes, node_id)
    # All ancestors above the last become directory components.
    rel = Path(*chain[:-1]) / f"{chain[-1]}.md"
    return layout.graph_dir / rel


# ---- whole-graph read / write --------------------------------------------


def write_node(layout: RunLayout, nodes: dict[str, TaskNode], node: TaskNode) -> None:
    """Atomically write a node's .md file at its canonical path."""
    path = node_md_path(layout, nodes, node.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # If the node has children, ensure the matching directory exists too.
    if node.children:
        child_dir = path.with_suffix("")
        child_dir.mkdir(exist_ok=True)
    _atomic_write(path, _dump_frontmatter(node))
    # Remove any STALE .md for this same id at a different path. The canonical
    # path can move -- e.g. load_graph re-roots an orphan (parent_id -> None when
    # its parent file was malformed/skipped), shifting the node from a nested
    # <parent>/<id>.md to a root <id>.md. The new file is written above; the old
    # nested one would otherwise linger and make load_graph's rglob find TWO .md
    # for one id (nondeterministic which wins). Crash-safety ordering: the new
    # canonical file is durable BEFORE we unlink the stale one, so a crash here
    # leaves at worst the recoverable pre-fix duplicate, never a missing node.
    _prune_stale_node_files(layout, node.id, keep=path)


def _prune_stale_node_files(layout: RunLayout, node_id: str, *, keep: Path) -> None:
    """Delete any other ``<node_id>.md`` under graph/ except ``keep``."""
    if not layout.graph_dir.is_dir():
        return
    keep_resolved = keep.resolve()
    for stale in layout.graph_dir.rglob(f"{node_id}.md"):
        if stale.resolve() == keep_resolved:
            continue
        with contextlib.suppress(OSError):
            stale.unlink()


def load_graph(layout: RunLayout) -> dict[str, TaskNode]:
    """Read every .md file under ``graph/`` and return a {id: TaskNode} map."""
    nodes: dict[str, TaskNode] = {}
    if not layout.graph_dir.is_dir():
        return nodes
    for md in layout.graph_dir.rglob("*.md"):
        try:
            node = _parse_frontmatter(md.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            # A hand-edited or torn node file must not brick resume; the rest of
            # the graph is still loadable, so degrade to a missing node (mirrors
            # the torn-line tolerance in iter_journal / _iter_recent_journal).
            sys.stderr.write(f"agent6: skipping malformed node file {md}: {exc}\n")
            continue
        nodes[node.id] = node
    # Reconcile integrity: skipping a malformed PARENT node above would leave its
    # children with a dangling parent_id. Re-root such orphans (parent_id -> None)
    # so every parent_id resolves and reads of parent_id can't observe a missing
    # node. (node_md_path is independently defended in _ancestor_chain.)
    for node_id, node in list(nodes.items()):
        if node.parent_id is not None and node.parent_id not in nodes:
            sys.stderr.write(
                f"agent6: re-rooting orphan node {node_id} (parent {node.parent_id} missing)\n"
            )
            nodes[node_id] = node.model_copy(update={"parent_id": None})
    return nodes


def write_journal(layout: RunLayout, entry: dict[str, object]) -> None:
    """Append one JSON event to graph.jsonl."""
    payload = dict(entry)
    payload.setdefault("ts", datetime.now(tz=UTC).isoformat())
    _append_line(layout.journal_path, json.dumps(payload, sort_keys=True))


def write_cursor(layout: RunLayout, node_id: str | None) -> None:
    payload = json.dumps({"node_id": node_id})
    _atomic_write(layout.cursor_path, payload)


def read_cursor(layout: RunLayout) -> str | None:
    if not layout.cursor_path.is_file():
        return None
    raw = json.loads(layout.cursor_path.read_text(encoding="utf-8"))
    cursor = raw.get("node_id")
    if cursor is None or isinstance(cursor, str):
        return cursor
    raise ValueError(f"malformed cursor.json: {raw!r}")


def write_snapshot(layout: RunLayout, node_id: str, snap: NodeSnapshot) -> None:
    path = layout.snapshots_dir / f"{node_id}.json"
    _atomic_write(path, snap.model_dump_json(indent=2))


def read_snapshot(layout: RunLayout, node_id: str) -> NodeSnapshot | None:
    path = layout.snapshots_dir / f"{node_id}.json"
    if not path.is_file():
        return None
    return NodeSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def list_checkpoint_turns(layout: RunLayout) -> list[int]:
    """Return the recorded checkpoint turn indices, ascending.

    Empty when the run predates the checkpoint store (no ``checkpoints/`` dir),
    which is how ``agent6 fork`` detects an old run and falls back to forking
    from ``loop_state.json`` only.
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


def write_dot(layout: RunLayout, nodes: dict[str, TaskNode]) -> None:
    """Render the graph to Graphviz DOT for visual debugging."""
    lines: list[str] = ["digraph agent6 {", "  rankdir=LR;"]
    for n in nodes.values():
        label = n.title.replace('"', "'")[:60]
        color = {
            "pending": "lightgray",
            "in_progress": "khaki",
            "passed": "palegreen",
            "failed": "salmon",
            "skipped": "lightblue",
            "obsolete": "gray60",
        }.get(n.status, "white")
        lines.append(
            f'  "{n.id}" [label="{label}\\n[{n.status}]", style=filled, fillcolor={color}];'
        )
    for n in nodes.values():
        for child_id in n.children:
            lines.append(f'  "{n.id}" -> "{child_id}";')
        for dep in n.depends_on:
            lines.append(f'  "{dep}" -> "{n.id}" [style=dashed, color=blue];')
    lines.append("}")
    _atomic_write(layout.dot_path, "\n".join(lines) + "\n")


def iter_journal(layout: RunLayout) -> Iterable[dict[str, object]]:
    """Yield every recorded journal entry in order."""
    if not layout.journal_path.is_file():
        return
    for raw in layout.journal_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError:
            # Tolerate a torn final line from a crash mid-append; the node .md
            # files are the source of truth, so a corrupt journal entry must not
            # crash readers (history graph, curator startup).
            sys.stderr.write(f"agent6: skipping malformed journal line: {stripped[:80]!r}\n")
            continue
