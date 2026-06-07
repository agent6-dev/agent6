# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Storage + frontmatter round-trip tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agent6.graph.models import TaskNode
from agent6.graph.storage import RunLayout, load_graph, write_node


def _mk_node(
    nid: str,
    *,
    parent: str | None = None,
    title: str = "t",
    rationale: str = "r",
    relevant_paths: tuple[str, ...] = (),
    children: tuple[str, ...] = (),
) -> TaskNode:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return TaskNode(
        id=nid,
        parent_id=parent,
        title=title,
        rationale=rationale,
        acceptance="a",
        relevant_paths=relevant_paths,
        depends_on=(),
        children=children,
        status="pending",
        created_at=now,
        updated_at=now,
        created_by="planner",
    )


def test_layout_ensure_creates_dirs(tmp_path: Path) -> None:
    layout = RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")
    layout.ensure()
    assert layout.graph_dir.is_dir()
    assert layout.snapshots_dir.is_dir()
    assert layout.transcripts_dir.is_dir()


def test_write_and_load_single_node(tmp_path: Path) -> None:
    layout = RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")
    layout.ensure()
    n = _mk_node("0" * 25 + "A", relevant_paths=("src/a.py", "src/b.py"))
    write_node(layout, {n.id: n}, n)
    loaded = load_graph(layout)
    assert n.id in loaded
    got = loaded[n.id]
    assert got.title == "t"
    assert got.relevant_paths == ("src/a.py", "src/b.py")


def test_frontmatter_quotes_special_chars(tmp_path: Path) -> None:
    layout = RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")
    layout.ensure()
    n = _mk_node(
        "0" * 25 + "B",
        title='has "quotes" and \\backslash',
        rationale="line1\nline2",
    )
    write_node(layout, {n.id: n}, n)
    loaded = load_graph(layout)
    assert loaded[n.id].title == 'has "quotes" and \\backslash'
    assert loaded[n.id].rationale == "line1\nline2"


def test_load_graph_reconstructs_parent_dir_layout(tmp_path: Path) -> None:
    layout = RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")
    layout.ensure()
    root = _mk_node("0" * 25 + "C", children=("0" * 25 + "D",))
    child = _mk_node("0" * 25 + "D", parent=root.id)
    nodes = {root.id: root, child.id: child}
    write_node(layout, nodes, root)
    write_node(layout, nodes, child)
    loaded = load_graph(layout)
    assert loaded[root.id].children == (child.id,)
    assert loaded[child.id].parent_id == root.id
