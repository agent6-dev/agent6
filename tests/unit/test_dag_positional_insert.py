# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`add_task(after=...)`: the model can place work, not only append it."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.sessions.layout import SessionLayout


def _curator(tmp_path: Path) -> GraphCurator:
    layout = SessionLayout(state_dir=tmp_path / "state", session_id="runny-one-AAAAAA")
    layout.ensure()
    return GraphCurator(layout)


def _add(cur: GraphCurator, parent: str | None, title: str, after: str | None = None) -> str:
    return cur.add_subtask(
        AddSubtaskIntent(
            parent_id=parent,
            draft=TaskNodeDraft(title=title, created_by="worker"),
            after=after,
        )
    ).id


def test_a_task_lands_after_the_sibling_it_names(tmp_path: Path) -> None:
    """Ordering was faked with add_dependency because add_task only appended:
    inserting work between two steps meant re-planning the whole tail."""
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    first = _add(cur, root, "first")
    last = _add(cur, root, "last")
    middle = _add(cur, root, "middle", after=first)
    assert cur.get(root).children == (first, middle, last)


def test_no_position_still_appends(tmp_path: Path) -> None:
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    a = _add(cur, root, "a")
    b = _add(cur, root, "b")
    assert cur.get(root).children == (a, b)


def test_after_must_name_a_sibling(tmp_path: Path) -> None:
    """A position under a different parent is a mistake, not a move: refusing
    keeps the children list a faithful order of THIS parent's work."""
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    branch = _add(cur, root, "branch")
    elsewhere = _add(cur, branch, "elsewhere")
    with pytest.raises(CuratorError, match="after"):
        _add(cur, root, "misplaced", after=elsewhere)


def test_the_inserted_task_is_focused_next(tmp_path: Path) -> None:
    """The point of placing work: the frontier surfaces it in its new
    position, not at the end of the list."""
    from agent6.workflows._dag_focus import first_ready_subtask

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    first = _add(cur, root, "first")
    _add(cur, root, "last")
    middle = _add(cur, root, "middle", after=first)
    nodes = cur.nodes()
    assert nodes[root].children[1] == middle
    # first is still open, so it stays the focus; the inserted task is next in
    # line rather than behind "last".
    assert first_ready_subtask(nodes) == first
    assert [c for c in nodes[root].children][1] == middle
