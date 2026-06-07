# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `GraphCurator` mutations."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    ObsoleteIntent,
    RecordCommitIntent,
    ReorderChildrenIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.graph.storage import RunLayout


def _layout(tmp_path: Path) -> RunLayout:
    return RunLayout(state_dir=tmp_path / ".agent6", run_id="run1")


def _draft(title: str = "do thing", deps: tuple[str, ...] = ()) -> TaskNodeDraft:
    return TaskNodeDraft(title=title, depends_on=deps, created_by="planner")


def test_add_subtask_with_no_parent_creates_root(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("root")))
    assert n.parent_id is None
    assert n.status == "pending"
    assert c.graph_version >= 1


def test_add_subtask_unknown_parent_raises(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    with pytest.raises(CuratorError, match="unknown parent"):
        c.add_subtask(AddSubtaskIntent(parent_id="X" * 26, draft=_draft()))


def test_add_subtask_links_child_to_parent(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    p = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("parent")))
    ch = c.add_subtask(AddSubtaskIntent(parent_id=p.id, draft=_draft("child")))
    assert c.get(p.id).children == (ch.id,)


def test_update_status_passed_then_obsolete_ok_other_rejected(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="passed"))
    # passed -> obsolete is fine
    c.update_status(UpdateStatusIntent(id=n.id, new_status="obsolete"))
    # but passed -> anything else would be rejected, set it back first
    n2 = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.update_status(UpdateStatusIntent(id=n2.id, new_status="passed"))
    with pytest.raises(CuratorError):
        c.update_status(UpdateStatusIntent(id=n2.id, new_status="failed"))


def test_add_dependency_detects_cycle(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    a = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("a")))
    b = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("b")))
    c.add_dependency(AddDependencyIntent(id=b.id, depends_on=a.id))
    with pytest.raises(CuratorError, match="cycle"):
        c.add_dependency(AddDependencyIntent(id=a.id, depends_on=b.id))


def test_reorder_children_requires_permutation(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    p = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("p")))
    a = c.add_subtask(AddSubtaskIntent(parent_id=p.id, draft=_draft("a")))
    b = c.add_subtask(AddSubtaskIntent(parent_id=p.id, draft=_draft("b")))
    p2 = c.reorder_children(ReorderChildrenIntent(parent_id=p.id, new_order=(b.id, a.id)))
    assert p2.children == (b.id, a.id)
    with pytest.raises(CuratorError, match="permutation"):
        c.reorder_children(ReorderChildrenIntent(parent_id=p.id, new_order=(a.id,)))


def test_obsolete_and_record_commit(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.record_commit(RecordCommitIntent(id=n.id, sha="abcd1234"))
    c.obsolete(ObsoleteIntent(id=n.id, reason="user-canceled"))
    final = c.get(n.id)
    assert final.commit_sha == "abcd1234"
    assert final.status == "obsolete"
    assert "user-canceled" in final.notes


def test_set_cursor_persists_and_validates(tmp_path: Path) -> None:
    c = GraphCurator(_layout(tmp_path))
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft()))
    c.set_cursor(SetCursorIntent(id=n.id))
    assert c.cursor() == n.id
    with pytest.raises(CuratorError):
        c.set_cursor(SetCursorIntent(id="Z" * 26))
    c.set_cursor(SetCursorIntent(id=None))
    assert c.cursor() is None


def test_curator_reload_preserves_state(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = GraphCurator(layout)
    n = c.add_subtask(AddSubtaskIntent(parent_id=None, draft=_draft("persist me")))
    c.update_status(UpdateStatusIntent(id=n.id, new_status="in_progress"))
    v_before = c.graph_version
    c2 = GraphCurator(layout)
    again = c2.get(n.id)
    assert again.status == "in_progress"
    assert c2.graph_version == v_before
