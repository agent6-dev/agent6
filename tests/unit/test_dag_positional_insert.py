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


def test_list_tasks_reads_back_the_order_the_frontier_executes(tmp_path: Path) -> None:
    """The point of placing a task is that the model can insert work between
    two steps. `list_tasks` iterated the node MAP -- insertion order live, and
    filesystem order after a resume -- so the model read back a plan it did not
    write: the task it placed second showed up last, and every id it planned
    around had moved by the next session.

    Both surfaces walk the same tree order now; the human-facing renderers
    already did.
    """
    from agent6.graph.order import tree_order
    from agent6.tools._dag_tools import list_tasks

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    first = _add(cur, root, "first")
    last = _add(cur, root, "last")
    middle = _add(cur, root, "middle", after=first)
    assert cur.get(root).children == (first, middle, last)

    listed = [t["id"] for t in list_tasks(cur, {}).to_wire()["tasks"]]
    assert listed == tree_order(cur.nodes()), "the model reads a different order than it planned"
    assert listed == [root, first, middle, last]


def test_standing_task_is_the_fallback_never_the_frontier(tmp_path: Path) -> None:
    """A standing task runs only when every ordinary subtask is settled:
    pending work always outranks it, new work preempts it (the cursor on a
    standing node yields to a fresh ordinary task), and it never passes."""
    import pytest

    from agent6.graph.curator import CuratorError
    from agent6.graph.models import TaskNodeDraft, UpdateStatusIntent
    from agent6.workflows._dag_focus import current_task_id, first_ready_subtask

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    standing = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root,
            draft=TaskNodeDraft(title="keep testing", created_by="worker", standing=True),
        )
    ).id
    work = _add(cur, root, "real work")
    # Ordinary work outranks the standing node.
    assert first_ready_subtask(cur.nodes()) == work
    cur.update_status(UpdateStatusIntent(id=work, new_status="passed"))
    # The queue drained: the standing node is the fallback.
    assert first_ready_subtask(cur.nodes()) == standing
    # New work preempts it, even with the cursor parked on the standing node.
    late = _add(cur, root, "late arrival")
    assert current_task_id(cur.nodes(), standing) == late
    # It never passes; retiring is skipped/obsolete.
    with pytest.raises(CuratorError, match="never passes"):
        cur.update_status(UpdateStatusIntent(id=standing, new_status="passed"))
    cur.update_status(UpdateStatusIntent(id=late, new_status="passed"))
    cur.update_status(UpdateStatusIntent(id=standing, new_status="skipped"))
    assert first_ready_subtask(cur.nodes()) is None


def test_standing_survives_the_storage_round_trip(tmp_path: Path) -> None:
    from agent6.graph.curator import GraphCurator
    from agent6.graph.models import TaskNodeDraft

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    sid = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root,
            draft=TaskNodeDraft(title="hunt bugs", created_by="worker", standing=True),
        )
    ).id
    # A fresh curator over the same layout re-reads the files from disk.
    layout = SessionLayout(state_dir=tmp_path / "state", session_id="runny-one-AAAAAA")
    reloaded = GraphCurator(layout)
    assert reloaded.nodes()[sid].standing is True
    assert reloaded.nodes()[root].standing is False


def test_the_model_cannot_retire_the_operators_standing_goal(tmp_path: Path) -> None:
    """`run --standing` seeds the goal with created_by="steering"; update_task
    (the model's route) refuses to skip/obsolete it, or the never-finishing
    fallback becomes an ordinary early finish (seen live: the goal worked
    once, marked skipped, run over at half budget). The model's OWN standing
    task (add_task standing=true) stays retirable; the curator itself stays
    permissive (the operator's surfaces go through it)."""
    from agent6.graph.models import TaskNodeDraft, UpdateStatusIntent
    from agent6.tools._dag_tools import update_task
    from agent6.tools.errors import ToolError

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    operators = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root,
            draft=TaskNodeDraft(
                title="docstrings everywhere", created_by="steering", standing=True
            ),
        )
    ).id
    models_own = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root,
            draft=TaskNodeDraft(title="keep tests green", created_by="worker", standing=True),
        )
    ).id
    for status in ("skipped", "obsolete"):
        with pytest.raises(ToolError, match="operator's standing goal"):
            update_task(cur, {"id": operators, "status": status})
    assert update_task(cur, {"id": models_own, "status": "skipped"}).status == "skipped"
    # The curator (the operator's route) can still retire it.
    cur.update_status(UpdateStatusIntent(id=operators, new_status="skipped"))
