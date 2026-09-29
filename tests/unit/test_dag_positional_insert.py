# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`add_task(after=...)`: the model can place work, not only append it."""

from __future__ import annotations

import pathlib

import pytest

from agent6.graph import curator, models
from agent6.sessions import layout as sessions_layout


def _curator(tmp_path: pathlib.Path) -> curator.GraphCurator:
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / "state", session_id="runny-one-AAAAAA"
    )
    layout.ensure()
    return curator.GraphCurator(layout)


def _add(
    cur: curator.GraphCurator, parent: str | None, title: str, after: str | None = None
) -> str:
    return cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=parent,
            draft=models.TaskNodeDraft(title=title, created_by="worker"),
            after=after,
        )
    ).id


def test_a_task_lands_after_the_sibling_it_names(tmp_path: pathlib.Path) -> None:
    """A task lands after the sibling it names, so work can be inserted between two steps."""
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    first = _add(cur, root, "first")
    last = _add(cur, root, "last")
    middle = _add(cur, root, "middle", after=first)
    assert cur.get(root).children == (first, middle, last)


def test_no_position_still_appends(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    a = _add(cur, root, "a")
    b = _add(cur, root, "b")
    assert cur.get(root).children == (a, b)


def test_after_must_name_a_sibling(tmp_path: pathlib.Path) -> None:
    """`after` must name a sibling; a position under another parent is refused."""
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    branch = _add(cur, root, "branch")
    elsewhere = _add(cur, branch, "elsewhere")
    with pytest.raises(curator.CuratorError, match="after"):
        _add(cur, root, "misplaced", after=elsewhere)


def test_the_inserted_task_is_focused_next(tmp_path: pathlib.Path) -> None:
    """The frontier surfaces an inserted task in its new position, not at the end."""
    from agent6.harness import _dag_focus

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    first = _add(cur, root, "first")
    _add(cur, root, "last")
    middle = _add(cur, root, "middle", after=first)
    nodes = cur.nodes()
    assert nodes[root].children[1] == middle
    # first is still open, so it stays the focus; the inserted task is next in line.
    assert _dag_focus.first_ready_subtask(nodes) == first
    assert [c for c in nodes[root].children][1] == middle


def test_list_tasks_reads_back_the_order_the_frontier_executes(tmp_path: pathlib.Path) -> None:
    """`list_tasks` reads back the tree order the frontier executes, not the node map's order."""
    from agent6.graph import order
    from agent6.tools import _dag_tools

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    first = _add(cur, root, "first")
    last = _add(cur, root, "last")
    middle = _add(cur, root, "middle", after=first)
    assert cur.get(root).children == (first, middle, last)

    listed = [t["id"] for t in _dag_tools.list_tasks(cur, {}).to_wire()["tasks"]]
    assert listed == order.tree_order(cur.nodes()), (
        "the model reads a different order than it planned"
    )
    assert listed == [root, first, middle, last]


def test_standing_task_is_the_fallback_never_the_frontier(tmp_path: pathlib.Path) -> None:
    """A standing task runs only when every ordinary subtask is settled, and never passes.

    New work preempts it: the cursor on a standing node yields to a fresh ordinary task.
    """
    import pytest

    from agent6.harness import _dag_focus

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    standing = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(title="keep testing", created_by="steering", standing=True),
        )
    ).id
    work = _add(cur, root, "real work")
    # Ordinary work outranks the standing node.
    assert _dag_focus.first_ready_subtask(cur.nodes()) == work
    cur.update_status(models.UpdateStatusIntent(id=work, new_status="passed"))
    # The queue drained: the standing node is the fallback.
    assert _dag_focus.first_ready_subtask(cur.nodes()) == standing
    # New work preempts it, even with the cursor parked on the standing node.
    late = _add(cur, root, "late arrival")
    assert _dag_focus.current_task_id(cur.nodes(), standing) == late
    # It never passes; retiring is skipped/obsolete.
    with pytest.raises(curator.CuratorError, match="never passes"):
        cur.update_status(models.UpdateStatusIntent(id=standing, new_status="passed"))
    cur.update_status(models.UpdateStatusIntent(id=late, new_status="passed"))
    cur.update_status(models.UpdateStatusIntent(id=standing, new_status="skipped"))
    assert _dag_focus.first_ready_subtask(cur.nodes()) is None


def test_standing_survives_the_storage_round_trip(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    sid = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(title="hunt bugs", created_by="steering", standing=True),
        )
    ).id
    # A fresh curator over the same layout re-reads the files from disk.
    layout = sessions_layout.SessionLayout(
        state_dir=tmp_path / "state", session_id="runny-one-AAAAAA"
    )
    reloaded = curator.GraphCurator(layout)
    assert reloaded.nodes()[sid].standing is True
    assert reloaded.nodes()[root].standing is False


def test_the_model_cannot_retire_the_operators_standing_goal(tmp_path: pathlib.Path) -> None:
    """update_task refuses to skip or obsolete the operator's standing goal.

    A model asking for a standing task of its own gets an ordinary one, which stays retirable.
    """
    from agent6.tools import _dag_tools, errors

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    operators = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(
                title="docstrings everywhere", created_by="steering", standing=True
            ),
        )
    ).id
    models_own = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(
                title="keep tests green", created_by="worker", standing=True
            ),
        )
    ).id
    assert not cur.get(models_own).standing
    for status in ("skipped", "obsolete"):
        with pytest.raises(errors.ToolError, match="operator's standing goal"):
            _dag_tools.update_task(cur, {"id": operators, "status": status})
    assert _dag_tools.update_task(cur, {"id": models_own, "status": "skipped"}).status == "skipped"
    # The curator (the operator's route) can still retire it.
    cur.update_status(models.UpdateStatusIntent(id=operators, new_status="skipped"))


def test_a_parent_over_a_failed_child_is_focused_and_its_refusal_names_the_child(
    tmp_path: pathlib.Path,
) -> None:
    """A parent over a failed child is focused, and its refusal to pass names the child."""
    import pytest

    from agent6.harness import _dag_focus

    cur = _curator(tmp_path)
    root = _add(cur, None, "root")
    phase = _add(cur, root, "phase")
    step = _add(cur, phase, "step")
    cur.update_status(models.UpdateStatusIntent(id=step, new_status="failed"))
    assert _dag_focus.first_ready_subtask(cur.nodes()) == phase
    with pytest.raises(curator.CuratorError, match=f"unresolved children \\({step}\\)"):
        cur.update_status(models.UpdateStatusIntent(id=phase, new_status="passed"))
    cur.update_status(models.UpdateStatusIntent(id=step, new_status="skipped"))
    cur.update_status(models.UpdateStatusIntent(id=phase, new_status="passed"))
    assert cur.nodes()[phase].status == "passed"
