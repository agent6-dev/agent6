# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Claiming a task: the worker picks what it works next by marking it
in_progress, and the harness honours that while the task stays workable.

The frontier's own order is unchanged when nothing claims, so this costs a run
that ignores it nothing. The anti-grind counter still bites a worker that
claims its way around the graph without finishing anything.
"""

from __future__ import annotations

from pathlib import Path

from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft, UpdateStatusIntent
from agent6.sessions.layout import SessionLayout
from agent6.tools._dag_tools import update_task
from agent6.workflows._dag_focus import current_task_id
from agent6.workflows._guards import FocusGuard


def _graph(tmp_path: Path) -> tuple[GraphCurator, str, list[str]]:
    c = GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))
    root = c.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="run", created_by="user"))
    ).id
    kids = [
        c.add_subtask(
            AddSubtaskIntent(parent_id=root, draft=TaskNodeDraft(title=title, created_by="worker"))
        ).id
        for title in ("first", "second", "third")
    ]
    return c, root, kids


def test_the_frontier_picks_in_tree_order_when_nothing_claims(tmp_path: Path) -> None:
    c, _root, kids = _graph(tmp_path)

    assert current_task_id(c.nodes(), c.cursor()) == kids[0]


def test_marking_a_task_in_progress_claims_it(tmp_path: Path) -> None:
    c, _root, kids = _graph(tmp_path)

    c.update_status(UpdateStatusIntent(id=kids[2], new_status="in_progress"))

    assert c.cursor() == kids[2]
    assert current_task_id(c.nodes(), c.cursor()) == kids[2]


def test_a_claim_the_frontier_cannot_honour_says_so(tmp_path: Path) -> None:
    """An unworkable claim leaves the frontier's own pick standing, and the
    tool result says which happened rather than looking like it took."""
    c, _root, kids = _graph(tmp_path)
    c.add_subtask(
        AddSubtaskIntent(parent_id=kids[1], draft=TaskNodeDraft(title="leaf", created_by="worker"))
    )

    result = update_task(c, {"id": kids[1], "status": "in_progress"})

    assert "not workable yet" in result.to_wire()["note"]
    # A decomposed parent is not a unit of work, so the frontier's own pick
    # (the first ready subtask in tree order) stands.
    assert current_task_id(c.nodes(), c.cursor()) == kids[0]


def test_a_claim_that_took_says_so(tmp_path: Path) -> None:
    c, _root, kids = _graph(tmp_path)

    result = update_task(c, {"id": kids[1], "status": "in_progress"})

    assert result.to_wire()["note"] == "claimed: this is the task the harness works next"


def test_an_ordinary_status_carries_no_note(tmp_path: Path) -> None:
    c, _root, kids = _graph(tmp_path)

    assert "note" not in update_task(c, {"id": kids[0], "status": "passed"}).to_wire()


def test_claiming_around_the_graph_still_trips_the_anti_grind_nudge() -> None:
    """The counter used to reset on every focus change, so a worker claiming a
    different task each turn was never nudged. It resets on progress only."""
    guard = FocusGuard()
    fired = [
        guard.note(f"task-{i % 3}", standing=False, progressed=False)
        for i in range(60)  # switching between three open tasks, finishing none
    ]

    assert any(fired)


def test_finishing_a_task_still_resets_the_counter() -> None:
    guard = FocusGuard()
    for _ in range(19):
        guard.note("task-a", standing=False)

    assert guard.turns_on_task == 18
    guard.note("task-b", standing=False, progressed=True)
    assert guard.turns_on_task == 0
