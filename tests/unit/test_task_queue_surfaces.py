# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta

"""A queued task reads as the operator's on every surface.

It arrives without the model having read anything, so the transcript says it
arrived, the task tree says whose it is, and a run that ends over one names it
as theirs rather than as work the model chose to leave.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent6.graph.models import TaskNode
from agent6.viewmodel.log_line import format_log_line
from agent6.viewmodel.state import task_tree_views
from agent6.viewmodel.transcript import fold_transcript
from agent6.workflows._advice import open_subtasks, with_open_tasks

_NOW = datetime(2026, 9, 16, tzinfo=UTC)


def _node(**kw: Any) -> TaskNode:
    base: dict[str, Any] = {
        "id": "01M2M10ABAH7YRMW5YXKY4MDEX",
        "parent_id": "01M2M10AB8RER75QYT5QYHQ2JK",
        "title": "add a --json flag",
        "created_by": "worker",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return TaskNode(**(base | kw))


def test_the_tree_view_carries_who_added_a_task() -> None:
    """The graph.update snapshot grew `created_by` so a surface can tell the
    operator's queued work from the model's own breakdown."""
    nodes = {
        "root": {"title": "run", "parent_id": None, "children": ["a", "b"], "created_by": "user"},
        "a": {"title": "model's", "parent_id": "root", "children": [], "created_by": "worker"},
        "b": {"title": "yours", "parent_id": "root", "children": [], "created_by": "user"},
    }

    views = {v.title: v for v in task_tree_views(nodes, cursor=None)}

    assert views["yours"].created_by == "user"
    assert views["model's"].created_by == "worker"


def test_an_old_run_dir_reads_as_the_models() -> None:
    """A snapshot written before the field existed simply lacks it."""
    nodes = {"a": {"title": "old", "parent_id": None, "children": []}}

    assert task_tree_views(nodes, cursor=None)[0].created_by == ""


def test_the_transcript_says_a_task_arrived() -> None:
    items = fold_transcript(
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "loop.task.queued", "id": "01M2", "title": "add a --json flag"},
        ]
    )

    assert any("task queued: add a --json flag" in (i.body or "") for i in items)


def test_the_log_line_names_the_task() -> None:
    line = format_log_line({"type": "loop.task.queued", "title": "add a --json flag"})

    assert "add a --json flag" in line


def test_an_end_over_an_operator_task_names_it_as_theirs() -> None:
    nodes = {
        "yours": _node(created_by="user", title="add a --json flag"),
        "mine": _node(created_by="worker", title="refactor the parser"),
    }

    summary = with_open_tasks("stopped", open_subtasks(nodes))

    assert "add a --json flag (queued by you)" in summary
    assert "refactor the parser" in summary
    assert "refactor the parser (queued by you)" not in summary
