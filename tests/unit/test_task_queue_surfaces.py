# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta

"""A queued task reads as the operator's on every surface.

It arrives without the model having read anything, so the transcript says it
arrived, the task tree says whose it is, and a run that ends over one names it
as theirs rather than as work the model chose to leave.
"""

from __future__ import annotations

import datetime
from typing import Any

from agent6.graph import models
from agent6.harness import _advice
from agent6.viewmodel import log_line, state, transcript

_NOW = datetime.datetime(2026, 9, 16, tzinfo=datetime.UTC)


def _node(**kw: Any) -> models.TaskNode:
    base: dict[str, Any] = {
        "id": "01M2M10ABAH7YRMW5YXKY4MDEX",
        "parent_id": "01M2M10AB8RER75QYT5QYHQ2JK",
        "title": "add a --json flag",
        "created_by": "worker",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return models.TaskNode(**(base | kw))


def test_the_tree_view_carries_who_added_a_task() -> None:
    """The graph.update snapshot carries `created_by`, telling queued work from the model's own."""
    nodes = {
        "root": {"title": "run", "parent_id": None, "children": ["a", "b"], "created_by": "user"},
        "a": {"title": "model's", "parent_id": "root", "children": [], "created_by": "worker"},
        "b": {"title": "yours", "parent_id": "root", "children": [], "created_by": "user"},
    }

    views = {v.title: v for v in state.task_tree_views(nodes, cursor=None)}

    assert views["yours"].created_by == "user"
    assert views["model's"].created_by == "worker"


def test_an_old_run_dir_reads_as_the_models() -> None:
    """A snapshot written before the field existed simply lacks it."""
    nodes = {"a": {"title": "old", "parent_id": None, "children": []}}

    assert state.task_tree_views(nodes, cursor=None)[0].created_by == ""


def test_the_transcript_says_a_task_arrived() -> None:
    items = transcript.fold_transcript(
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {"type": "loop.task.queued", "id": "01M2", "title": "add a --json flag"},
        ]
    )

    assert any("task queued: add a --json flag" in (i.body or "") for i in items)


def test_the_log_line_names_the_task() -> None:
    line = log_line.format_log_line({"type": "loop.task.queued", "title": "add a --json flag"})

    assert "add a --json flag" in line


def test_an_end_over_an_operator_task_names_it_as_theirs() -> None:
    nodes = {
        "yours": _node(created_by="user", title="add a --json flag"),
        "mine": _node(created_by="worker", title="refactor the parser"),
    }

    summary = _advice.with_open_tasks("stopped", _advice.open_subtasks(nodes))

    assert "add a --json flag (queued by you)" in summary
    assert "refactor the parser" in summary
    assert "refactor the parser (queued by you)" not in summary


def test_every_surface_reads_one_owner_note() -> None:
    """One owner note on the view marks queued and standing tasks, with the id leading every line.

    The TUI and web decided "queued by you" for themselves and the CLI tree marked nothing.
    """
    from agent6.ui.cli import _task_tree

    nodes = {
        "0001": {
            "title": "the run",
            "parent_id": None,
            "children": ["0002", "0003", "0004"],
            "status": "in_progress",
            "created_by": "user",
        },
        "0002": {
            "title": "the model's own",
            "parent_id": "0001",
            "children": [],
            "status": "pending",
            "created_by": "worker",
        },
        "0003": {
            "title": "queued",
            "parent_id": "0001",
            "children": [],
            "status": "pending",
            "created_by": "user",
        },
        "0004": {
            "title": "keep the suite green",
            "parent_id": "0001",
            "children": [],
            "status": "pending",
            "created_by": "steering",
            "standing": True,
        },
    }
    views = state.task_tree_views(nodes, "0002")
    assert [(v.short_id, v.note) for v in views] == [
        ("1", ""),
        ("2", ""),
        ("3", "queued by you"),
        ("4", "standing goal"),
    ]
    assert models.owner_note(created_by="user", parent_id=None, standing=False) == ""
    lines = _task_tree.task_tree_lines(nodes, "0002")
    assert lines[2].startswith("  3    ") and lines[2].endswith("queued  (queued by you)")
    assert lines[3].endswith("keep the suite green  (standing goal)")
