# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The queued-task bridge: the file a `/task` writes and the run drains.

Unlike the answer and steer bridges, this one is a queue: many entries, taken
oldest first, and none of it is cleared at a leg boundary.
"""

from __future__ import annotations

from pathlib import Path

from agent6.sessions.ipc import (
    clear_pending_answers,
    drain_queued_tasks,
    queue_path,
    queue_task,
)


def test_queued_tasks_drain_oldest_first_and_only_once(tmp_path: Path) -> None:
    for text in ("first thing", "second thing", "third thing"):
        queue_task(tmp_path, text)

    assert drain_queued_tasks(tmp_path) == ["first thing", "second thing", "third thing"]
    assert drain_queued_tasks(tmp_path) == []


def test_a_multi_line_task_keeps_every_line(tmp_path: Path) -> None:
    """The operator's full text is the payload: a considered spec queued from a
    phone reaches the run whole, not as its first line."""
    spec = "Add a --json flag\n\nIt should print the same fields as the table,\nkeyed by name."
    queue_task(tmp_path, spec)

    assert drain_queued_tasks(tmp_path) == [spec]


def test_a_blank_task_is_dropped(tmp_path: Path) -> None:
    queue_task(tmp_path, "   \n  ")

    assert drain_queued_tasks(tmp_path) == []


def test_draining_an_untouched_run_makes_no_directory(tmp_path: Path) -> None:
    """A read never creates the dir: the listings sort runs by the session
    dir's mtime, which making a directory would bump."""
    assert drain_queued_tasks(tmp_path) == []
    assert not queue_path(tmp_path).exists()


def test_a_queued_task_survives_a_leg_boundary(tmp_path: Path) -> None:
    """A task queued while the run was between legs is still wanted, so the
    leg-start sweep that drops answers and markers leaves the queue alone."""
    queue_task(tmp_path, "queued between legs")

    clear_pending_answers(tmp_path, started_at=0.0)

    assert drain_queued_tasks(tmp_path) == ["queued between legs"]
