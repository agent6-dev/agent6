# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The queued-task bridge: the file a `/task` writes and the run drains.

Unlike the answer and steer bridges, this one is a queue: many entries, taken
oldest first, and none of it is cleared at an execution boundary.
"""

from __future__ import annotations

import time
from pathlib import Path

from agent6.sessions.ipc import (
    OperatorRequest,
    clear_pending_answers,
    drain_requests,
    queue_path,
    queue_request,
)


def test_queued_tasks_drain_oldest_first_and_only_once(tmp_path: Path) -> None:
    for text in ("first thing", "second thing", "third thing"):
        queue_request(tmp_path, "task", text)

    assert [r.text for r in drain_requests(tmp_path)] == [
        "first thing",
        "second thing",
        "third thing",
    ]
    assert [r.text for r in drain_requests(tmp_path)] == []


def test_a_multi_line_task_keeps_every_line(tmp_path: Path) -> None:
    """The operator's full text is the payload: a considered spec queued from a
    phone reaches the run whole, not as its first line."""
    spec = "Add a --json flag\n\nIt should print the same fields as the table,\nkeyed by name."
    queue_request(tmp_path, "task", spec)

    assert [r.text for r in drain_requests(tmp_path)] == [spec]


def test_a_blank_task_is_dropped(tmp_path: Path) -> None:
    queue_request(tmp_path, "task", "   \n  ")

    assert [r.text for r in drain_requests(tmp_path)] == []


def test_draining_an_untouched_run_makes_no_directory(tmp_path: Path) -> None:
    """A read never creates the dir: the listings sort runs by the session
    dir's mtime, which making a directory would bump."""
    assert [r.text for r in drain_requests(tmp_path)] == []
    assert not queue_path(tmp_path).exists()


def test_a_queued_task_survives_a_execution_boundary(tmp_path: Path) -> None:
    """A task queued during the run's last turn is still wanted, so the
    execution-start sweep that drops answers and markers older than the execution's start
    leaves the queue alone."""
    queue_request(tmp_path, "task", "queued between executions")
    (tmp_path / "steer.answer").write_text("stale", encoding="utf-8")

    clear_pending_answers(tmp_path, started_at=time.time() + 600)

    assert not (tmp_path / "steer.answer").exists()  # the sweep did run
    assert [r.text for r in drain_requests(tmp_path)] == ["queued between executions"]


def test_every_request_keeps_its_kind_and_order(tmp_path: Path) -> None:
    """The standing goal and the retirement were a single slot and an
    append-only file, read then deleted by name: a goal typed between the
    read and the delete was lost after the composer said it was set."""
    queue_request(tmp_path, "standing", "first goal")
    queue_request(tmp_path, "task", "a task")
    queue_request(tmp_path, "standing", "second goal")
    queue_request(tmp_path, "retire", "0002")

    assert drain_requests(tmp_path) == [
        OperatorRequest("standing", "first goal"),
        OperatorRequest("task", "a task"),
        OperatorRequest("standing", "second goal"),
        OperatorRequest("retire", "0002"),
    ]
    assert drain_requests(tmp_path) == []
