# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The queued-task bridge: the file a `/task` writes and the run drains.

Unlike the answer and steer bridges, this one is a queue: many entries, taken
oldest first, and none of it is cleared at an execution boundary.
"""

from __future__ import annotations

import pathlib
import time

from agent6.sessions import ipc


def test_queued_tasks_drain_oldest_first_and_only_once(tmp_path: pathlib.Path) -> None:
    for text in ("first thing", "second thing", "third thing"):
        ipc.queue_request(tmp_path, "task", text)

    assert [r.text for r in ipc.drain_requests(tmp_path)] == [
        "first thing",
        "second thing",
        "third thing",
    ]
    assert [r.text for r in ipc.drain_requests(tmp_path)] == []


def test_a_multi_line_task_keeps_every_line(tmp_path: pathlib.Path) -> None:
    """The operator's full text is the payload, not its first line."""
    spec = "Add a --json flag\n\nIt should print the same fields as the table,\nkeyed by name."
    ipc.queue_request(tmp_path, "task", spec)

    assert [r.text for r in ipc.drain_requests(tmp_path)] == [spec]


def test_a_blank_task_is_dropped(tmp_path: pathlib.Path) -> None:
    ipc.queue_request(tmp_path, "task", "   \n  ")

    assert [r.text for r in ipc.drain_requests(tmp_path)] == []


def test_draining_an_untouched_run_makes_no_directory(tmp_path: pathlib.Path) -> None:
    """A read never creates the dir; the listings sort runs by the session dir's mtime."""
    assert [r.text for r in ipc.drain_requests(tmp_path)] == []
    assert not ipc.queue_path(tmp_path).exists()


def test_a_queued_task_survives_a_execution_boundary(tmp_path: pathlib.Path) -> None:
    """The execution-start sweep leaves the task queue alone.

    A task queued during the run's last turn is still wanted.
    """
    ipc.queue_request(tmp_path, "task", "queued between executions")
    (tmp_path / "steer.answer").write_text("stale", encoding="utf-8")

    ipc.clear_pending_answers(tmp_path, started_at=time.time() + 600)

    assert not (tmp_path / "steer.answer").exists()  # the sweep did run
    assert [r.text for r in ipc.drain_requests(tmp_path)] == ["queued between executions"]


def test_every_request_keeps_its_kind_and_order(tmp_path: pathlib.Path) -> None:
    """A goal typed between the read and the delete of the standing slot is never lost."""
    ipc.queue_request(tmp_path, "standing", "first goal")
    ipc.queue_request(tmp_path, "task", "a task")
    ipc.queue_request(tmp_path, "standing", "second goal")
    ipc.queue_request(tmp_path, "retire", "0002")

    assert ipc.drain_requests(tmp_path) == [
        ipc.OperatorRequest("standing", "first goal"),
        ipc.OperatorRequest("task", "a task"),
        ipc.OperatorRequest("standing", "second goal"),
        ipc.OperatorRequest("retire", "0002"),
    ]
    assert ipc.drain_requests(tmp_path) == []
