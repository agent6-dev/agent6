# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A headless run prints the degraded-jail marker the live view renders."""

from __future__ import annotations

import functools
import pathlib

from agent6.app import execution
from agent6.sessions import event_log, journal


def test_a_degraded_jail_is_said_once_on_the_headless_log(tmp_path: pathlib.Path) -> None:
    """A warning marker reaches the text log, a steer does not; the detail is folded to one line."""
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    lines: list[str] = []
    sink.subscribe(functools.partial(execution.headless_marker, log=lines.append))
    sink.emit(
        journal.JailDegraded(detail="agent6-jail: mount /proc: EPERM\n  continuing without it")
    )
    sink.emit(journal.SteerRequested(turn=1))
    assert lines == [
        "[agent6] sandbox degraded: agent6-jail: mount /proc: EPERM continuing without it"
    ]


def test_headless_log_names_child_activity_and_merge_results(tmp_path: pathlib.Path) -> None:
    """Redirecting a run must not hide its children's activity."""
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    lines: list[str] = []
    sink.subscribe(functools.partial(execution.headless_marker, log=lines.append))
    sink.emit(journal.ParallelChildActivity(child="c1", activity="stopped", detail="stop request"))
    sink.emit(journal.ParallelChildActivity(child="c2", activity="stopped", detail="stopped"))
    sink.emit(journal.ParallelChildContinued(group="g", child="c1", sequence=1))
    sink.emit(
        journal.ParallelChildMerged(
            group="g", child="c1", status="merged", ref="kept", sha="abc123", detail=""
        )
    )
    sink.emit(
        journal.ParallelReturned(
            group="g",
            children=[{"session_id": "c2", "status": "unmerged", "commits": 2, "branch": "kept"}],
        )
    )
    assert lines == [
        "[agent6] child c1 stopped: stop request",
        "[agent6] child c2 stopped",
        "[agent6] child c1 continued (1)",
        "[agent6] child c1 merged into parent chain at abc123",
        "[agent6] group g complete: 1 child\nretained  c2  kept  2 commits",
    ]
