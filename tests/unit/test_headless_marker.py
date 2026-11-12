# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A headless run prints the degraded-jail marker the live view renders."""

from __future__ import annotations

import functools
import pathlib

from agent6.app import execution
from agent6.sessions import event_log, journal


def test_a_degraded_jail_is_said_once_on_the_headless_log(tmp_path: pathlib.Path) -> None:
    """Only `jail.degraded` reaches the text log; the event's detail is folded to one line."""
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
