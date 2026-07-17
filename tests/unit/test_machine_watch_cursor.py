# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""MachineWatchCursor: the shared what-have-I-shown state for machine watchers.

Pins the three dedup rules the CLI watch loop and the TUI machine screen must
agree on: transitions by count, notifications by identity across the sliding
window, and byte-offset log tailing that never consumes a partial line.
"""

from __future__ import annotations

from pathlib import Path

from agent6.viewmodel import MachineWatchCursor, read_complete_lines
from agent6.viewmodel.machine_state import MachineState, NotificationView, TransitionView


def _ms(
    transitions: tuple[TransitionView, ...] = (),
    notifications: tuple[NotificationView, ...] = (),
) -> MachineState:
    return MachineState(
        machine="m",
        version=1,
        initial="a",
        current="a",
        states=(),
        transitions=transitions,
        ended=None,
        notifications=notifications,
    )


def _t(seq: int) -> TransitionView:
    return TransitionView(seq=seq, state="a", label="ok", goto="b")


def _n(ts: str, message: str = "hi") -> NotificationView:
    return NotificationView(ts=ts, state="a", message=message, level="info")


def test_new_transitions_are_yielded_once() -> None:
    cur = MachineWatchCursor()
    assert cur.new_transitions(_ms(transitions=(_t(1), _t(2)))) == [_t(1), _t(2)]
    assert cur.new_transitions(_ms(transitions=(_t(1), _t(2)))) == []
    assert cur.new_transitions(_ms(transitions=(_t(1), _t(2), _t(3)))) == [_t(3)]


def test_notifications_dedup_by_identity_across_the_sliding_window() -> None:
    """The viewmodel caps ms.notifications, so a count index would miss every
    notify past the cap; identity dedup must keep working when old entries
    slide out of the window."""
    cur = MachineWatchCursor()
    assert cur.new_notifications(_ms(notifications=(_n("1"), _n("2")))) == [_n("1"), _n("2")]
    # Window slid: "1" dropped out, "3" arrived. Only "3" is new.
    assert cur.new_notifications(_ms(notifications=(_n("2"), _n("3")))) == [_n("3")]


def test_seed_notifications_silences_history() -> None:
    cur = MachineWatchCursor()
    cur.seed_notifications(_ms(notifications=(_n("1"),)))
    assert cur.new_notifications(_ms(notifications=(_n("1"), _n("2")))) == [_n("2")]


def test_advance_log_switches_and_resets_offset(tmp_path: Path) -> None:
    s1 = tmp_path / "states" / "001-first"
    s1.mkdir(parents=True)
    (s1 / "logs.jsonl").write_text('{"a":1}\n', encoding="utf-8")
    cur = MachineWatchCursor()

    log, switched = cur.advance_log(tmp_path)
    assert switched and log == s1 / "logs.jsonl"
    assert cur.read_log_lines() == ['{"a":1}\n']

    # Same log again: no switch, nothing new to read.
    _, switched = cur.advance_log(tmp_path)
    assert not switched
    assert cur.read_log_lines() == []

    # A newer state appears: switch + offset reset to its start.
    s2 = tmp_path / "states" / "002-second"
    s2.mkdir(parents=True)
    (s2 / "logs.jsonl").write_text('{"b":2}\n', encoding="utf-8")
    log, switched = cur.advance_log(tmp_path)
    assert switched and log == s2 / "logs.jsonl"
    assert cur.read_log_lines() == ['{"b":2}\n']


def test_read_complete_lines_leaves_partial_tail_unconsumed(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":')
    lines, off = read_complete_lines(p, 0)
    assert lines == ['{"a":1}\n']
    assert off == len(b'{"a":1}\n')  # the partial line is re-read next poll
    # Writer finishes the line (multibyte content flushed across syscalls).
    p.write_bytes(b'{"a":1}\n{"b":"\xc3\xa9"}\n')
    lines, off = read_complete_lines(p, off)
    assert lines == ['{"b":"é"}\n']


def test_read_complete_lines_missing_file(tmp_path: Path) -> None:
    lines, off = read_complete_lines(tmp_path / "absent.jsonl", 7)
    assert lines == [] and off == 7
