# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure machine-journal fold in agent6.viewmodel.machine_state."""

from __future__ import annotations

from pathlib import Path

from agent6.machine import load_machine
from agent6.machine.journal import BranchFact, MachineEnd, MachineNotify, StepEvent
from agent6.viewmodel.machine_state import (
    _NOTIFY_KEEP,  # pyright: ignore[reportPrivateUsage]
    NotificationView,
    fold_machine,
    machine_status_word,
    newest_state_log,
    notification_key,
)

# A branch -> terminal machine: two states, no I/O, valid to load.
TINY = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _spec(tmp_path: Path):
    f = tmp_path / "tiny.asm.toml"
    f.write_text(TINY, encoding="utf-8")
    return load_machine(f)


def test_fold_empty_starts_at_initial(tmp_path: Path) -> None:
    # No journal yet: the machine is at its initial state, nothing visited.
    ms = fold_machine(_spec(tmp_path), [])
    assert (ms.machine, ms.version, ms.initial, ms.current) == ("tiny", 1, "route", "route")
    assert ms.transitions == ()
    assert ms.ended is None
    assert [s.name for s in ms.states] == ["route", "done"]  # spec order preserved
    route = next(s for s in ms.states if s.name == "route")
    assert route.is_current and not route.is_visited and route.kind == "branch"
    assert all(not s.is_visited for s in ms.states)


def test_fold_tracks_position_transitions_and_end(tmp_path: Path) -> None:
    events = [
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    ms = fold_machine(_spec(tmp_path), events)
    by = {s.name: s for s in ms.states}
    # current = goto of the last transition; both endpoints are visited.
    assert ms.current == "done"
    assert by["done"].is_current and not by["route"].is_current
    assert by["route"].is_visited and by["done"].is_visited
    path = [(t.seq, t.state, t.label, t.goto) for t in ms.transitions]
    assert path == [(0, "route", "else", "done")]
    assert ms.ended is not None
    assert (ms.ended.status, ms.ended.reason, ms.ended.state, ms.ended.transitions) == (
        "ok",
        "routed",
        "done",
        1,
    )


def test_machine_status_word_distinguishes_waiting_from_running(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    ended = fold_machine(
        spec, [MachineEnd(ts="t", status="failed", reason="boom", state="done", transitions=1)]
    )
    # A terminal instance reports its end status regardless of liveness probes.
    assert machine_status_word(ended, parked=True, alive=True) == "failed"

    live = fold_machine(spec, [])  # not ended
    # Parked (an armed --exit-on-wait wait) reads waiting, even if a stale pid
    # probe were to lie alive; running only when live and not parked; a dead pid
    # that is neither parked nor ended is stopped.
    assert machine_status_word(live, parked=True, alive=False) == "waiting"
    assert machine_status_word(live, parked=True, alive=True) == "waiting"
    assert machine_status_word(live, parked=False, alive=True) == "running"
    assert machine_status_word(live, parked=False, alive=False) == "stopped"


def test_newest_state_log_picks_highest_seq(tmp_path: Path) -> None:
    states = tmp_path / "states"
    for name in ("0000-greet", "0002-review", "0001-greet"):
        (states / name).mkdir(parents=True)
        (states / name / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    # A dir without a log yet (the agent hasn't written) must be ignored.
    (states / "0009-pending").mkdir()
    assert newest_state_log(tmp_path) == states / "0002-review" / "logs.jsonl"
    assert newest_state_log(tmp_path / "absent") is None


def test_fold_collects_notifications(tmp_path: Path) -> None:
    events = [
        MachineNotify(ts="t1", state="route", message="starting", level="info"),
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineNotify(ts="t2", state="done", message="all done", level="warn"),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    ms = fold_machine(_spec(tmp_path), events)
    assert [(n.state, n.message, n.level) for n in ms.notifications] == [
        ("route", "starting", "info"),
        ("done", "all done", "warn"),
    ]


def test_notifications_are_a_capped_sliding_window(tmp_path: Path) -> None:
    # notifications is capped to the recent tail: a front-end must dedup by
    # notification_key, NOT by a count index (which would miss every one past
    # the cap once the window slides).
    events = [
        MachineNotify(ts=f"t{i}", state="route", message=f"n{i}", level="info")
        for i in range(_NOTIFY_KEEP + 5)
    ]
    ms = fold_machine(_spec(tmp_path), events)
    assert len(ms.notifications) == _NOTIFY_KEEP
    assert ms.notifications[-1].message == f"n{_NOTIFY_KEEP + 4}"  # newest kept
    assert ms.notifications[0].message == "n5"  # oldest dropped


def test_notification_key_is_stable_identity() -> None:
    n = NotificationView(ts="t1", state="poll", message="hi", level="warn")
    assert notification_key(n) == ("t1", "poll", "hi")


def test_machine_state_as_dict_is_json_serializable(tmp_path: Path) -> None:
    import json

    from agent6.viewmodel.machine_state import machine_state_as_dict

    events = [
        StepEvent(
            ts="t", seq=0, state="route", label="else", goto="done", fact=BranchFact(clause_index=1)
        ),
        MachineEnd(ts="t", status="ok", reason="routed", state="done", transitions=1),
    ]
    d = machine_state_as_dict(fold_machine(_spec(tmp_path), events))
    assert d["machine"] == "tiny" and d["current"] == "done"
    assert d["states"][0]["name"] == "route"  # tuple -> list, dataclass -> dict
    assert d["ended"]["status"] == "ok"
    json.dumps(d)  # the wire form must serialize
