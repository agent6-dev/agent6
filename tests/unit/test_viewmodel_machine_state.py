# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure machine-journal fold in agent6.viewmodel.machine_state."""

from __future__ import annotations

from pathlib import Path

from agent6.machine import load_machine
from agent6.machine.journal import BranchFact, MachineEnd, StepEvent
from agent6.viewmodel.machine_state import fold_machine, newest_state_log

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


def test_newest_state_log_picks_highest_seq(tmp_path: Path) -> None:
    states = tmp_path / "states"
    for name in ("0000-greet", "0002-review", "0001-greet"):
        (states / name).mkdir(parents=True)
        (states / name / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    # A dir without a log yet (the agent hasn't written) must be ignored.
    (states / "0009-pending").mkdir()
    assert newest_state_log(tmp_path) == states / "0002-review" / "logs.jsonl"
    assert newest_state_log(tmp_path / "absent") is None
