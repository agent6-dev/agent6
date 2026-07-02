# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure fold of a machine's journal into a render-ready watch view.

The machine analogue of state.py: where RunState folds a run's logs.jsonl, this
folds a machine instance's journal (the StepEvent / MachineEnd stream) plus its
spec into a MachineState that the CLI `agent6 watch`, the TUI
MachineWatchScreen, and a future web client all render. The agent reasoning
inside an `agent` state is itself a run log, so it folds through RunState
(state.py); this module models only the machine level: which states exist, where
we are, the path taken, and how it ended.

Position is exposed semantically (is_current / is_visited), not as a marker
glyph, so each front-end picks its own (the CLI uses ".", the TUI "·", a web
client a CSS class) without the model dictating presentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent6.machine.journal import MachineEnd, MachineNotify, StepEvent
from agent6.machine.model import MachineSpec

# How many recent machine.notify events a MachineState carries. Front-ends render
# them as ephemeral surfaces, so only the tail matters; the journal keeps them all.
_NOTIFY_KEEP = 20


@dataclass(frozen=True, slots=True)
class MachineStateView:
    """One state in the overview: its name, kind, and where we are relative to it."""

    name: str
    kind: str
    is_current: bool
    is_visited: bool


@dataclass(frozen=True, slots=True)
class TransitionView:
    """One journaled transition: state --label--> goto, in order."""

    seq: int
    state: str
    label: str
    goto: str


@dataclass(frozen=True, slots=True)
class MachineEndView:
    """The terminal/failed end the journal recorded, if any."""

    status: str
    reason: str
    state: str
    transitions: int


@dataclass(frozen=True, slots=True)
class NotificationView:
    """One journaled `machine.notify` (a state's `notify` message), in order."""

    ts: str
    state: str
    message: str
    level: str


@dataclass(frozen=True, slots=True)
class MachineState:
    machine: str
    version: int
    initial: str
    current: str  # where the machine is, or is about to run
    states: tuple[MachineStateView, ...]  # spec order, position-flagged
    transitions: tuple[TransitionView, ...]  # the path taken, in order
    ended: MachineEndView | None
    notifications: tuple[NotificationView, ...]  # recent machine.notify, oldest first


def fold_machine(spec: MachineSpec, events: Sequence[object]) -> MachineState:
    """Reduce a machine journal (StepEvent/MachineEnd stream) to a watch view.

    current = the goto of the last transition (where the machine is, or is about
    to run), else the initial state. visited = every state entered or left.
    """
    steps = [e for e in events if isinstance(e, StepEvent)]
    end = next((e for e in reversed(events) if isinstance(e, MachineEnd)), None)
    current = steps[-1].goto if steps else spec.initial
    visited: set[str] = set()
    for s in steps:
        visited.update((s.state, s.goto))
    states = tuple(
        MachineStateView(
            name=name,
            kind=st.kind,
            is_current=(name == current),
            is_visited=(name in visited),
        )
        for name, st in spec.states.items()
    )
    transitions = tuple(
        TransitionView(seq=s.seq, state=s.state, label=s.label, goto=s.goto) for s in steps
    )
    ended = (
        MachineEndView(
            status=end.status, reason=end.reason, state=end.state, transitions=end.transitions
        )
        if end is not None
        else None
    )
    notes = [e for e in events if isinstance(e, MachineNotify)]
    notifications = tuple(
        NotificationView(ts=n.ts, state=n.state, message=n.message, level=n.level)
        for n in notes[-_NOTIFY_KEEP:]
    )
    return MachineState(
        machine=spec.machine,
        version=spec.version,
        initial=spec.initial,
        current=current,
        states=states,
        transitions=transitions,
        ended=ended,
        notifications=notifications,
    )


def newest_state_log(root: Path) -> Path | None:
    """The logs.jsonl of the most recent agent-state execution (highest seq), or
    None. That is the state whose reasoning a watcher should follow live."""
    states = root / "states"
    if not states.is_dir():
        return None

    def seq_of(p: Path) -> int:
        head = p.name.split("-", 1)[0]
        return int(head) if head.isdigit() else -1

    for d in sorted((p for p in states.iterdir() if p.is_dir()), key=seq_of, reverse=True):
        log = d / "logs.jsonl"
        if log.is_file():
            return log
    return None


def machine_state_as_dict(ms: MachineState) -> dict[str, Any]:
    """The JSON-able wire form of a MachineState, stable field names: what
    `agent6 watch --json` and a web client serialize."""
    return asdict(ms)
