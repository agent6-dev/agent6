# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure fold of a machine's journal into a render-ready watch view.

The machine analogue of state.py: where RunState folds a run's logs.jsonl, this
folds a machine instance's journal (the StepEvent / MachineEnd stream) plus its
spec into a MachineState that the CLI `agent6 attach`, the TUI
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
    """The terminal/failed end the journal recorded, if any: a render projection
    of the journal's `MachineEnd`, dropping its `type`/`ts` wire fields."""

    status: str
    reason: str
    state: str
    transitions: int

    @classmethod
    def from_end(cls, end: MachineEnd) -> MachineEndView:
        return cls(end.status, end.reason, end.state, end.transitions)


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
    ended = MachineEndView.from_end(end) if end is not None else None
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


def notification_key(n: NotificationView) -> tuple[str, str, str]:
    """A stable identity for a notification, for dedup across the sliding window
    (front-ends track which they have surfaced by this key, not by a count into
    the capped `notifications` tuple). Mirrors the web client's ts|state|message."""
    return (n.ts, n.state, n.message)


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


def read_complete_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete new lines of *path* past byte *offset*, plus the new offset
    (the start of any partial trailing line, re-read next poll).

    Byte reads: a poll can hit EOF mid multibyte UTF-8 sequence (the writer
    flushes long lines in several syscalls) and a text-mode readline would
    raise UnicodeDecodeError there. Only complete lines are decoded."""
    lines: list[str] = []
    pos = offset
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            while True:
                pos = fh.tell()
                raw = fh.readline()
                if not raw.endswith(b"\n"):
                    break
                lines.append(raw.decode("utf-8", errors="replace"))
    except OSError:
        pass
    return lines, pos


@dataclass
class MachineWatchCursor:
    """What a live machine watcher has already surfaced.

    One implementation of the three dedup rules every front-end (the CLI watch
    loop, the TUI machine screen) must agree on: transitions by count,
    notifications by identity (``ms.notifications`` is a sliding window, so a
    count index would miss every notify past its cap), and the newest state
    log by (path, byte offset) with only complete lines consumed."""

    seen_steps: int = 0
    seen_notifications: set[tuple[str, str, str]] | None = None
    log_path: Path | None = None
    log_offset: int = 0

    def seed_notifications(self, ms: MachineState) -> None:
        """Mark every already-recorded notification as seen, so opening a watch
        does not re-announce history."""
        self.seen_notifications = {notification_key(n) for n in ms.notifications}

    def new_transitions(self, ms: MachineState) -> list[TransitionView]:
        out = list(ms.transitions[self.seen_steps :])
        self.seen_steps = len(ms.transitions)
        return out

    def new_notifications(self, ms: MachineState) -> list[NotificationView]:
        if self.seen_notifications is None:
            self.seen_notifications = set()
        out: list[NotificationView] = []
        for n in ms.notifications:
            key = notification_key(n)
            if key not in self.seen_notifications:
                self.seen_notifications.add(key)
                out.append(n)
        return out

    def advance_log(self, root: Path) -> tuple[Path | None, bool]:
        """Track the newest per-state log under *root*. Returns the current log
        and True when it changed; the caller resets its render state (elapsed
        anchor, pending text) and announces the new agent state."""
        newest = newest_state_log(root)
        if newest != self.log_path:
            self.log_path, self.log_offset = newest, 0
            return newest, True
        return newest, False

    def read_log_lines(self) -> list[str]:
        """Complete new lines of the current state log since the last poll."""
        if self.log_path is None:
            return []
        lines, self.log_offset = read_complete_lines(self.log_path, self.log_offset)
        return lines


def machine_state_as_dict(ms: MachineState) -> dict[str, Any]:
    """The JSON-able wire form of a MachineState, stable field names: what
    `agent6 attach --json` and a web client serialize."""
    return asdict(ms)
