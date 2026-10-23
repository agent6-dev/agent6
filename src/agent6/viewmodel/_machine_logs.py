# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read agent-state logs for machine execution and spend."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
from collections.abc import Sequence

from agent6.machine import journal
from agent6.sessions import layout
from agent6.viewmodel import _state_types
from agent6.viewmodel import state as viewmodel_state
from agent6.viewmodel import tail as viewmodel_tail


@dataclasses.dataclass(frozen=True, slots=True)
class AgentExecution:
    """The newest agent state's execution as the operator verbs see it.

    Attributes:
        open: The execution has begun and not ended, so a steer has a reader.
        blocked_in: The state dir (`0001-attempt`) holding an unanswered approval or
            question, else "".
    """

    open: bool = False
    blocked_in: str = ""


def execution_of(state: _state_types.SessionState, log: pathlib.Path) -> AgentExecution:
    """Return the execution a folded state log describes.

    Args:
        state: The log's fold.
        log: The log's path; its parent names the state dir.

    Returns:
        The execution facts.
    """
    open_prompts = [*state.pending_approvals, *state.pending_questions]
    return AgentExecution(
        open=state.started and not state.finished,
        blocked_in=log.parent.name if any(not p.answered for p in open_prompts) else "",
    )


def newest_agent_execution(machine_dir: pathlib.Path) -> AgentExecution:
    """Return the newest state log's execution from one fold of that log.

    A poll loop holds a `NewestExecutionFold` instead.

    Args:
        machine_dir: The instance's dir.

    Returns:
        The execution facts, empty when no agent state has a log yet.
    """
    log = newest_state_log(machine_dir)
    if log is None:
        return AgentExecution()
    return execution_of(
        viewmodel_state.fold_session(viewmodel_tail.tail_events(log, follow=False)), log
    )


class NewestExecutionFold:
    """The newest state log folded across a poll loop.

    Each `refresh` folds the bytes appended since the last one into the held state,
    and starts over when the machine entered a newer agent state or the log was
    rewritten, so a tick reads the log once instead of whole.

    Attributes:
        state: The held fold.
    """

    def __init__(self) -> None:
        self._log: pathlib.Path | None = None
        self._tail: viewmodel_tail.LogTail | None = None
        self.state: _state_types.SessionState = viewmodel_state.initial_state()

    def refresh(self, machine_dir: pathlib.Path) -> pathlib.Path | None:
        """Fold what the newest state log gained since the last refresh.

        Args:
            machine_dir: The instance's dir.

        Returns:
            The newest state log, None when no agent state has one yet.
        """
        log = newest_state_log(machine_dir)
        if log != self._log:
            self._log, self._tail, self.state = (
                log,
                viewmodel_tail.LogTail(log) if log else None,
                viewmodel_state.initial_state(),
            )
        if self._tail is not None:
            events = self._tail.read()
            if self._tail.rewound:
                self.state = viewmodel_state.initial_state()
            for event in events:
                self.state = viewmodel_state.apply_event(self.state, event)
        return log

    @property
    def log(self) -> pathlib.Path | None:
        """The state log the held fold describes, None before any exists."""
        return self._log

    def execution(self) -> AgentExecution:
        """Return the execution facts of the held fold."""
        return execution_of(self.state, self._log) if self._log is not None else AgentExecution()


@dataclasses.dataclass(frozen=True, slots=True)
class Spend:
    """A dollar and token spend, summable so booked and live spend fold.

    Attributes:
        usd: The dollar figure.
        input_tokens: Input tokens.
        output_tokens: Output tokens.
        partial: An unpriced model contributed $0, so `usd` is a lower bound; it ORs
            across folds and the render adds the shared "~" marker.
        cache_read_tokens: The cached side of the input; 0 where never recorded.
        cache_creation_tokens: Cache-creation tokens; 0 where never recorded.
    """

    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    partial: bool = False
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: Spend) -> Spend:
        """Return the field-wise sum, `partial` ORed."""
        return Spend(
            self.usd + other.usd,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.partial or other.partial,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
        )


def read_budget_totals(log_path: pathlib.Path, *, from_offset: int = 0) -> Spend:
    """Read the latest running budget totals from an agent state's event log.

    Each `budget.update` carries cumulative totals from its call's own tracker, so
    the last one is that call's running total. This recovers the spend of a killed
    subprocess whose `result.json` never landed, and the live total of an in-flight
    state whose `StepEvent` is not written yet.

    Args:
        log_path: The state's journal.
        from_offset: Read only events appended after this byte offset; a caller
            salvaging one call on a shared log passes the size captured before its
            spawn, or a call that died before its first update double-books the prior one.

    Returns:
        The totals, or an empty `Spend` when there are none or the log is unreadable.
    """
    usd, tin, tout, cr, cc = 0.0, 0, 0, 0, 0
    partial = False
    with contextlib.suppress(OSError):
        with log_path.open("rb") as fh:
            if from_offset > 0:
                fh.seek(from_offset)
            body = fh.read().decode("utf-8", errors="replace")
        for line in body.splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "budget.update":
                usd = float(e.get("usd_total", usd) or 0.0)
                tin = int(e.get("input_total", tin) or 0)
                tout = int(e.get("output_total", tout) or 0)
                cr = int(e.get("cache_read_total", cr) or 0)
                cc = int(e.get("cache_creation_total", cc) or 0)
                partial = partial or bool(e.get("usd_partial", False))
    return Spend(usd, tin, tout, partial, cr, cc)


def state_dir_seq(dir_name: str) -> int | None:
    """Return the transition seq a `<seq>-<state>` dir name encodes, None when it has none."""
    head = dir_name.split("-", 1)[0]
    return int(head) if head.isdigit() else None


def machine_spend(
    events: Sequence[object], root: pathlib.Path, *, alive: bool
) -> tuple[Spend, str]:
    """Sum a machine instance's spend, the in-flight state's live figure included.

    A state books its StepEvent only when it completes, so the running state's log
    dir carries a seq no StepEvent holds: that log is the in-flight state, folded only
    while the worker is alive so a crashed in-flight log is ignored.

    Args:
        events: The journal's events.
        root: The instance's dir.
        alive: The worker pid is live.

    Returns:
        The total and the in-flight state's name, "" when none is running.
    """
    total = Spend()
    step_seqs: set[int] = set()
    for event in events:
        if isinstance(event, journal.StepEvent):
            step_seqs.add(event.seq)
            if isinstance(event.fact, journal.AgentFact):
                total += Spend(
                    event.fact.usd,
                    event.fact.input_tokens,
                    event.fact.output_tokens,
                    event.fact.usd_partial,
                )
        elif isinstance(event, journal.AttemptSpend):
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
        elif isinstance(event, journal.MachineEnd):
            # An unbooked slice rides on the end; gating on its usd would drop an unpriced one.
            total += Spend(event.usd, event.input_tokens, event.output_tokens, event.usd_partial)
    inflight_state = ""
    newest = newest_state_log(root) if alive else None
    if newest is not None:
        seq = state_dir_seq(newest.parent.name)
        if seq is not None and seq not in step_seqs:
            total += read_budget_totals(newest)
            inflight_state = newest.parent.name.split("-", 1)[-1]
    return total, inflight_state


def newest_state_log(root: pathlib.Path) -> pathlib.Path | None:
    """Return the newest agent state's journal, the one a watcher follows live, or None."""
    states = root / "states"
    if not states.is_dir():
        return None

    def seq_of(p: pathlib.Path) -> int:
        """Return the dir's seq, -1 for a dir without one."""
        head = p.name.split("-", 1)[0]
        return int(head) if head.isdigit() else -1

    for d in sorted((p for p in states.iterdir() if p.is_dir()), key=seq_of, reverse=True):
        log = d / layout.LOGS_NAME
        if log.is_file():
            return log
    return None


def read_complete_lines(path: pathlib.Path, offset: int) -> tuple[list[str], int]:
    """Read the complete lines a file gained past a byte offset.

    Reads bytes: a poll can hit EOF inside a multibyte sequence, where a text-mode
    readline would raise. Only complete lines are decoded.

    Args:
        path: The file.
        offset: Where the last read stopped.

    Returns:
        The new lines and the new offset, the start of any partial trailing line.
    """
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
