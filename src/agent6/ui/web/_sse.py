# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Stream a run or a machine to a browser as server-sent events.

A run stream folds logs.jsonl incrementally, coalesces delta bursts and closes on a
dead worker; a machine stream polls the journal with an idle heartbeat. A handler
binds its two socket writes into `SseChannel`, so the streams need no server to
exercise.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.machine import MachineError
from agent6.sessions.layout import LOGS_NAME
from agent6.ui.web import model
from agent6.viewmodel import (
    NewestExecutionFold,
    apply_event,
    died_without_end,
    initial_state,
    machine_snapshot,
    manifest_header,
    session_state_as_dict,
    summarize_session_dir,
    tail_events,
)

# Coalesce delta bursts, heartbeat idle streams so a gone client is noticed, poll at human cadence.
DELTA_COALESCE_S = 0.15
HEARTBEAT_S = 15.0
MACHINE_POLL_S = 0.5
STREAMING_DELTAS = frozenset({"role.text_delta", "role.thinking_delta"})


@dataclass(frozen=True, slots=True)
class SseChannel:
    """The two writes a stream makes, bound to one client's socket.

    Attributes:
        send: Write one JSON frame; False once the client has gone.
        ping: Write a heartbeat comment; False once the client has gone.
    """

    send: Callable[[Any], bool]
    ping: Callable[[], bool]


def _with_idle_age(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill the reasoning fold's idle age in from its epoch.

    The age is server-computed so a browser on another machine needs no clock
    agreement: the client anchors its timer to its own now minus the age.

    Args:
        payload: A machine frame with a `reasoning` half.

    Returns:
        The payload with `last_event_age_s` set, or unchanged without an epoch.
    """
    reasoning = payload.get("reasoning") or {}
    ep = reasoning.get("last_event_ep")
    if not isinstance(ep, (int, float)):
        return payload
    fresh = {**reasoning, "last_event_age_s": max(0.0, time.time() - ep)}
    return {**payload, "reasoning": fresh}


def _late_merge_header(
    session_dir: Path, repo: Path, header: dict[str, Any], *, finished: bool
) -> dict[str, Any] | None:
    """Re-read the manifest header on a finished run's heartbeat.

    A merge lands after session.end while a page is open, so the branch line and
    Merge button follow it; a left-open page settles once nothing changes.

    Args:
        session_dir: The run's directory.
        repo: The repository the run worked in.
        header: The header last sent.
        finished: The fold's finished flag.

    Returns:
        The refreshed header, or None while the run is live or nothing changed.
    """
    if not finished:
        return None
    refreshed = manifest_header(session_dir, repo=repo)
    return refreshed if refreshed != header else None


def stream_session(chan: SseChannel, session_dir: Path, *, repo: Path) -> None:  # noqa: PLR0915
    """Stream one run until it ends, the worker dies, or the client leaves.

    A tailer thread feeds a queue; the loop folds every queued event into one frame,
    coalesces delta bursts and heartbeats idle spans.

    Args:
        chan: The client's socket writes.
        session_dir: The run's directory.
        repo: The repository the run worked in.
    """
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    stop = threading.Event()

    def tail() -> None:
        src = session_dir / LOGS_NAME
        try:
            # A resumed run logs into this same file, so the stream outlives session.end.
            for ev in tail_events(
                src, follow=True, stop_when_finished=False, should_stop=stop.is_set
            ):
                events.put(ev)
        finally:
            # The sentinel goes even when the tailer raises, or the loop blocks on heartbeats.
            events.put(None)

    # Fixed while the run is live; re-read on the heartbeat once it finishes.
    header = manifest_header(session_dir, repo=repo)

    threading.Thread(target=tail, daemon=True).start()

    def frame(*, dead: bool = False) -> dict[str, Any]:
        # The dir is read per frame: a resumed parked run changes the label and `live`.
        d = {**session_state_as_dict(state, session_dir), **header}
        if state.last_event_ep is not None:
            # Server-computed so a browser on another machine needs no clock agreement.
            d["last_event_age_s"] = max(0.0, time.time() - state.last_event_ep)
        if dead:
            # A transport signal, distinct from `finished`: the client closes instead of retrying.
            d["stream_dead"] = True
        return d

    def drain(ev: dict[str, Any] | None) -> tuple[str, bool]:
        """Fold an event and everything queued behind it.

        Args:
            ev: The event taken from the queue; None is the tailer's sentinel.

        Returns:
            The last event's type, and whether the run ended.
        """
        nonlocal state
        last_type = ""
        while ev is not None:
            state = apply_event(state, ev)
            last_type = str(ev.get("type", ""))
            try:
                ev = events.get_nowait()
            except queue.Empty:
                return last_type, False
        return last_type, True

    try:
        state = initial_state()
        last_delta_emit = 0.0
        while True:
            try:
                ev: dict[str, Any] | None = events.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                if not chan.ping():
                    return
                # A run dead without session.end would pin this worker; parked is excluded.
                word = summarize_session_dir(session_dir).status
                if word != "parked" and died_without_end(word):
                    chan.send(frame(dead=True))
                    return
                refreshed = _late_merge_header(session_dir, repo, header, finished=state.finished)
                if refreshed is not None:
                    header = refreshed
                    if not chan.send(frame()):
                        return
                continue
            # One frame per queued batch: a frame per replayed event is quadratic (13 MB at 502).
            last_type, ended = drain(ev)
            if ended:
                chan.send(frame())
                return
            # A delta burst is one frame per window; the wait catches the burst's tail.
            wait = DELTA_COALESCE_S - (time.monotonic() - last_delta_emit)
            if last_type in STREAMING_DELTAS and wait > 0:
                time.sleep(wait)
                with contextlib.suppress(queue.Empty):
                    _, ended = drain(events.get_nowait())
                    if ended:
                        chan.send(frame())
                        return
            if not chan.send(frame()):
                return
            last_delta_emit = time.monotonic()
    finally:
        stop.set()


def stream_machine(chan: SseChannel, machine_dir: Path) -> None:
    """Stream one machine until it ends or the client leaves.

    Each poll folds the journal and pushes the snapshot when it changed, heartbeats
    when it did not, and closes on a journaled end.

    Args:
        chan: The client's socket writes.
        machine_dir: The machine instance's directory.
    """
    prev = ""
    idle = 0.0
    fold = NewestExecutionFold()  # the newest state log, read once per poll for both halves
    while True:
        try:
            reasoning = model.machine_reasoning_snapshot(machine_dir, fold=fold)
            payload = {
                "machine": machine_snapshot(machine_dir, execution=fold.execution()),
                "reasoning": reasoning,
            }
        except MachineError as exc:
            chan.send({"type": "error", "error": "; ".join(exc.problems)})
            return
        # A stopped machine's frame carries `worker_lost`; the stream stays open for `machine run`.
        blob = json.dumps(payload, sort_keys=True)
        if blob != prev:
            # The age is added after the comparison: it changes every poll, the epoch does not.
            if not chan.send(_with_idle_age(payload)):
                return
            prev = blob
            idle = 0.0
        else:
            idle += MACHINE_POLL_S
            if idle >= HEARTBEAT_S and not chan.ping():
                return
            if idle >= HEARTBEAT_S:
                idle = 0.0
        if payload["machine"].get("ended") is not None:
            return
        time.sleep(MACHINE_POLL_S)
