# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web front-end's server-sent-event streams: a run (an incremental fold
of logs.jsonl, streaming deltas coalesced, a dead worker closing the stream
truthfully) and a machine (a journal poll with an idle heartbeat).

HTTP-free: a handler binds its two socket writes into `SseChannel`, so the
streaming behaviour needs no server to exercise.
"""

from __future__ import annotations

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
    apply_event,
    died_without_end,
    initial_state,
    machine_snapshot,
    manifest_header,
    session_state_as_dict,
    summarize_session_dir,
    tail_events,
)

# SSE tuning: coalesce high-frequency streaming deltas, heartbeat idle streams so
# a gone client is noticed, and poll a machine's journal at human cadence.
DELTA_COALESCE_S = 0.15
HEARTBEAT_S = 15.0
MACHINE_POLL_S = 0.5
STREAMING_DELTAS = frozenset({"role.text_delta", "role.thinking_delta"})


@dataclass(frozen=True, slots=True)
class SseChannel:
    """The two writes a stream makes, bound to one client's socket; each
    returns False when the client has gone away."""

    send: Callable[[Any], bool]
    ping: Callable[[], bool]


def _with_idle_age(payload: dict[str, Any]) -> dict[str, Any]:
    """*payload* with the reasoning fold's idle age filled in from its epoch.

    Server-computed, like the run stream's, so a browser on another machine
    needs no clock agreement: the client anchors its "working... Ns" timer to
    (its own now) - age and ticks locally. Anchoring to the frame's arrival
    would show a state wedged for forty minutes as three seconds of work.
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
    """Refreshed manifest header to push on a finished run's heartbeat, or None
    when nothing changed. A merge (the run's own auto-merge, or any surface's)
    lands after session.end while a page is open, so its branch line and Merge
    button must follow it, as the TUI's `_branch_top` re-reads for a finished
    run. Only while finished, and only on a real change (a left-open page then
    settles instead of re-rendering every heartbeat)."""
    if not finished:
        return None
    refreshed = manifest_header(session_dir, repo=repo)
    return refreshed if refreshed != header else None


def stream_session(chan: SseChannel, session_dir: Path, *, repo: Path) -> None:  # noqa: PLR0915
    """Stream one run to *chan* until it ends, the worker dies, or the client
    leaves: the tailer thread feeds a queue, the loop folds every queued event
    into one frame, coalesces delta bursts, and heartbeats idle spans."""
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    stop = threading.Event()

    def tail() -> None:
        src = session_dir / LOGS_NAME
        try:
            # Not stop_when_finished: a finished run resumed from any other
            # surface logs into this same file, and a stream closing at
            # session.end would freeze the page on "stopped" while the hub
            # says "running". The TUI follows across legs the same way; the
            # client closes only on stream_dead (or navigation).
            for ev in tail_events(
                src, follow=True, stop_when_finished=False, should_stop=stop.is_set
            ):
                events.put(ev)
        finally:
            # Always enqueue the sentinel, even if the tailer raises: without
            # it the response loop would block on heartbeats forever.
            events.put(None)  # run ended (or tail cancelled/failed), tailer done

    # Manifest-derived header fields (branch facts + the fan-out compare
    # outcome). Fixed while the run is live; re-read on the heartbeat once it
    # finishes (see the queue-empty branch): the auto-merge (the run's own, or
    # any surface's) lands after session.end while a page is open, so the
    # branch line and Merge button follow it, as the TUI's `_branch_top` does.
    header = manifest_header(session_dir, repo=repo)

    threading.Thread(target=tail, daemon=True).start()

    def frame(*, dead: bool = False) -> dict[str, Any]:
        # session_dir per frame, not once at connect: a parked run the operator
        # resumes starts logging into this same stream, and the label (and
        # `live`) have to follow.
        d = {**session_state_as_dict(state, session_dir), **header}
        if state.last_event_ep is not None:
            # Server-computed so a browser on another machine needs no clock
            # agreement: the client anchors its "working… Ns" timer to
            # (its own now) - age, then ticks locally.
            d["last_event_age_s"] = max(0.0, time.time() - state.last_event_ep)
        if dead:
            # Transport signal, distinct from the fold's `finished`: this
            # stream will send nothing more (dead worker, no session.end), so
            # the client must close instead of letting EventSource retry
            # into a reconnect-refold loop. `finished` stays the fold truth:
            # a crashed run is stale, not "finished".
            d["stream_dead"] = True
        return d

    try:
        state = initial_state()
        last_delta_emit = 0.0
        while True:
            try:
                ev: dict[str, Any] | None = events.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                if not chan.ping():
                    return
                # A run that reached terminal without its own session.end
                # (crash, went quiet, killed in preflight) would otherwise
                # pin this worker forever; ask the codebase's own
                # died_without_end rather than one word of it. `parked` is
                # deliberately excluded: a parked submission the operator
                # resumes starts logging into this same stream.
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
            # Fold everything already queued into one frame. On connect the
            # tailer replays the whole history, and a full SessionState frame per
            # historical event is quadratic (13 MB probed on a 502-event run).
            last_type = ""
            while ev is not None:
                state = apply_event(state, ev)
                last_type = str(ev.get("type", ""))
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    break
            if ev is None:  # run ended: send the final snapshot and close
                chan.send(frame())
                return
            now = time.monotonic()
            if last_type in STREAMING_DELTAS and (now - last_delta_emit) < DELTA_COALESCE_S:
                continue  # coalesce bursts of text/thinking deltas
            if not chan.send(frame()):
                return
            last_delta_emit = now
    finally:
        # cancel the tailer so it exits on disconnect / dead run, not just session.end
        stop.set()


def stream_machine(chan: SseChannel, machine_dir: Path) -> None:
    """Stream one machine to *chan*: poll the journal fold, push the combined
    snapshot when it changes, heartbeat when it does not, and close truthfully
    on a journaled end or a dead worker."""
    prev = ""
    idle = 0.0
    while True:
        try:
            payload = {
                "machine": machine_snapshot(machine_dir),
                "reasoning": model.machine_reasoning_snapshot(machine_dir),
            }
        except MachineError as exc:
            chan.send({"type": "error", "error": "; ".join(exc.problems)})
            return
        if payload["machine"].get("status") == "stopped":
            # No worker and no armed wait (an operator stop or a death
            # mid-state, the same dir: the worker clears its pid on every
            # unwound exit): resumable, so the frame says so and the stream
            # stays open for `machine run`. A fabricated `ended` (a status the
            # journal vocabulary does not hold) would style it terminal;
            # `ended` stays reserved for a durable MachineEnd.
            payload["machine"]["worker_lost"] = {
                "reason": "no worker running",
                "state": payload["machine"].get("current", ""),
            }
        blob = json.dumps(payload, sort_keys=True)
        if blob != prev:
            # The age is derived at send time and deliberately outside the
            # comparison above: it changes every poll, so including it would
            # send a frame every poll. The epoch it comes from does not.
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
            return  # machine terminated: final snapshot sent, close the stream
        time.sleep(MACHINE_POLL_S)
