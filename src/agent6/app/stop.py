# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Stopping a session: the one function behind `agent6 stop`, the TUI's Stop
entries, the web's Stop buttons, `/stop` in a composer and an ACP cancel.

A stop now writes both bridges the loop reads (the abort steer, which cuts a
model call in flight, and the stop marker, which breaks a command or approval
wait), then waits for the run to end. A worker that has not ended by then is
not reading requests (parked on a prompt, wedged on a stream): it and the
background commands it started on the host get SIGTERM, then SIGKILL after a
grace; the jail's children die with it. The run then reads stale and resume
continues it. A stop after the step writes the marker alone: the finished
step's tool results and auto-commit land first.

A fan-out's stop ends its live lanes first, the same way, then stops the
coordinator, which imports and ranks what the lanes landed before it ends;
that drain gets its own, longer wait.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from agent6.sandbox.jail import signal_group
from agent6.sessions.ipc import (
    ProcessIdentity,
    process_is_alive,
    read_live_worker_identity,
    request_stop,
    submit_steer,
)
from agent6.sessions.layout import layout_of
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.background import SHELLS_DIR, shell_host_processes
from agent6.viewmodel import session_is_live, summarize_session_dir
from agent6.viewmodel.listing import lanes_of, produced_result

STOP_WAIT_S = 5.0  # a run that reads its requests ends within this
FANOUT_WAIT_S = 30.0  # a coordinator drains its ended lanes and imports what landed within this
KILL_GRACE_S = 3.0  # SIGTERM to SIGKILL
_POLL_S = 0.1


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """What a stop did: *how* is `after_step`, `stopped`, `killed`, `stale` (the
    run did not answer and nothing of it was left to signal), `not_live`
    (nothing to stop) or `failed` (a request could not be written); *message*
    is the line every surface shows; *resumable* says `agent6 resume` continues
    it (a fan-out has no loop to resume)."""

    session_id: str
    ok: bool
    how: str
    message: str
    resumable: bool = False


def stop_session(
    session_dir: Path,
    *,
    after_step: bool = False,
    wait_s: float | None = None,
    grace_s: float | None = None,
) -> StopOutcome:
    """Stop the session in *session_dir* (see the module docstring). *wait_s*
    and *grace_s* default to the module's constants at call time."""
    rid = session_dir.name
    wait_s = STOP_WAIT_S if wait_s is None else wait_s
    grace_s = KILL_GRACE_S if grace_s is None else grace_s
    if not session_is_live(session_dir):
        return StopOutcome(
            rid, False, "not_live", f"{rid} {_not_live_state(session_dir)}; nothing to stop"
        )
    fanout = _is_fanout(session_dir)
    lanes = _live_lanes(session_dir) if fanout else []
    for lane in lanes:
        stop_session(lane, after_step=after_step, wait_s=wait_s, grace_s=grace_s)
    with_lanes = f" with its {len(lanes)} live lane{'s' if len(lanes) > 1 else ''}" if lanes else ""
    worker = read_live_worker_identity(session_dir)
    if after_step:
        if not request_stop(session_dir):
            return StopOutcome(rid, False, "failed", f"could not write the stop request for {rid}")
        when = "their current steps" if lanes else "its current step"
        return StopOutcome(
            rid, True, "after_step", f"{rid} stops{with_lanes} after {when}", resumable=not fanout
        )
    if not (submit_steer(session_dir, "abort", now=True) and request_stop(session_dir)):
        return StopOutcome(rid, False, "failed", f"could not write the stop request for {rid}")
    return _ended(
        session_dir,
        worker,
        wait_s=FANOUT_WAIT_S if fanout else wait_s,
        grace_s=grace_s,
        fanout=fanout,
        with_lanes=with_lanes,
    )


def _ended(
    session_dir: Path,
    worker: ProcessIdentity | None,
    *,
    wait_s: float,
    grace_s: float,
    fanout: bool,
    with_lanes: str,
) -> StopOutcome:
    """The outcome once both bridges are written: the run ended within *wait_s*,
    or what the kill after it signalled."""
    rid = session_dir.name
    if _ended_within(session_dir, worker, wait_s):
        landed = "; what they landed is imported and ranked" if with_lanes else ""
        return StopOutcome(
            rid, True, "stopped", f"{rid} stopped{with_lanes}{landed}", resumable=not fanout
        )
    worker_killed, commands = _kill(session_dir, worker, grace_s)
    continues = "" if fanout else ", and resume continues it"
    if not worker_killed and not commands:
        return StopOutcome(
            rid,
            True,
            "stale",
            f"{rid} did not answer within {wait_s:g} s and nothing of it is left to signal;"
            f" it reads stale{continues}",
            resumable=not fanout,
        )
    killed = ["its worker"] if worker_killed else []
    if commands:
        killed.append(f"{commands} background command{'' if commands == 1 else 's'}")
    return StopOutcome(
        rid,
        True,
        "killed",
        f"{rid} did not answer within {wait_s:g} s: {' and '.join(killed)} killed;"
        f" it reads stale{continues}",
        resumable=not fanout,
    )


def _not_live_state(session_dir: Path) -> str:
    summary = summarize_session_dir(session_dir)
    if summary.status == "parked":
        return "is parked and has not started"
    if produced_result(summary.status):
        return f"is already {summary.status}"
    return f"is not running ({summary.status})"


def _is_fanout(session_dir: Path) -> bool:
    with contextlib.suppress(ManifestError):
        return read_manifest(session_dir).fanout is not None
    return False


def _live_lanes(session_dir: Path) -> list[Path]:
    """A fan-out's live lanes, in lane order."""
    state = layout_of(session_dir).state_dir
    return [
        lane_dir
        for lane in lanes_of(state, session_dir.name)
        if (lane_dir := session_dir.parent / lane.session_id).is_dir() and session_is_live(lane_dir)
    ]


def _ended_within(session_dir: Path, worker: ProcessIdentity | None, wait_s: float) -> bool:
    deadline = time.monotonic() + wait_s
    while session_is_live(session_dir):
        if read_live_worker_identity(session_dir) != worker:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)
    return True


def _kill(session_dir: Path, worker: ProcessIdentity | None, grace_s: float) -> tuple[bool, int]:
    """SIGTERM the worker and its host-side background commands that are still
    alive, SIGKILL what remains after *grace_s*. Returns whether the worker was
    signalled and how many commands were. Never this process: a front-end that
    is the worker (an ACP agent, a test) asks itself to stop through the
    bridges alone."""
    me = os.getpid()
    targets: list[ProcessIdentity] = []
    worker_live = worker is not None and worker[0] != me and process_is_alive(worker)
    if worker is not None and worker_live:
        targets.append(worker)
    commands = [
        identity
        for identity in shell_host_processes(session_dir / SHELLS_DIR)
        if identity[0] != me and process_is_alive(identity)
    ]
    targets.extend(commands)
    for identity in targets:
        signal_group(identity[0], signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while any(process_is_alive(identity) for identity in targets) and time.monotonic() < deadline:
        time.sleep(_POLL_S)
    for identity in targets:
        if process_is_alive(identity):
            signal_group(identity[0], signal.SIGKILL)
    return worker_live, len(commands)
