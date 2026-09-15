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
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

from agent6.sandbox.jail import signal_group
from agent6.sessions.ipc import read_worker_pid, request_stop, submit_steer, worker_is_alive
from agent6.tools.background import SHELLS_DIR, shell_host_pids
from agent6.viewmodel import session_is_live, summarize_session_dir
from agent6.viewmodel.listing import produced_result

STOP_WAIT_S = 5.0  # a run that reads its requests ends within this
KILL_GRACE_S = 3.0  # SIGTERM to SIGKILL
_POLL_S = 0.1


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """What a stop did: *how* is `after_step`, `stopped`, `killed`, `not_live`
    (nothing to stop) or `failed` (a request could not be written); *message*
    is the line every surface shows."""

    session_id: str
    ok: bool
    how: str
    message: str


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
    if after_step:
        if not request_stop(session_dir):
            return StopOutcome(rid, False, "failed", f"could not write the stop request for {rid}")
        return StopOutcome(rid, True, "after_step", f"{rid} stops after its current step")
    if not (submit_steer(session_dir, "abort", now=True) and request_stop(session_dir)):
        return StopOutcome(rid, False, "failed", f"could not write the stop request for {rid}")
    if _ended_within(session_dir, wait_s):
        return StopOutcome(rid, True, "stopped", f"{rid} stopped")
    commands = _kill(session_dir, grace_s)
    also = f" with {commands} background command{'' if commands == 1 else 's'}" if commands else ""
    return StopOutcome(
        rid,
        True,
        "killed",
        f"{rid} did not answer within {wait_s:g} s: its worker was killed{also};"
        " it reads stale, and resume continues it",
    )


def _not_live_state(session_dir: Path) -> str:
    summary = summarize_session_dir(session_dir)
    if summary.status == "parked":
        return "is parked and has not started"
    if produced_result(summary.status):
        return f"is already {summary.status}"
    return f"is not running ({summary.status})"


def _ended_within(session_dir: Path, wait_s: float) -> bool:
    deadline = time.monotonic() + wait_s
    while session_is_live(session_dir):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)
    return True


def _kill(session_dir: Path, grace_s: float) -> int:
    """SIGTERM the worker and its host-side background commands, SIGKILL what is
    still alive after *grace_s*. Returns how many commands were signalled."""
    worker = read_worker_pid(session_dir) if worker_is_alive(session_dir) else None
    commands = [pid for pid in shell_host_pids(session_dir / SHELLS_DIR) if _alive(pid)]
    # Never this process: a front-end that is the worker (an ACP agent, a test)
    # asks itself to stop through the bridges alone.
    targets = [
        pid for pid in ([worker] if worker is not None else []) + commands if pid != os.getpid()
    ]
    for pid in targets:
        signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while any(_alive(pid) for pid in targets) and time.monotonic() < deadline:
        time.sleep(_POLL_S)
    for pid in targets:
        if _alive(pid):
            signal_group(pid, signal.SIGKILL)
    return len(commands)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
