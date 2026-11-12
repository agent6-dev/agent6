# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Stop a session, for `agent6 stop`, the TUI and web Stop entries, `/stop` and an ACP cancel.

A stop now writes both bridges the loop reads (the abort steer cuts a model call in flight,
the stop marker, written with its `now` urgency, breaks a command or approval wait), then
waits for the run to end. A worker that has not ended by then is not reading requests: it and
the background commands it started on the host get SIGTERM, then SIGKILL after a grace, and
the jail's children die with it. A run's worker and a group's parent take the
SIGTERM as a Ctrl-C stop and journal their end; resume continues a run.
A request-only stop writes both bridges without waiting for or signalling the worker.
ACP cancel uses this form because its server is the worker. A stop after the step writes
the plain marker alone, so the step's command finishes and its tool results and auto-commit
land first.

A group's stop ends its live children first, then the parent, which imports and ranks what
the children landed before it ends; that drain gets a longer wait.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import signal
import time

from agent6.sandbox import jail
from agent6.sessions import ipc, lineage, process, shells
from agent6.viewmodel import listing, session_is_live, summarize_session_dir
from agent6.viewmodel.session import status as session_status

STOP_WAIT_S = 5.0  # a run that reads its requests ends within this
CHILDREN_WAIT_S = 30.0  # a run drains its ended children and imports what landed within this
KILL_GRACE_S = 3.0  # SIGTERM to SIGKILL
_POLL_S = 0.1


@dataclasses.dataclass(frozen=True, slots=True)
class StopOutcome:
    """Record what a stop did.

    Attributes:
        session_id: The stopped session.
        ok: Whether the stop took.
        how: `after_turn`, `requested`, `stopped`, `killed`, `stale` (no answer and nothing
            left to signal), `children_stopped` (the session itself had ended; its live children
            were stopped), `not_live` (nothing to stop) or `failed` (a request could not be
            written).
        message: The line every surface shows.
        resumable: `agent6 resume` continues it (a group has no loop to resume).
    """

    session_id: str
    ok: bool
    how: str
    message: str
    resumable: bool = False


def stop_session(
    session_dir: pathlib.Path,
    *,
    after_turn: bool = False,
    request_only: bool = False,
    wait_s: float | None = None,
    grace_s: float | None = None,
) -> StopOutcome:
    """Stop the session, as the module docstring describes.

    Args:
        session_dir: The session to stop.
        after_turn: Let the current turn finish first.
        request_only: Write the requests without waiting for or signalling the worker.
        wait_s: How long to wait for the run to end; None takes `STOP_WAIT_S`.
        grace_s: SIGTERM to SIGKILL; None takes `KILL_GRACE_S`.

    Returns:
        What the stop did.
    """
    rid = session_dir.name
    wait_s = STOP_WAIT_S if wait_s is None else wait_s
    grace_s = KILL_GRACE_S if grace_s is None else grace_s
    # Children are found by lineage, so a dead parent's count too.
    children = _live_children(session_dir)
    child_count = f"{len(children)} live {'children' if len(children) > 1 else 'child'}"
    if not session_is_live(session_dir):
        if not children:
            return StopOutcome(
                rid, False, "not_live", f"{rid} {_not_live_state(session_dir)}; nothing to stop"
            )
        for child in children:
            stop_session(
                child,
                after_turn=after_turn,
                request_only=request_only,
                wait_s=wait_s,
                grace_s=grace_s,
            )
        return StopOutcome(
            rid,
            True,
            "children_stopped",
            f"{rid} {_not_live_state(session_dir)}; its {child_count} stopped",
        )
    resumable = not listing.resume_blocker(session_dir)
    with_children = f" with its {child_count}" if children else ""
    worker = process.read_live_worker_identity(session_dir)
    # The session's own markers first: a parent reads its stop as its children end, and one
    # written after they ended is never read.
    written = (
        ipc.request_stop(session_dir)
        if after_turn
        else ipc.submit_steer(session_dir, "abort", now=True)
        and ipc.request_stop(session_dir, now=True)
    )
    if not written:
        return StopOutcome(rid, False, "failed", f"could not write the stop request for {rid}")
    for child in children:
        stop_session(
            child, after_turn=after_turn, request_only=request_only, wait_s=wait_s, grace_s=grace_s
        )
    if after_turn:
        when = "their current turns" if children else "its current turn"
        return StopOutcome(
            rid, True, "after_turn", f"{rid} stops{with_children} after {when}", resumable=resumable
        )
    if request_only:
        return StopOutcome(
            rid, True, "requested", f"{rid} stop requested{with_children}", resumable=resumable
        )
    return _ended(
        session_dir,
        worker,
        wait_s=max(wait_s, CHILDREN_WAIT_S) if children else wait_s,
        grace_s=grace_s,
        resumable=resumable,
        with_children=with_children,
    )


def _ended(
    session_dir: pathlib.Path,
    worker: process.ProcessIdentity | None,
    *,
    wait_s: float,
    grace_s: float,
    resumable: bool,
    with_children: str,
) -> StopOutcome:
    """Return the outcome once both bridges are written: ended in time, or what the kill did."""
    rid = session_dir.name
    if _ended_within(session_dir, worker, wait_s):
        return StopOutcome(
            rid, True, "stopped", f"{rid} stopped{with_children}", resumable=resumable
        )
    worker_killed, commands = _kill(session_dir, worker, grace_s)
    continues = ", and resume continues it" if resumable else ""
    if not worker_killed and not commands:
        return StopOutcome(
            rid,
            True,
            "stale",
            f"{rid} did not answer within {wait_s:g} s and nothing of it is left to signal;"
            f" it reads stale{continues}",
            resumable=resumable,
        )
    killed = ["its worker"] if worker_killed else []
    if commands:
        killed.append(f"{commands} background command{'' if commands == 1 else 's'}")
    return StopOutcome(
        rid,
        True,
        "killed",
        f"{rid} did not answer within {wait_s:g} s: {' and '.join(killed)} killed{continues}",
        resumable=resumable,
    )


def _not_live_state(session_dir: pathlib.Path) -> str:
    """Return the phrase for a session that is not running."""
    summary = summarize_session_dir(session_dir)
    if summary.status == "parked":
        return "is parked and has not started"
    if session_status.produced_result(summary.status):
        return f"is already {summary.status}"
    return f"is not running ({summary.status})"


def _live_children(session_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return live descendants across buckets and clone links, including dead intermediaries."""
    return [child for child in lineage.descendants_of(session_dir) if session_is_live(child)]


def _ended_within(
    session_dir: pathlib.Path, worker: process.ProcessIdentity | None, wait_s: float
) -> bool:
    """Return whether the run ended, or its worker changed, within the wait."""
    deadline = time.monotonic() + wait_s
    while session_is_live(session_dir):
        if process.read_live_worker_identity(session_dir) != worker:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_S)
    return True


def _kill(
    session_dir: pathlib.Path, worker: process.ProcessIdentity | None, grace_s: float
) -> tuple[bool, int]:
    """SIGTERM the live worker and its host-side background commands, then SIGKILL the rest.

    Never this process: a front-end that is the worker stops through the bridges alone.

    Args:
        session_dir: The session being stopped.
        worker: The worker's identity, when one was read.
        grace_s: SIGTERM to SIGKILL.

    Returns:
        Whether the worker was signalled, and how many commands were.
    """
    me = os.getpid()
    targets: list[process.ProcessIdentity] = []
    worker_live = worker is not None and worker[0] != me and process.process_is_alive(worker)
    if worker is not None and worker_live:
        targets.append(worker)
    commands = [
        identity
        for identity in shells.shell_host_processes(session_dir / shells.SHELLS_DIR)
        if identity[0] != me and process.process_is_alive(identity)
    ]
    targets.extend(commands)
    for identity in targets:
        jail.signal_group(identity[0], signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while (
        any(process.process_is_alive(identity) for identity in targets)
        and time.monotonic() < deadline
    ):
        time.sleep(_POLL_S)
    for identity in targets:
        if process.process_is_alive(identity):
            jail.signal_group(identity[0], signal.SIGKILL)
    return worker_live, len(commands)
