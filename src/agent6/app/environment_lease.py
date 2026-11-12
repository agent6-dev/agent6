# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Own a prepared execution environment across executions and sharing children.

A lease is a host-side record under the repository's state directory, one per selected
environment (the workspace's discovery under its runtime prefix, or a manual exec prefix).
The run that started the environment owns it; every other run attaches as a borrower. The
owner stops it through the operator's stop command only after every borrower released;
attaching grants no stop authority, and a successful start records what was observed, not
an assumed creation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import time
from collections.abc import Callable, Generator
from typing import Literal

import pydantic

from agent6 import kinds, paths, portable
from agent6.sandbox import container
from agent6.sessions import layout, process

# Under the state dir: one record and one lock per environment.
LEASES_DIR = "environments"
LeaseRole = Literal["owner", "borrower"]


class LeaseError(Exception):
    """The lease record is unreadable or malformed."""


class _Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class Owner(_Model):
    """The run that started the environment, and what it observed doing so."""

    session_id: str
    execution_id: str
    acquired: str
    observed_before: Literal["absent", "running", "unknown"]
    observed_after: Literal["running", "unknown"]
    container_id: str | None = None


class Borrower(_Model):
    """A run attached to an environment it does not own."""

    session_id: str
    execution_id: str
    since: str


class Stop(_Model):
    """The owner's stop command and what was observed after it."""

    ts: str
    outcome: kinds.EnvironmentLifecycle
    reason: str


class Lease(_Model):
    """One environment's ownership record; its `key` names the record file."""

    version: int = 1
    key: str
    selector: tuple[str, ...]
    container_id: str | None = None
    owner: Owner | None = None
    lost: str = ""
    borrowers: tuple[Borrower, ...] = ()
    stop: Stop | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Acquisition:
    """What a run holds after acquiring an environment.

    Attributes:
        ref: The lease key, recorded as the target's `lifecycle_ref`.
        role: Owner when this run started the environment, else borrower.
        reason: Why the run holds that role, in the operator's words.
        discovery: The selected environment, as discovered after any start.
        started: This run's start command made the container appear.
        start_s: How long the start command took; None when none ran.
    """

    ref: str
    role: LeaseRole
    reason: str
    discovery: container.ContainerDiscovery
    started: bool = False
    start_s: float | None = None


def selector(host_root: pathlib.Path, options: container.ContainerOptions) -> tuple[str, ...]:
    """Return what selects the environment: a manual exec prefix, or the workspace's discovery.

    A replacement of a discovered container keeps the selector, so the lease sees the swap.

    Args:
        host_root: The host workspace.
        options: The operator's selection.

    Returns:
        The selector words.
    """
    if options.exec:
        return ("exec", *options.exec)
    return ("discover", str(host_root.resolve()), *options.runtime)


def lease_key(selection: tuple[str, ...]) -> str:
    """Return the record key of a selector."""
    return hashlib.sha256(json.dumps(list(selection)).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _record_path(state_dir: pathlib.Path, key: str) -> pathlib.Path:
    return state_dir / LEASES_DIR / f"{key}.json"


def read_lease(state_dir: pathlib.Path, key: str) -> Lease | None:
    """Return the recorded lease, or None when the environment has none.

    Args:
        state_dir: The repository's state directory.
        key: The lease key.

    Returns:
        The lease.

    Raises:
        LeaseError: The record exists and cannot be read as a lease.
    """
    path = _record_path(state_dir, key)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LeaseError(f"cannot read the environment lease {path}: {exc}") from exc
    try:
        return Lease.model_validate_json(raw)
    except pydantic.ValidationError as exc:
        raise LeaseError(f"malformed environment lease {path}: {exc}") from exc


@contextlib.contextmanager
def _locked(state_dir: pathlib.Path, key: str) -> Generator[list[Lease | None]]:
    """Hold the environment's lock while the caller reads and rewrites its lease.

    Args:
        state_dir: The repository's state directory.
        key: The lease key.

    Yields:
        A one-cell list holding the lease (None when absent); the cell's value is written
        on exit when it is a lease.
    """
    record = _record_path(state_dir, key)
    paths.mkdir_for_real_user(record.parent)
    fd = os.open(record.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        portable.lock_exclusive(fd, blocking=True)
        cell: list[Lease | None] = [read_lease(state_dir, key)]
        yield cell
        if cell[0] is not None:
            portable.atomic_write(record, cell[0].model_dump_json(indent=2))
    finally:
        with contextlib.suppress(OSError):
            portable.unlock(fd)
        os.close(fd)


def _alive(state_dir: pathlib.Path, session_id: str) -> bool:
    found = layout.session_layout(state_dir, session_id)
    return found is not None and process.worker_is_alive(found.session_dir)


def _run_start(
    options: container.ContainerOptions, on_start: Callable[[tuple[str, ...]], None] | None
) -> float:
    """Run the operator's start command, saying so first.

    Args:
        options: The container options whose `start` runs.
        on_start: Told the command before it runs; None tells nobody.

    Returns:
        How long the start took in seconds.
    """
    if on_start is not None:
        on_start(options.start)
    began = time.monotonic()
    container.run_lifecycle(options.start, what="container.start")
    return time.monotonic() - began


def _with_borrower(lease: Lease, session_id: str, execution_id: str) -> Lease:
    others = tuple(b for b in lease.borrowers if b.session_id != session_id)
    joined = Borrower(session_id=session_id, execution_id=execution_id, since=_now())
    return lease.model_copy(update={"borrowers": (*others, joined)})


def acquire(
    state_dir: pathlib.Path,
    host_root: pathlib.Path,
    options: container.ContainerOptions,
    *,
    session_id: str,
    execution_id: str,
    on_start: Callable[[tuple[str, ...]], None] | None = None,
) -> Acquisition:
    """Select the environment, starting it when the operator gave a start command and none runs.

    The run owns the environment only when its own start made the container appear. A
    container already running, a manual exec prefix (whose presence cannot be observed) and
    a replacement of an owned container all leave the run a borrower, with the reason said.

    Args:
        state_dir: The repository's state directory; a child names its origin's, so the
            runs sharing one container meet on one lease.
        host_root: The host workspace the container mounts; a child's origin, not its clone.
        options: The operator's selection, start and stop commands included.
        session_id: The acquiring session.
        execution_id: Its current execution.
        on_start: Told the start command just before it runs; None tells nobody.

    Returns:
        The acquisition, with the discovery to prepare the transport from.

    Raises:
        container.ContainerNotFoundError: Nothing runs and no start command is set.
        container.ContainerLifecycleError: The start command failed, or nothing ran after it.
        container.ContainerError: Discovery refused, or is ambiguous after the start.
        LeaseError: The lease record is malformed.
    """
    before: Literal["absent", "running", "unknown"] = "unknown" if options.exec else "running"
    started = False
    start_s: float | None = None
    try:
        discovery = container.discover(host_root, options)
    except container.ContainerNotFoundError:
        if not options.start:
            raise
        before = "absent"
        start_s = _run_start(options, on_start)
        started = True
        try:
            discovery = container.discover(host_root, options)
        except container.ContainerNotFoundError as exc:
            raise container.ContainerLifecycleError(
                f"container.start ran and no running dev container followed: {exc}"
            ) from exc
    if options.exec and options.start:
        # Presence behind an opaque prefix cannot be observed, so a start proves nothing.
        start_s = _run_start(options, on_start)
    selection = selector(host_root, options)
    key = lease_key(selection)
    found_id = discovery.target.container_id
    with _locked(state_dir, key) as cell:
        lease = cell[0] or Lease(key=key, selector=selection, container_id=found_id)
        owner = lease.owner
        if started and found_id is not None:
            lost = ""
            if owner is not None and owner.session_id != session_id:
                lost = f"run {owner.session_id} owned a container that was gone before this start"
            lease = lease.model_copy(
                update={
                    "container_id": found_id,
                    "owner": Owner(
                        session_id=session_id,
                        execution_id=execution_id,
                        acquired=_now(),
                        observed_before="absent",
                        observed_after="running",
                        container_id=found_id,
                    ),
                    "lost": lost,
                    "stop": None,
                }
            )
            cell[0] = lease
            return Acquisition(
                key, "owner", "started by this run; owned", discovery, started=True, start_s=start_s
            )
        if owner is not None and owner.session_id == session_id:
            if owner.container_id == found_id and found_id is not None:
                lease = lease.model_copy(
                    update={"owner": owner.model_copy(update={"execution_id": execution_id})}
                )
                cell[0] = lease
                return Acquisition(
                    key, "owner", f"owned since {owner.acquired}", discovery, start_s=start_s
                )
            reason = (
                "ownership lost: the container was replaced"
                f" ({owner.container_id or 'unknown'} to {found_id or 'unknown'})"
            )
            lease = lease.model_copy(update={"owner": None, "lost": reason})
        elif owner is not None:
            holder = owner.session_id
            reason = (
                f"attached; owned by run {holder}"
                if _alive(state_dir, holder)
                else f"attached; the owner, run {holder}, is not running (orphaned lease)"
            )
        elif options.exec and options.start:
            reason = (
                "attached; container.start ran, ownership unproved behind an opaque exec prefix"
            )
        elif before == "unknown":
            reason = "attached; an opaque exec prefix grants no ownership"
        else:
            reason = "attached; already running before this run"
        cell[0] = _with_borrower(lease, session_id, execution_id)
        return Acquisition(key, "borrower", reason, discovery, start_s=start_s)


def release(
    state_dir: pathlib.Path,
    ref: str,
    *,
    session_id: str,
    host_root: pathlib.Path,
    options: container.ContainerOptions,
) -> kinds.EnvironmentOutcome:
    """Leave the environment; the owner stops it when every borrower has left and a stop is set.

    Args:
        state_dir: The repository's state directory.
        ref: The lease key the run acquired.
        session_id: The releasing session.
        host_root: The host workspace, for the observation after a stop.
        options: The operator's selection and stop command.

    Returns:
        What was observed: left running (with who still holds it or why no stop ran),
        stopped, or unknown when the stop command failed or the container still runs.
    """
    try:
        with _locked(state_dir, ref) as cell:
            lease = cell[0]
            if lease is None:
                return kinds.EnvironmentOutcome("unknown", "no lease record for this environment")
            lease = lease.model_copy(
                update={
                    "borrowers": tuple(b for b in lease.borrowers if b.session_id != session_id)
                }
            )
            cell[0], outcome = _settle(lease, state_dir, session_id, host_root, options)
            return outcome
    except LeaseError as exc:
        return kinds.EnvironmentOutcome("unknown", str(exc))


def _settle(
    lease: Lease,
    state_dir: pathlib.Path,
    session_id: str,
    host_root: pathlib.Path,
    options: container.ContainerOptions,
) -> tuple[Lease, kinds.EnvironmentOutcome]:
    """Return the lease after the session's release and what the release observed."""
    owner = lease.owner
    if owner is None or owner.session_id != session_id:
        return lease, _left(host_root, options, lease.lost or "attached, not owned")
    live = sorted(b.session_id for b in lease.borrowers if _alive(state_dir, b.session_id))
    ended = sorted(b.session_id for b in lease.borrowers if b.session_id not in live)
    if live:
        return lease, _left(host_root, options, f"still attached: {', '.join(live)}")
    if ended:
        reason = f"attached runs ended without releasing: {', '.join(ended)}"
        return lease, _left(host_root, options, reason)
    if not options.stop:
        return lease, _left(host_root, options, "owned; container.stop unset")
    outcome = _stop(host_root, options)
    lease = lease.model_copy(
        update={
            "stop": Stop(ts=_now(), outcome=outcome.outcome, reason=outcome.reason),
            "owner": None if outcome.outcome == "stopped" else owner,
        }
    )
    return lease, outcome


def _stop(host_root: pathlib.Path, options: container.ContainerOptions) -> kinds.EnvironmentOutcome:
    """Run the stop command and report what discovery sees afterwards.

    Args:
        host_root: The host workspace.
        options: The operator's selection and stop command.

    Returns:
        Stopped when nothing runs afterwards, else unknown with the reason.
    """
    try:
        container.run_lifecycle(options.stop, what="container.stop")
    except container.ContainerError as exc:
        return kinds.EnvironmentOutcome("unknown", str(exc))
    state, seen = presence(host_root, options)
    if state == "absent":
        return kinds.EnvironmentOutcome("stopped", f"container.stop ran; {seen}")
    if state == "running":
        return kinds.EnvironmentOutcome("unknown", "container.stop ran; the container still runs")
    return kinds.EnvironmentOutcome("unknown", f"container.stop ran; {seen}")


def presence(
    host_root: pathlib.Path, options: container.ContainerOptions
) -> tuple[Literal["running", "absent", "unknown"], str]:
    """Look for the environment now.

    Args:
        host_root: The host workspace.
        options: The operator's selection.

    Returns:
        Whether discovery sees the dev container, and the words for what it saw.
    """
    if options.exec:
        return "unknown", "behind container.exec its state is not observed"
    try:
        container.discover(host_root, dataclasses.replace(options, start=(), stop=()))
    except container.ContainerNotFoundError:
        return "absent", "no running dev container"
    except container.ContainerError as exc:
        return "unknown", f"observation failed: {exc}"
    return "running", "the dev container runs"


def _left(
    host_root: pathlib.Path, options: container.ContainerOptions, reason: str
) -> kinds.EnvironmentOutcome:
    """Return left_running once the container is seen running, else what was seen."""
    state, seen = presence(host_root, options)
    if state == "running":
        return kinds.EnvironmentOutcome("left_running", reason)
    if state == "absent":
        return kinds.EnvironmentOutcome("stopped", f"{reason}; {seen} when the run ended")
    return kinds.EnvironmentOutcome("unknown", f"{reason}; {seen}")
