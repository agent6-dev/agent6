# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Process identities, worker and network records, and live front-end claims."""

from __future__ import annotations

import contextlib
import os
import pathlib
import subprocess
import uuid
from typing import Any

from agent6 import event_log, paths, portable

FRONTENDS_DIR = "frontends"
WORKER_PID_FILE = "worker.pid"
# `agent6 exec` and `agent6 forward` join the session network through this pid's /proc entry.
NETNS_PID_FILE = "netns.pid"


def register_frontend(session_dir: pathlib.Path, pid: int) -> None:
    """Register a pid as a live answering front-end.

    One claim file per front-end, so any number watch concurrently and none deregisters
    another; the body is the process start time, which tells a live front-end from a
    recycled pid.

    Args:
        session_dir: The session directory.
        pid: The front-end's pid.
    """
    d = session_dir / FRONTENDS_DIR
    paths.mkdir_for_real_user(d)
    portable.atomic_write(d / str(pid), _proc_start_time(pid))


def unregister_frontend(session_dir: pathlib.Path, pid: int) -> None:
    """Drop a pid's own claim, leaving the other front-ends' claims.

    Args:
        session_dir: The session directory.
        pid: The front-end's pid.
    """
    with contextlib.suppress(OSError):
        (session_dir / FRONTENDS_DIR / str(pid)).unlink()


def pid_alive(pid: int) -> bool:
    """Return whether a live process we own has the pid.

    A foreign-owned pid reads as dead: agent6 probes only processes the same user spawned,
    so it means the number was reused by another user's process, and reading it as live
    would keep a dead run "running" forever. A zombie is dead, and 0 and -1 are not pids.

    Args:
        pid: The pid.

    Returns:
        True for a live, non-zombie process signal 0 reaches.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    if _HAS_PROC:
        fields = _proc_stat_fields(pid)
        return bool(fields) and fields[0] not in {"Z", "X", "x"}
    state = _ps_field(pid, "stat")
    return not state.startswith("Z")


# /proc exists on Linux; on macOS `ps` answers the same question instead.
_HAS_PROC = pathlib.Path("/proc").is_dir()


def _proc_stat_fields(pid: int) -> list[str]:
    """Return the fields after comm in `/proc/<pid>/stat`.

    Args:
        pid: The pid.

    Returns:
        The fields; [] when the entry vanished.
    """
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return []
    return stat.rpartition(")")[2].split()


def _ps_field(pid: int, column: str) -> str:
    """Return a process field via `ps -o <column>=`.

    A fixed argv over a pid agent6 itself recorded, never LLM output: the subprocess
    allowlist in docs/security.md.

    Args:
        pid: The pid.
        column: The fixed ps column, `stat` or `lstart`.

    Returns:
        The field text; "" for a dead pid or a host without ps.
    """
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", f"{column}="],
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode(errors="replace").strip()


def _proc_start_time(pid: int) -> str:
    """Return the start-time identity of a pid.

    Field 22 of `/proc/<pid>/stat` on Linux, split after the last `)` since comm may hold
    spaces and parens; `ps` where /proc is absent.

    Args:
        pid: The pid.

    Returns:
        The identity; "" when it cannot be read.
    """
    if not _HAS_PROC:
        return _ps_field(pid, "lstart")
    fields = _proc_stat_fields(pid)
    return fields[19] if len(fields) > 19 else ""


def write_session_netns_pid(session_dir: pathlib.Path, pid: int) -> None:
    """Publish the holder of this run's session network, for `agent6 exec`.

    Args:
        session_dir: The session directory.
        pid: The holder's pid.
    """
    portable.atomic_write(session_dir / NETNS_PID_FILE, pid_record(pid))


def read_session_netns_pid(session_dir: pathlib.Path) -> int | None:
    """Return the live holder of this run's session network, or None.

    The recorded start time is checked: joining on liveness alone would put `agent6 exec`
    inside an unrelated process's namespaces once the kernel reused the pid.

    Args:
        session_dir: The session directory.

    Returns:
        The holder's pid; None when the run has no session network, the process is gone,
        or the pid was reused.
    """
    rec = _parse_pid_record(session_dir / NETNS_PID_FILE)
    if rec is None:
        return None
    pid, recorded_start = rec
    if pid <= 0 or not pathlib.Path(f"/proc/{pid}/ns/net").exists():
        return None
    if recorded_start and _proc_start_time(pid) != recorded_start:
        return None
    return pid


def listening_ports(session_dir: pathlib.Path) -> list[int]:
    """Return the TCP ports something in the run is listening on.

    Read from `/proc/<holder>/net/`, that process's own view: a namespace's sockets are
    readable without entering it, so no fork and no setns.

    Args:
        session_dir: The session directory.

    Returns:
        The ports, sorted; [] without a live session network.
    """
    pid = read_session_netns_pid(session_dir)
    if pid is None:
        return []
    ports: set[int] = set()
    for name in ("tcp", "tcp6"):
        with contextlib.suppress(OSError):
            for line in (
                pathlib.Path(f"/proc/{pid}/net/{name}").read_text(encoding="utf-8").splitlines()
            ):
                cols = line.split()
                # st == 0A is LISTEN; the local address is host:port in hex.
                if len(cols) > 3 and cols[3] == "0A" and ":" in cols[1]:
                    ports.add(int(cols[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def clear_session_netns_pid(session_dir: pathlib.Path) -> None:
    """Drop the session-network holder record.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(OSError):
        (session_dir / NETNS_PID_FILE).unlink()


def write_worker_pid(session_dir: pathlib.Path, pid: int) -> None:
    """Record the worker pid with its start-time identity, so liveness probes need no events.

    The identity keeps a recycled pid, after a killed worker left the file behind, from
    reading a dead run as running forever.

    Args:
        session_dir: The session directory.
        pid: The worker's pid.
    """
    # Atomic: a truncating write would expose a prefix of the pid with the identity stripped.
    portable.atomic_write(session_dir / WORKER_PID_FILE, pid_record(pid))


def emit_session_start(
    events: event_log.EventSink, session_dir: pathlib.Path, event_type: str, /, **fields: Any
) -> None:
    """Emit a start-family event with the worker pid already on disk.

    The status fold reads a started session with no pid file as one whose worker exited.

    Args:
        events: The event sink.
        session_dir: The session directory.
        event_type: `session.start` or `loop.resume.start`.
        **fields: The event's fields.
    """
    write_worker_pid(session_dir, os.getpid())
    fields.setdefault("execution_id", uuid.uuid4().hex)
    events.emit(event_type, **fields)


def clear_worker_pid(session_dir: pathlib.Path) -> None:
    """Drop the worker pid record.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (session_dir / WORKER_PID_FILE).unlink()


def pid_record(pid: int) -> str:
    """Render a pid with the identity that distinguishes it from a later reuse of the number.

    Args:
        pid: The pid.

    Returns:
        `<pid> <start time>`, the start time omitted when unreadable.
    """
    return f"{pid} {_proc_start_time(pid)}".rstrip()


def _parse_pid_record(path: pathlib.Path) -> tuple[int, str] | None:
    """Parse a pid record file.

    Args:
        path: The file.

    Returns:
        `(pid, start_time)`, the start time "" when none was recorded; None when the file is
        absent or malformed.
    """
    try:
        tokens = path.read_text(encoding="utf-8").split(maxsplit=1)
        return int(tokens[0]), tokens[1].strip() if len(tokens) > 1 else ""
    except (OSError, ValueError, IndexError):
        return None


def _read_pid_record(session_dir: pathlib.Path) -> tuple[int, str] | None:
    """Parse the worker pid record.

    Args:
        session_dir: The session directory.

    Returns:
        `(pid, start_time)`, or None.
    """
    return _parse_pid_record(session_dir / WORKER_PID_FILE)


type ProcessIdentity = tuple[int, str]


def process_identity(pid: int) -> ProcessIdentity:
    """Return a pid with the start time that distinguishes it from later reuse.

    Args:
        pid: The pid.

    Returns:
        The identity.
    """
    return pid, _proc_start_time(pid)


def process_is_alive(identity: ProcessIdentity) -> bool:
    """Return whether an identity still names its live process.

    Args:
        identity: The pid and start time.

    Returns:
        True when the pid is alive and the start time matches.
    """
    return _still_the_process(*identity)


def read_worker_pid(session_dir: pathlib.Path) -> int | None:
    """Return the recorded worker pid, live or not.

    Args:
        session_dir: The session directory.

    Returns:
        The pid, or None when none is recorded.
    """
    rec = _read_pid_record(session_dir)
    return None if rec is None else rec[0]


def read_live_worker_identity(session_dir: pathlib.Path) -> ProcessIdentity | None:
    """Return the live worker's recorded identity, or None.

    The identity comes from the same read that validates it, so a caller keeps targeting one
    worker while another resume takes over the run dir.

    Args:
        session_dir: The session directory.

    Returns:
        The pid and start time, or None when no live worker is recorded.
    """
    rec = _read_pid_record(session_dir)
    if rec is None or not _still_the_process(*rec):
        return None
    return rec


def worker_is_alive(session_dir: pathlib.Path) -> bool:
    """Return whether worker.pid names a live process that is the recorded worker.

    Args:
        session_dir: The session directory.

    Returns:
        True when the pid is alive and its start time matches; a recycled pid reads dead.
    """
    return read_live_worker_identity(session_dir) is not None


def _still_the_process(pid: int, recorded_start: str) -> bool:
    """Return whether the pid is alive and still the process that recorded the start time.

    A dead front-end whose pid our own later process reused would otherwise read live, and
    an approval would wait out its whole timeout instead of the dead grace.

    Args:
        pid: The pid.
        recorded_start: The recorded start time; "" is trusted.

    Returns:
        True when alive and, with a start time recorded, matching.
    """
    if not pid_alive(pid):
        return False
    return not recorded_start or _proc_start_time(pid) == recorded_start


def _claim_is_live(claim: pathlib.Path, pid: int) -> bool:
    """Return whether a claim still names the front-end that wrote it.

    Args:
        claim: The claim file.
        pid: The pid the claim is named for.

    Returns:
        True when the pid is alive and its start time matches the claim's body.
    """
    try:
        recorded_start = claim.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return _still_the_process(pid, recorded_start)


def frontend_is_live(session_dir: pathlib.Path) -> bool:
    """Return whether any registered front-end is live, pruning dead claims in passing.

    Args:
        session_dir: The session directory.

    Returns:
        True when a claim names a live process with a matching start time.
    """
    try:
        entries = list((session_dir / FRONTENDS_DIR).iterdir())
    except OSError:
        return False
    live = False
    for f in entries:
        # atomic_write's hidden staging sibling is not a claim; deleting it fails the rename.
        if f.name.startswith("."):
            continue
        try:
            pid = int(f.name)
        except ValueError:
            pid = -1
        if _claim_is_live(f, pid):
            live = True
        else:
            with contextlib.suppress(OSError):
                f.unlink()
    return live
