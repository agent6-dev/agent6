# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The file bridge between the harness process and a front-end.

The harness emits a prompt event to logs.jsonl and polls `<session_dir>/approvals/<id>.answer`;
a front-end tails the log, shows the prompt, and writes the operator's literal choice
(`yes`, `no`, `session`, `session-deny`) to that file. With no live claim under
`<session_dir>/frontends/` the harness falls back to stdin. Files rather than a socket: the
log is already the cross-process contract, a front-end may crash without taking the harness
down, and every front-end mirrors the same files. An answer is staged whole, fsync'd and
hard linked into place, so a reader never consumes a torn file and the first answer stands.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from agent6 import events as agent6_events
from agent6 import paths, portable

APPROVAL_DIR_NAME = "approvals"
QUESTION_DIR_NAME = "questions"
# Operator requests, one file each, sorted oldest first; unlike the answer and steer bridges
# a request survives an execution boundary.
QUEUE_DIR_NAME = "queue"
FRONTENDS_DIR = "frontends"
WORKER_PID_FILE = "worker.pid"
# `agent6 exec` and `agent6 forward` join the session network through this pid's /proc entry.
NETNS_PID_FILE = "netns.pid"
STEER_ANSWER_FILE = "steer.answer"

# A page reload re-registers within seconds; without the grace one poll in that gap denies.
# 30s outlasts a reload while a gone front-end still fails over well before the answer timeout.
FRONTEND_DEAD_GRACE_S = 30.0


def approvals_dir(session_dir: pathlib.Path) -> pathlib.Path:
    """Return the approvals dir, created.

    A read asks `approvals_path` instead: creating the dir bumps the session dir's mtime,
    which the listings sort by.

    Args:
        session_dir: The session directory.

    Returns:
        The approvals dir.
    """
    p = session_dir / APPROVAL_DIR_NAME
    paths.mkdir_for_real_user(p)
    return p


def approvals_path(session_dir: pathlib.Path) -> pathlib.Path:
    """Return where the approvals dir is, never creating it.

    Args:
        session_dir: The session directory.

    Returns:
        The approvals dir.
    """
    return session_dir / APPROVAL_DIR_NAME


def queue_dir(session_dir: pathlib.Path) -> pathlib.Path:
    """Return the request queue dir, created.

    Args:
        session_dir: The session directory.

    Returns:
        The queue dir.
    """
    p = session_dir / QUEUE_DIR_NAME
    paths.mkdir_for_real_user(p)
    return p


def queue_path(session_dir: pathlib.Path) -> pathlib.Path:
    """Return where the request queue dir is, never creating it.

    Args:
        session_dir: The session directory.

    Returns:
        The queue dir.
    """
    return session_dir / QUEUE_DIR_NAME


RequestKind = Literal["task", "standing", "retire"]


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorRequest:
    """One thing the operator asked of a live run from a composer or `agent6 steer`.

    Attributes:
        kind: A task for the graph, a standing goal, or a task to retire.
        text: The task text, the goal, or the task id.
    """

    kind: RequestKind
    text: str


def queue_request(session_dir: pathlib.Path, kind: RequestKind, text: str) -> None:
    """Queue one request for the run to apply at its next turn boundary.

    The file name carries the clock, so the drain takes requests in the order asked; two
    standing goals in a row both land, the later replacing the earlier when adopted.

    Args:
        session_dir: The session directory.
        kind: The request kind.
        text: The task text, the goal, or the task id.
    """
    target = queue_dir(session_dir) / f"{time.time_ns():020d}-{os.getpid()}.{kind}"
    portable.atomic_write(target, text)


def drain_requests(session_dir: pathlib.Path) -> list[OperatorRequest]:
    """Take every queued request, oldest first, removing each as it is read.

    A file that vanishes under the read was drained by someone else; an unreadable or blank
    one is dropped rather than re-read forever.

    Args:
        session_dir: The session directory.

    Returns:
        The requests.
    """
    directory = queue_path(session_dir)
    if not directory.is_dir():
        return []
    out: list[OperatorRequest] = []
    for path in sorted(p for p in directory.iterdir() if p.suffix[1:] in _REQUEST_KINDS):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        with contextlib.suppress(OSError):
            path.unlink()
        if text.strip():
            out.append(OperatorRequest(cast("RequestKind", path.suffix[1:]), text.strip()))
    return out


_REQUEST_KINDS: frozenset[str] = frozenset({"task", "standing", "retire"})


def _contained(
    directory: pathlib.Path, filename: str, *, untrusted: str, what: str
) -> pathlib.Path:
    """Return `<directory>/<filename>`, refusing a name that is not one plain file inside it.

    A prompt id comes from a web request and an approval scope from a tool name the model
    chose; a separator or `..` would walk out of the run, so containment is a check on the
    write primitive rather than a caller's manners.

    Args:
        directory: The bridge dir.
        filename: The file name built from the untrusted string.
        untrusted: The string from outside.
        what: What it names, for the message.

    Returns:
        The contained path.

    Raises:
        ValueError: The string is empty, holds a separator or NUL, or escapes the directory.
    """
    if not untrusted or "/" in untrusted or os.sep in untrusted or "\x00" in untrusted:
        raise ValueError(f"unsafe {what}: {untrusted!r}")
    target = directory / filename
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"unsafe {what}: {untrusted!r}")
    return target


def _answer_path(directory: pathlib.Path, answer_id: str) -> pathlib.Path:
    """Return the contained `<id>.answer` path.

    Args:
        directory: The bridge dir.
        answer_id: The prompt or question id.

    Returns:
        The path.

    Raises:
        ValueError: The id is not a plain file name.
    """
    return _contained(directory, f"{answer_id}.answer", untrusted=answer_id, what="answer id")


# File timestamps trail `time.time()` by up to one kernel tick (4 ms at HZ=250).
TIMESTAMP_SLACK_S = 0.01


def clear_pending_answers(session_dir: pathlib.Path, *, started_at: float) -> None:
    """Drop the bridge state an execution inherits, at its start; best-effort.

    The `*.answer` files, the steer answer and marker (a phantom prompt no front-end
    answers), and the stop and compact markers (an instant re-stop) go; only files older
    than the start, less `TIMESTAMP_SLACK_S`: one written since belongs to the execution
    that is starting. Front-end claims need no sweep: `frontend_is_live` prunes dead ones,
    and a live watcher's must survive so its modals stay wired up.

    Args:
        session_dir: The session directory.
        started_at: This execution's start as `time.time()`; ACP passes its turn's start,
            a machine's crash recovery passes now.
    """
    answer_dirs = (session_dir / APPROVAL_DIR_NAME, session_dir / QUESTION_DIR_NAME)
    markers = (STEER_ANSWER_FILE, STEER_REQUEST_FILE, STOP_REQUEST_FILE, COMPACT_REQUEST_FILE)
    answers = (f for d in answer_dirs for f in d.glob("*.answer"))
    for path in (*answers, *(session_dir / name for name in markers)):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < started_at - TIMESTAMP_SLACK_S:
                path.unlink()


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
    state = _ps_state(pid)
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


def _ps_state(pid: int) -> str:
    """Return the process state via `ps -o stat=`, the zombie check where /proc is absent.

    Args:
        pid: The pid.

    Returns:
        The state, a leading `Z` for a zombie; "" for a dead pid or a host without ps.
    """
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode(errors="replace").strip()


def _ps_start_time(pid: int) -> str:
    """Return the start-time identity via `ps -o lstart=`.

    A fixed argv over a pid agent6 itself recorded, never LLM output: the subprocess
    allowlist in docs/security.md.

    Args:
        pid: The pid.

    Returns:
        The start time text; "" for a dead pid or a host without ps.
    """
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
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
        return _ps_start_time(pid)
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
    events: agent6_events.EventSink, session_dir: pathlib.Path, event_type: str, /, **fields: Any
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


def effective_away(session_dir: pathlib.Path) -> str:
    """Return this run's away answer: the env a launcher set, else the recorded one.

    The one owner, so the preflight and the approver agree; an invalid env value reads as
    unset, never as an unattended run having a policy.

    Args:
        session_dir: The session directory.

    Returns:
        A value of `AWAY_MODES`, or "".
    """
    marker = os.environ.get("AGENT6_DETACHED_AWAY", "")
    return marker if marker in AWAY_MODES else away_mode(session_dir)


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


def _consume_answer(target: pathlib.Path) -> str | None:
    """Read and delete an answer file, so it is never re-read on a later prompt or resume.

    Args:
        target: The answer file.

    Returns:
        Its text, or None when absent.
    """
    try:
        txt = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    with contextlib.suppress(FileNotFoundError):
        target.unlink()
    return txt


def _await_answer(
    target: pathlib.Path,
    live: pathlib.Path,
    *,
    timeout_s: float,
    poll_s: float,
    dead_grace_s: float,
) -> str | None:
    """Poll for an answer file, consume it, and return its text.

    A final consume runs before a timeout or dead verdict, so an answer landing between the
    round's read and the verdict is honoured rather than denied with its file left on disk.

    Args:
        target: The answer file.
        live: The dir whose front-end claims and stop markers gate the wait.
        timeout_s: The most to wait.
        poll_s: The poll interval.
        dead_grace_s: How long the front-end may stay dead before the wait ends.

    Returns:
        The text; None on a stop, a dead front-end past the grace, or the timeout.
    """
    deadline = time.monotonic() + timeout_s
    dead_since: float | None = None
    while time.monotonic() < deadline:
        if (txt := _consume_answer(target)) is not None:
            return txt
        if steer_answer_is_abort(live) or stop_request_pending(live):
            return None
        if frontend_is_live(live):
            dead_since = None
        else:
            now = time.monotonic()
            if dead_since is None:
                dead_since = now
            if now - dead_since >= dead_grace_s:
                break
        time.sleep(poll_s)
    return _consume_answer(target)


def await_frontend_reply[T](
    session_dir: pathlib.Path, read_once: Callable[[], T | None]
) -> T | None:
    """Block in the detach `wait` mode until an answer arrives or a stop ends the run.

    `read_once` is called even with no claim registered: a claim-less front-end (the web UI
    over HTTP) writes the same answer files, and the answer's existence is the proof. It
    paces itself through its dead grace; the sleep paces the no-claim loop.

    Args:
        session_dir: The session directory.
        read_once: One bounded read of the answer, None when none arrived.

    Returns:
        The reply; None on a steer abort or a stop marker.
    """
    while True:
        if steer_answer_is_abort(session_dir) or stop_request_pending(session_dir):
            return None
        reply = read_once()
        if reply is not None:
            return reply
        if not frontend_is_live(session_dir):
            time.sleep(1.0)


def write_answer(session_dir: pathlib.Path, prompt_id: str, answer: str) -> bool:
    """Write the operator's literal choice to an approval prompt, from a front-end.

    Args:
        session_dir: The session directory.
        prompt_id: The prompt id.
        answer: `yes`, `no`, `session` or `session-deny`.

    Returns:
        False when another surface answered first; that answer stands.

    Raises:
        ValueError: The prompt id is not a plain file name.
    """
    return _publish_answer(_answer_path(approvals_dir(session_dir), prompt_id), answer)


ANSWERED_ELSEWHERE = "already answered from another surface"


def _publish_answer(target: pathlib.Path, content: str) -> bool:
    """Publish an answer unless one is there already.

    Written whole to a staging file, fsync'd, hard linked into place, the directory fsync'd:
    a reader never sees a partial file, concurrent writers never share a name, and the entry
    survives a crash. The state dir's filesystem must support hard links.

    Args:
        target: The answer file.
        content: The answer.

    Returns:
        False when an answer was there already; the first stands.
    """
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".staged", dir=target.parent
    )
    staged = pathlib.Path(staged_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            return False
        portable.fsync_dir(target.parent)
        return True
    finally:
        staged.unlink(missing_ok=True)


def answer_written(session_dir: pathlib.Path, prompt_id: str) -> bool:
    """Return whether an answer to the prompt is on disk; a peek, nothing consumed.

    The terminal prompt polls it, so an answer from another route ends the prompt.

    Args:
        session_dir: The session directory.
        prompt_id: The prompt id.

    Returns:
        True when the answer file exists.
    """
    return _answer_path(approvals_path(session_dir), prompt_id).exists()


def clear_answer(session_dir: pathlib.Path, prompt_id: str) -> None:
    """Drop a pre-existing answer to a prompt, right before the prompt is emitted.

    Prompt ids are sequential counters, so a hostile POST could pre-write the next one and
    auto-approve a command the operator never saw; a legitimate answer is only written after
    the front-end renders the prompt, so none is lost.

    Args:
        session_dir: The session directory.
        prompt_id: The prompt id.
    """
    with contextlib.suppress(OSError):
        _answer_path(approvals_dir(session_dir), prompt_id).unlink(missing_ok=True)


def clear_question_answers(session_dir: pathlib.Path, question_id: str) -> None:
    """Drop a pre-existing answer to a question, as `clear_answer` does for a prompt.

    Args:
        session_dir: The session directory.
        question_id: The question id.
    """
    with contextlib.suppress(OSError):
        _answer_path(questions_dir(session_dir), question_id).unlink(missing_ok=True)


# One marker per scope: a standing answer grants what its prompt said and no more. The command
# tools share one scope; each MCP server has its own (a name is [A-Za-z0-9_-]+, a safe suffix).
# Markers are not `*.answer`s: they survive a detached run's resumes, and an interactive start
# drops the allow markers with the away mode.
COMMAND_SCOPE = "command"
MCP_SCOPE_PREFIX = "mcp."
SESSION_ALLOW_FILE = "session.allow"
SESSION_DENY_FILE = "session.deny"


def _marker_path(session_dir: pathlib.Path, stem: str, scope: str) -> pathlib.Path:
    """Return one scope's marker path; the writers create the dir, a probe never does.

    Args:
        session_dir: The session directory.
        stem: `SESSION_ALLOW_FILE` or `SESSION_DENY_FILE`.
        scope: The approval scope.

    Returns:
        The contained path.

    Raises:
        ValueError: The scope is not a plain file suffix.
    """
    return _contained(
        approvals_path(session_dir), f"{stem}.{scope}", untrusted=scope, what="approval scope"
    )


def set_session_allow(session_dir: pathlib.Path, scope: str) -> None:
    """Record the operator's "allow all of this scope for the session" choice.

    Args:
        session_dir: The session directory.
        scope: The approval scope.
    """
    target = _marker_path(session_dir, SESSION_ALLOW_FILE, scope)
    paths.mkdir_for_real_user(target.parent)
    portable.atomic_write(target, "1")


def session_allow_set(session_dir: pathlib.Path, scope: str) -> bool:
    """Return whether the scope's allow marker is set.

    Args:
        session_dir: The session directory.
        scope: The approval scope.

    Returns:
        True when the marker exists.
    """
    return _marker_path(session_dir, SESSION_ALLOW_FILE, scope).exists()


def set_session_deny(session_dir: pathlib.Path, scope: str) -> None:
    """Record the operator's "none of this scope for the rest of the session" choice.

    A session deny withdraws the tools rather than refusing each call, so the model stops
    spending turns on a door that will not open.

    Args:
        session_dir: The session directory.
        scope: The approval scope.
    """
    target = _marker_path(session_dir, SESSION_DENY_FILE, scope)
    paths.mkdir_for_real_user(target.parent)
    portable.atomic_write(target, "1")


def session_deny_set(session_dir: pathlib.Path, scope: str) -> bool:
    """Return whether the scope's deny marker is set.

    Args:
        session_dir: The session directory.
        scope: The approval scope.

    Returns:
        True when the marker exists.
    """
    return _marker_path(session_dir, SESSION_DENY_FILE, scope).exists()


def record_answer(session_dir: pathlib.Path, answer: str, scope: str | None) -> bool:
    """Apply the operator's literal answer and return the verdict for this call.

    The one place an answer's meaning is decided. Anything unrecognised is a deny, so a
    truncated or hand-written answer file cannot approve.

    Args:
        session_dir: The session directory.
        answer: The literal choice.
        scope: The approval scope a session choice persists under; None for a gate with no
            standing answer (`fetch`), where "allow all" grants this call only.

    Returns:
        True for `yes` or `session`.
    """
    if scope:
        if answer == "session":
            set_session_allow(session_dir, scope)
        elif answer == "session-deny":
            set_session_deny(session_dir, scope)
    return answer in {"yes", "session"}


def effective_run_commands(configured: str, session_dir: pathlib.Path) -> str:
    """Return the command policy in force right now.

    One answer from the configured knob, the session choice and the away mode, so every
    consumer agrees. Only `ask` moves: a configured `yes` or `no` is the standing policy.
    `no` withdraws the tools, the same wiring for `--no-commands`, a session deny and an
    away mode of `deny`.

    Args:
        configured: `[sandbox].run_commands`.
        session_dir: The session directory.

    Returns:
        `no`, `ask` or `yes`.
    """
    if configured != "ask":
        return configured
    if session_allow_set(session_dir, COMMAND_SCOPE):
        return "yes"
    if session_deny_set(session_dir, COMMAND_SCOPE) or away_mode(session_dir) == "deny":
        return "no"
    return "ask"


# How a detached run answers approvals and questions; `approve` sets the allow marker instead.
AWAY_MODE_FILE = "away.mode"
# The valid AGENT6_DETACHED_AWAY values; a typo must not read as a known intent, which would
# lift the preflight refusal and leave the run waiting forever at its first approval.
AwayMode = Literal["wait", "deny", "approve"]
AWAY_MODES: tuple[AwayMode, ...] = ("wait", "deny", "approve")


def set_away_mode(session_dir: pathlib.Path, mode: str) -> None:
    """Record the detach "while away" choice.

    Args:
        session_dir: The session directory.
        mode: `deny` or `wait`.

    Raises:
        ValueError: The mode is anything else; approve-all sets the allow marker instead.
    """
    if mode not in ("deny", "wait"):
        raise ValueError(
            f"away.mode is 'deny' or 'wait', got {mode!r} (approve-all reuses session.allow)"
        )
    portable.atomic_write(approvals_dir(session_dir) / AWAY_MODE_FILE, mode)


def away_mode(session_dir: pathlib.Path) -> str:
    """Return the recorded away mode.

    Args:
        session_dir: The session directory.

    Returns:
        `deny`, `wait`, or "" when unset.
    """
    try:
        return (approvals_path(session_dir) / AWAY_MODE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def clear_away_mode(session_dir: pathlib.Path) -> None:
    """Drop the away mode when an interactive run or resume starts: the operator is back.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (approvals_path(session_dir) / AWAY_MODE_FILE).unlink()


def clear_session_grants(session_dir: pathlib.Path) -> None:
    """Drop every per-scope allow marker; they expire with the away mode. `--auto-approve` stays.

    Args:
        session_dir: The session directory.
    """
    approvals = approvals_path(session_dir)
    if approvals.is_dir():
        for marker in approvals.glob(f"{SESSION_ALLOW_FILE}.*"):
            with contextlib.suppress(FileNotFoundError):
                marker.unlink()


def read_answer(
    session_dir: pathlib.Path,
    prompt_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: pathlib.Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Wait for the operator's answer to an approval prompt, from the harness.

    Args:
        session_dir: The session directory.
        prompt_id: The prompt id.
        timeout_s: The most to wait.
        poll_s: The poll interval.
        live_dir: The dir the liveness gate probes for claims; a machine agent state reads
            answers from its per-state dir while the front-end registers on the instance dir.
        dead_grace_s: How long the front-end may stay dead before the wait ends.

    Returns:
        The lower-cased choice; None on the timeout, a stop, or a dead front-end.

    Raises:
        ValueError: The prompt id is not a plain file name.
    """
    target = _answer_path(approvals_dir(session_dir), prompt_id)
    txt = _await_answer(
        target,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
    return None if txt is None else txt.strip().lower()


# The `ask_user` bridge: the approval shape with a free-string answer.


def questions_dir(session_dir: pathlib.Path) -> pathlib.Path:
    """Return the questions dir, created.

    Args:
        session_dir: The session directory.

    Returns:
        The questions dir.
    """
    p = session_dir / QUESTION_DIR_NAME
    paths.mkdir_for_real_user(p)
    return p


def write_question_answers(
    session_dir: pathlib.Path, question_id: str, answers: Sequence[str]
) -> bool:
    """Write the operator's answers to a question prompt, from a front-end.

    Args:
        session_dir: The session directory.
        question_id: The question id.
        answers: One per question of the prompt, in order; stored as a JSON list.

    Returns:
        False when another surface answered first; that answer stands.

    Raises:
        ValueError: The question id is not a plain file name.
    """
    return _publish_answer(
        _answer_path(questions_dir(session_dir), question_id), json.dumps(list(answers))
    )


def question_answers_written(session_dir: pathlib.Path, question_id: str) -> bool:
    """Return whether an answer to the question is on disk; a peek, nothing consumed.

    Args:
        session_dir: The session directory.
        question_id: The question id.

    Returns:
        True when the answer file exists.
    """
    return _answer_path(session_dir / QUESTION_DIR_NAME, question_id).exists()


def read_question_answers(
    session_dir: pathlib.Path,
    question_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: pathlib.Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> tuple[str, ...] | None:
    """Wait for the operator's answers to a question prompt, from the harness.

    Args:
        session_dir: The session directory.
        question_id: The question id.
        timeout_s: The most to wait.
        poll_s: The poll interval.
        live_dir: The dir the liveness gate probes, as in `read_answer`.
        dead_grace_s: How long the front-end may stay dead before the wait ends.

    Returns:
        The answers, one per question; a bare non-JSON file is one answer. None on the
        timeout, a stop, or a dead front-end.

    Raises:
        ValueError: The question id is not a plain file name.
    """
    target = _answer_path(questions_dir(session_dir), question_id)
    raw = _await_answer(
        target,
        live_dir or session_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return (raw,)
    return tuple(str(x) for x in data) if isinstance(data, list) else (str(data),)


# The steer bridge, single-slot: "" continues, "abort" stops, anything else is an instruction.


def write_steer_answer(session_dir: pathlib.Path, answer: str) -> None:
    """Write the steer answer, from a front-end.

    Args:
        session_dir: The session directory.
        answer: "" to continue, `abort` to stop, else the instruction.
    """
    portable.atomic_write(session_dir / STEER_ANSWER_FILE, answer)


def clear_steer_answer(session_dir: pathlib.Path) -> None:
    """Drop a steer answer.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_ANSWER_FILE).unlink()


def take_steer_answer(session_dir: pathlib.Path) -> str | None:
    """Consume a steer answer already on disk, so the tty prompt asks nothing it was told.

    Args:
        session_dir: The session directory.

    Returns:
        The answer (a `resume --steer` seed, an early front-end answer), or None.
    """
    return _consume_answer(session_dir / STEER_ANSWER_FILE)


def steer_answer_written(session_dir: pathlib.Path) -> bool:
    """Return whether a steer answer is on disk; a peek, nothing consumed.

    The pause menu polls it, so a steer from a front-end ends the menu.

    Args:
        session_dir: The session directory.

    Returns:
        True when the answer file exists.
    """
    return (session_dir / STEER_ANSWER_FILE).exists()


def steer_answer_is_abort(session_dir: pathlib.Path) -> bool:
    """Return whether a pending steer answer is a stop; a peek, nothing consumed.

    A streaming model turn bails on it at once instead of at the boundary, which still
    handles the answer if the stream ends first.

    Args:
        session_dir: The session directory.

    Returns:
        True for `abort` exactly, the word every front-end's Stop writes; a typed instruction,
        even "stop", is an instruction.
    """
    try:
        answer = (session_dir / STEER_ANSWER_FILE).read_text(encoding="utf-8").strip().lower()
    except (OSError, ValueError):  # missing, unreadable, or non-UTF-8
        return False
    return answer == "abort"


# A front-end's steer request; decoupled from signals so a watcher process can make one.
STEER_REQUEST_FILE = "steer.request"


def request_steer(session_dir: pathlib.Path, *, now: bool = False) -> bool:
    """Drop the steer marker the session polls at its next boundary.

    Args:
        session_dir: The session directory.
        now: Write the urgency into the marker, so the loop aborts the in-flight model call.

    Returns:
        Whether the marker landed.
    """
    try:
        portable.atomic_write(session_dir / STEER_REQUEST_FILE, "now" if now else "")
    except OSError:
        return False
    return True


def steer_request_pending(session_dir: pathlib.Path) -> bool:
    """Return whether a steer request is pending.

    Args:
        session_dir: The session directory.

    Returns:
        True when the marker exists.
    """
    return (session_dir / STEER_REQUEST_FILE).exists()


def steer_interrupt_pending(session_dir: pathlib.Path) -> bool:
    """Return whether a pending steer carries the `now` urgency.

    Only this aborts an in-flight model call; a plain steer waits for the boundary, since
    aborting wastes the streamed tokens and the step's partial work.

    Args:
        session_dir: The session directory.

    Returns:
        True when the marker reads `now`.
    """
    try:
        return (session_dir / STEER_REQUEST_FILE).read_text(encoding="utf-8").strip() == "now"
    except OSError:
        return False


def submit_steer(session_dir: pathlib.Path, text: str, *, now: bool = False) -> bool:
    """Queue the session's next steer: the answer first, then the request marker.

    The loop finds the answer the moment it notices the request and never waits on a modal.

    Args:
        session_dir: The session directory.
        text: The instruction.
        now: Abort the in-flight model call to take it.

    Returns:
        Whether both files landed; a failed marker removes its stranded answer.
    """
    try:
        write_steer_answer(session_dir, text)
    except OSError:
        return False
    if request_steer(session_dir, now=now):
        return True
    clear_steer_answer(session_dir)
    return False


def clear_steer_request(session_dir: pathlib.Path) -> None:
    """Drop a steer request.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_REQUEST_FILE).unlink()


STOP_REQUEST_FILE = "stop.request"


def request_stop(session_dir: pathlib.Path) -> bool:
    """Drop the "stop after this step" marker the session honors at its next boundary.

    The finished step's tool results and auto-commit land first; the immediate stop is the
    steer `abort`. The session dir is created here: ACP can cancel after assigning the id but
    before the lifecycle creates the layout.

    Args:
        session_dir: The session directory.

    Returns:
        Whether the marker landed; a failed write neither raises into a front-end action nor
        reads as a stop nothing will honor.
    """
    try:
        paths.mkdir_for_real_user(session_dir)
        (session_dir / STOP_REQUEST_FILE).write_text("", encoding="utf-8")
    except OSError:
        return False
    return True


def stop_request_pending(session_dir: pathlib.Path) -> bool:
    """Return whether a stop request is pending.

    Args:
        session_dir: The session directory.

    Returns:
        True when the marker exists.
    """
    return (session_dir / STOP_REQUEST_FILE).exists()


def clear_stop_request(session_dir: pathlib.Path) -> None:
    """Drop a stop request.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STOP_REQUEST_FILE).unlink()


COMPACT_REQUEST_FILE = "compact.request"


def request_compact(session_dir: pathlib.Path, focus: str = "") -> bool:
    """Drop the compaction marker the session honors at its next boundary.

    Published atomically: the run polls every boundary, and a plain write would expose a
    partial focus for it to consume as the real one.

    Args:
        session_dir: The session directory.
        focus: The summary focus from `/compact <focus>`; "" is a plain compact.

    Returns:
        Whether the marker landed; a failed write neither raises into a front-end action nor
        reads as a request nothing will honor.
    """
    try:
        portable.atomic_write(session_dir / COMPACT_REQUEST_FILE, focus)
    except OSError:
        return False
    return True


def read_compact_request(session_dir: pathlib.Path) -> str | None:
    """Return the pending compact request's focus.

    Args:
        session_dir: The session directory.

    Returns:
        The focus, "" for a plain compact; None when no request is pending.
    """
    try:
        return (session_dir / COMPACT_REQUEST_FILE).read_text(encoding="utf-8")
    except OSError:
        return None


def clear_compact_request(session_dir: pathlib.Path) -> None:
    """Drop a compact request.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (session_dir / COMPACT_REQUEST_FILE).unlink()


def read_steer_answer(
    session_dir: pathlib.Path, *, live_dir: pathlib.Path | None = None
) -> str | None:
    """Wait for the steer answer while a front-end is live, from the harness.

    Args:
        session_dir: The session directory.
        live_dir: The dir the liveness gate probes, as in `read_answer`.

    Returns:
        The answer, consumed; None after ten minutes, on a stop, or once the front-end has
        stayed dead past `FRONTEND_DEAD_GRACE_S`.
    """
    return _await_answer(
        session_dir / STEER_ANSWER_FILE,
        live_dir or session_dir,
        timeout_s=600.0,
        poll_s=0.2,
        dead_grace_s=FRONTEND_DEAD_GRACE_S,
    )
