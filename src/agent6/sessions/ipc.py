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
import math
import os
import pathlib
import tempfile
import time
from collections.abc import Callable, Collection, Sequence
from typing import Literal, cast

from agent6 import paths, portable
from agent6.sessions import process

APPROVAL_DIR_NAME = "approvals"
QUESTION_DIR_NAME = "questions"
# Operator requests, one file each, sorted oldest first; unlike the answer and steer bridges
# a request survives an execution boundary.
QUEUE_DIR_NAME = "queue"
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


def clear_pending_answers(
    session_dir: pathlib.Path, *, started_at: float, keep_questions: Collection[str] = ()
) -> None:
    """Drop the bridge state an execution inherits, at its start; best-effort.

    The `*.answer` files, the steer answer and marker (a phantom prompt no front-end
    answers), and the stop and compact markers (an instant re-stop) go; only files older
    than the start, less `TIMESTAMP_SLACK_S`: one written since belongs to the execution
    that is starting. The answer to a question still pending in the journal stays: the new
    execution delivers it. Front-end claims need no sweep: `frontend_is_live` prunes dead
    ones, and a live watcher's must survive so its modals stay wired up.

    Args:
        session_dir: The session directory.
        started_at: This execution's start as `time.time()`; ACP passes its turn's start,
            a machine's crash recovery passes now.
        keep_questions: The ids of the questions still pending.
    """
    answer_dirs = (session_dir / APPROVAL_DIR_NAME, session_dir / QUESTION_DIR_NAME)
    markers = (STEER_ANSWER_FILE, STEER_REQUEST_FILE, STOP_REQUEST_FILE, COMPACT_REQUEST_FILE)
    answers = (
        f
        for d in answer_dirs
        for f in d.glob("*.answer")
        if not (d.name == QUESTION_DIR_NAME and f.stem in keep_questions)
    )
    for path in (*answers, *(session_dir / name for name in markers)):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < started_at - TIMESTAMP_SLACK_S:
                path.unlink()


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
        if process.frontend_is_live(live):
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
        if not process.frontend_is_live(session_dir):
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
    consumer agrees. Only `ask` moves: a configured `allow` or `deny` is the standing policy.
    `deny` withdraws the tools, the same wiring for `--no-commands`, a session deny and an
    away mode of `deny`.

    Args:
        configured: `[sandbox].run_commands`.
        session_dir: The session directory.

    Returns:
        `deny`, `ask` or `allow`.
    """
    if configured != "ask":
        return configured
    if session_allow_set(session_dir, COMMAND_SCOPE):
        return "allow"
    if session_deny_set(session_dir, COMMAND_SCOPE) or away_mode(session_dir) == "deny":
        return "deny"
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
    timeout_s: float = math.inf,
    poll_s: float = 0.2,
    live_dir: pathlib.Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Wait for the operator's answer to an approval prompt, from the harness.

    Args:
        session_dir: The session directory.
        prompt_id: The prompt id.
        timeout_s: The most to wait; by default as long as the front-end lives.
        poll_s: The poll interval.
        live_dir: The dir the liveness gate probes for claims; a machine agent state reads
            answers from its per-state dir while the front-end registers on the instance dir.
        dead_grace_s: How long the front-end may stay dead before the wait ends.

    Returns:
        The lower-cased choice; None on a stop, a dead front-end, or a timeout given.

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


def saved_question_answers(session_dir: pathlib.Path) -> tuple[str, ...]:
    """Return the ids of the questions whose answers wait on disk for the run to read.

    Args:
        session_dir: The session directory.

    Returns:
        The ids, sorted; a file stays until the run consumes it at a turn or at its resume.
    """
    return tuple(sorted(p.stem for p in (session_dir / QUESTION_DIR_NAME).glob("*.answer")))


def peek_question_answers(session_dir: pathlib.Path, question_id: str) -> tuple[str, ...] | None:
    """Return the answers on disk for a question, consuming nothing; None when absent.

    Args:
        session_dir: The session directory.
        question_id: The question id.

    Returns:
        The answers, one per question; a bare non-JSON file is one answer.
    """
    try:
        raw = _answer_path(session_dir / QUESTION_DIR_NAME, question_id).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return (raw,)
    return tuple(str(x) for x in data) if isinstance(data, list) else (str(data),)


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
    timeout_s: float = math.inf,
    poll_s: float = 0.2,
    live_dir: pathlib.Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> tuple[str, ...] | None:
    """Wait for the operator's answers to a question prompt, from the harness.

    Args:
        session_dir: The session directory.
        question_id: The question id.
        timeout_s: The most to wait; by default as long as the front-end lives.
        poll_s: The poll interval.
        live_dir: The dir the liveness gate probes, as in `read_answer`.
        dead_grace_s: How long the front-end may stay dead before the wait ends.

    Returns:
        The answers, one per question; a bare non-JSON file is one answer. None on a
        stop, a dead front-end, or a timeout given.

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


def submit_steer(
    session_dir: pathlib.Path, text: str, *, now: bool = False, only_if_empty: bool = False
) -> bool:
    """Queue the session's next steer: the answer first, then the request marker.

    The loop finds the answer the moment it notices the request and never waits on a modal.

    Args:
        session_dir: The session directory.
        text: The instruction.
        now: Abort the in-flight model call to take it.
        only_if_empty: Refuse an occupied slot instead of replacing its instruction.

    Returns:
        Whether both files landed; a failed marker removes its stranded answer.
    """
    try:
        if only_if_empty:
            if steer_request_pending(session_dir) or not _publish_answer(
                session_dir / STEER_ANSWER_FILE, text
            ):
                return False
        else:
            write_steer_answer(session_dir, text)
    except OSError:
        return False
    if request_steer(session_dir, now=now):
        return True
    clear_steer_answer(session_dir)
    return False


REVIEW_ACTIONS_DIR = "review-actions"


def _review_path(session_dir: pathlib.Path, request_id: str, suffix: str) -> pathlib.Path:
    return _contained(
        session_dir / REVIEW_ACTIONS_DIR,
        f"{request_id}.{suffix}",
        untrusted=request_id,
        what="review request id",
    )


def queue_review_action(session_dir: pathlib.Path, request_id: str, payload: str) -> None:
    """Publish a whole action envelope for the session worker, keeping the first request.

    Args:
        session_dir: The session directory.
        request_id: The action's retry identity.
        payload: The JSON action envelope.

    Raises:
        ValueError: The request ID escapes the bridge or names a different pending action.
    """
    target = _review_path(session_dir, request_id, "request")
    paths.mkdir_for_real_user(target.parent)
    if not _publish_answer(target, payload) and target.read_text(encoding="utf-8") != payload:
        raise ValueError("request_id already has a different action envelope")


def review_requests(session_dir: pathlib.Path) -> list[tuple[str, str]]:
    """Read pending action envelopes without deleting evidence before the worker replies.

    Args:
        session_dir: The session directory.

    Returns:
        Request IDs and their whole JSON envelopes.
    """
    directory = session_dir / REVIEW_ACTIONS_DIR
    result: list[tuple[str, str]] = []
    for path in sorted(directory.glob("*.request")):
        target = _review_path(session_dir, path.stem, "request")
        result.append((path.stem, target.read_text(encoding="utf-8")))
    return result


def reply_review_action(session_dir: pathlib.Path, request_id: str, payload: str) -> None:
    """Publish the worker's reply before removing its request."""
    _publish_answer(_review_path(session_dir, request_id, "reply"), payload)
    _review_path(session_dir, request_id, "request").unlink(missing_ok=True)


def read_review_reply(session_dir: pathlib.Path, request_id: str) -> str | None:
    """Read an action reply without consuming it, so retries retain the same result.

    Args:
        session_dir: The session directory.
        request_id: The action's retry identity.

    Returns:
        The JSON reply, or None while no reply is recorded.
    """
    try:
        return _review_path(session_dir, request_id, "reply").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def clear_steer_request(session_dir: pathlib.Path) -> None:
    """Drop a steer request.

    Args:
        session_dir: The session directory.
    """
    with contextlib.suppress(FileNotFoundError):
        (session_dir / STEER_REQUEST_FILE).unlink()


STOP_REQUEST_FILE = "stop.request"


def request_stop(session_dir: pathlib.Path, *, now: bool = False) -> bool:
    """Drop the "stop after this step" marker the session honors at its next boundary.

    The finished step's tool results and auto-commit land first: a command the step is running
    finishes. The session dir is created here: ACP can cancel after assigning the id but
    before the lifecycle creates the layout.

    Args:
        session_dir: The session directory.
        now: Write the urgency into the marker, so a running foreground command is ended at once
            and the step ends there. The model call in flight is cut by the steer `abort`.

    Returns:
        Whether the marker landed; a failed write neither raises into a front-end action nor
        reads as a stop nothing will honor.
    """
    try:
        paths.mkdir_for_real_user(session_dir)
        (session_dir / STOP_REQUEST_FILE).write_text("now" if now else "", encoding="utf-8")
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


def stop_request_is_now(session_dir: pathlib.Path) -> bool:
    """Return whether a pending stop carries the `now` urgency.

    Args:
        session_dir: The session directory.

    Returns:
        True when the marker reads `now`.
    """
    try:
        return (session_dir / STOP_REQUEST_FILE).read_text(encoding="utf-8").strip() == "now"
    except OSError:
        return False


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
