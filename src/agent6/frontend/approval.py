# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""File-based approval bridge between the workflow process and a front-end.

The workflow process and a front-end (the Textual TUI or the `agent6 web`
server) run as separate OS processes; the front-end just tails JSONL and
answers prompts by writing files. When an approval is needed:

1. The workflow process writes an `approval.prompt` event to logs.jsonl
   and then polls `<run_dir>/approvals/<id>.answer` for a result.
2. If `<run_dir>/frontend.pid` exists and points at a live process, the
   workflow process waits for the front-end to write the answer file.
   Otherwise it falls back to a plain stdin prompt.
3. The front-end (when present) presents a modal / control, then writes
   `<run_dir>/approvals/<id>.answer` containing exactly `yes` or `no`.

We use the filesystem rather than a socket because:
- the JSONL log is already the cross-process contract,
- the front-end may crash without taking the workflow down with it,
- any front-end can mirror it (the TUI, the web server, a VS Code extension).
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

APPROVAL_DIR_NAME = "approvals"
QUESTION_DIR_NAME = "questions"
FRONTEND_PID_FILE = "frontend.pid"
WORKER_PID_FILE = "worker.pid"  # the run's worker process, for `agent6 runs show` liveness
STEER_ANSWER_FILE = "steer.answer"

# How long the answer polls keep waiting after the front-end liveness gate goes
# dark before falling back headless (deny / ""). A transient drop (a phone
# locking its browser, a page reload, a web server restart) re-registers within
# seconds; without the grace, one 0.2s poll landing in that gap silently denied
# a pending approval. 30s outlasts a reload while a truly-gone front-end still
# fails over well before the answer timeout.
FRONTEND_DEAD_GRACE_S = 30.0


def approvals_dir(run_dir: Path) -> Path:
    p = run_dir / APPROVAL_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _answer_path(directory: Path, answer_id: str) -> Path:
    """Resolve ``<directory>/<answer_id>.answer``, refusing an id that escapes the
    directory (a path separator, ``..``, or an absolute path). A front-end always
    answers an id from a prompt it rendered, but that id crosses a trust boundary
    in the web server, so containment stays a hard check on the write primitive."""
    target = directory / f"{answer_id}.answer"
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"unsafe answer id: {answer_id!r}")
    return target


def clear_pending_answers(run_dir: Path) -> None:
    """Drop stale bridge state at run/resume START: leftover `*.answer` files
    from a prior session (the id counters reset, so an old answer would be read
    instead of prompting), a leftover `steer.request` marker (which would
    otherwise trigger a phantom steer prompt that no live front-end answers), and a
    stale `frontend.pid` from a hard-killed front-end (which would otherwise make the
    answer-poll block until timeout). Best-effort.

    The `frontend.pid` is only dropped when NO live front-end owns it: a concurrently-live
    `agent6 watch` watcher must keep bridging the resumed run's approval/question
    modals, so we must not unlink a pid that still points at a running process."""
    for sub in (APPROVAL_DIR_NAME, QUESTION_DIR_NAME):
        d = run_dir / sub
        if d.is_dir():
            for f in d.glob("*.answer"):
                with contextlib.suppress(OSError):
                    f.unlink()
    clear_steer_answer(run_dir)
    clear_steer_request(run_dir)
    if not frontend_is_live(run_dir):  # only drop a STALE pid (hard-killed front-end)
        clear_frontend_pid(run_dir)


def write_frontend_pid(run_dir: Path, pid: int) -> None:
    (run_dir / FRONTEND_PID_FILE).write_text(str(pid), encoding="utf-8")


def clear_frontend_pid(run_dir: Path) -> None:
    p = run_dir / FRONTEND_PID_FILE
    with contextlib.suppress(FileNotFoundError):
        p.unlink()


def _pid_alive(pid: int) -> bool:
    """True if a process with *pid* exists (signal 0 probes without killing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but is owned by another user
    except OSError:
        return False
    return True


def write_worker_pid(run_dir: Path, pid: int) -> None:
    """Record the run's worker pid so `agent6 runs show` can probe liveness even
    while the worker is blocked in a long provider call (no events emitted)."""
    (run_dir / WORKER_PID_FILE).write_text(str(pid), encoding="utf-8")


def clear_worker_pid(run_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (run_dir / WORKER_PID_FILE).unlink()


def read_worker_pid(run_dir: Path) -> int | None:
    try:
        return int((run_dir / WORKER_PID_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def worker_is_alive(run_dir: Path) -> bool:
    """True iff the run dir has a worker.pid pointing at a live process."""
    pid = read_worker_pid(run_dir)
    return pid is not None and _pid_alive(pid)


def frontend_is_live(run_dir: Path) -> bool:
    p = run_dir / FRONTEND_PID_FILE
    if not p.exists():
        return False
    try:
        pid = int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _write_answer_atomic(target: Path, text: str) -> None:
    """Write an answer file via temp + fsync + rename (the journal.poke pattern).

    The reader polls on existence every 0.2s; a plain write_text exposes an
    empty/partial file it would consume as deny / ""."""
    tmp = target.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    with tmp.open("r", encoding="utf-8") as fh:
        os.fsync(fh.fileno())
    tmp.rename(target)


def _await_answer(
    target: Path, live: Path, *, timeout_s: float, poll_s: float, dead_grace_s: float
) -> str | None:
    """Poll for *target*, consume it, and return its text.

    Returns None when the front-end registered on *live* stays dead for
    *dead_grace_s* consecutive seconds (see FRONTEND_DEAD_GRACE_S) or when
    *timeout_s* elapses. A file that vanishes between polls is not-yet-answered,
    never an error."""
    deadline = time.monotonic() + timeout_s
    dead_since: float | None = None
    while time.monotonic() < deadline:
        try:
            txt = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            txt = None
        if txt is not None:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()  # consume: never re-read on a later prompt/resume
            return txt
        if frontend_is_live(live):
            dead_since = None
        else:
            now = time.monotonic()
            if dead_since is None:
                dead_since = now
            if now - dead_since >= dead_grace_s:
                return None
        time.sleep(poll_s)
    return None


def write_answer(run_dir: Path, prompt_id: str, *, approved: bool) -> None:
    """Called by a front-end (TUI or web)."""
    target = _answer_path(approvals_dir(run_dir), prompt_id)
    _write_answer_atomic(target, "yes" if approved else "no")


def read_answer(
    run_dir: Path,
    prompt_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> bool | None:
    """Called by the workflow. Returns True/False, or None on timeout or once the
    front-end has stayed dead past ``dead_grace_s`` (a shorter drop keeps waiting).

    ``live_dir`` overrides which dir the liveness gate probes for ``frontend.pid``
    (defaults to ``run_dir``). A machine agent state reads answers from its
    per-state dir but the front-end registers on the instance dir, so it passes
    the instance dir here."""
    target = approvals_dir(run_dir) / f"{prompt_id}.answer"
    txt = _await_answer(
        target, live_dir or run_dir, timeout_s=timeout_s, poll_s=poll_s, dead_grace_s=dead_grace_s
    )
    if txt is None:
        return None
    return txt.strip().lower() in {"yes", "y", "true", "1"}


# --- agent->user question bridge (the `ask_user` tool) -----------------------
# Same shape as approvals, but the answer is a free string (a selected option or
# typed text). The workflow emits `question.prompt`, polls for the answer file;
# the TUI shows a modal and writes it. Falls back to stdin (then a default) when
# no TUI is live, so headless runs never hang.


def questions_dir(run_dir: Path) -> Path:
    p = run_dir / QUESTION_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_question_answer(run_dir: Path, question_id: str, answer: str) -> None:
    """Called by a front-end when the user answers the question."""
    _write_answer_atomic(_answer_path(questions_dir(run_dir), question_id), answer)


def read_question_answer(
    run_dir: Path,
    question_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Called by the workflow. Returns the answer string, or None on timeout or
    once the front-end has stayed dead past ``dead_grace_s``. ``live_dir``
    overrides the liveness-gate dir (see :func:`read_answer`)."""
    target = questions_dir(run_dir) / f"{question_id}.answer"
    return _await_answer(
        target, live_dir or run_dir, timeout_s=timeout_s, poll_s=poll_s, dead_grace_s=dead_grace_s
    )


# --- mid-run steering bridge (Ctrl-C while the TUI owns the terminal) --------
# Single-slot: only one steer prompt is ever outstanding (the SIGINT handler
# sets a flag the loop drains at its next boundary). The run process triggers a
# steer by emitting `run.steer_requested`; the TUI shows a modal and writes the
# answer here; the run process reads it. The answer is a free string:
# "" = continue, "abort" = stop, anything else = a steering instruction.


def write_steer_answer(run_dir: Path, answer: str) -> None:
    """Called by a front-end when the user answers the steer prompt."""
    _write_answer_atomic(run_dir / STEER_ANSWER_FILE, answer)


def clear_steer_answer(run_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (run_dir / STEER_ANSWER_FILE).unlink()


def steer_answer_is_abort(run_dir: Path) -> bool:
    """Non-blocking peek: True if a pending steer answer is a stop. Lets a long
    streaming model turn bail immediately instead of only at the between-step
    boundary. Does NOT consume the answer -- the boundary still handles it if the
    stream ends first."""
    try:
        answer = (run_dir / STEER_ANSWER_FILE).read_text(encoding="utf-8").strip().lower()
    except (OSError, ValueError):  # missing/unreadable, or non-UTF-8: not an abort
        return False
    # Same stop-words the between-step boundary honors (_normalize_steer_choice).
    return answer in ("abort", "stop", "q", "quit")


# A steer can also be INITIATED from the TUI (the `s` key) without Ctrl-C: the
# dashboard drops this marker, the run notices it at its next safe boundary (same
# as the SIGINT flag), prompts via the modal, and clears it. Decoupled from
# signals so a watcher process can request a steer the run picks up.
STEER_REQUEST_FILE = "steer.request"


def request_steer(run_dir: Path) -> None:
    """TUI-initiated steer: drop a marker the run polls at its next boundary."""
    with contextlib.suppress(OSError):
        (run_dir / STEER_REQUEST_FILE).write_text("", encoding="utf-8")


def steer_request_pending(run_dir: Path) -> bool:
    return (run_dir / STEER_REQUEST_FILE).exists()


def clear_steer_request(run_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (run_dir / STEER_REQUEST_FILE).unlink()


def read_steer_answer(
    run_dir: Path,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
    live_dir: Path | None = None,
    dead_grace_s: float = FRONTEND_DEAD_GRACE_S,
) -> str | None:
    """Called by the workflow when the TUI is live. Returns the answer string
    (consuming the file), or None on timeout or once the front-end has stayed
    dead past ``dead_grace_s``. ``live_dir`` overrides the liveness-gate dir
    (see :func:`read_answer`)."""
    return _await_answer(
        run_dir / STEER_ANSWER_FILE,
        live_dir or run_dir,
        timeout_s=timeout_s,
        poll_s=poll_s,
        dead_grace_s=dead_grace_s,
    )
