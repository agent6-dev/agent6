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


def approvals_dir(run_dir: Path) -> Path:
    p = run_dir / APPROVAL_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def write_answer(run_dir: Path, prompt_id: str, *, approved: bool) -> None:
    """Called by a front-end (TUI or web)."""
    d = approvals_dir(run_dir)
    (d / f"{prompt_id}.answer").write_text("yes" if approved else "no", encoding="utf-8")


def read_answer(
    run_dir: Path,
    prompt_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
) -> bool | None:
    """Called by the workflow. Returns True/False or None on timeout."""
    target = approvals_dir(run_dir) / f"{prompt_id}.answer"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if target.exists():
            txt = target.read_text(encoding="utf-8").strip().lower()
            with contextlib.suppress(FileNotFoundError):
                target.unlink()  # consume: never re-read on a later prompt/resume
            return txt in {"yes", "y", "true", "1"}
        if not frontend_is_live(run_dir):
            return None
        time.sleep(poll_s)
    return None


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
    (questions_dir(run_dir) / f"{question_id}.answer").write_text(answer, encoding="utf-8")


def read_question_answer(
    run_dir: Path,
    question_id: str,
    *,
    timeout_s: float = 600.0,
    poll_s: float = 0.2,
) -> str | None:
    """Called by the workflow. Returns the answer string or None if the TUI
    died / timed out before answering."""
    target = questions_dir(run_dir) / f"{question_id}.answer"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if target.exists():
            txt = target.read_text(encoding="utf-8")
            with contextlib.suppress(FileNotFoundError):
                target.unlink()  # consume: never re-read on a later prompt/resume
            return txt
        if not frontend_is_live(run_dir):
            return None
        time.sleep(poll_s)
    return None


# --- mid-run steering bridge (Ctrl-C while the TUI owns the terminal) --------
# Single-slot: only one steer prompt is ever outstanding (the SIGINT handler
# sets a flag the loop drains at its next boundary). The run process triggers a
# steer by emitting `run.steer_requested`; the TUI shows a modal and writes the
# answer here; the run process reads it. The answer is a free string:
# "" = continue, "abort" = stop, anything else = a steering instruction.


def write_steer_answer(run_dir: Path, answer: str) -> None:
    """Called by a front-end when the user answers the steer prompt."""
    (run_dir / STEER_ANSWER_FILE).write_text(answer, encoding="utf-8")


def clear_steer_answer(run_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (run_dir / STEER_ANSWER_FILE).unlink()


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
) -> str | None:
    """Called by the workflow when the TUI is live. Returns the answer string
    (consuming the file) or None if the TUI died / timed out before answering."""
    target = run_dir / STEER_ANSWER_FILE
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if target.exists():
            txt = target.read_text(encoding="utf-8")
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            return txt
        if not frontend_is_live(run_dir):
            return None
        time.sleep(poll_s)
    return None
