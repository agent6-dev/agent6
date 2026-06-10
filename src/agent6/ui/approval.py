# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""File-based approval bridge between the workflow process and the TUI.

The workflow process and the TUI run as separate OS processes (the TUI
just tails JSONL). When an approval is needed:

1. The workflow process writes an `approval.prompt` event to logs.jsonl
   and then polls `<run_dir>/approvals/<id>.answer` for a result.
2. If `<run_dir>/tui.pid` exists and points at a live process, the
   workflow process waits for the TUI to write the answer file. Otherwise
   it falls back to a plain stdin prompt.
3. The TUI (when present) presents a modal, then writes
   `<run_dir>/approvals/<id>.answer` containing exactly `yes` or `no`.

We use the filesystem rather than a socket because:
- the JSONL log is already the cross-process contract,
- the TUI may crash without taking the workflow down with it,
- it's trivial to mirror in any other UI (including the VS Code extension).
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

APPROVAL_DIR_NAME = "approvals"
QUESTION_DIR_NAME = "questions"
TUI_PID_FILE = "tui.pid"
STEER_ANSWER_FILE = "steer.answer"


def approvals_dir(run_dir: Path) -> Path:
    p = run_dir / APPROVAL_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_pending_answers(run_dir: Path) -> None:
    """Drop stale bridge state at run/resume START: leftover `*.answer` files
    from a prior session (the id counters reset, so an old answer would be read
    instead of prompting) and a stale `tui.pid` from a hard-killed TUI (which
    would otherwise make the answer-poll block until timeout). Best-effort."""
    for sub in (APPROVAL_DIR_NAME, QUESTION_DIR_NAME):
        d = run_dir / sub
        if d.is_dir():
            for f in d.glob("*.answer"):
                with contextlib.suppress(OSError):
                    f.unlink()
    clear_steer_answer(run_dir)
    clear_tui_pid(run_dir)


def write_tui_pid(run_dir: Path, pid: int) -> None:
    (run_dir / TUI_PID_FILE).write_text(str(pid), encoding="utf-8")


def clear_tui_pid(run_dir: Path) -> None:
    p = run_dir / TUI_PID_FILE
    with contextlib.suppress(FileNotFoundError):
        p.unlink()


def tui_is_live(run_dir: Path) -> bool:
    p = run_dir / TUI_PID_FILE
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
    """Called by the TUI."""
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
        if not tui_is_live(run_dir):
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
    """Called by the TUI when the user answers the question modal."""
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
        if not tui_is_live(run_dir):
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
    """Called by the TUI when the user answers the steer modal."""
    (run_dir / STEER_ANSWER_FILE).write_text(answer, encoding="utf-8")


def clear_steer_answer(run_dir: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        (run_dir / STEER_ANSWER_FILE).unlink()


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
        if not tui_is_live(run_dir):
            return None
        time.sleep(poll_s)
    return None
