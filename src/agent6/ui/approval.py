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
TUI_PID_FILE = "tui.pid"


def approvals_dir(run_dir: Path) -> Path:
    p = run_dir / APPROVAL_DIR_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


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
            return txt in {"yes", "y", "true", "1"}
        if not tui_is_live(run_dir):
            return None
        time.sleep(poll_s)
    return None
