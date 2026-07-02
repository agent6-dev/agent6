# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared run-listing helpers, used by every front-end's hub/watch listing.

The last-activity time and the task snippet were copied into the CLI, the TUI,
and the web hub and drifted; this is the one place they live now.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def run_mtime(run_dir: Path) -> float:
    """Last-activity time of a run: the mtime of its ``logs.jsonl`` (when the run
    last appended an event), falling back to the dir mtime before the log exists.

    NOT the run-directory mtime: a viewer writes ``frontend.pid`` / ``approvals/``
    into the dir on open, bumping the DIRECTORY mtime, so sorting by it floats a
    merely-viewed run to "most recent". Keying off the log keeps "when" stable.
    """
    for candidate in (run_dir / "logs.jsonl", run_dir):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def first_task_line(lines: Iterable[str]) -> str | None:
    """First user-authored line, skipping the ask headers and the multi-line body
    of a ``<file ...>`` / ``<prior-run ...>`` block (a seeded ask prepends those).
    Returns None when nothing stands out."""
    skip_until: str | None = None
    for line in lines:
        s = line.strip()
        if skip_until is not None:
            if s == skip_until:
                skip_until = None
            continue
        if s in {"# agent6 ask", "## Question"}:
            continue
        if s == "## Answer":
            break
        if s.startswith("<file "):
            if "</file>" not in s:
                skip_until = "</file>"
            continue
        if s.startswith("<prior-run "):
            if "</prior-run>" not in s:
                skip_until = "</prior-run>"
            continue
        if s and not s.startswith("<"):
            return s
    return None


def task_snippet(text: str) -> str:
    """One-line summary of a task or ask transcript for a listing: the first
    user-authored line (block bodies skipped), else the stripped text."""
    return first_task_line(text.splitlines()) or text.strip()
