# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared run-listing helpers, used by every front-end's hub/watch listing.

The last-activity time and the task snippet were copied into the CLI, the TUI,
and the web hub and drifted; this is the one place they live now.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agent6.ui.bridge.approval import read_worker_pid, worker_is_alive

STALE_AFTER_S = 600.0


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


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One listing row: everything a hub or `runs list` needs, uncolored."""

    run_id: str
    mode: str  # run | plan | ask | ?
    task: str  # raw task text; callers snippet/truncate for their layout
    status: str  # running | stale | passed | finished | stopped | failed | ?
    reason: str  # end reason detail when status is "failed", else ""
    cost_usd: float
    mtime: float


def status_word(*, finished: bool, all_passed: bool, end_reason: str) -> tuple[str, str]:
    """Map an end state to ``(word, reason-detail)``.

    The single place that decides how a run's outcome reads -- shared by
    ``run_status_label`` (headers) and ``summarize_run_dir`` (listings) so the
    surfaces can never disagree. "stopped" is the operator's own act (not a
    failure), "passed" means all verify gates green, "finished" is a deliberate
    finish without all-passed, and anything else is "failed" with the reason
    (provider_error, went_quiet, ...).
    """
    if not finished:
        return "running", ""
    if end_reason in ("steer_abort", "interrupted"):
        return "stopped", ""  # both are the operator's own act, not a failure
    if all_passed:
        return "passed", ""
    if end_reason and end_reason != "finish_run":
        return "failed", end_reason
    return "finished", ""


def _running_is_stale(run_dir: Path, stale_after_s: float) -> bool:
    """Probe the worker pid when the run recorded one: a killed run reads
    "stale" at once (not after the silence window), and a live worker blocked
    in a long provider call stays "running" however quiet the log. Runs
    without a pid record keep the log-silence fallback."""
    if read_worker_pid(run_dir) is not None:
        return not worker_is_alive(run_dir)
    return (time.time() - run_mtime(run_dir)) > stale_after_s


def summarize_run_dir(run_dir: Path, *, stale_after_s: float = STALE_AFTER_S) -> RunSummary:
    """Single pass over ``logs.jsonl``: run.start (mode/task), the last run.end
    (status, un-finished again by a later resume), and the last budget.update
    (cost). An "ask" run's task is replaced by its transcript, which shows what
    was asked. Replaced the near-duplicate scanners in the TUI hub and the web
    hub that badged a provider_error death as a neutral "done"."""
    logs = run_dir / "logs.jsonl"
    mode, task = "?", ""
    finished, all_passed, end_reason = False, False, ""
    cost = 0.0
    if not logs.is_file():
        return RunSummary(
            run_id=run_dir.name,
            mode=mode,
            task="(no logs)",
            status="?",
            reason="",
            cost_usd=0.0,
            mtime=run_mtime(run_dir),
        )
    try:
        # errors="replace": a live writer can leave a torn multibyte UTF-8 tail;
        # strict decoding would take down the whole listing. The mangled line
        # just fails json.loads and is skipped.
        with logs.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                etype = ev.get("type")
                if etype == "run.start":
                    mode = str(ev.get("mode", mode))
                    task = str(ev.get("user_task", ""))
                elif etype == "run.end":
                    finished = True
                    all_passed = bool(ev.get("all_passed"))
                    end_reason = str(ev.get("reason", ""))
                elif etype == "loop.resume.start":
                    finished = False  # a resume un-finishes the run
                elif etype == "budget.update":
                    cost = float(ev.get("usd_total", cost) or 0.0)
    except OSError:
        pass
    word, reason = status_word(finished=finished, all_passed=all_passed, end_reason=end_reason)
    if word == "running" and _running_is_stale(run_dir, stale_after_s):
        word = "stale"
    if mode == "ask":
        with contextlib.suppress(OSError):
            task = (run_dir / "transcript.md").read_text(encoding="utf-8", errors="replace")
    return RunSummary(
        run_id=run_dir.name,
        mode=mode,
        task=task,
        status=word,
        reason=reason,
        cost_usd=cost,
        mtime=run_mtime(run_dir),
    )
