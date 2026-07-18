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

from agent6.runs.ipc import read_worker_pid, worker_is_alive
from agent6.runs.manifest import CompareStamp, ManifestError, read_manifest
from agent6.viewmodel.events import event_epoch

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


def newest_run_dir(buckets: Iterable[Path]) -> Path | None:
    """The most recently active run dir (by logs.jsonl mtime, not dir mtime: a
    viewer writing frontend.pid must not float a run to latest) across the given
    bucket dirs.

    The one run-recency query: callers name the buckets in scope explicitly --
    a lone ``runs/`` dir for run/plan/resume/fork/ask scope, or every
    ``RUN_BUCKETS`` dir for a cross-bucket listing (attach / runs stop). A
    missing bucket dir is skipped; returns None when no bucket holds a run.
    Callers that key off the id take ``.name`` of the result.
    """
    runs: list[Path] = []
    for bucket in buckets:
        if bucket.is_dir():
            runs.extend(p for p in bucket.iterdir() if p.is_dir())
    dirs = sorted(runs, key=run_mtime, reverse=True)
    return dirs[0] if dirs else None


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


def task_snippet(text: str, max_chars: int | None = None) -> str:
    """One-line summary of a task or ask transcript for a listing: the first
    user-authored line (block bodies skipped), else the stripped text; clipped
    to *max_chars* with an ellipsis (the bare slices each surface carried
    clipped mid-word and read as the whole task)."""
    snip = first_task_line(text.splitlines()) or text.strip()
    if max_chars is not None and len(snip) > max_chars:
        snip = snip[: max_chars - 1] + "…"
    return snip


def is_run_husk(run_dir: Path) -> bool:
    """True for a run dir that never really started: neither manifest.json nor
    logs.jsonl (a preflight refused it, or a crash orphaned it). Listings skip
    husks -- "(no logs)" forever is noise, not a run -- and id lookups must not
    let one shadow a real run of the same id in another bucket (runs/ vs asks/).

    Exception: a dir with a LIVE worker.pid is a just-launched run in its
    pre-manifest preflight window, not a husk -- keep it listed (it reads
    "starting"). Only a dir with no live worker is a true husk."""
    if (run_dir / "manifest.json").exists() or (run_dir / "logs.jsonl").exists():
        return False
    return not worker_is_alive(run_dir)


def run_compare(run_dir: Path) -> CompareStamp | None:
    """The ``compare`` stamp a fan-out's auto-compare recorded on an imported
    lane's manifest (rank/of/winner/ranked_by/rationale), or None for a run that
    was never part of a compared fan-out. The event fold doesn't carry it (it is
    post-import manifest state), so every run view reads it from here. Best effort:
    a missing/corrupt manifest reads as None, never an error."""
    try:
        manifest = read_manifest(run_dir)
    except ManifestError:
        return None
    return manifest.compare


def is_winner(run_dir: Path) -> bool:
    """True when a run is the fan-out compare winner (rank 1), for a listing
    marker. False for any run outside a compared fan-out."""
    compare = run_compare(run_dir)
    return compare is not None and compare.winner


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One listing row: everything a hub or `runs list` needs, uncolored."""

    run_id: str
    mode: str  # run | plan | ask | ?
    task: str  # raw task text; callers snippet/truncate for their layout
    # created|starting|running|waiting|stale|passed|answered|planned|finished|
    # stopped|failed
    status: str
    reason: str  # detail: the end reason when "failed", "needs answer" when "waiting", else ""
    cost_usd: float
    mtime: float


def status_word(*, finished: bool, all_passed: bool, end_reason: str) -> tuple[str, str]:
    """Map an end state to ``(word, reason-detail)``.

    The single place that decides how a run's outcome reads -- shared by
    ``run_status_label`` (headers) and ``summarize_run_dir`` (listings) so the
    surfaces can never disagree. "stopped" is the operator's own act (not a
    failure); "planned" and "answered" are the no-verify clean exits (a plan
    pass / an ask, where "passed" would mislead); "passed" means all verify
    gates green, "finished" is a deliberate finish without all-passed, and
    anything else is "failed" with the reason (provider_error, went_quiet, ...).
    """
    if not finished:
        return "running", ""
    if end_reason in ("steer_abort", "interrupted", "interactive_stop"):
        return "stopped", ""  # each is the operator's own act, not a failure
    # A clean exit that verified nothing gets its own word, never "passed": a
    # plan pass ends via finish_planning, an ask by answering, and a gateless
    # run settles with committed work no verify ever gated (deliberate, so
    # "finished"; never green, never "failed").
    no_verify = {
        "finish_planning": ("planned", ""),
        "answered": ("answered", ""),
        "settled": ("finished", "unverified"),
    }
    if end_reason in no_verify:
        return no_verify[end_reason]
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


@dataclass(frozen=True, slots=True)
class LogScan:
    """One tolerant pass over a run's ``logs.jsonl``: the shared scan behind the
    hub listing and ``runs show``. One owner, because the resume rules (bank
    cost legs, un-finish) and the torn-line tolerances drifted when each
    consumer scanned for itself.

    Token counters are the CURRENT leg's; ``cost_usd`` is cumulative across
    resume legs (None = no budget.update ever), matching the typed fold's
    BudgetView so no two surfaces can disagree on what a run cost. ``legs``
    lets a renderer say which scope a figure describes when they differ.
    """

    saw_start: bool = False  # run.start seen (a run without it is still launching)
    mode: str = "?"
    task: str = ""
    finished: bool = False  # a later resume un-finishes
    all_passed: bool = False
    end_reason: str = ""
    cost_usd: float | None = None
    usd_partial: bool = False  # sticky: unpriced spend in any leg -> under-estimate
    legs: int = 1  # 1 + completed resume legs
    input_tokens: int | None = None
    output_tokens: int | None = None
    iteration: int | None = None  # last event carrying an int iteration
    start_ep: float | None = None  # run.start ts (epoch seconds)
    last_ep: float | None = None  # last event with a parseable ts
    last_type: str | None = None  # last event's type


def _tolerant_usd(raw: object, last_good: float) -> float:
    """*raw* as a float when it is a real number or numeric string; else the
    last good figure. A torn/adversarial usd_total degrades like a torn line,
    never aborts the scan (the typed fold makes the same call in parse_event),
    and falsy junk (``""``, ``False``) must KEEP the figure; an ``or 0.0``
    fallback silently reset it."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return float(raw)
    return last_good


def scan_run_log(logs: Path) -> LogScan:  # noqa: PLR0915 (linear fold, like build_parser)
    """Fold ``logs.jsonl`` into a :class:`LogScan`: run.start (mode/task), the
    last run.end (un-finished again by a later resume), the running per-leg
    budget banked across resumes into a cumulative total, and the liveness
    anchors (timestamps, iteration, last event type) ``runs show`` reads.

    errors="replace": a live writer can leave a torn multibyte UTF-8 tail; strict
    decoding would take down the whole listing. The mangled line just fails
    json.loads and is skipped."""
    mode, task = "?", ""
    finished, all_passed, end_reason = False, False, ""
    saw_start = False
    usd_leg = 0.0  # latest leg's running total
    usd_prior_legs = 0.0  # summed totals of completed (resumed-past) legs
    saw_budget = False
    usd_partial = False
    legs = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    iteration: int | None = None
    start_ep: float | None = None
    last_ep: float | None = None
    last_type: str | None = None
    try:
        with logs.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue  # a valid-JSON non-object line (torn/adversarial)
                etype = ev.get("type")
                ep = event_epoch(ev.get("ts"))
                if ep is not None:
                    last_ep = ep
                if isinstance(etype, str):
                    last_type = etype
                if isinstance(ev.get("iteration"), int):
                    iteration = ev["iteration"]
                if etype == "run.start":
                    saw_start = True
                    mode = str(ev.get("mode", mode))
                    task = str(ev.get("user_task", ""))
                    if start_ep is None:
                        start_ep = ep
                elif etype == "run.end":
                    finished = True
                    all_passed = bool(ev.get("all_passed"))
                    end_reason = str(ev.get("reason", ""))
                elif etype == "loop.resume.start":
                    finished = False  # a resume un-finishes the run
                    # Each resume leg starts a FRESH budget (usd_total resets to
                    # 0), so bank the finished leg's total before it does -- the
                    # displayed cost is then the true cumulative spend across all
                    # legs, not just the latest leg's (per-leg budgets stay the
                    # enforcement mechanism; only the shown total changes). The
                    # typed fold applies the same rule (state.BudgetView), so the
                    # hub row and the run view can never disagree. Token counters
                    # reset too: they are documented as the current leg's, and
                    # leftovers would wear the "(latest leg)" label falsely until
                    # the resumed leg's first provider call lands.
                    usd_prior_legs += usd_leg
                    usd_leg = 0.0
                    input_tokens = output_tokens = None
                    legs += 1
                elif etype == "budget.update":
                    saw_budget = True
                    usd_leg = _tolerant_usd(ev.get("usd_total"), usd_leg)
                    usd_partial = bool(ev.get("usd_partial")) or usd_partial
                    ti, to = ev.get("input_total"), ev.get("output_total")
                    input_tokens = ti if isinstance(ti, int) else input_tokens
                    output_tokens = to if isinstance(to, int) else output_tokens
    except OSError:
        pass
    return LogScan(
        saw_start=saw_start,
        mode=mode,
        task=task,
        finished=finished,
        all_passed=all_passed,
        end_reason=end_reason,
        cost_usd=(usd_prior_legs + usd_leg) if saw_budget else None,
        usd_partial=usd_partial,
        legs=legs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        iteration=iteration,
        start_ep=start_ep,
        last_ep=last_ep,
        last_type=last_type,
    )


def summarize_run_dir(run_dir: Path, *, stale_after_s: float = STALE_AFTER_S) -> RunSummary:
    """One listing row from ``logs.jsonl`` + the manifest. Replaced the
    near-duplicate scanners in the TUI hub and the web hub that badged a
    provider_error death as a neutral "done". An "ask" run's task is replaced by
    its transcript, which shows what was asked."""
    logs = run_dir / "logs.jsonl"
    scan = scan_run_log(logs) if logs.is_file() else LogScan()
    mode, task = scan.mode, scan.task
    if not scan.saw_start:
        # Before run.start the log carries no mode/task: either a launching run
        # still in preflight (verify inference is a ~80s LLM call BEFORE the
        # loop's first turn), or a manifest-only `fork --no-run`. Read mode+task
        # from the manifest so the row shows its real work, not a blank
        # "? ? (no logs)". A truly empty husk (no manifest) keeps "(no logs)".
        mode, task = "?", "(no logs)"
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(run_dir)
            mode = manifest.mode
            task = manifest.user_task or "(no logs)"
    word, reason = status_word(
        finished=scan.finished, all_passed=scan.all_passed, end_reason=scan.end_reason
    )
    if word == "running":
        if not scan.saw_start:
            # No run.start yet. A live worker means the run is still launching
            # (egress + verify inference); show "starting", not a bare "running"
            # on an empty row. No live worker (a `fork --no-run`, or a run that
            # died before starting) reads "created": it never started work, so
            # neither a cryptic "?" nor a false "stale".
            word, reason = ("starting" if worker_is_alive(run_dir) else "created"), ""
        elif _running_is_stale(run_dir, stale_after_s):
            word = "stale"
        elif scan.last_type in ("approval.prompt", "question.prompt"):
            # Alive but blocked on the OPERATOR: an unanswered command
            # approval or ask_user question.
            word, reason = "waiting", "needs answer"
    if mode == "ask":
        with contextlib.suppress(OSError):
            task = (run_dir / "transcript.md").read_text(encoding="utf-8", errors="replace")
    return RunSummary(
        run_id=run_dir.name,
        mode=mode,
        task=task,
        status=word,
        reason=reason,
        cost_usd=scan.cost_usd or 0.0,
        mtime=run_mtime(run_dir),
    )
