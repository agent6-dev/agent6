# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Watch started lanes until they land, and say what they wait on.

The fan-out's await loop, the single-lane await and drain behind `run_lane_to_completion`,
the live symlink a lane gets under the origin's runs dir, and the pending-prompt probe the
status line uses. `app/parallel.py` drives the lanes.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.harness.subrun import LaneResult, LaneSpec
from agent6.paths import mkdir_for_real_user
from agent6.sessions.ipc import request_stop, worker_is_alive
from agent6.sessions.layout import LOGS_NAME, bucket_dir
from agent6.viewmodel import summarize_session_dir
from agent6.viewmodel.format import format_usd

# How often the await loop polls lane liveness.
POLL_INTERVAL_S = 2.0

# How long a stop waits for a lane to finish its in-flight step before giving up on it.
STOP_GRACE_S = 30.0


def lane_terminal(session_dir: Path, status: str, worker_is_alive: Callable[[Path], bool]) -> bool:
    """Return whether an awaited lane is terminal: the fold left "running" and the worker is gone.

    `session.end` lands before the teardown clears `worker.pid`, so status alone races it. A
    lane that dies without a `session.end` cannot hang the gate: the fold reads it "stale".
    """
    return status != "running" and not worker_is_alive(session_dir)


def await_lane(
    res: LaneResult,
    *,
    poll_interval_s: float = POLL_INTERVAL_S,
    should_stop: Callable[[], bool] | None = None,
) -> bool:
    """Block until the lane is terminal, or until the stop check goes true first.

    Args:
        res: The started lane, awaited on its real run dir.
        poll_interval_s: How often to poll.
        should_stop: The coordinator's abort channel, read between polls.

    Returns:
        True when the lane ended, False when the stop check ended the wait.
    """
    while True:
        summary = summarize_session_dir(res.session_dir)
        if lane_terminal(res.session_dir, summary.status, worker_is_alive):
            return True
        if should_stop is not None and should_stop():
            return False
        time.sleep(poll_interval_s)


def drain_lane(
    res: LaneResult, *, poll_interval_s: float, hard_stop: threading.Event | None
) -> bool:
    """Wait a bounded grace after a stop for the lane to land, so its work still imports.

    Args:
        res: The stopped lane.
        poll_interval_s: How often to poll.
        hard_stop: A process teardown, which skips the wait.

    Returns:
        True when the lane landed in time, False to leave it running un-imported.
    """
    deadline = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < deadline:
        if hard_stop is not None and hard_stop.is_set():
            return False
        summary = summarize_session_dir(res.session_dir)
        if lane_terminal(res.session_dir, summary.status, worker_is_alive):
            return True
        if hard_stop is not None:
            if hard_stop.wait(poll_interval_s):
                return False
        else:
            time.sleep(poll_interval_s)
    return False


def lane_link(origin_state: Path, session_id: str) -> Path:
    """Return where a lane's live symlink sits under the origin's runs dir."""
    return bucket_dir(origin_state, "runs") / session_id


def symlink_lane(origin_state: Path, res: LaneResult) -> None:
    """Symlink a lane's clone-side run dir into the origin's `runs/` so the hub shows it live.

    The import replaces the link with the real dir.
    """
    link = lane_link(origin_state, res.spec.session_id)
    mkdir_for_real_user(link.parent)
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    with contextlib.suppress(OSError):
        link.symlink_to(res.session_dir)


def await_lanes(
    started: list[LaneResult],
    *,
    already_interrupted: bool = False,
    should_stop: Callable[[], bool] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> bool:
    """Poll every started lane's real run dir until it is terminal, printing status changes.

    An interrupt requests a clean stop on each running lane and waits a bounded grace for
    their in-flight steps, so the caller imports what landed.

    Args:
        started: The started lanes.
        already_interrupted: A Ctrl+C landed before the await began; go straight to the stop.
        should_stop: The coordinator's own stop request, read between polls.
        reporter: Where the status lines go.

    Returns:
        True when interrupted, False when every lane ended.
    """
    pending = {r.spec.session_id: r for r in started}
    seen: dict[str, tuple[str, str, float]] = {}

    def poll_once() -> None:
        for rid, res in list(pending.items()):
            summary = summarize_session_dir(res.session_dir)
            # A "waiting" lane is blocked on a prompt no detached lane can answer.
            waiting = pending_prompt(res.session_dir) if summary.status == "waiting" else ""
            key = (summary.status, waiting, round(summary.cost_usd, 4))
            if seen.get(rid) != key:
                seen[rid] = key
                print_lane_status(
                    res.spec, summary.status, summary.cost_usd, waiting=waiting, reporter=reporter
                )
            if lane_terminal(res.session_dir, summary.status, worker_is_alive):
                pending.pop(rid)

    def stop_and_drain() -> None:
        reporter.err("\n[agent6] interrupted; stopping lanes...")
        for res in pending.values():
            if not request_stop(res.session_dir):
                reporter.err(f"[agent6] could not write the stop request for {res.spec.session_id}")
        deadline = time.monotonic() + STOP_GRACE_S
        with contextlib.suppress(KeyboardInterrupt):
            while pending and time.monotonic() < deadline:
                poll_once()
                if pending:
                    time.sleep(POLL_INTERVAL_S)

    if already_interrupted:
        stop_and_drain()
        return True
    try:
        while pending:
            poll_once()
            if pending and should_stop is not None and should_stop():
                stop_and_drain()
                return True
            if pending:
                time.sleep(POLL_INTERVAL_S)
        return False
    except KeyboardInterrupt:
        stop_and_drain()
        return True


# The two prompt/answer event pairs a lane can block on.
_PROMPT_KIND = {"approval.prompt": "approval", "question.prompt": "a question"}
_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


def pending_prompt(session_dir: Path) -> str:
    """Return "approval" or "a question" when the lane is blocked on an unanswered prompt.

    The last prompt or answer event in the log decides it: the worker blocks on its answer
    while its away mode is `wait`, and no request marker exists for prompts.

    Args:
        session_dir: The lane's run dir.

    Returns:
        The prompt kind, or "" when nothing is pending.
    """
    try:
        lines = (session_dir / LOGS_NAME).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        if "approval." not in raw and "question." not in raw:
            continue  # a fast reject before json.loads
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        etype = ev.get("type") if isinstance(ev, dict) else None
        if etype in _ANSWER_EVENTS:
            return ""
        if etype in _PROMPT_KIND:
            return _PROMPT_KIND[etype]
    return ""


def print_lane_status(
    spec: LaneSpec,
    status: str,
    cost: float,
    *,
    waiting: str = "",
    reporter: Reporter = STDIO_REPORTER,
) -> None:
    """Print one lane's status line."""
    model = f" ({spec.route.spec})" if spec.route else ""
    cost_s = f"  {format_usd(cost)}" if cost > 0 else ""
    state = (
        f"waiting on {waiting} (answer via agent6 attach {spec.session_id}, the web or TUI hub)"
        if waiting
        else status
    )
    reporter.note(f"lane {spec.lane} [{spec.session_id}]{model}: {state}{cost_s}")
