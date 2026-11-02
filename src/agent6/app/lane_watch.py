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
import pathlib
import threading
import time
from collections.abc import Callable

from agent6 import paths
from agent6.app import reporter as app_reporter
from agent6.harness import subrun
from agent6.sessions import ipc, layout, process
from agent6.viewmodel import summarize_session_dir

# How often the await loop polls lane liveness.
POLL_INTERVAL_S = 2.0

# How long a stop waits for a lane to finish its in-flight step before giving up on it.
STOP_GRACE_S = 30.0


def lane_terminal(
    session_dir: pathlib.Path, status: str, worker_is_alive: Callable[[pathlib.Path], bool]
) -> bool:
    """Return whether an awaited lane is terminal: the fold left "running" and the worker is gone.

    `session.end` lands before the teardown clears `worker.pid`, so status alone races it. A
    lane that dies without a `session.end` cannot hang the gate: the fold reads it "stale".

    Args:
        session_dir: The lane's run dir.
        status: The lane's status as the fold reads it.
        worker_is_alive: The liveness check for the run dir's worker.

    Returns:
        True when the status is not "running" and the worker is gone.
    """
    return status != "running" and not worker_is_alive(session_dir)


def await_lane(
    res: subrun.LaneResult,
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
        if lane_terminal(res.session_dir, summary.status, process.worker_is_alive):
            return True
        if should_stop is not None and should_stop():
            return False
        time.sleep(poll_interval_s)


def drain_lane(
    res: subrun.LaneResult, *, poll_interval_s: float, hard_stop: threading.Event | None
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
        if lane_terminal(res.session_dir, summary.status, process.worker_is_alive):
            return True
        if hard_stop is not None:
            if hard_stop.wait(poll_interval_s):
                return False
        else:
            time.sleep(poll_interval_s)
    return False


def lane_link(origin_state: pathlib.Path, session_id: str) -> pathlib.Path:
    """Return where a lane's live symlink sits under the origin's runs dir."""
    return layout.bucket_dir(origin_state, "runs") / session_id


def symlink_lane(origin_state: pathlib.Path, res: subrun.LaneResult) -> None:
    """Symlink a lane's clone-side run dir into the origin's `runs/` so the hub shows it live.

    The import replaces the link with the real dir.

    Args:
        origin_state: The origin's state dir.
        res: The started lane.
    """
    link = lane_link(origin_state, res.spec.session_id)
    paths.mkdir_for_real_user(link.parent)
    with contextlib.suppress(FileNotFoundError):
        link.unlink()
    with contextlib.suppress(OSError):
        link.symlink_to(res.session_dir)


def await_lanes(
    started: list[subrun.LaneResult],
    *,
    already_interrupted: bool = False,
    should_stop: Callable[[], bool] | None = None,
    stop_is_now: Callable[[], bool] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> bool:
    """Poll every started lane's real run dir until it is terminal, printing status changes.

    A stop reaches every running lane with the coordinator's own urgency: an interrupt or a
    `now` request hands their running commands back and waits a bounded grace, a plain request
    lets each lane finish its turn and waits for that, escalating when a `now` arrives.

    Args:
        started: The started lanes.
        already_interrupted: A Ctrl+C landed before the await began; go straight to the stop.
        should_stop: The coordinator's own stop request, read between polls.
        stop_is_now: Whether that request carries the `now` urgency; None reads as `now`.
        reporter: Where the status lines go.

    Returns:
        True when interrupted, False when every lane ended.
    """
    pending = {r.spec.session_id: r for r in started}
    seen: dict[str, tuple[str, str, str]] = {}

    def poll_once() -> None:
        for rid, res in list(pending.items()):
            summary = summarize_session_dir(res.session_dir)
            # A "waiting" lane is blocked on a prompt no detached lane can answer.
            waiting = pending_prompt(res.session_dir) if summary.status == "waiting" else ""
            key = (summary.status, waiting, summary.cost_cell)
            if seen.get(rid) != key:
                seen[rid] = key
                print_lane_status(
                    res.spec, summary.status, summary.cost_cell, waiting=waiting, reporter=reporter
                )
            if lane_terminal(res.session_dir, summary.status, process.worker_is_alive):
                pending.pop(rid)

    def escalates() -> bool:
        return stop_is_now is None or stop_is_now()

    def stop_and_drain(*, immediate: bool) -> None:
        reporter.err(
            "\n[agent6] interrupted; stopping lanes..."
            if immediate
            else "\n[agent6] stopping lanes after their current turns..."
        )
        delivered = True
        for res in pending.values():
            if not ipc.request_stop(res.session_dir, now=immediate):
                delivered = False
                reporter.err(f"[agent6] could not write the stop request for {res.spec.session_id}")
        # A lane that never got its marker ends on its own terms: the wait is the grace, not open.
        deadline = time.monotonic() + STOP_GRACE_S if immediate or not delivered else None
        try:
            while pending and (deadline is None or time.monotonic() < deadline):
                poll_once()
                if pending and deadline is None and escalates():
                    stop_and_drain(immediate=True)
                    return
                if pending:
                    time.sleep(POLL_INTERVAL_S)
        except KeyboardInterrupt:
            if deadline is None:
                stop_and_drain(immediate=True)

    if already_interrupted:
        stop_and_drain(immediate=True)
        return True
    try:
        while True:
            poll_once()
            # A request that landed while the lanes started still ends the fan-out as a stop.
            if should_stop is not None and should_stop():
                if pending:
                    stop_and_drain(immediate=escalates())
                return True
            if not pending:
                return False
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        stop_and_drain(immediate=True)
        return True


# The two prompt/answer event pairs a lane can block on.
_PROMPT_KIND = {"approval.prompt": "approval", "question.prompt": "a question"}
_ANSWER_EVENTS = frozenset({"approval.answer", "question.answer"})


def pending_prompt(session_dir: pathlib.Path) -> str:
    """Return "approval" or "a question" when the lane is blocked on an unanswered prompt.

    The last prompt or answer event in the log decides it: the worker blocks on its answer
    while its away mode is `wait`, and no request marker exists for prompts.

    Args:
        session_dir: The lane's run dir.

    Returns:
        The prompt kind, or "" when nothing is pending.
    """
    try:
        lines = (
            (session_dir / layout.LOGS_NAME)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
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
        if etype == "question.prompt" and ev.get("wait") is False:
            continue  # a queued question pauses nothing
        if etype in _PROMPT_KIND:
            return _PROMPT_KIND[etype]
    return ""


def print_lane_status(
    spec: subrun.LaneSpec,
    status: str,
    cost: str,
    *,
    waiting: str = "",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> None:
    """Print one lane's status line, its cost as the listing's cell spells it.

    Args:
        spec: The lane, shown by number, session id and model route, if any.
        status: The status, shown unless `waiting` is set.
        cost: The cost; "" shows none.
        waiting: The prompt kind the lane waits on, from `pending_prompt`; shown with where
            to answer.
        reporter: Receives the line.
    """
    model = f" ({spec.route.spec})" if spec.route else ""
    cost_s = f"  {cost}" if cost else ""
    state = (
        f"waiting on {waiting} (answer via agent6 attach {spec.session_id}, the web or TUI hub)"
        if waiting
        else status
    )
    reporter.note(f"lane {spec.lane} [{spec.session_id}]{model}: {state}{cost_s}")
