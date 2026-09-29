# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Answer `/btw`, a question asked beside a run without interrupting it.

A btw is an ask session seeded with the run's context. The run never waits for it; its answer
prints into the conversation view between a header and a footer, never into the run's own
transcript. It has no follow-up thread: the operator resumes it like any other ask.

It is not in-process (two loops sharing one dispatcher would race on tools) and not a plain
subprocess under `strict` (it would inherit the run's empty netns). It spawns the way a
`/parallel` lane does: through the host launcher when the run is netns-isolated, else directly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent6.sessions.layout import LOGS_NAME
from agent6.viewmodel import summarize_session_dir
from agent6.viewmodel.format import status_label

# Starts a btw: (cwd, agent6 argv without the exe, env extras) -> "" or an error.
BtwLaunch = Callable[[Path, list[str], dict[str, str]], str]

# The child pays a cold Python start plus config load before its session dir appears.
_START_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class BtwSession:
    """Identify a started btw.

    Attributes:
        id: The ask session's id.
        dir: The ask session's directory.
        question: The question as asked.
    """

    id: str
    dir: Path
    question: str


def start_btw(
    question: str,
    parent_id: str,
    *,
    cwd: Path,
    launch: BtwLaunch,
    list_asks: Callable[[], list[Path]],
) -> tuple[BtwSession | None, str]:
    """Open the btw and return as soon as its session dir exists.

    The launcher is fire-and-forget, so the new dir appearing is the confirmation it started.

    Args:
        question: The question to ask.
        parent_id: The run whose context seeds the ask.
        cwd: The repository the ask runs in.
        launch: The spawn callable the front-end injects.
        list_asks: Lists the ask session dirs.

    Returns:
        The started session and "", or None and the error.
    """
    if not question:
        return None, "ask something: `/btw <question>`"
    before = {d.name for d in list_asks()}
    # `--no-commands`: nobody can approve a command for a btw; resuming the ask has the tools.
    err = launch(
        cwd,
        ["ask", "--no-commands", "--from", parent_id, "--", question],
        {"AGENT6_SUBRUN": "1"},
    )
    if err:
        return None, err
    deadline = time.monotonic() + _START_TIMEOUT_S
    while time.monotonic() < deadline:
        # Newest, not first-seen: another ask starting in the same window is not this btw.
        fresh = sorted(
            (d for d in list_asks() if d.name not in before),
            key=lambda d: d.stat().st_mtime,
        )
        if fresh:
            newest = fresh[-1]
            return BtwSession(id=newest.name, dir=newest, question=question), ""
        time.sleep(0.1)
    return None, (
        f"the btw did not start within {_START_TIMEOUT_S:g}s: no new ask session appeared."
        " The spawn is fire-and-forget, so check `agent6 sessions` and that `agent6 ask` works"
        " from this directory."
    )


def btw_answer(session: BtwSession) -> str | None:
    """Return the btw's answer once it has finished, else None.

    An ask ends with its final prose as the answer, so the last assistant message is the
    answer. A session that ended without one says so rather than rendering blank.

    Args:
        session: The started btw.

    Returns:
        The answer text, or None while the btw is still running.
    """
    # "created" spans the dir appearing and the worker pid landing; it is not an ending.
    summary = summarize_session_dir(session.dir)
    if summary.status in {"created", "running", "starting", "waiting"}:
        return None
    label = status_label(summary.status, summary.reason)
    return _final_prose(session.dir) or f"(the btw ended without an answer: {label})"


def _final_prose(session_dir: Path) -> str:
    """Return the last assistant message in the session's journal, "" without one."""
    try:
        raw = (session_dir / LOGS_NAME).read_text(errors="replace")
    except OSError:
        return ""
    answer = ""
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a torn last line while it was still writing
        if isinstance(event, dict) and event.get("type") == "role.result":
            answer = str(event.get("text", "")) or answer
    return answer.strip()


def render_btw(session: BtwSession, answer: str) -> str:
    """Return the inline block, fenced so it never reads as the run's own output."""
    return (
        f"\n--- btw: {session.question}\n"
        f"{answer.strip()}\n"
        f"--- end btw · resume it with `agent6 resume {session.id}`\n"
    )
