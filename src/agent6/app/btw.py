# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/btw` -- a question asked beside a run, without interrupting it.

A btw is a real ask session seeded with the run's context. It opens at once and
runs in parallel: the run never waits for it, and its answer is printed into the
conversation view between a header and a footer, never inserted into the run's
own transcript. Copying anything useful across is the operator's move.

One-off by construction: a btw has no follow-up thread, which keeps both the
interface and the implementation simple. It is an ask like any other, so the
operator can resume it later from another agent6 instance to go deeper.

Not in-process (two loops sharing one dispatcher would race on tools) and not a
plain subprocess under `strict` (it would inherit the run's empty netns and have
no provider egress). It is spawned the way a `/parallel` lane is: through the
host launcher when the run is netns-isolated, else directly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent6.runs.layout import LOGS_NAME
from agent6.viewmodel import summarize_run_dir

# How a btw is started: (cwd, agent6 argv without the exe, env extras) -> "" or
# an error. Exactly `app.egress.HostLaneLaunch`, which the coordinator already
# injects for `/parallel`; a btw needs the same host-namespace spawn.
BtwLaunch = Callable[[Path, list[str], dict[str, str]], str]


@dataclass(frozen=True, slots=True)
class BtwSession:
    """A btw that was started. `answer` fills in once it finishes."""

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
    """Open the btw and return as soon as it exists. Never waits for an answer.

    Returns (session, error). The new session is found by diffing the ask dirs,
    the same way a `/parallel` lane's run dir is located: the launcher is
    fire-and-forget, so the dir appearing IS the confirmation it started.
    """
    if not question:
        return None, "ask something: `/btw <question>`"
    before = {d.name for d in list_asks()}
    err = launch(cwd, ["ask", "--from", parent_id, "--", question], {"AGENT6_SUBRUN": "1"})
    if err:
        return None, err
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        fresh = [d for d in list_asks() if d.name not in before]
        if fresh:
            return BtwSession(id=fresh[0].name, dir=fresh[0], question=question), ""
        time.sleep(0.1)
    return None, "the btw did not start within 30s"


def btw_answer(session: BtwSession) -> str | None:
    """The btw's answer once it has finished, else None (still thinking).

    An ask ends by emitting its final prose AS the answer (a silent finish, no
    finish_run), so the last assistant message is the answer. A session that
    ended without one says so rather than rendering blank.
    """
    status = summarize_run_dir(session.dir).status
    if status in {"running", "starting", "waiting"}:
        return None
    return _final_prose(session.dir) or f"(the btw ended without an answer: {status})"


def _final_prose(session_dir: Path) -> str:
    """The last assistant message in *session_dir*'s journal."""
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
    """The inline block. Fenced top and bottom so it can never be misread as
    the run's own output, and labelled with the id so it can be resumed."""
    return (
        f"\n--- btw: {session.question}\n"
        f"{answer.strip()}\n"
        f"--- end btw · resume it with `agent6 resume {session.id}`\n"
    )
