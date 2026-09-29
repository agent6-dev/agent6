# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run a `/btw` side question for every composer: spawn it, deliver the answer.

`app.btw` owns the session; this owns what only a front-end can do: spawning
through the run's escape from its namespace, and landing the answer in the
run's journal.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import threading
import time
from collections.abc import Callable

from agent6 import event_log
from agent6.app import btw
from agent6.sandbox import jail
from agent6.sessions import layout
from agent6.ui import spawn

# A poll costs one status fold off the session dir.
_POLL_S = 1.0
_GIVE_UP_S = 900.0


def direct_launch(cwd: pathlib.Path, argv: list[str], env_extra: dict[str, str]) -> str:
    """Spawn `agent6 <argv>` detached, for a run with no namespace to escape.

    Returns:
        "" once spawned, else why not; the session dir appearing is the confirmation.
    """
    try:
        proc = subprocess.Popen(
            [spawn.agent6_exe(), *argv],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, **env_extra},
        )
    except OSError as exc:
        return f"could not start the btw: {exc}"
    # Unregistered, the escapee sweep would kill it at the next background command's teardown.
    jail.keep_out_of_the_sweep(proc.pid)
    return ""


def make_btw_runner(
    parent_id: str,
    *,
    launch: btw.BtwLaunch,
    list_asks: Callable[[], list[pathlib.Path]],
    events: event_log.EventSink,
) -> Callable[[str, pathlib.Path], tuple[bool, str]]:
    """Build the `/btw <question>` handler the pause menu and the composers call.

    Args:
        parent_id: The run the question sits beside.
        launch: Spawns the side session.
        list_asks: Lists the ask sessions, to find the new one.
        events: The run's journal, where the answer lands as `btw.answered`.

    Returns:
        The handler; it returns at once with whether the question opened and the line
        to show, and never blocks the run.
    """

    def run_btw(question: str, _session_dir: pathlib.Path) -> tuple[bool, str]:
        session, err = btw.start_btw(
            question, parent_id, cwd=pathlib.Path.cwd(), launch=launch, list_asks=list_asks
        )
        if session is None:
            return False, f"[agent6] {err}"
        events.emit("btw.opened", btw_id=session.id, question=session.question)
        threading.Thread(target=_watch, args=(session, events), daemon=True).start()
        return True, f"[agent6] btw {session.id} opened; its answer prints here when it lands"

    return run_btw


def _watch(session: btw.BtwSession, events: event_log.EventSink) -> None:
    """Poll until the side question answers, then put the block on the run's journal.

    Runs on a daemon thread, so it never holds the run open. Past the give-up time
    the block says so, where silence would read as an answer still coming.
    """
    deadline = time.monotonic() + _GIVE_UP_S
    while time.monotonic() < deadline:
        answer = btw.btw_answer(session)
        if answer is not None:
            events.emit("btw.answered", btw_id=session.id, block=btw.render_btw(session, answer))
            return
        time.sleep(_POLL_S)
    late = (
        f"(no answer after {_GIVE_UP_S / 60:g} minutes; `agent6 sessions show {session.id}`"
        " reads it once it ends)"
    )
    events.emit("btw.answered", btw_id=session.id, block=btw.render_btw(session, late))


def asks_dir(session_dir: pathlib.Path) -> pathlib.Path:
    """Return the asks bucket beside the session's own, derived so the two agree under any XDG."""
    return layout.bucket_dir(layout.layout_of(session_dir).state_dir, "asks")


def open_btw(session_dir: pathlib.Path, question: str) -> tuple[bool, str]:
    """Open a side question beside a live run, from any composer.

    Args:
        session_dir: The live session.
        question: The question; an empty one is refused.

    Returns:
        Whether it opened, and the line to show; the answer lands later on the journal.
    """
    if not question.strip():
        return False, "[agent6] ask something: `/btw <question>`"
    runner = make_btw_runner(
        session_dir.name,
        launch=direct_launch,
        list_asks=lambda: (
            [d for d in asks_dir(session_dir).iterdir() if d.is_dir()]
            if asks_dir(session_dir).is_dir()
            else []
        ),
        events=event_log.EventSink(session_dir / layout.LOGS_NAME),
    )
    return runner(question, session_dir)
