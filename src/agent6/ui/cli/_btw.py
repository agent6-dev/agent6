# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The CLI's `/btw` runner: spawn the side question, deliver the answer.

The menu owns the grammar and `app.btw` owns the session; this owns the two
things only the front-end can do -- spawning through whatever escape the run
has from its namespace, and handing the finished answer to the console view,
which prints it whole at the next turn boundary.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from threading import Thread

from agent6.app.btw import BtwLaunch, BtwSession, btw_answer, render_btw, start_btw
from agent6.ui.cli._console_view import ConsoleView
from agent6.ui.spawn import agent6_exe

# How often the watcher looks for the answer. A btw is a short question, and
# the cost of a poll is one status fold off the session dir.
_POLL_S = 1.0
_GIVE_UP_S = 900.0


def direct_launch(cwd: Path, argv: list[str], env_extra: dict[str, str]) -> str:
    """Spawn `agent6 <argv>` detached, for a run with no namespace to escape.

    Fire-and-forget: the session dir appearing is the confirmation, exactly as
    for a `/parallel` lane.
    """
    try:
        subprocess.Popen(
            [agent6_exe(), *argv],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, **env_extra},
        )
    except OSError as exc:
        return f"could not start the btw: {exc}"
    return ""


def make_btw_runner(
    parent_id: str,
    *,
    launch: BtwLaunch,
    list_asks: Callable[[], list[Path]],
    console_view: Callable[[], ConsoleView | None],
) -> Callable[[str, Path], str]:
    """The `/btw <question>` handler the pause menu calls.

    Returns immediately with a line to print; the answer arrives later on the
    console view. A btw never blocks the run -- that is the whole point of
    asking beside it rather than steering it.
    """

    def run_btw(question: str, _run_dir: Path) -> str:
        session, err = start_btw(
            question, parent_id, cwd=Path.cwd(), launch=launch, list_asks=list_asks
        )
        if session is None:
            return f"[agent6] {err}"
        Thread(target=_watch, args=(session, console_view), daemon=True).start()
        return f"[agent6] btw {session.id} opened; its answer prints here when it lands"

    return run_btw


def _watch(session: BtwSession, console_view: Callable[[], ConsoleView | None]) -> None:
    """Poll until the btw answers, then hand the block to the view.

    Daemon thread: a btw must never hold the run open, and an unanswered one at
    exit is simply an ask the operator can resume.
    """
    deadline = time.monotonic() + _GIVE_UP_S
    while time.monotonic() < deadline:
        answer = btw_answer(session)
        if answer is not None:
            view = console_view()
            if view is not None:
                view.queue_btw(render_btw(session, answer))
            return
        time.sleep(_POLL_S)
