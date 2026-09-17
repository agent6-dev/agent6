# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 task`: add work to a live run's task graph without steering it.

A steer is a message: the run reads it at its next step and it colours the turn
in flight. A queued task is a node in the graph, which costs the run nothing
until its frontier reaches it. A session that is not running refuses, naming
`resume --steer`.
"""

from __future__ import annotations

from pathlib import Path

from agent6.sessions.id import SessionIdError
from agent6.sessions.ipc import queue_task
from agent6.ui.cli._common import error, refuse, resolve_session_layout
from agent6.viewmodel import session_is_live


def _cmd_task(target: str, text: str) -> int:
    if not text.strip():
        refuse('a task needs text: agent6 task ID "what to do"')
        return 2
    try:
        layout = resolve_session_layout(Path.cwd(), target)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if not session_is_live(layout.session_dir):
        refuse(
            f"session {layout.session_id} is not running, so nothing would drain the"
            f" queue. Start it with the task instead:"
            f" agent6 resume {layout.session_id} --steer TEXT"
        )
        return 2
    try:
        queue_task(layout.session_dir, text)
    except OSError as exc:
        error(f"could not queue the task for {layout.session_id}: {exc}")
        return 1
    print(
        f"task queued for {layout.session_id}: it joins the task graph at the next step,"
        " and runs once the open work drains."
    )
    return 0
