# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 steer`: queue a steering instruction for a live run.

Wraps the one steer channel every front-end uses (`sessions.ipc.submit_steer`),
so scripts and cron jobs can drive a running session. A session that is not
running refuses, naming `resume --steer`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.app.stop import stop_session
from agent6.directive import parse_now
from agent6.sessions.id import SessionIdError
from agent6.sessions.ipc import submit_steer
from agent6.ui.cli._common import error, refuse, resolve_session_layout
from agent6.ui.directives import act_on_directive
from agent6.viewmodel import session_is_live
from agent6.viewmodel.listing import summarize_session_dir


def _cmd_steer(target: str, text: str, *, now: bool = False) -> int:
    try:
        layout = resolve_session_layout(Path.cwd(), target)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if text.strip() == "/stop":
        out = stop_session(layout.session_dir)
        print(f"[agent6] {out.message}.", file=sys.stdout if out.ok else sys.stderr)
        return 0 if out.ok or out.how == "not_live" else 1
    if not session_is_live(layout.session_dir):
        refuse(
            f"session {layout.session_id} is not running; a steer needs a"
            f" live run. Queue one for its next leg instead:"
            f" agent6 resume {layout.session_id} --steer TEXT"
        )
        return 2
    # The same directives the composers act on, from the one owner: a line
    # typed here does what it does in the TUI, the web and the pause menu.
    handled = act_on_directive(layout.session_dir, text)
    if handled is not None:
        did, said = handled
        print(said) if did else error(said)
        return 0 if did else 1
    urgent = parse_now(text)  # `/now <text>`: what --now spells
    if urgent == "":
        refuse("/now needs the instruction: /now <text>")
        return 2
    if urgent is not None:
        text, now = urgent, True
    queued = submit_steer(layout.session_dir, text, now=now)
    if not queued:
        error(f"could not write the steer request for {layout.session_id}")
    else:
        picked = (
            "an in-flight model call is interrupted to take it"
            if now
            else "it lands at the next step boundary (--now interrupts the in-flight call)"
        )
        print(f"steer queued for {layout.session_id}: {picked}.")
        summary = summarize_session_dir(layout.session_dir)
        if summary.status == "waiting" and summary.reason:
            # Parked on an operator prompt: no boundaries arrive and no steer
            # (--now included) can break that wait; only the answer can.
            # `agent6 answer` takes a question; an approval needs a front-end.
            how = (
                f"agent6 answer {layout.session_id}"
                if summary.reason.startswith("question")
                else f"agent6 attach {layout.session_id}"
            )
            print(
                f"note: the run is waiting ({summary.reason}); the steer stays"
                f" queued until that is answered: {how}"
            )
    return 0 if queued else 1
