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
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import error, refuse, resolve_session_layout
from agent6.ui.directives import submit_composer_line
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
    # The one owner of what a typed line does: a line here acts as it does in
    # the TUI, the web and the pause menu.
    did, said = submit_composer_line(layout.session_dir, text, now=now)
    if not did:
        error(f"{said} ({layout.session_id})")
        return 1
    print(f"{said} for {layout.session_id}.")
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
            f"note: the run is waiting ({summary.reason}); what you sent stays"
            f" queued until that is answered: {how}"
        )
    return 0
