# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 steer`: queue a steering instruction for a live run.

Wraps the one steer channel every front-end uses (`sessions.ipc.submit_steer`), so scripts
and cron jobs can drive a running session. A session that is not running refuses, naming
`resume --steer`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent6.app.stop import stop_session
from agent6.directive import VIEW_COMMANDS
from agent6.sessions.id import SessionIdError
from agent6.ui.cli._common import error, refuse, resolve_session_layout
from agent6.ui.directives import submit_composer_line
from agent6.viewmodel import session_is_live
from agent6.viewmodel.listing import summarize_session_dir


def _cmd_steer(target: str, text: str, *, now: bool = False) -> int:
    """Send a composer line to a live session.

    Args:
        target: The session id or prefix.
        text: The line, as typed in a composer; `/stop` stops the session.
        now: Interrupt the current call instead of waiting for a boundary.

    Returns:
        The exit code; 2 when the session is unknown, not live, or the line is a view command.
    """
    try:
        layout = resolve_session_layout(Path.cwd(), target)
    except SessionIdError as exc:
        error(f"{exc}")
        return 2
    if text.strip() == "/stop":
        out = stop_session(layout.session_dir)
        print(f"[agent6] {out.message}.", file=sys.stdout if out.ok else sys.stderr)
        return 0 if out.ok or out.how == "not_live" else 1
    if text.strip().lower() in VIEW_COMMANDS:
        refuse(
            f"{text.strip()} acts on a view of the run, which a steer has none of:"
            f" type it in `agent6 attach {layout.session_id}`, the TUI or the web"
        )
        return 2
    if not session_is_live(layout.session_dir):
        refuse(
            f"session {layout.session_id} is not running; a steer needs a"
            f" live run. Queue one for its next execution instead:"
            f" agent6 resume {layout.session_id} --steer TEXT"
        )
        return 2
    # The one owner of what a typed line does, shared with the TUI, the web and the pause menu.
    did, said = submit_composer_line(layout.session_dir, text, now=now)
    if not did:
        error(f"{said} ({layout.session_id})")
        return 1
    print(f"{said} for {layout.session_id}.")
    summary = summarize_session_dir(layout.session_dir)
    if summary.status == "waiting" and summary.reason:
        # Parked on an operator prompt, only the answer ends the wait; approvals need a front-end.
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
