# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 stop`: one session, or every live one, now or after its step."""

from __future__ import annotations

import pathlib
import sys

from agent6 import paths
from agent6.app import stop
from agent6.sessions import id
from agent6.ui.cli import _common
from agent6.viewmodel import session_dirs, session_is_live


def _cmd_stop(session_id: str, *, all_sessions: bool, after_step: bool) -> int:
    """Stop a session, or every live one.

    Args:
        session_id: The session, or "" for the newest.
        all_sessions: Stop every live session instead.
        after_step: Stop at the end of the current step rather than now.

    Returns:
        The exit code: 1 when a stop failed, 2 when the session could not be resolved.
    """
    cwd = pathlib.Path.cwd()
    if all_sessions:
        targets = [d for d in session_dirs(paths.state_dir(cwd)) if session_is_live(d)]
        if not targets:
            print("[agent6] no live session to stop.", file=sys.stderr)
            return 0
    else:
        try:
            layout = _common.resolve_or_newest_layout(cwd, session_id)
        except id.SessionIdError as exc:
            _common.error(f"{exc}")
            return 2
        if layout is None:
            _common.print_nothing_yet()
            return 2
        targets = [layout.session_dir]
    code = 0
    for session_dir in targets:
        if all_sessions and not session_is_live(session_dir):
            continue  # a lane its coordinator's stop ended
        out = stop.stop_session(session_dir, after_step=after_step)
        print(f"[agent6] {out.message}.", file=sys.stdout if out.ok else sys.stderr)
        if out.resumable:
            print(f"  resume with:  agent6 resume {out.session_id}")
        elif out.how == "failed":
            code = 1
    return code
