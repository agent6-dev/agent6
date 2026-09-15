# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 stop`: one session, or every live one, now or after its step."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from agent6.app.stop import stop_session
from agent6.paths import state_dir
from agent6.sessions.id import SessionIdError
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.ui.cli._common import error, print_nothing_yet, resolve_or_newest_layout
from agent6.viewmodel import session_dirs, session_is_live
from agent6.viewmodel.listing import lanes_of


def _cmd_stop(session_id: str, *, all_sessions: bool, after_step: bool) -> int:
    cwd = Path.cwd()
    if all_sessions:
        targets = [d for d in session_dirs(state_dir(cwd)) if session_is_live(d)]
        if not targets:
            print("[agent6] no live session to stop.", file=sys.stderr)
            return 0
    else:
        try:
            layout = resolve_or_newest_layout(cwd, session_id)
        except SessionIdError as exc:
            error(f"{exc}")
            return 2
        if layout is None:
            print_nothing_yet()
            return 2
        targets = [layout.session_dir, *_live_lanes(layout.state_dir, layout.session_dir)]
    code = 0
    for session_dir in targets:
        out = stop_session(session_dir, after_step=after_step)
        print(f"[agent6] {out.message}.", file=sys.stdout if out.ok else sys.stderr)
        if out.ok and not _is_fanout(session_dir):  # a fan-out has no loop to resume
            print(f"  resume with:  agent6 resume {out.session_id}")
        elif out.how == "failed":
            code = 1
    return code


def _is_fanout(session_dir: Path) -> bool:
    with contextlib.suppress(ManifestError):
        return read_manifest(session_dir).fanout is not None
    return False


def _live_lanes(state: Path, session_dir: Path) -> list[Path]:
    """A fan-out's live lanes: a stop of the coordinator stops them too."""
    if not _is_fanout(session_dir):
        return []
    return [
        lane_dir
        for lane in lanes_of(state, session_dir.name)
        if (lane_dir := session_dir.parent / lane.session_id).is_dir() and session_is_live(lane_dir)
    ]
