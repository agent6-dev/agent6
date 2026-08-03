# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`session/new`, `session/prompt` and `session/cancel`.

A prompt runs on a worker thread, not on the read loop. Answering it inline
would block reading for the whole run, and a blocked loop cannot receive the
`session/cancel` that ACP requires to work DURING one -- so the cancel an
editor sends would arrive only after the thing it meant to stop had finished.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent6.runs.id import new_friendly_id
from agent6.runs.ipc import request_stop
from agent6.runs.layout import RunLayout
from agent6.ui.acp.rpc import INVALID_PARAMS, RpcError

# What ACP is told a turn ended as. `cancelled` is the operator's own act, so
# it is reported as itself rather than as a failure.
StopReason = str


@dataclass(slots=True)
class Session:
    """One ACP session: a working directory, and at most one live turn."""

    id: str
    cwd: Path
    run_id: str = ""
    thread: threading.Thread | None = None
    cancelled: bool = False
    # Cleared BEFORE the turn answers. `thread.is_alive()` is still true while
    # `finish` runs -- and `finish` IS the reply -- so a conforming editor that
    # writes its next prompt the instant it reads the answer was refused at
    # random.
    turn_live: bool = False

    def is_running(self) -> bool:
        return self.turn_live


@dataclass
class Sessions:
    """The connection's sessions, and how a prompt becomes a run."""

    # (session, prompt text) -> the stop reason. Injected so the transport can
    # be tested without a provider, and so the lifecycle stays in `app`.
    run: Callable[[Session, str], StopReason]
    state_dir_for: Callable[[Path], Path]
    _by_id: dict[str, Session] = field(default_factory=dict)

    def new(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_cwd = params.get("cwd")
        if not isinstance(raw_cwd, str) or not Path(raw_cwd).is_absolute():
            # The spec makes every path absolute; a relative one would resolve
            # against whatever directory the editor happened to launch us in.
            raise RpcError(INVALID_PARAMS, "cwd must be an absolute path")
        session = Session(id=new_friendly_id(), cwd=Path(raw_cwd))
        self._by_id[session.id] = session
        return {"sessionId": session.id}

    def get(self, params: dict[str, Any]) -> Session:
        session_id = params.get("sessionId")
        session = self._by_id.get(session_id) if isinstance(session_id, str) else None
        if session is None:
            raise RpcError(INVALID_PARAMS, f"no session {session_id!r}")
        return session

    def start_turn(
        self, session: Session, text: str, *, finish: Callable[[StopReason], None]
    ) -> None:
        """Run the prompt on a worker, and answer when it ends."""
        if session.is_running():
            raise RpcError(INVALID_PARAMS, "that session already has a turn in flight")
        session.cancelled = False

        def _work() -> None:
            try:
                reason = self.run(session, text)
            except Exception:  # a run that dies must still end the turn
                reason = "refusal"
            answer = "cancelled" if session.cancelled else reason
            session.turn_live = False
            finish(answer)

        session.turn_live = True
        session.thread = threading.Thread(target=_work, name=f"acp-{session.id}", daemon=True)
        session.thread.start()

    def wait_for_turns(self, *, timeout_s: float) -> None:
        """Let live turns finish before the process goes.

        Closing the editor is the ordinary way EOF arrives, and a daemon worker
        torn down mid-git holds the repo and worker single-writer locks and the
        run-dir pid. Cancel first so it stops at its next boundary rather than
        running to completion nobody is watching.
        """
        live = [s for s in self._by_id.values() if s.is_running()]
        for session in live:
            self.cancel(session)
        for session in live:
            if session.thread is not None:
                session.thread.join(timeout=timeout_s)

    def cancel(self, session: Session) -> None:
        """Ask the run to stop at its next boundary.

        A marker, not a kill: the finished step's tool results and auto-commit
        land first, so a cancelled turn leaves the workspace in a state the
        operator can read rather than halfway through one.
        """
        session.cancelled = True
        if session.run_id:
            request_stop(
                RunLayout(state_dir=self.state_dir_for(session.cwd), run_id=session.run_id).run_dir
            )


def prompt_text(params: dict[str, Any]) -> str:
    """The prompt's text blocks, joined.

    ACP sends content blocks; agent6's task is prose. Non-text blocks (an
    image, an embedded resource) are dropped rather than rendered as a
    placeholder the model would try to read.
    """
    blocks = params.get("prompt")
    if not isinstance(blocks, list):
        raise RpcError(INVALID_PARAMS, "prompt must be a list of content blocks")
    parts = [
        str(b.get("text", ""))
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
    ]
    text = "\n\n".join(parts).strip()
    if not text:
        raise RpcError(INVALID_PARAMS, "the prompt carried no text")
    return text
