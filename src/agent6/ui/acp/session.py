# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve `session/new`, `session/prompt` and `session/cancel`.

A prompt runs on a worker thread: a blocked read loop could not receive the
`session/cancel` ACP requires to work during one.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent6.app.preflight import git_repo_refusal
from agent6.app.stop import stop_session
from agent6.kinds import session_bucket
from agent6.sessions.id import friendly_token
from agent6.sessions.ipc import request_stop
from agent6.sessions.layout import SessionLayout
from agent6.ui.acp.rpc import INVALID_PARAMS, RpcError

# What ACP is told a turn ended as; `cancelled` is the operator's act, not a failure.
StopReason = str
# Every ACP turn is an `agent6 run`; the editor cannot ask for another mode.
ACP_MODE = "run"


@dataclass(slots=True)
class Session:
    """One ACP session: a working directory, and at most one live turn.

    Attributes:
        acp_id: ACP's id for the conversation, minted by `session/new`.
        cwd: The working directory.
        session_id: agent6's run id; "" until the first prompt mints it, then every
            later prompt resumes the same run.
        turn: The turn in progress, 1-based: one execution of the run.
        thread: The turn's worker.
        cancelled: The operator cancelled the turn.
        turn_live: Cleared before the turn answers, since the thread is still alive
            while `finish` runs and an editor may prompt again the instant it reads
            the reply.
    """

    acp_id: str
    cwd: Path
    session_id: str = ""
    turn: int = 0
    thread: threading.Thread | None = None
    cancelled: bool = False
    turn_live: bool = False

    def is_running(self) -> bool:
        """Return whether a turn is in flight."""
        return self.turn_live

    def layout(self, state_dir: Path) -> SessionLayout:
        """Return the turn's agent6 session dir, under ACP's own bucket."""
        return SessionLayout(
            state_dir=state_dir, session_id=self.session_id, subdir=session_bucket(ACP_MODE)
        )


@dataclass
class Sessions:
    """The connection's sessions, and how a prompt becomes a run.

    Attributes:
        run: Runs a prompt on a session and returns the stop reason; injected so the
            transport tests without a provider and the lifecycle stays in `app`.
        state_dir_for: The state dir for a working directory.
    """

    run: Callable[[Session, str], StopReason]
    state_dir_for: Callable[[Path], Path]
    _by_id: dict[str, Session] = field(default_factory=dict)

    def new(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serve `session/new`.

        Returns:
            The reply with the new session's id.

        Raises:
            RpcError: The cwd is relative or not a git repository, or the params carry
                MCP servers or additional directories, which agent6 takes from the
                operator's config only.
        """
        raw_cwd = params.get("cwd")
        if not isinstance(raw_cwd, str) or not Path(raw_cwd).is_absolute():
            raise RpcError(INVALID_PARAMS, "cwd must be an absolute path")
        cwd = Path(raw_cwd)
        # The wall `agent6 run` puts in front of a workspace: this directory becomes the jail's.
        refusal = git_repo_refusal(cwd)
        if refusal is not None:
            raise RpcError(INVALID_PARAMS, refusal)
        servers = params.get("mcpServers")
        if servers is not None and not isinstance(servers, list):
            raise RpcError(INVALID_PARAMS, "mcpServers must be a list")
        if isinstance(servers, list) and servers:
            # Accepting the session and not starting them would read as connected.
            raise RpcError(
                INVALID_PARAMS,
                "agent6 does not take MCP servers from the editor: configure them in "
                "agent6's own config ([mcp.servers], `agent6 mcp connect`) and remove "
                "them from this agent's entry.",
            )
        additional = params.get("additionalDirectories")
        if additional is not None and not isinstance(additional, list):
            raise RpcError(INVALID_PARAMS, "additionalDirectories must be a list")
        if isinstance(additional, list) and additional:
            raise RpcError(
                INVALID_PARAMS,
                "agent6 does not support additionalDirectories; remove them from this session",
            )
        session = Session(acp_id=friendly_token(), cwd=cwd)
        self._by_id[session.acp_id] = session
        return {"sessionId": session.acp_id}

    def get(self, params: dict[str, Any]) -> Session:
        """Return the session the params name.

        Raises:
            RpcError: No such session.
        """
        acp_session_id = params.get("sessionId")
        session = self._by_id.get(acp_session_id) if isinstance(acp_session_id, str) else None
        if session is None:
            raise RpcError(INVALID_PARAMS, f"no session {acp_session_id!r}")
        return session

    def start_turn(
        self, session: Session, text: str, *, finish: Callable[[StopReason], None]
    ) -> None:
        """Run the prompt on a worker, and answer when it ends.

        Raises:
            RpcError: The session has a turn in flight.
            RuntimeError: The worker thread could not start.
        """
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
        session.thread = threading.Thread(target=_work, name=f"acp-{session.acp_id}", daemon=True)
        try:
            session.thread.start()
        except RuntimeError:
            # A thread that never ran would leave the session busy and EOF's join raising.
            session.turn_live = False
            session.thread = None
            raise

    def wait_for_turns(self, *, timeout_s: float) -> None:
        """Cancel live turns and let them finish before the process goes.

        A daemon worker torn down mid-git holds the repo locks and the run-dir pid.

        Args:
            timeout_s: One deadline across every join.
        """
        live = [s for s in self._by_id.values() if s.is_running()]
        for session in live:
            self.cancel(session)
        deadline = time.monotonic() + timeout_s
        for session in live:
            if session.thread is not None:
                session.thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def cancel(self, session: Session) -> None:
        """Ask the run to stop at its next boundary.

        A marker, not a kill: the step's tool results and auto-commit land first.
        """
        if not session.is_running():
            return
        session.cancelled = True
        if not session.session_id:
            return
        session_dir = session.layout(self.state_dir_for(session.cwd)).session_dir
        out = stop_session(session_dir, after_step=True)
        if not out.ok:
            # The turn is live before the lifecycle records its worker; the marker survives startup.
            if out.how == "not_live" and session.is_running() and request_stop(session_dir):
                return
            # A notification has no reply: stderr is the one channel left.
            print(f"[agent6] {out.message}", file=sys.stderr)


def prompt_text(params: dict[str, Any]) -> str:
    """Join the prompt's text and resource_link blocks into the task.

    A `resource_link` renders as its uri, which the model reads through the
    ordinary tools; other non-text blocks are dropped, as `initialize` said.

    Returns:
        The task text.

    Raises:
        RpcError: The blocks are malformed or carry no text.
    """
    blocks = params.get("prompt")
    if not isinstance(blocks, list):
        raise RpcError(INVALID_PARAMS, "prompt must be a list of content blocks")
    parts: list[str] = []
    for index, b in enumerate(blocks, start=1):
        if not isinstance(b, dict):
            raise RpcError(INVALID_PARAMS, f"prompt content block {index} must be an object")
        if b.get("type") == "text":
            text = b.get("text")
            if not isinstance(text, str):
                raise RpcError(
                    INVALID_PARAMS, f"prompt content block {index} text must be a string"
                )
            if text:
                parts.append(text)
        elif b.get("type") == "resource_link":
            uri = b.get("uri")
            if not isinstance(uri, str):
                raise RpcError(INVALID_PARAMS, f"prompt content block {index} uri must be a string")
            if uri:
                parts.append(f"Attached: {uri}")
    text = "\n\n".join(parts).strip()
    if not text:
        raise RpcError(INVALID_PARAMS, "the prompt carried no text")
    return text
