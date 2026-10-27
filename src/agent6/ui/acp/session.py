# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve the session lifecycle: new, list, load, resume, close, prompt and cancel.

A prompt runs on a worker thread: a blocked read loop could not receive the
`session/cancel` ACP requires to work during one. A conversation's id is the run id its
first prompt records under, so the id an editor keeps across a restart names the run
`session/resume` reopens.
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from agent6 import kinds
from agent6.app import preflight, stop
from agent6.sessions import id, ipc, process
from agent6.sessions import layout as sessions_layout
from agent6.ui.acp import rpc, updates
from agent6.viewmodel import listing, tail

# What a prompt answers; `cancelled` is the operator's act, not a failure.
StopReason = str
# Every ACP session is an `agent6 run`; the editor cannot ask for another mode.
ACP_MODE = "run"


@dataclasses.dataclass(slots=True)
class Session:
    """One ACP session: a working directory, and at most one live execution.

    Attributes:
        acp_id: ACP's id for the conversation: the run id `session/new` reserved for its
            first prompt, or the recorded run `session/resume` reopened.
        cwd: The working directory.
        session_id: agent6's run id; "" until the first prompt mints it, then every
            later prompt resumes the same run.
        execution: The execution in progress, 1-based: the first prompt starts the run and
            each later prompt resumes it.
        thread: The execution's worker.
        cancelled: The operator cancelled the execution.
        execution_live: Cleared before the prompt answers, since the thread is still alive
            while `finish` runs and an editor may prompt again the instant it reads
            the reply.
        require_resume: Refuse a missing resume point instead of starting new work.
        closed: The editor closed the conversation while a prompt was in flight; the
            cancelled prompt still answers, then the entry goes.
    """

    acp_id: str
    cwd: pathlib.Path
    session_id: str = ""
    execution: int = 0
    thread: threading.Thread | None = None
    cancelled: bool = False
    execution_live: bool = False
    require_resume: bool = False
    closed: bool = False

    def is_running(self) -> bool:
        """Return whether an execution is in flight."""
        return self.execution_live

    def layout(self, state_dir: pathlib.Path) -> sessions_layout.SessionLayout:
        """Return the run's agent6 session dir, under ACP's own bucket."""
        return sessions_layout.SessionLayout(
            state_dir=state_dir, session_id=self.session_id, subdir=kinds.session_bucket(ACP_MODE)
        )


@dataclasses.dataclass
class Sessions:
    """The connection's sessions, and how a prompt becomes a run.

    Attributes:
        run: Runs a prompt on a session and returns the stop reason; injected so the
            transport tests without a provider and the lifecycle stays in `app`.
        state_dir_for: The state dir for a working directory.
    """

    run: Callable[[Session, str], StopReason]
    state_dir_for: Callable[[pathlib.Path], pathlib.Path]
    _by_id: dict[str, Session] = dataclasses.field(default_factory=dict)

    def new(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serve `session/new`.

        Args:
            params: The request's `cwd`, `mcpServers` and `additionalDirectories`.

        Returns:
            The reply with the new session's id.

        Raises:
            RpcError: The cwd is relative or not a git repository, or the params carry
                MCP servers or additional directories, which agent6 takes from the
                operator's config only.
        """
        cwd = _repository(params)
        _refuse_editor_extras(params)
        # The conversation's id is the run id its first prompt records under (the runner adopts
        # it unless a run took it meanwhile), so the id an editor keeps across a restart is one
        # `session/resume` and `session/list` know.
        session = Session(acp_id=id.friendly_token(), cwd=cwd)
        self._by_id[session.acp_id] = session
        return {"sessionId": session.acp_id}

    def list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serve `session/list`: the recorded runs an editor can resume, newest first.

        Args:
            params: An optional absolute `cwd` to list; without it, every repository this
                connection opened a session in.

        Returns:
            The reply: `sessions`, each with agent6's run id as its `sessionId`, its `cwd`,
            its task as `title` and its last activity as `updatedAt`; one page, no cursor.

        Raises:
            RpcError: The cwd is relative, or a cursor was passed.
        """
        if params.get("cursor"):
            raise rpc.RpcError(rpc.INVALID_PARAMS, "the listing is one page; it takes no cursor")
        cwds = (
            [_repository(params)]
            if params.get("cwd") is not None
            else sorted({s.cwd for s in self._by_id.values()})
        )
        bucket = kinds.session_bucket(ACP_MODE)
        sessions: list[dict[str, Any]] = []
        for cwd in cwds:
            state_dir = self.state_dir_for(cwd)
            rows, _winners = listing.rows_for(cwd, buckets=(bucket,))
            for row in _flat(rows):
                saved = sessions_layout.SessionLayout(
                    state_dir, row.summary.session_id, subdir=bucket
                )
                if listing.resume_blocker(saved.session_dir):
                    continue  # a coordinator, a lane, a run without a checkpoint: not reopened
                when = (
                    datetime.datetime.fromtimestamp(row.mtime, tz=datetime.UTC).isoformat(
                        timespec="seconds"
                    )
                    if row.mtime
                    else None
                )
                sessions.append(
                    {
                        "sessionId": row.summary.session_id,
                        "cwd": str(cwd),
                        "title": row.summary.task_label.line(),
                        "updatedAt": when,
                    }
                )
        return {"sessions": sessions}

    def resume(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serve `session/resume`: open a conversation on a recorded run.

        The run's id becomes the conversation's `sessionId`; the next prompt resumes the run
        with its text as the steering instruction, as every follow-up prompt does.

        Args:
            params: The run's `sessionId` and the repository's absolute `cwd`.

        Returns:
            An empty reply.

        Raises:
            RpcError: The cwd is not a repository, the id names no recorded run of it, the
                record is a plan or an ask, the run's worker is alive, or this connection
                has it open for another directory or with a prompt in flight.
        """
        cwd = _repository(params)
        _refuse_editor_extras(params)
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise rpc.RpcError(rpc.INVALID_PARAMS, "sessionId must be a non-empty string")
        saved = sessions_layout.session_layout(self.state_dir_for(cwd), session_id)
        if saved is None or saved.session_id != session_id or not saved.manifest_path.exists():
            raise rpc.RpcError(rpc.INVALID_PARAMS, f"no recorded run {session_id!r} in {cwd}")
        bucket = saved.session_dir.parent.name
        if bucket != kinds.session_bucket(ACP_MODE):
            raise rpc.RpcError(
                rpc.INVALID_PARAMS, f"{session_id} is a {bucket[:-1]}; an editor resumes runs"
            )
        if process.worker_is_alive(saved.session_dir):
            raise rpc.RpcError(
                rpc.INVALID_PARAMS,
                f"run {session_id} is running; `agent6 attach {session_id}` follows it",
            )
        if blocker := listing.resume_blocker(saved.session_dir):
            raise rpc.RpcError(rpc.INVALID_PARAMS, f"cannot resume {session_id}: {blocker}")
        known = self._by_id.get(session_id)
        if known is not None and not known.closed:
            if known.cwd.resolve() != cwd.resolve():
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS,
                    f"{session_id} is open for {known.cwd}; close it before resuming it here",
                )
            if known.is_running():
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, "that session already has a prompt in flight"
                )
            return {}  # the conversation is open already; its counters carry on
        self._by_id[session_id] = Session(acp_id=session_id, cwd=cwd, session_id=session_id)
        return {}

    def load(
        self, params: dict[str, Any], *, notify: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        """Open a recorded run and replay its conversation before returning.

        Args:
            params: The run's `sessionId` and repository's absolute `cwd`.
            notify: Sends each `session/update` notification.

        Returns:
            An empty reply after the replay.

        Raises:
            RpcError: The resume checks refuse the requested run or editor extras.
        """
        self.resume(params)
        session = self.get(params)
        saved = session.layout(self.state_dir_for(session.cwd))
        execution = updates.replay(
            tail.tail_events(saved.logs_path, follow=False),
            acp_session_id=session.acp_id,
            cwd=session.cwd,
            notify=notify,
        )
        session.execution = max(session.execution, execution)
        return {}

    def close(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serve `session/close`: end a conversation, cancelling a prompt in flight.

        Args:
            params: The `sessionId`.

        Returns:
            An empty reply.

        Raises:
            RpcError: No such session.
        """
        session = self.get(params)
        if session.is_running():
            self.cancel(session)  # the prompt answers `cancelled`, then the entry goes
            session.closed = True
            return {}
        del self._by_id[session.acp_id]
        return {}

    def get(self, params: dict[str, Any]) -> Session:
        """Return the session the params name.

        Args:
            params: The request's `sessionId`.

        Returns:
            The session.

        Raises:
            RpcError: No such session.
        """
        acp_session_id = params.get("sessionId")
        session = self._by_id.get(acp_session_id) if isinstance(acp_session_id, str) else None
        if session is None or session.closed:
            raise rpc.RpcError(rpc.INVALID_PARAMS, f"no session {acp_session_id!r}")
        return session

    def start_execution(
        self,
        session: Session,
        text: str,
        *,
        finish: Callable[[StopReason], None],
        require_resume: bool = False,
    ) -> None:
        """Run the prompt on a worker, and answer when it ends.

        Args:
            session: The session the prompt came on.
            text: The prompt, as `prompt_text` joined it.
            finish: Answers the prompt; called once with the stop reason, `cancelled` after
                a cancel whatever the run returned.
            require_resume: Refuse instead of starting new work if the resume point is missing.

        Raises:
            RpcError: The session has an execution in flight.
            RuntimeError: The worker thread could not start.
        """
        if session.is_running():
            raise rpc.RpcError(rpc.INVALID_PARAMS, "that session already has a prompt in flight")
        session.cancelled = False
        session.require_resume = require_resume

        def _work() -> None:
            try:
                reason = self.run(session, text)
            except Exception:  # a run that dies must still end the execution
                reason = "refusal"
            answer = "cancelled" if session.cancelled else reason
            session.execution_live = False
            session.require_resume = False
            finish(answer)
            if session.closed:
                self._by_id.pop(session.acp_id, None)

        session.execution_live = True
        session.thread = threading.Thread(target=_work, name=f"acp-{session.acp_id}", daemon=True)
        try:
            session.thread.start()
        except RuntimeError:
            # A thread that never ran would leave the session busy and EOF's join raising.
            session.execution_live = False
            session.thread = None
            raise

    def wait_for_executions(self, *, timeout_s: float) -> None:
        """Cancel live executions and let them finish before the process goes.

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
        """Cut a model call in flight and end a running foreground command at once.

        What the run had not committed stays in the tree for the next prompt. The worker is
        the ACP server, so the cancel never waits for or signals it.

        Args:
            session: The session to cancel; nothing happens when no execution is in flight.
        """
        if not session.is_running():
            return
        session.cancelled = True
        if not session.session_id:
            return
        session_dir = session.layout(self.state_dir_for(session.cwd)).session_dir
        out = stop.stop_session(session_dir, request_only=True)
        if not out.ok:
            # Live before the lifecycle records its worker; the marker survives startup.
            if (
                out.how == "not_live"
                and session.is_running()
                and ipc.request_stop(session_dir, now=True)
                and ipc.submit_steer(session_dir, "abort", now=True)
            ):
                return
            # A notification has no reply: stderr is the one channel left.
            print(f"[agent6] {out.message}", file=sys.stderr)


def _repository(params: dict[str, Any]) -> pathlib.Path:
    """Return the request's `cwd`: an absolute path to a git repository.

    Args:
        params: The request's params.

    Returns:
        The cwd.

    Raises:
        RpcError: The cwd is relative, or the wall `agent6 run` puts in front of a
            workspace refuses it.
    """
    raw_cwd = params.get("cwd")
    if not isinstance(raw_cwd, str) or not pathlib.Path(raw_cwd).is_absolute():
        raise rpc.RpcError(rpc.INVALID_PARAMS, "cwd must be an absolute path")
    cwd = pathlib.Path(raw_cwd)
    # This directory becomes the jail's.
    refusal = preflight.git_repo_refusal(cwd)
    if refusal is not None:
        raise rpc.RpcError(rpc.INVALID_PARAMS, refusal)
    return cwd


def _refuse_editor_extras(params: dict[str, Any]) -> None:
    """Refuse MCP servers and additional directories: agent6 takes both from its config only.

    Args:
        params: The request's params.

    Raises:
        RpcError: The request carries either.
    """
    servers = params.get("mcpServers")
    if servers is not None and not isinstance(servers, list):
        raise rpc.RpcError(rpc.INVALID_PARAMS, "mcpServers must be a list")
    if isinstance(servers, list) and servers:
        # Accepting the session and not starting them would read as connected.
        raise rpc.RpcError(
            rpc.INVALID_PARAMS,
            "agent6 does not take MCP servers from the editor: configure them in "
            "agent6's own config ([mcp.servers], `agent6 mcp add`) and remove "
            "them from this agent's entry.",
        )
    additional = params.get("additionalDirectories")
    if additional is not None and not isinstance(additional, list):
        raise rpc.RpcError(rpc.INVALID_PARAMS, "additionalDirectories must be a list")
    if isinstance(additional, list) and additional:
        raise rpc.RpcError(
            rpc.INVALID_PARAMS,
            "agent6 does not support additionalDirectories; remove them from this session",
        )


def _flat(rows: list[listing.ListingRow]) -> list[listing.ListingRow]:
    """Return the rows and their lanes, each once, in listing order."""
    out: list[listing.ListingRow] = []
    for row in rows:
        out.append(row)
        out.extend(_flat(list(row.lanes)))
    return out


def prompt_text(params: dict[str, Any]) -> str:
    """Join the prompt's text and resource_link blocks into the task.

    A `resource_link` renders as its uri, which the model reads through the
    ordinary tools; other non-text blocks are dropped, as `initialize` said.

    Args:
        params: The `session/prompt` params.

    Returns:
        The task text.

    Raises:
        RpcError: The blocks are malformed or carry no text.
    """
    blocks = params.get("prompt")
    if not isinstance(blocks, list):
        raise rpc.RpcError(rpc.INVALID_PARAMS, "prompt must be a list of content blocks")
    parts: list[str] = []
    for index, b in enumerate(blocks, start=1):
        if not isinstance(b, dict):
            raise rpc.RpcError(
                rpc.INVALID_PARAMS, f"prompt content block {index} must be an object"
            )
        if b.get("type") == "text":
            text = b.get("text")
            if not isinstance(text, str):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, f"prompt content block {index} text must be a string"
                )
            if text:
                parts.append(text)
        elif b.get("type") == "resource_link":
            uri = b.get("uri")
            if not isinstance(uri, str):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, f"prompt content block {index} uri must be a string"
                )
            if uri:
                parts.append(f"Attached: {uri}")
    text = "\n\n".join(parts).strip()
    if not text:
        raise rpc.RpcError(rpc.INVALID_PARAMS, "the prompt carried no text")
    return text
