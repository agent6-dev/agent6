# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Speak JSON-RPC 2.0 over stdio, with the `initialize` handshake.

Framing is line-delimited JSON with a bounded read, as `ui/mcp_server.py` frames:
an unbounded `readline` buffers a whole line before any size check.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any, BinaryIO

from agent6 import __version__
from agent6.app import frontend
from agent6.ui.acp import rpc, updates
from agent6.ui.acp import session as acp_session

# The client sends the newest version it supports and disconnects if this one will not do.
PROTOCOL_VERSION = 1
# 4 MiB, the MCP server's cap; past it the payload is dropped, not buffered.
MAX_LINE_BYTES = 1 << 22
# How long EOF waits for a turn to reach its next boundary: a verify's length, not a wedged run's.
EOF_GRACE_S = 30.0

# A handler's result when a worker thread replies later, so the read loop stays free for a cancel.
DEFERRED = object()


@dataclasses.dataclass
class _Pending:
    """One outstanding request to the client.

    Attributes:
        arrived: Set once the answer landed or the wait was abandoned.
        answer: The client's response frame.
    """

    arrived: threading.Event = dataclasses.field(default_factory=threading.Event)
    answer: dict[str, Any] | None = None


def capabilities_from(_client: dict[str, Any]) -> frontend.FrontendCapabilities:
    """Return the client's capabilities in the seam every front-end declares.

    Every ACP client must serve `session/request_permission`, so it can be asked.
    """
    return frontend.FrontendCapabilities(can_ask=True)


@dataclasses.dataclass
class ACPServer:
    """One ACP connection, owning the framing.

    Attributes:
        stdin: The client's requests.
        stdout: The replies and notifications.
        client_capabilities: What the client declared at `initialize`.
        sessions: How a prompt becomes a run; None in a transport-only test.
    """

    stdin: BinaryIO
    stdout: BinaryIO
    client_capabilities: frontend.FrontendCapabilities | None = None
    sessions: acp_session.Sessions | None = None
    _handlers: dict[str, Any] = dataclasses.field(default_factory=dict)
    # One writer at a time: a reply and a worker's update interleaved is a line no editor parses.
    _write_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _pending: dict[object, _Pending] = dataclasses.field(default_factory=dict)
    _pending_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _next_id: int = 0
    _gone: bool = False  # the client's end of the pipe is closed

    def __post_init__(self) -> None:
        self._handlers = {
            "initialize": self._initialize,
            "session/new": self._session_new,
            "session/prompt": self._session_prompt,
            "session/cancel": self._session_cancel,
        }

    def serve(self) -> None:
        """Read messages until EOF; requests are answered, notifications acted on."""
        while True:
            line = self.stdin.readline(MAX_LINE_BYTES + 1)
            if not line:
                # EOF: a live turn stops at a boundary rather than mid-git holding the locks.
                self.abandon_pending()
                if self.sessions is not None:
                    self.sessions.wait_for_turns(timeout_s=EOF_GRACE_S)
                return
            if len(line) > MAX_LINE_BYTES:
                # The payload is drained in bounded chunks and dropped; its id went with it.
                while line and not line.endswith(b"\n"):
                    line = self.stdin.readline(MAX_LINE_BYTES + 1)
                self.reply(
                    None,
                    error=(
                        rpc.INVALID_REQUEST,
                        f"a request line over {MAX_LINE_BYTES} bytes was dropped",
                    ),
                )
                continue
            if not line.strip():
                continue
            self._handle(line)

    def _handle(self, line: bytes) -> None:
        """Dispatch one message to its handler and send the reply it owes."""
        parsed = self._envelope(line)
        if parsed is None:
            return
        req_id, method, params = parsed
        handler = self._handlers.get(method)
        if handler is None:
            if req_id is not None:  # an unknown notification is ignorable
                self.reply(req_id, error=(rpc.METHOD_NOT_FOUND, f"unknown method: {method!r}"))
            return
        try:
            result = handler(params, req_id)
        except rpc.RpcError as exc:
            if req_id is not None:
                self.reply(req_id, error=(exc.code, exc.message))
            return
        except Exception as exc:  # a handler bug never kills the connection
            print(f"[agent6] {method}: {traceback.format_exc()}", file=sys.stderr)
            if req_id is not None:
                self.reply(req_id, error=(rpc.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}"))
            return
        if result is DEFERRED:
            return
        if req_id is not None:
            self.reply(req_id, result=result)

    def _envelope(  # noqa: PLR0911
        self, line: bytes
    ) -> tuple[object, str, dict[str, Any]] | None:
        """Parse one line into its id, method and params.

        Invalid JSON has no id to echo, so its parse error carries a null id; the
        connection stays open.

        Returns:
            The id, method and params, or None when there is nothing to act on.
        """
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            self.reply(
                None,
                error=(
                    rpc.PARSE_ERROR,
                    f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
                ),
            )
            return None
        if not isinstance(message, dict):
            self.reply(None, error=(rpc.INVALID_REQUEST, "the JSON-RPC message must be an object"))
            return None
        req_id = message.get("id")
        if "method" not in message and self._ours(req_id) and self._deliver(req_id, message):
            # The client's answer to a request of ours; the waiting slot vouches for the frame.
            return None
        if message.get("jsonrpc") != "2.0":
            self.reply(
                req_id
                if isinstance(req_id, str | int)
                and not isinstance(req_id, bool)
                and not self._ours(req_id)
                else None,
                error=(rpc.INVALID_REQUEST, "jsonrpc must be '2.0'"),
            )
            return None
        if req_id is not None and (isinstance(req_id, bool) or not isinstance(req_id, str | int)):
            self.reply(
                None,
                error=(rpc.INVALID_REQUEST, "id must be a string, number, or null"),
            )
            return None
        method = message.get("method")
        raw = message.get("params")
        if not isinstance(method, str):
            if req_id is not None:
                # An error frame naming an id this server minted would answer its own request.
                self.reply(
                    None if self._ours(req_id) else req_id,
                    error=(rpc.INVALID_REQUEST, "no method"),
                )
            return None
        if raw is not None and not isinstance(raw, dict):
            if req_id is not None:
                self.reply(
                    req_id,
                    error=(rpc.INVALID_PARAMS, f"params for {method!r} must be an object"),
                )
            return None
        return req_id, method, raw or {}

    def abandon_pending(self) -> None:
        """Answer every outstanding request with nothing, because nobody will.

        Only the read loop delivers a client's answer; without this a worker would
        wait the full permission timeout, past the EOF grace, and be killed mid-run.
        """
        with self._pending_lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for slot in waiting:
            slot.arrived.set()

    @staticmethod
    def _ours(req_id: object) -> bool:
        """Return whether the id is one `request` minted."""
        return isinstance(req_id, str) and req_id.startswith("agent6-")

    def _deliver(self, req_id: object, message: dict[str, Any]) -> bool:
        """Hand a client response to the slot waiting for it.

        Returns:
            Whether a slot was waiting; a frame with neither result nor error is not
            a response, and would otherwise deny the approval it named.
        """
        if "result" not in message and "error" not in message:
            return False
        with self._pending_lock:
            slot = self._pending.pop(req_id, None)
        if slot is None:
            return False
        slot.answer = message
        slot.arrived.set()
        return True

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float,
        until: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Ask the client something and wait for its answer.

        Called from a worker thread, never the read loop, which delivers the answer.

        Args:
            method: The request's method.
            params: Its params.
            timeout_s: How long to wait before answering with nothing.
            until: Polled every 0.2 s; True ends the wait with nothing, the prompt
                having been answered by another route.

        Returns:
            The client's result object, or empty.
        """
        if until is not None and until():
            return {}
        with self._pending_lock:
            self._next_id += 1
            req_id = f"agent6-{self._next_id}"
            slot = _Pending()
            self._pending[req_id] = slot
        self.notify_raw({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout_s
        while not slot.arrived.wait(timeout=min(0.2, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline or (until is not None and until()):
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                return {}
        answer = slot.answer or {}
        result = answer.get("result")
        return result if isinstance(result, dict) else {}

    def _session_new(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Return the reply to `session/new`: the new session's id."""
        return self._sessions().new(params)

    def _session_prompt(self, params: dict[str, Any], req_id: object) -> object:
        """Serve `session/prompt`; the worker replies with the stop reason.

        Returns:
            `DEFERRED`.

        Raises:
            RpcError: The prompt came as a notification, which nothing could answer.
        """
        if req_id is None:
            raise rpc.RpcError(
                rpc.INVALID_REQUEST, "session/prompt is a request, not a notification"
            )
        sessions = self._sessions()
        session = sessions.get(params)
        text = acp_session.prompt_text(params)
        sessions.start_turn(
            session,
            text,
            finish=lambda reason: self.reply(req_id, result={"stopReason": reason}),
        )
        return DEFERRED

    def _session_cancel(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Serve `session/cancel`, a notification.

        Returns:
            An empty result, never sent; an unknown session is told to the editor.
        """
        sessions = self._sessions()
        try:
            sessions.cancel(sessions.get(params))
        except rpc.RpcError as exc:
            self.notify_raw(
                updates.message_update(str(params.get("sessionId")), f"cancel: {exc.message}")
            )
        return {}

    def _sessions(self) -> acp_session.Sessions:
        """Return the session runner.

        Raises:
            RpcError: None is wired.
        """
        if self.sessions is None:
            raise rpc.RpcError(rpc.INTERNAL_ERROR, "this connection has no session runner wired")
        return self.sessions

    def _initialize(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Serve `initialize`: the capability exchange.

        Returns:
            The protocol version and the agent's capabilities.

        Raises:
            RpcError: The protocol version is not an integer from 0 to 65535.
        """
        protocol_version = params.get("protocolVersion")
        if (
            not isinstance(protocol_version, int)
            or isinstance(protocol_version, bool)
            or not 0 <= protocol_version <= 65_535
        ):
            raise rpc.RpcError(
                rpc.INVALID_PARAMS, "protocolVersion must be an integer from 0 to 65535"
            )
        raw = params.get("clientCapabilities")
        self.client_capabilities = capabilities_from(raw if isinstance(raw, dict) else {})
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                "loadSession": False,
                # A resource block's uri is client-controlled; passing it through is path injection.
                "promptCapabilities": {"embeddedContext": False},
            },
            "agentInfo": {"name": "agent6", "version": __version__},
            "authMethods": [],
        }

    def reply(
        self,
        req_id: object,
        *,
        result: dict[str, Any] | None = None,
        error: tuple[int, str] | None = None,
    ) -> None:
        """Send a reply carrying the result, or the error as code and message."""
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            body["error"] = {"code": error[0], "message": error[1]}
        else:
            body["result"] = result if result is not None else {}
        self.notify_raw(body)

    def notify_raw(self, body: dict[str, Any]) -> None:
        """Write one message, encoded lossily: a lone surrogate must not desync the stream."""
        line = json.dumps(body, ensure_ascii=False, default=str) + "\n"
        gone = False
        with self._write_lock:
            if self._gone:
                gone = True
            else:
                try:
                    self.stdout.write(line.encode("utf-8", "replace"))
                    self.stdout.flush()
                except BrokenPipeError:
                    # The editor closed; the run keeps going to its next boundary.
                    self._gone = True
                    gone = True
        if gone:
            self.abandon_pending()
