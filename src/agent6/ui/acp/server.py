# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Speak JSON-RPC 2.0 over stdio, with the `initialize` handshake.

Framing is line-delimited JSON with a bounded read, as `ui/mcp_server.py` frames:
an unbounded `readline` buffers a whole line before any size check.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any, BinaryIO

from agent6 import __version__
from agent6.app import frontend
from agent6.ui import operator_images
from agent6.ui.acp import model, review, rpc, updates, wire
from agent6.ui.acp import session as acp_session

# The protocol versions served, lowest first; `initialize` selects one for the connection.
VERSIONS: tuple[int, ...] = (1, 2)
# 4 MiB, the MCP server's cap, and a prompt's images; past it the payload is dropped, not buffered.
MAX_LINE_BYTES = (1 << 22) + operator_images.ENCODED_TOTAL_MAX
# How long EOF waits for an execution to reach a boundary: a verify's length, not a wedged run's.
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


@dataclasses.dataclass
class _PromptAnswer:
    """Answer one `session/prompt` as the connection's version orders it.

    v1 answers when the turn's work ends, with its stop reason or its error. v2 answers
    once the prompt's message is in the conversation, with the message's id; an error
    answers only a prompt whose message never got in. A retry that arrives before the
    answer shares it.

    Attributes:
        server: The connection.
        session: The session the prompt came on.
        message_id: The id of the prompt's message.
        request_ids: The requests the answer goes to: the prompt's, then each retry's.
    """

    server: ACPServer
    session: acp_session.Session
    message_id: str
    request_ids: list[object]
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _answered: bool = False

    def inserted(self) -> None:
        """Record the prompt's message as in the conversation, and answer a v2 prompt."""
        self.session.inserted.add(self.message_id)
        if self.server.protocol_version >= 2:
            self._once(result={"messageId": self.message_id})

    def ended(self, end: acp_session.TurnEnd) -> None:
        """Answer a v1 prompt for its turn's end, or a v2 prompt whose message never got in."""
        if isinstance(end, rpc.RpcError):
            self._once(error=end)
        elif self.server.protocol_version == 1:
            self._once(result={"stopReason": end})
        elif end == "cancelled":
            why = "the prompt was cancelled before its message was added"
            self._once(error=rpc.RpcError(rpc.REQUEST_CANCELLED, why))
        else:
            self.inserted()

    def join(self, request_id: object) -> bool:
        """Add a retry's request to the answer.

        Args:
            request_id: The retry's request.

        Returns:
            False when the answer already went out.
        """
        with self._lock:
            if not self._answered:
                self.request_ids.append(request_id)
            return not self._answered

    def _once(
        self, *, result: dict[str, Any] | None = None, error: rpc.RpcError | None = None
    ) -> None:
        with self._lock:
            if self._answered:
                return
            self._answered = True
        self.server.unanswered.pop((self.session.acp_id, self.message_id), None)
        for request_id in self.request_ids:
            self.server.reply(request_id, result=result, error=error)


def capabilities_from(_client: dict[str, Any]) -> frontend.FrontendCapabilities:
    """Return the client's capabilities in the seam every front-end declares.

    Every ACP client must serve `session/request_permission`, so it can be asked.

    Returns:
        `can_ask` true, whatever the client declared.
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
        protocol_version: The version `initialize` selected; 1 until then, as a v1 editor
            that skips the handshake is still served.
        unanswered: The prompts not answered yet, by session and message id, so a retry
            with the same idempotency key joins the first.
    """

    stdin: BinaryIO
    stdout: BinaryIO
    client_capabilities: frontend.FrontendCapabilities | None = None
    sessions: acp_session.Sessions | None = None
    protocol_version: int = 1
    unanswered: dict[tuple[str, str], _PromptAnswer] = dataclasses.field(default_factory=dict)
    _handlers: dict[str, Any] = dataclasses.field(default_factory=dict)
    # One writer at a time: a reply and a worker's update interleaved is a line no editor parses.
    _write_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _pending: dict[object, _Pending] = dataclasses.field(default_factory=dict)
    _pending_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _next_id: int = 0
    _asks: int = 0  # under `_pending_lock`; numbers the calls v1 makes up for a bare prompt
    _used: bool = False  # a request other than `initialize` arrived
    _gone: bool = False  # the client's end of the pipe is closed

    def __post_init__(self) -> None:
        self._handlers = {
            "initialize": self._initialize,
            "session/new": self._session_new,
            "session/list": self._session_list,
            "session/load": self._session_load,
            "session/resume": self._session_resume,
            "session/close": self._session_close,
            "session/prompt": self._session_prompt,
            "session/cancel": self._session_cancel,
            "session/set_config_option": self._session_set_config_option,
        }

    def serve(self) -> None:
        """Read messages until EOF; requests are answered, notifications acted on."""
        while True:
            line = self.stdin.readline(MAX_LINE_BYTES + 1)
            if not line:
                # EOF: a live execution stops at a boundary rather than mid-git holding the locks.
                self.abandon_pending()
                if self.sessions is not None:
                    self.sessions.wait_for_executions(timeout_s=EOF_GRACE_S)
                return
            if len(line) > MAX_LINE_BYTES:
                # The payload is drained in bounded chunks and dropped; its id went with it.
                while line and not line.endswith(b"\n"):
                    line = self.stdin.readline(MAX_LINE_BYTES + 1)
                self.reply(
                    None,
                    error=rpc.RpcError(
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
        self._used = self._used or (req_id is not None and method != "initialize")
        handler = self._handlers.get(method)
        if method == "session/load" and self.protocol_version >= 2:
            handler = None  # v2 has no load: `session/resume` replays
        if handler is None:
            if req_id is not None:  # an unknown notification is ignorable
                self.reply(
                    req_id, error=rpc.RpcError(rpc.METHOD_NOT_FOUND, f"unknown method: {method!r}")
                )
            return
        try:
            result = handler(params, req_id)
        except rpc.RpcError as exc:
            if req_id is not None:
                self.reply(req_id, error=exc)
            return
        except Exception as exc:  # a handler bug never kills the connection
            print(f"[agent6] {method}: {traceback.format_exc()}", file=sys.stderr)
            if req_id is not None:
                self.reply(
                    req_id, error=rpc.RpcError(rpc.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
                )
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

        Args:
            line: One line the client sent.

        Returns:
            The id, method and params, or None when there is nothing to act on.
        """
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            self.reply(
                None,
                error=rpc.RpcError(
                    rpc.PARSE_ERROR,
                    f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
                ),
            )
            return None
        if not isinstance(message, dict):
            self.reply(
                None,
                error=rpc.RpcError(rpc.INVALID_REQUEST, "the JSON-RPC message must be an object"),
            )
            return None
        req_id = message.get("id")
        if "method" not in message and self._ours(req_id) and self._deliver(req_id, message):
            # A reply to our request, including one whose wait already ended.
            return None
        if message.get("jsonrpc") != "2.0":
            self.reply(
                req_id
                if isinstance(req_id, str | int)
                and not isinstance(req_id, bool)
                and not self._ours(req_id)
                else None,
                error=rpc.RpcError(rpc.INVALID_REQUEST, "jsonrpc must be '2.0'"),
            )
            return None
        if req_id is None and "id" in message and self.protocol_version >= 2:
            self.reply(None, error=rpc.RpcError(rpc.INVALID_REQUEST, "id must not be null"))
            return None
        if req_id is not None and (isinstance(req_id, bool) or not isinstance(req_id, str | int)):
            self.reply(
                None,
                error=rpc.RpcError(rpc.INVALID_REQUEST, "id must be a string, number, or null"),
            )
            return None
        method = message.get("method")
        raw = message.get("params")
        if not isinstance(method, str):
            if req_id is not None:
                # An error frame naming an id this server minted would answer its own request.
                self.reply(
                    None if self._ours(req_id) else req_id,
                    error=rpc.RpcError(rpc.INVALID_REQUEST, "no method"),
                )
            return None
        if raw is not None and not isinstance(raw, dict):
            if req_id is not None:
                self.reply(
                    req_id,
                    error=rpc.RpcError(
                        rpc.INVALID_PARAMS, f"params for {method!r} must be an object"
                    ),
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

        Args:
            req_id: The id the frame answers, one `request` minted.
            message: The client's frame.

        Returns:
            Whether the frame is a response; a late one is consumed without an answer.
            A frame with neither result nor error leaves the approval pending.
        """
        if "result" not in message and "error" not in message:
            return False
        with self._pending_lock:
            slot = self._pending.pop(req_id, None)
        if slot is None:
            return True
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
            until: Polled every 0.2 s and checked when a reply arrives; True discards the
                answer because the prompt was cancelled or answered by another route.

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
                if self.protocol_version >= 2:
                    # The wait is over: a v2 editor dismisses the prompt it still shows.
                    cancel = {"jsonrpc": "2.0", "method": "$/cancel_request"}
                    self.notify_raw({**cancel, "params": {"requestId": req_id}})
                return {}
        if until is not None and until():
            return {}
        answer = slot.answer or {}
        result = answer.get("result")
        return result if isinstance(result, dict) else {}

    def request_permission(
        self,
        acp_session_id: str,
        request: model.PermissionRequest,
        *,
        timeout_s: float,
        until: Callable[[], bool],
    ) -> int | None:
        """Put one approval or question to the editor as the connection's version asks it.

        v1 names a tool call in every request, so a prompt that gates none announces a
        call of its own and closes it with the answer.

        Args:
            acp_session_id: The conversation the request addresses.
            request: The prompt, the call it gates and its answers.
            timeout_s: How long to wait before answering with nothing.
            until: Polled while the answer is pending; True ends the wait.

        Returns:
            The place of the chosen option, or None for no usable answer.
        """
        call_id = request.tool_call_id
        if not call_id and self.protocol_version == 1:
            with self._pending_lock:
                self._asks += 1
                call_id = f"ask-{acp_session_id}-{self._asks}"
        answer = self.request(
            "session/request_permission",
            wire.permission_request(self.protocol_version, acp_session_id, request, call_id),
            timeout_s=timeout_s,
            until=until,
        )
        chosen = wire.selected_option(answer, len(request.options))
        if call_id != request.tool_call_id:
            closed = "failed" if chosen is None else "completed"
            self.update(acp_session_id, model.ToolCallUpdate(call_id, closed))
        return chosen

    def _session_new(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Return the reply to `session/new`: the new session's id, options and commands."""
        return wire.result(self.protocol_version, self._sessions().new(params))

    def _session_list(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Return the reply to `session/list`: the recorded runs, newest first."""
        return self._sessions().list(params)

    def _session_load(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Return the reply to v1's `session/load`: a resume that replays from the start."""
        replaying = {**params, "replayFrom": {"type": "start"}}
        return wire.result(1, self._sessions().resume(replaying, notify=self.update))

    def _session_resume(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Return the reply to `session/resume`: the run's conversation, replayed when asked."""
        if self.protocol_version == 1:
            # v1's resume has no replay cursor: `session/load` is its replay.
            params = {key: value for key, value in params.items() if key != "replayFrom"}
        opened = self._sessions().resume(params, notify=self.update)
        return wire.result(self.protocol_version, opened)

    def _session_set_config_option(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Return the reply to `session/set_config_option`: the complete config options."""
        if self.protocol_version == 1:
            params = {**params, "type": "id"}  # v1 has selects alone, and no tag for them
        chosen = self._sessions().set_config_option(params)
        return wire.result(self.protocol_version, chosen)

    def _session_close(self, params: dict[str, Any], req_id: object) -> object:
        """Serve `session/close`: end a conversation, cancelling a prompt in flight.

        Args:
            params: The `sessionId`.
            req_id: The id the reply answers.

        Returns:
            Under v1 the empty reply at once, the cancelled prompt answering after it. Under
            v2 `DEFERRED`: the reply follows the cancelled work's last update, and closing a
            conversation that is not open succeeds.

        Raises:
            RpcError: Under v1, no such session.
        """
        sessions = self._sessions()
        closed: dict[str, Any] = {}
        if self.protocol_version == 1:
            sessions.close(params, settled=lambda: None)
            return closed
        try:
            sessions.close(params, settled=lambda: self.reply(req_id, result=closed))
        except rpc.RpcError as exc:
            if exc.code != rpc.RESOURCE_NOT_FOUND:
                raise
            return closed
        return DEFERRED

    def _session_prompt(self, params: dict[str, Any], req_id: object) -> object:
        """Serve `session/prompt`; the turn answers as the connection's version orders it.

        Args:
            params: The `sessionId` and the `prompt` content blocks.
            req_id: The id the turn's answer carries.

        Returns:
            `DEFERRED`; or at once the id of the message a retried prompt already added.

        Raises:
            RpcError: The prompt came as a notification, which nothing could answer, or its
                blocks, its images or its idempotency key are malformed.
        """
        if req_id is None:
            raise rpc.RpcError(
                rpc.INVALID_REQUEST, "session/prompt is a request, not a notification"
            )
        sessions = self._sessions()
        session = sessions.get(params)
        images = acp_session.prompt_images(params)
        strict = self.protocol_version >= 2
        text = acp_session.prompt_text(params, allow_empty=bool(images), strict=strict)
        if images:
            # Checked before the worker starts, so a refused image answers this request.
            if text.startswith("/"):
                raise rpc.RpcError(
                    rpc.INVALID_PARAMS, "a directive takes no image; send it with a prompt"
                )
            try:
                operator_images.decode(images)
            except operator_images.ImageRefusedError as exc:
                raise rpc.RpcError(rpc.INVALID_PARAMS, str(exc)) from exc

        message_id = self._prompt_id(session, params)
        waiting = self.unanswered.get((session.acp_id, message_id))
        if waiting is not None and waiting.join(req_id):
            return DEFERRED  # a retry of a prompt not answered yet shares its answer
        # Only a keyed prompt can repeat a message, so only it reads the run's record.
        keyed = self.protocol_version >= 2 and params.get("idempotencyKey") is not None
        if keyed and sessions.has_message(session, message_id):
            return {"messageId": message_id}
        answer = _PromptAnswer(self, session, message_id, [req_id])
        self.unanswered[session.acp_id, message_id] = answer
        turn = acp_session.Turn(
            finish=answer.ended,
            inserted=answer.inserted,
            notify=functools.partial(self.update, session.acp_id),
            message_id=answer.message_id,
        )
        try:
            if review.handle_command(sessions, session, text, turn=turn):
                return DEFERRED
            if self.protocol_version >= 2 and session.is_running():
                # v2 takes a prompt during a run; v1 refuses it as busy.
                sessions.steer(session, text, turn=turn, images=images)
            else:
                sessions.start_execution(session, text, turn=turn, images=images)
        except rpc.RpcError as exc:
            answer.ended(exc)  # refused before its message was added: its retries hear it too
        return DEFERRED

    def _prompt_id(self, session: acp_session.Session, params: dict[str, Any]) -> str:
        """Return the id of the prompt's message, derived from its idempotency key when it has one.

        The same key on the same session names the same message, across restarts too, and
        the key itself is never stored.

        Args:
            session: The session the prompt came on.
            params: The `session/prompt` params.

        Returns:
            A fresh id for a prompt without a key; v1 has no key.

        Raises:
            RpcError: The key is not a string of 1 to 255 characters.
        """
        key = params.get("idempotencyKey") if self.protocol_version >= 2 else None
        if key is None:
            return uuid.uuid4().hex
        if not isinstance(key, str) or not 1 <= len(key) <= 255:
            raise rpc.RpcError(
                rpc.INVALID_PARAMS, "idempotencyKey must be a string of 1 to 255 characters"
            )
        return uuid.uuid5(uuid.NAMESPACE_OID, f"agent6.acp.prompt\n{session.acp_id}\n{key}").hex

    def _session_cancel(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Serve `session/cancel`, a notification.

        Args:
            params: The `sessionId` of the session to cancel.

        Returns:
            An empty result, never sent; a v1 editor is told of an unknown session, and v2
            has a cancel for one ignored.
        """
        sessions = self._sessions()
        try:
            sessions.cancel(sessions.get(params))
        except rpc.RpcError as exc:
            if self.protocol_version == 1:
                self.update(str(params.get("sessionId")), updates.notice(f"cancel: {exc.message}"))
        return {}

    def _sessions(self) -> acp_session.Sessions:
        """Return the session runner.

        Returns:
            The connection's session table.

        Raises:
            RpcError: None is wired.
        """
        if self.sessions is None:
            raise rpc.RpcError(rpc.INTERNAL_ERROR, "this connection has no session runner wired")
        return self.sessions

    def _initialize(self, params: dict[str, Any], _req_id: object) -> dict[str, Any]:
        """Serve `initialize`: select the connection's version and exchange capabilities.

        Args:
            params: The client's `protocolVersion` and its capabilities.

        Returns:
            The highest version served that is at most the one requested, and the agent's
            capabilities in that version's shape.

        Raises:
            RpcError: The protocol version is not an integer from 0 to 65535, no version
                served is that low, or a v2 connection repeats `initialize` after another
                request.
        """
        requested = params.get("protocolVersion")
        if (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or not 0 <= requested <= 65_535
        ):
            raise rpc.RpcError(
                rpc.INVALID_PARAMS, "protocolVersion must be an integer from 0 to 65535"
            )
        if self._used and self.protocol_version >= 2:
            raise rpc.RpcError(
                rpc.INVALID_REQUEST, "initialize cannot be repeated after another request"
            )
        served = [version for version in VERSIONS if version <= requested]
        if not served:
            raise rpc.RpcError(
                rpc.UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version",
                {"requested": requested, "supported": list(VERSIONS)},
            )
        self.protocol_version = served[-1]
        raw = params.get("clientCapabilities" if self.protocol_version == 1 else "capabilities")
        self.client_capabilities = capabilities_from(raw if isinstance(raw, dict) else {})
        return wire.initialize_result(self.protocol_version, __version__)

    def reply(
        self,
        req_id: object,
        *,
        result: dict[str, Any] | None = None,
        error: rpc.RpcError | None = None,
    ) -> None:
        """Send a reply carrying the result, or the error as the connection's version reads it."""
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            body["error"] = wire.error(self.protocol_version, error)
        else:
            body["result"] = result if result is not None else {}
        self.notify_raw(body)

    def update(self, acp_session_id: str, update: model.Update) -> None:
        """Send one session update as the connection's version reads it, when it has a form."""
        body = wire.render(self.protocol_version, update)
        if body is not None:
            self.notify_raw(wire.notification(acp_session_id, body))

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
