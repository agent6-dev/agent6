# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""JSON-RPC 2.0 over stdio, and the `initialize` handshake.

Framing is line-delimited JSON with a bounded read, the same shape
`ui/mcp_server.py` uses and for the same reason: an unbounded `readline`
buffers a whole line before any size check, so a runaway client could exhaust
memory before the cap could refuse it. The dispatch is NOT shared -- different
protocol, different methods.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from agent6 import __version__
from agent6.app.run import FrontendCapabilities

# The ACP version this front-end speaks. Negotiation is bilateral: the client
# sends the newest it supports, we answer with this, and the client disconnects
# if it cannot live with the answer.
PROTOCOL_VERSION = 1
# 4 MiB, mirroring the MCP server's cap. A prompt with a large pasted context
# is the legitimate big case; past this the payload is dropped, not buffered.
MAX_LINE_BYTES = 1 << 22

# JSON-RPC 2.0 reserved codes, the only ones this front-end originates.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """A JSON-RPC error to return to the client, rather than a crash."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def capabilities_from(client: dict[str, Any]) -> FrontendCapabilities:
    """What the CLIENT said it can do, as the seam every front-end declares.

    The whole reason `FrontendCapabilities` exists: an editor that cannot show
    a permission prompt must never be asked one, and a surface that knows what
    it cannot do never offers it. An absent capability reads as absent, so a
    client that says nothing gets the cautious answer rather than the generous
    one.
    """
    fs = client.get("fs") if isinstance(client.get("fs"), dict) else {}
    terminal = bool(client.get("terminal"))
    return FrontendCapabilities(
        # The editor renders our session/update notifications.
        live_view=True,
        # session/request_permission is required of every ACP client, so a
        # connected one can always be asked.
        can_ask=True,
        # A prompt into a live session is how ACP steers.
        can_steer=True,
        # Sibling sessions need somewhere to put them; without a terminal or
        # filesystem capability the client has nowhere to show one.
        can_spawn=terminal or bool(fs),
    )


@dataclass
class ACPServer:
    """One ACP connection. Owns the framing; the methods live beside it."""

    stdin: BinaryIO
    stdout: BinaryIO
    client_capabilities: FrontendCapabilities | None = None
    _handlers: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._handlers = {"initialize": self._initialize}

    def serve(self) -> None:
        """Read messages until EOF. Requests are answered; notifications are
        acted on and not answered, per JSON-RPC."""
        while True:
            line = self.stdin.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            if len(line) > MAX_LINE_BYTES:
                # Drain the rest of the oversized line in bounded chunks and
                # drop the whole payload: refusing beats buffering it.
                while line and not line.endswith(b"\n"):
                    line = self.stdin.readline(MAX_LINE_BYTES + 1)
                continue
            if not line.strip():
                continue
            self._handle(line)

    def _handle(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            # No id to answer against, so this is the one case with no reply.
            return
        if not isinstance(message, dict):
            return
        req_id = message.get("id")
        method = message.get("method")
        raw = message.get("params")
        params = raw if isinstance(raw, dict) else {}
        if not isinstance(method, str):
            if req_id is not None:
                self._reply(req_id, error=(INVALID_REQUEST, "no method"))
            return
        handler = self._handlers.get(method)
        if handler is None:
            if req_id is not None:  # a notification we do not know is ignorable
                self._reply(req_id, error=(METHOD_NOT_FOUND, f"unknown method: {method!r}"))
            return
        try:
            result = handler(params)
        except RpcError as exc:
            if req_id is not None:
                self._reply(req_id, error=(exc.code, exc.message))
            return
        except Exception as exc:  # a handler bug must not kill the connection
            if req_id is not None:
                self._reply(req_id, error=(INTERNAL_ERROR, f"{type(exc).__name__}"))
            return
        if req_id is not None:
            self._reply(req_id, result=result)

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        raw = params.get("clientCapabilities")
        self.client_capabilities = capabilities_from(raw if isinstance(raw, dict) else {})
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {
                # `session/load` is what v2 reorganises, and resume is where
                # agent6 has the most of its own semantics. Absent, not half.
                "loadSession": False,
                "promptCapabilities": {"embeddedContext": True},
            },
            "agentInfo": {"name": "agent6", "version": __version__},
            "authMethods": [],
        }

    def _reply(
        self,
        req_id: object,
        *,
        result: dict[str, Any] | None = None,
        error: tuple[int, str] | None = None,
    ) -> None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            body["error"] = {"code": error[0], "message": error[1]}
        else:
            body["result"] = result if result is not None else {}
        self.notify_raw(body)

    def notify_raw(self, body: dict[str, Any]) -> None:
        """Write one message. Encoded lossily on purpose: a lone surrogate in
        model-emitted text would otherwise raise mid-write and desynchronise
        the stream, which is worse than a replacement character."""
        line = json.dumps(body, ensure_ascii=False, default=str) + "\n"
        self.stdout.write(line.encode("utf-8", "replace"))
        self.stdout.flush()


def serve_acp() -> int:
    """Entry point: speak ACP on this process's stdio until EOF."""
    ACPServer(stdin=sys.stdin.buffer, stdout=sys.stdout.buffer).serve()
    return 0
