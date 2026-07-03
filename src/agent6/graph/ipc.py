# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Length-prefixed JSON IPC over Unix-domain socket.

Wire format (per message, both directions):

    8 ASCII digits decimal length, then exactly that many UTF-8 JSON bytes.

We deliberately avoid newline framing because graph node titles can contain
newlines and we never want a parser to depend on payload content.

Request envelope: ``{"id": <int>, "intent": <intent dict>}`` where ``intent``
is one of the discriminated-union models in `agent6.graph.models`. The server
inspects the ``op`` field to pick the right pydantic class.

Response envelope: ``{"id": <int>, "ok": true, "result": <jsonable>}`` or
``{"id": <int>, "ok": false, "error": "<message>"}``.
"""

from __future__ import annotations

import json
import socket
from typing import Any

_LEN_BYTES = 8
# One cap for both directions. Also protects the 8-digit decimal header, which
# would silently desync framing at 10^8 bytes.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise IpcError(f"message too large to frame: {len(body)} bytes (cap {MAX_MESSAGE_BYTES})")
    header = f"{len(body):0{_LEN_BYTES}d}".encode("ascii")
    sock.sendall(header + body)


def recv_message(sock: socket.socket) -> dict[str, Any] | None:
    header = _recv_exact(sock, _LEN_BYTES)
    if header is None:
        return None
    try:
        length = int(header.decode("ascii"))
    except ValueError as exc:
        raise IpcError(f"bad length header: {header!r}") from exc
    if length < 0 or length > MAX_MESSAGE_BYTES:
        raise IpcError(f"message length out of range: {length}")
    body = _recv_exact(sock, length)
    if body is None:
        raise IpcError("connection closed mid-message")
    result = json.loads(body.decode("utf-8"))
    if not isinstance(result, dict):
        raise IpcError(f"expected JSON object, got {type(result).__name__}")
    return result


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise IpcError("connection closed during read")
        buf.extend(chunk)
    return bytes(buf)


class IpcError(Exception):
    """Wire-protocol level failure (framing / decoding)."""
