# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Talk to an MCP server the OPERATOR is running, over HTTP.

The stdio transport has agent6 spawn the server, which means agent6 owns its
environment, its lifetime and its confinement. For a server that wants a
browser, a device or a network of its own, that is the wrong owner: the
operator runs it however they like -- their container, their sandbox, their
credentials -- and agent6 only connects.

One request, one response: JSON-RPC over POST. What that buys in simplicity it
does not buy in trust, so this side carries the same defences the `fetch` tool
does -- no compression, a streamed cap, a total deadline -- plus the id check
the stdio reader has always applied. A response is only this call's answer if
it says so.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx2

# The same bound the stdio reader applies, and applied the same way: while the
# body arrives, not after. `response.content` materializes first, so a 400 MiB
# body reached 849 MiB of RSS before the check, and a 1 MiB gzip bomb reached
# 2 GiB -- enough to OOM the process that owns the run and the provider keys.
MAX_BODY_BYTES = 8 << 20
# The version agent6 negotiates in `initialize`, echoed on every later request
# as the spec requires.
PROTOCOL_VERSION = "2024-11-05"


class MCPHttpError(Exception):
    """The server could not be reached, or answered with something unusable."""


@dataclass(frozen=True, slots=True)
class HttpTransport:
    """A connection to one operator-run MCP server."""

    name: str
    url: str
    # The env var holding the bearer token, named in config. The VALUE is read
    # here and never logged, never written to a transcript, and never part of
    # an error message.
    token_env: str = ""
    # Echoed on every request after the server assigns one.
    session_id: str = ""

    def _auth(self) -> str:
        """The bearer header value, or "" -- refusing a token that cannot be
        one. A stray CR (a token file with CRLF endings) makes the HTTP layer
        raise with the header VALUE in its message, and that message reaches
        stderr, the launch log and the model's context."""
        token = os.environ.get(self.token_env, "") if self.token_env else ""
        if not token:
            return ""
        if any(ch in token for ch in "\r\n\x00") or not token.isprintable():
            raise MCPHttpError(
                f"the token in ${self.token_env} is not a usable header value"
                " (it contains a newline or a control character)"
            )
        return f"Bearer {token}"

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            # Streamable HTTP: a server may answer with either.
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": PROTOCOL_VERSION,
            # No compression: the cap below counts what ARRIVES, and a decoded
            # stream would let a small body expand past it before any check.
            "accept-encoding": "identity",
        }
        if auth := self._auth():
            headers["authorization"] = auth
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers

    def send(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any] | None:
        """POST one JSON-RPC message; return the response, or None for a
        notification the server acknowledged with no body.

        `trust_env=False`: an ambient `HTTP_PROXY` would otherwise capture this
        connection -- loopback included, since httpx has no implicit bypass --
        sending the bearer token to the proxy in cleartext while the operator's
        own server received nothing.
        """
        deadline = time.monotonic() + timeout_s
        body = bytearray()
        try:
            with (
                httpx2.Client(timeout=timeout_s, follow_redirects=False, trust_env=False) as client,
                client.stream(
                    "POST",
                    self.url,
                    headers=self._headers(),
                    content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                ) as response,
            ):
                if response.status_code >= 400:
                    raise MCPHttpError(f"server {self.name!r} returned HTTP {response.status_code}")
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BODY_BYTES:
                        raise MCPHttpError(
                            f"server {self.name!r} sent more than {MAX_BODY_BYTES} bytes"
                        )
                    if time.monotonic() > deadline:
                        # A total deadline, not the per-operation one httpx
                        # applies: a server dribbling a byte at a time never
                        # trips a read timeout and held the run for as long as
                        # it liked.
                        raise MCPHttpError(
                            f"server {self.name!r} was still answering after {timeout_s:g}s"
                        )
        except MCPHttpError:
            raise
        except Exception as exc:
            # Deliberately broad: httpx2.InvalidURL does not derive from
            # HTTPError, so an operator typo in `url` escaped a narrower catch
            # and crashed the run instead of being logged and skipped. The
            # message is the exception's TYPE, never its text, which can quote
            # a rejected header value back at us.
            raise MCPHttpError(f"server {self.name!r} unreachable ({type(exc).__name__})") from None
        if not body.strip():
            return None  # an accepted notification
        message = _parse(bytes(body), name=self.name)
        return message


def _parse(raw: bytes, *, name: str) -> dict[str, Any]:
    """The JSON-RPC message in *raw*, whether it arrived bare or as SSE."""
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    if text.lstrip().startswith(("event:", "data:", "id:", "retry:", ":")):
        text = _sse_data(text, name=name)
    try:
        message = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MCPHttpError(f"server {name!r} sent invalid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise MCPHttpError(f"server {name!r} sent a non-object response")
    return message


def _sse_data(text: str, *, name: str) -> str:
    """The `data` payload of the first SSE event carrying one.

    A real field parser, not a line scan: an event may open with `id:` or
    `retry:` (resumability), may carry `data` across several lines the spec
    says to join with newlines, and its line endings may be CR, LF or CRLF.
    `str.splitlines()` also splits on U+2028/U+2029/U+0085, which are LEGAL
    raw characters inside a JSON string -- so a tool result containing one was
    cut in half, every time, and the model could plant one deliberately.
    """
    data: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith("data:"):
            data.append(line[len("data:") :].removeprefix(" "))
        elif not line.strip() and data:
            break  # end of the first event that carried data
    if not data:
        raise MCPHttpError(f"server {name!r} sent an SSE response with no data")
    return "\n".join(data)
