# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Talk to an MCP server the operator is running, over HTTP.

The stdio transport has agent6 spawn the server and own its environment, lifetime and
confinement. A server that wants a browser, a device or a network of its own is run by the
operator however they like, and agent6 only connects.

One request, one response: JSON-RPC over POST, with the defences the `fetch` tool carries (no
compression, a streamed cap, a total deadline) plus the stdio reader's id check.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.tools.http_body import BodyRefusedError, read_capped

# The stdio reader's bound, applied while the body arrives: `response.content` materializes
# first, so a 400 MiB body reaches 849 MiB of RSS before any check and a 1 MiB gzip bomb 2 GiB.
MAX_BODY_BYTES = 8 << 20

# How much of a non-2xx body rides into the error message.
_MAX_ERROR_DETAIL_CHARS = 2048


def _clean_session_id(value: str) -> str:
    """Return a server-assigned session id the transport can echo in a header, or "".

    The value crosses the wire, so it is untrusted like the token in `_auth`: a non-ASCII byte
    makes the HTTP layer raise on the next send with the value in its message (which reaches
    stderr, the launch log and the model's context), and a control character rides into the
    outgoing header. The spec restricts a session id to visible ASCII (0x21-0x7E); anything
    else is dropped, never quoted, and the caller treats the response as stateless.
    """
    if value and all("\x21" <= ch <= "\x7e" for ch in value):
        return value
    return ""


# Negotiated in `initialize` and echoed on every later request, as the spec requires.
PROTOCOL_VERSION = "2024-11-05"


class MCPHttpError(Exception):
    """The server could not be reached, or answered with something unusable."""


class MCPSessionExpiredError(MCPHttpError):
    """A stateful server answered a request carrying this transport's session id with 404.

    That is the spec's signal that it expired the session; the manager re-initializes.
    """


@dataclass(slots=True)
class HttpTransport:
    """A connection to one operator-run MCP server.

    Not frozen: `session_id` is live connection state; the rest is config.

    Attributes:
        name: The server's name in config.
        url: The server's endpoint.
        token_env: The env var holding the bearer token; its value is read here and never
            logged, written to a transcript or quoted in an error.
        httpx_trust_env: Whether httpx honours the ambient HTTP(S)_PROXY; off by default so the
            bearer token never routes to a proxy.
        session_id: The streamable-HTTP session id the server assigns on `initialize`, echoed
            on every later request and cleared on the 404 that means it expired; "" for a
            stateless server.
    """

    name: str
    url: str
    token_env: str = ""
    httpx_trust_env: bool = False
    session_id: str = ""

    def _auth(self) -> str:
        """Return the bearer header value, or "" when no token is configured.

        Raises:
            MCPHttpError: The token cannot be a header value; a stray CR would make the HTTP
                layer raise with the value in its message, which reaches the model's context.
        """
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
            # Compression is declined here and refused by read_capped if sent anyway.
            "accept-encoding": "identity",
        }
        if auth := self._auth():
            headers["authorization"] = auth
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers

    def send(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any] | None:
        """POST one JSON-RPC message and return the server's answer.

        `trust_env` is off by default: an ambient `HTTP_PROXY` would otherwise capture this
        connection, loopback included, sending the bearer token to the proxy.
        `[mcp.servers.<name>].httpx_trust_env` opts a server in.

        Args:
            payload: The JSON-RPC message.
            timeout_s: The total deadline for the request and its body.

        Returns:
            The response message, or None for a notification acknowledged with no body.

        Raises:
            MCPSessionExpiredError: The server answered the session id with 404.
            MCPHttpError: The server is unreachable, answered outside 2xx, or sent a body that
                is compressed, too large, too slow or not JSON-RPC.
        """
        try:
            with (
                httpx2.Client(
                    timeout=timeout_s, follow_redirects=False, trust_env=self.httpx_trust_env
                ) as client,
                client.stream(
                    "POST",
                    self.url,
                    headers=self._headers(),
                    content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                ) as response,
            ):
                if response.status_code == 404 and self.session_id:
                    # Dropped so the transport stops echoing a dead id.
                    self.session_id = ""
                    raise MCPSessionExpiredError(
                        f"server {self.name!r} expired its session (HTTP 404)"
                    )
                deadline = time.monotonic() + timeout_s
                try:
                    body = read_capped(
                        response, cap=MAX_BODY_BYTES, deadline=deadline, timeout_s=timeout_s
                    )
                except BodyRefusedError as exc:
                    raise MCPHttpError(f"server {self.name!r}: {exc}") from exc
                if not 200 <= response.status_code < 300:
                    # A 3xx is no JSON-RPC answer either; the body's own words are kept, bounded.
                    detail = body.decode("utf-8", errors="replace").strip()
                    if len(detail) > _MAX_ERROR_DETAIL_CHARS:
                        detail = detail[:_MAX_ERROR_DETAIL_CHARS] + " …[agent6: truncated]"
                    suffix = f": {detail}" if detail else ""
                    raise MCPHttpError(
                        f"server {self.name!r} returned HTTP {response.status_code}{suffix}"
                    )
                # A stateless server sends no id, so this leaves session_id "".
                assigned = _clean_session_id(response.headers.get("mcp-session-id", ""))
                if assigned:
                    self.session_id = assigned
        except MCPHttpError:
            raise
        except Exception as exc:
            # Broad: httpx2.InvalidURL is no HTTPError. The type only: the text can quote a header.
            raise MCPHttpError(f"server {self.name!r} unreachable ({type(exc).__name__})") from None
        if not body.strip():
            return None  # an accepted notification
        message = _parse(body, name=self.name)
        return message


def _parse(raw: bytes, *, name: str) -> dict[str, Any]:
    """Return the JSON-RPC message in the body, whether it arrived bare or as SSE.

    Raises:
        MCPHttpError: The body is not a JSON object, or an SSE stream without data.
    """
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
    """Return the `data` payload of the first SSE event carrying one.

    An event may open with `id:` or `retry:`, carry `data` across several lines joined with
    newlines, and end lines with CR, LF or CRLF. `str.splitlines()` is avoided: it also splits
    on U+2028, U+2029 and U+0085, which are legal inside a JSON string.

    Raises:
        MCPHttpError: No event carries data.
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
