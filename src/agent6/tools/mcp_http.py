# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Talk to an MCP server the OPERATOR is running, over HTTP.

The stdio transport has agent6 spawn the server, which means agent6 owns its
environment, its lifetime and its confinement. For a server that wants a
browser, a device or a network of its own, that is the wrong owner: the
operator runs it however they like -- their container, their sandbox, their
credentials -- and agent6 only connects.

One request, one response: JSON-RPC over POST, no reader thread and no
pending-id bookkeeping, because HTTP already pairs them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx2

# The same bound the stdio reader applies, for the same reason: a runaway
# server must not be able to buffer an unbounded body into the agent.
MAX_BODY_BYTES = 8 << 20


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

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            # Streamable HTTP: a server may answer with either, and agent6
            # reads the JSON body in both cases.
            "accept": "application/json, text/event-stream",
        }
        token = os.environ.get(self.token_env) if self.token_env else None
        if token:
            headers["authorization"] = f"Bearer {token}"
        return headers

    def send(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any] | None:
        """POST one JSON-RPC message; return the response, or None for a
        notification the server acknowledged with no body."""
        try:
            with httpx2.Client(timeout=timeout_s, follow_redirects=False) as client:
                response = client.post(
                    self.url,
                    headers=self._headers(),
                    content=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                )
        except httpx2.HTTPError as exc:
            raise MCPHttpError(f"server {self.name!r} unreachable: {exc}") from exc
        if response.status_code >= 400:
            raise MCPHttpError(f"server {self.name!r} returned HTTP {response.status_code}")
        body = response.content[: MAX_BODY_BYTES + 1]
        if len(body) > MAX_BODY_BYTES:
            raise MCPHttpError(f"server {self.name!r} sent more than {MAX_BODY_BYTES} bytes")
        if not body.strip():
            return None  # an accepted notification
        try:
            message = json.loads(_json_from(body.decode("utf-8", errors="replace")))
        except (json.JSONDecodeError, ValueError) as exc:
            raise MCPHttpError(f"server {self.name!r} sent invalid JSON: {exc}") from exc
        if not isinstance(message, dict):
            raise MCPHttpError(f"server {self.name!r} sent a non-object response")
        return message


def _json_from(text: str) -> str:
    """The JSON-RPC message in *text*, whether it arrived bare or as SSE.

    Streamable HTTP lets a server answer a single request with an
    `event: message` / `data: {...}` frame instead of a plain body. Reading
    only the first form left every such server looking like it sent garbage.
    """
    if not text.lstrip().startswith(("event:", "data:", ":")):
        return text
    for line in text.splitlines():
        if line.startswith("data:"):
            return line[len("data:") :].strip()
    raise ValueError("an SSE response with no data frame")
