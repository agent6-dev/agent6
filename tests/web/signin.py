# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Sign in to a test `agent6 web` server and carry its token."""

from __future__ import annotations

import json
from http import client

# The secret a test server starts with unless the test names its own.
SECRET = "amber-acorn-agile-alder-alert-arrow"


def login(port: int, secret: str = SECRET) -> tuple[int, dict[str, object], dict[str, str]]:
    """POST the secret to `/api/login`.

    Returns:
        The status, the JSON body, and the response headers.
    """
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps({"secret": secret}).encode()
        conn.request("POST", "/api/login", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read()), dict(resp.getheaders())
    finally:
        conn.close()


def token(port: int, secret: str = SECRET) -> str:
    """Return the token a login with the secret earns."""
    status, body, _headers = login(port, secret)
    assert status == 200, body
    return str(body["token"])


def bearer(port: int, secret: str = SECRET) -> str:
    """Return the `Authorization` header value a login with the secret earns."""
    return f"Bearer {token(port, secret)}"


class SignedIn(client.HTTPConnection):
    """A connection to a test server whose every request carries the token of a real login."""

    def __init__(self, port: int, *, timeout: float = 10, secret: str = SECRET) -> None:
        super().__init__("127.0.0.1", port, timeout=timeout)
        self._bearer = bearer(port, secret)

    def putrequest(
        self, method: str, url: str, skip_host: bool = False, skip_accept_encoding: bool = False
    ) -> None:
        """Start a request and add the token header."""
        super().putrequest(method, url, skip_host, skip_accept_encoding)
        self.putheader("Authorization", self._bearer)
