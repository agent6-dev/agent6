# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The terminal side of an OAuth sign-in: the loopback callback and the pasted URL.

`agent6 connect` picks the flow; this module runs the parts that touch the browser and
the terminal. The callback receiver binds a loopback address only, answers one path,
logs nothing and keeps the code in memory; nothing a remote returns is executed.
"""

from __future__ import annotations

import contextlib
import getpass
import html
import http.server
import os
import sys
import threading
import time
import webbrowser
from urllib import parse

from agent6.providers import _oauth_protocol
from agent6.ui.cli import _common

CALLBACK_WAIT_S = 300.0
# A client that connects and sends nothing is dropped after this long; others are served meanwhile.
_CONNECTION_TIMEOUT_S = 10.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


class CallbackServer:
    """A one-shot loopback receiver for the OAuth redirect.

    Binds 127.0.0.1 on the given port (0 picks a free one) and serves, one thread per
    connection, until the callback path answers this sign-in: a code completes it, a
    state-matching refusal ends it with the refusal. Every other path 404s, a request
    for another host 400s, and neither completes anything. The authorization code is
    held in memory only, never logged.
    """

    def __init__(self, state: str, *, port: int, path: str = "/auth/callback") -> None:
        self._code: str | None = None
        self._refusal: str | None = None
        self._got = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            """The one request handler: the callback path, or 404."""

            timeout = _CONNECTION_TIMEOUT_S

            def do_GET(self) -> None:  # BaseHTTPRequestHandler API name
                """Take the authorization code from the callback query, checking its state."""
                host = self.headers.get("host", "")
                name, _, port_text = (
                    host.rpartition(":") if not host.endswith("]") else (host, "", "")
                )
                if name.lower() not in _LOOPBACK_HOSTS or port_text != str(outer.port):
                    self.send_error(400, "unexpected host")
                    return
                parts = parse.urlsplit(self.path)
                if parts.path != path:
                    self.send_error(404)
                    return
                refusal: str | None = None
                try:
                    code = _oauth_protocol.parse_callback(parts.query, state=state)
                    body = b"<html><body>Signed in. Return to the terminal.</body></html>"
                    status = 200
                except ValueError as exc:
                    # Escaped: a reflected error_description would run script on this origin.
                    body = f"<html><body>{html.escape(str(exc))}</body></html>".encode()
                    status = 400
                    code = None
                    if str(exc).startswith("sign-in was refused"):
                        refusal = str(exc)
                self.send_response(status)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-security-policy", "default-src 'none'")
                self.send_header("x-content-type-options", "nosniff")
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                outer._settle(code, refusal)

            def log_message(self, format: str, *args: object) -> None:
                """Log nothing: the code must not reach the terminal."""
                del format, args  # silent: the code must not reach the terminal

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="agent6-oauth-callback", daemon=True
        )
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound port."""
        return int(self._server.server_address[1])

    @property
    def refusal(self) -> str | None:
        """The refusal this sign-in's callback carried, once one has."""
        return self._refusal

    def _settle(self, code: str | None, refusal: str | None) -> None:
        """Record the first answer to this sign-in; later callbacks change nothing."""
        if self._got.is_set() or (code is None and refusal is None):
            return
        self._code = code
        self._refusal = refusal
        self._got.set()

    def wait(self, timeout_s: float) -> str | None:
        """Wait for this sign-in's answer, polling so Ctrl-C lands promptly.

        Args:
            timeout_s: The longest wait.

        Returns:
            The code; None on timeout or when the answer was a refusal (`refusal` names it).
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._got.wait(0.25):
                return self._code
        return None

    def close(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()


def gui_browser_available() -> bool:
    """Return whether a GUI browser can take the sign-in URL.

    On a display-less Linux box `webbrowser.open` falls back to a console browser, which
    takes over the terminal the sign-in prompt lives on; there the printed URL is the flow.

    Returns:
        True on macOS and Windows; elsewhere, when `DISPLAY` or `WAYLAND_DISPLAY` is set.
    """
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def code_via_callback_server(
    url: str, state: str, *, port: int, path: str = "/auth/callback"
) -> str | None:
    """Open the browser and wait on the loopback for the redirect: the GUI path.

    Args:
        url: The authorize URL the browser opens.
        state: The state the callback must echo.
        port: The loopback port the redirect names.
        path: The callback path the redirect names.

    Returns:
        The code, or None on timeout, Ctrl-C or an unbindable port; the caller falls back.
    """
    try:
        server = CallbackServer(state, port=port, path=path)
    except OSError as exc:
        print(f"(no local callback: port {port} unavailable: {exc})")
        return None
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    print(f"Waiting for the sign-in redirect on localhost:{server.port}")
    print("(Ctrl-C to paste the callback URL by hand instead)")
    try:
        code = server.wait(timeout_s=CALLBACK_WAIT_S)
    except KeyboardInterrupt:
        print()
        return None
    finally:
        server.close()
    if server.refusal:
        _common.error(server.refusal)
    return code


def paste_callback(state: str, *, redirect_uri: str = "") -> str | None:
    """Ask for the callback URL the browser landed on and read its code.

    The URL carries the code, so it is read hidden, like a password.

    Args:
        state: The state the sign-in started with.
        redirect_uri: The redirect a full URL must match; "" accepts any origin.

    Returns:
        The code; None after printing why none was taken (no input, or a refused URL).
    """
    try:
        pasted = getpass.getpass("Paste the callback URL the browser landed on (hidden): ").strip()
    except EOFError:
        _common.error("no callback input.")
        return None
    try:
        return _oauth_protocol.parse_callback(pasted, state=state, redirect_uri=redirect_uri)
    except ValueError as exc:
        _common.error(f"{exc}")
        return None
