# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The terminal side of an OAuth sign-in: the loopback callback and the pasted URL.

`agent6 connect` picks the flow; this module runs the parts that touch the browser and
the terminal. The callback receiver binds a loopback address only, answers one path,
logs nothing and keeps the code in memory; nothing a remote returns is executed.
"""

from __future__ import annotations

import contextlib
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


class CallbackServer:
    """A one-shot loopback receiver for the OAuth redirect.

    Binds 127.0.0.1 on the given port (0 picks a free one) and serves until the callback
    path is hit; every other path 404s. The authorization code is held in memory only,
    never logged.
    """

    def __init__(self, state: str, *, port: int, path: str = "/auth/callback") -> None:
        self._code: str | None = None
        self._got = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            """The one request handler: the callback path, or 404."""

            def do_GET(self) -> None:  # BaseHTTPRequestHandler API name
                """Take the authorization code from the callback query, checking its state."""
                parts = parse.urlsplit(self.path)
                if parts.path != path:
                    self.send_error(404)
                    return
                try:
                    outer._code = _oauth_protocol.parse_callback(parts.query, state=state)
                    body = b"<html><body>Signed in. Return to the terminal.</body></html>"
                    status = 200
                except ValueError as exc:
                    # Escaped: a reflected error_description would run script on this origin.
                    body = f"<html><body>{html.escape(str(exc))}</body></html>".encode()
                    status = 400
                self.send_response(status)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-security-policy", "default-src 'none'")
                self.send_header("x-content-type-options", "nosniff")
                self.send_header("cache-control", "no-store")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if status == 200:
                    outer._got.set()

            def log_message(self, format: str, *args: object) -> None:
                """Log nothing: the code must not reach the terminal."""
                del format, args  # silent: the code must not reach the terminal

        self._server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="agent6-oauth-callback", daemon=True
        )
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound port."""
        return int(self._server.server_address[1])

    def wait(self, timeout_s: float) -> str | None:
        """Wait for the code, polling so Ctrl-C lands promptly.

        Args:
            timeout_s: The longest wait.

        Returns:
            The code, or None on timeout.
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
        return server.wait(timeout_s=CALLBACK_WAIT_S)
    except KeyboardInterrupt:
        print()
        return None
    finally:
        server.close()


def paste_callback(state: str) -> str | None:
    """Ask for the callback URL the browser landed on and read its code.

    Args:
        state: The state the sign-in started with.

    Returns:
        The code; None after printing why none was taken (no input, or a refused URL).
    """
    try:
        pasted = input("Paste the callback URL the browser landed on: ").strip()
    except EOFError:
        _common.error("no callback input.")
        return None
    try:
        return _oauth_protocol.parse_callback(pasted, state=state)
    except ValueError as exc:
        _common.error(f"{exc}")
        return None
