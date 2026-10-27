# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The terminal side of an OAuth sign-in: the loopback callback and the pasted URL.

This module selects and runs browser, device and paste flows for resolved profiles.
The callback receiver binds a loopback address only, answers one path,
logs nothing and keeps the code in memory; nothing a remote returns is executed.
"""

from __future__ import annotations

import getpass
import html
import http.server
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from urllib import parse

from agent6 import child_env
from agent6.providers import _headers, _oauth_profile, _oauth_protocol, chatgpt_oauth, types
from agent6.ui.cli import _common, _terminal_guard

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

CALLBACK_WAIT_S = 300.0
# A client that connects and sends nothing is dropped after this long; others are served meanwhile.
_CONNECTION_TIMEOUT_S = 10.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


class _Server(http.server.ThreadingHTTPServer):
    """The receiver over IPv4."""

    daemon_threads = True


class _Server6(_Server):
    """The receiver over IPv6, for a `[::1]` redirect."""

    address_family = socket.AF_INET6


class CallbackServer:
    """A one-shot loopback receiver for the OAuth redirect.

    Binds the loopback address on the given port (0 picks a free one) and serves, one
    thread per connection, until the callback path answers this sign-in: a code completes
    it, a state-matching refusal ends it with the refusal. Every other path 404s, a request
    for another host 400s, and neither completes anything. The authorization code is
    held in memory only, never logged.
    """

    def __init__(
        self, state: str, *, port: int, path: str = "/auth/callback", host: str = "127.0.0.1"
    ) -> None:
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
                    code = _oauth_protocol.parse_callback(f"?{parts.query}", state=state)
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

        self._server = (_Server6 if ":" in host else _Server)((host, port), Handler)
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


def open_browser(url: str) -> bool:
    """Hand the sign-in URL to the platform's URL opener, found on PATH.

    `webbrowser` would honour `$BROWSER`, an arbitrary command template, and hand the child
    the whole environment; the opener gets the curated environment and the URL as one
    argument, with nothing a remote supplied.

    Args:
        url: The authorize URL, built from the operator's configuration.

    Returns:
        Whether an opener was started; False leaves the printed URL as the way in.
    """
    if sys.platform == "win32":
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            return False
        startfile(url)  # the shell's own URL handler, no command line involved
        return True
    opener = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
    if opener is None:
        return False
    try:
        subprocess.Popen(  # fixed argv from operator config; on the subprocess allowlist
            [opener, url],
            env=child_env.curated_env(desktop=True),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    return True


def wait_for_redirect(server: CallbackServer, url: str) -> str | None:
    """Open the browser and wait on a bound receiver for the redirect, then close it.

    Args:
        server: The bound receiver the redirect names.
        url: The authorize URL the browser opens.

    Returns:
        The code, or None on timeout, Ctrl-C or a refusal; the caller falls back.
    """
    if not open_browser(url):
        print("(no URL opener found; open the URL above by hand)")
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


def _read_masked_callback(prompt: str) -> str:
    """Read a callback with stars on a terminal, or with getpass elsewhere.

    Args:
        prompt: The text before the hidden input.

    Returns:
        The line as entered, without its newline.

    Raises:
        EOFError: The stream ended or Ctrl-D cancelled input.
        KeyboardInterrupt: Ctrl-C cancelled input.
    """
    if not sys.stdin.isatty() or termios is None or tty is None:
        return getpass.getpass(prompt)
    try:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (OSError, ValueError, termios.error):
        return getpass.getpass(prompt)
    out = _terminal_guard.raw_stream(sys.stderr)
    try:
        print(prompt, end="", file=out, flush=True)
        characters: list[str] = []
        while True:
            char = sys.stdin.read(1)
            if char in ("", "\x04"):
                raise EOFError
            if char == "\x03":
                raise KeyboardInterrupt
            if char in ("\r", "\n"):
                return "".join(characters)
            if char in ("\b", "\x7f"):
                if characters:
                    characters.pop()
                    out.write("\b \b")
            else:
                characters.append(char)
                out.write("*")
            out.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print(file=out, flush=True)


def paste_callback(state: str, *, redirect_uri: str = "") -> str | None:
    """Ask for the callback URL the browser landed on and read its code.

    The URL carries the code; only stars, the host and the length are shown.

    Args:
        state: The state the sign-in started with.
        redirect_uri: The redirect a full URL must match; "" accepts any origin.

    Returns:
        The code; None after printing why none was taken (no input, or a refused URL).
    """
    try:
        pasted = _read_masked_callback(
            "Paste the callback URL the browser landed on (hidden): "
        ).strip()
    except KeyboardInterrupt:
        return None
    except EOFError:
        _common.error("no callback input.")
        return None
    try:
        parts = parse.urlsplit(pasted)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if host and parts.port is not None:
            host += f":{parts.port}"
        print(f"received {len(pasted)} characters{f' for {host}' if host else ''}")
        return _oauth_protocol.parse_callback(pasted, state=state, redirect_uri=redirect_uri)
    except ValueError as exc:
        _common.error(f"{exc}")
        return None


def _chatgpt_device_grant(
    issuer: str, client_id: str, provider: str
) -> _oauth_protocol.TokenGrant | None:
    """Show a short code and poll while the person enters it at the issuer's device page.

    The no-display path: nothing to forward over SSH.

    Args:
        issuer: The OAuth issuer.
        client_id: The client registration.
        provider: The provider name, for the reconnect hint.

    Returns:
        The grant, or None when the issuer has the flow disabled, on an error or on Ctrl-C;
        the caller falls back to pasting the callback URL.
    """
    try:
        device = chatgpt_oauth.start_device_auth(issuer, client_id)
    except types.ProviderError as exc:
        print(f"(device sign-in unavailable: {exc})")
        return None
    if device is None:
        return None
    print(f"On any device, open  {issuer.rstrip('/')}{chatgpt_oauth.DEVICE_VERIFY_PATH}")
    print(f"and enter the code:  {device.user_code}")
    print("(waiting; Ctrl-C to paste the callback URL by hand instead)")
    try:
        return chatgpt_oauth.poll_device_auth(issuer, client_id, device, provider=provider)
    except KeyboardInterrupt:
        print()
        return None
    except types.ProviderError as exc:
        print(f"(device sign-in failed: {exc})")
        return None


def _pick_flow(flow: str, issuer: _oauth_profile.OAuthProfile) -> str | None:
    """Return the configured issuer's way in: browser, device or paste.

    Args:
        flow: The `--flow` value.
        issuer: The resolved sign-in profile.

    Returns:
        The flow, or None after printing why the issuer's settings cannot run it.
    """
    has_code = issuer.authorization_endpoint is not None
    fixed_port = bool(parse.urlsplit(issuer.redirect_uri).port)
    if flow == "auto":
        if has_code and sys.stdin.isatty() and gui_browser_available():
            return "browser"
        if issuer.device_authorization_endpoint is not None and (
            issuer.device_flow != "chatgpt" or sys.stdin.isatty()
        ):
            return "device"
        flow = "paste"
    if flow == "device" and issuer.device_authorization_endpoint is None:
        _common.error("--flow device needs oauth.device_authorization_endpoint in the provider.")
        return None
    if flow in ("browser", "paste") and not has_code:
        _common.error(f"--flow {flow} needs oauth.authorization_endpoint in the provider.")
        return None
    if flow == "paste" and not fixed_port:
        _common.error(
            "pasting the callback needs a fixed port in oauth.redirect_uri (port 0 serves the"
            " local listener only); set one, or add oauth.device_authorization_endpoint."
        )
        return None
    return flow


def _with_port(uri: str, port: int) -> str:
    """Return the redirect with the listener's bound port in place of port 0."""
    parts = parse.urlsplit(uri)
    host = parts.hostname or ""
    netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    return parse.urlunsplit(parts._replace(netloc=netloc))


def _code_grant(
    name: str,
    issuer: _oauth_profile.OAuthProfile,
    headers: _headers.HeaderSet,
    *,
    browser: bool,
    device: bool = False,
) -> _oauth_protocol.TokenGrant | None:
    """Run authorization code with PKCE: the loopback redirect, else the pasted callback.

    Args:
        name: The provider name.
        issuer: The resolved sign-in profile.
        headers: The entry's extra headers; those addressed to `oauth` ride the exchange.
        browser: Open the browser and listen for the redirect first.
        device: Try the ChatGPT device flow before the pasted callback.

    Returns:
        The grant, or None after printing why none was taken.

    Raises:
        ProviderError: The token endpoint was unreachable or refused the code.
    """
    assert issuer.authorization_endpoint is not None
    verifier, challenge = _oauth_protocol.pkce_pair()
    state = secrets.token_urlsafe(24)
    parts = parse.urlsplit(issuer.redirect_uri)
    server: CallbackServer | None = None
    if browser:
        try:
            server = CallbackServer(
                state, port=parts.port or 0, path=parts.path, host=issuer.callback_host
            )
        except OSError as exc:
            print(f"(no local callback: {exc}; paste the callback URL instead)")
    if server is None and not parts.port:
        _common.error(
            "pasting the callback needs a fixed port in oauth.redirect_uri; set one and retry."
        )
        return None
    redirect = (
        issuer.redirect_uri
        if server is None or parts.port
        else _with_port(issuer.redirect_uri, server.port)
    )
    url = _oauth_protocol.authorization_url(
        issuer.authorization_endpoint,
        issuer.client_id,
        redirect_uri=redirect,
        scopes=issuer.scopes,
        state=state,
        challenge=challenge,
        extra=dict(issuer.authorization_params),
    )
    where = f"with your {issuer.label} account" if issuer.label else f"at {issuer.issuer}"
    print(f"Open this URL to sign in {where}:\n\n  {url}\n")
    if device:
        grant = _chatgpt_device_grant(issuer.issuer, issuer.client_id, name)
        if grant is not None:
            return grant
    code = wait_for_redirect(server, url) if server is not None else None
    if code is None:
        code = paste_callback(state, redirect_uri=redirect)
        if code is None:
            return None
    return _oauth_protocol.exchange_code(
        issuer.token_endpoint,
        issuer.client_id,
        code=code,
        verifier=verifier,
        redirect_uri=redirect,
        provider=name,
        label=issuer.label or name,
        encoding=issuer.request_encoding,
        headers=headers.renderer("oauth"),
    )


def _device_grant(
    name: str, issuer: _oauth_profile.OAuthProfile, headers: _headers.HeaderSet
) -> _oauth_protocol.TokenGrant | None:
    """Show the issuer's user code and poll while the person enters it on any device.

    Args:
        name: The provider name.
        issuer: The resolved sign-in profile.
        headers: The entry's extra headers; those addressed to `oauth` ride the requests.

    Returns:
        The grant, or None on Ctrl-C.

    Raises:
        ProviderError: The issuer was unreachable, refused, or the code expired.
    """
    assert issuer.device_authorization_endpoint is not None
    started = _oauth_protocol.start_device_grant(
        issuer.device_authorization_endpoint,
        issuer.client_id,
        scopes=issuer.scopes,
        encoding=issuer.request_encoding,
        label=issuer.label or name,
        headers=headers.renderer("oauth"),
    )
    print(f"On any device, open  {started.verification_uri}")
    print(f"and enter the code:  {started.user_code}")
    if started.verification_uri_complete:
        print(f"(or open {started.verification_uri_complete}, which carries the code)")
    print("(waiting; Ctrl-C to stop)")
    try:
        return _oauth_protocol.poll_device_grant(
            issuer.token_endpoint,
            issuer.client_id,
            started,
            encoding=issuer.request_encoding,
            label=issuer.label or name,
            headers=headers.renderer("oauth"),
        )
    except KeyboardInterrupt:
        print()
        return None


def sign_in_grant(
    name: str, profile: _oauth_profile.OAuthProfile, headers: _headers.HeaderSet, flow: str
) -> _oauth_protocol.TokenGrant | None:
    """Run the selected sign-in flow and return its grant.

    Args:
        name: The provider key for reconnect hints.
        profile: The resolved issuer settings and flow policy.
        headers: The provider's request headers.
        flow: Auto, browser, device or paste.

    Returns:
        The grant, or None after a refusal or cancelled input.

    Raises:
        ProviderError: The issuer refused or could not be reached.
    """
    chosen = _pick_flow(flow, profile)
    if chosen is None:
        return None
    if chosen == "device" and profile.device_flow != "chatgpt":
        return _device_grant(name, profile, headers)
    return _code_grant(
        name, profile, headers, browser=chosen == "browser", device=chosen == "device"
    )
