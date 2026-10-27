# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The OAuth callback: one state, one code, the right origin, a listener no slow client holds."""

from __future__ import annotations

import http.client
import socket
import urllib.error
import urllib.request

import pytest

from agent6.providers import _oauth_protocol
from agent6.ui.cli import _oauth

REDIRECT = "http://127.0.0.1:1455/auth/callback"


@pytest.mark.parametrize(
    ("query", "word"),
    [
        ("state=S&state=S&code=C", "one state"),
        ("state=S", "code"),
        ("state=S&code=C&code=D", "one code"),
        ("state=S&code=C&error=access_denied", "one of"),
    ],
)
def test_a_callback_takes_exactly_one_state_and_one_code(query: str, word: str) -> None:
    with pytest.raises(ValueError, match=word):
        _oauth_protocol.parse_callback(query, state="S")


def test_a_pasted_full_url_must_match_the_redirect() -> None:
    """A URL from another origin or path is not this sign-in's callback, whatever its query."""
    assert (
        _oauth_protocol.parse_callback(
            f"{REDIRECT}?code=C&state=S", state="S", redirect_uri=REDIRECT
        )
        == "C"
    )
    assert _oauth_protocol.parse_callback("code=C&state=S", state="S", redirect_uri=REDIRECT) == "C"
    for wrong in (
        "http://127.0.0.1:1455/other?code=C&state=S",
        "http://localhost:1455/auth/callback?code=C&state=S",
        "https://attacker.example/auth/callback?code=C&state=S",
        "http://127.0.0.1:9/auth/callback?code=C&state=S",
    ):
        with pytest.raises(ValueError, match="redirect"):
            _oauth_protocol.parse_callback(wrong, state="S", redirect_uri=REDIRECT)


def test_a_slow_client_does_not_hold_the_callback_listener() -> None:
    """A connection that sends nothing leaves the real callback free to land."""
    srv = _oauth.CallbackServer("S1", port=0)
    idle = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{srv.port}/auth/callback?code=OK&state=S1", timeout=5
        ) as resp:
            assert resp.status == 200
        assert srv.wait(timeout_s=5.0) == "OK"
    finally:
        idle.close()
        srv.close()


def test_a_request_for_another_host_completes_nothing() -> None:
    """A request whose Host is not this loopback listener is answered 400 and ignored."""
    srv = _oauth.CallbackServer("S1", port=0)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
        conn.request("GET", "/auth/callback?code=OK&state=S1", headers={"Host": "evil.example"})
        assert conn.getresponse().status == 400
        conn.close()
        assert srv.wait(timeout_s=0.6) is None
        with urllib.request.urlopen(
            f"http://127.0.0.1:{srv.port}/auth/callback?code=REAL&state=S1", timeout=5
        ) as resp:
            assert resp.status == 200
        assert srv.wait(timeout_s=5.0) == "REAL"
    finally:
        srv.close()


def test_a_refusal_callback_ends_the_attempt() -> None:
    """A state-matching refusal is this sign-in's answer: the wait ends and names it."""
    srv = _oauth.CallbackServer("S1", port=0)
    try:
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(
                f"http://127.0.0.1:{srv.port}/auth/callback?state=S1&error=access_denied",
                timeout=5,
            )
        assert refused.value.code == 400
        assert srv.wait(timeout_s=5.0) is None
        assert srv.refusal and "access_denied" in srv.refusal
    finally:
        srv.close()


def test_the_pasted_callback_is_read_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pasted URL carries the code, so it is read like a password, never echoed."""

    def no_echo(prompt: str = "") -> str:
        raise AssertionError("input() would echo the callback")

    monkeypatch.setattr("builtins.input", no_echo)
    monkeypatch.setattr(
        "agent6.ui.cli._oauth.getpass.getpass", lambda prompt="": f"{REDIRECT}?code=C9&state=S"
    )
    assert _oauth.paste_callback("S", redirect_uri=REDIRECT) == "C9"


def test_the_browser_opens_through_a_trusted_launcher_with_a_curated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sign-in URL goes to the platform opener found on PATH, not to `$BROWSER`'s command."""
    import subprocess

    launched: list[dict[str, object]] = []

    class _Child:
        pid = 1

    def fake_popen(argv: list[str], **kwargs: object) -> _Child:
        launched.append({"argv": argv, **kwargs})
        return _Child()

    monkeypatch.setenv("BROWSER", "/tmp/evil-browser %s")
    monkeypatch.setenv("PROBE_PROVIDER_KEY", "sk-secret")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    def which_xdg(name: str) -> str | None:
        return "/usr/bin/xdg-open" if name == "xdg-open" else None

    monkeypatch.setattr("shutil.which", which_xdg)
    monkeypatch.setattr("sys.platform", "linux")
    assert _oauth.open_browser("https://login.example/authorize?x=1") is True
    (call,) = launched
    assert call["argv"] == ["/usr/bin/xdg-open", "https://login.example/authorize?x=1"]
    env = call["env"]
    assert isinstance(env, dict) and "BROWSER" not in env and "PROBE_PROVIDER_KEY" not in env
    assert call["start_new_session"] is True

    def which_none(name: str) -> str | None:
        return None

    monkeypatch.setattr("shutil.which", which_none)
    assert _oauth.open_browser("https://login.example/authorize") is False


def test_the_callback_query_is_parsed_as_a_query_even_with_an_unencoded_url_in_it() -> None:
    """An `iss` parameter carrying `https://` must not turn the query into a URL."""
    import urllib.request

    from agent6.ui.cli import _oauth

    srv = _oauth.CallbackServer("S1", port=0)
    try:
        url = (
            f"http://127.0.0.1:{srv.port}/auth/callback?code=OK&state=S1&iss=https://issuer.example"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200
        assert srv.wait(timeout_s=5.0) == "OK"
    finally:
        srv.close()
