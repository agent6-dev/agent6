# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The OAuth callback: one state, one code, the right origin, a listener no slow client holds."""

from __future__ import annotations

import http.client
import io
import os
import pty
import select
import signal
import socket
import subprocess
import sys
import termios
import time
import tty
import urllib.error
import urllib.request
from unittest import mock

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
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{REDIRECT}?code=C9&state=S\n"))
    assert _oauth.paste_callback("S", redirect_uri=REDIRECT) == "C9"


def _read_pty_until(master: int, expected: bytes) -> bytes:
    """Read a terminal's output until the expected text arrives, with a bounded wait."""
    seen = b""
    deadline = time.monotonic() + 5
    while expected not in seen and time.monotonic() < deadline:
        if select.select([master], [], [], 0.1)[0]:
            try:
                seen += os.read(master, 4096)
            except OSError:
                break
    assert expected in seen, repr(seen)
    return seen


@pytest.mark.filterwarnings("ignore:This process.*is multi-threaded.*:DeprecationWarning")
@pytest.mark.parametrize("finish", [b"\r", b"\n", b"\x03", b"\x04"])
def test_callback_paste_masks_edits_and_restores_the_terminal(finish: bytes) -> None:
    """A real terminal shows stars and erases, returns the code, and restores on Enter or abort."""
    pasted = f"{REDIRECT}?code=private-code&state=S"
    expected = "private-code" if finish in (b"\r", b"\n") else None
    script = (
        "import termios\n"
        "from agent6.ui.cli import _oauth, _terminal_guard\n"
        "saved = termios.tcgetattr(0)\n"
        "with _terminal_guard.guarded_terminal():\n"
        f"    code = _oauth.paste_callback('S', redirect_uri={REDIRECT!r})\n"
        f"assert code == {expected!r}\n"
        "assert termios.tcgetattr(0) == saved\n"
        "print('RESULT_OK restored', flush=True)\n"
    )
    try:
        probe_master, probe_slave = os.openpty()
    except OSError as exc:
        pytest.skip(f"pty unavailable: {exc}")
    os.close(probe_master)
    os.close(probe_slave)
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - exec isolates the child from pytest's streams
        os.execl(sys.executable, sys.executable, "-c", script)
    reaped = False
    try:
        output = _read_pty_until(master, b"): ")
        os.write(master, pasted.encode())
        output += _read_pty_until(master, b"*" * len(pasted))
        os.write(master, b"X\x7fY\b")
        output += _read_pty_until(master, b"*\b \b*\b \b")
        os.write(master, finish)
        output += _read_pty_until(master, b"RESULT_OK restored")
        _, status = os.waitpid(pid, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
        assert output.count(b"*") == len(pasted) + 2
        assert b"private-code" not in output and b"?code=" not in output
        if expected:
            assert f"received {len(pasted)} characters for 127.0.0.1:1455".encode() in output
        else:
            assert b"received" not in output
    finally:
        if not reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        os.close(master)


@pytest.mark.parametrize(
    ("pasted", "host"),
    [
        (f"{REDIRECT}?code=private-code&state=S", " for 127.0.0.1:1455"),
        ("code=private-code&state=S", ""),
    ],
)
def test_callback_paste_non_tty_reads_one_line_and_confirms_without_the_code(
    pasted: str, host: str
) -> None:
    """Off a terminal the paste is one line of stdin, and the confirmation names no code."""
    script = (
        "import sys\n"
        "from agent6.ui.cli import _oauth\n"
        f"assert _oauth.paste_callback('S', redirect_uri={REDIRECT!r}) == 'private-code'\n"
        "assert sys.stdin.readline() == 'next-line\\n'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=f"{pasted}\nnext-line\n",
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert f"received {len(pasted)} characters{host}" in output
    assert "private-code" not in output and "?code=" not in output


def test_callback_paste_cancels_on_interrupt_off_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C while reading the pasted line cancels the sign-in instead of escaping the prompt."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    class _Interrupting(io.StringIO):
        def readline(self, size: int = -1) -> str:
            raise KeyboardInterrupt

    monkeypatch.setattr("sys.stdin", _Interrupting())
    try:
        assert _oauth.paste_callback("S") is None
    except KeyboardInterrupt:
        pytest.fail("the callback prompt let Ctrl-C escape")


class _TerminalInput(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 99


@pytest.mark.parametrize("finish", ["\r", "\n", "\x03", "\x04", ""])
def test_callback_paste_reader_masks_characters_and_restores_on_every_exit(
    monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    """The reader masks Unicode characters, erases both backspaces, and restores on EOF or abort."""
    from agent6.ui.cli import _terminal_guard

    pasted = f"{REDIRECT}?code=private-code&state=S"
    stream = _TerminalInput("\b\x7f" + pasted + "é\bX\x7f" + finish)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(sys, "stderr", _terminal_guard.ScrubbedStream(output))
    saved: list[object] = [1, 2, 3, 4, 5, 6, []]
    monkeypatch.setattr(termios, "tcgetattr", mock.Mock(return_value=saved))
    cbreak = mock.Mock()
    restore = mock.Mock()
    monkeypatch.setattr(tty, "setcbreak", cbreak)
    monkeypatch.setattr(termios, "tcsetattr", restore)
    monkeypatch.setattr(
        "getpass.getpass", mock.Mock(side_effect=AssertionError("unexpected fallback"))
    )
    code = _oauth.paste_callback("S", redirect_uri=REDIRECT)
    assert code == ("private-code" if finish in ("\r", "\n") else None)
    cbreak.assert_called_once_with(99)
    restore.assert_called_once_with(99, termios.TCSADRAIN, saved)
    assert "*" * len(pasted) + "*\b \b*\b \b" in output.getvalue()
    assert "private-code" not in output.getvalue()


@pytest.mark.parametrize("failure", ["attributes", "cbreak"])
def test_callback_paste_uses_getpass_when_terminal_mode_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    """An unavailable terminal mode falls back to getpass without losing the confirmation."""
    monkeypatch.setattr(sys, "stdin", _TerminalInput())
    attrs = mock.Mock(return_value=[1, 2, 3, 4, 5, 6, []])
    cbreak = mock.Mock()
    (attrs if failure == "attributes" else cbreak).side_effect = termios.error("unavailable")
    monkeypatch.setattr(termios, "tcgetattr", attrs)
    monkeypatch.setattr(tty, "setcbreak", cbreak)
    monkeypatch.setattr(termios, "tcsetattr", mock.Mock(side_effect=termios.error("unavailable")))
    pasted = f"{REDIRECT}?code=private-code&state=S"
    monkeypatch.setattr("getpass.getpass", mock.Mock(return_value=pasted))
    assert _oauth.paste_callback("S", redirect_uri=REDIRECT) == "private-code"
    output = capsys.readouterr()
    assert f"received {len(pasted)} characters for 127.0.0.1:1455" in output.out
    assert "private-code" not in output.out + output.err


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
