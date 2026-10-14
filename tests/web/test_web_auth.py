# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 web` requires a secret: every state route needs the cookie a login sets."""

from __future__ import annotations

import json
import pathlib
import re
import socket
import threading
from collections.abc import Iterator
from http import client

import pytest

from agent6 import paths
from agent6._data import words
from agent6.ui.cli import main
from agent6.ui.web import auth, page
from agent6.ui.web import server as web_server
from tests.web import signin

_RUN = "brave-oak-AAAAAA"
_ESCAPE = {"key": "sandbox.isolation", "value": "none", "repo": True}


def _serve(cwd: pathlib.Path, secret: str) -> Iterator[int]:
    srv = web_server.WebServer(("127.0.0.1", 0), cwd, "", secret=secret)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server on an ephemeral loopback port, holding one run, started with `signin.SECRET`."""
    run = paths.state_dir(tmp_path) / "sessions" / "runs" / _RUN
    run.mkdir(parents=True)
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    yield from _serve(tmp_path, signin.SECRET)


def _request(
    port: int, method: str, path: str, body: object = None, *, cookie: str = ""
) -> tuple[int, dict[str, str], bytes]:
    """Send one request with no `Origin`, as a local process would.

    Returns:
        The status, the response headers, and the body (unread for an event stream).
    """
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {"Cookie": cookie} if cookie else {}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, payload, headers)
        resp = conn.getresponse()
        stream = "text/event-stream" in resp.getheader("Content-Type", "")
        return resp.status, dict(resp.getheaders()), b"" if stream else resp.read()
    finally:
        conn.close()


def test_a_config_write_with_no_cookie_is_refused_and_changes_nothing(
    port: int, tmp_path: pathlib.Path
) -> None:
    """A local process could switch the sandbox off: a POST with no `Origin` passed every guard.

    The CSRF guard waves a request with no `Origin` through, so `sandbox.isolation = none`
    landed in the repo config from any process that could reach the port.
    """
    status, _headers, body = _request(port, "POST", "/api/config", _ESCAPE)
    assert status == 401
    assert json.loads(body)["login"] is True
    assert not paths.repo_config_path(tmp_path).exists()


def test_the_same_config_write_with_the_cookie_succeeds(port: int, tmp_path: pathlib.Path) -> None:
    """The 401 answers the missing cookie: signed in, the same write sets the config."""
    cookie = signin.cookie(port)
    status, _headers, body = _request(port, "POST", "/api/config", _ESCAPE, cookie=cookie)
    assert (status, json.loads(body)["ok"]) == (200, True)
    assert 'isolation = "none"' in paths.repo_config_path(tmp_path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    [
        "/api/hub",
        "/api/meta",
        "/api/config",
        f"/api/session/{_RUN}",
        f"/api/session/{_RUN}/conversation",
        f"/api/session/{_RUN}/events",
        "/api/machine/nope/events",
        "/api/session/nope",
    ],
)
def test_a_state_route_without_the_cookie_answers_401(port: int, path: str) -> None:
    """Transcripts and state the jail masks were served to any process that asked.

    An event stream answers 401 before its stream headers, and an unknown id answers 401 like
    a known one, so the answer tells an outsider nothing.
    """
    status, headers, body = _request(port, "GET", path)
    assert status == 401
    assert "application/json" in headers["Content-Type"]
    assert json.loads(body)["login"] is True
    signed_in, _headers, _body = _request(port, "GET", path, cookie=signin.cookie(port))
    assert signed_in != 401


def test_a_wrong_secret_is_refused_and_sets_no_cookie(port: int) -> None:
    """A login that does not match answers 403 with the reason and no `Set-Cookie`."""
    status, body, set_cookie = signin.login(port, "amber-acorn-agile-alder-alert-atlas")
    assert status == 403
    assert body["error"]
    assert set_cookie == ""


def test_a_login_through_a_tls_proxy_gets_a_secure_cookie(port: int) -> None:
    """Behind a proxy that terminated HTTPS the cookie carries `Secure`; on loopback it does not."""
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps({"secret": signin.SECRET}).encode()
        headers = {"Content-Type": "application/json", "X-Forwarded-Proto": "https"}
        conn.request("POST", "/api/login", payload, headers)
        resp = conn.getresponse()
        assert resp.status == 200, resp.read()
        set_cookie = resp.getheader("Set-Cookie", "")
    finally:
        conn.close()
    assert set_cookie.endswith("; Secure")
    assert not signin.login(port)[2].endswith("Secure")


def test_the_secret_typed_with_spaces_and_capitals_signs_in(port: int) -> None:
    """A phone keyboard capitalizes and spaces words: both sides normalize before comparing."""
    status, body, set_cookie = signin.login(port, "  Amber Acorn  agile-alder ALERT - arrow ")
    assert (status, body) == (200, {"ok": True})
    name, _, value = set_cookie.split(";", 1)[0].partition("=")
    assert name == f"agent6_web_{port}"
    assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert signin.SECRET not in set_cookie
    attributes = [part.strip() for part in set_cookie.split(";")[1:]]
    assert attributes == ["HttpOnly", "SameSite=Strict", "Path=/", "Max-Age=34560000"]
    assert _request(port, "GET", "/api/hub", cookie=f"{name}={value}")[0] == 200


def test_login_goes_through_the_cross_site_guard(port: int) -> None:
    """A page on another origin cannot sign a browser in: login keeps the `Origin` check."""
    conn = client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(
            "POST",
            "/api/login",
            json.dumps({"secret": signin.SECRET}).encode(),
            {"Content-Type": "application/json", "Origin": "https://evil.example"},
        )
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 403
        assert resp.getheader("Set-Cookie") is None
    finally:
        conn.close()


def test_a_cookie_from_another_secret_is_refused(port: int, tmp_path: pathlib.Path) -> None:
    """The cookie is keyed by the secret: one server's cookie opens no server started with another.

    A restart with a new secret on the same port meets the old cookie under the same name.
    """
    other_secret = "blue-birch-bold-brook-brave-cedar"
    probe = web_server.WebServer(("127.0.0.1", 0), tmp_path, "", secret=other_secret)
    threading.Thread(target=probe.serve_forever, daemon=True).start()
    try:
        other_port = int(probe.server_address[1])
        other_value = signin.cookie(other_port, other_secret).partition("=")[2]
    finally:
        probe.shutdown()
        probe.server_close()
    mine = signin.cookie(port)
    assert other_value != mine.partition("=")[2]
    forged = f"agent6_web_{port}={other_value}"
    assert _request(port, "GET", "/api/hub", cookie=forged)[0] == 401
    assert _request(port, "GET", "/api/hub", cookie=mine)[0] == 200
    # A stable secret keeps a browser signed in: the value depends on the secret alone.
    assert auth.session_token(signin.SECRET) == mine.partition("=")[2]


def test_the_minted_secret_is_three_adjective_noun_pairs() -> None:
    """The printed secret is six wordlist words a phone keyboard types, new on every launch."""
    minted = [auth.mint() for _ in range(20)]
    for secret in minted:
        parts = secret.split("-")
        assert len(parts) == 6
        assert all(word in words.ADJECTIVES for word in parts[0::2]), secret
        assert all(word in words.NOUNS for word in parts[1::2]), secret
        assert auth.normalize(secret) == secret
    assert len(set(minted)) == len(minted)


def test_normalization_is_one_function_for_both_sides() -> None:
    """Case, outer space and any run of spaces or hyphens between words do not matter."""
    assert auth.normalize("Amber Acorn  agile-alder") == "amber-acorn-agile-alder"
    assert auth.normalize(" -amber--ACORN - agile\talder- ") == "amber-acorn-agile-alder"


def test_the_flag_wins_over_the_variable_and_a_short_secret_is_refused() -> None:
    """`--secret` beats `AGENT6_WEB_SECRET`; under 12 normalized characters nothing starts."""
    assert auth.resolve(None, None).origin == "minted"
    assert auth.resolve("Flag Secret Value", "env-secret-value") == auth.Secret(
        "flag-secret-value", "flag"
    )
    assert auth.resolve(None, "env-secret-value") == auth.Secret("env-secret-value", "env")
    short = (("amber acorn", None), (None, "amber-acorn"), ("", "env-secret-value"), (None, ""))
    for flag, env in short:
        with pytest.raises(auth.WebSecretError, match="12 characters") as refusal:
            auth.resolve(flag, env)
        assert "amber" not in str(refusal.value)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> Iterator[list[web_server.WebServer]]:
    """Let `agent6 web` bind and print, then return instead of serving; yields the servers."""
    started: list[web_server.WebServer] = []

    def bind_only(srv: web_server.WebServer) -> None:
        started.append(srv)

    monkeypatch.setattr(web_server.WebServer, "serve_forever", bind_only)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(auth.SECRET_ENV, raising=False)
    yield started


def test_a_launch_prints_the_minted_secret_and_a_url_that_signs_in(
    launch: list[web_server.WebServer], capsys: pytest.CaptureFixture[str]
) -> None:
    """The secret reaches the operator on the terminal alone: once in the URL, once to type."""
    web_port = _free_port()
    assert main(["web", "--port", str(web_port)]) == 0
    err = capsys.readouterr().err.splitlines()
    url = re.fullmatch(
        rf"agent6 web: serving on http://127\.0\.0\.1:{web_port}/#secret=(\S+)  \(Ctrl-C to stop\)",
        err[0],
    )
    assert url, err
    secret = url.group(1)
    assert len(secret.split("-")) == 6
    assert err[1] == (
        f"agent6 web: secret: {secret} (type it once per browser;"
        " --secret or AGENT6_WEB_SECRET sets your own)"
    )
    assert launch[0].session_token == auth.session_token(secret)


def test_an_operator_secret_is_never_printed(
    launch: list[web_server.WebServer],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A secret the operator set stays off the terminal; a flag value gets the process-list note."""
    web_port = _free_port()
    monkeypatch.setenv(auth.SECRET_ENV, "Env Secret Value")
    assert main(["web", "--port", str(web_port)]) == 0
    err = capsys.readouterr().err
    assert "env-secret-value" not in err.lower().replace(" ", "-")
    assert f"http://127.0.0.1:{web_port}  (Ctrl-C to stop)" in err
    assert "#secret" not in err
    assert "AGENT6_WEB_SECRET" in err and "process list" not in err
    assert launch[0].session_token == auth.session_token("env-secret-value")

    assert main(["web", "--port", str(web_port), "--secret", "Flag Secret Value"]) == 0
    err = capsys.readouterr().err
    assert "flag-secret-value" not in err.lower().replace(" ", "-")
    assert "#secret" not in err
    assert "--secret" in err and "process list" in err
    assert launch[1].session_token == auth.session_token("flag-secret-value")


def test_a_short_operator_secret_refuses_to_start(
    launch: list[web_server.WebServer], capsys: pytest.CaptureFixture[str]
) -> None:
    """A guessable secret is refused before anything binds, naming the rule and not the value."""
    assert main(["web", "--port", str(_free_port()), "--secret", "amber acorn"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("REFUSING:") and "12 characters" in err
    assert "amber" not in err
    assert launch == []


_PUBLIC = {
    "GET": {"/", "/manifest.webmanifest", "/sw.js", "/icon.svg", "/favicon.svg"},
    "POST": {"/api/login"},
}


def _routed_paths() -> dict[str, set[str]]:
    """Every literal path the two routers compare against, by method."""
    source = pathlib.Path(web_server.__file__).read_text(encoding="utf-8")
    found: dict[str, set[str]] = {}
    for method, router in (("GET", "def _route("), ("POST", "def _route_post(")):
        body = source[source.index(router) :]
        body = body[: body.index("\n    def ", 1)]
        found[method] = set(re.findall(r'"(/[a-z0-9_./]*)"', body))
    return found


def test_only_the_page_shell_and_login_answer_without_a_cookie(port: int) -> None:
    """A new route is gated unless someone adds it to the public list on purpose.

    The lists hold the page, the assets a browser fetches to show it, and the login itself;
    every other path the routers name, and any path they do not, answers 401.
    """
    assert {"GET": web_server.PUBLIC_GET, "POST": web_server.PUBLIC_POST} == _PUBLIC
    routed = _routed_paths()
    assert _PUBLIC["GET"] <= routed["GET"] and len(routed["GET"]) > 8, routed
    assert _PUBLIC["POST"] <= routed["POST"] and len(routed["POST"]) > 6, routed
    for path in sorted(routed["GET"] | {"/nope", "/api", "/api/login", "/icon.svg/x", "/%2e"}):
        status, _headers, body = _request(port, "GET", path)
        if path in _PUBLIC["GET"]:
            assert (status, bool(body)) == (200, True), path
        else:
            assert status == 401, path
    for path in sorted(routed["POST"] | {"/", "/nope", "/api/session/x/steer"}):
        status, _headers, _body = _request(port, "POST", path, {})
        assert (status == 401) == (path not in _PUBLIC["POST"]), (path, status)


def test_the_page_meets_a_401_in_one_place() -> None:
    """Every fetch and stream of the page leads a signed-out browser to the sign-in form.

    A call past the two helpers would paint an error instead, and a stream the browser gave
    up on would stay frozen on its last frame.
    """
    js = page.CLIENT_JS
    assert js.count("fetch(") == js.count("signedOut(await fetch(") == 2
    assert js.count("new EventSource(") == js.count("es.onerror = () => streamFailed(es, ") == 2
    form = js[js.index("function showLogin") : js.index("async function boot")]
    for attribute in ("autocapitalize: 'none'", "autocorrect: 'off'", "spellcheck: 'false'"):
        assert attribute in form
    assert "autocomplete: 'off'" in form and "input.type = 'text'" in form
    assert "`agent6 web` printed in its terminal" in form
    boot = js[js.index("async function boot") : js.index("function streamFailed")]
    steps = ("postJSON('/api/login'", "history.replaceState(", "route();")
    assert [boot.index(step) for step in steps] == sorted(boot.index(step) for step in steps)
    assert js.rstrip().endswith("boot();")
