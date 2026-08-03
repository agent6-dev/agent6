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
from agent6.tools.dispatch import ToolDenied, ToolDispatcher
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


def test_a_host_the_operator_never_named_is_asked_about(tmp_path: Path) -> None:
    """The list is the standing approval. A host off it is the operator's call,
    and an absent operator is a no -- the away-mode approver refuses."""
    asked: list[str] = []

    def _deny(prompt: str) -> bool:
        asked.append(prompt)
        return False

    d = ToolDispatcher(root=tmp_path, config=Config(), approver=_deny)
    with pytest.raises(ToolDenied, match="fetch not approved"):
        d.dispatch("fetch", {"url": "https://example.com/x"})
    assert asked == ["Allow fetch: https://example.com/x"]


def test_an_allowed_host_is_never_prompted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.tools import dispatch as dispatch_mod
    from agent6.tools.fetch import Fetched

    def _loud(_prompt: str) -> bool:
        return pytest.fail("an allowed host must not prompt")

    def _checked(_url: str) -> str:
        return "example.com"

    def _fetched(url: str) -> Fetched:
        return Fetched(url=url, status=200, content_type="text/plain", body="hello")

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
