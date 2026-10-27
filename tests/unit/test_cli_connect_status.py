# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Live credential checks and token life in `agent6 connect --status`."""

from __future__ import annotations

import dataclasses
import pathlib
import subprocess
import time
import tomllib

import httpx2
import pytest

from agent6 import paths, secret_store
from agent6.config import ChatGPTProviderEntry, OpenAIProviderEntry
from agent6.providers import _oauth_protocol, _transport, oauth
from agent6.ui.cli import _connect_status, main


@pytest.fixture
def status_config(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    path = paths.global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("credential", ["key", "command"])
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (200, "verified"),
        (401, "refused: HTTP 401"),
        (403, "refused: HTTP 403"),
        (503, "refused: HTTP 503"),
        ("connection", "unreachable: connection refused"),
        ("timeout", "unreachable: timed out"),
    ],
)
def test_status_probes_keys_and_command_tokens(
    status_config: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    credential: str,
    response: int | str,
    expected: str,
) -> None:
    """A status row probes once, reports the reply and bounds both command and request time."""
    command = 'token_command = ["mint-token"]\n' if credential == "command" else ""
    status_config.write_text(
        '[providers.work]\napi_format = "openai"\n'
        'base_url = "https://openrouter.ai/api/v1"\n' + command
    )
    secret_store.save_secret("work", "stored-secret")
    commands: list[list[str]] = []
    requests: list[str] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        assert kwargs["timeout"] == 7.0
        return subprocess.CompletedProcess(argv, 0, stdout="command-secret\n", stderr="")

    def request(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> httpx2.Response:
        requests.append(url)
        assert method == "GET" and timeout == 7.0
        token = "command-secret" if credential == "command" else "stored-secret"
        assert headers["authorization"] == f"Bearer {token}"
        if response == "connection":
            raise httpx2.ConnectError("connection refused")
        if response == "timeout":
            raise httpx2.ReadTimeout("timed out")
        assert isinstance(response, int)
        return httpx2.Response(response, json={"data": {}}, request=httpx2.Request(method, url))

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(_transport, "http_request", request)
    assert main(["connect", "--status"]) == 0
    output = capsys.readouterr()
    assert output.out.splitlines()[1].endswith(expected)
    assert output.out.splitlines()[0].split() == [
        "name",
        "api_format",
        "authentication",
        "state",
        "live",
    ]
    assert requests == ["https://openrouter.ai/api/v1/key"]
    assert commands == ([["mint-token"]] if credential == "command" else [])
    assert "stored-secret" not in output.out + output.err
    assert "command-secret" not in output.out + output.err


@pytest.mark.parametrize("kind", ["chatgpt", "issuer", "exchange"])
@pytest.mark.parametrize("response", [200, 401])
def test_status_lists_with_bound_bearer_without_refresh_or_exchange(
    status_config: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    response: int,
) -> None:
    """Even an expired bearer is sent as stored, with no refresh, exchange or cache fallback."""
    api_format = "chatgpt" if kind == "chatgpt" else "openai"
    config = f'[providers.work]\napi_format = "{api_format}"\n'
    if kind != "chatgpt":
        config += (
            'base_url = "https://models.example/v1"\n'
            '[providers.work.oauth]\nissuer = "https://login.example"\n'
            'client_id = "agent6"\ntoken_endpoint = "https://login.example/token"\n'
            'authorization_endpoint = "https://login.example/auth"\n'
        )
    if kind == "exchange":
        config += '[providers.work.oauth.exchange]\nendpoint = "https://login.example/exchange"\n'
    status_config.write_text(config)
    raw = tomllib.loads(config)["providers"]["work"]
    entry = (
        ChatGPTProviderEntry.model_validate(raw)
        if kind == "chatgpt"
        else OpenAIProviderEntry.model_validate(raw)
    )
    grant = oauth.stored_grant_for(
        _oauth_protocol.TokenGrant("access-secret", "refresh-secret", 3600), entry, now=1000
    )
    grant = dataclasses.replace(
        grant,
        account_id="account-7",
        exchange_access_token="exchanged-secret" if kind == "exchange" else "",
        exchange_issued_at=1000 if kind == "exchange" else 0,
        exchange_expires_at=4600 if kind == "exchange" else None,
    )
    secret_store.save_oauth_grant("work", grant)
    before = paths.secrets_path().read_bytes()
    monkeypatch.setattr(time, "time", lambda: 5000.0)
    requests: list[str] = []

    def request(
        method: str, url: str, *, headers: dict[str, str], timeout: float
    ) -> httpx2.Response:
        requests.append(url)
        assert method == "GET" and timeout == 7.0
        token = "exchanged-secret" if kind == "exchange" else "access-secret"
        assert headers["authorization"] == f"Bearer {token}"
        if kind == "chatgpt":
            assert headers["chatgpt-account-id"] == "account-7"
        return httpx2.Response(
            response, json={"data": [], "models": []}, request=httpx2.Request(method, url)
        )

    monkeypatch.setattr(_transport, "http_request", request)
    assert main(["connect", "--status", "work"]) == 0
    output = capsys.readouterr()
    row = " ".join(output.out.splitlines()[1].split())
    expected = "verified" if response == 200 else "refused: HTTP 401"
    assert row.endswith(f"ok account-7 expired {expected}")
    endpoint = (
        "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0"
        if kind == "chatgpt"
        else "https://models.example/v1/models"
    )
    assert requests == [endpoint]
    assert paths.secrets_path().read_bytes() == before
    for secret in ("access-secret", "refresh-secret", "exchanged-secret"):
        assert secret not in output.out + output.err


@pytest.mark.parametrize("signed_in", [True, False])
def test_status_checks_claude_code_login_without_a_model_call(
    status_config: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    signed_in: bool,
) -> None:
    """Claude Code uses the configured binary's login check, never a model invocation."""
    status_config.write_text(
        '[providers.claude]\napi_format = "claude_code"\nbinary = "claude-work"\n'
    )
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(argv)
        assert kwargs["timeout"] == 7.0
        return subprocess.CompletedProcess(
            argv,
            0 if signed_in else 1,
            stdout=b'{"loggedIn": true}' if signed_in else b'{"loggedIn": false}',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert main(["connect", "--status"]) == 0
    row = capsys.readouterr().out.splitlines()[1]
    if signed_in:
        assert row.endswith("verified")
    else:
        assert "refused: Claude Code is not signed in" in row
    assert commands == [["claude-work", "auth", "status", "--json"]]


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ('api_format = "chatgpt"\n', "missing"),
        ('api_format = "openai"\n', "key unset"),
        (
            'api_format = "openai"\ndeployment = "azure"\nauth_style = "none"\n'
            'base_url = "https://azure.example"\nextra_query = {api-version = "2025-01-01"}\n',
            "no /models listing for this deployment",
        ),
    ],
)
def test_status_skips_without_credentials_or_listing(
    status_config: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config: str,
    reason: str,
) -> None:
    """A missing credential or listing never sends a request and gives a skip reason."""
    status_config.write_text("[providers.work]\n" + config)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a skipped provider must not send a request")

    monkeypatch.setattr(_transport, "http_request", forbidden)
    assert main(["connect", "--status"]) == 0
    assert capsys.readouterr().out.splitlines()[1].endswith(f"skipped: {reason}")


@pytest.mark.parametrize(
    ("remaining", "expected"),
    [
        (14367 * 60, "expires in 9 d 23 h"),
        (192 * 60, "expires in 3 h 12 min"),
        (42 * 60, "expires in 42 min"),
        (0, "expired"),
    ],
)
def test_status_token_life_uses_the_largest_two_units(
    monkeypatch: pytest.MonkeyPatch, remaining: int, expected: str
) -> None:
    """A token's life uses days and hours, hours and minutes, minutes, or expired."""
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    grant = oauth.stored_grant_for(
        _oauth_protocol.TokenGrant("access-secret", "refresh-secret", remaining), entry, now=1000
    )
    secret_store.save_oauth_grant("work", grant)
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    _, state = _connect_status._credential_status("work", entry, secret_store.load_secrets())
    assert state == f"ok {expected}"
