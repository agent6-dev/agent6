# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`[providers.<name>].oauth`: a configured issuer loads, and every unsafe shape refuses by name."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from agent6.config import Config, OAuthConfig, OpenAIProviderEntry, load_config

_WORK = """
[providers.work]
api_format = "openai"
base_url = "https://models.example/v1"

[providers.work.oauth]
issuer = "https://login.example"
client_id = "operator-registered-public-client"
scopes = ["model.invoke"]
authorization_endpoint = "https://login.example/authorize"
device_authorization_endpoint = "https://login.example/device"
token_endpoint = "https://login.example/token"
revocation_endpoint = "https://login.example/revoke"
redirect_uri = "http://127.0.0.1:8765/auth/callback"
"""


def _load(text: str, tmp_path: pathlib.Path) -> Config:
    cfg = tmp_path / "agent6.toml"
    cfg.write_text(text, encoding="utf-8")
    return load_config(cfg)


def test_a_configured_issuer_loads_with_both_flows(tmp_path: pathlib.Path) -> None:
    entry = _load(_WORK, tmp_path).providers["work"]
    assert isinstance(entry, OpenAIProviderEntry)
    oauth = entry.oauth
    assert isinstance(oauth, OAuthConfig)
    assert oauth.flows() == ("authorization_code", "device")
    assert oauth.scopes == ("model.invoke",)
    assert oauth.refresh_skew_s == 60.0 and oauth.exchange is None
    assert oauth.redirect_uri == "http://127.0.0.1:8765/auth/callback"


def test_the_second_exchange_loads_with_its_selectors(tmp_path: pathlib.Path) -> None:
    text = (
        _WORK
        + """
[providers.work.oauth.exchange]
endpoint = "https://credentials.example/session-token"
method = "GET"
access_token_field = "token"
expires_field = "expires_at"
expires_format = "unix_seconds"
"""
    )
    entry = _load(text, tmp_path).providers["work"]
    assert isinstance(entry, OpenAIProviderEntry) and entry.oauth is not None
    oauth = entry.oauth
    assert oauth.exchange is not None
    assert (oauth.exchange.method, oauth.exchange.expires_format) == ("GET", "unix_seconds")
    assert oauth.exchange.access_token_field == "token"


_UNSAFE: list[tuple[Callable[[str], str], str]] = [
    # no way to sign in
    (
        lambda t: t.replace(
            'authorization_endpoint = "https://login.example/authorize"\n', ""
        ).replace('device_authorization_endpoint = "https://login.example/device"\n', ""),
        "authorization_endpoint",
    ),
    # a plaintext endpoint off loopback
    (
        lambda t: t.replace(
            'token_endpoint = "https://login.example/token"',
            'token_endpoint = "http://login.example/token"',
        ),
        "plain http",
    ),
    # userinfo in an endpoint
    (
        lambda t: t.replace("https://login.example/token", "https://u:p@login.example/token"),
        "userinfo",
    ),
    # a redirect to another host, with a query, or to `localhost`
    (
        lambda t: t.replace("http://127.0.0.1:8765/auth/callback", "https://app.example/cb"),
        "redirect_uri",
    ),
    (
        lambda t: t.replace("http://127.0.0.1:8765/auth/callback", "http://127.0.0.1:8765/cb?x=1"),
        "query",
    ),
    (
        lambda t: t.replace(
            "http://127.0.0.1:8765/auth/callback", "http://localhost:8765/auth/callback"
        ),
        "127.0.0.1",
    ),
    # a reserved authorization parameter, in the table or in the endpoint's own query
    (lambda t: t + '\n[providers.work.oauth.authorization_params]\nstate = "fixed"\n', "state"),
    (
        lambda t: t.replace(
            'authorization_endpoint = "https://login.example/authorize"',
            'authorization_endpoint = "https://login.example/authorize?client_id=other"',
        ),
        "client_id",
    ),
    # a plaintext base_url off loopback would carry the grant readable
    (
        lambda t: t.replace(
            'base_url = "https://models.example/v1"', 'base_url = "http://10.0.0.5/v1"'
        ),
        "plain http",
    ),
    # an unknown key
    (lambda t: t + 'client_secret = "nope"\n', "client_secret"),
]


@pytest.mark.parametrize(("change", "word"), _UNSAFE)
def test_an_unsafe_issuer_shape_refuses_at_load(
    change: Callable[[str], str], word: str, tmp_path: pathlib.Path
) -> None:
    text = change(_WORK)
    with pytest.raises(Exception, match=word):
        _load(text, tmp_path)


@pytest.mark.parametrize(
    ("extra", "word"),
    [
        ('token_command = ["get-token"]\n', "token_command"),
        ('api_key_env = "WORK_KEY"\n', "api_key_env"),
        ('auth_style = "x_api_key"\n', "bearer"),
    ],
)
def test_a_configured_issuer_is_the_one_credential_source(
    extra: str, word: str, tmp_path: pathlib.Path
) -> None:
    """Configured OAuth replaces the key sources and rides the bearer header only."""
    text = _WORK.replace(
        'base_url = "https://models.example/v1"\n',
        'base_url = "https://models.example/v1"\n' + extra,
    )
    with pytest.raises(Exception, match=word):
        _load(text, tmp_path)


def test_the_chatgpt_profile_takes_no_custom_issuer(tmp_path: pathlib.Path) -> None:
    text = _WORK.replace(
        'api_format = "openai"\nbase_url = "https://models.example/v1"\n',
        'api_format = "chatgpt"\n',
    )
    with pytest.raises(Exception, match="chatgpt"):
        _load(text, tmp_path)


def test_the_config_reference_documents_every_oauth_field() -> None:
    """The generated reference walks the nested table, so the page says what the code takes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_config", pathlib.Path(__file__).resolve().parents[2] / "docs" / "gen_config.py"
    )
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rows = gen.leaves()
    for leaf in ("issuer", "client_id", "token_endpoint", "redirect_uri", "exchange.endpoint"):
        assert f"providers.<name>.oauth.{leaf}" in rows
