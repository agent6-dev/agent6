# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 connect <name>` at a configured issuer: the three ways in, the stored grant, logout."""

from __future__ import annotations

import dataclasses
import io
import pathlib
import threading
import urllib.request
from typing import Any
from urllib import parse

import httpx2
import pytest

from agent6 import paths, secret_store
from agent6.app import _setup
from agent6.config import Config, OpenAIProviderEntry, load_config
from agent6.models import cache
from agent6.providers import _transport, oauth, types
from agent6.ui.cli import main

_CONFIG = """
[providers.work]
api_format = "openai"
base_url = "https://models.example/v1"

[providers.work.oauth]
issuer = "https://login.example"
client_id = "client-1"
scopes = ["model.invoke"]
authorization_endpoint = "https://login.example/authorize"
device_authorization_endpoint = "https://login.example/device"
token_endpoint = "https://login.example/token"
revocation_endpoint = "https://login.example/revoke"
redirect_uri = "http://127.0.0.1:0/auth/callback"
"""


class _Resp:
    def __init__(self, status: int, body: object) -> None:
        self.status_code = status
        self._body = body
        self.text = str(body)
        self.headers = httpx2.Headers()

    def json(self) -> object:
        return self._body


@pytest.fixture
def iso(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # Enter keeps the block's values
    return tmp_path / "g"


def _write_config(text: str = _CONFIG) -> pathlib.Path:
    gc = paths.global_config_path()
    gc.parent.mkdir(parents=True, exist_ok=True)
    gc.write_text(text, encoding="utf-8")
    return gc


def _entry(gc: pathlib.Path) -> Any:
    return load_config(gc).providers["work"]


def test_the_device_flow_signs_in_headless_and_binds_the_grant(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a terminal, auto takes the device grant; the grant stores bound to the entry."""
    gc = _write_config()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    posts: list[tuple[str, dict[str, str]]] = []
    answers = [
        _Resp(
            200,
            {
                "device_code": "DC",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://login.example/activate",
                "interval": 1,
            },
        ),
        _Resp(200, {"access_token": "AT1", "refresh_token": "RT1", "expires_in": 3600}),
    ]

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        posts.append((url, data))
        return answers.pop(0)

    def fake_get(method: str, url: str, headers: dict[str, str], **_kw: Any) -> httpx2.Response:
        assert headers["authorization"] == "Bearer AT1"
        return httpx2.Response(
            200, json={"data": [{"id": "m1"}, {"id": "m2"}]}, request=httpx2.Request("GET", url)
        )

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_post)
    monkeypatch.setattr(_transport, "http_request", fake_get)
    assert main(["connect", "work"]) == 0
    out = capsys.readouterr().out
    assert "open  https://login.example/activate" in out and "enter the code:  ABCD-EFGH" in out
    assert "Sign-in validated: 2 models listed." in out
    assert "DC" not in out
    assert posts[0] == (
        "https://login.example/device",
        {"client_id": "client-1", "scope": "model.invoke"},
    )
    assert posts[1][1]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None
    assert (stored.access_token, stored.refresh_token) == ("AT1", "RT1")
    assert stored.binding == oauth.binding_fingerprint(_entry(gc))
    assert gc.read_text(encoding="utf-8") == _CONFIG  # the block already held its values


def test_the_paste_flow_sends_the_fixed_redirect_and_exchanges_the_code(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(_CONFIG.replace("http://127.0.0.1:0/auth/callback", "http://127.0.0.1:8765/cb"))
    monkeypatch.setattr("agent6.ui.cli._oauth.secrets.token_urlsafe", lambda n=24: "STATE1")
    monkeypatch.setattr("sys.stdin", io.StringIO("http://127.0.0.1:8765/cb?code=C1&state=STATE1\n"))
    exchanges: list[dict[str, str]] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        exchanges.append({"url": url, **data})
        return _Resp(200, {"access_token": "AT1", "refresh_token": "RT1", "expires_in": 3600})

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_post)
    assert main(["connect", "work", "--flow", "paste", "--no-verify"]) == 0
    out = capsys.readouterr().out
    url = next(line.strip() for line in out.splitlines() if line.strip().startswith("https://"))
    query = dict(parse.parse_qsl(parse.urlsplit(url).query))
    assert url.startswith("https://login.example/authorize?")
    assert query["redirect_uri"] == "http://127.0.0.1:8765/cb" and query["scope"] == "model.invoke"
    assert query["code_challenge_method"] == "S256" and query["state"] == "STATE1"
    assert exchanges[0]["url"] == "https://login.example/token"
    assert exchanges[0]["grant_type"] == "authorization_code" and exchanges[0]["code"] == "C1"
    assert exchanges[0]["redirect_uri"] == "http://127.0.0.1:8765/cb"
    assert "Checking the sign-in" not in out
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None and stored.refresh_token == "RT1"


def test_a_pasted_callback_with_another_state_stores_nothing(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(_CONFIG.replace("http://127.0.0.1:0/auth/callback", "http://127.0.0.1:8765/cb"))
    monkeypatch.setattr("agent6.ui.cli._oauth.secrets.token_urlsafe", lambda n=24: "STATE1")
    monkeypatch.setattr("sys.stdin", io.StringIO("http://127.0.0.1:8765/cb?code=C1&state=EVIL\n"))

    def never(*_a: object, **_k: object) -> None:
        raise AssertionError("no token request may leave")

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", never)
    assert main(["connect", "work", "--flow", "paste"]) == 2
    assert secret_store.load_oauth_grant("work") is None


def test_the_browser_flow_names_the_bound_port_in_the_redirect(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Port 0 takes a free port before the authorization starts; the exchange names that port."""
    _write_config()
    monkeypatch.setattr("agent6.ui.cli._oauth.gui_browser_available", lambda: True)
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        query = dict(parse.parse_qsl(parse.urlsplit(url).query))
        callback = f"{query['redirect_uri']}?code=C9&state={query['state']}"

        def land() -> None:
            with urllib.request.urlopen(callback, timeout=5) as resp:
                assert resp.status == 200

        threading.Thread(target=land, daemon=True).start()
        return True

    monkeypatch.setattr("agent6.ui.cli._oauth.open_browser", fake_open)
    exchanges: list[dict[str, str]] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        exchanges.append(data)
        return _Resp(200, {"access_token": "AT1", "refresh_token": "RT1", "expires_in": 3600})

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_post)
    assert main(["connect", "work", "--no-verify"]) == 0
    redirect = parse.urlsplit(exchanges[0]["redirect_uri"])
    assert redirect.hostname == "127.0.0.1" and redirect.port not in (None, 0)
    assert redirect.path == "/auth/callback" and exchanges[0]["code"] == "C9"
    assert opened and parse.urlsplit(opened[0]).netloc == "login.example"
    assert secret_store.load_oauth_grant("work") is not None


@pytest.mark.parametrize(
    ("flow", "config", "word"),
    [
        (
            "device",
            _CONFIG.replace('device_authorization_endpoint = "https://login.example/device"\n', ""),
            "device_authorization_endpoint",
        ),
        (
            "browser",
            _CONFIG.replace('authorization_endpoint = "https://login.example/authorize"\n', ""),
            "authorization_endpoint",
        ),
        ("paste", _CONFIG, "fixed port"),
        # auto without a terminal, no device endpoint, port 0: the paste fallback cannot run
        (
            "auto",
            _CONFIG.replace('device_authorization_endpoint = "https://login.example/device"\n', ""),
            "fixed port",
        ),
    ],
)
def test_a_flow_the_issuer_settings_cannot_run_refuses_by_name(
    iso: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flow: str,
    config: str,
    word: str,
) -> None:
    _write_config(config)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["connect", "work", "--flow", flow]) == 2
    assert word in capsys.readouterr().err
    assert secret_store.load_oauth_grant("work") is None


def test_logout_revokes_at_the_configured_endpoint_and_removes_the_grant(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gc = _write_config()
    entry = _entry(gc)
    secret_store.save_oauth_grant(
        "work",
        secret_store.StoredGrant(
            binding=oauth.binding_fingerprint(entry),
            access_token="AT1",
            refresh_token="RT1",
            issued_at=1.0,
            expires_at=None,
            exchange_access_token="",
            exchange_issued_at=0.0,
            exchange_expires_at=None,
        ),
    )
    revoked: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> _Resp:
        revoked.append({"url": url, "method": method, **kwargs})
        return _Resp(200, {})

    monkeypatch.setattr(_transport, "http_request", fake_request)
    assert main(["connect", "work", "--logout"]) == 0
    assert revoked[0]["url"] == "https://login.example/revoke"
    body = dict(parse.parse_qsl(revoked[0]["content"].decode("ascii")))
    assert body == {"token": "RT1", "token_type_hint": "refresh_token", "client_id": "client-1"}
    assert revoked[0]["method"] == "POST"
    assert secret_store.load_oauth_grant("work") is None
    out = capsys.readouterr().out
    assert "Revoked the sign-in for 'work' at https://login.example/revoke" in out
    assert "Removed stored credentials" in out

    # Without a revocation endpoint the grant is removed locally and nothing is dialled.
    _write_config(_CONFIG.replace('revocation_endpoint = "https://login.example/revoke"\n', ""))
    secret_store.save_secret("work", "unused")
    revoked.clear()
    assert main(["connect", "work", "--logout"]) == 0
    assert revoked == [] and secret_store.resolve_api_key("work", None) is None


def test_the_run_preflight_wants_a_bound_grant(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight names the reconnect for a missing or rebound grant, and lists through it."""
    cfg = Config.model_validate(
        {
            **load_config(_write_config()).model_dump(mode="json", exclude_defaults=True),
            "models": {"worker": {"provider": "work", "model": "m1"}},
        }
    )
    entry = cfg.providers["work"]
    assert isinstance(entry, OpenAIProviderEntry)
    assert _setup.check_provider_keys(cfg) == (
        "No sign-in stored for provider 'work'; run `agent6 connect work` to sign in again."
    )
    grant = secret_store.StoredGrant(
        binding="stale",
        access_token="AT1",
        refresh_token="",
        issued_at=1.0,
        expires_at=None,
        exchange_access_token="",
        exchange_issued_at=0.0,
        exchange_expires_at=None,
    )
    secret_store.save_oauth_grant("work", grant)
    refusal = _setup.check_provider_keys(cfg)
    assert (
        refusal is not None and "other provider settings" in refusal and "connect work" in refusal
    )
    secret_store.save_oauth_grant(
        "work", dataclasses.replace(grant, binding=oauth.binding_fingerprint(entry))
    )
    listed: list[tuple[str, str | None]] = []

    def fake_list(name: str, _entry: object, key: str | None, **_kw: Any) -> list[str]:
        listed.append((name, key))
        return ["m1"]

    monkeypatch.setattr(cache, "list_models", fake_list)
    assert _setup.check_provider_keys(cfg) is None
    assert listed == [("work", None)]


def test_model_set_names_the_missing_or_rebound_sign_in(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gc = _write_config()

    def listed(*_a: object, **_k: object) -> list[str]:
        return ["m1"]

    monkeypatch.setattr(cache, "list_models", listed)
    assert main(["model", "worker", "work/m1"]) == 0
    err = capsys.readouterr().err
    cause = _setup.check_provider_keys(load_config(gc))
    assert cause is not None and cause in err
    assert "agent6 connect work" in err

    grant = secret_store.StoredGrant(
        binding="stale",
        access_token="AT1",
        refresh_token="",
        issued_at=1.0,
        expires_at=None,
        exchange_access_token="",
        exchange_issued_at=0.0,
        exchange_expires_at=None,
    )
    secret_store.save_oauth_grant("work", grant)
    assert main(["model", "worker", "work/m1"]) == 0
    assert "other provider settings" in capsys.readouterr().err

    secret_store.save_oauth_grant(
        "work", dataclasses.replace(grant, binding=oauth.binding_fingerprint(_entry(gc)))
    )
    assert main(["model", "worker", "work/m1"]) == 0
    assert "WARNING" not in capsys.readouterr().err


def test_logout_does_not_revoke_a_grant_made_under_other_settings(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rebound grant's token never goes to the current config's revocation endpoint."""
    _write_config()
    secret_store.save_oauth_grant(
        "work",
        secret_store.StoredGrant(
            binding="made-elsewhere",
            access_token="AT1",
            refresh_token="RT1",
            issued_at=1.0,
            expires_at=None,
            exchange_access_token="",
            exchange_issued_at=0.0,
            exchange_expires_at=None,
        ),
    )

    def never(*_a: object, **_k: object) -> None:
        raise AssertionError("no token may leave")

    monkeypatch.setattr(_transport, "http_request", never)
    assert main(["connect", "work", "--logout"]) == 0
    output = capsys.readouterr()
    assert "other provider settings" in output.err and "locally only" in output.out
    assert secret_store.load_oauth_grant("work") is None


def test_logout_removes_the_grant_when_the_revocation_read_is_refused(
    iso: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A revocation body over the cap is a warning, not a kept grant."""
    gc = _write_config()
    secret_store.save_oauth_grant(
        "work",
        secret_store.StoredGrant(
            binding=oauth.binding_fingerprint(_entry(gc)),
            access_token="AT1",
            refresh_token="RT1",
            issued_at=1.0,
            expires_at=None,
            exchange_access_token="",
            exchange_issued_at=0.0,
            exchange_expires_at=None,
        ),
    )

    def oversized(*_a: object, **_k: object) -> None:
        raise types.ProviderError("provider response exceeded 1048576 bytes; refusing to buffer it")

    monkeypatch.setattr(_transport, "http_request", oversized)
    assert main(["connect", "work", "--logout"]) == 0
    assert "revocation failed" in capsys.readouterr().err
    assert secret_store.load_oauth_grant("work") is None
