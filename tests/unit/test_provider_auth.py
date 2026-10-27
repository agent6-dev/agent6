# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Authentication outcomes reach the journal without credentials or premature success."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import threading
import time
from collections.abc import Generator
from typing import Any

import httpx2
import pytest

from agent6 import budget, config, event_log, paths, portable, secret_store
from agent6.app import providers
from agent6.providers import (
    _stream,
    _transport,
    anthropic,
    chatgpt,
    oauth,
    openai,
    responses,
    types,
)
from agent6.viewmodel import fold_transcript

_AT = "issuer-bearer-secret"
_RT = "refresh-token-secret"
_NEW = "renewed-bearer-secret"
_EX = "exchanged-bearer-secret"
_ACCOUNT = "private-account"


def _route(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_format: str = "openai",
    exchange: bool = False,
    expired: bool = False,
    observed: bool = True,
    builder: str = "seat",
) -> tuple[
    openai.OpenAIProvider
    | anthropic.AnthropicProvider
    | responses.ResponsesProvider
    | chatgpt.ChatGPTProvider,
    oauth.OAuthCredential,
    event_log.EventSink,
]:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    entry: dict[str, Any] = {"api_format": api_format, "base_url": "https://models.example/v1"}
    if api_format != "chatgpt":
        entry["auth_style"] = "bearer"
        oauth_settings: dict[str, Any] = {
            "issuer": "https://login.example",
            "client_id": "client",
            "token_endpoint": "https://login.example/token",
            "device_authorization_endpoint": "https://login.example/device",
        }
        if exchange:
            oauth_settings["exchange"] = {"endpoint": "https://exchange.example/token"}
        entry["oauth"] = oauth_settings
    cfg = config.Config.model_validate(
        {"providers": {"work": entry}, "models": {"worker": {"provider": "work", "model": "m"}}}
    )
    bound_entry = cfg.providers["work"]
    assert not isinstance(bound_entry, config.ClaudeCodeProviderEntry)
    secret_store.save_oauth_grant(
        "work",
        secret_store.StoredGrant(
            binding=oauth.binding_fingerprint(bound_entry),
            access_token=_AT,
            refresh_token=_RT,
            issued_at=time.time() - 4000,
            expires_at=1 if expired else time.time() + 3600,
            exchange_access_token="",
            exchange_issued_at=0,
            exchange_expires_at=None,
            account_id=_ACCOUNT,
        ),
    )
    events = event_log.EventSink(tmp_path / "logs.jsonl")
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    sink = types.TranscriptSink(tmp_path / "transcripts")
    if builder == "panel":
        wrapped = providers.build_review_seats(
            cfg, transcript_sink=sink, budget=tracker, n=1, events=events if observed else None
        )[0].provider
    else:
        wrapped = providers.reviewer_seat_provider(
            cfg,
            "summariser",
            transcript_sink=sink,
            budget=tracker,
            events=events if observed else None,
        )
    inner = wrapped.inner if isinstance(wrapped, providers.InstrumentedProvider) else wrapped
    assert isinstance(
        inner,
        (
            openai.OpenAIProvider,
            anthropic.AnthropicProvider,
            responses.ResponsesProvider,
            chatgpt.ChatGPTProvider,
        ),
    )
    assert isinstance(inner.credential, oauth.OAuthCredential)
    return inner, inner.credential, events


def _auth(events: event_log.EventSink) -> list[dict[str, Any]]:
    if not events.path.exists():
        return []
    lines = events.path.read_text().splitlines()
    for line in lines:
        for secret in (_AT, _RT, _NEW, _EX, _ACCOUNT, "https://login.example", "private-error"):
            assert secret not in line
    raw = [json.loads(line) for line in lines]
    return [event_log.payload(event) for event in raw if event["type"] == "provider.auth"]


def _fact(kind: str, outcome: str) -> dict[str, str]:
    return {"provider": "work", "kind": kind, "outcome": outcome}


def _serve(
    monkeypatch: pytest.MonkeyPatch, *answers: httpx2.Response, method: str = "http_request"
) -> None:
    pending = iter(answers)

    def request(*_args: object, **_kwargs: object) -> httpx2.Response:
        return next(pending)

    monkeypatch.setattr(_transport, method, request)


@pytest.mark.parametrize("api_format", ["openai", "anthropic", "responses", "chatgpt"])
@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize("builder", ["seat", "panel"])
def test_refresh_journals_only_the_persisted_or_refused_outcome(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    api_format: str,
    status: int,
    builder: str,
) -> None:
    """Both OAuth profiles and side-call builders emit exactly three safe payload fields."""
    _, holder, events = _route(
        tmp_path, monkeypatch, api_format=api_format, expired=True, builder=builder
    )
    persisted: list[str] = []

    def observe(event: dict[str, Any]) -> None:
        if event["type"] == "provider.auth":
            grant = secret_store.load_oauth_grant("work")
            persisted.append(grant.access_token if grant else "missing")

    events.subscribe(observe)

    def request(*_args: object, **_kwargs: object) -> httpx2.Response:
        assert _auth(events) == []
        return httpx2.Response(
            status, json={"access_token": _NEW, "expires_in": 3600, "error": "private-error"}
        )

    monkeypatch.setattr(_transport, "http_request", request)
    if status == 200:
        assert holder.token() == _NEW
        assert holder.token() == _NEW  # a cached read emits nothing
    else:
        with pytest.raises(types.ProviderError):
            holder.token()
    assert _auth(events) == [_fact("refresh", "success" if status == 200 else "failure")]
    assert persisted == [_NEW if status == 200 else _AT]
    raw = [json.loads(line) for line in events.path.read_text().splitlines()]
    assert fold_transcript(raw) == []


@pytest.mark.parametrize("status", [200, 500])
def test_exchange_journals_its_own_outcome(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    """An exchange reports only after its token is stored or its endpoint refuses."""
    _, holder, events = _route(tmp_path, monkeypatch, exchange=True)
    stored: list[str] = []

    def observe(event: dict[str, Any]) -> None:
        if event["type"] == "provider.auth":
            grant = secret_store.load_oauth_grant("work")
            stored.append(grant.exchange_access_token if grant else "missing")

    events.subscribe(observe)
    _serve(
        monkeypatch,
        httpx2.Response(
            status, json={"access_token": _EX, "expires_in": 3600, "error": "private-error"}
        ),
    )
    if status == 200:
        assert holder.token() == _EX
        assert holder.token() == _EX
    else:
        with pytest.raises(types.ProviderError):
            holder.token()
    assert _auth(events) == [_fact("exchange", "success" if status == 200 else "failure")]
    assert stored == [_EX if status == 200 else ""]


def test_exchange_401_records_each_exchange_and_the_intervening_refresh(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused exchange is a failure even when refreshing the issuer rescues the next one."""
    _, holder, events = _route(tmp_path, monkeypatch, exchange=True)
    _serve(
        monkeypatch,
        httpx2.Response(401, json={"error": "private-error"}),
        httpx2.Response(200, json={"access_token": _NEW, "expires_in": 3600}),
        httpx2.Response(200, json={"access_token": _EX, "expires_in": 3600}),
    )
    assert holder.token() == _EX
    assert _auth(events) == [
        _fact("exchange", "failure"),
        _fact("refresh", "success"),
        _fact("exchange", "success"),
    ]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("statuses", [(401, 200), (401, 401), (401, 403), (403,)])
def test_recovery_waits_for_the_retry_answer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    statuses: tuple[int, ...],
) -> None:
    """Arming a rotation emits no recovery; one retry reports its answer, while a 403 is quiet."""
    provider, _, events = _route(tmp_path, monkeypatch)
    sent: list[str] = []
    _serve(monkeypatch, httpx2.Response(200, json={"access_token": _NEW, "expires_in": 3600}))

    def post(_url: str, *, headers: dict[str, str], **_kw: Any) -> httpx2.Response:
        assert not any(e["kind"] == "recovery" for e in _auth(events))
        sent.append(headers["authorization"])
        status = statuses[len(sent) - 1]
        if status != 200:
            return httpx2.Response(status, json={"error": "private-error"})
        data = {
            "choices": [
                {"message": {"content": "ok"}, "delta": {"content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        if streaming:
            return httpx2.Response(200, text=f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n")
        return httpx2.Response(200, json=data)

    @contextlib.contextmanager
    def stream(_method: str, url: str, **kwargs: Any) -> Generator[httpx2.Response]:
        yield post(url, **kwargs)

    monkeypatch.setattr(_transport, "http_post", post)
    monkeypatch.setattr(_stream, "http_stream", stream)
    if statuses[-1] == 200:
        assert (
            provider.call(
                system="s",
                messages=[],
                text_delta_callback=(lambda _t: None) if streaming else None,
            ).text
            == "ok"
        )
    else:
        with pytest.raises(types.ProviderError):
            provider.call(
                system="s",
                messages=[],
                text_delta_callback=(lambda _t: None) if streaming else None,
            )
    if len(statuses) == 1:
        assert _auth(events) == [] and sent == [f"Bearer {_AT}"]
    else:
        assert _auth(events) == [
            _fact("refresh", "success"),
            _fact("recovery", "success" if statuses[-1] == 200 else "failure"),
        ]
        assert sent == [f"Bearer {_AT}", f"Bearer {_NEW}"]


def test_recovery_reports_a_refusal_inside_a_2xx_body(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2xx error envelope is a refusal, not a successful recovery."""
    provider, _, events = _route(tmp_path, monkeypatch)
    _serve(monkeypatch, httpx2.Response(200, json={"access_token": _NEW, "expires_in": 3600}))
    _serve(
        monkeypatch,
        httpx2.Response(401, json={}),
        httpx2.Response(200, json={"error": {"code": "invalid_api_key"}}),
        method="http_post",
    )
    with pytest.raises(types.ProviderError) as refused:
        provider.call(system="s", messages=[])
    assert refused.value.status_code == 401
    assert _auth(events) == [_fact("refresh", "success"), _fact("recovery", "failure")]


def test_adoption_and_cached_reads_do_not_report_authentication(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling's persisted bearer is not a refresh by this holder."""
    _, holder, events = _route(tmp_path, monkeypatch)
    assert holder.token() == holder.token() == _AT
    assert holder.invalidate(403, _AT) is False
    grant = secret_store.load_oauth_grant("work")
    assert grant is not None
    secret_store.save_oauth_grant("work", dataclasses.replace(grant, access_token=_NEW))
    assert holder.invalidate(401, _AT) is True
    assert _auth(events) == []
    assert holder.token() == _NEW
    assert _auth(events) == []


@pytest.mark.parametrize("exchange", [False, True])
@pytest.mark.parametrize("status", [200, 500])
def test_auth_observer_runs_after_both_locks_are_released(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    exchange: bool,
    status: int,
) -> None:
    """A journal listener can read the holder and take the file lock from another thread."""
    _, holder, events = _route(tmp_path, monkeypatch, exchange=exchange, expired=not exchange)
    unlocked: list[bool] = []
    threads: list[threading.Thread] = []

    def observe(_event: dict[str, Any]) -> None:
        done = threading.Event()

        def read() -> None:
            holder.account_id()
            with portable.locked_file(paths.secrets_path()) as held:
                if held:
                    done.set()

        thread = threading.Thread(target=read, daemon=True)
        threads.append(thread)
        thread.start()
        unlocked.append(done.wait(timeout=1))

    events.subscribe(observe)
    _serve(monkeypatch, httpx2.Response(status, json={"access_token": _NEW, "expires_in": 3600}))
    if status == 200:
        holder.token()
    else:
        with pytest.raises(types.ProviderError):
            holder.token()
    for thread in threads:
        thread.join(timeout=2)
    assert unlocked == [True]


def test_a_holder_built_without_a_journal_observer_is_quiet(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listing and connect holders keep the observer optional."""
    _, holder, events = _route(tmp_path, monkeypatch, observed=False, expired=True)
    _serve(monkeypatch, httpx2.Response(200, json={"access_token": _NEW, "expires_in": 3600}))
    assert holder.token() == _NEW
    assert _auth(events) == []
