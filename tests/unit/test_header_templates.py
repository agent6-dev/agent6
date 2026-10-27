# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`extra_headers` templates: the grammar refuses at load, facts render per request class."""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import httpx2
import pydantic
import pytest

from agent6 import __version__, secret_store
from agent6 import budget as agent6_budget
from agent6.app import _conversations, providers
from agent6.config import Config, OpenAIProviderEntry
from agent6.models import cache
from agent6.providers import ProviderResponse, _headers, _oauth_protocol, _transport, oauth, types

_HEX32 = "0123456789abcdef"


def _entry(headers: dict[str, Any], **extra: Any) -> OpenAIProviderEntry:
    return OpenAIProviderEntry.model_validate(
        {
            "api_format": "openai",
            "base_url": "https://m.example/v1",
            "extra_headers": headers,
            **extra,
        }
    )


def _context(**overrides: Any) -> _headers.RequestContext:
    fields: dict[str, Any] = {
        "session_id": "sid-1",
        "is_first_turn": True,
        "is_subagent": False,
        "initiator": "user",
    }
    return _headers.RequestContext(**{**fields, **overrides})


def test_a_literal_string_stays_literal_and_rides_the_api_classes_only() -> None:
    """Braces in an existing header string are text; the string never reaches an issuer."""
    hs = _headers.HeaderSet.from_config(_entry({"X-A": "{request_id}"}).extra_headers)
    assert hs.render(_headers.plain_facts("inference")) == {"x-a": "{request_id}"}
    assert hs.render(_headers.plain_facts("models")) == {"x-a": "{request_id}"}
    assert hs.render(_headers.plain_facts("oauth")) == {}
    assert hs.render(_headers.plain_facts("exchange")) == {}


@pytest.mark.parametrize(
    ("headers", "word"),
    [
        ({"authorization": "Bearer x"}, "transport owns"),
        ({"Cookie": "a=b"}, "transport owns"),
        ({"API-Key": "x"}, "transport owns"),
        ({"x-t": {"template": "a\nb"}}, "control character"),
        ({"x-t": {"template": "caf\u00e9 {model}"}}, "control character"),
        ({"X-A": "1", "x-a": "2"}, "same header twice"),
        ({"x-a": "bad\r\nvalue"}, "control character"),
        ({"bad name": "v"}, "not a valid header name"),
        ({"x-t": {"template": "{model.attr}"}}, "fact name and nothing else"),
        ({"x-t": {"template": "{model:>5}"}}, "fact name and nothing else"),
        ({"x-t": {"template": "{home}"}}, "no such fact"),
        ({"x-t": {"template": "{model"}}, "never closes"),
        ({"x-t": {"template": "a}b"}}, "never opened"),
        ({"x-t": {"template": "{session_id}", "targets": ["oauth"]}}, "model-call fact"),
        ({"x-t": {"template": "x", "when": "has_images", "targets": ["oauth"]}}, "inference only"),
        ({"x-t": {"template": "x", "targets": ["models", "models"]}}, "repeats"),
        ({"x-t": {"template": "x", "targets": []}}, "at least 1"),
        ({"x-t": {"template": "x", "extra": "y"}}, "extra"),
    ],
)
def test_an_unsendable_header_map_refuses_at_load(headers: dict[str, Any], word: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=word):
        _entry(headers)


def test_every_fact_renders_for_the_class_that_has_it() -> None:
    entry = _entry(
        {
            "x-id": {"template": "{request_id}"},
            "x-ver": {"template": "agent6/{agent6_version}", "targets": ["inference", "models"]},
            "x-sess": {"template": "{session_id}"},
            "x-model": {"template": "{model}"},
            "x-init": {"template": "{initiator}"},
            "x-first": {"template": "{is_first_turn}"},
            "x-sub": {"template": "{is_subagent}"},
            "x-tools": {"template": "{has_tool_results}"},
            "x-vision": {"template": "true", "when": "has_images"},
            "x-brace": {"template": "{{literal}}"},
        }
    )
    hs = _headers.HeaderSet.from_config(entry.extra_headers)
    facts = _headers.call_facts("m1", (True, False), _context()).attempt()
    rendered = hs.render(facts)
    assert rendered.pop("x-id") == facts.request_id and len(facts.request_id) == 32
    assert rendered == {
        "x-ver": f"agent6/{__version__}",
        "x-sess": "sid-1",
        "x-model": "m1",
        "x-init": "user",
        "x-first": "true",
        "x-sub": "false",
        "x-tools": "true",
        "x-brace": "{literal}",
    }
    with_image = _headers.call_facts("m1", (False, True), _context()).attempt()
    assert hs.render(with_image)["x-vision"] == "true"
    listing = hs.render(_headers.plain_facts("models"))
    assert set(listing) == {"x-ver"}  # the model-call facts are not the listing's to render


def test_body_facts_read_the_serialized_request_per_format() -> None:
    assert _headers.openai_body_facts(
        {
            "messages": [
                {"role": "tool", "content": "ok"},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
            ]
        }
    ) == (True, True)
    assert _headers.openai_body_facts({"messages": [{"role": "user", "content": "hi"}]}) == (
        False,
        False,
    )
    assert _headers.anthropic_body_facts(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t",
                            "content": [{"type": "image", "source": {}}],
                        }
                    ],
                }
            ]
        }
    ) == (True, True)
    assert _headers.responses_body_facts(
        {
            "input": [
                {"type": "function_call_output", "call_id": "c", "output": "ok"},
                {"role": "user", "content": [{"type": "input_image", "image_url": "x"}]},
            ]
        }
    ) == (True, True)


def test_every_attempt_mints_its_own_request_id_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential recovery retries the call: two attempts, two ids, two `provider.request`s."""
    hs = _headers.HeaderSet.from_config(
        _entry({"x-id": {"template": "{request_id}"}}).extra_headers
    )
    seen: list[str] = []
    observed: list[_headers.RequestFacts] = []

    class _Unauthorized:
        status_code = 401
        text = "{}"
        headers = httpx2.Headers()

        def json(self) -> dict[str, Any]:
            return {}

    def fake_post(url: str, *, headers: dict[str, str], **_kw: Any) -> _Unauthorized:
        seen.append(headers["x-id"])
        return _Unauthorized()

    class _Cred:
        def token(self) -> str:
            return "t"

        def invalidate(self, status: int = 401) -> bool:
            return True

    monkeypatch.setattr(_transport, "http_post", fake_post)
    call = _transport.ProviderCall(
        api_label="OpenAI",
        api_format="openai",
        url="https://api.test/v1/chat/completions",
        body={"model": "m"},
        timeout_s=5.0,
        api_key="",
        credential=_Cred(),
        transcript_sink=None,
        budget=None,
        model="m",
        facts=_headers.call_facts("m", (False, False), _context(observe=observed.append)),
        build_headers=lambda token, facts: hs.render(facts),
        adapt_400=lambda _s, _t, _b: False,
        adapt_attempts=0,
        require_metered=lambda _d: None,
        parse=lambda _d: _resp(),
    )
    with pytest.raises(types.ProviderError, match="401"):
        call.run()
    assert len(seen) == 2 and seen[0] != seen[1]
    assert [f.request_id for f in observed] == seen
    assert all(f.target == "inference" and f.model == "m" for f in observed)


def _resp() -> ProviderResponse:
    return ProviderResponse(
        text="",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={},
    )


def test_a_conversation_persists_across_a_resume_and_starts_new_on_a_fork(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "s1" / _conversations.FILE_NAME
    path.parent.mkdir()
    first = _conversations.Conversations(path=path)
    ctx = first.context("worker", "work", initiator="user")
    assert ctx.is_first_turn and not ctx.is_subagent and len(ctx.session_id) == 32
    first.completed("worker", "work")
    resumed = _conversations.Conversations(path=path)
    again = resumed.context("worker", "work", initiator="agent")
    assert again.session_id == ctx.session_id and again.is_first_turn is False
    other = resumed.context("review:security", "work", initiator="agent")
    assert other.session_id != ctx.session_id and other.is_first_turn
    forked = _conversations.Conversations(path=tmp_path / "s2" / _conversations.FILE_NAME)
    assert forked.context("worker", "work", initiator="user").session_id != ctx.session_id
    lane = _conversations.Conversations(is_subagent=True)
    assert lane.context("worker", "work", initiator="user").is_subagent
    assert lane.path is None  # a path-less state writes nothing


class _Inner:
    """A provider that records its context and reports one attempt through it."""

    def __init__(self) -> None:
        self.contexts: list[_headers.RequestContext] = []

    def call(self, **kw: Any) -> ProviderResponse:
        context = kw["context"]
        assert isinstance(context, _headers.RequestContext)
        self.contexts.append(context)
        if context.observe is not None:
            context.observe(_headers.call_facts("m", (False, False), context).attempt())
        return _resp()


def _instrumented(inner: _Inner) -> providers.InstrumentedProvider:
    return providers.InstrumentedProvider(
        inner=inner,  # type: ignore[arg-type]
        role="worker",
        model="m",
        provider_name="work",
        events=mock.MagicMock(),
        budget=agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1),
    )


def test_the_instrumented_provider_fills_the_context_and_reports_each_attempt() -> None:
    """A caller names the initiator; the conversation supplies the id and the turn state."""
    inner = _Inner()
    wrapped = _instrumented(inner)
    task = [{"role": "user", "content": "do the task"}]
    wrapped.call(system="s", messages=task, context=_headers.RequestContext.initiated_by("user"))
    wrapped.call(system="s", messages=task, context=_headers.RequestContext.initiated_by("agent"))
    wrapped.call(system="s", messages=task)
    first, second, third = inner.contexts
    assert (first.initiator, first.is_first_turn) == ("user", True)
    assert (second.initiator, second.is_first_turn) == ("agent", False)
    assert third.initiator == "agent" and len(third.session_id) == 32
    assert first.session_id == second.session_id == third.session_id
    events = wrapped.events
    assert isinstance(events, mock.MagicMock)
    requests = [c for c in events.emit.call_args_list if c.args[0] == "provider.request"]
    assert len(requests) == 3
    assert requests[0].kwargs["provider"] == "work" and requests[0].kwargs["target"] == "inference"
    assert len(requests[0].kwargs["request_id"]) == 32
    assert "x-" not in str(requests[0].kwargs)  # no header value rides the event


def test_a_call_outside_a_conversation_keeps_its_model_and_body_facts() -> None:
    entry = _entry(
        {
            "x-sess": {"template": "{session_id}"},
            "x-first": {"template": "{is_first_turn}"},
            "x-init": {"template": "{initiator}"},
            "x-id": {"template": "{request_id}"},
            "x-model": {"template": "{model}"},
            "x-tools": {"template": "{has_tool_results}"},
        }
    )
    hs = _headers.HeaderSet.from_config(entry.extra_headers)
    rendered = hs.render(_headers.call_facts("m", (True, False), None).attempt())
    assert set(rendered) == {"x-id", "x-model", "x-tools"}
    assert (rendered["x-model"], rendered["x-tools"]) == ("m", "true")
    # A partial context names the initiator only; the conversation facts stay out.
    partial = _headers.RequestContext.initiated_by("user")
    rendered = hs.render(_headers.call_facts("m", (False, False), partial).attempt())
    assert set(rendered) == {"x-id", "x-model", "x-tools", "x-init"}
    assert rendered["x-init"] == "user"


def test_a_malformed_conversation_file_starts_empty(tmp_path: pathlib.Path) -> None:
    path = tmp_path / _conversations.FILE_NAME
    path.write_text('{"routes": {"worker@work": {"id": "abc", "completed": "x"}}}')
    ctx = _conversations.Conversations(path=path).context("worker", "work", initiator="agent")
    assert ctx.is_first_turn and ctx.session_id != "abc"


def test_an_oauth_targeted_template_reaches_the_token_endpoint_and_a_literal_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    cfg = Config.model_validate(
        {
            "providers": {
                "work": {
                    "api_format": "openai",
                    "base_url": "https://m.example/v1",
                    "extra_headers": {
                        "x-static": "literal",
                        "user-agent": {
                            "template": "agent6/{agent6_version}",
                            "targets": ["oauth", "exchange"],
                        },
                    },
                    "oauth": {
                        "issuer": "https://login.example",
                        "client_id": "c",
                        "device_authorization_endpoint": "https://login.example/device",
                        "token_endpoint": "https://login.example/token",
                    },
                }
            }
        }
    )
    entry = cfg.providers["work"]
    assert isinstance(entry, OpenAIProviderEntry)
    secret_store.save_oauth_grant(
        "work",
        secret_store.StoredGrant(
            binding=oauth.binding_fingerprint(entry),
            access_token="AT1",
            refresh_token="RT1",
            issued_at=1_000.0,
            expires_at=4_600.0,
            exchange_access_token="",
            exchange_issued_at=0.0,
            exchange_expires_at=None,
        ),
    )
    posted: list[dict[str, str] | None] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> Any:
        posted.append(headers)

        class _R:
            status_code = 200
            text = "{}"
            headers = httpx2.Headers()

            def json(self) -> dict[str, Any]:
                return {"access_token": "AT2", "expires_in": 3600}

        return _R()

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    cred = oauth.OAuthCredential("work", entry, now=lambda: 4_590.0)  # inside the skew: refresh
    assert cred.token() == "AT2"
    assert posted == [{"user-agent": f"agent6/{__version__}"}]


def test_the_listing_renders_its_own_class(tmp_path: pathlib.Path) -> None:
    entry = _entry(
        {
            "x-static": "literal",
            "x-req": {"template": "{request_id}", "targets": ["models"]},
            "x-model": {"template": "{model}"},
        }
    )
    _url, headers = cache._models_endpoint(entry, "k")
    assert headers["x-static"] == "literal" and len(headers["x-req"]) == 32
    assert "x-model" not in headers and headers["authorization"] == "Bearer k"


def test_a_changed_header_rebinds_the_grant() -> None:
    issuer = {
        "issuer": "https://l.example",
        "client_id": "c",
        "token_endpoint": "https://l.example/t",
        "device_authorization_endpoint": "https://l.example/d",
    }
    base = _entry({}, oauth=issuer)
    changed = _entry({"x-route": "b"}, oauth=issuer)
    assert oauth.binding_fingerprint(base) != oauth.binding_fingerprint(changed)


def test_the_config_reference_documents_the_template_table() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_config", pathlib.Path(__file__).resolve().parents[2] / "docs" / "gen_config.py"
    )
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    rows = gen.leaves()
    for leaf in ("template", "targets", "when"):
        assert f"providers.<name>.extra_headers.<name>.{leaf}" in rows
