# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Exercise the serving facts over HTTP, through the journal and budget."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import budget, config, kinds
from agent6.app import preflight, role_providers
from agent6.models import cache
from agent6.providers import provider_types
from agent6.sessions import event_log
from tests import price_env, provider_endpoint


def _reply(
    backend: str, stream: bool, facts: dict[str, str]
) -> dict[str, Any] | list[dict[str, Any]]:
    usage = {"input_tokens": 10, "output_tokens": 2}
    if backend == "anthropic":
        body = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": usage,
            "stop_reason": "end_turn",
            **facts,
        }
        return (
            [
                {"type": "message_start", "message": {**facts, "usage": {"input_tokens": 10}}},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
                {"type": "message_stop"},
            ]
            if stream
            else body
        )
    if backend == "openai":
        usage = {"prompt_tokens": 10, "completion_tokens": 2}
        return (
            [
                {**facts, "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": usage},
            ]
            if stream
            else {
                **facts,
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": usage,
            }
        )
    body = {
        "status": "completed",
        "usage": usage,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
    }
    return (
        [
            {"type": "response.created", "response": facts},
            {"type": "response.completed", "response": body},
        ]
        if stream
        else {**body, **facts}
    )


def _worker(
    tmp_path: pathlib.Path, url: str, backend: str, stream: bool, **settings: object
) -> role_providers.InstrumentedProvider:
    cfg = config.Config.model_validate(
        {
            "providers": {
                "work": {"backend": backend, "base_url": url, "auth_style": "none", **settings}
            },
            "models": {"default": {"model": "work/alias"}},
        }
    )
    tracker = budget.BudgetTracker(
        max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1, providers=cfg.providers
    )
    sink = provider_types.TranscriptSink(tmp_path / "transcripts", objects_dir=tmp_path / "objects")
    return role_providers.InstrumentedProvider(
        inner=role_providers.build_role_provider(
            cfg, "worker", budget=tracker, transcript_sink=sink
        ),
        actor="worker",
        model="alias",
        provider_name="work",
        events=event_log.EventSink(tmp_path / "logs.jsonl"),
        budget=tracker,
        stream_text=stream,
    )


def _records(tmp_path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (transcript,) = (tmp_path / "transcripts").glob("*.json")
    rows = [json.loads(line) for line in (tmp_path / "logs.jsonl").read_text().splitlines()]
    return json.loads(transcript.read_text()), [
        row for row in rows if row["type"] in ("model.result", "model.attempt")
    ]


@pytest.mark.parametrize("backend", ["openai", "responses", "anthropic"])
@pytest.mark.parametrize("stream", [True, False])
def test_serving_an_alias_is_a_recorded_fact_not_a_failure(
    tmp_path: pathlib.Path, backend: str, stream: bool
) -> None:
    with provider_endpoint.serve(_reply(backend, stream, {"model": "snapshot"})) as (url, calls):
        worker = _worker(tmp_path, url, backend, stream)
        worker.call(system="s", messages=[{"role": "user", "content": "hi"}])
    transcript, rows = _records(tmp_path)
    assert len(calls) == 1 and [r["type"] for r in rows] == ["model.attempt", "model.result"]
    assert all(r["model"] == "alias" and r["served_model"] == "snapshot" and r["ok"] for r in rows)
    assert transcript["response"]["body"]["model"] == "snapshot"
    assert "asked alias, served snapshot" in worker.budget.format_summary()


@pytest.mark.parametrize("backend", ["openai", "responses"])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    ("tier", "asked", "served", "ok"),
    [
        ("fast", "priority", "priority", True),
        ("fast", "priority", "fast", True),
        ("auto", "auto", "flex", True),
        ("standard", "default", "priority", False),
        ("flex", "flex", "default", False),
        (None, None, "priority", True),
    ],
)
def test_service_tier_is_enforced_after_accounting(
    tmp_path: pathlib.Path,
    backend: str,
    stream: bool,
    tier: str | None,
    asked: str | None,
    served: str,
    ok: bool,
) -> None:
    settings = {"service_tier": tier} if tier else {}
    with provider_endpoint.serve(
        _reply(backend, stream, {"model": "m", "service_tier": served})
    ) as (url, calls):
        worker = _worker(tmp_path, url, backend, stream, **settings)
        if ok:
            worker.call(system="s", messages=[])
        else:
            with pytest.raises(
                provider_types.ProviderError, match=r"providers\.work\.service_tier"
            ) as exc:
                worker.call(system="s", messages=[])
            assert exc.value.fatal
    assert [c.get("service_tier") for c in calls] == [asked]
    transcript, rows = _records(tmp_path)
    assert transcript["response"]["body"]["service_tier"] == served
    assert [r["type"] for r in rows] == ["model.attempt", "model.result"]
    assert all(
        (
            r.get("requested_service_tier"),
            r["served_service_tier"],
            r["ok"],
            r["input_tokens"],
            r["served_model"],
        )
        == (asked, served, ok, 10, "m")
        for r in rows
    )
    assert worker.budget.snapshot().unmetered_tokens == 12


@pytest.mark.parametrize(("backend", "stream"), [("responses", True), ("openai", False)])
def test_a_tier_refusal_names_the_key_and_keeps_the_reason(
    tmp_path: pathlib.Path, backend: str, stream: bool
) -> None:
    error = {"param": "service_tier", "message": "The account cannot use priority."}
    with provider_endpoint.serve({"error": error}, status=400) as (url, calls):
        worker = _worker(tmp_path, url, backend, stream, service_tier="fast")
        with pytest.raises(provider_types.ProviderError) as exc:
            worker.call(system="s", messages=[])
    assert "providers.work.service_tier" in str(exc.value) and error["message"] in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["openai", "responses"])
def test_the_direct_default_is_standard_and_a_tier_in_extra_body_is_refused(backend: str) -> None:
    entry = {"backend": backend, "base_url": "https://api.openai.com/v1"}
    cfg = config.Config.model_validate({"providers": {"work": entry}})
    assert cfg.providers["work"].model_dump()["service_tier"] == "standard"
    with pytest.raises(ValueError, match=r"extra_body\.service_tier"):
        config.Config.model_validate(
            {"providers": {"work": {**entry, "extra_body": {"service_tier": "auto"}}}}
        )


@pytest.mark.parametrize("backend", ["anthropic", "chatgpt"])
def test_an_entry_without_the_tier_key_takes_a_tier_in_extra_body(backend: str) -> None:
    """The refusal named `service_tier`, a key these entries refuse."""
    entry = {"backend": backend, "extra_body": {"service_tier": "standard_only"}}
    cfg = config.Config.model_validate({"providers": {"work": entry}})
    assert cfg.providers["work"].model_dump()["extra_body"] == {"service_tier": "standard_only"}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        config.Config.model_validate(
            {"providers": {"work": {"backend": backend, "service_tier": "standard"}}}
        )


@pytest.mark.parametrize("stream", [True, False])
def test_a_tier_in_extra_body_is_only_a_body_field_on_an_entry_without_the_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """Read as a tier request, it left the route unpriced at the meter."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with provider_endpoint.serve(_reply("anthropic", stream, {"model": "alias"})) as (url, calls):
        worker = _worker(
            tmp_path, url, "anthropic", stream, extra_body={"service_tier": "standard_only"}
        )
        price_env.write({"alias": [1.0, 2.0]}, name="work", entry=worker.budget.providers["work"])
        worker.call(system="s", messages=[{"role": "user", "content": "hi"}])
    assert [c.get("service_tier") for c in calls] == ["standard_only"]
    _transcript, rows = _records(tmp_path)
    assert rows and all("requested_service_tier" not in r for r in rows)
    snap = worker.budget.snapshot()
    assert snap.unmetered_tokens == 0 and snap.usd_own_calls == pytest.approx(14e-6)


@pytest.mark.parametrize("stream", [True, False])
def test_a_2xx_tier_refusal_keeps_the_key(tmp_path: pathlib.Path, stream: bool) -> None:
    error = {
        "code": "invalid_request_error",
        "message": "Account not enrolled",
        "param": "service_tier",
    }
    body = {"status": "failed", "error": error}
    with provider_endpoint.serve(
        [{"type": "response.failed", "response": body}] if stream else body
    ) as (url, calls):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        with pytest.raises(provider_types.ProviderError) as exc:
            worker.call(system="s", messages=[])
    assert exc.value.fatal and "providers.work.service_tier" in str(exc.value)
    assert "Account not enrolled" in str(exc.value) and len(calls) == 1


def test_a_tier_permission_refusal_does_not_refresh_the_bearer(tmp_path: pathlib.Path) -> None:
    with provider_endpoint.serve(
        {"error": {"param": "service_tier", "message": "Not enrolled"}}, status=403
    ) as (url, calls):
        worker = _worker(
            tmp_path,
            url,
            "responses",
            False,
            service_tier="fast",
            auth_style="bearer",
            token_command=["echo", "stand-in-key"],
        )
        with pytest.raises(provider_types.ProviderError, match="Not enrolled") as exc:
            worker.call(system="s", messages=[])
    assert exc.value.fatal and exc.value.status_code == 403
    assert "providers.work.service_tier" in str(exc.value) and len(calls) == 1


def test_a_chat_error_frame_can_name_the_tier_by_param_only(tmp_path: pathlib.Path) -> None:
    error = {"code": "invalid_request_error", "param": "service_tier", "message": "Not enrolled"}
    with provider_endpoint.serve([{"error": error}]) as (url, calls):
        worker = _worker(tmp_path, url, "openai", True, service_tier="fast")
        with pytest.raises(provider_types.ProviderError, match="Not enrolled") as exc:
            worker.call(system="s", messages=[])
    assert exc.value.fatal and "providers.work.service_tier" in str(exc.value) and len(calls) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_a_known_tier_mismatch_is_final_even_without_a_usage_trailer(
    tmp_path: pathlib.Path, stream: bool
) -> None:
    facts = {"model": "snapshot", "service_tier": "priority"}
    with provider_endpoint.serve(
        [{"type": "response.created", "response": facts}] if stream else facts
    ) as (url, calls):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        with pytest.raises(provider_types.ProviderError) as exc:
            worker.call(system="s", messages=[])
    assert exc.value.fatal and "providers.work.service_tier" in str(exc.value) and len(calls) == 1
    assert exc.value.usage is None and exc.value.serving.model == "snapshot"


def test_only_matching_tier_quotes_price_an_auto_call(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route can serve two different prices, or no listed price, during one run."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    listing = {
        "data": [
            {"id": "alias", "pricing": {"prompt": "1", "completion": "1"}},
            {
                "id": "alias",
                "service_tier": "priority",
                "pricing": {"prompt": "0.01", "completion": "0.02"},
            },
        ]
    }
    with provider_endpoint.serve(listing) as (url, _calls):
        cfg = config.Config.model_validate(
            {
                "models": {"default": {"model": "work/alias"}},
                "providers": {
                    "work": {"backend": "responses", "base_url": url, "service_tier": "auto"}
                },
                "budget": {"max_tokens_fallback": 0},
            }
        )
        assert cache.fetch_models_live("work", cfg.providers["work"], None)
    assert preflight.budget_preflight(cfg) is not None
    tracker = budget.BudgetTracker(
        providers=cfg.providers, max_usd=-1, max_tokens_fallback=12, max_plan_points=-1
    )
    for tier, reported in [
        ("priority", None),
        ("flex", None),
        ("default", 0.25),
        ("default", None),
    ]:
        tracker.record(
            route=kinds.ModelRoute("work", "alias"),
            serving=kinds.Serving(requested_tier="auto", tier=tier),
            input_tokens=10,
            output_tokens=2,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=reported,
        )
    snap = tracker.snapshot()
    assert snap.usd_own_calls == pytest.approx(0.14 + 0.25 + 12)
    assert snap.usd_partial and snap.unmetered_tokens == 12 and snap.exhausted


@pytest.mark.parametrize(
    ("tier", "serving", "listed", "preflight_priced", "usd"),
    [
        ("standard", kinds.Serving(requested_tier="default", tier="default"), None, True, 14),
        ("standard", kinds.Serving(requested_tier="default"), None, True, 14),
        ("standard", kinds.Serving(requested_tier="default", tier="default"), "default", True, 70),
        (None, kinds.Serving(tier="default"), None, True, 14),
        ("auto", kinds.Serving(requested_tier="auto", tier="default"), None, False, 14),
    ],
)
def test_the_standard_tier_is_priced_at_the_unqualified_rates(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: str | None,
    serving: kinds.Serving,
    listed: str | None,
    preflight_priced: bool,
    usd: float,
) -> None:
    """A listing without a standard-tier row leaves `max_usd` unbound on a standard route."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows = [{"id": "alias", "pricing": {"prompt": "1", "completion": "2"}}]
    if listed is not None:
        rows.append(
            {"id": "alias", "service_tier": listed, "pricing": {"prompt": "5", "completion": "10"}}
        )
    settings = {"service_tier": tier} if tier is not None else {}
    with provider_endpoint.serve({"data": rows}) as (url, _calls):
        cfg = config.Config.model_validate(
            {
                "models": {"default": {"model": "work/alias"}},
                "providers": {"work": {"backend": "responses", "base_url": url, **settings}},
                "budget": {"max_tokens_fallback": 0},
            }
        )
        assert cache.fetch_models_live("work", cfg.providers["work"], None)
    assert (preflight.budget_preflight(cfg) is None) == preflight_priced
    tracker = budget.BudgetTracker(
        providers=cfg.providers, max_usd=-1, max_tokens_fallback=12, max_plan_points=-1
    )
    tracker.record(
        route=kinds.ModelRoute("work", "alias"),
        serving=serving,
        input_tokens=10,
        output_tokens=2,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    snap = tracker.snapshot()
    assert snap.usd_own_calls == pytest.approx(usd) and snap.unmetered_tokens == 0


@pytest.mark.parametrize(("tier", "wire"), [("flex", "flex"), ("fast", "priority")])
@pytest.mark.parametrize("listed", [False, True])
def test_flex_and_fast_need_a_listed_rate_and_a_serving_tier(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tier: str, wire: str, listed: bool
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    row = {
        "id": "alias",
        "service_tier": wire if listed else None,
        "pricing": {"prompt": "1", "completion": "2"},
    }
    with provider_endpoint.serve({"data": [row]}) as (url, _calls):
        worker = _worker(tmp_path, url, "responses", False, service_tier=tier)
        providers = worker.budget.providers
        assert cache.fetch_models_live("work", providers["work"], None)
    cfg = config.Config.model_validate(
        {
            "providers": providers,
            "models": {"default": {"model": "work/alias"}},
            "budget": {"max_tokens_fallback": 0},
        }
    )
    assert (preflight.budget_preflight(cfg) is None) == listed
    tracker = budget.BudgetTracker(
        providers=providers, max_usd=-1, max_tokens_fallback=12, max_plan_points=-1
    )
    tracker.record(
        route=kinds.ModelRoute("work", "alias"),
        serving=kinds.Serving(requested_tier=wire, tier=None if listed else wire),
        input_tokens=10,
        output_tokens=2,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    snap = tracker.snapshot()
    assert snap.usd_partial and snap.unmetered_tokens == 12 and snap.exhausted


@pytest.mark.parametrize("stream", [True, False])
def test_responses_reported_dollars_take_precedence_over_a_missing_tier_price(
    tmp_path: pathlib.Path, stream: bool
) -> None:
    body: dict[str, Any] = {
        "status": "completed",
        "service_tier": "flex",
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.2},
        "output": [],
    }
    with provider_endpoint.serve(
        [{"type": "response.completed", "response": body}] if stream else body
    ) as (url, _):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        worker.call(system="s", messages=[])
    assert worker.budget.snapshot().usd_own_calls == 0.2
    assert worker.budget.snapshot().unmetered_tokens == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tier", ["auto", "flex"])
def test_echoed_credentials_in_serving_facts_never_reach_the_journal_or_summary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tier: str, stream: bool
) -> None:
    key = "stand-in-bearer-for-this-call"
    monkeypatch.setenv("TEST_SERVING_KEY", key)
    facts = {"model": f"snapshot:{key}", "service_tier": key}
    with provider_endpoint.serve(_reply("responses", stream, facts)) as (url, calls):
        worker = _worker(
            tmp_path,
            url,
            "responses",
            stream,
            service_tier=tier,
            auth_style="bearer",
            api_key_env="TEST_SERVING_KEY",
        )
        if tier == "auto":
            reply = worker.call(system="s", messages=[])
            assert key not in repr(reply.serving)
        else:
            with pytest.raises(provider_types.ProviderError) as exc:
                worker.call(system="s", messages=[])
            assert exc.value.fatal and key not in str(exc.value) + repr(exc.value.serving)
    transcript, rows = _records(tmp_path)
    assert len(calls) == 1 and len(rows) == 2
    assert key not in json.dumps([rows, transcript]) + worker.budget.format_summary()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("context", ["model flex-worker", "the flex service tier"])
def test_a_rate_limit_is_not_a_tier_refusal(
    tmp_path: pathlib.Path, stream: bool, context: str
) -> None:
    message = f"Rate limit exceeded for {context}"
    error = {"code": "rate_limit_exceeded", "message": message}
    with provider_endpoint.serve({"error": error}, status=429) as (url, calls):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        with pytest.raises(provider_types.ProviderError, match=message) as exc:
            worker.call(system="s", messages=[])
    assert exc.value.status_code == 429 and not exc.value.fatal
    assert len(calls) == 1 and "providers.work.service_tier" not in str(exc.value)


@pytest.mark.parametrize("stream", [False, True])
def test_a_tier_refusal_without_a_parameter_keeps_the_reason(
    tmp_path: pathlib.Path, stream: bool
) -> None:
    message = "The flex service tier is not enabled for this project"
    with provider_endpoint.serve({"error": {"message": message}}, status=403) as (url, calls):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        with pytest.raises(provider_types.ProviderError, match=message) as exc:
            worker.call(system="s", messages=[])
    assert exc.value.fatal and "providers.work.service_tier" in str(exc.value) and len(calls) == 1


@pytest.mark.parametrize("served", ["priority", "flex"])
def test_malformed_json_output_is_billed_before_checking_the_tier(
    tmp_path: pathlib.Path, served: str
) -> None:
    """A wire shape error keeps the bill and is final only for a tier mismatch."""
    body = {
        "status": "completed",
        "model": "snapshot",
        "service_tier": served,
        "output": [{"type": "message", "content": 1}],
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.25},
    }
    with provider_endpoint.serve(body) as (url, calls):
        worker = _worker(tmp_path, url, "responses", False, service_tier="flex")
        with pytest.raises(
            provider_types.ProviderError, match="did not match the wire shape"
        ) as exc:
            worker.call(system="s", messages=[])
    mismatch = served != "flex"
    assert len(calls) == 1
    assert (worker.budget.snapshot().usd_own_calls, exc.value.fatal) == (0.25, mismatch)
    assert ("providers.work.service_tier" in str(exc.value)) == mismatch
    if mismatch:
        assert "asked 'flex', served 'priority'" in str(exc.value)
    else:
        assert exc.value.status_code is None
    transcript, rows = _records(tmp_path)
    assert len(rows) == 2 and transcript["usage"]["cost_usd"] == 0.25
    assert all(
        (row["input_tokens"], row["output_tokens"], row["served_service_tier"]) == (10, 2, served)
        for row in rows
    )


@pytest.mark.parametrize(
    ("content", "stream"),
    [(1, False), ([], False), ([], True)],
    ids=["malformed", "parsed", "streamed"],
)
def test_a_cost_no_float_holds_is_unreported_and_the_tier_is_still_checked(
    tmp_path: pathlib.Path, content: object, stream: bool
) -> None:
    """An integer cost past the float range keeps the token counts and the tier refusal."""
    body = {
        "status": "completed",
        "model": "snapshot",
        "service_tier": "priority",
        "output": [{"type": "message", "content": content}],
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 10**400},
    }
    with provider_endpoint.serve(
        [{"type": "response.completed", "response": body}] if stream else body
    ) as (url, calls):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        with pytest.raises(
            provider_types.ProviderError, match="asked 'flex', served 'priority'"
        ) as exc:
            worker.call(system="s", messages=[])
    assert exc.value.fatal and len(calls) == 1
    snapshot = worker.budget.snapshot()
    assert (snapshot.usd_own_calls, snapshot.unmetered_tokens) == (0, 12)
    _, rows = _records(tmp_path)
    assert len(rows) == 2 and all(
        (row["input_tokens"], row["output_tokens"], row["served_service_tier"])
        == (10, 2, "priority")
        for row in rows
    )


@pytest.mark.parametrize("stream", [False, True])
def test_billed_tier_mismatch_keeps_the_failed_generations_usage(
    tmp_path: pathlib.Path, stream: bool
) -> None:
    body = {
        "status": "failed",
        "model": "snapshot",
        "service_tier": "priority",
        "error": {"code": "server_error", "message": "Generation failed"},
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.25},
    }
    with provider_endpoint.serve(
        [{"type": "response.failed", "response": body}] if stream else body
    ) as (url, calls):
        worker = _worker(tmp_path, url, "responses", stream, service_tier="flex")
        with pytest.raises(provider_types.ProviderError, match="Generation failed") as exc:
            worker.call(system="s", messages=[])
    reason = str(exc.value)
    assert (
        exc.value.fatal
        and "providers.work.service_tier" in reason
        and "asked 'flex', served 'priority'" in reason
    )
    assert len(calls) == 1 and worker.budget.snapshot().usd_own_calls == 0.25
    transcript, rows = _records(tmp_path)
    assert len(rows) == 2 and transcript["usage"]["cost_usd"] == 0.25
    assert all(
        (row["input_tokens"], row["output_tokens"], row["cost_usd"], row["served_service_tier"])
        == (10, 2, 0.25, "priority")
        for row in rows
    )
