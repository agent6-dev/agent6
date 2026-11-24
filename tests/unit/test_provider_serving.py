# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Exercise the serving facts over HTTP, through the journal and budget."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import budget, config
from agent6.app import role_providers
from agent6.providers import provider_types
from agent6.sessions import event_log
from tests import provider_endpoint


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
