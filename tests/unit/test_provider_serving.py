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
