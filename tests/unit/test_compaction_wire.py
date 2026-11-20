from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import time
from collections.abc import Generator
from typing import Any

import httpx2
import pytest

from agent6 import budget, config
from agent6.app import conversation_state, role_providers
from agent6.config import secret_store
from agent6.harness import loop_snapshot, loop_state
from agent6.harness.compaction import driver
from agent6.providers import (
    anthropic,
    chatgpt,
    oauth,
    openai,
    responses,
    responses_codec,
    transport,
)
from agent6.sessions import event_log
from tests.unit import test_compaction_writers as fixtures
from tests.unit import test_harness


@pytest.mark.parametrize(
    "signal,reason",
    [
        (fixtures.provider_types.ProviderAborted("stop"), "steer_abort"),
        (budget.BudgetExceededError("spent"), "budget_exhausted"),
        (fixtures.provider_types.ProviderError("sign in", status_code=401), "provider_error"),
    ],
)
def test_summary_signals_end_the_harness_without_another_call(
    signal: Exception,
    reason: str,
) -> None:
    worker = fixtures.Writer(signal)
    helper = fixtures.Writer(fixtures.response())
    wf = test_harness._wf(provider=worker, compaction=fixtures.context(worker, helper, []).settings)
    state = loop_state.LoopState("task", 1, system="system")
    conv = fixtures.history()
    result = wf._turn_provider_call(
        "system",
        conv,
        [],
        state,
        turn=2,
        ctx=wf._turn_context(state, turn=2, execution_start=1),
    )
    assert isinstance(result, loop_snapshot.SessionResult)
    assert result.reason == reason
    assert len(worker.calls) == 1
    assert helper.calls == []


def test_shared_budget_stops_the_fallback_before_it_spends() -> None:
    worker = fixtures.Writer(fixtures.response(""))
    helper = fixtures.Writer(fixtures.response())
    comp = fixtures.context(worker, helper, [])

    def check() -> None:
        if worker.calls:
            raise budget.BudgetExceededError("spent by the worker summary")

    comp = dataclasses.replace(comp, check_budget=check)
    conv = comp.conversation
    before = conv.to_wire()
    with pytest.raises(budget.BudgetExceededError):
        driver.Tiered().compact(comp)
    assert helper.calls == []
    assert conv.to_wire() == before


@pytest.mark.parametrize("backend", ["anthropic", "openai", "openrouter", "responses", "chatgpt"])
def test_worker_summary_keeps_the_wire_prefix_and_bills_the_same_tracker(
    backend: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary keeps tools, system, effort, reasoning replay and conversation identity."""
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    events = event_log.EventSink(tmp_path / "logs.jsonl")
    captured: list[dict[str, Any]] = []
    contexts: list[str] = []
    common: dict[str, Any] = {"model": "test-model", "reasoning_effort": "high", "budget": tracker}
    if backend == "anthropic":
        inner = anthropic.AnthropicProvider(
            api_key="fake", model="test-model", effort="high", budget=tracker
        )
    elif backend in ("openai", "openrouter"):
        url = (
            "https://openrouter.ai/api/v1"
            if backend == "openrouter"
            else "https://api.openai.com/v1"
        )
        inner = openai.OpenAIProvider(api_key="fake", base_url=url, **common)
    elif backend == "responses":
        inner = responses.ResponsesProvider(
            api_key="fake", base_url="https://models.example/v1", **common
        )
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        entry = config.ChatGPTProviderEntry(backend="chatgpt")
        secret_store.save_oauth_grant(
            "test",
            secret_store.StoredGrant(
                binding=oauth.binding_fingerprint(entry),
                access_token="fake",
                refresh_token="fake",
                issued_at=time.time(),
                expires_at=time.time() + 3600,
                account_id="test",
                exchange_access_token="",
                exchange_issued_at=0,
                exchange_expires_at=None,
            ),
        )
        inner = chatgpt.ChatGPTProvider(
            base_url="https://chatgpt.com/backend-api/codex",
            credential=oauth.OAuthCredential("test", entry),
            provider_name="test",
            account_id="test",
            **common,
        )
    worker = role_providers.InstrumentedProvider(
        inner=inner,
        actor="worker",
        model="test-model",
        provider_name="test",
        events=events,
        budget=tracker,
        conversations=conversation_state.Conversations(),
    )
    body: dict[str, Any]
    if backend == "anthropic":
        body = {
            "content": [{"type": "text", "text": "summary"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 20, "output_tokens": 5},
        }
    elif backend in ("openai", "openrouter"):
        body = {
            "choices": [{"message": {"content": "summary"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens": 5,
            },
        }
    else:
        body = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "summary"}],
                }
            ],
            "usage": {
                "input_tokens": 30,
                "input_tokens_details": {"cached_tokens": 20},
                "output_tokens": 5,
            },
        }

    def post(*args: Any, **kwargs: Any) -> httpx2.Response:
        captured.append(json.loads(kwargs["content"]))
        return httpx2.Response(200, json=body, request=httpx2.Request("POST", "https://test"))

    @contextlib.contextmanager
    def stream(*args: Any, **kwargs: Any) -> Generator[httpx2.Response]:
        if args[0] == "GET":
            yield httpx2.Response(503, request=httpx2.Request("GET", "https://test/usage"))
            return
        captured.append(json.loads(kwargs["content"]))
        data = "data: " + json.dumps({"type": "response.completed", "response": body}) + "\n\n"
        yield httpx2.Response(
            200, content=data.encode(), request=httpx2.Request("POST", "https://test")
        )

    monkeypatch.setattr(transport, "http_post", post)
    monkeypatch.setattr(httpx2, "stream", stream)
    conv = fixtures.history()
    if backend in ("chatgpt", "responses"):
        # The opaque item is route-bound; a helper-style conversion would discard it.
        route = responses_codec.replay_route(backend, inner.base_url, "test-model")
        conv.assistant(
            [
                {
                    "type": "thinking",
                    "thinking": "reason",
                    responses_codec.REPLAY_KEY: {
                        "wire": "responses",
                        "route": route,
                        "item": {
                            "type": "reasoning",
                            "id": "opaque",
                            "encrypted_content": "ciphertext",
                        },
                    },
                },
                {"type": "text", "text": "next"},
            ]
        )
        conv.notice("continue")
    tools = [fixtures.provider_types.ToolDefinition("read_file", "Read a file", {"type": "object"})]
    previous = worker.call(system="system", messages=conv.to_wire(), tools=tools, temperature=0.0)
    conv.assistant(previous.raw["content"])
    conv.notice("The next turn is ready.")
    conv.roll_cache_marks()
    context_before = worker.conversations.context("worker", "test", initiator="agent")
    contexts.append(context_before.conversation_id)
    turn = fixtures.context(
        worker, fixtures.Writer(fixtures.response()), [], conversation=conv, tools=tools
    )
    assert driver.Tiered().compact(
        dataclasses.replace(turn, emit=events.emit, check_budget=tracker.check)
    )
    contexts.append(
        worker.conversations.context("worker", "test", initiator="agent").conversation_id
    )
    assert contexts[0] == contexts[1]
    before, summary = captured
    key = "input" if "input" in before else "messages"
    assert summary[key][: len(before[key])] == before[key]
    for field in (
        "system",
        "instructions",
        "tools",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "output_config",
        "prompt_cache_key",
    ):
        assert summary.get(field) == before.get(field), field
    if backend in ("chatgpt", "responses"):
        assert "ciphertext" in json.dumps(summary)
    if backend == "chatgpt":
        assert summary["prompt_cache_key"] == contexts[0]
    rows = [json.loads(line) for line in events.path.read_text().splitlines()]
    results = [r for r in rows if r["type"] == "model.result"]
    assert len(results) == 2
    assert all(r["input_tokens"] == 10 and r["cache_read_tokens"] == 20 for r in results)
    assert len([r for r in rows if r["type"] == "budget.update"]) == 2
    assert tracker.snapshot().input_total == 20
    assert tracker.snapshot().cache_read_total == 40
    summary_call = next(r for r in rows if r["type"] == "compact.summarise.call")
    assert summary_call["model_call_id"] == results[-1]["model_call_id"]
    assert summary_call["writer"] == "worker"
