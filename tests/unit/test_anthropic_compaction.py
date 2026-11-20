"""Pin the Messages wire to captured compaction and prefix-check replies."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6 import budget
from agent6.providers import anthropic, anthropic_parse, provider_types, transport
from tests.unit import test_anthropic_thinking as wire

SAMPLES: dict[str, Any] = json.loads(
    (pathlib.Path(__file__).parent / "fixtures/anthropic_compaction.json").read_text()
)


@pytest.mark.parametrize(
    "sample", ["compact", "adopt-strict", "clearing-strict", "own-restart-drop_block"]
)
def test_captured_replies_report_the_endpoint_work(sample: str) -> None:
    raw = SAMPLES[sample]["response"]
    response = anthropic_parse.parse_response(raw)
    assert response.raw == raw and response.stop_reason == raw["stop_reason"]
    assert response.thinking_blocks_dropped == (sample == "own-restart-drop_block")
    assert response.tool_results_cleared == (2 if sample == "clearing-strict" else 0)
    if sample == "compact":
        assert response.summary_block == raw["content"][0]
        assert response.text == raw["content"][0]["content"]
        assert response.usage == provider_types.ProviderUsage(60, 434, 33050, 0)
    else:
        assert response.summary_block is None


@pytest.mark.parametrize("sample", ["compact", "adopt-strict", "clearing-strict"])
def test_request_keeps_messages_and_merges_required_betas(
    monkeypatch: pytest.MonkeyPatch, sample: str
) -> None:
    body: dict[str, Any] = {}
    headers: dict[str, str] = {}

    def post(url: str, **kwargs: Any) -> wire._FakeResponse:
        body.update(json.loads(kwargs["content"]))
        headers.update(kwargs["headers"])
        return wire._FakeResponse(status_code=200, payload=SAMPLES[sample]["response"])

    monkeypatch.setattr(transport, "http_post", post)
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    provider = anthropic.AnthropicProvider(
        api_key="test", model="claude-haiku-5-5", effort="high", budget=tracker
    )
    messages = SAMPLES["adopt-strict" if sample == "adopt-strict" else "turn3"]["messages"]
    options: dict[str, Any] = {}
    if sample == "compact":
        options["compaction"] = provider_types.Summarise("Keep the goal.")
    elif sample == "clearing-strict":
        options["compaction"] = provider_types.ClearToolResults(3, exclude_tools=("ask_user",))
    response = provider.call(system="system", messages=messages, **options)
    assert body["messages"] == messages
    assert body["thinking"] == SAMPLES["own-restart-drop_block"]["thinking"]
    betas = set(headers["anthropic-beta"].split(","))
    assert betas == {
        *SAMPLES[sample]["betas"],
        "thinking-binding-controls-2026-08-01",
        "prompt-caching-2024-07-31",
    }
    if sample == "compact":
        assert body["compaction"] == {"type": "summarize", "instructions": "Keep the goal."}
        assert response.input_tokens == 60
        assert "context_management" not in body
    elif sample == "clearing-strict":
        # The harness decides when and how much: the edit always applies and keeps what it names.
        assert body["context_management"] == {
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 1},
                    "keep": {"type": "tool_uses", "value": 3},
                    "exclude_tools": ["ask_user"],
                }
            ]
        }
        assert "compaction" not in body


@pytest.mark.parametrize("deployment", ["direct", "vertex"])
@pytest.mark.parametrize(
    "model",
    [
        "fable-5-2",
        "mythos-5-2",
        "fable-5-1",
        "mythos-5-1",
        "opus-5-5",
        "sonnet-5-5",
        "haiku-5-5",
        "sonnet-4-6",
        "haiku-5-50",
    ],
)
@pytest.mark.parametrize("effort", ["high", "off"])
def test_prefix_net_follows_model_and_thinking(
    monkeypatch: pytest.MonkeyPatch, deployment: Any, model: str, effort: str
) -> None:
    bodies: list[dict[str, Any]] = []
    wire._capture_body(monkeypatch, bodies)
    provider = anthropic.AnthropicProvider(
        api_key="test", model=f"claude-{model}", effort=effort, deployment=deployment
    )
    provider.call(system="system", messages=[{"role": "user", "content": "go"}])
    assert ("block_binding" in bodies[0].get("thinking", {})) is (
        effort != "off" and model not in ("sonnet-4-6", "haiku-5-50")
    )
    assert "compaction" not in bodies[0] and "context_management" not in bodies[0]


def test_an_operator_prefix_setting_stands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the operator's refusal: the net replaced `error` with `drop_block`."""
    bodies: list[dict[str, Any]] = []
    wire._capture_body(monkeypatch, bodies)
    strict = {"block_binding": {"prefix_mismatch_behavior": "error"}}
    provider = anthropic.AnthropicProvider(
        api_key="test",
        model="claude-haiku-5-5",
        effort="high",
        extra_body={"thinking": {"type": "adaptive", **strict}},
    )
    provider.call(system="system", messages=[{"role": "user", "content": "go"}])
    assert bodies[0]["thinking"]["block_binding"] == strict["block_binding"]


def test_every_usage_iteration_is_billed() -> None:
    """Pin the sum: a reply with a message iteration beside the summary lost the message's cost."""
    raw = SAMPLES["compact"]["response"]
    summary = raw["usage"]["iterations"][0]
    message = {"type": "message", "input_tokens": 7, "output_tokens": 100}
    message["cache_creation_input_tokens"] = 688
    usage = {**raw["usage"], "iterations": [summary, message]}
    response = anthropic_parse.parse_response({**raw, "usage": usage})
    assert response.usage == provider_types.ProviderUsage(67, 534, 33050, 688)


def test_a_summary_stream_cut_before_its_end_is_billed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the cut stream: its cost sits in the iterations and the budget stayed at zero."""
    raw = SAMPLES["compact"]["response"]
    zero = {k: v for k, v in raw["usage"].items() if k != "iterations"}
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"message": {"usage": zero}}),
        ("content_block_start", {"index": 0, "content_block": raw["content"][0]}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "compaction"}, "usage": raw["usage"]}),
    ]

    def stream(*_args: Any, **_kwargs: Any) -> wire._FakeStreamResponse:
        return wire._FakeStreamResponse(status_code=200, lines=wire._sse(events))

    monkeypatch.setattr(wire.httpx2, "stream", stream)
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    provider = anthropic.AnthropicProvider(api_key="test", model="claude-haiku-5-5", budget=tracker)
    with pytest.raises(provider_types.ProviderError) as cut:
        provider.call(
            system="s",
            messages=SAMPLES["turn3"]["messages"],
            text_delta_callback=lambda _: None,
            compaction=provider_types.Summarise("Keep the goal."),
        )
    assert cut.value.usage == provider_types.ProviderUsage(60, 434, 33050, 0)
    assert (tracker._input_total, tracker._output_total, tracker._cache_read_total) == (
        60,
        434,
        33050,
    )


@pytest.mark.parametrize("sample", ["compact", "own-restart-drop_block", "clearing-strict"])
def test_stream_reports_the_same_history_and_cost(
    monkeypatch: pytest.MonkeyPatch, sample: str
) -> None:
    raw = SAMPLES[sample]["response"]
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "message": {
                    "usage": raw["usage"],
                    "input_transformations": raw.get("input_transformations", []),
                }
            },
        )
    ]
    for i, block in enumerate(raw["content"]):
        events.extend(
            [
                ("content_block_start", {"index": i, "content_block": block}),
                ("content_block_stop", {"index": i}),
            ]
        )
    events.extend(
        [
            (
                "message_delta",
                {
                    "delta": {
                        "stop_reason": raw["stop_reason"],
                        "context_management": raw.get("context_management"),
                    },
                    "usage": raw["usage"],
                },
            ),
            ("message_stop", {}),
        ]
    )

    def stream(*_args: Any, **_kwargs: Any) -> wire._FakeStreamResponse:
        return wire._FakeStreamResponse(status_code=200, lines=wire._sse(events))

    monkeypatch.setattr(wire.httpx2, "stream", stream)
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    provider = anthropic.AnthropicProvider(api_key="test", model="claude-haiku-5-5", budget=tracker)
    reply = provider.call(
        system="s", messages=SAMPLES["turn3"]["messages"], text_delta_callback=lambda _: None
    )
    parsed = anthropic_parse.parse_response(raw)
    assert reply.raw["content"] == raw["content"]
    assert reply.usage == parsed.usage and reply.summary_block == parsed.summary_block
    assert reply.thinking_blocks_dropped == parsed.thinking_blocks_dropped
    assert reply.tool_results_cleared == parsed.tool_results_cleared


@pytest.mark.parametrize(
    "fields",
    [
        {"input_transformations": [None]},
        {"context_management": "invalid"},
        {"context_management": {"applied_edits": ["invalid"]}},
        {"usage": {"iterations": 1}},
    ],
)
def test_malformed_history_fields_raise_a_provider_error(fields: dict[str, Any]) -> None:
    with pytest.raises(provider_types.ProviderError):
        anthropic_parse.parse_response({**SAMPLES["compact"]["response"], **fields})
