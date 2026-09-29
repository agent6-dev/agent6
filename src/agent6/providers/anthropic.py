# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Anthropic Messages provider.

The transport and the SSE lifecycle are shared with the OpenAI provider; both use
httpx2 directly, with no SDK, for a smaller audit surface and a pinned URL. Prompt
caching rides the `cache_control` field on system and tool entries.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers._stream import SseCall, StreamClock, record_billed_usage, sse_events
from agent6.providers._transport import ProviderCall, envelope_status, meter_completion
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
    scrub_secret_values,
)
from agent6.providers.wire import AuthStyle, Deployment, auth_header, request_url

if TYPE_CHECKING:
    from agent6.providers.token_command import CommandToken

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
# Vertex carries the protocol version in the body, under its own value.
ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
DEFAULT_MAX_TOKENS = 8192


def _anthropic_version(deployment: str) -> tuple[str, str]:
    """Return where the protocol version goes and its value.

    Direct sends the `anthropic-version` header; Vertex sends an `anthropic_version`
    body field with its own value.
    """
    if deployment == "vertex":
        return ("body", ANTHROPIC_VERTEX_VERSION)
    return ("header", ANTHROPIC_VERSION)


# The `budget_tokens` per effort level for models without adaptive thinking; the wire requires
# `budget_tokens < max_tokens`, so the call lifts `max_tokens` to leave room to answer.
_THINKING_BUDGET_TOKENS: dict[str, int] = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 16384,
    "max": 16384,
}

# Models that reject `budget_tokens` (a 400) and take `thinking: {type: adaptive}` plus
# `output_config.effort` instead.
_ADAPTIVE_THINKING_MARKERS = (
    "fable-5",
    "mythos-5",
    "mythos-preview",
    "opus-4-6",
    "opus-4-7",
    "opus-4-8",
    "opus-5",
    "sonnet-4-6",
    "sonnet-5",
)
# Of those, the models whose thinking display defaults to omitted; a summary streams progress.
_SUMMARISE_DISPLAY_MARKERS = (
    "fable-5",
    "mythos-5",
    "mythos-preview",
    "opus-4-7",
    "opus-4-8",
    "opus-5",
    "sonnet-5",
)


def _is_adaptive_thinking(model: str) -> bool:
    """Return whether the model takes adaptive thinking rather than a token budget."""
    return any(m in model for m in _ADAPTIVE_THINKING_MARKERS)


def _summarise_thinking_display(model: str) -> bool:
    """Return whether the model's thinking display defaults to omitted."""
    return any(m in model for m in _SUMMARISE_DISPLAY_MARKERS)


def _non_negative_integer(value: Any, field_name: str) -> int:
    """Return a response value as a non-negative integer.

    Raises:
        ProviderError: The value is not one.
    """  # noqa: DOC501  # the TypeError is raised and caught in the same try
    try:
        if isinstance(value, bool):
            raise TypeError
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderError(
            f"Anthropic response {field_name} was not a non-negative integer"
        ) from exc
    if count < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ProviderError(f"Anthropic response {field_name} was not a non-negative integer")
    return count


def _usage_count(usage: Mapping[str, Any], key: str) -> int:
    """Return one usage count; 0 when absent."""
    value = usage.get(key)
    if value is None:
        return 0
    return _non_negative_integer(value, f"usage.{key}")


def _usage_mapping(value: Any) -> Mapping[str, Any]:
    """Return a `usage` object, empty when absent.

    Raises:
        ProviderError: The value is present but not an object.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProviderError("Anthropic response usage was not an object")
    return value


def _response_string(value: Any, field_name: str, *, empty: bool = True) -> str:
    """Return a response field as a string.

    Args:
        value: The field's value.
        field_name: The field's name, for the error.
        empty: Whether a blank string is accepted.

    Raises:
        ProviderError: The value is not a string, or is blank when one is required.
    """
    if not isinstance(value, str) or (not empty and not value.strip()):
        qualifier = "a nonempty string" if not empty else "a string"
        raise ProviderError(f"Anthropic response {field_name} was not {qualifier}")
    return value


def _require_metered_usage(usage: object, *, source: str) -> None:
    """Refuse a budgeted call whose usage cannot be metered.

    A gateway with usage tracking off returns all-zero counts, so presence is not
    enough; the input side must be positive with the cache counters summed in,
    since a fully cached turn reports `input_tokens: 0`.

    Args:
        usage: The response's `usage` value.
        source: The name that leads the error.

    Raises:
        ProviderError: No positive input usage (retryable: a usage-less reply is a
            gateway integrity failure, not a request defect).
    """
    if isinstance(usage, Mapping):
        total_input = (
            _usage_count(usage, "input_tokens")
            + _usage_count(usage, "cache_read_input_tokens")
            + _usage_count(usage, "cache_creation_input_tokens")
        )
        if total_input > 0:
            return
    raise ProviderError(
        f"{source} reported no usage input tokens (usage.input_tokens missing or 0); "
        "budgeted runs require provider usage accounting"
    )


def strip_cache_control_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the messages with every `cache_control` marker removed, copy-on-write.

    The harness places rolling breakpoints in the message list; with
    `prompt_caching = false` they are stripped before the request is built. The
    caller's list, shared with resume snapshots, is never mutated.
    """
    out: list[dict[str, Any]] = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            out.append(msg)
            continue
        new_content = [
            {k: v for k, v in b.items() if k != "cache_control"}
            if isinstance(b, dict) and "cache_control" in b
            else b
            for b in content
        ]
        out.append({**msg, "content": new_content})
        changed = True
    return out if changed else messages


def shape_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the messages the Anthropic wire accepts, without mutating history.

    Reasoning from another wire has no Anthropic signature and cannot be replayed;
    a message it empties is dropped, since the API rejects an empty one.

    Raises:
        ProviderError: No message is left (fatal).
    """

    def foreign(block: Any) -> bool:
        """Return whether a block is reasoning this wire cannot replay."""
        if not isinstance(block, dict):
            return False
        if "chatgpt_reasoning" in block:
            return True
        return block.get("type") == "thinking" and not (
            isinstance(block.get("signature"), str) and block["signature"]
        )

    out: list[dict[str, Any]] = []
    changed = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        kept = [block for block in content if not foreign(block)]
        if not kept:
            changed = True
            continue
        if kept != content:
            out.append({**msg, "content": kept})
            changed = True
        else:
            out.append(msg)
    shaped = out if changed else messages
    if not shaped:
        raise ProviderError("Anthropic request has no nonempty messages", fatal=True)
    return shaped


def _is_temperature_400(status: int | None, text: str, body: dict[str, Any]) -> bool:
    """Return whether a 400 rejects a `temperature` the body still carries."""
    return status == 400 and "temperature" in body and "temperature" in (text or "").lower()


@dataclass(frozen=True, slots=True)
class AnthropicProvider:
    """The Anthropic provider, constructed once per run.

    Attributes:
        api_key: The static credential, used when `credential` is None.
        model: The model id.
        base_url: The endpoint's base URL.
        deployment: The URL profile.
        auth_style: The auth header style: `x_api_key` direct, `bearer` on Vertex.
        prompt_caching: Whether the system block and the last tool are cache marked.
        timeout_s: The read budget in seconds.
        transcript_sink: Where each round-trip is recorded.
        budget: The run's tracker; None skips metering.
        effort: The reasoning level; anything but "off" enables extended thinking
            and drops `temperature`, which the wire rejects while thinking.
        extra_headers: Operator headers merged over the built ones.
        extra_body: Operator body keys merged last; the structural keys are reserved.
        extra_query: Operator query parameters.
        credential: A short-lived bearer source; a 401 or 403 re-mints it once.
    """

    api_key: str
    model: str
    base_url: str = ANTHROPIC_DEFAULT_BASE_URL
    deployment: Deployment = "direct"
    auth_style: AuthStyle = "x_api_key"
    prompt_caching: bool = True
    timeout_s: float = 120.0
    transcript_sink: TranscriptRecorder | None = None
    budget: BudgetTracker | None = None
    effort: str | None = None
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_query: dict[str, str] = field(default_factory=dict)
    credential: CommandToken | None = None
    # Latched on a "temperature is deprecated" 400 so the rest of the run omits it; a list, since
    # the dataclass is frozen.
    _omit_temperature: list[bool] = field(default_factory=lambda: [False])

    def _adapt_body_for_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Drop `temperature` on a 400 that rejects it and latch the omission.

        Args:
            status: The HTTP status.
            text: The error text.
            body: The request body, rewritten in place.

        Returns:
            Whether the body was adapted, so the transport retries once.
        """
        if not _is_temperature_400(status, text, body):
            return False
        self._omit_temperature[0] = True
        body.pop("temperature", None)
        return True

    def _build_headers(self, token: str) -> dict[str, str]:
        """Build one attempt's request headers from its token.

        Args:
            token: The attempt's credential.

        Returns:
            The headers, with an operator `anthropic-beta` merged into the built one.
        """
        headers: dict[str, str] = {"content-type": "application/json"}
        authed = auth_header(self.auth_style, token)
        if authed is not None:
            headers[authed[0]] = authed[1]
        version_placement, version_value = _anthropic_version(self.deployment)
        if version_placement == "header":
            headers["anthropic-version"] = version_value
        if self.prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        for k, v in self.extra_headers:
            key = k.lower()
            if key == "anthropic-beta" and key in headers:
                betas = [part.strip() for part in f"{headers[key]},{v}".split(",")]
                headers[key] = ",".join(dict.fromkeys(part for part in betas if part))
            else:
                headers[key] = v
        return headers

    def call(  # noqa: PLR0912
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        """Make one Messages call, streaming when a delta callback is set.

        Args:
            system: The system prompt.
            messages: The conversation in Anthropic shape.
            tools: The tools the model may call.
            max_tokens: The output cap; lifted to leave room to answer after thinking.
            temperature: The sampling temperature; dropped while thinking.
            reasoning_effort: Ignored; thinking is configured on the provider's `effort`.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning text as it streams.
            should_abort: Polled during a stream; True abandons the turn.
            should_interrupt: Polled during a stream; True ends the turn to steer.

        Returns:
            The parsed response.

        Raises:
            ProviderError: The run is over budget, `extra_body.max_tokens` sits under
                the thinking budget, or the call failed.
        """
        del reasoning_effort
        if self.budget is not None:
            self.budget.check()
        streaming = text_delta_callback is not None or thinking_delta_callback is not None
        url, model_in_body = request_url(
            api_format="anthropic",
            deployment=self.deployment,
            base_url=self.base_url,
            model=self.model,
            streaming=streaming,
            extra_query=self.extra_query,
        )
        version_placement, version_value = _anthropic_version(self.deployment)

        # Of the four cache breakpoints, this marks the system block and the last tool; the harness
        # places the other two in `messages`.
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if self.prompt_caching:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}
        else:
            messages = strip_cache_control_messages(messages)

        tool_payload: list[dict[str, Any]] = []
        if tools:
            for i, t in enumerate(tools):
                block: dict[str, Any] = {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                if self.prompt_caching and i == len(tools) - 1:
                    block["cache_control"] = {"type": "ephemeral"}
                tool_payload.append(block)

        level = self.effort or "off"
        adaptive_thinking = level != "off" and _is_adaptive_thinking(self.model)
        thinking_budget = None if adaptive_thinking else _THINKING_BUDGET_TOKENS.get(level)
        if adaptive_thinking or thinking_budget is not None:
            # Adaptive carries no budget, so it reserves the deepest fixed one as headroom.
            reserve = thinking_budget or _THINKING_BUDGET_TOKENS["high"]
            max_tokens = max(max_tokens, reserve + DEFAULT_MAX_TOKENS)

        body: dict[str, Any] = {
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": shape_anthropic_messages(messages),
        }
        if model_in_body:
            body["model"] = self.model
        if version_placement == "body":
            body["anthropic_version"] = version_value
        if adaptive_thinking:
            thinking_cfg: dict[str, Any] = {"type": "adaptive"}
            if _summarise_thinking_display(self.model):
                thinking_cfg["display"] = "summarized"
            body["thinking"] = thinking_cfg
            # The wire's effort tops out at high.
            body["output_config"] = {"effort": "high" if level in ("xhigh", "max") else level}
        elif thinking_budget is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        elif temperature is not None and not self._omit_temperature[0]:
            body["temperature"] = temperature
        if tool_payload:
            body["tools"] = tool_payload
        if self.extra_body:
            # The structural keys stay agent6's; tuning keys merge last and win.
            reserved = {
                "system",
                "messages",
                "model",
                "stream",
                "anthropic_version",
                "tools",
                "tool_choice",
            }
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})
        if thinking_budget is not None:
            # An extra_body max_tokens under the budget is the operator's to fix.
            configured_max = body.get("max_tokens")
            if not isinstance(configured_max, int) or isinstance(configured_max, bool):
                raise ProviderError("Anthropic request max_tokens was not an integer", fatal=True)
            if configured_max <= thinking_budget:
                raise ProviderError(
                    f"extra_body.max_tokens {configured_max} is not above the thinking budget"
                    f" {thinking_budget} (effort {level}); raise it or lower the effort",
                    fatal=True,
                )

        return ProviderCall(
            api_label="Anthropic",
            api_format="anthropic",
            url=url,
            body=body,
            timeout_s=self.timeout_s,
            api_key=self.api_key,
            credential=self.credential,
            transcript_sink=self.transcript_sink,
            budget=self.budget,
            model=self.model,
            build_headers=self._build_headers,
            adapt_400=self._adapt_body_for_400,
            adapt_attempts=int("temperature" in body),
            require_metered=lambda data: _require_metered_usage(
                data.get("usage"), source="Anthropic response"
            ),
            parse=_parse_response,
            stream=(
                lambda attempt_headers: self._call_streaming(
                    url=url,
                    headers=attempt_headers,
                    body=body,
                    text_delta_callback=text_delta_callback,
                    thinking_delta_callback=thinking_delta_callback,
                    should_abort=should_abort,
                    should_interrupt=should_interrupt,
                )
            )
            if streaming
            else None,
        ).run()

    def _call_streaming(  # noqa: C901, PLR0915  # one streaming state machine; a split hides the event order
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        """Make the call over SSE; this method owns the Messages event shape.

        Deltas fan to their callbacks as they arrive; at `message_stop` the response
        is synthesised in the non-streaming shape, so no caller needs a streaming path.

        Args:
            url: The URL dialled.
            headers: The attempt's request headers.
            body: The request body.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning text as it streams.
            should_abort: Polled each watchdog tick; True abandons the turn.
            should_interrupt: Polled each watchdog tick; True ends the turn to steer.

        Returns:
            The parsed response.

        Raises:
            ProviderError: A stream error frame, a malformed event, a stream cut
                before `message_stop`, or missing usage on a budgeted run; what the
                cut turn already cost is recorded first.
        """
        body = dict(body)
        # Vertex selects streaming by the URL and rejects a `stream` body field.
        if self.deployment == "direct":
            body["stream"] = True
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        content_blocks: list[dict[str, Any]] = []
        # Per-index builders; the wire indexes a message's blocks and opens one at a time.
        text_acc: dict[int, list[str]] = {}
        tool_acc: dict[int, dict[str, Any]] = {}
        json_partial: dict[int, list[str]] = {}
        unknown_acc: dict[int, dict[str, Any]] = {}
        open_blocks: set[int] = set()
        # A thinking block's signature must round-trip, or a tool call after it breaks next turn.
        thinking_acc: dict[int, list[str]] = {}
        signature_acc: dict[int, list[str]] = {}
        stop_reason: str = ""
        # A clean EOF before `message_stop` is a cut mid-message, not a completion.
        saw_message_stop = False
        usage_input = 0
        usage_output = 0
        usage_cache_read = 0
        usage_cache_creation = 0
        saw_input_usage = False
        saw_output_usage = False

        call = SseCall(
            api_label="Anthropic",
            api_format="anthropic",
            url=url,
            headers=stream_headers,
            body=body,
            timeout_s=self.timeout_s,
            transcript_sink=self.transcript_sink,
            should_abort=should_abort,
            should_interrupt=should_interrupt,
        )

        def consume(resp: httpx2.Response, clock: StreamClock) -> None:  # noqa: C901, PLR0912, PLR0915  # one streaming state machine; a split hides the event order
            """Read the stream's events into the accumulators.

            Raises:
                ProviderError: A malformed event or a stream error frame.
            """
            nonlocal stop_reason, saw_message_stop, usage_input, usage_output
            nonlocal usage_cache_read, usage_cache_creation, saw_input_usage, saw_output_usage
            for event_type, data_str in sse_events(resp):
                if not data_str:
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    raise ProviderError("Anthropic stream event was not JSON") from exc
                et = event_type or str(evt.get("type", ""))
                # `message_start` is metadata, so output is marked only once a content block starts.
                if et != "ping":
                    clock.mark_data()
                if et in ("content_block_start", "content_block_delta"):
                    clock.mark_output()
                if et == "message_start":
                    msg = evt.get("message")
                    if not isinstance(msg, Mapping):
                        raise ProviderError(
                            "Anthropic response message_start.message was not an object"
                        )
                    u = _usage_mapping(msg.get("usage"))
                    if "input_tokens" in u and u.get("input_tokens") is not None:
                        saw_input_usage = True
                    usage_input = _usage_count(u, "input_tokens")
                    usage_output = _usage_count(u, "output_tokens")
                    usage_cache_read = _usage_count(u, "cache_read_input_tokens")
                    usage_cache_creation = _usage_count(u, "cache_creation_input_tokens")
                elif et == "content_block_start":
                    idx = _non_negative_integer(evt.get("index", 0), "content block index")
                    if idx in open_blocks:
                        raise ProviderError(f"Anthropic content block {idx} started twice")
                    open_blocks.add(idx)
                    cb = evt.get("content_block")
                    if not isinstance(cb, Mapping):
                        raise ProviderError("Anthropic response content block was not an object")
                    btype = _response_string(cb.get("type"), "content block type", empty=False)
                    if btype == "text":
                        text_acc[idx] = [_response_string(cb.get("text", ""), "content text")]
                    elif btype == "thinking":
                        clock.enter_thinking()
                        thinking_acc[idx] = [
                            _response_string(cb.get("thinking", ""), "content thinking")
                        ]
                        signature_acc[idx] = [
                            _response_string(cb.get("signature", ""), "content signature")
                        ]
                    elif btype == "redacted_thinking":
                        content_blocks.append(
                            {
                                "type": "redacted_thinking",
                                "data": _response_string(
                                    cb.get("data", ""),
                                    "content redacted_thinking data",
                                    empty=False,
                                ),
                            }
                        )
                    elif btype == "tool_use":
                        tool_input = cb.get("input", {})
                        if not isinstance(tool_input, dict):
                            raise ProviderError(
                                "Anthropic response content tool_use.input was not an object"
                            )
                        tool_acc[idx] = {
                            "type": "tool_use",
                            "id": _response_string(
                                cb.get("id"), "content tool_use.id", empty=False
                            ),
                            "name": _response_string(
                                cb.get("name"), "content tool_use.name", empty=False
                            ),
                            "input": tool_input,
                        }
                        json_partial[idx] = []
                    else:
                        # A block type this client does not know is kept as it arrived.
                        unknown_acc[idx] = dict(cb)
                elif et == "content_block_delta":
                    idx = _non_negative_integer(evt.get("index", 0), "content block index")
                    d = evt.get("delta")
                    if not isinstance(d, Mapping):
                        raise ProviderError("Anthropic response content delta was not an object")
                    dt = _response_string(d.get("type"), "content delta type", empty=False)
                    if idx in unknown_acc:
                        raise ProviderError(
                            f"Anthropic content block {idx} of type"
                            f" {unknown_acc[idx].get('type')!r} streamed a {dt}"
                        )
                    if dt == "text_delta":
                        piece = _response_string(d.get("text", ""), "content text delta")
                        text_acc.setdefault(idx, []).append(piece)
                        if piece and text_delta_callback is not None:
                            # A callback is a cosmetic surface; its failure never breaks the stream.
                            with contextlib.suppress(Exception):
                                text_delta_callback(piece)
                    elif dt == "thinking_delta":
                        piece = _response_string(d.get("thinking", ""), "content thinking delta")
                        thinking_acc.setdefault(idx, []).append(piece)
                        if piece and thinking_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                thinking_delta_callback(piece)
                    elif dt == "signature_delta":
                        signature_acc.setdefault(idx, []).append(
                            _response_string(d.get("signature", ""), "content signature delta")
                        )
                    elif dt == "input_json_delta":
                        json_partial.setdefault(idx, []).append(
                            _response_string(d.get("partial_json", ""), "content input JSON delta")
                        )
                elif et == "content_block_stop":
                    idx = _non_negative_integer(evt.get("index", 0), "content block index")
                    if idx not in open_blocks:
                        raise ProviderError(
                            f"Anthropic content block {idx} stopped before it started"
                        )
                    open_blocks.remove(idx)
                    if idx in text_acc:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": "".join(text_acc.pop(idx)),
                            }
                        )
                    elif idx in thinking_acc:
                        clock.exit_thinking()
                        block_out: dict[str, Any] = {
                            "type": "thinking",
                            "thinking": "".join(thinking_acc.pop(idx)),
                        }
                        sig = "".join(signature_acc.pop(idx, []))
                        if not sig:
                            raise ProviderError(
                                "Anthropic content thinking block omitted its signature"
                            )
                        block_out["signature"] = sig
                        content_blocks.append(block_out)
                    elif idx in tool_acc:
                        tu = tool_acc.pop(idx)
                        partial = "".join(json_partial.pop(idx, []))
                        if partial:
                            try:
                                tu["input"] = json.loads(partial)
                            except json.JSONDecodeError as exc:
                                raise ProviderError(
                                    "Anthropic content tool_use input JSON was incomplete"
                                ) from exc
                        content_blocks.append(tu)
                    elif idx in unknown_acc:
                        content_blocks.append(unknown_acc.pop(idx))
                elif et == "message_delta":
                    d = evt.get("delta")
                    if not isinstance(d, Mapping):
                        raise ProviderError(
                            "Anthropic response message_delta.delta was not an object"
                        )
                    if "stop_reason" in d:
                        stop_reason = _response_string(
                            d.get("stop_reason"), "stop_reason", empty=False
                        )
                    u = _usage_mapping(evt.get("usage"))
                    if u.get("output_tokens") is not None:
                        saw_output_usage = True
                        usage_output = _usage_count(u, "output_tokens")
                elif et == "message_stop":
                    if open_blocks:
                        idx = min(open_blocks)
                        raise ProviderError(
                            f"Anthropic message stopped before content block {idx} stopped"
                        )
                    saw_message_stop = True
                    return
                elif et == "error":
                    err = evt.get("error")
                    if isinstance(err, Mapping):
                        label = err.get("type") or err.get("code") or "error"
                        detail = err.get("message") or err
                        status = envelope_status(err)
                    else:
                        label = "error"
                        detail = err or evt.get("message") or "unknown stream error"
                        status = None
                    # The frame is recorded first; the upstream status keeps a permanent error so.
                    call.record(status=0, response=data_str[:8192])
                    detail_text = scrub_secret_values(str(detail), headers)
                    raise ProviderError(
                        f"Anthropic stream error: {label}: {detail_text}",
                        status_code=status,
                    )

        def _record_billed() -> None:
            """Record what the turn cost so far."""
            record_billed_usage(
                self.budget,
                self.model,
                input_tokens=usage_input,
                output_tokens=usage_output,
                cache_read_tokens=usage_cache_read,
                cache_creation_tokens=usage_cache_creation,
            )

        try:
            call.run(consume)
        except BaseException:
            # Input usage arrives in `message_start`, so a turn ended early was billed.
            _record_billed()
            raise

        # A truncated turn returned as finished would read as the model going quiet.
        if not saw_message_stop:
            _record_billed()
            call.record(status=0, response="stream ended without message_stop (truncated)")
            raise ProviderError(
                f"Anthropic SSE stream from {url} ended prematurely "
                "(no message_stop); upstream appears cut off."
            )

        synthesised: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage_input,
                "output_tokens": usage_output,
                "cache_read_input_tokens": usage_cache_read,
                "cache_creation_input_tokens": usage_cache_creation,
            },
        }
        call.record(status=200, response=synthesised)
        if self.budget is not None:
            try:
                if not (saw_input_usage and saw_output_usage):
                    raise ProviderError(
                        "Anthropic stream omitted usage.input_tokens/output_tokens; "
                        "budgeted runs require provider usage accounting"
                    )
                _require_metered_usage(synthesised.get("usage"), source="Anthropic stream")
            except ProviderError:
                _record_billed()
                raise
        try:
            parsed = _parse_response(synthesised)
        except ProviderError:
            _record_billed()
            raise
        meter_completion(self.budget, self.model, parsed, "Anthropic")
        return parsed


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    """Parse one Messages body.

    Args:
        data: The response body.

    Returns:
        The response in agent6's canonical shape.

    Raises:
        ProviderError: The body is malformed (retryable).
    """
    content = data.get("content") or []
    if not isinstance(content, list):
        raise ProviderError(
            f"Anthropic response `content` was {type(content).__name__}, not a"
            " list (malformed 2xx from upstream gateway)"
        )
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    tool_use_ids: set[str] = set()
    for block in content:
        if not isinstance(block, dict):
            raise ProviderError(
                "Anthropic response content block was not an object"
                " (malformed 2xx from upstream gateway)"
            )
        block_type = _response_string(block.get("type"), "content block type", empty=False)
        if block_type == "text":
            text_parts.append(_response_string(block.get("text", ""), "content text"))
        elif block_type == "tool_use":
            tool_input = block.get("input", {})
            if not isinstance(tool_input, dict):
                raise ProviderError("Anthropic response content tool_use.input was not an object")
            tool_use_id = _response_string(block.get("id"), "content tool_use.id", empty=False)
            if tool_use_id in tool_use_ids:
                raise ProviderError(
                    f"Anthropic response had duplicate content tool_use.id {tool_use_id!r}"
                )
            tool_use_ids.add(tool_use_id)
            tool_uses.append(
                {
                    "id": tool_use_id,
                    "name": _response_string(
                        block.get("name"), "content tool_use.name", empty=False
                    ),
                    "input": tool_input,
                }
            )
        elif block_type == "thinking":
            _response_string(block.get("thinking", ""), "content thinking")
            _response_string(block.get("signature", ""), "content signature", empty=False)
        elif block_type == "redacted_thinking":
            _response_string(block.get("data", ""), "content redacted_thinking data", empty=False)
    usage = _usage_mapping(data.get("usage"))
    return ProviderResponse(
        text="\n\n".join(text_parts),
        tool_uses=tuple(tool_uses),
        stop_reason=_response_string(data.get("stop_reason"), "stop_reason", empty=False),
        input_tokens=_usage_count(usage, "input_tokens"),
        output_tokens=_usage_count(usage, "output_tokens"),
        cache_read_tokens=_usage_count(usage, "cache_read_input_tokens"),
        cache_creation_tokens=_usage_count(usage, "cache_creation_input_tokens"),
        raw=data,
    )
