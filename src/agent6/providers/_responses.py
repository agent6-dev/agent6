# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Responses API codec and stream machine the ChatGPT and generic adapters share.

Messages in Anthropic shape become `input` items, output items become the canonical
response, and one SSE consumer folds the event stream into `ResponsesStreamState`. A
reasoning item is replayed only to the route that produced it: the item rides a
`thinking` block under `reasoning_replay` with the route's fingerprint, never as display
text.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx2

from agent6.providers import _openai_messages, _openai_recovery, _stream, types

# The thinking-block key an opaque reasoning item rides under, with its route.
REPLAY_KEY = "reasoning_replay"
# Each carries the final response object; `response.failed` carries an envelope instead.
TERMINAL_EVENTS = frozenset({"response.completed", "response.done", "response.incomplete"})
# Body keys the adapter or the harness owns; `extra_body` never reaches them: retention,
# remote continuation, the output cap and the output format stay agent6's.
RESERVED_BODY = frozenset(
    {
        "background",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "response_format",
        "store",
        "stream",
        "text",
        "tool_choice",
        "tools",
    }
)


def replay_route(api_format: str, base_url: str, model: str) -> str:
    """Return the fingerprint of the route a reasoning item may be replayed to.

    Args:
        api_format: The wire format.
        base_url: The endpoint.
        model: The model.

    Returns:
        A hex digest.
    """
    canonical = json.dumps([api_format, base_url.rstrip("/"), model])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def responses_input(messages: list[dict[str, Any]], *, route: str) -> list[dict[str, Any]]:
    """Translate Anthropic-shaped messages into Responses `input` items.

    A text run is one message item, a `tool_use` a `function_call`, a `tool_result` a
    `function_call_output` with its content as a string. A `thinking` block carrying a
    `reasoning_replay` made on this route replays its item in place, only with its
    following kept item; any other `thinking` block (another route's, a legacy item with
    no route, display-only text) is dropped.

    Args:
        messages: The conversation in Anthropic shape.
        route: This call's `replay_route`.

    Returns:
        The input items, in wire order.
    """
    items: list[dict[str, Any]] = []
    # A blank-name or id-less tool_use is dropped, and its paired tool_result with it.
    dropped_ids: set[str] = set()
    for msg in messages:
        role = str(msg.get("role", "user"))
        if role not in ("user", "assistant"):
            role = "user"
        content = msg.get("content", "")
        if isinstance(content, list):
            items.extend(_content_items(role, content, dropped_ids, route))
        elif content:
            items.append(_message_item(role, str(content)))
    return items


def replay_item(block: dict[str, Any], route: str) -> dict[str, Any] | None:
    """Return the reasoning item a thinking block replays on this route, else None."""
    replay = block.get(REPLAY_KEY)
    if not isinstance(replay, dict) or replay.get("route") != route:
        return None
    item = replay.get("item")
    return item if isinstance(item, dict) and item.get("type") == "reasoning" else None


def _content_items(
    role: str, blocks: list[Any], dropped_ids: set[str], route: str
) -> list[dict[str, Any]]:
    """Return one message's content blocks as input items, text runs batched."""
    items: list[dict[str, Any]] = []
    text_run: list[str] = []
    # An orphaned reasoning item violates the wire's pairing rules and 400s the request.
    pending_reasoning: list[dict[str, Any]] = []

    def flush() -> None:
        """Emit the pending text run as one message item."""
        if text_run:
            items.append(_message_item(role, "\n\n".join(text_run)))
            text_run.clear()

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = str(block.get("text", ""))
            if not text:
                continue
            items.extend(pending_reasoning)
            pending_reasoning.clear()
            text_run.append(text)
        elif btype == "thinking" and role == "assistant":
            item = replay_item(block, route)
            if item is not None:
                flush()
                pending_reasoning.append(item)
        elif btype == "tool_use" and role == "assistant":
            call_id = str(block.get("id") or "")
            if not str(block.get("name") or "").strip() or not call_id.strip():
                dropped_ids.add(call_id)
                pending_reasoning.clear()
                continue
            flush()
            items.extend(pending_reasoning)
            pending_reasoning.clear()
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": str(block.get("name", "")),
                    "arguments": json.dumps(block.get("input") or {}),
                }
            )
        elif btype == "tool_result":
            if str(block.get("tool_use_id", "")) in dropped_ids:
                continue
            flush()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(block.get("tool_use_id", "")),
                    "output": _openai_messages.tool_result_text(block.get("content", "")),
                }
            )
    flush()
    return items


def _message_item(role: str, text: str) -> dict[str, Any]:
    """Return one message item of the role's text kind."""
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def tools_to_responses(tools: list[types.ToolDefinition]) -> list[dict[str, Any]]:
    """Return the tools as Responses function tools (flat, not nested)."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
            "strict": False,
        }
        for t in tools
    ]


def _tool_use_of(item: dict[str, Any], *, n: int) -> dict[str, Any] | None:
    """Translate a `function_call` item into a tool_use.

    Args:
        item: The output item.
        n: The call's ordinal, for a synthesised id.

    Returns:
        The tool_use, or None for a blank name, which never enters history.
        Unparseable arguments keep the lenient repair or the `_raw_arguments`
        sentinel so dispatch can ask for a resend.
    """
    name = str(item.get("name", "")).strip()
    if not name:
        return None
    args_raw = item.get("arguments", "")
    try:
        parsed = json.loads(args_raw) if args_raw else {}
        if not isinstance(parsed, dict):
            parsed = {"_value": parsed}
    except (json.JSONDecodeError, TypeError):
        repaired = _openai_recovery.lenient_json_object(str(args_raw))
        parsed = repaired if repaired is not None else {"_raw_arguments": str(args_raw)[:500]}
    return {
        "id": str(item.get("call_id") or item.get("id") or f"call_{n}"),
        "name": name,
        "input": parsed,
    }


def usage_count(value: Any, field_name: str, *, label: str) -> int:
    """Return one usage count; 0 when absent.

    Args:
        value: The decoded value; a bool or a fractional float is refused.
        field_name: The field's path under `usage`, for the error.
        label: The API the error names.

    Returns:
        The count.

    Raises:
        ProviderError: The value is not a non-negative integer.
    """  # noqa: DOC501  # the TypeError is raised and caught in the same try
    if value is None:
        return 0
    try:
        if isinstance(value, bool):
            raise TypeError
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise types.ProviderError(
            f"{label} response usage.{field_name} was not a non-negative integer"
        ) from exc
    if count < 0 or (isinstance(value, float) and not value.is_integer()):
        raise types.ProviderError(
            f"{label} response usage.{field_name} was not a non-negative integer"
        )
    return count


def parse_output_items(
    items: list[Any], *, usage: Mapping[str, Any], stop_reason: str, label: str, route: str
) -> types.ProviderResponse:
    """Parse the final output items into a response.

    `raw["content"]` holds one block per item in wire order: a message's text, a reasoning
    item as a `thinking` block (its summary for display, the raw item under
    `reasoning_replay` with this route for replay), a function_call as `tool_use`.

    Args:
        items: The output items.
        usage: The terminal usage object; `input_tokens` is the cached plus fresh total,
            normalised to fresh-only as the OpenAI parser does.
        stop_reason: The stop reason read off the terminal event.
        label: The API the errors name.
        route: This call's `replay_route`.

    Returns:
        The response in agent6's canonical shape, its cost unpriced.
    """
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            text = "".join(
                str(part.get("text", ""))
                for part in item.get("content") or []
                if isinstance(part, dict) and part.get("type") in ("output_text", "text")
            )
            if text:
                text_parts.append(text)
                blocks.append({"type": "text", "text": text})
        elif itype == "reasoning":
            summary = "\n\n".join(
                str(part["text"])
                for part in item.get("summary") or []
                if isinstance(part, dict) and str(part.get("text", ""))
            )
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": summary,
                    REPLAY_KEY: {"wire": "responses", "route": route, "item": item},
                }
            )
        elif itype == "function_call":
            tool_use = _tool_use_of(item, n=len(tool_uses))
            if tool_use is not None:
                tool_uses.append(tool_use)
                blocks.append({"type": "tool_use", **tool_use})
    text = "\n\n".join(text_parts)
    if tool_uses and stop_reason == "end_turn":
        # A turn that stopped to call tools says so, which arms the loop's empty-call detector.
        stop_reason = "tool_use"
    details = usage.get("input_tokens_details")
    cached = (
        usage_count(details.get("cached_tokens"), "input_tokens_details.cached_tokens", label=label)
        if isinstance(details, Mapping)
        else 0
    )
    prompt_total = usage_count(usage.get("input_tokens"), "input_tokens", label=label)
    cached = min(cached, prompt_total)
    return types.ProviderResponse(
        text=text,
        tool_uses=tuple(tool_uses),
        stop_reason=stop_reason,
        input_tokens=prompt_total - cached,
        output_tokens=usage_count(usage.get("output_tokens"), "output_tokens", label=label),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        cost_usd=0.0,
        raw={"content": blocks, "usage": dict(usage), "output": items},
    )


def stop_reason_of(response: Mapping[str, Any], status: str) -> str:
    """Return the canonical stop reason of a terminal response object.

    Args:
        response: The response object.
        status: Its status: `completed`, `done` or `incomplete`.

    Returns:
        `max_tokens` for an output cap, the reported reason for another incompleteness,
        else `end_turn`.
    """
    if status != "incomplete":
        return "end_turn"
    reason = str((response.get("incomplete_details") or {}).get("reason") or "")
    return "max_tokens" if reason == "max_output_tokens" else (reason or "incomplete")


def stream_error(
    evt: dict[str, Any], *, label: str, usage_limit_codes: frozenset[str] = frozenset()
) -> types.ProviderError:
    """Return the error a `response.failed` or `error` frame carries, classified.

    Args:
        evt: The frame.
        label: The API the error names.
        usage_limit_codes: Codes that mean a plan window is exhausted, carried as a 429 so
            the loop backs off rather than failing.

    Returns:
        The error, with the status the retry policy reads.
    """
    response = evt.get("response")
    err = response.get("error") if isinstance(response, dict) else evt.get("error", evt)
    if not isinstance(err, dict):
        err = {"message": str(err or evt.get("message") or "unknown stream error")}
    code = str(err.get("code") or "")
    status = 429 if code in usage_limit_codes else None
    detail = str(err.get("message") or code or "unknown error")
    if code == "unsupported_value" and err.get("param") == "reasoning.effort":
        status = 400
        detail += ' (param: "reasoning.effort")'
    if code in usage_limit_codes:
        plan = str(err.get("plan_type") or "")
        detail += f" ({label} {plan} plan usage limit)" if plan else f" ({label} usage limit)"
    return types.ProviderError(
        f"{label} stream error: {code or 'error'}: {detail}", status_code=status
    )


@dataclasses.dataclass(slots=True)
class ResponsesStreamState:
    """What one Responses stream has delivered so far.

    Attributes:
        items: The final output items, from `output_item.done` or the terminal response.
        delta_text: The visible text pieces, in order.
        usage: The usage object, from the terminal or the failure response.
        stop_reason: The canonical stop reason once the terminal event arrived.
        done: Whether the terminal event arrived.
    """

    items: list[Any] = dataclasses.field(default_factory=list)
    delta_text: list[str] = dataclasses.field(default_factory=list)
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    stop_reason: str = ""
    done: bool = False


def consume_stream(  # noqa: PLR0912
    call: _stream.SseCall,
    resp: httpx2.Response,
    clock: _stream.StreamClock,
    state: ResponsesStreamState,
    *,
    label: str,
    error_of: Callable[[dict[str, Any]], types.ProviderError],
    text_delta_callback: Callable[[str], None] | None,
    thinking_delta_callback: Callable[[str], None] | None,
) -> None:
    """Read a Responses event stream into the state.

    Deltas feed the callbacks only; the content comes from the `output_item.done` items,
    reconciled against the terminal response object, where usage lives.

    Args:
        call: The SSE call, which records the failure frame.
        resp: The open 2xx response.
        clock: The watchdog's idle clock; every event marks it, and the first output ends
            prefill.
        state: The accumulator.
        label: The API the errors name.
        error_of: Classifies a failure frame.
        text_delta_callback: Receives visible text as it streams.
        thinking_delta_callback: Receives reasoning summaries as they stream.

    Raises:
        ProviderError: A failed response or an unknown terminal status.
    """  # `error_of` builds the ProviderError named above
    for _event, data in _stream.sse_events(resp):
        clock.mark_data()
        data_str = data.strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            evt: dict[str, Any] = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        kind = str(evt.get("type", ""))
        if kind == "response.output_text.delta":
            piece = str(evt.get("delta", ""))
            if piece:
                clock.mark_output()
                state.delta_text.append(piece)
                if text_delta_callback is not None:
                    with contextlib.suppress(Exception):
                        text_delta_callback(piece)
        elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            piece = str(evt.get("delta", ""))
            if piece:
                clock.mark_output()
                if thinking_delta_callback is not None:
                    with contextlib.suppress(Exception):
                        thinking_delta_callback(piece)
        elif kind == "response.output_item.done":
            clock.mark_output()
            state.items.append(evt.get("item"))
        elif kind in ("response.failed", "error"):
            failed = evt.get("response")
            failed_usage = failed.get("usage") if isinstance(failed, dict) else None
            if isinstance(failed_usage, dict):
                state.usage = failed_usage
            call.record(status=0, response=data_str[:8192])
            raise error_of(evt)
        elif kind in TERMINAL_EVENTS:
            _terminal(state, evt, kind, data_str, call=call, label=label, error_of=error_of)
            return


def _terminal(
    state: ResponsesStreamState,
    evt: dict[str, Any],
    kind: str,
    data_str: str,
    *,
    call: _stream.SseCall,
    label: str,
    error_of: Callable[[dict[str, Any]], types.ProviderError],
) -> None:
    """Fold the terminal event into the state.

    Args:
        state: The accumulator.
        evt: The terminal frame.
        kind: Its event type.
        data_str: Its raw data, recorded on a failure.
        call: The SSE call, which records the failure frame.
        label: The API the errors name.
        error_of: Classifies a failure frame.

    Raises:
        ProviderError: The response failed or ended with an unknown status.
    """  # `error_of` builds the ProviderError named above
    response = evt.get("response") or {}
    evt_usage = response.get("usage")
    if isinstance(evt_usage, dict):
        state.usage = evt_usage
    status = str(response.get("status") or kind.removeprefix("response."))
    if status == "failed":
        call.record(status=0, response=data_str[:8192])
        raise error_of(evt)
    if status not in ("completed", "incomplete", "done"):
        call.record(status=0, response=data_str[:8192])
        raise types.ProviderError(f"{label} response ended with status {status}")
    final_items = response.get("output")
    if isinstance(final_items, list) and final_items:
        # The terminal response is the whole output; item.done events may be absent.
        state.items[:] = final_items
    state.stop_reason = stop_reason_of(response, status)
    state.done = True


def finish_stream(
    state: ResponsesStreamState, *, call: _stream.SseCall, url: str, label: str, route: str
) -> types.ProviderResponse:
    """Parse a finished stream and record its transcript entry.

    Args:
        state: The accumulator after the terminal event.
        call: The SSE call, which records the round-trip.
        url: The URL dialled, for the truncation error.
        label: The API the errors name.
        route: This call's `replay_route`.

    Returns:
        The parsed response; text deltas stand in for a missing final message item.

    Raises:
        ProviderError: The stream ended without a terminal response event (truncated).
    """
    if not state.done:
        call.record(status=0, response="stream ended without a terminal response event")
        raise types.ProviderError(
            f"{label} stream from {url} ended without response.completed; upstream appears cut off."
        )
    parsed = parse_output_items(
        state.items, usage=state.usage, stop_reason=state.stop_reason, label=label, route=route
    )
    if not parsed.text and state.delta_text:
        # Text deltas without a final message item keep what the operator watched arrive.
        text = "".join(state.delta_text)
        parsed = dataclasses.replace(
            parsed,
            text=text,
            raw={**parsed.raw, "content": [{"type": "text", "text": text}, *parsed.raw["content"]]},
        )
    call.record(
        status=200,
        response={
            "output": parsed.raw.get("output", []),
            "usage": state.usage,
            "status": state.stop_reason,
        },
    )
    return parsed
