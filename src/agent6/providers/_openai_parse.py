# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Parse a Chat Completions response into a `ProviderResponse`.

The response-parsing half of the OpenAI provider, shared by its non-streaming
path and its synthesised streaming response: tool_calls become tool_uses,
reasoning becomes a leading thinking block in `raw`, and prompt_tokens are
normalised to fresh-input semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent6.providers import _openai_recovery, types


def response_string(value: Any, field: str, *, empty: bool = True) -> str:
    """Return a response field as a string.

    Args:
        value: The field's value.
        field: The field's name, for the error.
        empty: Whether a blank string is accepted.

    Raises:
        ProviderError: The value is not a string, or is blank when one is required.
    """
    if not isinstance(value, str) or (not empty and not value.strip()):
        qualifier = "nonempty " if not empty else ""
        raise types.ProviderError(f"OpenAI response {field} was not a {qualifier}string")
    return value


def usage_mapping(value: Any) -> Mapping[str, Any]:
    """Return the response's `usage` object, empty when absent.

    Raises:
        ProviderError: The value is present but not an object.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise types.ProviderError("OpenAI response usage was not an object")
    return value


def usage_count(usage: Mapping[str, Any], field: str, *, path: str = "") -> int:
    """Read one non-negative integer count out of a usage object.

    Args:
        usage: The usage object.
        field: The field's key.
        path: The field's place in the body, for the error, when nested.

    Returns:
        The count; 0 when the field is absent.

    Raises:
        ProviderError: The value is not a non-negative integer.
    """  # noqa: DOC501  # the TypeError is raised and caught in the same try
    name = path or field
    value = usage.get(field)
    if value is None:
        return 0
    try:
        if isinstance(value, bool):
            raise TypeError
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise types.ProviderError(
            f"OpenAI response usage.{name} was not a non-negative integer"
        ) from exc
    if count < 0 or (isinstance(value, float) and not value.is_integer()):
        raise types.ProviderError(f"OpenAI response usage.{name} was not a non-negative integer")
    return count


def parse_response(  # noqa: C901, PLR0912, PLR0915  # one branch per provider dialect
    data: dict[str, Any],
    *,
    tool_names: frozenset[str] = frozenset(),
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> types.ProviderResponse:
    """Parse one Chat Completions body.

    Args:
        data: The response body.
        tool_names: The tools offered; a call the model wrote into its text is
            recovered only when this is non-empty and no native call exists.
        tool_schemas: The offered tools' input schemas, for coercing recovered
            Qwen-XML parameter strings.

    Returns:
        The response in agent6's canonical shape.

    Raises:
        ProviderError: The body is malformed (a retryable failure, never an
            AttributeError that would bypass the retry wrapper).
    """
    choices = data.get("choices")
    if choices is None:
        choices = []
    if not isinstance(choices, list):
        raise types.ProviderError("OpenAI response choices was not an array")
    text = ""
    reasoning_text = ""
    stop_reason = ""
    tool_uses: tuple[dict[str, Any], ...] = ()
    if choices:
        first = choices[0]
        if not isinstance(first, dict):
            raise types.ProviderError(
                f"OpenAI choices[0] is {type(first).__name__}, not an object (malformed 2xx body)"
            )
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise types.ProviderError("OpenAI response choices[0].message was not an object")
        raw_text = message.get("content")
        text = "" if raw_text is None else response_string(raw_text, "content")
        # Kimi spells it `reasoning_content`; DeepSeek-R1 and OpenRouter spell it `reasoning`.
        raw_reasoning = message.get("reasoning_content")
        if raw_reasoning is None:
            raw_reasoning = message.get("reasoning")
        reasoning_text = (
            "" if raw_reasoning is None else response_string(raw_reasoning, "reasoning")
        )
        raw_stop = first.get("finish_reason")
        stop_reason = "" if raw_stop is None else response_string(raw_stop, "finish_reason")
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise types.ProviderError("OpenAI response tool_calls was not an array")
        parsed_calls: list[dict[str, Any]] = []
        tool_call_ids: set[str] = set()
        for i, call in enumerate(raw_calls):
            if not isinstance(call, Mapping):
                raise types.ProviderError("OpenAI response tool_call was not an object")
            func = call.get("function")
            if not isinstance(func, Mapping):
                raise types.ProviderError("OpenAI response tool_call.function was not an object")
            # A blank-name native call (some open-weight backends) never enters history.
            raw_name = func.get("name") or ""
            name = response_string(raw_name, "tool_call.function.name")
            if not name.strip():
                continue
            raw_args = func.get("arguments")
            args_raw = response_string(
                "" if raw_args is None else raw_args, "tool_call.function.arguments"
            )
            # The `_raw_arguments` diagnostic is capped so a degenerate payload cannot re-enter.
            raw_args_cap = 500
            try:
                parsed_input = json.loads(args_raw) if args_raw else {}
                if not isinstance(parsed_input, dict):
                    parsed_input = {"_value": parsed_input}
            except (json.JSONDecodeError, TypeError):
                # A lenient re-parse saves a round-trip; dispatch turns the sentinel into an error.
                repaired = _openai_recovery.lenient_json_object(args_raw)
                if repaired is not None:
                    parsed_input = repaired
                else:
                    raw_str = str(args_raw)
                    if len(raw_str) > raw_args_cap:
                        raw_str = (
                            raw_str[:raw_args_cap]
                            + f"... <truncated; original was {len(str(args_raw))} chars>"
                        )
                    parsed_input = {"_raw_arguments": raw_str}
            raw_id = call.get("id")
            tool_call_id = (
                f"call_auto_{i}"
                if raw_id is None or raw_id == ""
                else response_string(raw_id, "tool_call.id", empty=False)
            )
            if tool_call_id in tool_call_ids:
                raise types.ProviderError(
                    f"OpenAI response had duplicate tool_call.id {tool_call_id!r}"
                )
            tool_call_ids.add(tool_call_id)
            parsed_calls.append(
                {
                    "id": tool_call_id,
                    "name": name,
                    "input": parsed_input,
                }
            )
        tool_uses = tuple(parsed_calls)
        # Native calls take precedence over a call leaked into the text.
        if not tool_uses and tool_names:
            recovered, remaining_text = _openai_recovery.coerce_text_tool_calls(
                text, tool_names, tool_schemas
            )
            if recovered:
                tool_uses = tuple(
                    {"id": f"call_text_{i}", "name": r["name"], "input": r["input"]}
                    for i, r in enumerate(recovered)
                )
                text = remaining_text
    usage = usage_mapping(data.get("usage"))
    # `prompt_tokens` is the whole prompt; `input_tokens` means fresh input, as Anthropic counts it.
    cached = 0
    details = usage.get("prompt_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        raise types.ProviderError("OpenAI response usage.prompt_tokens_details was not an object")
    if isinstance(details, Mapping):
        cached = usage_count(details, "cached_tokens", path="prompt_tokens_details.cached_tokens")
    prompt_total = usage_count(usage, "prompt_tokens")
    # The clamp keeps input_tokens non-negative when an upstream reports cached > prompt.
    cached = min(cached, prompt_total)
    fresh_input = prompt_total - cached
    # `raw["content"]` mirrors Anthropic's block shape; the loop rebuilds the assistant message.
    raw_content: list[dict[str, Any]] = []
    if reasoning_text:
        # Reasoning stays out of `text`, or every surface echoing it would print it twice.
        raw_content.append({"type": "thinking", "thinking": reasoning_text})
    if text:
        raw_content.append({"type": "text", "text": text})
    for tu in tool_uses:
        raw_content.append(
            {
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": tu["input"],
            }
        )
    enriched_raw = {**data, "content": raw_content}
    # A gateway's `usage.cost` (OpenRouter) is authoritative; a bool would read as a phantom dollar.
    reported_cost = 0.0
    raw_cost = usage.get("cost")
    if isinstance(raw_cost, int | float) and not isinstance(raw_cost, bool) and raw_cost > 0:
        reported_cost = float(raw_cost)
    return types.ProviderResponse(
        text=text,
        tool_uses=tool_uses,
        stop_reason=stop_reason,
        input_tokens=fresh_input,
        output_tokens=usage_count(usage, "completion_tokens"),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        cost_usd=reported_cost,
        raw=enriched_raw,
    )
