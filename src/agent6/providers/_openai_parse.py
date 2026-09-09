# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions response parsing.

The response-parsing half of the provider: choices[0].message ->
`ProviderResponse` in agent6's canonical Anthropic shape (tool_calls ->
tool_uses, reasoning_content -> a leading thinking block in `raw`,
prompt_tokens normalised to fresh-input semantics). Both the non-streaming
path and the synthesised streaming response in `providers/openai.py` call
it; the text-embedded tool-call fallback lives in
`providers/_openai_recovery.py`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent6.providers._openai_recovery import (
    coerce_text_tool_calls,
    lenient_json_object,
)
from agent6.providers.types import ProviderError, ProviderResponse


def response_string(value: Any, field: str, *, empty: bool = True) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        qualifier = "nonempty " if not empty else ""
        raise ProviderError(f"OpenAI response {field} was not a {qualifier}string")
    return value


def usage_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProviderError("OpenAI response usage was not an object")
    return value


def usage_count(usage: Mapping[str, Any], field: str, *, path: str = "") -> int:
    """A non-negative integer count, named by *path* (the field's place in the
    body) when it is nested."""
    name = path or field
    value = usage.get(field)
    if value is None:
        return 0
    try:
        if isinstance(value, bool):
            raise TypeError
        count = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderError(f"OpenAI response usage.{name} was not a non-negative integer") from exc
    if count < 0 or (isinstance(value, float) and not value.is_integer()):
        raise ProviderError(f"OpenAI response usage.{name} was not a non-negative integer")
    return count


def parse_response(  # noqa: PLR0912, PLR0915
    data: dict[str, Any],
    *,
    tool_names: frozenset[str] = frozenset(),
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> ProviderResponse:
    choices = data.get("choices")
    if choices is None:
        choices = []
    if not isinstance(choices, list):
        raise ProviderError("OpenAI response choices was not an array")
    text = ""
    reasoning_text = ""
    stop_reason = ""
    tool_uses: tuple[dict[str, Any], ...] = ()
    if choices:
        first = choices[0]
        if not isinstance(first, dict):
            # A malformed 2xx (choices[0] null/string from a flaky local
            # endpoint) must surface as a retryable ProviderError, not an
            # AttributeError that bypasses the loop's retry wrapper.
            raise ProviderError(
                f"OpenAI choices[0] is {type(first).__name__}, not an object (malformed 2xx body)"
            )
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ProviderError("OpenAI response choices[0].message was not an object")
        raw_text = message.get("content")
        text = "" if raw_text is None else response_string(raw_text, "content")
        # Kimi (`reasoning_content`), DeepSeek-R1 /
        # OpenRouter (`reasoning`), and OpenAI o-series surface
        # reasoning in a sibling field. Capture both spellings.
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
            raise ProviderError("OpenAI response tool_calls was not an array")
        parsed_calls: list[dict[str, Any]] = []
        tool_call_ids: set[str] = set()
        for i, call in enumerate(raw_calls):
            if not isinstance(call, Mapping):
                raise ProviderError("OpenAI response tool_call was not an object")
            func = call.get("function")
            if not isinstance(func, Mapping):
                raise ProviderError("OpenAI response tool_call.function was not an object")
            # Small open-weight models (via some OpenRouter backends,
            # Novita backend) sometimes emit a NATIVE tool_call with a blank
            # `function.name`. Dispatching it yields "Unknown tool: " and, worse,
            # echoing the blank-name call back in the next request makes strict
            # backends reject the whole conversation with a 400
            # invalid_request_error, killing the run. Drop the malformed call
            # here so it never enters history; valid calls in the same turn
            # still proceed.
            raw_name = func.get("name") or ""
            name = response_string(raw_name, "tool_call.function.name")
            if not name.strip():
                continue
            raw_args = func.get("arguments")
            args_raw = response_string(
                "" if raw_args is None else raw_args, "tool_call.function.arguments"
            )
            # OpenAI returns arguments as a JSON string. Convert to dict
            # for the Anthropic-shape input field. Malformed JSON
            # surfaces as an empty dict + the raw string under
            # `_raw_arguments` so debugging is possible.
            #
            # A degenerate tool-arg payload (tens of KB of repeated escape
            # sequences) would be echoed in the tool_error message and re-enter
            # the context, priming the same degeneration next turn. Cap the
            # diagnostic at 500 chars so the
            # repetition doesn't survive the round-trip.
            _RAW_ARGS_CAP = 500
            try:
                parsed_input = json.loads(args_raw) if args_raw else {}
                if not isinstance(parsed_input, dict):
                    parsed_input = {"_value": parsed_input}
            except (json.JSONDecodeError, TypeError):
                # Before giving up, try a lenient re-parse. Weak/open models
                # commonly emit args that strict JSON rejects: a raw newline in a
                # multiline code/regex param, or trailing junk (a leaked
                # `</invoke>` / prose). Recovering here means the tool just runs,
                # instead of a wasted round-trip on a validation error. Only a
                # parse that yields an object is accepted, so a bad guess can't
                # feed the handler garbage; anything still unparseable becomes the
                # `_raw_arguments` sentinel (dispatch turns that into a clear
                # "resend valid JSON" error).
                repaired = lenient_json_object(args_raw)
                if repaired is not None:
                    parsed_input = repaired
                else:
                    raw_str = str(args_raw)
                    if len(raw_str) > _RAW_ARGS_CAP:
                        raw_str = (
                            raw_str[:_RAW_ARGS_CAP]
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
                raise ProviderError(f"OpenAI response had duplicate tool_call.id {tool_call_id!r}")
            tool_call_ids.add(tool_call_id)
            parsed_calls.append(
                {
                    # Synthesise a distinct id when the backend omits one
                    # (some open-weight models stream tool_calls with no id).
                    # Two native tool_calls both with id="" would otherwise
                    # collapse to ambiguous/duplicate tool_call_id pairing on
                    # the next request, tripping a strict-backend 400. Mirrors
                    # the call_text_{i} fallback used for recovered calls.
                    "id": tool_call_id,
                    "name": name,
                    "input": parsed_input,
                }
            )
        tool_uses = tuple(parsed_calls)
        # Fallback: no NATIVE tool_calls but the model leaked a tool call
        # into its text content (small local models via Ollama/llama.cpp).
        # Guarded by `tool_names` so this only fires when tools were offered.
        # Every form must name one of them except the explicit `<tool_call>`
        # tag, which keeps an unknown name for the dispatcher's error. Native
        # calls take precedence.
        if not tool_uses and tool_names:
            recovered, remaining_text = coerce_text_tool_calls(text, tool_names, tool_schemas)
            if recovered:
                tool_uses = tuple(
                    {"id": f"call_text_{i}", "name": r["name"], "input": r["input"]}
                    for i, r in enumerate(recovered)
                )
                text = remaining_text
    usage = usage_mapping(data.get("usage"))
    # OpenAI's cached_tokens field, when present, lives under
    # usage.prompt_tokens_details.cached_tokens. Treat absent as 0.
    #
    # Provider-format asymmetry: Anthropic's `input_tokens`
    # already EXCLUDES cache-read tokens (they're surfaced separately under
    # `cache_read_input_tokens`). OpenAI's `prompt_tokens`, by contrast, is
    # the TOTAL prompt size, cached + fresh. We normalise to Anthropic's
    # semantics here so `ProviderResponse.input_tokens` consistently means
    # "fresh, non-cached input" across providers. Without this, the
    # BudgetTracker would charge cached tokens against the input-token cap
    # at full rate (causing premature budget exhaustion on cache-heavy
    # OpenAI runs) AND the cost formula in budget.py would double-count the
    # cache portion (full input rate plus an additional 10% cache-read
    # surcharge).
    cached = 0
    details = usage.get("prompt_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        raise ProviderError("OpenAI response usage.prompt_tokens_details was not an object")
    if isinstance(details, Mapping):
        cached = usage_count(details, "cached_tokens", path="prompt_tokens_details.cached_tokens")
    # `or 0` throughout: a gateway returning `"prompt_tokens": null` on a 2xx
    # would make bare int(None) raise TypeError, which escapes the loop's
    # ProviderError-only retry wrapper and kills the run.
    prompt_total = usage_count(usage, "prompt_tokens")
    # Clamp cached to the prompt total as the SINGLE source of truth: a
    # misbehaving upstream that reports cached > prompt would otherwise drive
    # input_tokens negative AND leave cache_read_tokens -- billed at the 10%
    # cache rate in budget.py -- inconsistent with it. One clamp keeps both
    # fields consistent.
    cached = min(cached, prompt_total)
    fresh_input = prompt_total - cached
    # Build a content-blocks raw payload mirroring Anthropic's response
    # shape so callers that inspect resp.raw["content"] (the worker_loop
    # does this to reconstruct the assistant message verbatim) see the
    # same structure regardless of provider.
    raw_content: list[dict[str, Any]] = []
    if reasoning_text:
        # A leading `thinking` block, so every provider yields one shape.
        # Reasoning is NOT promoted into `text`: surfaces that echo
        # `resp.text` (CLI logger, transcripts) would double-print it.
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
    # Prefer provider-reported USD cost when the upstream
    # gateway includes it (OpenRouter does, OpenAI direct does not).
    # Treat negative or non-numeric values as absent.
    reported_cost = 0.0
    raw_cost = usage.get("cost")
    # not-bool: bool subclasses int, and float(True) == 1.0 would record a
    # phantom dollar per call that becomes the AUTHORITATIVE reported figure.
    if isinstance(raw_cost, int | float) and not isinstance(raw_cost, bool) and raw_cost > 0:
        reported_cost = float(raw_cost)
    return ProviderResponse(
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
