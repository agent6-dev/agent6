# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The provider for any endpoint speaking the OpenAI Chat Completions API.

OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp's server, Moonshot and
DeepSeek. The transport and the SSE lifecycle are shared with the Anthropic
provider. Anthropic content blocks are agent6's internal shape; the translation
both ways lives in `_openai_messages` and `_openai_parse`. `cache_control` markers
are dropped (this wire caches server-side), and reasoning is the
`reasoning_effort` knob from `[models.<role>].effort`.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers._openai_messages import anthropic_to_openai_messages, tools_to_openai
from agent6.providers._openai_parse import parse_response, response_string
from agent6.providers._stream import SseCall, StreamClock, record_billed_usage, sse_events
from agent6.providers._transport import ProviderCall, envelope_status, meter_completion
from agent6.providers.token_command import CommandToken
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptRecorder,
)
from agent6.providers.wire import AuthStyle, Deployment, auth_header, request_url

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192

# Reasoning tokens count against `max_tokens`; below this floor they starve the answer.
REASONING_MODEL_MIN_MAX_TOKENS = 32768
_REASONING_MODEL_HINTS: tuple[str, ...] = (
    "thinking",
    "reasoning",
    "deepseek-r1",
    "qwq",
    "o1-",
    "o3-",
    "o4-",
    # Families that reason without saying so in the name.
    "kimi-k",
    "minimax-m2",
    "nemotron",
    "glm",
)


def _require_metered_usage(usage: object, *, source: str) -> None:
    """Refuse a budgeted call whose usage cannot be metered.

    A gateway with usage tracking off returns `prompt_tokens: 0`, so presence is
    not enough; the total input is never 0 for a real call, so it must be positive.

    Args:
        usage: The response's `usage` value.
        source: The name that leads the error.

    Raises:
        ProviderError: No positive input usage (retryable: a usage-less reply is a
            gateway integrity failure, and a permanent class would let one mangled
            stream end a budgeted run).
    """
    if isinstance(usage, Mapping):
        # A count serialised as a float or a string is meterable, as `usage_count` reads it.
        try:
            prompt = int(usage.get("prompt_tokens") or 0)
        except (TypeError, ValueError):
            prompt = 0
        if prompt > 0:
            return
    raise ProviderError(
        f"{source} reported no usage input tokens (usage.prompt_tokens missing or 0); "
        "budgeted runs require provider usage accounting"
    )


def _is_reasoning_model(model: str) -> bool:
    """Return whether the model reasons in a separate channel.

    This set gates the effort default, a measured behaviour, so it stays the
    measured families; the token floor matches more broadly.
    """
    lowered = model.lower()
    return any(hint in lowered for hint in _REASONING_MODEL_HINTS)


# Aliases that get the token floor but not the effort default (the floor changes no behaviour).
_REASONING_FLOOR_ONLY_HINTS: tuple[str, ...] = ("kimi-latest",)


def _needs_reasoning_headroom(model: str) -> bool:
    """Return whether the model gets the `max_tokens` floor; a false positive costs nothing."""
    lowered = model.lower()
    return (
        _is_reasoning_model(model)
        or _is_openai_direct_reasoning_model(model)
        or any(h in lowered for h in _REASONING_FLOOR_ONLY_HINTS)
    )


# OpenAI's own reasoning families, which on api.openai.com reject `max_tokens` and a temperature.
_OPENAI_DIRECT_REASONING_PREFIXES: tuple[str, ...] = ("o1", "o3", "o4", "gpt-5")


def _is_openai_direct_reasoning_model(model: str) -> bool:
    """Return whether the model is one of OpenAI's own reasoning families."""
    lowered = model.lower()
    return any(
        lowered == p or lowered.startswith(p + "-") for p in _OPENAI_DIRECT_REASONING_PREFIXES
    )


_EFFORT_LEVELS = ("off", "low", "medium", "high", "xhigh", "max")


def is_openai_direct_host(base_url: str, deployment: str) -> bool:
    """Return whether requests go to api.openai.com itself, whose parameters differ."""
    return deployment == "direct" and urlsplit(base_url).hostname == "api.openai.com"


def sent_reasoning_effort(
    model: str, configured: str | None, *, direct_openai: bool = False
) -> str | None:
    """Resolve the reasoning effort a call sends; the one owner of the rule `config show` prints.

    Args:
        model: The model id.
        configured: The role's `effort` or a per-call override; None defers to the
            `AGENT6_REASONING_EFFORT` variable, then `low`.
        direct_openai: Whether the request targets api.openai.com.

    Returns:
        The level, or None when the model takes no reasoning knob. `off` is a
        level, not a wire value: the call omits the parameter on api.openai.com
        and sends `{"enabled": false}` elsewhere.
    """
    if not (
        _is_reasoning_model(model) or (direct_openai and _is_openai_direct_reasoning_model(model))
    ):
        return None
    if configured is not None:
        return configured.strip().lower()
    env_override = os.environ.get("AGENT6_REASONING_EFFORT", "").strip().lower()
    return env_override if env_override in _EFFORT_LEVELS else "low"


@dataclass(frozen=True, slots=True)
class OpenAIProvider:
    """The Chat Completions provider, constructed once per run.

    Attributes:
        api_key: The static credential; "" sends no auth header (a local endpoint).
        model: The model id.
        base_url: The endpoint's base URL.
        deployment: The URL profile.
        auth_style: The auth header style: `bearer`, `api_key_header` (Azure) or `none`.
        extra_headers: Operator headers merged over the built ones.
        extra_body: Operator body keys merged last; the structural keys are reserved.
        extra_query: Operator query parameters (Azure's `api-version`).
        timeout_s: The read budget in seconds.
        transcript_sink: Where each round-trip is recorded.
        budget: The run's tracker; None skips metering.
        reasoning_effort: The role's effort; a per-call argument takes precedence.
        credential: A short-lived bearer source; a 401 or 403 re-mints it once.
    """

    api_key: str
    model: str
    base_url: str = OPENAI_DEFAULT_BASE_URL
    deployment: Deployment = "direct"
    auth_style: AuthStyle = "bearer"
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_query: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    transcript_sink: TranscriptRecorder | None = None
    budget: BudgetTracker | None = None
    reasoning_effort: str | None = None
    credential: CommandToken | None = None
    # Latched on a parameter-rejection 400 (an Azure reasoning deployment has an arbitrary name)
    # so the rest of the run builds the right body first; lists, since the dataclass is frozen.
    _use_max_completion_tokens: list[bool] = field(default_factory=lambda: [False])
    _omit_temperature: list[bool] = field(default_factory=lambda: [False])

    @property
    def endpoint(self) -> str:
        """The direct chat completions URL."""
        return self.base_url.rstrip("/") + "/chat/completions"

    def _adapt_body_for_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Rewrite the body for a parameter-rejection 400 and latch the adaptation.

        Covers "use max_completion_tokens" and "temperature is not supported".

        Args:
            status: The HTTP status.
            text: The error text.
            body: The request body, rewritten in place.

        Returns:
            Whether the body was adapted, so the transport retries once.
        """
        if status != 400:
            return False
        if "max_tokens" in body and "max_completion_tokens" in (text or ""):
            self._use_max_completion_tokens[0] = True
            body["max_completion_tokens"] = body.pop("max_tokens")
            return True
        if "temperature" in body and "temperature" in (text or "").lower():
            self._omit_temperature[0] = True
            body.pop("temperature", None)
            return True
        return False

    def _build_headers(self, token: str) -> dict[str, str]:
        """Return one attempt's request headers, built from its token."""
        headers: dict[str, str] = {"content-type": "application/json"}
        authed = auth_header(self.auth_style, token)
        if authed is not None:
            headers[authed[0]] = authed[1]
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        return headers

    def call(  # noqa: PLR0912
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        extended_thinking: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> ProviderResponse:
        """Make one Chat Completions call, streaming when a delta callback is set.

        Args:
            system: The system prompt.
            messages: The conversation in Anthropic shape.
            tools: The tools the model may call.
            max_tokens: The output cap; lifted to the reasoning floor for a reasoning model.
            temperature: The sampling temperature; omitted where the host rejects one.
            extended_thinking: Ignored; this wire has no equivalent of a thinking budget.
            reasoning_effort: A per-call effort over the provider's own.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning text as it streams.
            should_abort: Polled during a stream; True abandons the turn.
            should_interrupt: Polled during a stream; True ends the turn to steer.

        Returns:
            The parsed response.

        Raises:
            ProviderError: The run is over budget, or the call failed.
        """
        del extended_thinking
        if self.budget is not None:
            self.budget.check()

        oai_messages = anthropic_to_openai_messages(system, messages)

        effective_max_tokens = max_tokens
        if (
            _needs_reasoning_headroom(self.model)
            and effective_max_tokens < REASONING_MODEL_MIN_MAX_TOKENS
        ):
            effective_max_tokens = REASONING_MODEL_MIN_MAX_TOKENS

        streaming = text_delta_callback is not None or thinking_delta_callback is not None
        url, model_in_body = request_url(
            api_format="openai",
            deployment=self.deployment,
            base_url=self.base_url,
            model=self.model,
            streaming=streaming,
            extra_query=self.extra_query,
        )
        # Only api.openai.com's own reasoning models reject `max_tokens` and a temperature.
        is_openai_direct = is_openai_direct_host(self.base_url, self.deployment)
        is_openai_direct_reasoning = is_openai_direct and _is_openai_direct_reasoning_model(
            self.model
        )
        body: dict[str, Any] = {"messages": oai_messages}
        if is_openai_direct_reasoning or self._use_max_completion_tokens[0]:
            body["max_completion_tokens"] = effective_max_tokens
        else:
            body["max_tokens"] = effective_max_tokens
        if model_in_body:
            body["model"] = self.model
        # The reasoning knob differs per host and the wrong one is ignored, never rejected:
        # OpenRouter takes a nested `reasoning.effort` and needs `{"enabled": false}` for off;
        # api.openai.com takes a top-level `reasoning_effort` and cannot switch reasoning off.
        effort = sent_reasoning_effort(
            self.model,
            reasoning_effort if reasoning_effort is not None else self.reasoning_effort,
            direct_openai=is_openai_direct,
        )
        if effort is not None:
            if is_openai_direct_reasoning:
                if effort != "off":
                    body["reasoning_effort"] = effort
            elif effort == "off":
                body["reasoning"] = {"enabled": False}
            else:
                body["reasoning"] = {"effort": effort}
        if (
            temperature is not None
            and not is_openai_direct_reasoning
            and not self._omit_temperature[0]
        ):
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools_to_openai(tools)
        if self.extra_body:
            # The structural keys stay agent6's; tuning keys merge last and win.
            reserved = {
                "messages",
                "model",
                "stream",
                "stream_options",
                "tools",
                "tool_choice",
                "response_format",
                "n",
            }
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})
        # The recovery of a call leaked into text is guarded by the tools offered this turn.
        tool_names = frozenset(t.name for t in tools) if tools else frozenset()
        tool_schemas = {t.name: t.input_schema for t in tools} if tools else {}

        # Streaming is the only reliable path through a gateway whose heartbeats corrupt a body.
        return ProviderCall(
            api_label="OpenAI",
            api_format="openai",
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
            adapt_attempts=int("max_tokens" in body) + int("temperature" in body),
            require_metered=lambda data: _require_metered_usage(
                data.get("usage"), source="OpenAI response"
            ),
            parse=lambda data: parse_response(
                data, tool_names=tool_names, tool_schemas=tool_schemas
            ),
            stream=(
                lambda attempt_headers: self._call_streaming(
                    url=url,
                    headers=attempt_headers,
                    body=body,
                    text_delta_callback=text_delta_callback,
                    thinking_delta_callback=thinking_delta_callback,
                    should_abort=should_abort,
                    should_interrupt=should_interrupt,
                    tool_names=tool_names,
                    tool_schemas=tool_schemas,
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
        tool_names: frozenset[str] = frozenset(),
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> ProviderResponse:
        """Make the call over SSE; this method owns the Chat Completions event shape.

        Each frame is one `data:` JSON object whose `choices[0].delta` carries text,
        reasoning (`reasoning_content` or `reasoning`) or indexed `tool_calls` whose
        arguments arrive in pieces; usage lands in a trailing chunk with empty
        `choices`, and `[DONE]` ends the stream.

        Args:
            url: The URL dialled.
            headers: The attempt's request headers.
            body: The request body.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning text as it streams.
            should_abort: Polled each watchdog tick; True abandons the turn.
            should_interrupt: Polled each watchdog tick; True ends the turn to steer.
            tool_names: The tools offered, guarding the recovery of a leaked call.
            tool_schemas: The offered tools' input schemas.

        Returns:
            The parsed response.

        Raises:
            ProviderError: A stream error frame, a malformed delta, a stream cut
                before its end, or missing usage on a budgeted run; what the cut
                turn already cost is recorded first.
        """
        body = dict(body)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Keyed by the chunk's `index`; a call's id sometimes arrives late.
        tool_calls: dict[int, dict[str, Any]] = {}
        tool_arg_buf: dict[int, list[str]] = {}
        finish_reason = ""
        usage: dict[str, Any] = {}
        # A stream ending with neither `[DONE]` nor a finish_reason was cut, not completed.
        done_seen = False

        call = SseCall(
            api_label="OpenAI",
            api_format="openai",
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
                ProviderError: A malformed delta or a stream error frame.
            """
            nonlocal finish_reason, usage, done_seen
            # An empty role delta arrives at once, so output is marked on the first content token.
            for _event, data in sse_events(resp):
                clock.mark_data()
                data_str = data.strip()
                if data_str == "[DONE]":
                    done_seen = True
                    return
                if not data_str:
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # A gateway delivers an upstream error as a frame; its status keeps the retry class.
                err = evt.get("error")
                if isinstance(err, dict):
                    call.record(status=0, response=data_str[:8192])
                    raise ProviderError(
                        f"OpenAI stream error: {err.get('code')}: {err.get('message')}",
                        status_code=envelope_status(err),
                    )
                evt_usage = evt.get("usage")
                if isinstance(evt_usage, dict):
                    usage = evt_usage
                choices = evt.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                fr = choice.get("finish_reason")
                if fr is not None:
                    finish_reason = response_string(fr, "finish_reason")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if content is not None and not isinstance(content, str):
                    raise ProviderError("OpenAI response content delta was not a string")
                if isinstance(content, str) and content:
                    clock.mark_output()
                    text_parts.append(content)
                    if text_delta_callback is not None:
                        with contextlib.suppress(Exception):
                            text_delta_callback(content)
                reasoning = delta.get("reasoning_content")
                if reasoning is None:
                    reasoning = delta.get("reasoning")
                if reasoning is not None and not isinstance(reasoning, str):
                    raise ProviderError("OpenAI response reasoning delta was not a string")
                if isinstance(reasoning, str) and reasoning:
                    clock.mark_output()
                    reasoning_parts.append(reasoning)
                    if thinking_delta_callback is not None:
                        with contextlib.suppress(Exception):
                            thinking_delta_callback(reasoning)
                raw_tc = delta.get("tool_calls") or []
                if not isinstance(raw_tc, list):
                    continue
                if raw_tc:
                    clock.mark_output()
                for tc in raw_tc:
                    if not isinstance(tc, dict):
                        continue
                    raw_idx = tc.get("index")
                    raw_id = tc.get("id")
                    if raw_id is not None and not isinstance(raw_id, str):
                        raise ProviderError("OpenAI response tool_call.id delta was not a string")
                    tc_id = raw_id or ""
                    if raw_idx is not None:
                        idx = int(raw_idx)
                    elif tc_id and any(s["id"] == tc_id for s in tool_calls.values()):
                        idx = next(i for i, s in tool_calls.items() if s["id"] == tc_id)
                    elif tc_id and tool_calls:
                        # An indexless chunk with a new id opens its own slot rather than slot 0.
                        idx = max(tool_calls) + 1
                    else:
                        idx = max(tool_calls) if tool_calls else 0
                    slot = tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc.get("id"):
                        slot["id"] = str(tc["id"])
                    func = tc.get("function")
                    if func is None:
                        continue
                    if not isinstance(func, dict):
                        raise ProviderError(
                            "OpenAI response tool_call.function delta was not an object"
                        )
                    name = func.get("name")
                    if name is not None and not isinstance(name, str):
                        raise ProviderError(
                            "OpenAI response tool_call.function.name delta was not a string"
                        )
                    if name:
                        slot["function"]["name"] = name
                    args_piece = func.get("arguments")
                    if args_piece is not None and not isinstance(args_piece, str):
                        raise ProviderError(
                            "OpenAI response tool_call.function.arguments delta was not a string"
                        )
                    if args_piece:
                        tool_arg_buf.setdefault(idx, []).append(args_piece)

        def _record_billed() -> None:
            """Record what the turn cost so far, through the one owner of the usage mapping."""
            if not usage:
                return
            billed = parse_response(
                {"choices": [], "usage": usage},
                tool_names=tool_names,
                tool_schemas=tool_schemas,
            )
            record_billed_usage(
                self.budget,
                self.model,
                input_tokens=billed.input_tokens,
                output_tokens=billed.output_tokens,
                cache_read_tokens=billed.cache_read_tokens,
                cache_creation_tokens=billed.cache_creation_tokens,
                cost_usd=billed.cost_usd,
            )

        try:
            call.run(consume)
        except BaseException:
            # A usage shape the parser refuses must not replace the reason the stream ended.
            with contextlib.suppress(ProviderError):
                _record_billed()
            raise

        # A truncated turn returned as finished would read as the model going quiet.
        if not done_seen and not finish_reason:
            _record_billed()
            call.record(
                status=0,
                response="stream ended without [DONE] or finish_reason (truncated)",
            )
            raise ProviderError(
                f"OpenAI stream from {url} ended prematurely "
                "(no [DONE], no finish_reason); upstream appears cut off."
            )

        final_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls):
            slot = tool_calls[idx]
            args = "".join(tool_arg_buf.get(idx, []))
            slot["function"]["arguments"] = args
            final_tool_calls.append(slot)

        # The recorded body is read back as a transcript, so it carries the role.
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if final_tool_calls:
            message["tool_calls"] = final_tool_calls

        synthesised: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        if self.budget is not None and not done_seen and not usage:
            # The usage chunk follows finish_reason, so a stream stopped between them was cut.
            call.record(
                status=0,
                response="stream cut before its usage trailer (truncated)",
            )
            raise ProviderError(
                f"OpenAI stream from {url} was cut off before its usage trailer"
                " (finish_reason seen, no [DONE], no usage); truncated response."
            )
        call.record(status=200, response=synthesised)
        if self.budget is not None:
            try:
                _require_metered_usage(usage, source="OpenAI stream")
            except ProviderError:
                _record_billed()
                raise
        parsed = parse_response(synthesised, tool_names=tool_names, tool_schemas=tool_schemas)
        meter_completion(self.budget, self.model, parsed, "OpenAI")
        return parsed
