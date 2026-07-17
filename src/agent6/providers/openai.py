# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions-compatible provider.

Works against any endpoint that speaks the OpenAI Chat Completions API:
OpenAI itself, OpenRouter, Ollama (`/v1`), vLLM, LM Studio, llama.cpp's
server, Kimi via Moonshot, DeepSeek-V3 via the official API or via
OpenRouter. Any sub-agent role (planner, worker, critic, reviewer,
summarizer) can be routed through this provider via
`[models.<role>]` in your config.

HTTP transport and SSE lifecycle are shared with the Anthropic provider
(`providers/_transport.py`, `providers/_stream.py`); both use httpx2
directly (no SDK) for a smaller audit surface.

**- tool-use translation (Shape B)**: agent6's internal
"lingua franca" is Anthropic content-blocks (the most expressive
format - text + tool_use + tool_result inline). OpenAI Chat
Completions uses a parallel `tool_calls` array on assistant
messages and a separate `role=tool` message for tool results.
This provider translates IN/OUT internally so the workflow code
(worker_loop, architect_loop) sees uniform Anthropic-shape
behaviour across providers. Translation covers:

- Anthropic `tool_use` block in assistant content -> OpenAI
  `tool_calls` array (function-shape).
- Anthropic `tool_result` block in user message -> OpenAI separate
  `role=tool` message with `tool_call_id`.
- `ToolDefinition` -> OpenAI `tools=[{"type":"function","function":
  {...}}]`.
- OpenAI `choices[0].message.tool_calls` -> agent6's
  `ProviderResponse.tool_uses` tuple (Anthropic shape: id / name /
  input).

What's still per-provider (intentionally not abstracted):

- Anthropic `cache_control` markers: OpenAI does automatic prompt
  caching server-side; no explicit marker needed; we strip them.
- Anthropic `extended_thinking={"type":"enabled","budget_tokens":N}`:
  OpenAI's reasoning models use `reasoning_effort` not budget
  tokens; the params don't translate 1:1. Currently we silently no-op the
  Anthropic-shape param on OpenAI; if you want OpenAI reasoning,
  add a separate per-provider knob later.
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
from agent6.providers._openai_parse import parse_response as _parse_response
from agent6.providers._stream import SseCall, StreamClock
from agent6.providers._transport import ProviderCall
from agent6.providers.token_command import CommandToken
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
)
from agent6.providers.wire import AuthStyle, Deployment, auth_header, request_url

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192

# Kimi-K2-Thinking, DeepSeek-R1, QwQ, and similar reasoning
# models stream a separate ``reasoning_content`` (or ``reasoning``)
# field whose tokens count against ``max_tokens`` on the server side.
# At our default per-call cap of 16384, reasoning eats the budget and
# the actual assistant ``content`` / ``tool_calls`` get truncated
# mid-message - the loop then sees stop_reason="length" with an empty
# text and no tool calls, and stalls. Bump the floor for these models
# so reasoning has room without changing the budget tracker (which
# still accounts for every emitted token via usage.completion_tokens).
REASONING_MODEL_MIN_MAX_TOKENS = 32768
_REASONING_MODEL_HINTS: tuple[str, ...] = (
    "thinking",
    "reasoning",
    "deepseek-r1",
    "qwq",
    "o1-",
    "o3-",
    "o4-",
    # Reasoning channel emitters that do NOT advertise it in the
    # model name. Each entry below was added after observing
    # finish_reason="length" with empty content + empty tool_calls and a
    # populated reasoning_content field in the raw response:
    #   - kimi-k2:    Moonshot's K2.x family (k2.5, k2.6, k2.6-thinking,
    #                 k2-0905, etc.). K2.6 starves at the 16k loop default
    #                 (~16k reasoning tokens for a single perf-takehome
    #                 turn), then the loop sees no tool_use and emits
    #                 went_quiet.
    #   - minimax-m2: minimax-m2 / minimax-m2.7 (and presumably future
    #                 m2.x). Same pattern observed on click-short-help
    #                 and werkzeug-safe-join (19k reasoning tokens).
    #   - nemotron:   NVIDIA Nemotron-3 nano (e.g. nemotron-3-nano-30b-a3b)
    #                 streams reasoning_content even on the non-"reasoning"
    #                 variant; observed starving (loop.reasoning_starvation)
    #                 at the 16k default on the synthetic edit tasks.
    #   - glm:        Zhipu GLM-4.x/5.x (z-ai/glm-4.6, glm-4.7, glm-5.2).
    #                 All stream a separate ``reasoning`` channel; a direct
    #                 OpenRouter probe at max_tokens=40 returned
    #                 finish_reason="length" with empty content/tool_calls and
    #                 ~all 40 tokens charged as reasoning_tokens. The "v"
    #                 vision variants (glm-4.5v etc.) match too, which is fine:
    #                 they reason as well, and the floor only raises a ceiling.
    "kimi-k2",
    "minimax-m2",
    "nemotron",
    "glm",
)


def _require_metered_usage(usage: object, *, source: str) -> None:
    """Fail closed when a budgeted OpenAI-compatible call cannot be metered.

    Presence alone is not enough: a gateway with usage tracking disabled returns
    ``prompt_tokens: 0`` and every turn records zero, so the budget never trips.
    ``prompt_tokens`` is total input (cached + fresh) and is never legitimately 0
    for a real call, so require it strictly positive; a run must not proceed on a
    call it cannot meter."""
    if isinstance(usage, Mapping):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int) and prompt > 0:
            return
    raise ProviderError(
        f"{source} reported no usage input tokens (usage.prompt_tokens missing or 0); "
        "budgeted runs require provider usage accounting",
        status_code=422,
    )


def _is_reasoning_model(model: str) -> bool:
    """True if ``model`` looks like a reasoning model that emits
    ``reasoning_content`` separately from ``content``."""
    lowered = model.lower()
    return any(hint in lowered for hint in _REASONING_MODEL_HINTS)


# OpenAI's OWN reasoning families (o-series + gpt-5). On the api.openai.com
# direct host these reject the legacy ``max_tokens`` param (400, "Use
# max_completion_tokens") and reject ``temperature != 1``. Kept narrower than
# ``_is_reasoning_model`` on purpose: third-party reasoning models (kimi,
# deepseek, qwq) are never served by api.openai.com, so they must NOT trigger
# the rename even if someone points them at the default base_url.
_OPENAI_DIRECT_REASONING_PREFIXES: tuple[str, ...] = ("o1", "o3", "o4", "gpt-5")


def _is_openai_direct_reasoning_model(model: str) -> bool:
    """True if ``model`` is one of OpenAI's own o-series/gpt-5 reasoning
    models (only meaningful when the request targets api.openai.com)."""
    lowered = model.lower()
    return any(
        lowered == p or lowered.startswith(p + "-") for p in _OPENAI_DIRECT_REASONING_PREFIXES
    )


@dataclass(frozen=True, slots=True)
class OpenAIProvider:
    """Stateless OpenAI Chat Completions-compatible provider.

    `api_key` may be empty for unauthenticated local endpoints (Ollama,
    llama.cpp's `server`); when empty, no `Authorization` header is sent.
    """

    api_key: str
    model: str
    base_url: str = OPENAI_DEFAULT_BASE_URL
    deployment: Deployment = "direct"
    # Auth header style (config AuthConfig.style): "bearer" (default),
    # "api_key_header" (Azure's `api-key`), or "none" (local endpoints).
    auth_style: AuthStyle = "bearer"
    extra_headers: tuple[tuple[str, str], ...] = ()
    # Provider-specific JSON merged into every request body (e.g. OpenRouter
    # `provider` routing, see config OpenAIProviderEntry.extra_body). Keys here
    # override computed body fields, EXCEPT the load-bearing
    # messages/model/stream/stream_options (filtered in `call`).
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Static URL query params merged onto every request (e.g. Azure's
    # api-version). See config extra_query.
    extra_query: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    transcript_sink: TranscriptSink | None = None
    budget: BudgetTracker | None = None
    # Default reasoning effort for this provider (off|low|medium|high), wired
    # from the role's `[models.<role>].thinking`. A per-call `reasoning_effort`
    # argument takes precedence; below this sits the AGENT6_REASONING_EFFORT
    # env override. Only affects OpenAI-compatible reasoning models.
    reasoning_effort: str | None = None
    # Short-lived bearer source (config `token_command`). When set, it mints
    # the `Authorization` token per call instead of `api_key`, and a 401/403
    # triggers one refresh + retry. The object is internally mutable (cache),
    # which is why the otherwise-frozen provider holds only a reference to it.
    credential: CommandToken | None = None
    # Some OpenAI-compatible backends we cannot fingerprint up front (an Azure
    # o-series/gpt-5 deployment has an arbitrary deployment name) reject the
    # legacy ``max_tokens`` with a 400 saying to use ``max_completion_tokens``,
    # and/or reject any explicit ``temperature``. On that 400 the call adapts
    # the body and retries once, latching here so the rest of the run builds
    # the right body first time. 1-element lists because the dataclass is
    # frozen but the lists are mutable (same pattern as AnthropicProvider).
    _use_max_completion_tokens: list[bool] = field(default_factory=lambda: [False])
    _omit_temperature: list[bool] = field(default_factory=lambda: [False])

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _adapt_body_for_400(self, status: int | None, text: str, body: dict[str, Any]) -> bool:
        """Mutate ``body`` to satisfy a parameter-rejection 400 and latch the
        provider so later calls build the right body first time. Covers the
        two rejections a reasoning deployment we cannot fingerprint up front
        (an Azure o-series/gpt-5 deployment has an arbitrary name) sends:
        "use max_completion_tokens" and "temperature is not supported".
        Returns True when an adaptation was made (caller retries once)."""
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
        """Per-attempt request headers. Rebuilt each attempt because a
        `token_command` credential mints a short-lived bearer that takes
        precedence over the static api_key; on a 401/403 the transport
        refreshes it once and retries, so an expired token self-heals."""
        headers: dict[str, str] = {"content-type": "application/json"}
        authed = auth_header(self.auth_style, token)
        if authed is not None:
            headers[authed[0]] = authed[1]
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        return headers

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        env_var: str | None,
        base_url: str = OPENAI_DEFAULT_BASE_URL,
        extra_headers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
        transcript_sink: TranscriptSink | None = None,
        budget: BudgetTracker | None = None,
    ) -> OpenAIProvider:
        # env_var is optional: Ollama and similar local endpoints take no
        # API key. If it's set, we still allow empty (treated as "no key").
        key = "" if env_var is None else os.environ.get(env_var, "").strip()
        return cls(
            api_key=key,
            model=model,
            base_url=base_url,
            extra_headers=tuple(sorted((extra_headers or {}).items())),
            timeout_s=timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
        )

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
        # extended_thinking is Anthropic-shaped (`budget_tokens`).
        # OpenAI reasoning models use `reasoning_effort` instead; no
        # 1:1 mapping. Silently no-op so cross-provider workflow code
        # doesn't have to branch.
        del extended_thinking
        if self.budget is not None:
            self.budget.check()

        oai_messages = anthropic_to_openai_messages(system, messages)

        # Lift max_tokens for reasoning models so reasoning_content
        # doesn't starve the actual assistant content + tool_calls. See
        # REASONING_MODEL_MIN_MAX_TOKENS for the rationale.
        effective_max_tokens = max_tokens
        if (
            _is_reasoning_model(self.model)
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
        # OpenAI-direct o-series/reasoning models (o1/o3/o4/gpt-5-style)
        # REJECT the legacy ``max_tokens`` parameter with a hard 400
        # ("Use max_completion_tokens"), and reject ``temperature != 1``.
        # They are reached only on the OpenAI-direct host; other
        # openai-compatible hosts (OpenRouter, Azure, vLLM, llama.cpp) still
        # require ``max_tokens`` and accept arbitrary temperature, so gate the
        # rename on host + model. OpenRouter masked this by normalising
        # ``max_tokens`` -> ``max_completion_tokens`` itself.
        is_openai_direct = (
            self.deployment == "direct" and urlsplit(self.base_url).hostname == "api.openai.com"
        )
        is_openai_direct_reasoning = is_openai_direct and _is_openai_direct_reasoning_model(
            self.model
        )
        body: dict[str, Any] = {"messages": oai_messages}
        if is_openai_direct_reasoning or self._use_max_completion_tokens[0]:
            body["max_completion_tokens"] = effective_max_tokens
        else:
            body["max_tokens"] = effective_max_tokens
        # Direct/Vertex carry the model in the body; Azure carries the
        # deployment name in the URL path, so omit it from the body there.
        if model_in_body:
            body["model"] = self.model
        # Cap reasoning tokens on OpenRouter-style gateways.
        #
        # Background: Kimi K2.6 (and other reasoning models) routinely
        # spend their *entire* ``max_tokens`` budget on
        # ``reasoning_content`` alone, returning empty
        # ``content`` + empty ``tool_calls`` with ``finish_reason=length``.
        # added ``REASONING_MODEL_MIN_MAX_TOKENS=32768`` so a
        # full reasoning burst still leaves room for one tool call --
        # but on perf-takehome runs we observed K2.6 emitting
        # multiple consecutive 32768-token reasoning bursts, blowing
        # through the 120k-token output budget in 3-4 turns with zero
        # forward progress (``budget_exhausted`` runs cost $0.45-$0.58
        # for a 1.00x speedup).
        #
        # OpenRouter accepts a ``reasoning`` block on the request body
        # that maps to a per-provider reasoning cap. empirical
        # probe (5 variants x kimi-k2.6 on a heavy-reasoning prompt):
        #   - top-level ``reasoning_effort=low``: NO effect (4096 tok
        #     budget consumed in full, finish=length, 0 content chars)
        #   - nested ``reasoning.max_tokens=2000``: NO effect (hangs)
        #   - nested ``reasoning.effort=low``: HONORED -- reasoning
        #     dropped from ~3700 tok (medium) to ~2700 tok with
        #     finish=stop and real content emitted
        # originally sent ``reasoning.max_tokens=8000``; on K2.6
        # this was silently ignored (logs showed 32768-token bursts
        # still happening with reasoning_starvation events). Switching
        # to nested ``effort=low`` is the verified-working knob.
        #
        # ``AGENT6_REASONING_EFFORT`` env override (low|medium|
        # high|off) lets bench scripts vary the knob without code edits.
        #
        # per-call ``reasoning_effort`` argument takes
        # precedence over the env override. (Used by the CLI/config and
        # bench scripts; the run loop no longer drives it automatically
        # -- see below.)
        #
        # ``off`` must send ``reasoning={"enabled": False}``, NOT
        # omit the block. Empirically (direct OpenRouter probe, K2.6),
        # omitting the reasoning object leaves reasoning ON by default
        # (~2546 reasoning tokens on a heavy prompt) -- so the
        # "suppression" was a no-op and recovery turns still starved.
        # ``{"enabled": False}`` truly disables the reasoning channel
        # (0 reasoning tokens); the model writes any chain-of-thought
        # into ``content`` and still emits a tool_use.
        #
        # With the fix making "off" *actually* disable
        # reasoning, an N=8 K2.6 perf-takehome batch that forced
        # reasoning off on starvation-recovery turns scored WORSE than
        # leaving it on (25% vs ~38% win-rate, best 1.50x vs 7.76x):
        # K2.6's large speedups come *from* reasoning, so suppressing it
        # on recovery trades the occasional big win for reliable-but-
        # mediocre output. The automatic loop-level suppression
        # (per-turn + latch) was therefore removed. The
        # "off" knob remains for explicit operator/bench use.
        # `is_openai_direct_reasoning` (gpt-5, bare o1/o3) is a separate
        # predicate from `_is_reasoning_model` and does NOT imply it, so gate on
        # both: otherwise the configured `thinking`/reasoning_effort is silently
        # dropped for exactly the api.openai.com models whose only reasoning
        # control IS top-level `reasoning_effort`.
        if _is_reasoning_model(self.model) or is_openai_direct_reasoning:
            # Precedence: per-call argument > provider default (from config
            # `thinking`) > AGENT6_REASONING_EFFORT env > "low".
            effective_reasoning = (
                reasoning_effort if reasoning_effort is not None else self.reasoning_effort
            )
            if effective_reasoning is None:
                env_override = os.environ.get("AGENT6_REASONING_EFFORT", "").strip().lower()
                effective_reasoning = (
                    env_override if env_override in ("off", "low", "medium", "high") else "low"
                )
            effort = effective_reasoning.strip().lower()
            if is_openai_direct_reasoning:
                # api.openai.com Chat Completions o-series/gpt-5 take a TOP-LEVEL
                # ``reasoning_effort`` (low/medium/high), NOT the nested
                # ``reasoning`` object OpenRouter invented -- sending the nested
                # object there is an unknown parameter and 400s. Reasoning cannot
                # be disabled on o-series, so "off" omits the param (server
                # default) rather than sending {"enabled": False}.
                if effort != "off":
                    body["reasoning_effort"] = effort
            elif effort == "off":
                body["reasoning"] = {"enabled": False}
            else:
                body["reasoning"] = {"effort": effort}
        # OpenAI-direct o-series/reasoning models reject any explicit
        # ``temperature`` (only the server default is accepted), so omit it
        # there. Other hosts forward it as-is (until a 400 latches the omit).
        if (
            temperature is not None
            and not is_openai_direct_reasoning
            and not self._omit_temperature[0]
        ):
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools_to_openai(tools)
        # Operator-supplied body extras (e.g. OpenRouter `provider` routing to
        # pin a caching/fast backend). Merged last so it can override computed
        # tuning keys, but NEVER the load-bearing request shape: replacing
        # `messages`/`model` would silently send a different request, and
        # flipping `stream` would make the non-streaming path get an SSE body
        # that `resp.json()` can't parse. Those are filtered out.
        if self.extra_body:
            reserved = {"messages", "model", "stream", "stream_options"}
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})
        # Names of the tools actually offered this turn. Used purely as
        # a guard for the text-embedded-tool-call recovery in
        # `_parse_response`: we only ever coerce a text blob into a
        # tool_use when its `name` matches a tool we really offered, so
        # well-behaved models (native tool_calls) and models that happen
        # to answer with JSON are never affected.
        tool_names = frozenset(t.name for t in tools) if tools else frozenset()
        # Per-tool input JSON Schemas, keyed by name. Used by the
        # text-embedded-tool-call recovery to coerce a `<parameter>` string
        # value to its declared type (array/object/integer/...) so a leaked
        # Qwen-style XML call rebuilds correctly. Empty when no tools.
        tool_schemas = {t.name: t.input_schema for t in tools} if tools else {}

        # Streaming is chosen by the caller supplying a delta callback. It is
        # also the only reliable path for OpenRouter-style gateways whose
        # `: OPENROUTER PROCESSING` SSE comment heartbeats land in `resp.text`
        # as garbage on the non-streaming path and break `resp.json()`; bench
        # shell scripts force it via AGENT6_FORCE_STREAM=1 (the CLI translates
        # that into a no-op callback).
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
            parse=lambda data: _parse_response(
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

    def _call_streaming(  # noqa: PLR0915
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
        """SSE streaming variant of the OpenAI Chat Completions call.

        The stream lifecycle (idle watchdog, operator stop/steer, teardown
        classification) is ``providers._stream.SseCall``; this method owns
        the Chat Completions event shape:

        * Single ``data:`` line per frame (no ``event:`` typing); frames
          are JSON objects with a ``choices`` array carrying ``delta``.
        * Tool calls stream as ``choices[0].delta.tool_calls[]`` with an
          ``index`` field; id + name arrive once, ``function.arguments``
          arrives across many chunks and must be concatenated per
          index.
        * Reasoning models surface a separate ``delta.reasoning_content``
          (Kimi, DeepSeek) or ``delta.reasoning`` (OpenRouter).
        * Usage only arrives if ``stream_options.include_usage`` is set
          and lands in a terminal chunk whose ``choices`` is ``[]``.
        * ``data: [DONE]`` marks end of stream.
        * Gateways like OpenRouter emit SSE comment heartbeats
          (``:OPENROUTER PROCESSING``) for long requests. ``iter_lines``
          surfaces them as lines starting with ``:``; we skip those.
        """
        body = dict(body)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # tool_calls keyed by chunk-level ``index`` (not the call's
        # external id, which sometimes arrives late).
        tool_calls: dict[int, dict[str, Any]] = {}
        tool_arg_buf: dict[int, list[str]] = {}
        finish_reason = ""
        usage: dict[str, Any] = {}
        # Stream-completion tracking: a legit stream ends with `[DONE]` and/or a
        # non-empty `finish_reason`. A stream that ends with neither was cut off
        # (gateway timed out the upstream and closed the body cleanly, the same
        # failure family OpenRouter delivers as a mid-stream `error` frame); its
        # half-assembled content must NOT be returned as a completed turn.
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

        def consume(resp: httpx2.Response, clock: StreamClock) -> None:  # noqa: PLR0912, PLR0915
            nonlocal finish_reason, usage, done_seen
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                # SSE comment heartbeats (OpenRouter, etc). Deliberately NOT
                # marked on the clock -- heartbeats are exactly the bytes that
                # mask an upstream hang from httpx2's read timeout.
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                # Real SSE data line. Reset the idle clock; the watchdog is
                # satisfied as long as we keep seeing these at all (even
                # ``[DONE]`` counts as progress). NOTE: mark_output (the switch
                # to the short mid-stream idle timeout) happens later, only on
                # the first real CONTENT token -- an empty role/keepalive delta
                # arrives immediately and must not end the generous prefill
                # budget before the model has actually started producing output.
                clock.mark_data()
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    if data_str == "[DONE]":
                        done_seen = True
                        return
                    continue
                try:
                    evt: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # Mid-stream error frame (OpenRouter/OpenAI/LiteLLM deliver an
                # upstream 5xx/429 this way, then end the stream). Surface it
                # instead of silently returning the partial turn, mirroring the
                # Anthropic `error` event. No status_code -> retryable, so
                # _call_with_retry re-issues the request.
                err = evt.get("error")
                if isinstance(err, dict):
                    call.record(status=0, response=data_str[:8192])
                    raise ProviderError(
                        f"OpenAI stream error: {err.get('code')}: {err.get('message')}"
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
                if fr:
                    finish_reason = str(fr)
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    clock.mark_output()  # real output: the mid-stream idle budget applies
                    # Accumulation is unconditional; the callback is optional
                    # (streaming may be triggered by thinking_delta alone).
                    text_parts.append(content)
                    if text_delta_callback is not None:
                        with contextlib.suppress(Exception):
                            text_delta_callback(content)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    clock.mark_output()  # streamed reasoning counts as output too
                    reasoning_parts.append(reasoning)
                    if thinking_delta_callback is not None:
                        with contextlib.suppress(Exception):
                            thinking_delta_callback(reasoning)
                raw_tc = delta.get("tool_calls") or []
                if not isinstance(raw_tc, list):
                    continue
                if raw_tc:
                    clock.mark_output()  # tool-call tokens are real output
                for tc in raw_tc:
                    if not isinstance(tc, dict):
                        continue
                    raw_idx = tc.get("index")
                    tc_id = str(tc.get("id") or "")
                    if raw_idx is not None:
                        idx = int(raw_idx)
                    elif tc_id and any(s["id"] == tc_id for s in tool_calls.values()):
                        # Indexless delta continuing a known call: route by id.
                        idx = next(i for i, s in tool_calls.items() if s["id"] == tc_id)
                    elif tc_id and tool_calls:
                        # Indexless chunk carrying a NEW id (a gateway that
                        # sends whole calls in one chunk without index
                        # fields): open a fresh slot instead of collapsing
                        # every call onto slot 0 (which overwrote the first
                        # call and concatenated both argument strings).
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
                    func = tc.get("function") or {}
                    if isinstance(func, dict):
                        name = func.get("name")
                        if isinstance(name, str) and name:
                            slot["function"]["name"] = name
                        args_piece = func.get("arguments")
                        if isinstance(args_piece, str) and args_piece:
                            tool_arg_buf.setdefault(idx, []).append(args_piece)

        call.run(consume)

        # A stream that ended without `[DONE]` and without any `finish_reason`
        # was cut off mid-generation (a clean EOF is not a completion signal).
        # Returning the accumulated partial text / half-built tool call as a
        # finished turn feeds the loop a bogus silent_finish or a truncated
        # tool_use; raise a retryable ProviderError so the call is re-issued.
        if not done_seen and not finish_reason:
            call.record(
                status=0,
                response="stream ended without [DONE] or finish_reason (truncated)",
            )
            raise ProviderError(
                f"OpenAI stream from {url} ended prematurely "
                "(no [DONE], no finish_reason); upstream appears cut off."
            )

        # Finalise tool_call arguments.
        final_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls):
            slot = tool_calls[idx]
            args = "".join(tool_arg_buf.get(idx, []))
            slot["function"]["arguments"] = args
            final_tool_calls.append(slot)

        message: dict[str, Any] = {"content": "".join(text_parts)}
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
        call.record(status=200, response=synthesised)
        if self.budget is not None:
            _require_metered_usage(usage, source="OpenAI stream")
        parsed = _parse_response(synthesised, tool_names=tool_names, tool_schemas=tool_schemas)
        if self.budget is not None:
            self.budget.record(
                model=self.model,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                cache_read_tokens=parsed.cache_read_tokens,
                cache_creation_tokens=parsed.cache_creation_tokens,
                cost_usd=parsed.cost_usd,
            )
        return parsed
