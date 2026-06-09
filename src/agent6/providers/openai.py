# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions-compatible provider.

Works against any endpoint that speaks the OpenAI Chat Completions API:
OpenAI itself, OpenRouter, Ollama (`/v1`), vLLM, LM Studio, llama.cpp's
server, Kimi via Moonshot, DeepSeek-V3 via the official API or via
OpenRouter. Any sub-agent role (planner, worker, critic, reviewer,
summarizer) can be routed through this provider via
`[models.<role>]` in your config.

Single audited HTTP call site, same shape as
`agent6.providers.anthropic`. Uses httpx directly (no SDK) for a
smaller audit surface.

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
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from agent6.budget import BudgetTracker
from agent6.providers.anthropic import (
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
)
from agent6.providers.egress import http_post, http_stream

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192

# SSE idle-since-last-DATA watchdog.
#
# httpx's ``timeout`` (whether float or ``httpx.Timeout`` with ``read=``)
# is reset on EVERY received byte. OpenRouter (and other gateways
# fronted by Cloudflare) emits ``:`` SSE comment heartbeats every
# ~15s while a request is in flight. Those heartbeats reset the
# read-timeout indefinitely. If the upstream model truly hangs (we
# observed Kimi K2.6 sessions held in ESTABLISHED state with 0 bytes
# in the recv-queue for 800+ seconds while connection-level heartbeats
# continued), the harness never times out -- the orchestrator is
# parked in ``poll_schedule_timeout`` forever, eating wall-clock with
# no progress and no spend cap to save it.
#
# Fix: track time since the last meaningful SSE ``data:`` line
# (anything that isn't a heartbeat comment). If that goes past
# ``_STREAM_IDLE_TIMEOUT_S``, close the response from a watchdog
# thread; the blocking ``iter_lines`` then raises an ``httpx.HTTPError``
# that we re-raise as ``ProviderError`` with a descriptive message so
# the loop can retry-or-quit at its own layer.
#
# 180s is generous: real Kimi K2.6 reasoning bursts produce a data
# event every few seconds even mid-reasoning (token-level streaming).
# Going 3 minutes with only heartbeats means the upstream is wedged.
_STREAM_IDLE_TIMEOUT_S = 180.0
_STREAM_WATCHDOG_TICK_S = 5.0

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
    "kimi-k2",
    "minimax-m2",
)


def _is_reasoning_model(model: str) -> bool:
    """True if ``model`` looks like a reasoning model that emits
    ``reasoning_content`` separately from ``content``."""
    lowered = model.lower()
    return any(hint in lowered for hint in _REASONING_MODEL_HINTS)


@dataclass(frozen=True, slots=True)
class OpenAIProvider:
    """Stateless OpenAI Chat Completions-compatible provider.

    `api_key` may be empty for unauthenticated local endpoints (Ollama,
    llama.cpp's `server`); when empty, no `Authorization` header is sent.
    """

    api_key: str
    model: str
    base_url: str = OPENAI_DEFAULT_BASE_URL
    extra_headers: tuple[tuple[str, str], ...] = ()
    timeout_s: float = 120.0
    transcript_sink: TranscriptSink | None = None
    budget: BudgetTracker | None = None
    # Default reasoning effort for this provider (off|low|medium|high), wired
    # from the role's `[models.<role>].thinking`. A per-call `reasoning_effort`
    # argument takes precedence; below this sits the AGENT6_REASONING_EFFORT
    # env override. Only affects OpenAI-compatible reasoning models.
    reasoning_effort: str | None = None

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

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
    ) -> ProviderResponse:
        # extended_thinking is Anthropic-shaped (`budget_tokens`).
        # OpenAI reasoning models use `reasoning_effort` instead; no
        # 1:1 mapping. Silently no-op so cross-provider workflow code
        # doesn't have to branch.
        del extended_thinking
        if self.budget is not None:
            self.budget.check()
        headers: dict[str, str] = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        for k, v in self.extra_headers:
            headers[k.lower()] = v

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

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": oai_messages,
        }
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
        if _is_reasoning_model(self.model):
            # Precedence: per-call argument > provider default (from config
            # `thinking`) > AGENT6_REASONING_EFFORT env > "low".
            effective_reasoning = (
                reasoning_effort if reasoning_effort is not None else self.reasoning_effort
            )
            if effective_reasoning is not None:
                if effective_reasoning.strip().lower() == "off":
                    body["reasoning"] = {"enabled": False}
                else:
                    body["reasoning"] = {"effort": effective_reasoning.strip().lower()}
            else:
                override = os.environ.get("AGENT6_REASONING_EFFORT", "").strip().lower()
                if override == "off":
                    body["reasoning"] = {"enabled": False}
                elif override in ("low", "medium", "high"):
                    body["reasoning"] = {"effort": override}
                else:
                    body["reasoning"] = {"effort": "low"}
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools_to_openai(tools)
        # Names of the tools actually offered this turn. Used purely as
        # a guard for the text-embedded-tool-call recovery in
        # `_parse_response`: we only ever coerce a text blob into a
        # tool_use when its `name` matches a tool we really offered, so
        # well-behaved models (native tool_calls) and models that happen
        # to answer with JSON are never affected.
        tool_names = frozenset(t.name for t in tools) if tools else frozenset()

        # SSE streaming. When the caller supplies a
        # text_delta_callback we POST with `stream: true` (and
        # `stream_options.include_usage` so usage still arrives) and
        # synthesise a non-streaming-shape response at terminal chunk.
        # Streaming is also the only reliable path for OpenRouter-style
        # gateways that emit `: OPENROUTER PROCESSING` SSE comment
        # heartbeats during long requests; on the non-streaming path
        # those heartbeats land in `resp.text` as garbage and break
        # `resp.json()` mid-body. Bench shell scripts force this on
        # via AGENT6_FORCE_STREAM=1 (the CLI translates that into a
        # no-op callback so we exercise the streaming code path even
        # when stderr is redirected).
        if text_delta_callback is not None or thinking_delta_callback is not None:
            return self._call_streaming(
                headers=headers,
                body=body,
                text_delta_callback=text_delta_callback,
                thinking_delta_callback=thinking_delta_callback,
                tool_names=tool_names,
            )

        try:
            resp = http_post(
                self.endpoint,
                headers=headers,
                content=json.dumps(body).encode("utf-8"),
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    request_headers=headers,
                    request_body=body,
                    response_status=0,
                    response_body=f"HTTPError: {exc}",
                )
            raise ProviderError(f"HTTP error calling OpenAI: {exc}") from exc
        if resp.status_code >= 400:
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    request_headers=headers,
                    request_body=body,
                    response_status=resp.status_code,
                    response_body=resp.text[:8192],
                )
            raise ProviderError(
                f"OpenAI API error {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
            )
        data: dict[str, Any] = resp.json()
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                request_headers=headers,
                request_body=body,
                response_status=resp.status_code,
                response_body=data,
            )
        parsed = _parse_response(data, tool_names=tool_names)
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

    def _call_streaming(  # noqa: PLR0912, PLR0915
        self,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
        text_delta_callback: Callable[[str], None] | None = None,
        thinking_delta_callback: Callable[[str], None] | None = None,
        tool_names: frozenset[str] = frozenset(),
    ) -> ProviderResponse:
        """SSE streaming variant of the OpenAI Chat Completions call.

        Differences from Anthropic SSE we need to handle:

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

        # idle-since-last-DATA watchdog. ``last_data_at`` is
        # updated only on real SSE ``data:`` lines; heartbeats (``:``)
        # do not count. A background thread closes the response if the
        # idle threshold is exceeded; the blocking ``iter_lines`` then
        # raises and we surface a descriptive ProviderError.
        last_data_at = time.monotonic()
        idle_killed = threading.Event()
        watchdog_stop = threading.Event()
        # Mutable holder so the watchdog can reach the response without
        # racing on assignment (the ``with`` block runs in a different
        # frame from the watchdog closure).
        resp_holder: dict[str, httpx.Response] = {}

        def _watchdog() -> None:
            while not watchdog_stop.wait(_STREAM_WATCHDOG_TICK_S):
                if time.monotonic() - last_data_at <= _STREAM_IDLE_TIMEOUT_S:
                    continue
                resp = resp_holder.get("resp")
                if resp is None:
                    continue
                idle_killed.set()
                with contextlib.suppress(Exception):
                    resp.close()
                return

        watchdog = threading.Thread(
            target=_watchdog, name="agent6-openai-sse-watchdog", daemon=True
        )
        watchdog.start()

        try:
            with http_stream(
                "POST",
                self.endpoint,
                headers=stream_headers,
                content=json.dumps(body).encode("utf-8"),
                timeout=self.timeout_s,
            ) as resp:
                resp_holder["resp"] = resp
                if resp.status_code >= 400:
                    error_body = resp.read().decode("utf-8", errors="replace")[:8192]
                    if self.transcript_sink is not None:
                        self.transcript_sink.record(
                            request_headers=stream_headers,
                            request_body=body,
                            response_status=resp.status_code,
                            response_body=error_body,
                        )
                    raise ProviderError(
                        f"OpenAI API error {resp.status_code}: {error_body[:500]}",
                        status_code=resp.status_code,
                    )
                for raw_line in resp.iter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    # SSE comment heartbeats (OpenRouter, etc).
                    # Deliberately do NOT update last_data_at
                    # here -- heartbeats are exactly the bytes that mask
                    # an upstream hang from httpx's read timeout.
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    # Real SSE data line. Reset the idle clock; the
                    # watchdog is satisfied as long as we keep seeing
                    # these at all (even ``[DONE]`` counts as progress).
                    last_data_at = time.monotonic()
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        if data_str == "[DONE]":
                            break
                        continue
                    try:
                        evt: dict[str, Any] = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
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
                    if isinstance(content, str) and content and text_delta_callback is not None:
                        text_parts.append(content)
                        with contextlib.suppress(Exception):
                            text_delta_callback(content)
                    elif isinstance(content, str) and content:
                        text_parts.append(content)
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)
                        if thinking_delta_callback is not None:
                            with contextlib.suppress(Exception):
                                thinking_delta_callback(reasoning)
                    raw_tc = delta.get("tool_calls") or []
                    if not isinstance(raw_tc, list):
                        continue
                    for tc in raw_tc:
                        if not isinstance(tc, dict):
                            continue
                        idx = int(tc.get("index", 0))
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
        except httpx.HTTPError as exc:
            if idle_killed.is_set():
                # Convert the watchdog-induced HTTPError into a
                # purpose-specific ProviderError so the loop's error
                # path can log a meaningful reason rather than a
                # generic "ReadError" / "connection closed".
                watchdog_stop.set()
                if self.transcript_sink is not None:
                    self.transcript_sink.record(
                        request_headers=stream_headers,
                        request_body=body,
                        response_status=0,
                        response_body=(
                            f"SSE idle watchdog: no data event for "
                            f"{_STREAM_IDLE_TIMEOUT_S:.0f}s "
                            f"(only heartbeats). Upstream model appears wedged."
                        ),
                    )
                raise ProviderError(
                    f"OpenAI SSE stream idle for >{_STREAM_IDLE_TIMEOUT_S:.0f}s "
                    "(only heartbeats received); upstream model appears wedged."
                ) from exc
            watchdog_stop.set()
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    request_headers=stream_headers,
                    request_body=body,
                    response_status=0,
                    response_body=f"HTTPError: {exc}",
                )
            raise ProviderError(f"HTTP error streaming OpenAI: {exc}") from exc
        finally:
            watchdog_stop.set()

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
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                request_headers=stream_headers,
                request_body=body,
                response_status=200,
                response_body=synthesised,
            )
        parsed = _parse_response(synthesised, tool_names=tool_names)
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


def anthropic_to_openai_messages(  # noqa: PLR0912
    system: str, anthropic_msgs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Translate agent6's canonical Anthropic-shape messages into the
    OpenAI Chat Completions ``messages`` array.

    Three block types in Anthropic content are non-trivial:

    - ``text`` -> string content on the message (concatenated for
      multi-text-block messages).
    - ``tool_use`` (assistant) -> moved into ``message.tool_calls`` as
      OpenAI function-call objects; the assistant's text content
      stays in ``message.content``.
    - ``tool_result`` (user) -> emitted as a SEPARATE message with
      ``role="tool"`` and ``tool_call_id`` set; cannot stay in the
      user-message position because OpenAI puts tool replies in their
      own role.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in anthropic_msgs:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue
        # Walk content blocks. Behaviour depends on the block types
        # present.
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_chunks.append(str(block.get("text", "")))
            elif btype == "tool_use" and role == "assistant":
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            # OpenAI requires arguments as a JSON string,
                            # not an object.
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif btype == "tool_result":
                # Tool results become separate role=tool messages.
                # `content` field may be a string or a list of text
                # blocks; OpenAI accepts either string or its own
                # content-blocks shape. Flatten to string for the
                # broadest compatibility (Ollama, Kimi, etc).
                tr_content = block.get("content", "")
                if isinstance(tr_content, list):
                    parts = [
                        str(b.get("text", ""))
                        for b in tr_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    tr_text = "".join(parts) if parts else json.dumps(tr_content)
                else:
                    tr_text = str(tr_content)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": tr_text,
                    }
                )
        if role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_chunks:
                assistant_msg["content"] = "".join(text_chunks)
            else:
                assistant_msg["content"] = None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
        else:
            # user (or other) message: tool_results MUST come first
            # because OpenAI requires every `role=tool` message to
            # immediately follow the assistant turn whose `tool_calls`
            # it answers. we emitted text_chunks FIRST then
            # tool_results, which (a) inserted a user message between
            # the assistant's tool_calls and the tool replies — most
            # OpenAI-compatible gateways tolerate this but it is
            # technically malformed — and (b) made injected
            # "[loop-guard]" / "[harness]" / "[critic]" notices arrive
            # before the tool result they were commenting on, so weak
            # models lost the causal link entirely. Tool results first,
            # then any operator/harness text as a follow-up user turn.
            for tr in tool_results:
                out.append(tr)
            if text_chunks:
                out.append({"role": role, "content": "".join(text_chunks)})
    return out


def tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Translate ``ToolDefinition`` tuples into OpenAI function-tool entries."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


# Some OpenAI-compatible servers (notably certain Ollama / llama.cpp
# chat templates for Qwen, Hermes, and other small local models) do NOT
# parse the model's tool call into the native ``tool_calls`` array.
# Instead the call leaks into the assistant ``content`` as plain text —
# either a bare JSON object ``{"name": ..., "arguments": {...}}``, the
# same wrapped in a ```json fence, or Hermes/Qwen ``<tool_call>...
# </tool_call>`` tags. Without recovery the run loop sees text + no
# tool_use and stalls ("went quiet"). We recover these into real
# tool_uses, but ONLY as a fallback: see `_parse_response` for the
# guards (no native tool_calls present AND the recovered name matches a
# tool that was actually offered). Flagship models that emit native
# tool_calls — and any model that legitimately answers with JSON — never
# hit this path.
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_tool_call_obj(  # noqa: PLR0911
    candidate: str, tool_names: frozenset[str]
) -> dict[str, Any] | None:
    """Parse a single ``{"name", "arguments"}`` tool call from a text
    candidate, or return None if it isn't a tool call for an offered tool."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    # Accept the common spellings local templates use for the args object.
    raw_args = obj.get("arguments")
    if raw_args is None:
        raw_args = obj.get("parameters")
    if raw_args is None:
        raw_args = obj.get("input")
    if raw_args is None:
        raw_args = {}
    # A few templates double-encode the args as a JSON string.
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw_args, dict):
        return None
    return {"name": name, "input": raw_args}


def _coerce_text_tool_calls(
    text: str, tool_names: frozenset[str]
) -> tuple[list[dict[str, Any]], str]:
    """Best-effort recovery of tool calls a local model emitted as text.

    Returns ``(tool_uses, remaining_text)``. ``tool_uses`` is empty when
    nothing tool-call-shaped is found, in which case ``remaining_text``
    equals the original ``text``. The parsing is deliberately strict
    (exact JSON, a single fenced JSON object, or ``<tool_call>`` tags) so
    prose that merely mentions a tool name is never misread as a call.
    """
    if not text or not tool_names:
        return [], text
    # 1) Hermes / Qwen ``<tool_call>...</tool_call>`` wrappers (≥1).
    tag_matches = list(_TOOL_CALL_TAG_RE.finditer(text))
    if tag_matches:
        recovered: list[dict[str, Any]] = []
        for match in tag_matches:
            obj = _extract_tool_call_obj(match.group(1), tool_names)
            if obj is not None:
                recovered.append(obj)
        if recovered:
            remaining = _TOOL_CALL_TAG_RE.sub("", text).strip()
            return recovered, remaining
    # 2) A single fenced JSON object that is itself a tool call.
    fence = _JSON_FENCE_RE.search(text)
    if fence is not None:
        obj = _extract_tool_call_obj(fence.group(1), tool_names)
        if obj is not None:
            return [obj], _JSON_FENCE_RE.sub("", text).strip()
    # 3) The whole content is exactly one bare JSON tool-call object.
    obj = _extract_tool_call_obj(text, tool_names)
    if obj is not None:
        return [obj], ""
    return [], text


def _parse_response(  # noqa: PLR0912, PLR0915
    data: dict[str, Any], *, tool_names: frozenset[str] = frozenset()
) -> ProviderResponse:
    choices = data.get("choices") or []
    text = ""
    reasoning_text = ""
    stop_reason = ""
    tool_uses: tuple[dict[str, Any], ...] = ()
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        text = str(message.get("content") or "")
        # Kimi (``reasoning_content``), DeepSeek-R1 /
        # OpenRouter (``reasoning``), and OpenAI o-series surface
        # reasoning in a sibling field. Capture both spellings.
        raw_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        reasoning_text = str(raw_reasoning) if raw_reasoning else ""
        stop_reason = str(first.get("finish_reason") or "")
        raw_calls = message.get("tool_calls") or []
        parsed_calls: list[dict[str, Any]] = []
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function") or {}
            args_raw = func.get("arguments", "")
            # OpenAI returns arguments as a JSON string. Convert to dict
            # for the Anthropic-shape input field. Malformed JSON
            # surfaces as an empty dict + the raw string under
            # `_raw_arguments` so debugging is possible.
            #
            # When a model degenerates and emits a 30+ KB
            # tool-arg payload of repeated escape sequences (observed live
            # with Kimi K2.6 looping on `\\n\\n\\n...` until hitting the
            # completion_tokens cap), the raw blob ends up echoed in the
            # subsequent tool_error message and re-enters the model's
            # context window — priming the same degeneration on the next
            # turn. Cap the diagnostic string at 500 chars so the
            # repetition doesn't survive the round-trip.
            _RAW_ARGS_CAP = 500
            try:
                parsed_input = json.loads(args_raw) if args_raw else {}
                if not isinstance(parsed_input, dict):
                    parsed_input = {"_value": parsed_input}
            except (json.JSONDecodeError, TypeError):
                raw_str = str(args_raw)
                if len(raw_str) > _RAW_ARGS_CAP:
                    raw_str = (
                        raw_str[:_RAW_ARGS_CAP]
                        + f"... <truncated; original was {len(str(args_raw))} chars>"
                    )
                parsed_input = {"_raw_arguments": raw_str}
            parsed_calls.append(
                {
                    "id": str(call.get("id", "")),
                    "name": str(func.get("name", "")),
                    "input": parsed_input,
                }
            )
        tool_uses = tuple(parsed_calls)
        # Fallback: no NATIVE tool_calls but the model leaked a tool call
        # into its text content (small local models via Ollama/llama.cpp).
        # Guarded by `tool_names` so this only fires for tools actually
        # offered and never for flagship models (which populate
        # tool_calls) or models legitimately answering with JSON.
        if not tool_uses and tool_names:
            recovered, remaining_text = _coerce_text_tool_calls(text, tool_names)
            if recovered:
                tool_uses = tuple(
                    {"id": f"call_text_{i}", "name": r["name"], "input": r["input"]}
                    for i, r in enumerate(recovered)
                )
                text = remaining_text
    usage = data.get("usage") or {}
    # OpenAI's cached_tokens field, when present, lives under
    # usage.prompt_tokens_details.cached_tokens. Treat absent as 0.
    #
    # CRITICAL provider-format asymmetry: Anthropic's `input_tokens`
    # already EXCLUDES cache-read tokens (they're surfaced separately under
    # `cache_read_input_tokens`). OpenAI's `prompt_tokens`, by contrast, is
    # the TOTAL prompt size — cached + fresh. We normalise to Anthropic's
    # semantics here so `ProviderResponse.input_tokens` consistently means
    # "fresh, non-cached input" across providers. Without this, the
    # BudgetTracker would charge cached tokens against the input-token cap
    # at full rate (causing premature budget exhaustion on cache-heavy
    # OpenAI runs) AND the cost formula in budget.py would double-count the
    # cache portion (full input rate plus an additional 10% cache-read
    # surcharge).
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    prompt_total = int(usage.get("prompt_tokens", 0))
    # Defensive clamp: a misbehaving upstream that reports cached > prompt
    # would otherwise make input_tokens negative.
    fresh_input = max(prompt_total - cached, 0)
    # Build a content-blocks raw payload mirroring Anthropic's response
    # shape so callers that inspect resp.raw["content"] (the worker_loop
    # does this to reconstruct the assistant message verbatim) see the
    # same structure regardless of provider.
    raw_content: list[dict[str, Any]] = []
    if reasoning_text:
        # Surface reasoning as a leading text block wrapped in
        # ``<thinking>`` tags so downstream code that already knows the
        # tag (e.g. workflows.loop._summarise_assistant_text_for_commit)
        # behaves identically across providers. We do NOT promote
        # reasoning into the user-visible ``text`` field: workflows that
        # echo ``resp.text`` (CLI logger, transcript surface) would
        # double-print it and the auto-commit summariser already strips
        # the prefix. Keeping it in raw is enough for inspection /
        # debugging while leaving the assistant's actual answer clean.
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
    if isinstance(raw_cost, int | float) and raw_cost > 0:
        reported_cost = float(raw_cost)
    return ProviderResponse(
        text=text,
        tool_uses=tool_uses,
        stop_reason=stop_reason,
        input_tokens=fresh_input,
        output_tokens=int(usage.get("completion_tokens", 0)),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        cost_usd=reported_cost,
        raw=enriched_raw,
    )
