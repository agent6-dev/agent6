# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Anthropic provider.

Single HTTP call site. Uses httpx2 directly (no SDK) for a smaller
audit surface and pinned URL. Supports prompt caching via the
`cache_control` block field on system / tool entries.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers.egress import http_post, http_stream
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
    parse_retry_after,
)
from agent6.providers.wire import AuthStyle, Deployment, auth_header, request_url

if TYPE_CHECKING:
    # Imported only for typing (used in a type hint); no runtime import needed.
    from agent6.providers.token_command import CommandToken

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
# Vertex carries the protocol version in the request body (not a header) under
# a Vertex-specific value; see _anthropic_version.
ANTHROPIC_VERTEX_VERSION = "vertex-2023-10-16"
DEFAULT_MAX_TOKENS = 8192

# Idle-since-last-data watchdog for the SSE streaming path. ``http_stream``
# forwards ``timeout`` to httpx2 as a per-read gap that resets on every byte;
# Anthropic ``ping`` heartbeats are bytes, so a wedged-but-pinging upstream
# would otherwise block ``iter_lines`` forever with no spend cap (budget is
# only recorded after the loop completes). We track time since the last
# MEANINGFUL data event (anything that isn't a ``ping``) and have a daemon
# thread close the response once it goes past ``_STREAM_IDLE_TIMEOUT_S``; the
# blocking ``iter_lines`` then raises an ``httpx2.HTTPError`` we convert to a
# retryable ProviderError. Mirrors the OpenAI provider's watchdog; the constants
# are duplicated in both providers for now (a shared providers/_stream.py would
# consolidate the watchdog and these constants).
#
# Two phases (see the OpenAI provider for the rationale): a generous budget for
# prefill / time-to-first-token (legitimately long on a big context), then a
# short idle once tokens have started -- a 45s mid-stream gap means the stream
# wedged, and recovering in 45s instead of 180s is 4x faster with no
# false-positive on prefill.
_STREAM_FIRST_DATA_TIMEOUT_S = 120.0
_STREAM_IDLE_TIMEOUT_S = 45.0
# The watchdog also polls should_abort/should_interrupt each tick, so keep it
# short: this bounds how long a Stop/steer/detach waits to end a long in-flight
# turn. A quarter second reads as immediate without the impatient second Ctrl-C.
_STREAM_WATCHDOG_TICK_S = 0.25


def _anthropic_version(deployment: str) -> tuple[str, str]:
    """Return ``(placement, value)`` for the Anthropic protocol version.

    Direct sends it as the ``anthropic-version`` HEADER; Vertex (and future
    Bedrock) send it as an ``anthropic_version`` BODY field with a
    deployment-specific value.
    """
    if deployment == "vertex":
        return ("body", ANTHROPIC_VERTEX_VERSION)
    return ("header", ANTHROPIC_VERSION)


# Maps the cross-provider ``thinking`` level (off/low/medium/high) onto an
# Anthropic extended-thinking ``budget_tokens`` value. Anthropic requires
# ``budget_tokens >= 1024`` and ``budget_tokens < max_tokens``; the call
# site lifts ``max_tokens`` above the chosen budget so the model always has
# room to answer after it finishes thinking.
_THINKING_BUDGET_TOKENS: dict[str, int] = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
}


def _require_metered_usage(usage: object, *, source: str) -> None:
    """Fail closed when a budgeted Anthropic call cannot be metered.

    Presence alone is not enough: a gateway with usage tracking disabled returns
    all-zero counts and every turn records zero, so the budget never trips.
    Require a positive input side, but sum in the cache counters: a fully-cached
    turn legitimately reports ``input_tokens: 0`` with ``cache_read_input_tokens``
    > 0, so a plain ``input_tokens > 0`` check would false-reject it."""
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_creation = usage.get("cache_creation_input_tokens") or 0
        if (
            isinstance(input_tokens, int)
            and isinstance(output_tokens, int)
            and input_tokens + cache_read + cache_creation > 0
        ):
            return
    raise ProviderError(
        f"{source} reported no usage input tokens (usage.input_tokens missing or 0); "
        "budgeted runs require provider usage accounting",
        status_code=422,
    )


def strip_cache_control_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``messages`` with every ``cache_control`` marker removed.

    The workflow places rolling breakpoints in the message list (see
    ``agent6.workflows._cache``); when the operator sets
    ``prompt_caching = false`` this strips them before the request is built.
    Copy-on-write: unmarked messages pass through untouched, marked blocks are
    shallow-copied so the caller's list (shared with resume snapshots) is
    never mutated.
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


def _is_temperature_400(status: int | None, text: str, body: dict[str, Any]) -> bool:
    """True when a 400 says the model rejects ``temperature`` (e.g. claude-opus-4-8:
    "temperature is deprecated for this model") AND temperature is still in the
    request body -- the signal to drop it and retry once."""
    return status == 400 and "temperature" in body and "temperature" in (text or "").lower()


@dataclass(frozen=True, slots=True)
class AnthropicProvider:
    """Stateless provider; constructed once per run."""

    api_key: str
    model: str
    base_url: str = ANTHROPIC_DEFAULT_BASE_URL
    deployment: Deployment = "direct"
    # Auth header style (config AuthConfig.style). "x_api_key" for direct
    # Anthropic, "bearer" for Vertex (Google OAuth via token_command).
    auth_style: AuthStyle = "x_api_key"
    prompt_caching: bool = True
    timeout_s: float = 120.0
    transcript_sink: TranscriptSink | None = None
    budget: BudgetTracker | None = None
    # Extended-thinking level (off/low/medium/high). When not "off" the
    # call enables Anthropic extended thinking with a budget drawn from
    # ``_THINKING_BUDGET_TOKENS`` and drops ``temperature`` (Anthropic
    # rejects temperature overrides while thinking is enabled).
    thinking: str | None = None
    extra_headers: tuple[tuple[str, str], ...] = ()
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_query: dict[str, str] = field(default_factory=dict)
    # Short-lived bearer source (config auth.token_command). When set it mints
    # the auth token per call instead of api_key, and a 401/403 triggers one
    # refresh + retry. Internally mutable (cache), hence held by reference.
    credential: CommandToken | None = None
    # Some newer models (e.g. claude-opus-4-8) reject ANY ``temperature`` with a
    # 400 "temperature is deprecated for this model". agent6 pins temperature for
    # determinism, so on that 400 we drop it and retry, latching this flag so the
    # rest of the run omits it (avoids re-sending the full context every call).
    # A 1-element list because the dataclass is frozen but the list is mutable.
    _omit_temperature: list[bool] = field(default_factory=lambda: [False])

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        env_var: str,
        prompt_caching: bool = True,
        timeout_s: float = 120.0,
        transcript_sink: TranscriptSink | None = None,
        budget: BudgetTracker | None = None,
        thinking: str | None = None,
    ) -> AnthropicProvider:
        key = os.environ.get(env_var, "").strip()
        if not key:
            raise ProviderError(f"Environment variable {env_var!r} is empty or unset")
        return cls(
            api_key=key,
            model=model,
            prompt_caching=prompt_caching,
            timeout_s=timeout_s,
            transcript_sink=transcript_sink,
            budget=budget,
            thinking=thinking,
        )

    def call(  # noqa: PLR0912, PLR0915
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
        # ``reasoning_effort`` is the OpenAI-reasoning-model knob; Anthropic
        # extended thinking uses a different shape and is configured on the
        # provider itself (``self.thinking``), so the cross-provider call
        # argument is ignored here.
        del reasoning_effort
        # Hard-stop: refuse the call up front if we're already over budget.
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

        # Breakpoint budget (Anthropic max 4 per request): this provider marks
        # the system block and the last tool (2); the workflow's rolling pair
        # in `messages` (agent6.workflows._cache) accounts for the other 2.
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
                # Cache the last tool entry too, anthropic caches up to that block.
                if self.prompt_caching and i == len(tools) - 1:
                    block["cache_control"] = {"type": "ephemeral"}
                tool_payload.append(block)

        thinking_budget = _THINKING_BUDGET_TOKENS.get(self.thinking or "off")
        if thinking_budget is not None:
            # Anthropic requires ``budget_tokens < max_tokens``; lift the
            # ceiling so the model keeps room to answer after thinking.
            max_tokens = max(max_tokens, thinking_budget + DEFAULT_MAX_TOKENS)

        body: dict[str, Any] = {
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        # Direct carries the model in the body; Vertex carries it in the URL
        # path and moves the protocol version into the body.
        if model_in_body:
            body["model"] = self.model
        if version_placement == "body":
            body["anthropic_version"] = version_value
        if thinking_budget is not None:
            # Extended thinking is incompatible with temperature overrides,
            # so only pass temperature when thinking is disabled.
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        elif temperature is not None and not self._omit_temperature[0]:
            body["temperature"] = temperature
        if tool_payload:
            body["tools"] = tool_payload
        if self.extra_body:
            reserved = {"system", "messages", "model", "stream", "anthropic_version"}
            body.update({k: v for k, v in self.extra_body.items() if k not in reserved})

        # Auth header is (re)built per attempt: a token_command credential mints
        # a short-lived bearer (Vertex Google OAuth); on a 401/403 we refresh it
        # once and retry. Without a credential there is exactly one attempt.
        cred = self.credential
        # +1 attempt reserved for a one-shot "temperature is deprecated" 400 retry
        # (only possible while temperature is still in the body).
        max_attempts = (2 if cred is not None else 1) + (1 if "temperature" in body else 0)
        for attempt in range(max_attempts):
            headers: dict[str, str] = {"content-type": "application/json"}
            token = cred.token() if cred is not None else self.api_key
            authed = auth_header(self.auth_style, token)
            if authed is not None:
                headers[authed[0]] = authed[1]
            if version_placement == "header":
                headers["anthropic-version"] = version_value
            if self.prompt_caching:
                headers["anthropic-beta"] = "prompt-caching-2024-07-31"
            for k, v in self.extra_headers:
                headers[k.lower()] = v

            # opt-in SSE streaming; otherwise the non-streaming POST below.
            if streaming:
                try:
                    return self._call_streaming(
                        url=url,
                        headers=headers,
                        body=body,
                        text_delta_callback=text_delta_callback,
                        thinking_delta_callback=thinking_delta_callback,
                        should_abort=should_abort,
                        should_interrupt=should_interrupt,
                    )
                except ProviderError as exc:
                    if _is_temperature_400(exc.status_code, str(exc), body):
                        self._omit_temperature[0] = True
                        body.pop("temperature", None)
                        continue
                    if (
                        cred is not None
                        and attempt + 1 < max_attempts
                        and exc.status_code in (401, 403)
                    ):
                        cred.invalidate()
                        continue
                    raise

            try:
                resp = http_post(
                    url,
                    headers=headers,
                    content=json.dumps(body).encode("utf-8"),
                    timeout=self.timeout_s,
                )
            except httpx2.HTTPError as exc:
                if self.transcript_sink is not None:
                    self.transcript_sink.record(
                        url=url,
                        request_headers=headers,
                        request_body=body,
                        response_status=0,
                        response_body=f"HTTPError: {exc}",
                    )
                raise ProviderError(f"HTTP error calling {url} (anthropic format): {exc}") from exc
            if cred is not None and attempt + 1 < max_attempts and resp.status_code in (401, 403):
                cred.invalidate()
                continue
            if resp.status_code >= 400:
                if self.transcript_sink is not None:
                    self.transcript_sink.record(
                        url=url,
                        request_headers=headers,
                        request_body=body,
                        response_status=resp.status_code,
                        response_body=resp.text[:8192],
                    )
                if _is_temperature_400(resp.status_code, resp.text, body):
                    self._omit_temperature[0] = True
                    body.pop("temperature", None)
                    continue
                raise ProviderError(
                    f"Anthropic API error {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code,
                    retry_after_s=parse_retry_after(resp.headers),
                )
            try:
                data: dict[str, Any] = resp.json()
            except (json.JSONDecodeError, ValueError) as exc:
                # A 2xx with a non-JSON body (transient proxy/gateway glitch)
                # would otherwise raise a JSONDecodeError that the retry loop
                # doesn't catch (it only handles ProviderError), aborting the
                # run. Convert to a retryable ProviderError. Leaving
                # status_code unset marks it retryable.
                if self.transcript_sink is not None:
                    self.transcript_sink.record(
                        url=url,
                        request_headers=headers,
                        request_body=body,
                        response_status=resp.status_code,
                        response_body=resp.text[:8192],
                    )
                raise ProviderError(
                    f"non-JSON response from Anthropic (status {resp.status_code}): "
                    f"{resp.text[:500]}"
                ) from exc
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    url=url,
                    request_headers=headers,
                    request_body=body,
                    response_status=resp.status_code,
                    response_body=data,
                )
            if self.budget is not None:
                _require_metered_usage(data.get("usage"), source="Anthropic response")
            parsed = _parse_response(data)
            if self.budget is not None:
                self.budget.record(
                    model=self.model,
                    input_tokens=parsed.input_tokens,
                    output_tokens=parsed.output_tokens,
                    cache_read_tokens=parsed.cache_read_tokens,
                    cache_creation_tokens=parsed.cache_creation_tokens,
                )
            return parsed
        raise ProviderError("Anthropic auth retry exhausted")  # pragma: no cover

    def _call_streaming(  # noqa: PLR0912, PLR0915
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
        """SSE streaming variant.

        Iterates the Anthropic Messages SSE stream, fans text_delta and
        thinking_delta deltas to their callbacks as they arrive, and at
        message_stop returns a ProviderResponse whose .raw is shaped
        identically to a non-streaming response so callers (Workflow,
        transcript replay) don't need a streaming-aware code path.
        """
        body = dict(body)
        # Direct enables streaming with a body flag; Vertex selects it via the
        # `:streamRawPredict` URL suffix (already baked into `url`) and rejects
        # a `stream` body field.
        if self.deployment == "direct":
            body["stream"] = True
        stream_headers = dict(headers)
        stream_headers["accept"] = "text/event-stream"

        # Accumulators for the synthesised non-streaming-shape response.
        content_blocks: list[dict[str, Any]] = []
        # Per-index in-flight builders. Anthropic indexes content blocks
        # 0..N within a single message; one block at a time is "open".
        text_acc: dict[int, list[str]] = {}
        tool_acc: dict[int, dict[str, Any]] = {}
        json_partial: dict[int, list[str]] = {}
        # Extended-thinking builders. ``thinking_acc`` collects the visible
        # reasoning text and ``signature_acc`` the cryptographic signature
        # Anthropic requires to be echoed back on the next turn when a tool
        # call follows a thinking block. Dropping either breaks multi-turn
        # tool use under extended thinking, so both must round-trip.
        thinking_acc: dict[int, list[str]] = {}
        signature_acc: dict[int, list[str]] = {}
        stop_reason: str = ""
        # The stream is complete only when a `message_stop` event arrives. A
        # clean EOF before it means the connection was cut mid-message; the
        # accumulated blocks are a truncated turn, not a finished one.
        saw_message_stop = False
        usage_input = 0
        usage_output = 0
        usage_cache_read = 0
        usage_cache_creation = 0
        saw_input_usage = False
        saw_output_usage = False

        sse_lines: list[str] = []  # for transcript audit trail

        # Background watchdog. It closes the response (unblocking the blocking
        # ``iter_lines`` below, which then raises) in two cases: an idle-since-
        # last-data hang past the threshold (``ping`` heartbeats do NOT count --
        # they mask an upstream hang from httpx2's read timeout), or the operator
        # stopping the run mid-turn (``should_abort``). The ``except`` turns each
        # into a purpose-specific error.
        last_data_at = time.monotonic()
        seen_data = threading.Event()  # set on the first data event; picks the timeout
        idle_killed = threading.Event()
        aborted = threading.Event()
        interrupted = threading.Event()
        watchdog_stop = threading.Event()
        # Mutable holder so the watchdog can reach the response without racing
        # on assignment (the ``with`` body runs in a different frame).
        resp_holder: dict[str, httpx2.Response] = {}

        def _watchdog() -> None:
            while not watchdog_stop.wait(_STREAM_WATCHDOG_TICK_S):
                resp = resp_holder.get("resp")
                if resp is None:
                    continue
                should_stop = False
                if should_abort is not None:
                    # A poll failure must not kill the watchdog -- that would also
                    # disable idle-hang detection. Treat it as "not aborting".
                    with contextlib.suppress(Exception):
                        should_stop = should_abort()
                if should_stop:
                    aborted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                # A steer request (Ctrl-C / TUI `s`) closes the stream so a long
                # thinking turn reaches the loop's steer boundary at once.
                should_steer = False
                if should_interrupt is not None:
                    with contextlib.suppress(Exception):
                        should_steer = should_interrupt()
                if should_steer:
                    interrupted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                timeout = (
                    _STREAM_IDLE_TIMEOUT_S if seen_data.is_set() else _STREAM_FIRST_DATA_TIMEOUT_S
                )
                if time.monotonic() - last_data_at <= timeout:
                    continue
                idle_killed.set()
                with contextlib.suppress(Exception):
                    resp.close()
                return

        watchdog = threading.Thread(
            target=_watchdog, name="agent6-anthropic-sse-watchdog", daemon=True
        )
        watchdog.start()

        try:
            with http_stream(
                "POST",
                url,
                headers=stream_headers,
                content=json.dumps(body).encode("utf-8"),
                timeout=self.timeout_s,
            ) as resp:
                resp_holder["resp"] = resp
                if resp.status_code >= 400:
                    error_body = resp.read().decode("utf-8", errors="replace")[:8192]
                    if self.transcript_sink is not None:
                        self.transcript_sink.record(
                            url=url,
                            request_headers=stream_headers,
                            request_body=body,
                            response_status=resp.status_code,
                            response_body=error_body,
                        )
                    raise ProviderError(
                        f"Anthropic API error {resp.status_code}: {error_body[:500]}",
                        status_code=resp.status_code,
                        retry_after_s=parse_retry_after(resp.headers),
                    )
                event_type: str = ""
                for line in resp.iter_lines():
                    sse_lines.append(line)
                    if not line:
                        event_type = ""
                        continue
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        evt: dict[str, Any] = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    et = event_type or str(evt.get("type", ""))
                    # Reset the idle clock on every MEANINGFUL event. ``ping``
                    # heartbeats are deliberately excluded: they are exactly
                    # the bytes that would otherwise mask a wedged upstream.
                    # seen_data (the switch to the short mid-stream idle timeout)
                    # is set only when actual content starts (content_block_*
                    # below), NOT on message_start -- that metadata arrives before
                    # the model has produced anything, and ending the generous
                    # prefill budget there would false-kill a long silent reason.
                    if et != "ping":
                        last_data_at = time.monotonic()
                    if et in ("content_block_start", "content_block_delta"):
                        seen_data.set()
                    if et == "message_start":
                        msg = evt.get("message", {})
                        u = msg.get("usage", {}) or {}
                        if "input_tokens" in u and u.get("input_tokens") is not None:
                            saw_input_usage = True
                        usage_input = int(u.get("input_tokens") or usage_input)
                        usage_cache_read = int(u.get("cache_read_input_tokens") or usage_cache_read)
                        usage_cache_creation = int(
                            u.get("cache_creation_input_tokens") or usage_cache_creation
                        )
                    elif et == "content_block_start":
                        idx = int(evt.get("index", 0))
                        cb = evt.get("content_block", {}) or {}
                        btype = cb.get("type")
                        if btype == "text":
                            text_acc[idx] = [str(cb.get("text", ""))]
                        elif btype == "thinking":
                            thinking_acc[idx] = [str(cb.get("thinking", ""))]
                            signature_acc[idx] = [str(cb.get("signature", ""))]
                        elif btype == "redacted_thinking":
                            # Opaque encrypted block, pass straight through.
                            content_blocks.append(
                                {"type": "redacted_thinking", "data": cb.get("data", "")}
                            )
                        elif btype == "tool_use":
                            tool_acc[idx] = {
                                "type": "tool_use",
                                "id": cb.get("id", ""),
                                "name": cb.get("name", ""),
                                "input": cb.get("input", {}) or {},
                            }
                            json_partial[idx] = []
                    elif et == "content_block_delta":
                        idx = int(evt.get("index", 0))
                        d = evt.get("delta", {}) or {}
                        dt = d.get("type")
                        if dt == "text_delta":
                            piece = str(d.get("text", ""))
                            text_acc.setdefault(idx, []).append(piece)
                            if piece and text_delta_callback is not None:
                                # Callback failure must never break the
                                # stream, cosmetic surface.
                                with contextlib.suppress(Exception):
                                    text_delta_callback(piece)
                        elif dt == "thinking_delta":
                            piece = str(d.get("thinking", ""))
                            thinking_acc.setdefault(idx, []).append(piece)
                            if piece and thinking_delta_callback is not None:
                                with contextlib.suppress(Exception):
                                    thinking_delta_callback(piece)
                        elif dt == "signature_delta":
                            signature_acc.setdefault(idx, []).append(str(d.get("signature", "")))
                        elif dt == "input_json_delta":
                            json_partial.setdefault(idx, []).append(str(d.get("partial_json", "")))
                    elif et == "content_block_stop":
                        idx = int(evt.get("index", 0))
                        if idx in text_acc:
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": "".join(text_acc.pop(idx)),
                                }
                            )
                        elif idx in thinking_acc:
                            block_out: dict[str, Any] = {
                                "type": "thinking",
                                "thinking": "".join(thinking_acc.pop(idx)),
                            }
                            sig = "".join(signature_acc.pop(idx, []))
                            if sig:
                                block_out["signature"] = sig
                            content_blocks.append(block_out)
                        elif idx in tool_acc:
                            tu = tool_acc.pop(idx)
                            partial = "".join(json_partial.pop(idx, []))
                            if partial:
                                try:
                                    tu["input"] = json.loads(partial)
                                except json.JSONDecodeError:
                                    # Stream truncated mid-JSON; surface
                                    # what we have rather than dropping
                                    # the tool_use entirely.
                                    tu["input"] = {"_partial_json": partial}
                            content_blocks.append(tu)
                    elif et == "message_delta":
                        d = evt.get("delta", {}) or {}
                        if "stop_reason" in d:
                            stop_reason = str(d.get("stop_reason", "") or "")
                        u = evt.get("usage", {}) or {}
                        if "output_tokens" in u:
                            if u.get("output_tokens") is not None:
                                saw_output_usage = True
                            usage_output = int(u.get("output_tokens") or usage_output)
                    elif et == "message_stop":
                        saw_message_stop = True
                        break
                    elif et == "error":
                        err = evt.get("error", {}) or {}
                        # Record the frame before raising so the upstream failure
                        # is auditable in the transcript (parity with the OpenAI
                        # provider's mid-stream error handling).
                        if self.transcript_sink is not None:
                            self.transcript_sink.record(
                                url=url,
                                request_headers=stream_headers,
                                request_body=body,
                                response_status=0,
                                response_body=data_str[:8192],
                            )
                        raise ProviderError(
                            f"Anthropic stream error: {err.get('type')}: {err.get('message')}"
                        )
        except httpx2.HTTPError as exc:
            watchdog_stop.set()
            if interrupted.is_set():
                # The operator asked to steer; the watchdog closed the stream so the
                # loop reaches its steer boundary without waiting out the turn.
                raise ProviderInterrupted("steer requested mid-stream") from exc
            if aborted.is_set():
                # The operator stopped the run; the watchdog closed the stream.
                raise ProviderAborted("run stopped by operator") from exc
            if idle_killed.is_set():
                # Convert the watchdog-induced HTTPError into a purpose-specific
                # ProviderError so the loop's retry/quit path can log a
                # meaningful reason rather than a generic "ReadError".
                phase_s = (
                    _STREAM_IDLE_TIMEOUT_S if seen_data.is_set() else _STREAM_FIRST_DATA_TIMEOUT_S
                )
                where = "mid-stream" if seen_data.is_set() else "before any data (prefill)"
                if self.transcript_sink is not None:
                    self.transcript_sink.record(
                        url=url,
                        request_headers=stream_headers,
                        request_body=body,
                        response_status=0,
                        response_body=(
                            f"SSE idle watchdog: no data event for {phase_s:.0f}s {where} "
                            f"(only heartbeats). Upstream model appears wedged."
                        ),
                    )
                raise ProviderError(
                    f"Anthropic SSE stream idle for >{phase_s:.0f}s {where} "
                    "(only heartbeats received); upstream appears wedged."
                ) from exc
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    url=url,
                    request_headers=stream_headers,
                    request_body=body,
                    response_status=0,
                    response_body=f"HTTPError: {exc}",
                )
            raise ProviderError(
                f"HTTP error streaming from {url} (anthropic format): {exc}"
            ) from exc
        finally:
            watchdog_stop.set()

        # No `message_stop` means the stream was cut mid-message (a clean EOF is
        # not a completion signal). The accumulated blocks are a truncated turn,
        # possibly with text already fanned to the TUI; returning them as a
        # finished response feeds the loop a bogus went_quiet/silent_finish and
        # records input tokens for a call that never completed. Raise a retryable
        # ProviderError so _call_with_retry re-issues the request.
        if not saw_message_stop:
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    url=url,
                    request_headers=stream_headers,
                    request_body=body,
                    response_status=0,
                    response_body="stream ended without message_stop (truncated)",
                )
            raise ProviderError(
                f"Anthropic SSE stream from {url} ended prematurely "
                "(no message_stop); upstream appears cut off."
            )

        # Synthesise the non-streaming-shaped response body so
        # downstream consumers (transcript replay, assistant_blocks
        # reconstruction in Workflow) see the same shape they would
        # see from a non-streaming call.
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
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                url=url,
                request_headers=stream_headers,
                request_body=body,
                response_status=200,
                response_body=synthesised,
            )
        if self.budget is not None:
            if not (saw_input_usage and saw_output_usage):
                raise ProviderError(
                    "Anthropic stream omitted usage.input_tokens/output_tokens; "
                    "budgeted runs require provider usage accounting",
                    status_code=422,
                )
            _require_metered_usage(synthesised.get("usage"), source="Anthropic stream")
        parsed = _parse_response(synthesised)
        if self.budget is not None:
            self.budget.record(
                model=self.model,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                cache_read_tokens=parsed.cache_read_tokens,
                cache_creation_tokens=parsed.cache_creation_tokens,
            )
        return parsed


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    content = data.get("content") or []
    text_parts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_uses.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
            )
    usage = data.get("usage") or {}
    # `or 0` throughout: a gateway returning null token fields on a 2xx would
    # make bare int(None) raise TypeError, which escapes the loop's
    # ProviderError-only retry wrapper and kills the run.
    return ProviderResponse(
        text="".join(text_parts),
        tool_uses=tuple(tool_uses),
        stop_reason=str(data.get("stop_reason", "")),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        raw=data,
    )
