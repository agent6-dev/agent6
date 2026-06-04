# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Anthropic provider.

Single audited HTTP call site. Uses httpx directly (no SDK) for a smaller
audit surface and pinned URL. Supports prompt caching via the
`cache_control` block field on system / tool entries.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agent6.budget import BudgetTracker
from agent6.providers.egress import http_post, http_stream

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192

_REDACT_HEADER_NAMES = frozenset({"x-api-key", "authorization", "proxy-authorization"})
_REDACTED = "<REDACTED>"


class ProviderError(Exception):
    """Anthropic call failed.

    ``status_code`` is the upstream HTTP status when the failure originated
    from an API error response (None for network/parse failures). The loop's
    retry wrapper uses it to skip retrying permanent client errors such as
    401/402/403 that will never succeed on a second attempt.
    """

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of `headers` with secret-bearing entries redacted."""
    return {k: (_REDACTED if k.lower() in _REDACT_HEADER_NAMES else v) for k, v in headers.items()}


class TranscriptSink:
    """Append-only writer of one JSON file per LLM round-trip.

    Files live under `transcripts_dir/<utc-iso>-<seq>.json`. The seq counter
    is per-sink, monotonically increasing, and thread-safe. Secrets in
    request headers are redacted before any bytes hit disk.
    """

    __slots__ = ("_dir", "_lock", "_seq")

    def __init__(self, transcripts_dir: Path) -> None:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._dir = transcripts_dir
        self._lock = threading.Lock()
        self._seq = 0

    def record(
        self,
        *,
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> Path:
        with self._lock:
            self._seq += 1
            seq = self._seq
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._dir / f"{ts}-{seq:06d}.json"
        payload = {
            "ts": ts,
            "seq": seq,
            "request": {
                "url": ANTHROPIC_URL,
                "headers": _redact_headers(request_headers),
                "body": request_body,
            },
            "response": {
                "status": response_status,
                "body": response_body,
            },
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One tool exposed to the model. `input_schema` is generated from a pydantic model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Response from a single Anthropic call."""

    text: str
    tool_uses: tuple[dict[str, Any], ...]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    # provider-reported USD cost for this single call. Currently
    # populated only by the OpenAI-compatible provider when the upstream
    # gateway returns ``usage.cost`` (OpenRouter does; OpenAI direct does
    # not; Anthropic does not). Zero means "no authoritative figure was
    # supplied" — callers fall back to the price-table estimate in
    # ``BudgetTracker.estimate_usd``.
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnthropicProvider:
    """Stateless provider; constructed once per run."""

    api_key: str
    model: str
    prompt_caching: bool = True
    timeout_s: float = 120.0
    transcript_sink: TranscriptSink | None = None
    budget: BudgetTracker | None = None

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
        )

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
    ) -> ProviderResponse:
        # reasoning_effort is an OpenAI-reasoning-model knob; Anthropic
        # extended thinking uses a different shape. Silently no-op so
        # cross-provider loop code doesn't have to branch.
        del reasoning_effort
        # Hard-stop: refuse the call up front if we're already over budget.
        if self.budget is not None:
            self.budget.check()
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self.prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if self.prompt_caching:
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}

        tool_payload: list[dict[str, Any]] = []
        if tools:
            for i, t in enumerate(tools):
                block: dict[str, Any] = {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                # Cache the last tool entry too — anthropic caches up to that block.
                if self.prompt_caching and i == len(tools) - 1:
                    block["cache_control"] = {"type": "ephemeral"}
                tool_payload.append(block)

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if tool_payload:
            body["tools"] = tool_payload

        # opt-in SSE streaming. When the caller passes a
        # text_delta_callback we POST with `stream: true` and feed text
        # deltas to the callback as they arrive, then synthesise a
        # ProviderResponse identical in shape to the non-streaming path
        # at message_stop. Non-streaming is the default to keep bench
        # runs and the existing test suite on the audited code path.
        if text_delta_callback is not None:
            return self._call_streaming(
                headers=headers,
                body=body,
                text_delta_callback=text_delta_callback,
            )

        try:
            resp = http_post(
                ANTHROPIC_URL,
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
            raise ProviderError(f"HTTP error calling Anthropic: {exc}") from exc
        if resp.status_code >= 400:
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    request_headers=headers,
                    request_body=body,
                    response_status=resp.status_code,
                    response_body=resp.text[:8192],
                )
            raise ProviderError(
                f"Anthropic API error {resp.status_code}: {resp.text[:500]}",
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

    def _call_streaming(  # noqa: PLR0912, PLR0915
        self,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
        text_delta_callback: Callable[[str], None],
    ) -> ProviderResponse:
        """SSE streaming variant.

        Iterates the Anthropic Messages SSE stream, fans text_delta
        deltas to ``text_delta_callback`` as they arrive, and at
        message_stop returns a ProviderResponse whose .raw is shaped
        identically to a non-streaming response so callers (Workflow,
        transcript replay) don't need a streaming-aware code path.
        """
        body = dict(body)
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
        stop_reason: str = ""
        usage_input = 0
        usage_output = 0
        usage_cache_read = 0
        usage_cache_creation = 0

        sse_lines: list[str] = []  # for transcript audit trail
        try:
            with http_stream(
                "POST",
                ANTHROPIC_URL,
                headers=stream_headers,
                content=json.dumps(body).encode("utf-8"),
                timeout=self.timeout_s,
            ) as resp:
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
                        f"Anthropic API error {resp.status_code}: {error_body[:500]}"
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
                    if et == "message_start":
                        msg = evt.get("message", {})
                        u = msg.get("usage", {}) or {}
                        usage_input = int(u.get("input_tokens", usage_input))
                        usage_cache_read = int(u.get("cache_read_input_tokens", usage_cache_read))
                        usage_cache_creation = int(
                            u.get("cache_creation_input_tokens", usage_cache_creation)
                        )
                    elif et == "content_block_start":
                        idx = int(evt.get("index", 0))
                        cb = evt.get("content_block", {}) or {}
                        btype = cb.get("type")
                        if btype == "text":
                            text_acc[idx] = [str(cb.get("text", ""))]
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
                            if piece:
                                # Callback failure must never break the
                                # stream — cosmetic surface.
                                with contextlib.suppress(Exception):
                                    text_delta_callback(piece)
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
                            usage_output = int(u.get("output_tokens", usage_output))
                    elif et == "message_stop":
                        break
                    elif et == "error":
                        err = evt.get("error", {}) or {}
                        raise ProviderError(
                            f"Anthropic stream error: {err.get('type')}: {err.get('message')}"
                        )
        except httpx.HTTPError as exc:
            if self.transcript_sink is not None:
                self.transcript_sink.record(
                    request_headers=stream_headers,
                    request_body=body,
                    response_status=0,
                    response_body=f"HTTPError: {exc}",
                )
            raise ProviderError(f"HTTP error streaming Anthropic: {exc}") from exc

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
                request_headers=stream_headers,
                request_body=body,
                response_status=200,
                response_body=synthesised,
            )
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
    return ProviderResponse(
        text="".join(text_parts),
        tool_uses=tuple(tool_uses),
        stop_reason=str(data.get("stop_reason", "")),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0)),
        raw=data,
    )
