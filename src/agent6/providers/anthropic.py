# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Anthropic provider.

Single audited HTTP call site. Uses httpx directly (no SDK) for a smaller
audit surface and pinned URL. Supports prompt caching via the
`cache_control` block field on system / tool entries.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agent6.budget import BudgetTracker

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192

_REDACT_HEADER_NAMES = frozenset({"x-api-key", "authorization", "proxy-authorization"})
_REDACTED = "<REDACTED>"


class ProviderError(Exception):
    """Anthropic call failed."""


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
    ) -> ProviderResponse:
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
        if tool_payload:
            body["tools"] = tool_payload

        try:
            resp = httpx.post(
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
            raise ProviderError(f"Anthropic API error {resp.status_code}: {resp.text[:500]}")
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
