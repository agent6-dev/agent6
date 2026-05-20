# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""OpenAI Chat Completions-compatible provider.

Works against any endpoint that speaks the OpenAI Chat Completions API:
OpenAI itself, OpenRouter, Ollama (`/v1`), vLLM, LM Studio, llama.cpp's
server. Any sub-agent role (planner, worker, critic, reviewer, summarizer)
can be routed through this provider via `[models.<role>]` in `agent6.toml`.

Single audited HTTP call site, same shape as `agent6.providers.anthropic`.
Uses httpx directly (no SDK) for a smaller audit surface.

Tools (function-calling) are intentionally NOT supported in v1. The current
sub-agents route all tool dispatch through Python (`agent6.tools.dispatch`),
not the provider's tool-call API, so this is fine for every existing role.
If the caller passes a non-empty tools list we fail loudly rather than
silently dropping them.
"""

from __future__ import annotations

import json
import os
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

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 8192


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
        if tools:
            raise ProviderError("OpenAIProvider does not support tool use in v1")
        if self.budget is not None:
            self.budget.check()
        headers: dict[str, str] = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        for k, v in self.extra_headers:
            headers[k.lower()] = v
        # Translate Anthropic-style messages (role=user|assistant, content=str|blocks)
        # into OpenAI Chat Completions shape. We only emit text content, since the
        # reviewer never passes tool_use or tool_result blocks.
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                # Flatten Anthropic content blocks into plain text.
                text_chunks = [
                    str(b.get("text", ""))
                    for b in content
                    if isinstance(b, dict)
                    if b.get("type") == "text"
                ]
                text = "".join(text_chunks)
            else:
                text = str(content)
            oai_messages.append({"role": role, "content": text})

        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
        }

        try:
            resp = httpx.post(
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
            raise ProviderError(f"OpenAI API error {resp.status_code}: {resp.text[:500]}")
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
    choices = data.get("choices") or []
    text = ""
    stop_reason = ""
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        text = str(message.get("content") or "")
        stop_reason = str(first.get("finish_reason") or "")
    usage = data.get("usage") or {}
    # OpenAI's cached_tokens field, when present, lives under
    # usage.prompt_tokens_details.cached_tokens. Treat absent as 0.
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    return ProviderResponse(
        text=text,
        tool_uses=(),
        stop_reason=stop_reason,
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        cache_read_tokens=cached,
        cache_creation_tokens=0,
        raw=data,
    )
