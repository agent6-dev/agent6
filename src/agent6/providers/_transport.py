# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared request transport for the provider call paths.

Both providers execute one API call the same way: an attempt loop with
per-attempt auth headers (a ``token_command`` credential mints a short-lived
bearer, and a 401/403 refreshes it once and retries), one-shot 4xx body
adaptation (each provider decides which parameter-rejection 400s it can fix
by rewriting the body and latching), transcript recording, a retryable error
for a 2xx with a non-JSON body, usage metering, and the budget charge.
:class:`ProviderCall` owns that loop; request-body construction, header
composition, 400 adaptation, metering rules, and response parsing stay
per-provider via the hook fields.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.budget import BudgetTracker
from agent6.providers.egress import http_post
from agent6.providers.token_command import CommandToken
from agent6.providers.types import (
    ProviderError,
    ProviderResponse,
    TranscriptRecorder,
    parse_retry_after,
)


def _has_assistant_output(data: dict[str, Any]) -> bool:
    """Whether a 2xx body carries a REAL assistant response, so a top-level
    ``error`` key beside it is incidental rather than an error envelope. Covers
    both wire shapes: OpenAI ``choices[].message`` (content or tool_calls),
    Anthropic top-level ``content``. A placeholder choice with null content is
    not output."""
    choices = data.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            msg = ch.get("message") if isinstance(ch, dict) else None
            if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
                return True
    return isinstance(data.get("content"), list) and bool(data.get("content"))


def _envelope_status(err: object) -> int | None:
    """The upstream HTTP status carried in an error envelope's ``code`` (int or
    all-digit string), if it is a real 4xx/5xx; else None (retryable). Threading
    it into ``ProviderError.status_code`` lets ``NON_RETRYABLE_HTTP_STATUSES``
    classify a 402 as permanent while a 429/5xx stays retryable."""
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code if 400 <= code <= 599 else None
    if isinstance(code, str) and code.isdigit() and 400 <= int(code) <= 599:
        return int(code)
    return None


def _envelope_detail(err: object) -> str:
    """A readable ``code: message`` for an error envelope, tolerating a bare
    string error and an empty object."""
    if isinstance(err, dict):
        label = err.get("code") or err.get("type") or "error"
        return f"{label}: {err.get('message') or err}"
    return str(err)


@dataclass(frozen=True, slots=True)
class ProviderCall:
    """One provider API call: the attempt loop around a built request body.

    ``adapt_400`` receives ``(status, error_text, body)`` and returns True
    after mutating ``body`` (and latching provider state) so the next attempt
    sends the adapted request; ``adapt_attempts`` reserves one extra attempt
    per adaptation the provider considers possible for this body. ``stream``,
    when set, replaces the non-streaming POST and receives the per-attempt
    headers; errors it raises flow through the same adapt/refresh logic.
    """

    api_label: str  # "OpenAI" / "Anthropic"; leads API-error messages
    api_format: str  # "openai" / "anthropic"; names the wire format
    url: str
    body: dict[str, Any]
    timeout_s: float
    api_key: str
    credential: CommandToken | None
    transcript_sink: TranscriptRecorder | None
    budget: BudgetTracker | None
    model: str
    build_headers: Callable[[str], dict[str, str]]
    adapt_400: Callable[[int | None, str, dict[str, Any]], bool]
    adapt_attempts: int
    require_metered: Callable[[dict[str, Any]], None]
    parse: Callable[[dict[str, Any]], ProviderResponse]
    stream: Callable[[dict[str, str]], ProviderResponse] | None = None

    def record(self, headers: dict[str, str], status: int, response: dict[str, Any] | str) -> None:
        """Write one transcript entry for this request (no-op without a sink)."""
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                url=self.url,
                request_headers=headers,
                request_body=self.body,
                response_status=status,
                response_body=response,
            )

    def run(self) -> ProviderResponse:
        cred = self.credential
        # A credential reserves one refresh + retry for an expired bearer;
        # each possible one-shot body adaptation reserves one more attempt.
        max_attempts = (2 if cred is not None else 1) + self.adapt_attempts
        for attempt in range(max_attempts):
            token = cred.token() if cred is not None else self.api_key
            headers = self.build_headers(token)

            if self.stream is not None:
                try:
                    return self.stream(headers)
                except ProviderError as exc:
                    if attempt + 1 < max_attempts and self.adapt_400(
                        exc.status_code, str(exc), self.body
                    ):
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
                    self.url,
                    headers=headers,
                    content=json.dumps(self.body).encode("utf-8"),
                    timeout=self.timeout_s,
                )
            except httpx2.HTTPError as exc:
                self.record(headers, 0, f"HTTPError: {exc}")
                raise ProviderError(
                    f"HTTP error calling {self.url} ({self.api_format} format): {exc}"
                ) from exc
            if cred is not None and attempt + 1 < max_attempts and resp.status_code in (401, 403):
                # Record BEFORE refreshing: this 401/403 hit the wire, and the
                # transcript contract is one file per round-trip (the streaming
                # path already records it; only this branch dropped it).
                self.record(headers, resp.status_code, resp.text[:8192])
                cred.invalidate()
                continue
            if resp.status_code >= 400:
                self.record(headers, resp.status_code, resp.text[:8192])
                if attempt + 1 < max_attempts and self.adapt_400(
                    resp.status_code, resp.text, self.body
                ):
                    continue
                raise ProviderError(
                    f"{self.api_label} API error {resp.status_code}: {resp.text[:500]}",
                    status_code=resp.status_code,
                    retry_after_s=parse_retry_after(resp.headers),
                )
            return self._decode_success(headers, resp)
        raise ProviderError(f"{self.api_label} auth retry exhausted")  # pragma: no cover

    def _decode_success(self, headers: dict[str, str], resp: httpx2.Response) -> ProviderResponse:
        """A 2xx body -> ProviderResponse: decode, record, meter, budget."""
        try:
            # Annotated Any: json() returns whatever the body holds; the
            # dict shape is PROVEN by the guard below, not assumed.
            data: Any = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            # A 2xx with a non-JSON body (transient proxy/gateway glitch)
            # would otherwise raise a JSONDecodeError that the retry loop
            # doesn't catch (it only handles ProviderError), aborting the
            # run. Convert to a retryable ProviderError. Leaving
            # status_code unset marks it retryable.
            self.record(headers, resp.status_code, resp.text[:8192])
            raise ProviderError(
                f"non-JSON response from {self.api_label} "
                f"(status {resp.status_code}): {resp.text[:500]}"
            ) from exc
        if not isinstance(data, dict):
            # A 2xx whose valid JSON is not an object (array/string from a
            # glitching gateway): every consumer downstream assumes a dict,
            # and the AttributeError it would raise bypasses the loop's
            # ProviderError-only retry. Same retryable conversion as the
            # non-JSON branch above.
            self.record(headers, resp.status_code, resp.text[:8192])
            raise ProviderError(
                f"{self.api_label} returned a non-object JSON body "
                f"(status {resp.status_code}): {resp.text[:500]}"
            )
        self.record(headers, resp.status_code, data)
        # An in-band error envelope on a 2xx (OpenRouter/LiteLLM deliver an
        # upstream 5xx/429/4xx this way; Anthropic's error object has the same
        # top-level key). require_metered below finds no usage and blamed
        # agent6's own accounting ("no usage input tokens", 422), so a transient
        # upstream failure killed the run with no retry AND a permanent one
        # (402 "Insufficient credits") was RETRIED every turn because the status
        # was dropped. Key on "no usable assistant output" -- a placeholder
        # `choices` entry with null content is not output, a bare string `error`
        # is still an envelope -- and carry the upstream code as the status so
        # NON_RETRYABLE_HTTP_STATUSES makes 4xx permanent, 429/5xx retryable.
        err = data.get("error")
        if err is not None and not _has_assistant_output(data):
            raise ProviderError(
                f"{self.api_label} error in 2xx body: {_envelope_detail(err)}",
                status_code=_envelope_status(err),
            )
        if self.budget is not None:
            self.require_metered(data)
        parsed = self.parse(data)
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
