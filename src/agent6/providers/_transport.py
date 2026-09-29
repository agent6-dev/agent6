# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The request transport the HTTP providers' call paths share.

`ProviderCall` owns the attempt loop: per-attempt auth headers (a refreshable
credential is re-minted once on a 401 or 403), one-shot 400 body adaptation,
transcript recording, a retryable error for a malformed 2xx, usage metering and
the budget charge. Body construction, headers, adaptation, metering rules and
parsing stay per provider through its hook fields.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

import httpx2

from agent6 import budget as agent6_budget
from agent6.providers import types

# The handshake bound; only a blackholed connect takes longer, and the watchdog cannot cut one.
CONNECT_TIMEOUT_S = 20.0


def granular_timeout(timeout: float) -> httpx2.Timeout:
    """Return the timeout for read, write and pool, with the connect phase bounded on its own."""
    return httpx2.Timeout(timeout, connect=min(CONNECT_TIMEOUT_S, timeout))


# A full-window reply is a few MiB; the cap stops a hostile endpoint from being buffered whole.
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def http_post(
    url: str, *, headers: dict[str, str], content: bytes, timeout: float
) -> httpx2.Response:
    """Make one POST and read the body under the size cap; the seam tests stub.

    Args:
        url: The URL.
        headers: The request headers.
        content: The request body.
        timeout: The read budget in seconds.

    Returns:
        The response with its body read.

    Raises:
        ProviderError: The body exceeded `MAX_RESPONSE_BYTES` (retryable).
    """
    with httpx2.stream(
        "POST", url, headers=headers, content=content, timeout=granular_timeout(timeout)
    ) as resp:
        body = bytearray()
        for chunk in resp.iter_bytes():
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise types.ProviderError(
                    f"provider response exceeded {MAX_RESPONSE_BYTES} bytes; refusing to buffer it"
                )
        # The body is decoded, so the wire's representation headers no longer describe it.
        response_headers = [
            (k, v)
            for k, v in httpx2.Headers(resp.headers).multi_items()
            if k not in ("content-encoding", "content-length")
        ]
        return httpx2.Response(
            resp.status_code,
            headers=response_headers,
            content=bytes(body),
            request=resp.request,
        )


def _has_assistant_output(data: dict[str, Any]) -> bool:
    """Return whether a 2xx body carries a real assistant response, in either wire shape.

    A top-level `error` beside real output is incidental, not an envelope; a
    placeholder choice with null content is not output.
    """
    choices = data.get("choices")
    if isinstance(choices, list):
        for ch in choices:
            msg = ch.get("message") if isinstance(ch, dict) else None
            if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
                return True
    return isinstance(data.get("content"), list) and bool(data.get("content"))


# The documented statuses behind string error codes, so an in-band error keeps its retry class.
_ERROR_CODE_STATUS: dict[str, int] = {
    # OpenAI-family `code`
    "insufficient_quota": 402,
    "invalid_api_key": 401,
    "model_not_found": 404,
    # Anthropic `type`
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


def envelope_status(err: object) -> int | None:
    """Read the HTTP status an error envelope carries.

    A numeric `code` is read directly; a documented string `code` or `type` maps
    to its status. Threaded into `ProviderError.status_code`, it lets the retry
    wrapper treat a 402 as permanent and a 429 or 5xx as retryable.

    Args:
        err: The envelope's `error` value.

    Returns:
        The status when it is a real 4xx or 5xx, else None (retryable).
    """
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if isinstance(code, bool):
        code = None
    if isinstance(code, int):
        return code if 400 <= code <= 599 else None
    if isinstance(code, str) and code.isdigit() and 400 <= int(code) <= 599:
        return int(code)
    for label in (code, err.get("type")):
        if isinstance(label, str) and label in _ERROR_CODE_STATUS:
            return _ERROR_CODE_STATUS[label]
    return None


def _envelope_detail(err: object) -> str:
    """Return an error envelope as `code: message`, tolerating a bare string or an empty object."""
    if isinstance(err, dict):
        label = err.get("code") or err.get("type") or "error"
        return f"{label}: {err.get('message') or err}"
    return str(err)


def meter_completion(
    budget: agent6_budget.BudgetTracker | None,
    model: str,
    parsed: types.ProviderResponse,
    api_label: str,
) -> None:
    """Book the response's usage, then refuse a completion the upstream failed.

    OpenRouter reports an upstream failure as a 200 whose choice carries
    `finish_reason: "error"` and no content; returned as a finished turn it would
    read as the model going quiet, raised it retries. The usage is booked first:
    the provider billed those tokens either way. Both the decoded and the streamed
    response shape pass through here, so neither drifts from the other.

    Args:
        budget: The run's tracker; None books nothing.
        model: The model billed.
        parsed: The parsed response.
        api_label: The name that leads the error.

    Raises:
        ProviderError: The upstream failed the completion.
    """
    if budget is not None:
        budget.record(
            model=model,
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            cache_read_tokens=parsed.cache_read_tokens,
            cache_creation_tokens=parsed.cache_creation_tokens,
            cost_usd=parsed.cost_usd,
        )
    if (
        parsed.stop_reason.strip().lower() == "error"
        and not parsed.text.strip()
        and not parsed.tool_uses
    ):
        raise types.ProviderError(
            f"{api_label} response carries finish_reason='error' with no content:"
            " the upstream failed this completion"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderCall:
    """One provider API call: the attempt loop around a built request body.

    Attributes:
        api_label: The name that leads API error messages ("OpenAI", "Anthropic").
        api_format: The wire format ("openai", "anthropic", "chatgpt").
        url: The URL dialled.
        body: The request body; an adaptation mutates it in place.
        timeout_s: The read budget in seconds.
        api_key: The static credential, used when `credential` is None.
        credential: A refreshable bearer; a 401 or 403 re-mints it once and retries.
        transcript_sink: Where each round-trip is recorded; None records nothing.
        budget: The run's tracker; None skips metering.
        model: The model billed.
        build_headers: Builds the attempt's headers from its token.
        adapt_400: Receives `(status, error_text, body)` and returns True after
            rewriting the body (and latching provider state) so the next attempt
            sends the adapted request.
        adapt_attempts: One extra attempt per adaptation possible for this body.
        require_metered: Refuses a 2xx body that lacks usage when a budget is set.
        parse: Parses a 2xx body.
        stream: Replaces the non-streaming POST when set; receives the attempt's
            headers, and its errors flow through the same adapt and refresh logic.
    """

    api_label: str
    api_format: str
    url: str
    body: dict[str, Any]
    timeout_s: float
    api_key: str
    credential: types.BearerCredential | None
    transcript_sink: types.TranscriptRecorder | None
    budget: agent6_budget.BudgetTracker | None
    model: str
    build_headers: Callable[[str], dict[str, str]]
    adapt_400: Callable[[int | None, str, dict[str, Any]], bool]
    adapt_attempts: int
    require_metered: Callable[[dict[str, Any]], None]
    parse: Callable[[dict[str, Any]], types.ProviderResponse]
    stream: Callable[[dict[str, str]], types.ProviderResponse] | None = None

    def record(self, headers: dict[str, str], status: int, response: dict[str, Any] | str) -> None:
        """Write one transcript entry for this request; nothing without a sink."""
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                url=self.url,
                request_headers=headers,
                request_body=self.body,
                response_status=status,
                response_body=response,
            )

    def run(self) -> types.ProviderResponse:
        """Make the call, retrying once per credential refresh and per body adaptation.

        Returns:
            The parsed response.

        Raises:
            ProviderError: A transport error, an API error status, a malformed 2xx,
                or a completion the upstream failed.
        """
        cred = self.credential
        max_attempts = (2 if cred is not None else 1) + self.adapt_attempts
        for attempt in range(max_attempts):
            token = cred.token() if cred is not None else self.api_key
            headers = self.build_headers(token)

            if self.stream is not None:
                try:
                    return self.stream(headers)
                except types.ProviderError as exc:
                    if attempt + 1 < max_attempts and self.adapt_400(
                        exc.status_code, str(exc), self.body
                    ):
                        continue
                    if (
                        cred is not None
                        and attempt + 1 < max_attempts
                        and exc.status_code in (401, 403)
                        and cred.invalidate(exc.status_code)
                    ):
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
                raise types.ProviderError(
                    f"HTTP error calling {self.url} ({self.api_format} format): "
                    f"{types.scrub_secret_values(str(exc), headers)}"
                ) from exc
            recorded = False
            if cred is not None and attempt + 1 < max_attempts and resp.status_code in (401, 403):
                # The transcript contract is one file per round-trip, the refused one included.
                self.record(headers, resp.status_code, resp.text[:8192])
                recorded = True
                if cred.invalidate(resp.status_code):
                    continue
            if not 200 <= resp.status_code < 300:
                if not recorded:
                    self.record(headers, resp.status_code, resp.text[:8192])
                if attempt + 1 < max_attempts and self.adapt_400(
                    resp.status_code, resp.text, self.body
                ):
                    continue
                raise types.ProviderError(
                    f"{self.api_label} API error {resp.status_code}: "
                    f"{types.scrub_secret_values(resp.text, headers)[:500]}",
                    status_code=resp.status_code,
                    retry_after_s=types.parse_retry_after(resp.headers),
                )
            return self._decode_success(headers, resp)
        raise types.ProviderError(f"{self.api_label} auth retry exhausted")  # pragma: no cover

    def _decode_success(
        self, headers: dict[str, str], resp: httpx2.Response
    ) -> types.ProviderResponse:
        """Decode, record, check and meter a 2xx response.

        Args:
            headers: The attempt's request headers.
            resp: The response.

        Returns:
            The parsed response.

        Raises:
            ProviderError: A non-JSON or non-object body, an error envelope, or a
                body off the wire shape (each retryable unless the envelope's
                status says otherwise).
        """
        try:
            data: Any = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            # A gateway glitch; without a status the error is retryable.
            self.record(headers, resp.status_code, resp.text[:8192])
            raise types.ProviderError(
                f"non-JSON response from {self.api_label} "
                f"(status {resp.status_code}): "
                f"{types.scrub_secret_values(resp.text, headers)[:500]}"
            ) from exc
        if not isinstance(data, dict):
            self.record(headers, resp.status_code, resp.text[:8192])
            raise types.ProviderError(
                f"{self.api_label} returned a non-object JSON body "
                ""
                f"(status {resp.status_code}): "
                f"{types.scrub_secret_values(resp.text, headers)[:500]}"
            )
        self.record(headers, resp.status_code, data)
        # OpenRouter and LiteLLM deliver an upstream error in a 2xx body; refused before metering.
        err = data.get("error")
        if err and not _has_assistant_output(data):
            raise types.ProviderError(
                f"{self.api_label} error in 2xx body: "
                f"{types.scrub_secret_values(_envelope_detail(err), headers)}",
                status_code=envelope_status(err),
                retry_after_s=types.parse_retry_after(resp.headers),
            )
        if self.budget is not None:
            self.require_metered(data)
        try:
            parsed = self.parse(data)
        except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
            # The one parse seam: a malformed body never bypasses the retry wrapper as a traceback.
            raise types.ProviderError(
                f"{self.api_label} 2xx body did not match the wire shape: {exc!r}"
            ) from exc
        meter_completion(self.budget, self.model, parsed, self.api_label)
        return parsed
