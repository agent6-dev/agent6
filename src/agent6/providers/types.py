# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider-neutral vocabulary shared by every provider.

The errors, the response and tool shapes, the transcript sink and the Retry-After
parser live in this leaf so every provider imports down to it, never sideways.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import pathlib
import threading
from collections.abc import Mapping
from email import utils
from typing import Any, Protocol

from agent6 import paths, portable


class ProviderError(Exception):
    """A provider call failed.

    Attributes:
        status_code: The upstream HTTP status of an API error response; None for a
            network or parse failure. The retry wrapper skips permanent client
            errors (401, 402, 403) by it.
        retry_after_s: The upstream `Retry-After` hint in seconds on a 429 or 503,
            the floor of the retry wrapper's wait; None when absent.
        provider: The configured provider name, "" until the instrumented wrapper
            stamps it, so a credential hint can name the config key.
        fatal: A permanent failure with no HTTP status (a missing binary, a
            signed-out login); the retry wrapper re-raises it at once.
        attempts: The calls the retry wrapper spent before re-raising; 1 when it
            never retried.
    """

    def __init__(
        self,
        *args: object,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        provider: str = "",
        fatal: bool = False,
        attempts: int = 1,
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.provider = provider
        self.fatal = fatal
        self.attempts = attempts


class ProviderAborted(ProviderError):  # noqa: N818  # a signal, not an error
    """The operator stopped the run mid-call; the loop ends the run instead of retrying."""


class ProviderInterrupted(ProviderError):  # noqa: N818  # a signal, not an error
    """The operator asked to steer mid-call, so the watchdog closed the stream.

    Unlike `ProviderAborted` this does not end the run: the loop shows the steer
    menu and then redoes the turn, or stops or detaches as the operator chooses.
    """


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Parse an HTTP `Retry-After` header in either RFC 7231 form.

    Args:
        headers: The response headers; both spellings of the name are tried.

    Returns:
        The delay in seconds, never negative; None when the header is absent or
        unparseable.
    """
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        secs = float(raw)
        # inf and nan parse as floats but are malformed headers.
        return max(0.0, secs) if math.isfinite(secs) else None
    except ValueError:
        pass
    try:
        when = utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)
    delta = (when - datetime.datetime.now(tz=datetime.UTC)).total_seconds()
    return max(0.0, delta)


_REDACT_HEADER_NAMES = frozenset(
    {"x-api-key", "authorization", "proxy-authorization", "api-key", "chatgpt-account-id"}
)
_REDACTED = "<REDACTED>"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return the request headers with each secret-bearing value replaced by the marker."""
    return {k: (_REDACTED if k.lower() in _REDACT_HEADER_NAMES else v) for k, v in headers.items()}


def scrub_secret_values(text: str, headers: dict[str, str]) -> str:
    """Replace the credential values the headers carry wherever they appear in text.

    Covers a server that echoes the credential in a body or an error excerpt. The
    raw value, the bare token behind a scheme prefix and each one's JSON-escaped
    form are scrubbed; values under 8 characters are skipped, since replacing them
    would shred the text.

    Args:
        text: The text about to be written.
        headers: The request headers, the source of the values.

    Returns:
        The text with every spelling of each credential replaced by the marker.
    """
    for name, value in headers.items():
        if name.lower() not in _REDACT_HEADER_NAMES or not value:
            continue
        for cand in {value, value.split(" ", 1)[-1]}:
            if len(cand) < 8:
                continue
            for spelling in {cand, json.dumps(cand)[1:-1]}:
                text = text.replace(spelling, _REDACTED)
    return text


def _max_seq_in_dir(transcripts_dir: pathlib.Path) -> int:
    """Return the highest seq recorded in the directory, or 0 when it holds none.

    The seq is the suffix of each `<ts>-<seq>.json` file; an in-flight temp file
    and a stray `.json` without the suffix are skipped.
    """
    seqs = [
        int(tail)
        for p in transcripts_dir.glob("*.json")
        if (tail := p.stem.rsplit("-", 1)[-1]).isdigit()
    ]
    return max(seqs, default=0)


class TranscriptRecorder(Protocol):
    """What a provider needs of a transcript sink: the shared sink or a seat's view."""

    def record(
        self,
        *,
        url: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> pathlib.Path:
        """Record one round-trip.

        Args:
            url: The URL dialled.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body.
            response_status: The HTTP status.
            response_body: The response body, parsed or raw.

        Returns:
            The transcript file written.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class RoleTranscriptSink:
    """A `TranscriptSink` view that stamps one seat on every record.

    The seq counter stays run-global; the seat lets a transcript consumer tell the
    worker's conversation from a compaction side-call's round-trip.

    Attributes:
        inner: The shared sink.
        seat: The seat stamped on each record.
    """

    inner: TranscriptSink
    seat: str

    def record(
        self,
        *,
        url: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> pathlib.Path:
        """Record one round-trip through the shared sink with the seat stamped.

        Args:
            url: The URL dialled.
            request_headers: The request headers.
            request_body: The request body.
            response_status: The HTTP status.
            response_body: The response body, parsed or raw.

        Returns:
            The transcript file written.
        """
        return self.inner.record(
            url=url,
            request_headers=request_headers,
            request_body=request_body,
            response_status=response_status,
            response_body=response_body,
            seat=self.seat,
        )


class TranscriptSink:
    """Write one JSON file per model round-trip; append-only and thread-safe.

    Files are `<utc-iso>-<seq>.json`. The seq is run-global: a sink over a directory
    that already holds transcripts continues from the highest present, so every
    consumer can treat it as a unique, monotonic key across resumes. Secret headers
    are redacted and an echoed credential scrubbed before any bytes hit disk.
    """

    __slots__ = ("_dir", "_lock", "_seq")

    def __init__(self, transcripts_dir: pathlib.Path) -> None:
        """Open the sink over a directory, creating it and continuing its seq."""
        paths.mkdir_for_real_user(transcripts_dir)
        self._dir = transcripts_dir
        self._lock = threading.Lock()
        self._seq = _max_seq_in_dir(transcripts_dir)

    def for_seat(self, seat: str) -> RoleTranscriptSink:
        """Return a view of this sink that stamps the seat on everything it records."""
        return RoleTranscriptSink(self, seat)

    def record(
        self,
        *,
        url: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
        seat: str = "",
    ) -> pathlib.Path:
        """Write one round-trip as the next transcript file.

        Args:
            url: The URL dialled.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body.
            response_status: The HTTP status.
            response_body: The response body, parsed or raw.
            seat: The seat that made the call; "" for the worker.

        Returns:
            The transcript file written.
        """
        with self._lock:
            self._seq += 1
            seq = self._seq
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = self._dir / f"{ts}-{seq:06d}.json"
        payload = {
            "ts": ts,
            "seq": seq,
            # Without it a compaction side-call's one-message request reads as a restart.
            "seat": seat,
            "request": {
                "url": url,
                "headers": _redact_headers(request_headers),
                "body": request_body,
            },
            "response": {
                "status": response_status,
                "body": response_body,
            },
        }
        # atomic_write's unpredictable O_EXCL temp name is not symlink-followable.
        text = scrub_secret_values(json.dumps(payload, indent=2, sort_keys=True), request_headers)
        portable.atomic_write(path, text)
        return path


class BearerCredential(Protocol):
    """A refreshable bearer source (`CommandToken`, `ChatGPTCredential`)."""

    def token(self) -> str:
        """Return a fresh-enough bearer."""
        ...

    def invalidate(self, status: int = 401) -> bool:
        """React to one auth failure.

        Args:
            status: The 401 or 403 the transport saw.

        Returns:
            Whether a retry is worthwhile; False when recovery changed nothing.
        """
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One tool exposed to the model.

    Attributes:
        name: The tool's name.
        description: The model-facing description.
        input_schema: The JSON schema of the arguments, generated from a pydantic model.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderResponse:
    """The response to one provider call.

    Attributes:
        text: The visible text.
        tool_uses: The tool calls the model made.
        stop_reason: The provider's stop reason, as spelled on the wire.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        cache_read_tokens: Prompt tokens served from the cache.
        cache_creation_tokens: Prompt tokens written to the cache.
        cost_usd: The gateway-reported cost of this call (OpenRouter's `usage.cost`);
            0 means no authoritative figure, and the price table estimates instead.
        raw: The response body as received.
        refused: Tool-use ids the provider's own front-end answered with an error
            before agent6 could run them, with its text; the loop records the error
            as the result and runs nothing.
    """

    text: str
    tool_uses: tuple[dict[str, Any], ...]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float = 0.0
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)
    refused: dict[str, str] = dataclasses.field(default_factory=dict)


# The output-cap stop reasons (OpenAI "length", Anthropic "max_tokens"), case-folded.
_OUTPUT_CAP_STOP_REASONS = frozenset({"length", "max_tokens"})


def output_cap_truncated(resp: ProviderResponse) -> bool:
    """Return whether the output cap cut the response off.

    A reasoning model can spend its whole cap before emitting content, so a
    truncated call is a failed call, never an empty verdict.
    """
    return resp.stop_reason.strip().lower() in _OUTPUT_CAP_STOP_REASONS
