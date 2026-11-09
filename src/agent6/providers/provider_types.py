# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider-neutral vocabulary shared by every provider.

The errors, the response and tool shapes, the transcript sink and the Retry-After
parser live in this leaf so every provider imports down to it, never sideways.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime
import json
import math
import pathlib
import threading
import uuid
from collections.abc import Callable, Generator, Mapping
from email import utils
from typing import Any, Literal, Protocol

from agent6 import media, paths, portable
from agent6.sessions import objects, retention


@dataclasses.dataclass(frozen=True, slots=True)
class CallContext:
    """Identify one logical model call across its request attempts.

    Attributes:
        model_call_id: The call's id, shared by its events and transcripts.
        panel_id: The review panel's id; None outside a panel.
        seat_index: The seat's zero-based roster position; None outside a review seat.
    """

    model_call_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    panel_id: str | None = None
    seat_index: int | None = None

    def event_fields(self) -> dict[str, str | int]:
        """Return the call identity, omitting absent seat fields."""
        fields: dict[str, str | int] = {"model_call_id": self.model_call_id}
        if self.panel_id is not None:
            fields["panel_id"] = self.panel_id
        if self.seat_index is not None:
            fields["seat_index"] = self.seat_index
        return fields


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderUsage:
    """The usage one provider call reported, even if the call failed.

    Attributes:
        input_tokens: Fresh input tokens billed.
        output_tokens: Completion tokens billed.
        cache_read_tokens: Prompt tokens served from the cache.
        cache_creation_tokens: Prompt tokens written to the cache.
        cost_usd: The reported cost; None means unreported and zero means a free call.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float | None = None


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
        usage: The failed call's reported usage; None when none was observed.
    """

    def __init__(
        self,
        *args: object,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        provider: str = "",
        fatal: bool = False,
        attempts: int = 1,
        usage: ProviderUsage | None = None,
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.provider = provider
        self.fatal = fatal
        self.attempts = attempts
        self.usage = usage


class MediaInputError(ProviderError):
    """A recognized image or document rejection, without upstream payload text.

    Attributes:
        modality: The rejected input modality.
        reason: Unsupported input, malformed content or a size limit.
        call_ids: Affected tool calls; empty means all attachments of this modality.
    """

    def __init__(
        self,
        *,
        modality: Literal["image", "document"],
        reason: Literal["unsupported", "invalid", "too_large"],
        call_ids: tuple[str, ...] = (),
        status_code: int = 400,
    ) -> None:
        super().__init__(
            f"{modality} input rejected: {reason}", status_code=status_code, fatal=True
        )
        self.modality = modality
        self.reason = reason
        self.call_ids = call_ids


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
    {
        "api-key",
        "authorization",
        "chatgpt-account-id",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
    }
)
_REDACTED = "<REDACTED>"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return the headers with each secret-bearing value replaced by the marker."""
    return {k: (_REDACTED if k.lower() in _REDACT_HEADER_NAMES else v) for k, v in headers.items()}


def scrub_secret_values(text: str, headers: dict[str, str]) -> str:
    """Replace the credential values the headers carry wherever they appear in text.

    Covers a server that echoes the credential in a body or an error excerpt. The
    raw value, the bare token behind a scheme prefix and each one's JSON-escaped
    form are scrubbed; values under 8 characters are skipped, since replacing them
    would shred the text.

    Args:
        text: The text about to be written.
        headers: The headers carrying the credential values.

    Returns:
        The text with every spelling of each credential replaced by the marker.
    """
    spellings: set[str] = set()
    for name, value in headers.items():
        if name.lower() not in _REDACT_HEADER_NAMES or not value:
            continue
        candidates = {value}
        if name.lower() in {"authorization", "proxy-authorization"}:
            candidates.add(value.split(" ", 1)[-1])
        for candidate in candidates:
            if len(candidate) >= 8:
                spellings.update((candidate, json.dumps(candidate)[1:-1]))
    for spelling in sorted(spellings, key=len, reverse=True):
        text = text.replace(spelling, _REDACTED)
    return text


type TranscriptOutcome = Literal["incomplete", "completed", "failed", "interrupted", "aborted"]


@dataclasses.dataclass(slots=True)
class TranscriptAttempt:
    """Keep one attempt's evidence until its atomic final rewrite.

    A missing path disables disk writes. The record holds no raw stream events: a
    credential split across two deltas would escape the whole-string scrub.
    """

    path: pathlib.Path | None = None
    objects_dir: pathlib.Path | None = None
    ts: str = ""
    seq: int = 0
    model_call_id: str | None = None
    request_id: str = ""
    seat: str = ""
    url: str = ""
    request_headers: dict[str, str] = dataclasses.field(default_factory=dict)
    request_body: dict[str, Any] = dataclasses.field(default_factory=dict)
    response_status: int = 0
    response_headers: dict[str, str] = dataclasses.field(default_factory=dict)
    response_body: dict[str, Any] | str = ""
    usage: ProviderUsage | None = None
    error: str = ""

    def write(self, outcome: TranscriptOutcome) -> None:
        """Publish the redacted attempt atomically.

        Attributes:
            ts: The UTC start timestamp in `YYYYMMDDTHHMMSSffffffZ` form.
            seq: The run-global sequence, starting at 1 and continuing across resumes.
            model_call_id: The logical call id from `role.call`; null without a call context.
            request_id: The HTTP attempt id from `provider.request`; empty for a local process.
            seat: The role or review focus label, not a unique seat identity.
            outcome: `incomplete`, `completed`, `failed`, `interrupted` or `aborted`.
            request: The request formed for the provider.
            request.url: The URL dialled.
            request.headers: The request headers, with secret-bearing values redacted.
            request.body: The redacted body, with message and prompt references into objects/.
            response: What the attempt received before it ended.
            response.status: The HTTP status, or 0 when no response opened.
            response.headers: The response headers, with secret-bearing values redacted.
            response.body: The parsed response body or raw error text.
            usage: The observed token counts and cost; null when the provider reported none.
            error: The exception type and reason; empty when none was raised.

        Args:
            outcome: The attempt's current outcome.
        """
        if self.path is None:
            return
        payload = {
            "ts": self.ts,
            "seq": self.seq,
            "model_call_id": self.model_call_id,
            "request_id": self.request_id,
            "seat": self.seat,
            "outcome": outcome,
            "request": {
                "url": self.url,
                "headers": _redact_headers(self.request_headers),
                "body": self.request_body,
            },
            "response": {
                "status": self.response_status,
                "headers": _redact_headers(self.response_headers),
                "body": self.response_body,
            },
            "usage": dataclasses.asdict(self.usage) if self.usage is not None else None,
            "error": self.error,
        }
        text = scrub_secret_values(json.dumps(payload), self.request_headers)
        clean = json.loads(scrub_secret_values(text, self.response_headers))
        clean = media.omit_payloads(clean)
        assert self.objects_dir is not None
        clean["request"]["body"] = objects.reference_fields(
            clean["request"]["body"], self.objects_dir
        )
        portable.atomic_write(self.path, json.dumps(clean, indent=2, sort_keys=True))

    @contextlib.contextmanager
    def capture(self) -> Generator[TranscriptAttempt]:
        """Finalize the attempt on every unwind, including cancellation.

        Yields:
            The mutable response evidence.
        """
        outcome: TranscriptOutcome = "completed"
        try:
            yield self
        except BaseException as exc:
            if isinstance(exc, ProviderInterrupted):
                outcome = "interrupted"
            elif isinstance(exc, ProviderAborted) or not isinstance(exc, Exception):
                outcome = "aborted"
            else:
                outcome = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            if not self.response_body:
                self.response_body = self.error
            if isinstance(exc, ProviderError):
                self.response_status = self.response_status or exc.status_code or 0
                if exc.usage is not None:
                    self.usage = exc.usage
            raise
        finally:
            if outcome == "completed" and not 200 <= self.response_status < 300:
                outcome = "failed"
            self.write(outcome)


class TranscriptRecorder(Protocol):
    """The shared transcript sink or a seat's view."""

    def open(
        self,
        *,
        url: str = "",
        call_context: CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
    ) -> TranscriptAttempt:
        """Write the redacted request before send and return its attempt."""
        ...

    def record(
        self,
        *,
        url: str = "",
        call_context: CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> pathlib.Path:
        """Record a settled local-process round-trip and return its transcript path."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class RoleTranscriptSink:
    """Stamp a seat on records while sharing the run-global sequence."""

    inner: TranscriptSink
    seat: str

    def open(
        self,
        *,
        url: str = "",
        call_context: CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
    ) -> TranscriptAttempt:
        """Open an attempt through the shared sink with the seat stamped.

        Args:
            url: The URL dialled.
            call_context: The logical call identity; None when the caller supplies none.
            request_id: The HTTP attempt id; empty for a local process.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body.

        Returns:
            The open attempt.
        """
        return self.inner.open(
            url=url,
            call_context=call_context,
            request_id=request_id,
            request_headers=request_headers,
            request_body=request_body,
            seat=self.seat,
        )

    def record(
        self,
        *,
        url: str = "",
        call_context: CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> pathlib.Path:
        """Record a settled local-process round-trip with the seat stamped.

        Args:
            url: The URL dialled.
            call_context: The logical call identity; None when the caller supplies none.
            request_id: The HTTP attempt id; empty for a local process.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body.
            response_status: The HTTP status.
            response_body: The response body, parsed or raw.

        Returns:
            The transcript file written.
        """
        return self.inner.record(
            url=url,
            call_context=call_context,
            request_id=request_id,
            request_headers=request_headers,
            request_body=request_body,
            response_status=response_status,
            response_body=response_body,
            seat=self.seat,
        )


class TranscriptSink:
    """Write one JSON file per attempt with a thread-safe, run-global sequence.

    Each request is published before send and atomically rewritten on finalization.
    A sink over existing transcripts continues from the highest sequence.
    """

    __slots__ = ("_dir", "_lock", "_objects_dir", "_seq")

    def __init__(self, transcripts_dir: pathlib.Path, *, objects_dir: pathlib.Path) -> None:
        """Open the transcript directory with an explicit store for referenced values."""
        paths.mkdir_for_real_user(transcripts_dir)
        self._dir = transcripts_dir
        self._objects_dir = objects_dir
        self._lock = threading.Lock()
        self._seq = retention.last_sequence(transcripts_dir)

    def for_seat(self, seat: str) -> RoleTranscriptSink:
        """Return a view of this sink that stamps the seat on everything it records."""
        return RoleTranscriptSink(self, seat)

    def open(
        self,
        *,
        url: str = "",
        call_context: CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        seat: str = "",
    ) -> TranscriptAttempt:
        """Write the attempt's redacted request before the send.

        Args:
            url: The URL dialled.
            call_context: The logical call identity; None when the caller supplies none.
            request_id: The HTTP attempt id; empty for a local process.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body, copied: a retry may adapt the original in place.
            seat: The seat that made the call; "" for the worker.

        Returns:
            The open attempt, on disk with outcome incomplete.
        """
        with self._lock:
            self._seq += 1
            seq = self._seq
        ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        attempt = TranscriptAttempt(
            path=self._dir / f"{ts}-{seq:06d}.json",
            objects_dir=self._objects_dir,
            ts=ts,
            seq=seq,
            model_call_id=call_context.model_call_id if call_context is not None else None,
            request_id=request_id,
            seat=seat,
            url=url,
            request_headers=dict(request_headers),
            request_body=copy.deepcopy(request_body),
        )
        attempt.write("incomplete")
        return attempt

    def record(
        self,
        *,
        url: str = "",
        call_context: CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
        seat: str = "",
    ) -> pathlib.Path:
        """Record a settled local-process round-trip.

        Args:
            url: The URL dialled.
            call_context: The logical call identity; None when the caller supplies none.
            request_id: The HTTP attempt id; empty for a local process.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body.
            response_status: The HTTP status.
            response_body: The response body, parsed or raw.
            seat: The seat that made the call; "" for the worker.

        Returns:
            The transcript file written.
        """
        attempt = self.open(
            url=url,
            call_context=call_context,
            request_id=request_id,
            request_headers=request_headers,
            request_body=request_body,
            seat=seat,
        )
        attempt.response_status = response_status
        attempt.response_body = response_body
        attempt.write("completed" if 200 <= response_status < 300 else "failed")
        assert attempt.path is not None
        return attempt.path


@dataclasses.dataclass(frozen=True, slots=True)
class AuthEvent:
    """A settled authentication operation, without credential or endpoint data.

    Attributes:
        kind: The operation whose outcome is known.
        outcome: Whether it succeeded or was refused.
    """

    kind: Literal["refresh", "exchange", "recovery"]
    outcome: Literal["success", "failure"]


type AuthObserver = Callable[[AuthEvent], None]


class BearerCredential(Protocol):
    """A refreshable bearer source (`CommandToken`, `OAuthCredential`)."""

    def token(self) -> str:
        """Return a fresh-enough bearer."""
        ...

    def invalidate(self, status: int, rejected_bearer: str) -> bool:
        """Return whether the rejected bearer arms credential recovery."""
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
            None means unreported, so the price table estimates instead; zero means a free call.
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
    cost_usd: float | None = None
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)
    refused: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def usage(self) -> ProviderUsage:
        """The response's usage, without its content."""
        return ProviderUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
            cost_usd=self.cost_usd,
        )


# The output-cap stop reasons (OpenAI "length", Anthropic "max_tokens"), case-folded.
_OUTPUT_CAP_STOP_REASONS = frozenset({"length", "max_tokens"})


def output_cap_truncated(resp: ProviderResponse) -> bool:
    """Return whether the output cap cut the response off.

    A reasoning model can spend its whole cap before emitting content, so a
    truncated call is a failed call, never an empty verdict.

    Args:
        resp: The response.

    Returns:
        True for the stop reason `length` or `max_tokens`, ignoring case.
    """
    return resp.stop_reason.strip().lower() in _OUTPUT_CAP_STOP_REASONS
