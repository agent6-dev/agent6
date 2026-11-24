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
import sys
import threading
from collections.abc import Callable, Generator, Mapping
from email import utils
from typing import Any, Literal, Protocol

from agent6 import kinds, media, paths, portable
from agent6.sessions import objects, retention


@dataclasses.dataclass(frozen=True, slots=True)
class EndpointCompaction:
    """Describe the history the endpoint manages for one model."""

    clears_tool_results: bool = False
    writes_summary: bool = False
    keeps_session: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class Summarize:
    """Ask for a replayable summary of the request's messages."""

    instructions: str


@dataclasses.dataclass(frozen=True, slots=True)
class ClearToolResults:
    """Clear tool output older than the newest results, except for the named tools.

    Attributes:
        keep: The newest tool uses left whole, not counting the named tools.
        exclude_tools: Tools whose results are never cleared.
    """

    keep: int
    exclude_tools: tuple[str, ...]


type CompactionRequest = Summarize | ClearToolResults


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderUsage:
    """The usage one provider call reported, even if the call failed.

    Attributes:
        input_tokens: Fresh input tokens billed.
        output_tokens: Completion tokens billed.
        cache_read_tokens: Prompt tokens served from cache; None when unreported.
        cache_write_tokens: Prompt tokens written to cache; None when unreported.
        cost_usd: The reported cost; None means unreported and zero means a free call.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost_usd: float | None = None


def reported_cost(usage: Mapping[str, Any]) -> float | None:
    """Return gateway-reported dollars, including a measured zero."""
    cost = usage.get("cost")
    # The bound compares exactly; `float` and `math.isfinite` raise on an int past the range.
    if (
        isinstance(cost, int | float)
        and not isinstance(cost, bool)
        and 0 <= cost <= sys.float_info.max
    ):
        return float(cost)
    return None


def requested_tier(backend: str, body: Mapping[str, Any]) -> str | None:
    """Return the tier the entry's `service_tier` key sent.

    Only `openai` and `responses` entries have the key; elsewhere the body field is the
    operator's `extra_body` value and no tier rule applies to it.

    Args:
        backend: The wire format.
        body: The request body.

    Returns:
        The wire tier, or None.
    """
    return body.get("service_tier") if backend in ("openai", "responses") else None


def refuses_service_tier(error: object, status: int | None) -> bool:
    """Recognize a refused parameter or a request error that names the tier field.

    A rate limit can name the model or tier it applies to without rejecting either.

    Args:
        error: An error frame or the scrubbed body of an HTTP error.
        status: The HTTP status or its in-band equivalent.

    Returns:
        Whether the endpoint identified the service tier as the refused setting.
    """
    if isinstance(error, str):
        with contextlib.suppress(ValueError):
            error = json.loads(error)
    if isinstance(error, dict):
        error = error.get("error", error)
    if isinstance(error, dict):
        if error.get("param") is not None:
            return error["param"] == "service_tier"
        error = error.get("message")
    return (
        status in (400, 401, 403, 422)
        and isinstance(error, str)
        and "service tier" in error.lower().replace("_", " ")
    )


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
        fatal: A failure the retry wrapper re-raises at once.
        tier_refused: The endpoint rejected the service tier.
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
        tier_refused: bool = False,
        attempts: int = 1,
        usage: ProviderUsage | None = None,
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.provider = provider
        self.serving = kinds.UNREPORTED
        self.fatal = fatal
        self.tier_refused = tier_refused
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
    actor: str = "worker"
    url: str = ""
    request_headers: dict[str, str] = dataclasses.field(default_factory=dict)
    request_body: dict[str, Any] = dataclasses.field(default_factory=dict)
    response_status: int = 0
    response_headers: dict[str, str] = dataclasses.field(default_factory=dict)
    response_body: dict[str, Any] | str = ""
    usage: ProviderUsage | None = None
    serving: kinds.Serving = kinds.UNREPORTED
    error: str = ""

    def observe_serving(self, response: object) -> None:
        """Keep only serving facts safe to publish in the journal and token summary."""
        if not isinstance(response, dict):
            return
        fields = {
            key: scrub_secret_values(value, self.request_headers)
            for key in ("model", "service_tier")
            if isinstance(value := response.get(key), str) and value
        }
        if fields:
            self.serving = self.serving.observe(fields)

    def write(self, outcome: TranscriptOutcome) -> None:
        """Publish the redacted attempt atomically.

        Attributes:
            ts: The UTC start timestamp in `YYYYMMDDTHHMMSSffffffZ` form.
            seq: The run-global sequence, starting at 1 and continuing across resumes.
            model_call_id: The logical call id from `model.call`; null without a call context.
            request_id: The HTTP attempt id from `provider.request`; empty for a local process.
            actor: The calling role, optionally followed by a colon and detail.
            outcome: `incomplete`, `completed`, `failed`, `interrupted` or `aborted`.
            request: The request formed for the provider.
            request.url: The URL dialled.
            request.headers: The request headers, with secret-bearing values redacted.
            request.body: The redacted body, with message and prompt references into objects/.
            response: What the attempt received before it ended.
            response.service_tier: The processing tier that served the call, when reported.
            response.model: The model that served the call, when reported.
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
            "actor": self.actor,
            "outcome": outcome,
            "request": {
                "url": self.url,
                "headers": _redact_headers(self.request_headers),
                "body": self.request_body,
            },
            "response": {
                **self.serving.wire_fields(),
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
                exc.serving = self.serving
                self.response_status = self.response_status or exc.status_code or 0
                if exc.usage is not None:
                    self.usage = exc.usage
            raise
        finally:
            if outcome == "completed" and not 200 <= self.response_status < 300:
                outcome = "failed"
            self.write(outcome)


class TranscriptRecorder(Protocol):
    """The shared transcript sink or an actor's view."""

    def open(
        self,
        *,
        url: str = "",
        call_context: kinds.CallContext | None = None,
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
        call_context: kinds.CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> pathlib.Path:
        """Record a settled local-process round-trip and return its transcript path."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ActorTranscriptSink:
    """Stamp an actor on records while sharing the run-global sequence."""

    inner: TranscriptSink
    actor: str

    def open(
        self,
        *,
        url: str = "",
        call_context: kinds.CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
    ) -> TranscriptAttempt:
        """Open an attempt through the shared sink with the actor stamped.

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
            actor=self.actor,
        )

    def record(
        self,
        *,
        url: str = "",
        call_context: kinds.CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
    ) -> pathlib.Path:
        """Record a settled local-process round-trip with the actor stamped.

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
            actor=self.actor,
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

    def for_actor(self, actor: str) -> ActorTranscriptSink:
        """Return a view of this sink that stamps the actor on everything it records."""
        return ActorTranscriptSink(self, actor)

    def open(
        self,
        *,
        url: str = "",
        call_context: kinds.CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        actor: str = "worker",
    ) -> TranscriptAttempt:
        """Write the attempt's redacted request before the send.

        Args:
            url: The URL dialled.
            call_context: The logical call identity; None when the caller supplies none.
            request_id: The HTTP attempt id; empty for a local process.
            request_headers: The request headers; secrets are redacted on write.
            request_body: The request body, copied: a retry may adapt the original in place.
            actor: The actor that made the call.

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
            actor=actor,
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
        call_context: kinds.CallContext | None = None,
        request_id: str = "",
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        response_status: int,
        response_body: dict[str, Any] | str,
        actor: str = "worker",
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
            actor: The actor that made the call.

        Returns:
            The transcript file written.
        """
        attempt = self.open(
            url=url,
            call_context=call_context,
            request_id=request_id,
            request_headers=request_headers,
            request_body=request_body,
            actor=actor,
        )
        attempt.response_status = response_status
        attempt.response_body = response_body
        attempt.observe_serving(response_body)
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


type ToolChoice = Literal["auto", "none"]


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
class WebSearch:
    """One search the provider's hosted tool ran inside a response.

    Attributes:
        id: The provider's id for the search.
        query: The query the model sent; empty when the reply names none.
        status: `completed`, or the provider's word for another outcome; empty when unreported.
        urls: The result URLs on Messages; on Responses, the URLs the reply cites after the
            search.
    """

    id: str
    query: str = ""
    status: str = ""
    urls: tuple[str, ...] = ()


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
        cache_write_tokens: Prompt tokens written to the cache.
        cost_usd: The gateway-reported cost of this call (OpenRouter's `usage.cost`);
            None means unreported, so the price table estimates instead; zero means a free call.
        raw: The response body as received.
        serving: The serving facts the backend reported.
        summary_block: The block to replay verbatim after an endpoint summary.
        thinking_blocks_dropped: The number of thinking blocks the endpoint dropped.
        tool_results_cleared: The number of tool results the endpoint cleared.
        refused: Tool-use ids the provider's own front-end answered with an error
            before agent6 could run them, with its text; the loop records the error
            as the result and runs nothing.
        web_searches: The hosted searches the response holds, in order.
        web_searches_billed: The hosted searches the provider bills for this response.
    """

    text: str
    tool_uses: tuple[dict[str, Any], ...]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None = None
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)
    refused: dict[str, str] = dataclasses.field(default_factory=dict)
    summary_block: dict[str, Any] | None = None
    thinking_blocks_dropped: int = 0
    tool_results_cleared: int = 0
    serving: kinds.Serving = kinds.UNREPORTED
    web_searches: tuple[WebSearch, ...] = ()
    web_searches_billed: int = 0

    @property
    def usage(self) -> ProviderUsage:
        """The response's usage, without its content."""
        raw_usage = self.raw.get("usage") or {}
        details = raw_usage.get("input_tokens_details") or raw_usage.get("prompt_tokens_details")
        read_reported = raw_usage.get("cache_read_input_tokens") is not None or (
            isinstance(details, dict) and details.get("cached_tokens") is not None
        )
        write_reported = raw_usage.get("cache_creation_input_tokens") is not None
        return ProviderUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens
            if read_reported or self.cache_read_tokens
            else None,
            cache_write_tokens=self.cache_write_tokens
            if write_reported or self.cache_write_tokens
            else None,
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
