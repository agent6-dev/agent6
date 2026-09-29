# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The SSE lifecycle the HTTP providers' streaming paths share.

The event framing, an idle watchdog that heartbeats cannot satisfy, operator stop
and steer ending an in-flight turn, and the teardown classified into
`ProviderAborted`, `ProviderInterrupted` or a retryable `ProviderError`; what an
event means stays per provider.

httpx2's read timeout resets on every byte, and gateways send heartbeat bytes
(`:` comment lines every ~15s, Anthropic `ping` events) while an upstream model
hangs, so a wedged stream never times out on its own. The consume loop marks each
meaningful event on a `StreamClock` and a watchdog thread closes the response once
the gap exceeds the phase's budget: patient before the first output token
(prefill), tight once output flows, and patient again inside a display-omitted
thinking block, which streams pings only by design.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.budget import BudgetTracker, PlanUsage
from agent6.providers._transport import granular_timeout
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    TranscriptRecorder,
    parse_retry_after,
    scrub_secret_values,
)

STREAM_FIRST_DATA_TIMEOUT_S = 120.0
STREAM_IDLE_TIMEOUT_S = 45.0
# A thinking block streams pings only; the tight mid-stream budget would kill a long think.
STREAM_THINKING_IDLE_TIMEOUT_S = 300.0
# The tick also bounds how long a stop or steer waits; a quarter second reads as immediate.
STREAM_WATCHDOG_TICK_S = 0.25
_ERROR_BODY_PREFIX_BYTES = 8192


@contextlib.contextmanager
def http_stream(
    method: str, url: str, *, headers: dict[str, str], content: bytes, timeout: float
) -> Generator[httpx2.Response]:
    """Open a streaming request; the seam tests stub in place of `httpx2`.

    The connect phase gets its own bound: the watchdog has no response to close
    until the connect returns, so a blackholed connect is httpx2's to cut.

    Args:
        method: The HTTP method.
        url: The URL.
        headers: The request headers.
        content: The request body.
        timeout: The read budget in seconds.

    Yields:
        The open response.
    """
    with httpx2.stream(
        method, url, headers=headers, content=content, timeout=granular_timeout(timeout)
    ) as resp:
        yield resp


def bounded_lines(
    resp: httpx2.Response, *, max_line_bytes: int = 8 * 1024 * 1024
) -> Generator[str]:
    """Iterate the response's lines with a ceiling on each; every consume loop reads through it.

    A line that never ends is bounded by the watchdog, since `iter_lines` reads it
    whole first.

    Args:
        resp: The open response.
        max_line_bytes: The ceiling.

    Yields:
        Each line.

    Raises:
        ProviderError: A line exceeded the ceiling (retryable).
    """
    for line in resp.iter_lines():
        if len(line) * 4 > max_line_bytes and len(line.encode("utf-8")) > max_line_bytes:
            raise ProviderError(
                f"stream frame exceeded {max_line_bytes} bytes; refusing to buffer it"
            )
        yield line


def sse_events(
    resp: httpx2.Response, *, max_event_bytes: int = 8 * 1024 * 1024
) -> Generator[tuple[str, str]]:
    """Frame the response's SSE events.

    Several `data:` fields in one event join with newlines, and an event is
    dispatched only at its blank-line boundary, so a stream cut mid-event ends
    with no event. Comments and unused fields are ignored.

    Args:
        resp: The open response.
        max_event_bytes: The ceiling on one event's data.

    Yields:
        Each event's name and its complete data payload.

    Raises:
        ProviderError: An event exceeded the ceiling (retryable).
    """
    event_type = ""
    data: list[str] = []
    size = 0
    for line in bounded_lines(resp):
        if not line:
            if data:
                yield event_type, "\n".join(data)
            event_type = ""
            data = []
            size = 0
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_type = value
        elif field == "data":
            size += len(value)
            if size > max_event_bytes:
                raise ProviderError(
                    f"SSE event exceeded {max_event_bytes} bytes; refusing to buffer it"
                )
            data.append(value)


def _error_body_prefix(resp: httpx2.Response) -> str:
    """Return the first 8 KiB of an error response's body as text."""
    body = bytearray()
    for chunk in resp.iter_bytes():
        remaining = _ERROR_BODY_PREFIX_BYTES - len(body)
        body.extend(chunk[:remaining])
        if len(body) == _ERROR_BODY_PREFIX_BYTES:
            break
    return body.decode("utf-8", errors="replace")


class StreamClock:
    """The idle bookkeeping the consume loop feeds and the watchdog reads.

    Heartbeats are never marked: they are the bytes that mask a wedged upstream.

    Attributes:
        last_data_at: The monotonic time of the last meaningful wire event.
    """

    __slots__ = ("_in_thinking", "_seen_output", "last_data_at")

    def __init__(self) -> None:
        self.last_data_at = time.monotonic()
        self._seen_output = threading.Event()
        self._in_thinking = threading.Event()

    def mark_data(self) -> None:
        """Mark a meaningful wire event."""
        self.last_data_at = time.monotonic()

    def mark_output(self) -> None:
        """Mark the first real content (text, reasoning or tool tokens); prefill is over."""
        self._seen_output.set()

    def enter_thinking(self) -> None:
        """Enter a display-omitted thinking block, which streams pings only."""
        self._in_thinking.set()

    def exit_thinking(self) -> None:
        """Leave the thinking block."""
        self._in_thinking.clear()

    def idle_budget(self) -> tuple[float, str]:
        """Return the active idle timeout and its label.

        A thinking block gets the patient budget, prefill the first-data budget,
        and output the tight mid-stream budget.
        """
        if self._in_thinking.is_set():
            return (STREAM_THINKING_IDLE_TIMEOUT_S, "mid-thinking")
        if self._seen_output.is_set():
            return (STREAM_IDLE_TIMEOUT_S, "mid-stream")
        return (STREAM_FIRST_DATA_TIMEOUT_S, "before any data (prefill)")


def safe_poll(fn: Callable[[], bool] | None) -> bool:
    """Return an operator-state callback's answer; absent or raising, it reads False."""
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def record_billed_usage(
    budget: BudgetTracker | None,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_usd: float = 0.0,
    plan_usage: PlanUsage | None = None,
) -> None:
    """Record what a call that did not complete already cost.

    A stream that dies after the provider reported usage has been billed, and every
    retry is billed again; counting only completed calls would hide that spend from
    `max_usd`. Nothing is recorded when the provider reported nothing; a reported
    plan window counts as a report even without token counts.

    Args:
        budget: The run's tracker; None records nothing.
        model: The model billed.
        input_tokens: Prompt tokens reported.
        output_tokens: Completion tokens reported.
        cache_read_tokens: Cache-read tokens reported.
        cache_creation_tokens: Cache-write tokens reported.
        cost_usd: The gateway-reported cost.
        plan_usage: The plan window the response reported.
    """
    if budget is None:
        return
    if (
        plan_usage is None
        and cost_usd <= 0
        and not (input_tokens or output_tokens or cache_read_tokens or cache_creation_tokens)
    ):
        return
    budget.record(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cost_usd=cost_usd,
        plan_usage=plan_usage,
    )


@dataclass(frozen=True, slots=True)
class SseCall:
    """One provider SSE request, as the shared lifecycle needs it.

    Attributes:
        api_label: The name that leads API error messages ("OpenAI", "Anthropic").
        api_format: The wire format ("openai", "anthropic", "chatgpt").
        url: The URL dialled.
        headers: The request headers.
        body: The request body.
        timeout_s: The read budget in seconds.
        transcript_sink: Where the round-trip is recorded; None records nothing.
        should_abort: Polled each watchdog tick; True ends the turn as aborted.
        should_interrupt: Polled each watchdog tick; True ends the turn as interrupted.
        response_headers: Receives the response headers once the stream opens.
    """

    api_label: str
    api_format: str
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    timeout_s: float
    transcript_sink: TranscriptRecorder | None
    should_abort: Callable[[], bool] | None
    should_interrupt: Callable[[], bool] | None
    response_headers: Callable[[Mapping[str, str]], None] | None = None

    def record(self, *, status: int, response: dict[str, Any] | str) -> None:
        """Write one transcript entry for this request; nothing without a sink."""
        if self.transcript_sink is not None:
            self.transcript_sink.record(
                url=self.url,
                request_headers=self.headers,
                request_body=self.body,
                response_status=status,
                response_body=response,
            )

    def run(  # noqa: PLR0915
        self, consume: Callable[[httpx2.Response, StreamClock], None]
    ) -> None:
        """Open the stream, run the consumer under the watchdog and classify the teardown.

        The consumer parses the provider's events and marks the clock; accumulation
        happens in its closure.

        Args:
            consume: Reads the open response to its end.

        Raises:
            ProviderInterrupted: The operator asked to steer mid-stream.
            ProviderAborted: The operator stopped the run mid-stream.
            ProviderError: An API error status, the idle watchdog, a transport
                error, or a malformed 2xx frame (a shape error from the consumer is
                normalised here so it never bypasses the retry wrapper); one the
                consumer raises itself propagates unchanged.
        """
        clock = StreamClock()
        aborted = threading.Event()
        interrupted = threading.Event()
        idle_killed = threading.Event()
        watchdog_stop = threading.Event()
        # The watchdog closure reaches the response through the holder, never racing its assignment.
        resp_holder: dict[str, httpx2.Response] = {}

        def _watchdog() -> None:
            while not watchdog_stop.wait(STREAM_WATCHDOG_TICK_S):
                resp = resp_holder.get("resp")
                if resp is None:
                    continue
                if safe_poll(self.should_abort):
                    aborted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                if safe_poll(self.should_interrupt):
                    interrupted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                timeout, _ = clock.idle_budget()
                if time.monotonic() - clock.last_data_at <= timeout:
                    continue
                idle_killed.set()
                with contextlib.suppress(Exception):
                    resp.close()
                return

        watchdog = threading.Thread(
            target=_watchdog, name=f"agent6-{self.api_format}-sse-watchdog", daemon=True
        )
        watchdog.start()

        def _raise_watchdog(cause: Exception | None = None) -> None:
            if interrupted.is_set():
                raise ProviderInterrupted("steer requested mid-stream") from cause
            if aborted.is_set():
                raise ProviderAborted("run stopped by operator") from cause
            if idle_killed.is_set():
                phase_s, where = clock.idle_budget()
                self.record(
                    status=0,
                    response=(
                        f"SSE idle watchdog: no data event for {phase_s:.0f}s {where} "
                        f"(only heartbeats). Upstream model appears wedged."
                    ),
                )
                raise ProviderError(
                    f"{self.api_label} SSE stream idle for >{phase_s:.0f}s {where} "
                    "(only heartbeats received); upstream model appears wedged."
                ) from cause

        try:
            with http_stream(
                "POST",
                self.url,
                headers=self.headers,
                content=json.dumps(self.body).encode("utf-8"),
                timeout=self.timeout_s,
            ) as resp:
                resp_holder["resp"] = resp
                if self.response_headers is not None:
                    self.response_headers(resp.headers)
                if not 200 <= resp.status_code < 300:
                    error_body = _error_body_prefix(resp)
                    self.record(status=resp.status_code, response=error_body)
                    raise ProviderError(
                        f"{self.api_label} API error {resp.status_code}: "
                        f"{scrub_secret_values(error_body, self.headers)[:500]}",
                        status_code=resp.status_code,
                        retry_after_s=parse_retry_after(resp.headers),
                    )
                try:
                    consume(resp, clock)
                except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
                    raise ProviderError(
                        f"{self.api_label} stream frame did not match the wire shape:"
                        f" {exc!r} (malformed 2xx event; retryable)"
                    ) from exc
                # A cross-thread close may end iteration as a clean EOF rather than an HTTPError.
                _raise_watchdog()
        except httpx2.HTTPError as exc:
            _raise_watchdog(exc)
            self.record(status=0, response=f"HTTPError: {exc}")
            raise ProviderError(
                f"HTTP error streaming from {self.url} ({self.api_format} format): {exc}"
            ) from exc
        finally:
            watchdog_stop.set()
