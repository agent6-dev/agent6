# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared SSE lifecycle for the provider streaming paths.

Both providers speak Server-Sent Events over a single POST and need the same
machinery around their event loops: an idle watchdog that heartbeats cannot
satisfy, operator stop/steer that ends an in-flight turn promptly, and
classification of the teardown into ``ProviderAborted`` /
``ProviderInterrupted`` / a retryable ``ProviderError``. Event parsing stays
per-provider (the two wire formats share nothing); this module owns
everything around it.

Why a watchdog at all: httpx2's ``timeout`` (float or ``httpx2.Timeout`` with
``read=``) resets on EVERY received byte, and gateways emit heartbeat bytes
while a request is in flight (OpenRouter/Cloudflare send ``:`` SSE comment
lines every ~15s; Anthropic sends ``ping`` events). If the upstream model
truly hangs (observed: Kimi K2.6 sessions held in ESTABLISHED state with 0
bytes of payload for 800+ seconds while heartbeats continued), the read
timeout never fires and the orchestrator parks forever with no spend cap to
save it. The fix: the per-provider consume loop marks each MEANINGFUL event
on a :class:`StreamClock` (heartbeats deliberately do not count), and a
watchdog thread closes the response once the gap exceeds the threshold. The
blocking ``iter_lines`` then raises an ``httpx2.HTTPError`` that
:meth:`SseCall.run` re-raises as a descriptive error so the loop can
retry-or-quit at its own layer.

Two idle phases, because "no data yet" and "data stopped" mean different
things:

- Before the first real output token the gap is prefill / time-to-first-token,
  which legitimately runs long on a big context or a slow model, so be patient
  (``STREAM_FIRST_DATA_TIMEOUT_S``).
- Once tokens have started, real models emit a data event every few seconds
  even mid-reasoning; a 45s gap then means the stream wedged. Recovering a
  mid-stream wedge in 45s instead of 180s is 4x faster with no false-positive
  on prefill (``STREAM_IDLE_TIMEOUT_S``).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2

from agent6.providers.egress import http_stream
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    TranscriptSink,
    parse_retry_after,
)

STREAM_FIRST_DATA_TIMEOUT_S = 120.0
STREAM_IDLE_TIMEOUT_S = 45.0
# The watchdog also polls should_abort/should_interrupt each tick, so keep it
# short: this bounds how long a Stop/steer/detach waits to end a long in-flight
# turn. A quarter second reads as immediate without the impatient second Ctrl-C.
STREAM_WATCHDOG_TICK_S = 0.25


class StreamClock:
    """Idle bookkeeping the per-provider consume loop feeds.

    ``mark_data()`` on every meaningful wire event; heartbeats must not be
    marked, they are exactly the bytes that mask a wedged upstream.
    ``mark_output()`` when the model has produced real content (text /
    reasoning / tool tokens), which ends the generous prefill budget and
    starts the short mid-stream idle budget.
    """

    __slots__ = ("_seen_output", "last_data_at")

    def __init__(self) -> None:
        self.last_data_at = time.monotonic()
        self._seen_output = threading.Event()

    def mark_data(self) -> None:
        self.last_data_at = time.monotonic()

    def mark_output(self) -> None:
        self._seen_output.set()

    def seen_output(self) -> bool:
        return self._seen_output.is_set()


@dataclass(frozen=True, slots=True)
class SseCall:
    """One provider SSE request: what the shared lifecycle needs around the
    per-provider event loop."""

    api_label: str  # "OpenAI" / "Anthropic"; leads API-error messages
    api_format: str  # "openai" / "anthropic"; names the wire format
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    timeout_s: float
    transcript_sink: TranscriptSink | None
    should_abort: Callable[[], bool] | None
    should_interrupt: Callable[[], bool] | None

    def record(self, *, status: int, response: dict[str, Any] | str) -> None:
        """Write one transcript entry for this request (no-op without a sink)."""
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
        """Open the stream, run ``consume`` under the watchdog, classify teardown.

        ``consume`` iterates ``resp.iter_lines()`` and parses the provider's
        events, marking the clock as it goes; accumulation happens in the
        caller's closure. A ``ProviderError`` it raises (mid-stream error
        frame) propagates unchanged.
        """
        clock = StreamClock()
        aborted = threading.Event()
        interrupted = threading.Event()
        idle_killed = threading.Event()
        watchdog_stop = threading.Event()
        # Mutable holder so the watchdog can reach the response without racing
        # on assignment (the ``with`` body runs in a different frame from the
        # watchdog closure).
        resp_holder: dict[str, httpx2.Response] = {}

        def _watchdog() -> None:
            while not watchdog_stop.wait(STREAM_WATCHDOG_TICK_S):
                resp = resp_holder.get("resp")
                if resp is None:
                    continue
                should_stop = False
                if self.should_abort is not None:
                    # A poll failure must not kill the watchdog -- that would also
                    # disable idle-hang detection. Treat it as "not aborting".
                    with contextlib.suppress(Exception):
                        should_stop = self.should_abort()
                if should_stop:
                    aborted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                # A steer request (Ctrl-C / TUI `s`) closes the stream so a long
                # thinking turn reaches the loop's steer boundary at once.
                should_steer = False
                if self.should_interrupt is not None:
                    with contextlib.suppress(Exception):
                        should_steer = self.should_interrupt()
                if should_steer:
                    interrupted.set()
                    with contextlib.suppress(Exception):
                        resp.close()
                    return
                timeout = (
                    STREAM_IDLE_TIMEOUT_S if clock.seen_output() else STREAM_FIRST_DATA_TIMEOUT_S
                )
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

        try:
            with http_stream(
                "POST",
                self.url,
                headers=self.headers,
                content=json.dumps(self.body).encode("utf-8"),
                timeout=self.timeout_s,
            ) as resp:
                resp_holder["resp"] = resp
                if resp.status_code >= 400:
                    error_body = resp.read().decode("utf-8", errors="replace")[:8192]
                    self.record(status=resp.status_code, response=error_body)
                    raise ProviderError(
                        f"{self.api_label} API error {resp.status_code}: {error_body[:500]}",
                        status_code=resp.status_code,
                        retry_after_s=parse_retry_after(resp.headers),
                    )
                consume(resp, clock)
        except httpx2.HTTPError as exc:
            if interrupted.is_set():
                # The operator asked to steer; the watchdog closed the stream so
                # the loop reaches its steer boundary without waiting out the turn.
                raise ProviderInterrupted("steer requested mid-stream") from exc
            if aborted.is_set():
                # The operator stopped the run; the watchdog closed the stream.
                raise ProviderAborted("run stopped by operator") from exc
            if idle_killed.is_set():
                # Convert the watchdog-induced HTTPError into a purpose-specific
                # ProviderError so the loop's retry/quit path can log a meaningful
                # reason rather than a generic "ReadError" / "connection closed".
                phase_s = (
                    STREAM_IDLE_TIMEOUT_S if clock.seen_output() else STREAM_FIRST_DATA_TIMEOUT_S
                )
                where = "mid-stream" if clock.seen_output() else "before any data (prefill)"
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
                ) from exc
            self.record(status=0, response=f"HTTPError: {exc}")
            raise ProviderError(
                f"HTTP error streaming from {self.url} ({self.api_format} format): {exc}"
            ) from exc
        finally:
            watchdog_stop.set()
