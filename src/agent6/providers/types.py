# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider-neutral vocabulary shared by every provider implementation.

The errors, the response/tool shapes, the transcript sink, and the Retry-After
parser live here rather than in one concrete provider so the others (and generic
helpers like token_command) import DOWN to this leaf instead of sideways into
`anthropic`, which would be an import cycle.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


class ProviderError(Exception):
    """A provider call failed.

    ``status_code`` is the upstream HTTP status when the failure originated
    from an API error response (None for network/parse failures). The loop's
    retry wrapper uses it to skip retrying permanent client errors such as
    401/402/403 that will never succeed on a second attempt.

    ``retry_after_s`` carries the upstream ``Retry-After`` hint (seconds) on a
    rate-limit/unavailable response (429/503) when present, so the retry wrapper
    waits at least as long as the server asked instead of its own shorter
    backoff. None when absent.
    """

    def __init__(
        self,
        *args: object,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class ProviderAborted(ProviderError):
    """The operator stopped the run mid-call (a streaming turn was interrupted).
    A distinct type so the loop ends the run instead of retrying like a fault."""


class ProviderInterrupted(ProviderError):
    """The operator asked to steer mid-call, so the watchdog closed the (possibly
    long, thinking) stream to bring the loop to its steer boundary promptly. Unlike
    ProviderAborted this does not end the run: the loop shows the steer menu and
    then re-does the turn (or stops/detaches per the operator's choice)."""


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Parse an HTTP ``Retry-After`` header to a non-negative seconds value.

    Accepts the two RFC 7231 forms: a delta in seconds (``"120"``) or an
    HTTP-date (``"Wed, 21 Oct 2026 07:28:00 GMT"``, converted to a delay from
    now). Returns None when the header is absent or unparseable. Case-insensitive
    lookup works with httpx2's header mapping.
    """
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        secs = float(raw)
        # Reject inf/nan (a malformed header); a real delta is finite seconds.
        return max(0.0, secs) if math.isfinite(secs) else None
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - datetime.now(tz=UTC)).total_seconds()
    return max(0.0, delta)


_REDACT_HEADER_NAMES = frozenset({"x-api-key", "authorization", "proxy-authorization", "api-key"})
_REDACTED = "<REDACTED>"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact secret-bearing request headers before they are written to disk."""
    return {k: (_REDACTED if k.lower() in _REDACT_HEADER_NAMES else v) for k, v in headers.items()}


def _max_seq_in_dir(transcripts_dir: Path) -> int:
    """The highest seq already recorded in *transcripts_dir*, or 0 if empty.

    The seq is the 6-digit suffix of each committed `<ts>-<seq>.json` file
    (in-flight `.json.tmp` files don't match the glob). A non-conforming stray
    `.json` is skipped, not fatal."""
    seqs = [
        int(tail)
        for p in transcripts_dir.glob("*.json")
        if (tail := p.stem.rsplit("-", 1)[-1]).isdigit()
    ]
    return max(seqs, default=0)


class TranscriptSink:
    """Append-only writer of one JSON file per LLM round-trip.

    Files live under `transcripts_dir/<utc-iso>-<seq>.json`. The seq counter is
    per-RUN (not per-sink): a sink opened over a dir that already holds a prior
    leg's transcripts continues the counter from the highest present, so seq
    stays globally unique and monotonic across resume legs -- every consumer
    (the load_transcripts sort, the `(seq N)` label, the `--seq` window) treats
    it as a run-global key. Monotonically increasing and thread-safe. Secrets in
    request headers are redacted before any bytes hit disk.
    """

    __slots__ = ("_dir", "_lock", "_seq")

    def __init__(self, transcripts_dir: Path) -> None:
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._dir = transcripts_dir
        self._lock = threading.Lock()
        self._seq = _max_seq_in_dir(transcripts_dir)

    def record(
        self,
        *,
        url: str = "",
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
                "url": url,
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
    """Response from a single provider call."""

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
    # supplied", callers fall back to the price-table estimate in
    # ``BudgetTracker.estimate_usd``.
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
