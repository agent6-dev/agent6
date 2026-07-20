# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structured JSONL event sink.

Emits one JSON object per line to `<run-dir>/logs.jsonl` (the run dir under the
per-repo state dir), to give external tools (and the planned VS Code extension)
a stable, tail-able view of an agent run without parsing the freeform `print` log.

Design notes:
- Write-only and append-only. No reads, no rotation, no schema validation,
  consumers should be defensive.
- Each call opens, writes one line, flushes, closes. Durable events fsync too;
  the high-frequency streaming deltas (see ``_EPHEMERAL_EVENTS``) only flush, so
  a reasoning model's tens of thousands of deltas don't fsync-throttle the run.
- Best-effort: if the directory has been deleted or the FS errors, we swallow
  it. Losing telemetry should never crash the agent.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

# High-frequency streaming deltas: written + flushed (so tailers see them live)
# but NOT fsynced. They are ephemeral UI, reconstructable from the lossless
# transcripts, and a reasoning model can emit tens of thousands per run -- an
# fsync each throttles the SSE reader on a slow disk and stalls the stream.
_EPHEMERAL_EVENTS = frozenset({"role.text_delta", "role.thinking_delta"})


@dataclass(slots=True)
class EventSink:
    """Append structured JSON events to a JSONL file. Thread-safe.

    Uses a *reentrant* lock so emitting from a SIGINT handler (the Ctrl-C steer
    path emits ``run.steer_requested``) cannot deadlock against the main thread
    being mid-``emit``, the handler runs in the same thread and re-acquires.
    """

    path: Path
    _lock: RLock
    _listeners: list[Callable[[dict[str, Any]], None]]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._listeners = []

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Also hand each emitted event to an in-process consumer, as it happens.
        The live CLI renderer uses this; the file stays the source for
        out-of-process viewers (TUI, `watch`, web)."""
        self._listeners.append(listener)

    def emit(self, event_type: str, /, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
            "type": event_type,
        }
        payload.update(fields)
        try:
            line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError):
            # If a field can't be serialized, drop the event rather than crash.
            return
        # Encode HERE, lossily: json.dumps(ensure_ascii=False) passes a lone
        # surrogate (a split emoji escape in model-emitted tool args, a
        # surrogateescape-decoded argv) through as a str, and a text-mode write
        # would then raise UnicodeEncodeError -- a ValueError the OSError guard
        # below never caught, breaking the "telemetry must never break the run"
        # contract. Replacing keeps the event recorded (a dropped run.start
        # makes a run invisible to watch/web) and the file strictly valid UTF-8
        # for every reader.
        data = (line + "\n").encode("utf-8", "replace")
        with self._lock, contextlib.suppress(OSError):  # telemetry must never break the run
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as fh:
                fh.write(data)
                fh.flush()
                if event_type not in _EPHEMERAL_EVENTS:
                    os.fsync(fh.fileno())
        for listener in self._listeners:
            with contextlib.suppress(Exception):  # a UI consumer must never break the run
                listener(payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return repr(value)


@dataclass(slots=True)
class UserInputSink:
    """Append a structured audit record per interactive prompt.

     Separate from :class:`EventSink` (which carries machine telemetry) so
     the human-decision trail lives in its own readable JSONL file at
     ``<run-dir>/user_inputs.jsonl``. Each line has a fixed shape
    , ``ts``, ``kind``, ``prompt``, ``answer``, ``source``, plus any
     extra fields the caller passes. The strict schema is the point: a
     reviewer reading the file can answer "what did the operator decide
     and when" without grepping past unrelated events.
    """

    path: Path
    _lock: RLock

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()

    def record(
        self,
        *,
        kind: str,
        prompt: str,
        answer: str,
        source: str = "stdin",
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
            "kind": kind,
            "prompt": prompt,
            "answer": answer,
            "source": source,
        }
        # Reserved keys win, caller cannot shadow the schema.
        for k, v in extra.items():
            if k not in payload:
                payload[k] = v
        try:
            line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        # Same lossy-encode-then-binary-write as EventSink.emit: a lone
        # surrogate in a prompt/answer must not crash the audit trail.
        data = (line + "\n").encode("utf-8", "replace")
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("ab") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError:
                return
