# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structured JSONL event sink.

Emits one JSON object per line to a path under `.agent6/runs/<id>/logs.jsonl`,
to give external tools (and the planned VS Code extension) a stable, tail-able
view of an agent run without parsing the freeform `print` log.

Design notes:
- Write-only and append-only. No reads, no rotation, no schema validation —
  consumers should be defensive.
- Each call opens, writes one line, fsyncs, closes. Cheap enough at our event
  rate (a handful per step) and removes any "did the buffer flush" worry for
  external tailers.
- Best-effort: if the directory has been deleted or the FS errors, we swallow
  it. Losing telemetry should never crash the agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass(slots=True)
class EventSink:
    """Append structured JSON events to a JSONL file. Thread-safe.

    Uses a *reentrant* lock so emitting from a SIGINT handler (the Ctrl-C steer
    path emits ``run.steer_requested``) cannot deadlock against the main thread
    being mid-``emit`` — the handler runs in the same thread and re-acquires.
    """

    path: Path
    _lock: RLock

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()

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
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError:
                # Telemetry must never break the run.
                return


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
    ``.agent6/runs/<id>/user_inputs.jsonl``. Each line has a fixed shape
    — ``ts``, ``kind``, ``prompt``, ``answer``, ``source`` — plus any
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
        # Reserved keys win — caller cannot shadow the schema.
        for k, v in extra.items():
            if k not in payload:
                payload[k] = v
        try:
            line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError:
                return
