# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Append structured events to a session's journal.

One JSON object per line in `logs.jsonl`, so a front-end follows a run by
tailing one file. Append-only, with no reads, rotation or schema validation.
Each durable event is written, flushed and fsynced, and fails loudly: the
journal is the read model every surface trusts. Streaming deltas only flush
and stay best-effort; the transcripts keep the lossless copy.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from threading import RLock
from typing import Any

from agent6.paths import mkdir_for_real_user

# Flushed but never fsynced: a reasoning model emits tens of thousands per run.
_EPHEMERAL_EVENTS = frozenset({"role.text_delta", "role.thinking_delta"})


class EventWriteError(Exception):
    """A durable event could not be appended to the journal.

    A run whose terminal events cannot land would render live forever, so the
    lifecycle stops loudly; a cleanup emit that must not mask an exit suppresses it.
    """


@dataclass(slots=True)
class EventSink:
    """Append events to a journal file; thread-safe.

    The lock is reentrant, so a SIGINT handler emitting mid-emit cannot deadlock.

    Attributes:
        path: The journal file.
    """

    path: Path
    _lock: RLock
    _listeners: list[Callable[[dict[str, Any]], None]]

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._listeners = []

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Hand each emitted event to an in-process consumer too, after it lands."""
        self._listeners.append(listener)

    def emit(self, event_type: str, /, **fields: Any) -> None:
        """Append one event, then notify the listeners.

        Args:
            event_type: The event's `type`.
            **fields: Its other fields.

        Raises:
            EventWriteError: A durable event could not be serialized or written; the
                live view never shows an event the record lost.
        """
        ephemeral = event_type in _EPHEMERAL_EVENTS
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
            "type": event_type,
        }
        payload.update(fields)
        try:
            line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            if ephemeral:
                return
            raise EventWriteError(f"cannot serialize event {event_type!r}: {exc}") from exc
        # A lone surrogate from model output would fail a text-mode write; replacing keeps the
        # file valid UTF-8 for every reader.
        data = (line + "\n").encode("utf-8", "replace")
        try:
            with self._lock:
                # Only when missing: on every event the handback would walk the dir under sudo.
                if not self.path.parent.is_dir():
                    mkdir_for_real_user(self.path.parent)
                with self.path.open("ab") as fh:
                    fh.write(data)
                    fh.flush()
                    if not ephemeral:
                        os.fsync(fh.fileno())
        except OSError as exc:
            if not ephemeral:
                raise EventWriteError(f"event journal unwritable at {self.path}: {exc}") from exc
        for listener in self._listeners:
            with contextlib.suppress(Exception):  # a UI consumer must never break the run
                listener(payload)


def _json_default(value: Any) -> Any:
    """Return a path or a datetime as text, and anything else as its repr."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return repr(value)
