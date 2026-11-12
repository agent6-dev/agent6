# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Append structured events to a session's journal.

One JSON object per line in `logs.jsonl`, so a front-end follows a run by
tailing one file. Complete lines stay append-only; a torn tail is dropped before appending.
There is no rotation or production schema validation.
Each durable event is written, flushed and fsynced, and fails loudly: the
journal is the read model every surface trusts. Streaming deltas only flush
and stay best-effort; the transcripts keep the lossless copy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import json
import os
import pathlib
import threading
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agent6 import journal_tail, paths
from agent6.sessions import journal

# Flushed but never fsynced: a reasoning model emits tens of thousands per run.
_EPHEMERAL_EVENTS = frozenset({"model.text_delta", "model.thinking_delta"})
# The fields `emit` adds around a payload; a reader validating a payload strips them.
ENVELOPE_KEYS = frozenset({"ts", "type", "event_id"})


def payload(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return the event's own fields, the envelope left out."""
    return {key: value for key, value in event.items() if key not in ENVELOPE_KEYS}


class EventWriteError(Exception):
    """A durable event could not be appended to the journal.

    A run whose terminal events cannot land would render live forever, so the
    lifecycle stops loudly; a cleanup emit that must not mask an exit suppresses it.
    """


@dataclasses.dataclass(slots=True)
class EventSink:
    """Append events to a journal file; thread-safe.

    The lock is reentrant, so a SIGINT handler emitting mid-emit cannot deadlock.

    Attributes:
        path: The journal file.
    """

    path: pathlib.Path
    _lock: threading.RLock
    _listeners: list[Callable[[dict[str, Any]], None]]

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._listeners = []

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Hand each emitted event to an in-process consumer too, after it lands."""
        self._listeners.append(listener)

    def emit(self, event: journal.Event) -> None:
        """Append one record under its envelope (`ENVELOPE_KEYS`), then notify the listeners.

        A field the writer left unset is left out of the row.

        Attributes:
            ts: The write time in UTC ISO 8601 with microseconds.
            type: The record's `TYPE`.
            event_id: A UUID hex identifier on durable events; absent on streaming deltas.

        Args:
            event: The record.

        Raises:
            EventWriteError: A durable event could not be serialized or written; the
                live view never shows an event the record lost.
        """
        event_type = event.TYPE
        ephemeral = event_type in _EPHEMERAL_EVENTS
        payload: dict[str, Any] = {
            "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="microseconds"),
            "type": event_type,
        }
        # Python mode: a path, a datetime or a stray object meets `_json_default` below.
        payload.update(event.model_dump(exclude_unset=True))
        if not ephemeral:
            payload["event_id"] = uuid.uuid4().hex
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
                    paths.mkdir_for_real_user(self.path.parent)
                journal_tail.heal_torn_tail(self.path)
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
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    return repr(value)
