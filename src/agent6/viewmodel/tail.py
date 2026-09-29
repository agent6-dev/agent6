# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tail a JSONL journal with the stdlib alone."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def tail_events(
    path: Path,
    *,
    poll_s: float = 0.25,
    follow: bool = True,
    stop_when_finished: bool = False,
    should_stop: Callable[[], bool] | None = None,
    start_at: int | None = None,
    on_position: Callable[[int], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the journal's events as they are appended.

    Reads bytes and splits on newlines before decoding: a writer flushes a long line
    in several syscalls, so a poll can hit EOF inside a multibyte sequence. Only
    complete lines are decoded; the byte tail stays pending until its newline arrives.
    A malformed line is skipped (a partial write in flight is picked up next poll).

    Args:
        path: The journal file; waited for while it does not exist and `follow` is set.
        poll_s: Seconds between polls.
        follow: Keep polling for new lines after the existing ones; false returns after them.
        stop_when_finished: Return after a `session.end` that nothing follows in its batch.
        should_stop: Polled each round; once true, the lines already appended are
            yielded and the tail returns.
        start_at: A byte offset to start from, skipping the prior executions a viewer
            has seen; measured before the execution starts, so nothing it appends is lost.
        on_position: Hears the byte offset an event ends at, before that event is yielded,
            so a caller can order its own lines against the journal read so far.

    Yields:
        Each JSON-decoded event, in file order.
    """
    while follow and not path.exists():
        if should_stop is not None and should_stop():
            return
        time.sleep(poll_s)
    if not path.exists():
        return

    pos = start_at or 0
    pending = b""
    heard = on_position or _ignore_position
    final_drain = False
    while True:
        if should_stop is not None and not final_drain and should_stop():
            # A worker that exits within one poll leaves its last events unread otherwise.
            final_drain = True
        try:
            with path.open("rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except FileNotFoundError:
            if not follow or final_drain:
                return
            time.sleep(poll_s)
            continue

        if chunk:
            base = pos - len(chunk) - len(pending)
            parsed, pending = _complete_lines(pending + chunk, base)
            # A resume appends past a session.end; only the batch's last one is the real end.
            for i, (end, evt) in enumerate(parsed):
                heard(end)
                yield evt
                if stop_when_finished and i == len(parsed) - 1 and evt.get("type") == "session.end":
                    return

        if not follow or final_drain:
            evt = _parse_event_line(pending)
            if evt is not None:
                heard(pos)
                yield evt
            return
        time.sleep(poll_s)


def _ignore_position(_end: int) -> None:
    return None


def _complete_lines(buffer: bytes, base: int) -> tuple[list[tuple[int, dict[str, Any]]], bytes]:
    """Split a buffer into its complete events and the trailing fragment.

    Args:
        buffer: The bytes read so far.
        base: The byte offset the buffer starts at.

    Returns:
        The (end offset, event) pairs of the complete lines, malformed ones skipped,
        and the bytes after the last newline.
    """
    lines = buffer.split(b"\n")
    parsed: list[tuple[int, dict[str, Any]]] = []
    end = base
    for raw in lines[:-1]:
        end += len(raw) + 1
        if (event := _parse_event_line(raw)) is not None:
            parsed.append((end, event))
    return parsed, lines[-1]


def journal_size(path: Path) -> int:
    """Return the journal's size in bytes, 0 when it does not exist yet."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _parse_event_line(line: bytes) -> dict[str, Any] | None:
    """Return one journal line decoded, None for a blank, malformed or non-object line."""
    if not line.strip():
        return None
    try:
        evt = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return evt if isinstance(evt, dict) else None


class LogTail:
    """Read a journal incrementally for a UI poll loop.

    One reader follows a run and its same-dir resume by byte offset, tolerating a
    partial line at EOF.

    Attributes:
        rewound: The last read found the file shorter than its position (rewritten)
            and started over from its head, so a holder folding the events starts over too.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._pos = 0
        self._pending = b""
        self.rewound = False

    def read(self) -> list[dict[str, Any]]:
        """Return the events appended since the last read, none when the file is unreadable."""
        out: list[dict[str, Any]] = []
        self.rewound = False
        try:
            if self._path.stat().st_size < self._pos:
                self._pos, self._pending, self.rewound = 0, b"", True
            with self._path.open("rb") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
                self._pos = fh.tell()
        except OSError:
            return out
        parsed, self._pending = _complete_lines(self._pending + chunk, 0)
        return [evt for _end, evt in parsed]
