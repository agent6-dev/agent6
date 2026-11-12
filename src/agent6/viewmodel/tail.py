# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tail a JSONL journal with the stdlib alone."""

from __future__ import annotations

import json
import os
import pathlib
import time
from collections.abc import Callable, Iterator
from typing import Any


def tail_events(
    path: pathlib.Path,
    *,
    poll_s: float = 0.25,
    follow: bool = True,
    stop_when_finished: bool = False,
    should_stop: Callable[[], bool] | None = None,
    start_at: int | None = None,
    on_position: Callable[[int], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the journal's events as they are appended.

    The decoding reader over `_tail_batches`: a malformed line is skipped (a partial write
    in flight is picked up next poll).

    Args:
        path: The journal file; waited for while it does not exist and `follow` is set.
        poll_s: Seconds between polls.
        follow: Keep polling for new lines after the existing ones; false returns after them.
        stop_when_finished: Return after a `session.stop` that nothing follows in its batch.
        should_stop: Polled each round; once true, the lines already appended are
            yielded and the tail returns.
        start_at: A byte offset to start from, skipping the prior executions a viewer
            has seen; measured before the execution starts, so nothing it appends is lost.
        on_position: Hears the byte offset an event ends at, before that event is yielded,
            so a caller can order its own lines against the journal read so far.

    Yields:
        Each JSON-decoded event, in file order.
    """
    heard = on_position or _ignore_position
    batches = _tail_batches(
        path, poll_s=poll_s, follow=follow, should_stop=should_stop, start_at=start_at, torn=True
    )
    for batch in batches:
        parsed = [(end, evt) for end, raw in batch if (evt := _parse_event_line(raw)) is not None]
        # A resume appends past a session.stop; only the batch's last one is the real end.
        for i, (end, evt) in enumerate(parsed):
            heard(end)
            yield evt
            if stop_when_finished and i == len(parsed) - 1 and evt.get("type") == "session.stop":
                return


def tail_lines(
    path: pathlib.Path,
    *,
    poll_s: float = 0.25,
    should_stop: Callable[[], bool] | None = None,
    start_at: int | None = None,
) -> Iterator[bytes]:
    """Yield the journal's complete lines as bytes, as they are appended.

    The raw reader over the same framing as `tail_events`: a malformed line is yielded as it
    is, and a line torn at EOF waits for its newline. It follows until `should_stop` says so,
    then hands over what was appended.

    Args:
        path: The journal file; waited for while it does not exist.
        poll_s: Seconds between polls.
        should_stop: Polled each round; once true, the lines already appended are
            yielded and the tail returns.
        start_at: The byte offset to start from.

    Yields:
        Each complete line without its newline, in file order.
    """
    batches = _tail_batches(
        path, poll_s=poll_s, follow=True, should_stop=should_stop, start_at=start_at, torn=False
    )
    for batch in batches:
        for _end, raw in batch:
            yield raw


def _tail_batches(
    path: pathlib.Path,
    *,
    poll_s: float,
    follow: bool,
    should_stop: Callable[[], bool] | None,
    start_at: int | None,
    torn: bool,
) -> Iterator[list[tuple[int, bytes]]]:
    """Yield one batch of complete lines per poll, each with the offset it ends at.

    Reads bytes and splits on newlines before decoding: a writer flushes a long line in
    several syscalls, so a poll can hit EOF inside a multibyte sequence. Each poll rereads
    from the last complete line's end, so a repaired tail cannot leave stale partial bytes
    in the reader; a file shorter than that end (rotated, truncated) is read from its head.

    Args:
        path: The journal file; waited for while it does not exist and `follow` is set.
        poll_s: Seconds between polls.
        follow: Keep polling after the existing lines; false returns after them.
        should_stop: Polled each round; once true, one last read drains the file.
        start_at: A byte offset to start from.
        torn: Whether the last batch carries a line left torn at EOF.

    Yields:
        The (end offset, raw line) pairs of each poll's complete lines.
    """
    while follow and not path.exists():
        if should_stop is not None and should_stop():
            return
        time.sleep(poll_s)
    if not path.exists():
        return

    pos = start_at or 0
    final_drain = False
    while True:
        if should_stop is not None and not final_drain and should_stop():
            # A worker that exits within one poll leaves its last events unread otherwise.
            final_drain = True
        try:
            with path.open("rb") as fh:
                if fh.seek(0, os.SEEK_END) < pos:
                    pos = 0
                fh.seek(pos)
                chunk = fh.read()
        except FileNotFoundError:
            if not follow or final_drain:
                return
            time.sleep(poll_s)
            continue

        lines, pending = _complete_lines(chunk, pos)
        pos += len(chunk) - len(pending)
        last = not follow or final_drain
        if last and torn and pending.strip():
            lines.append((pos + len(pending), pending))
        yield lines
        if last:
            return
        time.sleep(poll_s)


def _ignore_position(_end: int) -> None:
    return None


def _complete_lines(buffer: bytes, base: int) -> tuple[list[tuple[int, bytes]], bytes]:
    """Split a buffer into its complete lines and the trailing fragment.

    Args:
        buffer: The bytes read so far.
        base: The byte offset the buffer starts at.

    Returns:
        The (end offset, raw line) pairs of the complete lines, and the bytes after the
        last newline.
    """
    lines = buffer.split(b"\n")
    out: list[tuple[int, bytes]] = []
    end = base
    for raw in lines[:-1]:
        end += len(raw) + 1
        out.append((end, raw))
    return out, lines[-1]


def journal_size(path: pathlib.Path) -> int:
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

    One reader follows a run and its same-dir resume from the last complete line's
    end, rereading a partial line at EOF until it is completed or repaired.

    Attributes:
        rewound: The last read found the file shorter than its last complete line's end
            and started over from its head, so a holder folding the events starts over too.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._pos = 0
        self.rewound = False

    def read(self) -> list[dict[str, Any]]:
        """Return the events appended since the last read, none when the file is unreadable."""
        self.rewound = False
        try:
            with self._path.open("rb") as fh:
                if fh.seek(0, os.SEEK_END) < self._pos:
                    self._pos, self.rewound = 0, True
                fh.seek(self._pos)
                chunk = fh.read()
        except OSError:
            return []
        lines, pending = _complete_lines(chunk, self._pos)
        self._pos += len(chunk) - len(pending)
        return [evt for _end, raw in lines if (evt := _parse_event_line(raw)) is not None]
