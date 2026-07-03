# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Stdlib JSONL tail-follower. No third-party deps."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def tail_events(  # noqa: PLR0912
    path: Path,
    *,
    poll_s: float = 0.25,
    follow: bool = True,
    stop_when_finished: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSON-decoded events from *path* as they are appended.

    - Waits for the file to appear (up to forever if follow=True).
    - Yields each existing line on startup, then tails for new ones.
    - If *stop_when_finished* is true, exits after a `run.end` event.
    - If *should_stop* is given, exits at the next poll boundary once it returns
      True (lets a caller cancel a follow, e.g. on client disconnect).
    - If *follow* is false, yields existing lines and returns.
    - Skips malformed JSON lines silently (the writer may have a partial
      write in flight; we'll pick it up on the next poll).

    Reads bytes and splits on b"\\n" before decoding: writers flush long lines
    in multiple syscalls, so a poll can hit EOF mid multibyte UTF-8 sequence and
    a text-mode read() would raise UnicodeDecodeError. Only complete lines are
    decoded; the byte tail stays pending until its newline arrives.
    """
    while follow and not path.exists():
        if should_stop is not None and should_stop():
            return
        time.sleep(poll_s)
    if not path.exists():
        return

    pos = 0
    pending = b""
    while True:
        if should_stop is not None and should_stop():
            return
        try:
            with path.open("rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except FileNotFoundError:
            if not follow:
                return
            time.sleep(poll_s)
            continue

        if chunk:
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines[-1]  # last fragment may be incomplete
            for line in lines[:-1]:
                evt = _parse_event_line(line)
                if evt is None:
                    continue
                yield evt
                if stop_when_finished and evt.get("type") == "run.end":
                    return

        if not follow:
            evt = _parse_event_line(pending)
            if evt is not None:
                yield evt
            return
        time.sleep(poll_s)


def _parse_event_line(line: bytes) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        evt = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return evt if isinstance(evt, dict) else None
