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
    """
    while follow and not path.exists():
        if should_stop is not None and should_stop():
            return
        time.sleep(poll_s)
    if not path.exists():
        return

    pos = 0
    pending = ""
    while True:
        if should_stop is not None and should_stop():
            return
        try:
            with path.open("r", encoding="utf-8") as fh:
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
            lines = pending.split("\n")
            pending = lines[-1]  # last fragment may be incomplete
            for line in lines[:-1]:
                if not line.strip():
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(evt, dict):
                    continue
                yield evt
                if stop_when_finished and evt.get("type") == "run.end":
                    return

        if not follow:
            return
        time.sleep(poll_s)
