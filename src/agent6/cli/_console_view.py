# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a run's event stream to a terminal as a live conversation.

The CLI skin over `viewmodel.TranscriptFold`: assistant reasoning and text stream
inline as they arrive, every tool call shows with its result, and nothing prints
a blank block. One `ConsoleView` serves both `agent6 run` (in-process, subscribed
to the EventSink) and `agent6 watch` (out-of-process, fed by the log tailer), so
the two render identically.

Reasoning/text deltas are streamed by this class (the live-typing feel); the
structural steps (tool call+result, commit, verdict) come from `TranscriptFold`,
which is shared with the TUI and web skins.
"""

from __future__ import annotations

import sys
import time
from threading import RLock
from typing import Any, TextIO

from agent6.viewmodel.transcript import (
    CALL,
    COMMIT,
    DONE,
    RESULT,
    THINK,
    TranscriptFold,
    TranscriptItem,
)

_ANSI = {
    "dim": "\033[2m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
}

_FLUSH_EVERY_S = 0.03  # coalesce streaming-delta flushes; see ConsoleView._raw


class ConsoleView:
    """Fold events to styled terminal lines. `feed`/`__call__` take one event;
    thread-safe so it can subscribe to an EventSink that several roles emit to."""

    def __init__(self, out: TextIO | None = None, *, color: bool | None = None) -> None:
        self._out = out if out is not None else sys.stderr
        self._color = self._out.isatty() if color is None else color
        self._fold = TranscriptFold()
        # Reentrant: the SIGINT steer handler emits an event (re-entering feed on
        # the same thread) while a delta write may hold the lock.
        self._lock = RLock()
        self._phase: str | None = None  # None | "thinking" | "text": the open prose block
        self._last_flush = 0.0

    def __call__(self, event: dict[str, Any]) -> None:
        self.feed(event)

    def feed(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        with self._lock:
            if etype in ("role.thinking_delta", "role.text_delta"):
                self._stream(str(event.get("text", "")), thinking=etype == "role.thinking_delta")
                return
            if etype in ("role.call", "role.result"):
                self._end_block()  # a provider call boundary closes any open prose
                return
            if etype == "run.steer_requested":
                # A Ctrl-C pause message is about to print to the same terminal;
                # close any open (dim) block so it doesn't bleed into the message.
                self._end_block()
                return
            if etype == "run.start":
                task = " ".join(str(event.get("user_task", "")).split())
                self._line(self._c("bold", self._c("cyan", DONE) + " " + task) + "\n")
                return
            for item in self._fold.feed(event):
                self._end_block()
                self._render(item)

    # -- inline prose streaming --------------------------------------------
    def _stream(self, piece: str, *, thinking: bool) -> None:
        want = "thinking" if thinking else "text"
        if self._phase != want:
            if not piece.strip():
                return  # never open a block on whitespace: kills empty response blocks
            self._end_block()
            self._phase = want
            self._raw("  " + (self._dim() + THINK + " " if thinking else ""))
            piece = piece.lstrip()
        # keep wrapped lines under the block's indent; dim (thinking) spans them all
        self._raw(piece.replace("\n", "\n    " if thinking else "\n  "))

    def _end_block(self) -> None:
        if self._phase == "thinking":
            self._raw(self._reset())
        if self._phase is not None:
            self._raw("\n")
            self._flush()  # show the completed prose block now
        self._phase = None

    # -- structural items ---------------------------------------------------
    def _render(self, item: TranscriptItem) -> None:
        if item.kind == "thinking":
            self._line("  " + self._c("dim", THINK + " " + item.body) + "\n")
        elif item.kind == "text":
            self._line("".join(f"  {ln}\n" for ln in item.body.splitlines()))
        elif item.kind == "tool":
            head = "  " + self._c("cyan", CALL) + " " + self._c("bold", item.name)
            if item.arg:
                head += "  " + self._c("dim", item.arg)
            self._line(head + "\n")
            mark = self._c("green" if item.ok else "red", RESULT)
            self._line("    " + mark + " " + self._c("dim", item.detail) + "\n")
            if item.tail:
                self._line("      " + self._c("dim", " ".join(item.tail.split())[:100]) + "\n")
        elif item.kind == "commit":
            self._line(
                "  "
                + self._c("magenta", COMMIT + " commit")
                + self._c("dim", f"  {item.detail}")
                + "\n"
            )
        elif item.kind == "marker":
            self._line("  " + self._c("dim", f"── {item.body} ──") + "\n")
        elif item.kind == "done":
            badge = (
                self._c("green", DONE + " done")
                if item.ok
                else self._c("yellow", DONE + f" {item.name or 'stopped'}")
            )
            self._line("\n" + badge + (f"  {item.body}" if item.body else "") + "\n")
            self._line("  " + self._c("dim", item.detail) + "\n")

    # -- output helpers -----------------------------------------------------
    def _c(self, name: str, text: str) -> str:
        return f"{_ANSI[name]}{text}{_ANSI['reset']}" if self._color else text

    def _dim(self) -> str:
        return _ANSI["dim"] if self._color else ""

    def _reset(self) -> str:
        return _ANSI["reset"] if self._color else ""

    def _raw(self, text: str) -> None:
        # Streaming path: flush at most every _FLUSH_EVERY_S. A per-token flush on
        # a slow terminal (SSH, a busy emulator) backpressures the SSE read in the
        # same thread and can stall the stream; ~30ms is imperceptible and cuts
        # thousands of flushes to a few dozen a second.
        self._out.write(text)
        now = time.monotonic()
        if now - self._last_flush >= _FLUSH_EVERY_S:
            self._out.flush()
            self._last_flush = now

    def _line(self, text: str) -> None:
        # Structural lines (tool call/result, commit, verdict) are discrete: show
        # them at once.
        self._out.write(text)
        self._flush()

    def _flush(self) -> None:
        self._out.flush()
        self._last_flush = time.monotonic()
