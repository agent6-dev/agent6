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
from threading import Event, RLock, Thread
from typing import Any, TextIO

from agent6.cli._task_tree import tree_lines_from_event_nodes
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
_HEARTBEAT_TICK_S = 0.5  # how often the spinner refreshes
_STALL_AFTER_S = 1.5  # show the heartbeat once output has been silent this long
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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
        self._plan_count = 0  # tasks shown in the last plan block; reprint when it grows
        # Live heartbeat: a turn can stream text then wedge mid-token (a stalled
        # SSE stream) or pause between turns with nothing on screen. A background
        # thread shows a spinner + "working… Ns" during silence so the run never
        # looks hung; only on a real terminal (no spinner in a pipe or a test).
        self._last_output_at = time.monotonic()
        self._active = False  # run is between run.start and run.end (a turn or a tool)
        self._status_active = False  # a transient spinner line is on screen now
        self._spin = 0
        self._stop = Event()
        self._heartbeat: Thread | None = None
        if self._out.isatty():
            self._heartbeat = Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat.start()

    def __call__(self, event: dict[str, Any]) -> None:
        self.feed(event)

    def feed(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        with self._lock:
            # The heartbeat spins whenever the run is active and output has gone
            # silent -- covering BOTH a thinking provider call AND a long tool /
            # verify command running in the jail (which happens between role.result
            # and the next role.call, so a role-only flag would miss it and the
            # CLI would look frozen through a whole test suite).
            if etype in ("run.start", "role.call", "tool.call"):
                self._active = True
            elif etype in ("run.end", "run.steer_requested"):
                self._active = False
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
            if etype == "graph.update":
                self._render_plan(event)
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
        self._last_output_at = time.monotonic()  # a delta is real progress
        # keep wrapped lines under the block's indent; dim (thinking) spans them all
        self._raw(piece.replace("\n", "\n    " if thinking else "\n  "))

    def _end_block(self) -> None:
        if self._phase == "thinking":
            self._raw(self._reset())
        if self._phase is not None:
            self._raw("\n")
            self._flush()  # show the completed prose block now
        self._phase = None

    def _render_plan(self, event: dict[str, Any]) -> None:
        """Print the decomposed task tree when it first appears and each time it
        grows (new subtasks as the model explores), so a headless run's plan is
        visible in the stream, not only in the TUI pane. A single root (no
        decomposition) is not a plan worth a block."""
        nodes = event.get("nodes", {}) or {}
        if not isinstance(nodes, dict) or len(nodes) <= 1 or len(nodes) <= self._plan_count:
            return
        self._plan_count = len(nodes)
        cursor = event.get("cursor")
        lines = tree_lines_from_event_nodes(nodes, cursor if isinstance(cursor, str) else None)
        if not lines:
            return
        self._end_block()
        self._line("\n" + self._c("bold", f"plan ({len(nodes)} tasks)") + "\n")
        for line in lines:
            self._line(self._c("dim", "  " + line) + "\n")

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

    def _clear_status(self) -> None:
        """Erase the transient spinner line so real output prints cleanly. Caller
        holds the lock (or is the constructor before the thread starts)."""
        if self._status_active:
            self._out.write("\r\x1b[2K")  # carriage return + erase whole line
            self._status_active = False

    def _raw(self, text: str) -> None:
        # Low-level writer, used by streaming deltas AND internal block-closing;
        # it clears the spinner but does NOT bump _last_output_at (that tracks
        # real model output -- set by _stream / _line -- so closing a block from
        # the heartbeat can't reset the idle timer and suppress the spinner).
        # Streaming path: flush at most every _FLUSH_EVERY_S. A per-token flush on
        # a slow terminal (SSH, a busy emulator) backpressures the SSE read in the
        # same thread and can stall the stream; ~30ms is imperceptible and cuts
        # thousands of flushes to a few dozen a second.
        self._clear_status()
        self._out.write(text)
        now = time.monotonic()
        if now - self._last_flush >= _FLUSH_EVERY_S:
            self._out.flush()
            self._last_flush = now

    def _line(self, text: str) -> None:
        # Structural lines (tool call/result, commit, verdict) are discrete model
        # progress: show them at once and reset the idle timer.
        self._clear_status()
        self._last_output_at = time.monotonic()
        self._out.write(text)
        self._flush()

    def _heartbeat_loop(self) -> None:
        """Refresh a transient "⠋ working… Ns" line while a turn is in flight and
        output has gone silent (a stalled stream, or a between-turn pause), so the
        run never looks hung. Runs only on a real terminal."""
        while not self._stop.wait(_HEARTBEAT_TICK_S):
            with self._lock:
                idle = time.monotonic() - self._last_output_at
                if not self._active or idle < _STALL_AFTER_S:
                    self._clear_status()  # output flowing or turn done: no spinner
                    if self._status_active is False:
                        self._out.flush()
                    continue
                # Silent mid-turn: close any open prose block so the cursor sits on
                # a clean line, then draw/refresh the spinner in place.
                if self._phase is not None:
                    self._end_block()
                self._spin += 1
                glyph = _SPINNER[self._spin % len(_SPINNER)]
                hint = "  (Ctrl-C to steer or stop)" if idle >= 20 else ""
                body = f"{glyph} working… {int(idle)}s{hint}"
                self._out.write("\r\x1b[2K" + (self._c("dim", body) if self._color else body))
                self._out.flush()
                self._status_active = True

    def close(self) -> None:
        """Stop the heartbeat thread and clear any spinner line. Safe to call more
        than once; the daemon thread also dies with the process."""
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)
        with self._lock:
            self._clear_status()
            self._out.flush()

    def _flush(self) -> None:
        self._out.flush()
        self._last_flush = time.monotonic()
