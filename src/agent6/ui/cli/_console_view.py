# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a run's event stream to a terminal as a live conversation.

The CLI skin over `viewmodel.TranscriptFold`: this class streams the reasoning and
text deltas inline; the structural items (tool call and result, commit, verdict)
come from the fold the TUI and web skins share. One `ConsoleView` serves `run`
(subscribed to the EventSink) and `attach` (fed by the log tailer).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Callable, Generator
from typing import Any, TextIO

from agent6.ui.cli import _task_tree, _terminal_guard
from agent6.viewmodel import events, format, listing, transcript, transcript_style

_ANSI = {
    "dim": "\033[2m",
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "italic": "\033[3m",
}

# Style name to ANSI escape; the TUI has the sibling Rich map over the same `item_lines()`.
_STYLE_ANSI: dict[transcript_style.StyleName, str] = {
    "thinking": _ANSI["dim"],
    "think-marker": _ANSI["blue"],
    "text": "",
    "call": _ANSI["bold"] + _ANSI["cyan"],
    "verify": _ANSI["bold"] + _ANSI["yellow"],
    "arg": _ANSI["dim"],
    "ok": _ANSI["green"],
    "fail": _ANSI["red"],
    "detail": _ANSI["dim"],
    "more": _ANSI["dim"] + _ANSI["italic"],
    "tail": _ANSI["dim"],
    "commit": _ANSI["magenta"],
    "marker": _ANSI["dim"] + _ANSI["italic"],
    "done-ok": _ANSI["bold"] + _ANSI["green"],
    "done-fail": _ANSI["bold"] + _ANSI["yellow"],
    "done-neutral": _ANSI["bold"],
    "body": "",
    "done-detail": _ANSI["dim"],
    "operator": _ANSI["bold"] + _ANSI["green"],
}

_FLUSH_EVERY_S = 0.03  # coalesces streaming-delta flushes; see ConsoleView._raw
_HEARTBEAT_TICK_S = 0.5  # how often the spinner refreshes
_STALL_AFTER_S = 1.5  # the heartbeat shows once output has been silent this long
# Drawing the spinner mid-block splits a streamed word in two, and slow token cadence
# routinely pauses a few seconds; only a real stall is worth that cost.
_MID_BLOCK_STALL_S = 10.0


class ConsoleView:
    """Fold events to styled terminal lines, one event per `feed` call.

    Thread-safe, so it can subscribe to an EventSink several roles emit to.
    """

    def __init__(
        self,
        out: TextIO | None = None,
        *,
        color: bool | None = None,
        policy: Callable[[], str] | None = None,
    ) -> None:
        # The policy line, read when the task prints: the gate is pinned between then and now.
        self._policy = policy
        # Finished /btw answers, printed whole at the next turn boundary.
        self._btw: list[str] = []
        self._out = out if out is not None else sys.stderr
        self._color = self._out.isatty() if color is None else color
        self._fold = transcript.TranscriptFold()
        # Reentrant: the SIGINT steer handler emits an event while a delta write holds the lock.
        self._lock = threading.RLock()
        self._phase: str | None = None  # the open prose block: None, "thinking" or "text"
        self._text_streamed = False
        self._last_flush = 0.0
        self._plan_count = 0  # tasks shown in the last plan block; reprinted when it grows
        # The heartbeat thread draws a spinner during silence; only on a real terminal.
        self._last_output_at = time.monotonic()
        # The fed event's own ts anchors the idle timer, so a replay measures from the run's time.
        self._event_ep: float | None = None
        self._active = False  # between session.start and session.end
        self._status_active = False  # a transient spinner line is on screen
        self._paused = False  # an interactive /dev/tty prompt owns the line
        self._spin = 0
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        if self._out.isatty():
            self._heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat.start()

    def __call__(self, event: dict[str, Any]) -> None:
        """Feed one event."""
        self.feed(event)

    def queue_btw(self, block: str) -> None:
        """Queue a finished /btw answer, printed whole at the next turn boundary."""
        with self._lock:
            self._btw.append(block)

    def settle_dead(self, reason: str) -> None:
        """Render the open tool calls as never returned, since no session.end will settle them."""
        with self._lock:
            for item in self._fold.settle_open_calls(reason):
                self._render(item)

    def _drain_btw(self) -> None:
        """Print the queued btw answers; the caller holds the lock and closed the open block."""
        for block in self._btw:
            self._line(block)
        self._btw.clear()

    def _bump_idle(self) -> None:
        """Reset the idle timer to the fed event's own age, or to now for a ts-less event."""
        age = 0.0 if self._event_ep is None else max(0.0, time.time() - self._event_ep)
        self._last_output_at = time.monotonic() - age

    def feed(self, event: dict[str, Any]) -> None:  # noqa: PLR0911, PLR0912  # one branch per event type
        """Render one event."""
        etype = event.get("type", "")
        with self._lock:
            # Anchored per event: a replay can end on an event that renders nothing yet.
            self._event_ep = events.event_epoch(event.get("ts"))
            self._bump_idle()
            # Active through a jailed command too, which runs between role.result and role.call.
            if etype in ("session.start", "role.call", "tool.call"):
                self._active = True
            elif etype in ("session.end", "session.steer_requested"):
                self._active = False
                # The run ending is a clean break for a btw that landed after the last turn.
                self._end_block()
                self._drain_btw()
            if etype in ("role.thinking_delta", "role.text_delta"):
                self._stream(str(event.get("text", "")), thinking=etype == "role.thinking_delta")
                if etype == "role.text_delta" and self._phase == "text":
                    self._text_streamed = True
                return
            if etype == "role.call":
                self._end_block()
                self._drain_btw()
                self._text_streamed = False
                self._fold.feed(event)
                return
            if etype == "role.result":
                self._end_block()
                self._drain_btw()
                items = self._fold.feed(event)
                if not self._text_streamed:
                    for item in items:
                        self._render(item)
                return
            if etype == "session.steer_requested":
                # A pause message is about to print; an open dim block would bleed into it.
                self._end_block()
                return
            if etype == "session.start":
                # Clipped: a `--from` task carries the whole plan.
                task = listing.task_snippet(str(event.get("user_task", "")), max_chars=200)
                self._line(self._c("bold", self._c("cyan", transcript.DONE) + " " + task) + "\n")
                policy = self._policy() if self._policy is not None else ""
                if policy:
                    self._line(self._c("dim", f"  {policy}") + "\n")
                # The fold reads the start for the receipt; its operator item is this headline.
                self._fold.feed(event)
                return
            if etype == "btw.answered":
                # Queued: printed now, it would break up a streaming turn.
                self._btw.append(str(event.get("block", "")))
                return
            if etype == "graph.update":
                self._render_plan(event)
                return
            if etype == "loop.provider.retry":
                # A retry resets the idle clock; unsaid, a run wedged on failures reads as fresh.
                self._end_block()
                attempt = event.get("attempt")
                self._line(
                    self._c("dim", f"  retrying after a provider error (attempt {attempt}): ")
                    + self._c("dim", str(event.get("error", "")))
                    + "\n"
                )
                return
            for item in self._fold.feed(event):
                self._end_block()
                self._render(item)

    def _stream(self, piece: str, *, thinking: bool) -> None:
        """Write a delta inline, opening the prose block it belongs to."""
        # A control sequence split across deltas cannot reassemble: the tail prints inert.
        piece = transcript.scrub_terminal_controls(piece)
        want = "thinking" if thinking else "text"
        if self._phase != want:
            if not piece.strip():
                return  # never open a block on whitespace
            self._end_block()
            self._phase = want
            self._raw("  " + (self._dim() + transcript.THINK + " " if thinking else ""))
            piece = piece.lstrip()
        self._bump_idle()
        # Wrapped lines stay under the block's indent.
        self._raw(piece.replace("\n", "\n    " if thinking else "\n  "))

    def _end_block(self) -> None:
        """Close the open prose block, if any."""
        if self._phase == "thinking":
            self._raw(self._reset())
        if self._phase is not None:
            self._raw("\n")
            self._flush()
        self._phase = None

    def _render_plan(self, event: dict[str, Any]) -> None:
        """Print the task tree when it first appears and each time it grows.

        A single root is not a plan worth a block.
        """
        nodes = event.get("nodes", {}) or {}
        if not isinstance(nodes, dict) or len(nodes) <= 1 or len(nodes) <= self._plan_count:
            return
        self._plan_count = len(nodes)
        cursor = event.get("cursor")
        lines = _task_tree.task_tree_lines(nodes, cursor if isinstance(cursor, str) else None)
        if not lines:
            return
        self._end_block()
        self._line("\n" + self._c("bold", f"plan ({len(nodes)} tasks)") + "\n")
        for line in lines:
            self._line(self._c("dim", "  " + line) + "\n")

    def _render(self, item: transcript.TranscriptItem) -> None:
        """Print a fold item's lines in ANSI behind a two-space gutter.

        An in-flight tool call prints nothing; the settled item prints the call whole.
        """
        if item.kind == "tool" and item.ok is None:
            return
        for line in transcript_style.item_lines(item, detail="collapsed"):
            rendered = "".join(
                f"{_STYLE_ANSI[style]}{text}{_ANSI['reset']}"
                if self._color and _STYLE_ANSI[style]
                else text
                for text, style in line
            )
            self._line(("  " + rendered if rendered else "") + "\n")

    def _c(self, name: str, text: str) -> str:
        """Return the text in the named colour, when colour is on."""
        return f"{_ANSI[name]}{text}{_ANSI['reset']}" if self._color else text

    def _dim(self) -> str:
        """Return the dim escape, when colour is on."""
        return _ANSI["dim"] if self._color else ""

    def _reset(self) -> str:
        """Return the reset escape, when colour is on."""
        return _ANSI["reset"] if self._color else ""

    def _clear_status(self) -> None:
        """Erase the transient spinner line; the caller holds the lock."""
        if self._status_active:
            _terminal_guard.raw_stream(self._out).write("\r\x1b[2K")
            self._status_active = False

    def _raw(self, text: str) -> None:
        """Write without bumping the idle timer, flushing at most every `_FLUSH_EVERY_S`.

        A per-token flush on a slow terminal backpressures the SSE read in the same
        thread and can stall the stream.
        """
        self._clear_status()
        self._out.write(text)
        now = time.monotonic()
        if now - self._last_flush >= _FLUSH_EVERY_S:
            self._out.flush()
            self._last_flush = now

    def _line(self, text: str) -> None:
        """Write a structural line at once and reset the idle timer."""
        self._clear_status()
        self._bump_idle()
        self._out.write(text)
        self._flush()

    def _heartbeat_loop(self) -> None:
        """Refresh a transient "working… Ns" line while a turn is in flight and silent."""
        while not self._stop.wait(_HEARTBEAT_TICK_S):
            with self._lock:
                if self._paused:
                    continue
                idle = time.monotonic() - self._last_output_at
                stall_after = _MID_BLOCK_STALL_S if self._phase is not None else _STALL_AFTER_S
                if not self._active or idle < stall_after:
                    # No spinner; the flush pushes out the partial line `_raw` left buffered.
                    self._clear_status()
                    self._out.flush()
                    continue
                # Close any open prose block so the spinner draws on a clean line.
                if self._phase is not None:
                    self._end_block()
                self._spin += 1
                glyph = format.spinner_frame(self._spin)
                hint = "  (Ctrl-C to steer or stop)" if idle >= 20 else ""
                body = f"{glyph} working… {int(idle)}s{hint}"
                _terminal_guard.raw_stream(self._out).write("\r\x1b[2K")
                self._out.write(self._c("dim", body) if self._color else body)
                self._out.flush()
                self._status_active = True

    def notice(self, msg: str) -> None:
        """Print a harness notice on the view's stream, under the lock, after the spinner."""
        with self._lock:
            self._clear_status()
            self._out.write(msg if msg.endswith("\n") else msg + "\n")
            self._out.flush()

    @contextlib.contextmanager
    def pause(self) -> Generator[None]:
        """Suspend the spinner for the block, so an interactive prompt owns the terminal.

        The lock is released across the yield, so the blocking prompt cannot stall `feed`.

        Yields:
            Nothing; the spinner is suspended for the block.
        """
        with self._lock:
            self._paused = True
            self._clear_status()
            self._out.flush()
        try:
            yield
        finally:
            with self._lock:
                self._paused = False

    def close(self) -> None:
        """Stop the heartbeat thread and clear any spinner line; idempotent."""
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)
        with self._lock:
            self._clear_status()
            self._out.flush()

    def _flush(self) -> None:
        """Flush and record the time."""
        self._out.flush()
        self._last_flush = time.monotonic()
