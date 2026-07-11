# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[cli] input = "bar"` always-on input bar.

A persistent prompt_toolkit line stays at the bottom of a live run while the
ConsoleView output scrolls above it (inline -- no alternate screen, so terminal
scrollback and copy/paste are preserved). The synchronous Workflow loop runs in a
background thread; a line typed in the bar steers the run through a thread-safe
holder that backs a `SteerState` (the same seam Ctrl-C uses in modal mode). Abort
in bar mode is cooperative: Ctrl-C or /stop requests stop and the loop ends at its
next boundary, but a wedged call can't be force-interrupted the way modal's
main-thread Ctrl-C can, since the loop runs off-thread. Bar mode is gated on a tty
+ prompt_toolkit before it engages (run.py), so a start failure here is a real
error and propagates rather than being swallowed.

Design: one owner of terminal output at a time. `patch_stdout()` makes the
ConsoleView's writes (routed through `_BarStream`, which delegates to the CURRENT
sys.stdout) land above the bar; the heartbeat spinner is paused for the duration
so it cannot fight the bar for the line."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout

from agent6.ui.cli._steer import SteerState, normalize_steer_choice

if TYPE_CHECKING:
    from agent6.ui.cli._console_view import ConsoleView


class _BarStream:
    """A stdout-delegating stream so ConsoleView writes route through whatever
    sys.stdout is at write time -- i.e. prompt_toolkit's patch_stdout proxy, so
    the live output lands ABOVE the input bar instead of on top of it."""

    def write(self, s: str) -> int:
        return sys.stdout.write(s)

    def flush(self) -> None:
        sys.stdout.flush()

    def isatty(self) -> bool:
        return sys.stdout.isatty()


class BarController:
    """Owns the steer holder + the bar loop for one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: str | None = None
        self._has = False  # a real steer (not an empty continue) is queued
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- the SteerState seam the Workflow polls (from the loop thread) ---

    def steer_state(self) -> SteerState:
        return SteerState(
            requested=self._requested,
            clear=self._clear,
            prompt=self._take,
            restore=lambda: None,
            abort_pending=self._abort_pending,
        )

    def _requested(self) -> bool:
        with self._lock:
            return self._has

    def _clear(self) -> None:
        with self._lock:
            self._has, self._pending = False, None

    def _take(self) -> str | None:
        with self._lock:
            value, self._has, self._pending = self._pending, False, None
            return value

    def _abort_pending(self) -> bool:
        with self._lock:
            return self._pending == "abort"

    def submit(self, line: str) -> None:
        # Same mapping as the modal steer prompt, also accepting the slash form
        # (/stop, /detach) shown in the toolbar: ""->continue (no-op), stop/q->abort,
        # detach/d->detach, else the typed instruction.
        choice = normalize_steer_choice(line.strip().lstrip("/"))
        if not choice:  # "" (empty enter) or None -> continue, nothing to queue
            return
        with self._lock:
            self._pending, self._has = choice, True

    # --- wiring ---

    def stream(self) -> TextIO:
        """The stream to build the ConsoleView with in bar mode."""
        return _BarStream()  # type: ignore[return-value]

    def prompt(self, fn: Callable[[], Any]) -> Any:
        """From the loop thread: run `fn` (a blocking terminal prompt -- an ask_user
        radio, a run_command approval) on the bar's main thread with the bar
        suspended, so it owns the terminal instead of fighting the bar. Returns fn's
        result (a direct call if the bar loop is not up yet)."""
        loop = self._loop
        if loop is None:
            return fn()

        async def _run() -> Any:
            return await run_in_terminal(fn, in_executor=True)

        return asyncio.run_coroutine_threadsafe(_run(), loop).result()

    def run(self, run_call: Callable[[], Any], console_view: ConsoleView) -> Any:
        """Run `run_call` (the Workflow.run) in a background thread while the bar
        owns the bottom line; return its result."""

        async def drive() -> Any:
            self._loop = asyncio.get_running_loop()
            session: PromptSession[str] = PromptSession()

            def toolbar() -> HTML:
                return HTML("  <b>● running</b>   type a steer + Enter · <b>/stop</b> ends ")

            async def bar_loop() -> None:
                while True:
                    try:
                        line = await session.prompt_async("steer > ", bottom_toolbar=toolbar)
                    except KeyboardInterrupt:
                        # Keep the bar up so it keeps owning the terminal (and every
                        # further Ctrl-C) until the run winds down; each press just
                        # re-requests stop.
                        self.submit("stop")
                        continue
                    except EOFError:
                        self.submit("stop")  # Ctrl-D ends the bar and stops the run
                        return
                    self.submit(line.strip())

            # raw=True: the ConsoleView writes its own vt100 ANSI (color, resets);
            # patch_stdout's default (raw=False) would replace every ESC byte with
            # '?', turning the whole live transcript into "?[36m" garbage.
            with patch_stdout(raw=True), console_view.pause():
                loop = asyncio.get_running_loop()
                run_future = loop.run_in_executor(None, run_call)  # the sync loop, off-thread
                bar_task = asyncio.ensure_future(bar_loop())
                try:
                    return await run_future
                finally:
                    # Cleanup only: a cancelled (or otherwise failed) bar teardown
                    # must not mask the run's own result or exception.
                    bar_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await bar_task

        return asyncio.run(drive())
