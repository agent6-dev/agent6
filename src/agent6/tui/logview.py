# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of one run's logs.jsonl (current or past).

The dashboard's live log pane is a small sliding window that snaps to the
bottom on every new line, so a fast run "plays through" with no way to scroll
back, and finished runs in the hub had no log view at all. ``LogScreen`` reads
a run's whole logs.jsonl, renders each event with the SAME one-line formatter
the dashboard uses (so the two read identically), and lets the operator scroll
freely. It is read-only; ``r`` re-reads the file (a live run keeps appending).

Deliberately lighter chrome than HomeScreen/ConfigScreen: a read-only pager
needs no File/View menus, so it skips the MenuBar/MENUS/palette convention and
keeps just a dock-top title and a terse binding set (Esc/q back, r refresh,
g/G scroll) -- the same minimal shape as its sibling ConversationScreen.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from agent6.viewmodel.state import format_log_line
from agent6.viewmodel.tail import LogTail


class LogScreen(Screen[None]):
    """Scrollable, read-only log of a single run (live or finished)."""

    CSS = """
    LogScreen { background: $surface; }
    #logview-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #logview-body { height: 1fr; border: none; padding: 0 1; }
    """

    BINDINGS: ClassVar = [
        # q and Esc both close the pager (back out one level); shown as one "Esc/q
        # Back" footer entry. Only the root hub quits on q -- every other screen
        # backs out; Ctrl+Q is the app-wide hard quit.
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        Binding("r", "reload", "Refresh"),
        Binding("g", "scroll_top", "Top"),
        Binding("G", "scroll_bottom", "End"),
    ]

    def __init__(self, logs_path: Path, *, title: str) -> None:
        super().__init__()
        self._logs_path = logs_path
        self._title = title
        self._tail = LogTail(logs_path)

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="logview-title")
        # markup/highlight off: log lines carry raw tool args (brackets) that Rich
        # would try to parse. auto_scroll off so reading doesn't fight live writes.
        yield RichLog(
            id="logview-body", highlight=False, markup=False, wrap=False, auto_scroll=False
        )
        yield Footer()

    def on_mount(self) -> None:
        self._reload()
        # Follow live: a resume appends to the same file, so keep reading.
        self.set_interval(0.5, self._poll)

    def _reload(self) -> None:
        log = self.query_one("#logview-body", RichLog)
        log.clear()
        self._tail = LogTail(self._logs_path)
        count = 0
        for event in self._tail.read():
            log.write(format_log_line(event))
            count += 1
        if count == 0:
            log.write("(no events yet)")
        log.scroll_end(animate=False)
        log.focus()

    def _poll(self) -> None:
        log = self.query_one("#logview-body", RichLog)
        new_events = self._tail.read()
        if not new_events:
            return
        # Sticky bottom: only snap to the newest line if the operator was there,
        # so scrolling up to read holds position.
        at_bottom = (log.max_scroll_y - log.scroll_offset.y) <= 1
        for event in new_events:
            log.write(format_log_line(event), scroll_end=False)
        if at_bottom:
            log.scroll_end(animate=False)

    def action_reload(self) -> None:
        self._reload()

    def action_scroll_top(self) -> None:
        self.query_one("#logview-body", RichLog).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#logview-body", RichLog).scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
