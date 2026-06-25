# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of one run's logs.jsonl (current or past).

The dashboard's live log pane is a small sliding window that snaps to the
bottom on every new line, so a fast run "plays through" with no way to scroll
back, and finished runs in the hub had no log view at all. ``LogScreen`` reads
a run's whole logs.jsonl, renders each event with the SAME one-line formatter
the dashboard uses (so the two read identically), and lets the operator scroll
freely. It is read-only; ``r`` re-reads the file (a live run keeps appending).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from agent6.ui.state import format_log_line
from agent6.ui.tail import tail_events


class LogScreen(Screen[None]):
    """Scrollable, read-only log of a single run (live or finished)."""

    CSS = """
    LogScreen { background: $surface; }
    #logview-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #logview-body { height: 1fr; border: none; padding: 0 1; }
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("r", "reload", "Reload"),
        Binding("g", "scroll_top", "Top"),
        Binding("G", "scroll_bottom", "End"),
    ]

    def __init__(self, logs_path: Path, *, title: str) -> None:
        super().__init__()
        self._logs_path = logs_path
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="logview-title")
        # markup/highlight off: log lines carry raw tool args (brackets) that Rich
        # would try to parse. auto_scroll off so reading doesn't fight live writes.
        yield RichLog(
            id="logview-body", highlight=False, markup=False, wrap=False, auto_scroll=False
        )
        yield Footer()

    def on_mount(self) -> None:
        self._load()

    def _load(self) -> None:
        log = self.query_one("#logview-body", RichLog)
        log.clear()
        count = 0
        for event in tail_events(self._logs_path, follow=False):
            log.write(format_log_line(event))
            count += 1
        if count == 0:
            log.write("(no events yet — press r to reload)")
        log.scroll_end(animate=False)
        log.focus()

    def action_reload(self) -> None:
        self._load()

    def action_scroll_top(self) -> None:
        self.query_one("#logview-body", RichLog).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#logview-body", RichLog).scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
