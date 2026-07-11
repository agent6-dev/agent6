# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of one run's logs.jsonl (current or past).

The dashboard's live log pane is a small sliding window that snaps to the
bottom on every new line, so a fast run "plays through" with no way to scroll
back, and finished runs in the hub had no log view at all. ``LogScreen`` reads
a run's whole logs.jsonl, renders each STRUCTURAL event with the SAME one-line
formatter the dashboard uses (so the two read identically), and lets the
operator scroll -- and select/copy -- freely. It is read-only; ``r`` re-reads
the file (a live run keeps appending).

Two deliberate choices:
- Ephemeral streaming deltas are skipped (see STREAM_DELTA_EVENTS): a reasoning
  model emits thousands of contentless `role.thinking_delta` events, which are
  noise in an audit log (the reasoning itself is in the conversation view).
- The body is a `Static` inside a `VerticalScroll`, not a `RichLog`. A `RichLog`
  renders as line Strips, which the framework's text selection can't extract, so
  its text is not copyable; a `Static` renders as `Content` and is selectable.

Deliberately lighter chrome than HomeScreen/ConfigScreen: a read-only pager needs
no File/View menus, so it skips the MenuBar/MENUS/palette convention and keeps
just a dock-top title and a terse binding set (Esc/q back, r refresh, g/G scroll)
-- the same minimal shape as its sibling ConversationScreen.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static

from agent6.viewmodel.state import STREAM_DELTA_EVENTS, format_log_line
from agent6.viewmodel.tail import LogTail


class LogScreen(Screen[None]):
    """Scrollable, read-only, selectable log of a single run (live or finished)."""

    CSS = """
    LogScreen { background: $surface; }
    #logview-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #logview-scroll { height: 1fr; }
    #logview-body { height: auto; padding: 0 1; }
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
        self._text = Text()

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="logview-title")
        with VerticalScroll(id="logview-scroll"):
            yield Static(id="logview-body")  # renders as Content -> its text is selectable
        yield Footer()

    def on_mount(self) -> None:
        self._reload()
        # Follow live: a resume appends to the same file, so keep reading.
        self.set_interval(0.5, self._poll)

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#logview-scroll", VerticalScroll)

    def _append(self, events: list[dict[str, object]]) -> bool:
        added = False
        for event in events:
            if event.get("type") in STREAM_DELTA_EVENTS:
                continue
            self._text.append(format_log_line(event) + "\n")
            added = True
        return added

    def _reload(self) -> None:
        self._tail = LogTail(self._logs_path)
        self._text = Text()
        self._append(self._tail.read())
        shown = self._text if len(self._text) else Text("(no events yet)", style="dim italic")
        self.query_one("#logview-body", Static).update(shown)
        self._scroll().scroll_end(animate=False)
        self._scroll().focus()

    def _poll(self) -> None:
        scroll = self._scroll()
        at_bottom = scroll.is_vertical_scroll_end
        if not self._append(self._tail.read()):
            return
        self.query_one("#logview-body", Static).update(self._text)
        if at_bottom:  # sticky bottom: hold position if the operator scrolled up
            scroll.scroll_end(animate=False)

    def action_reload(self) -> None:
        self._reload()

    def action_scroll_top(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll().scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
