# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of one session's `logs.jsonl`.

The dashboard's log pane is a sliding window that snaps to the bottom, so a fast
run plays through with no way back. This screen renders the whole file with the
dashboard's one-line formatter, follows a live file, and lets the operator
scroll, select and copy. Streaming deltas and the loop-side mirrors are skipped,
as in the dashboard's tail. The body is a `Static` in a `VerticalScroll`, not a
`RichLog`: a `RichLog` renders as line strips, which text selection cannot
extract.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Static

from agent6.ui.tui.menubar import (
    SCROLL_ITEMS,
    Menu,
    MenuBar,
    MenuItem,
    menu_bindings,
)
from agent6.ui.tui.screen_chrome import ScreenChrome
from agent6.viewmodel.log_line import format_log_line
from agent6.viewmodel.state import LOG_NOISE_EVENTS, STREAM_DELTA_EVENTS
from agent6.viewmodel.tail import LogTail


class LogScreen(ScreenChrome, Screen[None]):
    """The scrollable, read-only, selectable log of one session, live or finished."""

    CSS = """
    LogScreen { background: $surface; }
    #logview-scroll { height: 1fr; }
    #logview-body { height: auto; padding: 0 1; pointer: text; }  /* selectable: I-beam */
    """

    HELP_TITLE: ClassVar = "agent6 — log"
    MENUS: ClassVar = (
        Menu("File", (MenuItem("Back", "close"),)),
        Menu("View", (*SCROLL_ITEMS, MenuItem("Reload", "reload"))),
        Menu(
            "Help",
            (MenuItem("Keys & actions", "help"), MenuItem("Command palette", "command_palette")),
        ),
    )
    FOOTER: ClassVar = (("close", "Back"), ("reload", "Reload"), ("help", "Help"))
    BINDINGS: ClassVar = menu_bindings("event log", MENUS, footer=FOOTER)

    def __init__(self, logs_path: Path, *, title: Callable[[], str]) -> None:
        """Bind the screen to a log file and the callable naming its session."""
        super().__init__()
        self._logs_path = logs_path
        self._title = title
        self._tail = LogTail(logs_path)
        self._text = Text()

    def compose(self) -> ComposeResult:
        """Lay out the screen.

        Yields:
            The menu bar, the scrollable body and the footer.
        """
        yield MenuBar(self.MENUS)
        with VerticalScroll(id="logview-scroll"):
            yield Static(id="logview-body")
        yield Footer()

    def on_mount(self) -> None:
        """Load the file and keep following it; a resume appends to the same file."""
        self.app.sub_title = self._title()
        self._reload()
        self.set_interval(0.5, self._poll)

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#logview-scroll", VerticalScroll)

    def _append(self, events: list[dict[str, object]]) -> bool:
        added = False
        for event in events:
            if event.get("type") in STREAM_DELTA_EVENTS or event.get("type") in LOG_NOISE_EVENTS:
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
        if at_bottom:  # hold the position when the operator scrolled up
            scroll.scroll_end(animate=False)

    def action_reload(self) -> None:
        """Re-read the file from the start."""
        self._reload()

    def action_page_up(self) -> None:
        """Scroll a page up; instant, since animation reads as lag."""
        self._scroll().scroll_page_up(animate=False)

    def action_page_down(self) -> None:
        """Scroll a page down."""
        self._scroll().scroll_page_down(animate=False)

    def action_scroll_top(self) -> None:
        """Scroll to the first line."""
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll to the last line."""
        self._scroll().scroll_end(animate=False)

    def action_close(self) -> None:
        """Return to the previous screen."""
        self.dismiss()
