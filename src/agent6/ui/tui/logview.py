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

import pathlib
from collections.abc import Callable
from typing import ClassVar

from rich import text
from textual import app, containers, screen, widgets

from agent6.ui.tui import menubar, screen_chrome
from agent6.viewmodel import log_line, state, tail


class LogScreen(screen_chrome.ScreenChrome, screen.Screen[None]):
    """The scrollable, read-only, selectable log of one session, live or finished."""

    CSS = """
    LogScreen { background: $surface; }
    #logview-scroll { height: 1fr; }
    #logview-body { height: auto; padding: 0 1; pointer: text; }  /* selectable: I-beam */
    """

    HELP_TITLE: ClassVar = "agent6 — log"
    MENUS: ClassVar = (
        menubar.Menu("File", (menubar.MenuItem("Back", "close"),)),
        menubar.Menu("View", (*menubar.SCROLL_ITEMS, menubar.MenuItem("Reload", "reload"))),
        menubar.Menu(
            "Help",
            (
                menubar.MenuItem("Keys & actions", "help"),
                menubar.MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    FOOTER: ClassVar = (("close", "Back"), ("reload", "Reload"), ("help", "Help"))
    BINDINGS: ClassVar = menubar.menu_bindings("event log", MENUS, footer=FOOTER)

    def __init__(self, logs_path: pathlib.Path, *, title: Callable[[], str]) -> None:
        """Bind the screen to a log file and the callable naming its session."""
        super().__init__()
        self._logs_path = logs_path
        self._title = title
        self._tail = tail.LogTail(logs_path)
        self._text = text.Text()

    def compose(self) -> app.ComposeResult:
        """Lay out the screen.

        Yields:
            The menu bar, the scrollable body and the footer.
        """
        yield menubar.MenuBar(self.MENUS)
        with containers.VerticalScroll(id="logview-scroll"):
            yield widgets.Static(id="logview-body")
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Load the file and keep following it; a resume appends to the same file."""
        self.app.sub_title = self._title()
        self._reload()
        self.set_interval(0.5, self._poll)

    def _scroll(self) -> containers.VerticalScroll:
        return self.query_one("#logview-scroll", containers.VerticalScroll)

    def _append(self, events: list[dict[str, object]]) -> bool:
        added = False
        for event in events:
            if (
                event.get("type") in state.STREAM_DELTA_EVENTS
                or event.get("type") in state.LOG_NOISE_EVENTS
            ):
                continue
            self._text.append(log_line.format_log_line(event) + "\n")
            added = True
        return added

    def _reload(self) -> None:
        self._tail = tail.LogTail(self._logs_path)
        self._text = text.Text()
        self._append(self._tail.read())
        shown = self._text if len(self._text) else text.Text("(no events yet)", style="dim italic")
        self.query_one("#logview-body", widgets.Static).update(shown)
        self._scroll().scroll_end(animate=False)
        self._scroll().focus()

    def _poll(self) -> None:
        scroll = self._scroll()
        at_bottom = scroll.is_vertical_scroll_end
        if not self._append(self._tail.read()):
            return
        self.query_one("#logview-body", widgets.Static).update(self._text)
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
