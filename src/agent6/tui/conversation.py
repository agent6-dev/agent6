# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of a run's LLM conversation (current or past).

The companion to ``LogScreen``: where that shows the terse ``logs.jsonl`` event
stream, this folds the SAME stream through the shared ``TranscriptFold`` into the
conversation -- assistant reasoning and text, every tool call with its result,
commits, and the verdict -- rendered with the same glyphs the CLI stream uses.

It follows live: a poll appends new turns as they arrive and, unless the operator
has scrolled up to read, sticks to the bottom. So resuming a run (which appends
to the same log) keeps updating here instead of freezing. ``t`` toggles thinking,
``r`` re-reads from scratch, ``g``/``G`` jump to top/bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from agent6.viewmodel.tail import LogTail
from agent6.viewmodel.transcript import (
    CALL,
    COMMIT,
    DONE,
    RESULT,
    THINK,
    TranscriptFold,
    TranscriptItem,
)


def _item_renderables(item: TranscriptItem, *, show_thinking: bool) -> list[Text]:
    """Render one folded conversation item as styled Rich lines (the TUI skin)."""
    out: list[Text] = []
    if item.kind == "thinking":
        if show_thinking:
            out = [Text(f"{THINK} {item.body}", style="dim italic"), Text("")]
    elif item.kind == "text":
        out = [Text(item.body), Text("")]
    elif item.kind == "tool":
        head = Text(f"{CALL} {item.name}", style="bold cyan")
        if item.arg:
            head.append(f"  {item.arg}", style="dim")
        result = Text(f"  {RESULT} ", style="green" if item.ok else "red")
        result.append(item.detail, style="dim")
        out = [head, result]
        if item.tail:
            out.append(Text(f"     {' '.join(item.tail.split())[:160]}", style="dim red"))
        out.append(Text(""))
    elif item.kind == "commit":
        out = [Text(f"{COMMIT} commit  {item.detail}", style="magenta"), Text("")]
    elif item.kind == "marker":
        out = [Text(f"── {item.body} ──", style="dim italic"), Text("")]
    elif item.kind == "done":
        badge = (
            Text(f"{DONE} done", style="bold green")
            if item.ok
            else Text(f"{DONE} {item.name or 'stopped'}", style="bold yellow")
        )
        if item.body:
            badge.append(f"  {item.body}", style="default")
        out = [Text(""), badge, Text(item.detail, style="dim"), Text("")]
    return out


class ConversationScreen(Screen[None]):
    """Scrollable, live-following LLM conversation for a single run."""

    CSS = """
    ConversationScreen { background: $surface; }
    #conv-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #conv-body { height: 1fr; border: none; padding: 0 1; }
    """

    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        Binding("r", "reload", "Refresh"),
        Binding("t", "toggle_thinking", "Thinking"),
        Binding("g", "scroll_top", "Top"),
        Binding("G", "scroll_bottom", "End"),
    ]

    def __init__(self, logs_path: Path, *, title: str) -> None:
        super().__init__()
        self._logs_path = logs_path
        self._title = title
        self._show_thinking = True
        self._tail = LogTail(logs_path)
        self._fold = TranscriptFold()

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="conv-title")
        # wrap: prose reads better wrapped; markup off (tool args carry brackets).
        # auto_scroll off: the poll manages sticky-bottom so a scroll-up holds.
        yield RichLog(id="conv-body", highlight=False, markup=False, wrap=True, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        self._reload()
        self.set_interval(0.5, self._poll)

    def _body(self) -> RichLog:
        return self.query_one("#conv-body", RichLog)

    def _write(self, item: TranscriptItem, log: RichLog) -> bool:
        wrote = False
        for line in _item_renderables(item, show_thinking=self._show_thinking):
            log.write(line, scroll_end=False)
            wrote = True
        return wrote

    def _reload(self) -> None:
        """Re-read the whole log from scratch (mount, `r`, thinking toggle)."""
        log = self._body()
        log.clear()
        self._tail = LogTail(self._logs_path)
        self._fold = TranscriptFold()
        wrote = False
        for event in self._tail.read():
            for item in self._fold.feed(event):
                wrote = self._write(item, log) or wrote
        if not wrote:
            log.write(
                Text("(no conversation yet — it appears as the run streams)", style="dim italic")
            )
        log.scroll_end(animate=False)
        log.focus()

    def _poll(self) -> None:
        """Append turns from any newly-written events, sticking to the bottom
        unless the operator scrolled up to read."""
        log = self._body()
        new_events = self._tail.read()
        if not new_events:
            return
        at_bottom = (log.max_scroll_y - log.scroll_offset.y) <= 1
        wrote = False
        for event in new_events:
            for item in self._fold.feed(event):
                wrote = self._write(item, log) or wrote
        if wrote and at_bottom:
            log.scroll_end(animate=False)

    def action_reload(self) -> None:
        self._reload()

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        self._reload()

    def action_scroll_top(self) -> None:
        self._body().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._body().scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
