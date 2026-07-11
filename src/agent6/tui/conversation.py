# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of a run's LLM conversation (current or past).

The companion to ``LogScreen``: it folds the same ``logs.jsonl`` stream through
the shared ``TranscriptFold`` into the conversation -- assistant reasoning and
text, every tool call with its result, commits, and the verdict -- with the same
glyphs the CLI stream uses.

Completed turns scroll in the main pane; a docked live pane at the bottom streams
the turn IN PROGRESS (a reasoning model can think for 30-60s before it produces a
tool call, so without this the view looks frozen). Follows live: new turns append
and, unless the operator scrolled up to read, the pane sticks to the bottom.
``t`` toggles thinking, ``r`` re-reads, ``g``/``G`` jump to top/bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
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

_LIVE_TAIL = 1600  # chars of the in-progress turn kept in the live pane


def _tail(text: str, n: int) -> str:
    return text if len(text) <= n else "…" + text[-n:]


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
    #conv-live {
        height: auto; max-height: 12; padding: 0 1;
        border-top: solid $border; background: $surface;
    }
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
        self._live_think: list[str] = []
        self._live_text: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="conv-title")
        with Vertical():
            # wrap: prose reads wrapped; markup off (tool args carry brackets).
            # auto_scroll off: the poll manages sticky-bottom so a scroll-up holds.
            yield RichLog(
                id="conv-body", highlight=False, markup=False, wrap=True, auto_scroll=False
            )
            yield Static("", id="conv-live")
        yield Footer()

    def on_mount(self) -> None:
        self._reload()
        self.set_interval(0.3, self._poll)

    def _body(self) -> RichLog:
        return self.query_one("#conv-body", RichLog)

    def _write(self, item: TranscriptItem, log: RichLog) -> bool:
        wrote = False
        for line in _item_renderables(item, show_thinking=self._show_thinking):
            log.write(line, scroll_end=False)
            wrote = True
        return wrote

    def _track_live(self, event: dict[str, object]) -> None:
        etype = event.get("type")
        if etype in ("role.call", "role.result"):
            self._live_think.clear()
            self._live_text.clear()
        elif etype == "role.thinking_delta":
            self._live_think.append(str(event.get("text", "")))
        elif etype == "role.text_delta":
            self._live_text.append(str(event.get("text", "")))

    def _render_live(self) -> None:
        live = self.query_one("#conv-live", Static)
        think = "".join(self._live_think).strip() if self._show_thinking else ""
        text = "".join(self._live_text).strip()
        if not think and not text:
            live.display = False
            return
        body = Text()
        if think:
            body.append(f"{THINK} thinking… ", style="bold cyan")
            body.append(_tail(think, _LIVE_TAIL), style="dim italic")
        if text:
            if think:
                body.append("\n\n")
            body.append(_tail(text, _LIVE_TAIL))
        live.display = True
        live.update(body)

    def _reload(self) -> None:
        """Re-read the whole log from scratch (mount, `r`, thinking toggle)."""
        log = self._body()
        log.clear()
        self._tail = LogTail(self._logs_path)
        self._fold = TranscriptFold()
        self._live_think.clear()
        self._live_text.clear()
        wrote = False
        for event in self._tail.read():
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._write(item, log) or wrote
        if not wrote:
            log.write(
                Text("(no conversation yet — it appears as the run streams)", style="dim italic")
            )
        self._render_live()
        log.scroll_end(animate=False)
        log.focus()

    def _poll(self) -> None:
        """Append newly-completed turns (sticking to the bottom unless scrolled
        up) and refresh the live in-progress pane."""
        log = self._body()
        new_events = self._tail.read()
        if not new_events:
            return
        at_bottom = log.is_vertical_scroll_end
        wrote = False
        for event in new_events:
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._write(item, log) or wrote
        if wrote and at_bottom:
            log.scroll_end(animate=False)
        self._render_live()

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
