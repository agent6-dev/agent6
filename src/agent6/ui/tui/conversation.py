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
``t`` toggles thinking, ``r`` re-reads, ``g``/``G`` jump to top/bottom. ``c`` copies
the mouse selection (or the whole transcript) via the ``copy_method`` UI preference;
``s``/``p``/``w`` copy via the native terminal / pager / a file.

The scrollback is a ``Static`` in a ``VerticalScroll`` (not a ``RichLog``): a
``RichLog`` renders as line Strips, which the framework's text selection cannot
extract, so its text is not copyable; a ``Static`` renders as ``Content`` and is
selectable -- matching the live pane, which is already a ``Static``.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Footer, Static

from agent6.ui.tui import clipboard
from agent6.ui.tui.settings import get_copy_method
from agent6.ui.viewmodel.tail import LogTail
from agent6.ui.viewmodel.transcript import (
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


class _ChromeStatic(Static):
    """A Static that never joins a text selection, so dragging over the title or
    the live pane doesn't grab their text (or stall the auto-scroll) mid-select.
    Only the transcript body (`#conv-body`) is selectable/copyable."""

    ALLOW_SELECT = False


class ConversationScreen(Screen[None]):
    """Scrollable, live-following, selectable LLM conversation for a single run."""

    CSS = """
    ConversationScreen { background: $surface; }
    #conv-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #conv-scroll { height: 1fr; }
    #conv-body { height: auto; padding: 0 1; }
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
        Binding("c", "copy", "Copy"),
        Binding("s", "suspend_copy", "Copy via terminal", show=False),
        Binding("p", "pager", "Pager", show=False),
        Binding("w", "write_file", "Save to file", show=False),
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
        self._content = Text()  # accumulated completed-turn lines (selectable)
        self._live_think: list[str] = []
        self._live_text: list[str] = []

    def compose(self) -> ComposeResult:
        yield _ChromeStatic(self._title, id="conv-title")
        with Vertical():
            with VerticalScroll(id="conv-scroll"):
                yield Static(id="conv-body")  # renders as Content -> selectable
            yield _ChromeStatic("", id="conv-live")  # chrome: not part of a selection
        yield Footer()  # Footer is ALLOW_SELECT=False in textual already

    def on_mount(self) -> None:
        self._reload()
        self.set_interval(0.3, self._poll)

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#conv-scroll", VerticalScroll)

    def _append(self, item: TranscriptItem) -> bool:
        wrote = False
        for line in _item_renderables(item, show_thinking=self._show_thinking):
            self._content.append_text(line)
            self._content.append("\n")
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
        self._tail = LogTail(self._logs_path)
        self._fold = TranscriptFold()
        self._content = Text()
        self._live_think.clear()
        self._live_text.clear()
        wrote = False
        for event in self._tail.read():
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        empty = Text("(no conversation yet — it appears as the run streams)", style="dim italic")
        self.query_one("#conv-body", Static).update(self._content if wrote else empty)
        self._render_live()
        self._scroll().scroll_end(animate=False)
        self._scroll().focus()

    def _poll(self) -> None:
        """Append newly-completed turns (sticking to the bottom unless scrolled
        up) and refresh the live in-progress pane."""
        new_events = self._tail.read()
        if not new_events:
            return
        scroll = self._scroll()
        at_bottom = scroll.is_vertical_scroll_end
        wrote = False
        for event in new_events:
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        if wrote:
            self.query_one("#conv-body", Static).update(self._content)
            if at_bottom:
                scroll.scroll_end(animate=False)
        self._render_live()

    # -- copy ---------------------------------------------------------------
    def _emit(self, seq: str) -> None:
        """Write a raw terminal escape (an OSC 52 clipboard-set) through the driver."""
        driver = self.app._driver  # pyright: ignore[reportPrivateUsage]
        if driver is not None:
            driver.write(seq)

    def _selected_or_all(self) -> tuple[str, str]:
        """The current transcript-body selection if any, else the whole transcript."""
        body_selection = self._body_selection()
        if body_selection and body_selection.strip():
            return body_selection, "selection"
        return self._content.plain, "whole transcript"

    def _body_selection(self) -> str | None:
        """Selected text from the transcript BODY only, so a drag that strays over
        the footer or live pane never copies their text -- Textual's screen-wide
        get_selected_text() would otherwise include them (they are chrome)."""
        try:
            body = self.query_one("#conv-body", Static)
        except NoMatches:
            return None
        selection = self.selections.get(body)
        if selection is None:
            return None
        grabbed = body.get_selection(selection)
        return grabbed[0] if grabbed is not None else None

    def _copy_text(self, text: str, *, method: str) -> str:
        """Copy *text* using the resolved *method*; returns a short status."""
        return clipboard.emit_clipboard(text, clipboard.resolve_method(method), self._emit)

    def action_copy(self) -> None:
        text, what = self._selected_or_all()
        if not text:
            self.notify("nothing to copy yet")
            return
        try:
            status = self._copy_text(text, method=get_copy_method())
        except (OSError, subprocess.CalledProcessError) as exc:
            self.notify(f"copy failed: {exc}", severity="error")
            return
        self.notify(f"copied {what} — {status}")

    def action_write_file(self) -> None:
        path = clipboard.write_transcript_file(self._content.plain)
        self.notify(f"wrote transcript to {path}")

    def action_suspend_copy(self) -> None:
        """Drop to the native terminal, print the text (scroll + select + copy with
        the terminal, which always works), Enter to return."""
        text, what = self._selected_or_all()
        with self.app.suspend():
            print(f"\n===== COPY BELOW ({what}) — select + copy in your terminal =====\n")
            print(text)
            print("\n===== END — press Enter to return =====")
            with contextlib.suppress(EOFError):
                input()

    def action_pager(self) -> None:
        """Open the text in $PAGER (scroll + select + copy natively)."""
        text, _ = self._selected_or_all()
        pager = os.environ.get("PAGER") or "less"
        cmd = [pager, "-R"] if Path(pager).name.startswith("less") else [pager]
        with self.app.suspend():
            try:
                subprocess.run(cmd, input=text, text=True, check=False)
            except OSError as exc:
                print(f"pager {pager!r} failed: {exc}\nPress Enter to return.")
                with contextlib.suppress(EOFError):
                    input()

    def action_reload(self) -> None:
        self._reload()

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        self._reload()

    def action_scroll_top(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll().scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
