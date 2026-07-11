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
A live run opens with the steer bar focused: type + Enter sends a steer, Ctrl-J
newlines. Ctrl+C copies the mouse selection (or the whole transcript) via the
``copy_method`` UI preference; PageUp/PageDown scroll, Ctrl+Home/End jump to
top/bottom, Esc returns to the dashboard. The menu bar (File/View/Help) and the
command palette (Ctrl+P) hold the rest -- the detail cycle, reload, and the
pager/terminal/file copies -- each showing its shortcut from the live bindings.

The scrollback is a ``Static`` in a ``VerticalScroll`` (not a ``RichLog``): a
``RichLog`` renders as line Strips, which the framework's text selection cannot
extract, so its text is not copyable; a ``Static`` renders as ``Content`` and is
selectable -- matching the live pane, which is already a ``Static``.
"""

from __future__ import annotations

import bisect
import contextlib
import inspect
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Static, TextArea

from agent6.ui.bridge.approval import clear_steer_answer, request_steer, write_steer_answer
from agent6.ui.tui import clipboard
from agent6.ui.tui.menubar import (
    HelpScreen,
    Menu,
    MenuBar,
    MenuItem,
    menu_bindings,
)
from agent6.ui.tui.settings import get_copy_method
from agent6.ui.viewmodel.tail import LogTail
from agent6.ui.viewmodel.transcript import (
    THINK,
    TranscriptFold,
    TranscriptItem,
)
from agent6.ui.viewmodel.transcript_style import DetailLevel, StyleName, item_lines

_LIVE_TAIL = 1600  # chars of the in-progress turn kept in the live pane

# The single detail shortcut cycles through these in order.
_DETAIL_CYCLE: dict[DetailLevel, DetailLevel] = {
    "hidden": "collapsed",
    "collapsed": "expanded",
    "expanded": "hidden",
}


def _tail(text: str, n: int) -> str:
    return text if len(text) <= n else "…" + text[-n:]


# Semantic style name -> Rich style. The CLI has the sibling ANSI map; both skins
# render item_lines(), so the structure and which element is coloured live in ONE
# place (transcript_style) and can't drift.
_STYLE_RICH: dict[StyleName, str] = {
    "thinking": "#6C7086",
    "text": "",
    "call": "bold cyan",
    "arg": "dim",
    "ok": "green",
    "fail": "red",
    "detail": "dim",
    "more": "dim italic",
    "tail": "dim",
    "commit": "magenta",
    "marker": "dim italic",
    "done-ok": "bold green",
    "done-fail": "bold yellow",
    "body": "",
    "done-detail": "dim",
}


def _item_renderables(item: TranscriptItem, *, detail: DetailLevel) -> list[Text]:
    """The TUI skin over the shared item_lines(): one Rich Text per line, mapping
    each span's semantic style, with a blank line after the item for spacing."""
    lines = item_lines(item, detail=detail)
    if not lines:
        return []
    out: list[Text] = []
    for line in lines:
        text = Text()
        for chunk, style in line:
            text.append(chunk, style=_STYLE_RICH[style] or None)
        out.append(text)
    out.append(Text(""))  # one blank line after the item
    return out


class _ChromeStatic(Static):
    """A Static that never joins a text selection, so dragging over the title or
    the live pane doesn't grab their text (or stall the auto-scroll) mid-select.
    Only the transcript body (`#conv-body`) is selectable/copyable."""

    ALLOW_SELECT = False


_INPUT_MAX_ROWS = 6  # the steer bar grows to this many rows, then scrolls internally


class SteerInput(TextArea):
    """The bottom steer bar: a TextArea that submits on Enter (Ctrl+J / Shift+Enter
    insert a newline instead) and grows with its content up to _INPUT_MAX_ROWS."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_mount(self) -> None:
        self.border_title = "steer the run"
        self.border_subtitle = "Enter sends · Ctrl-J newline"
        self._resize()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self._resize()

    def _resize(self) -> None:
        rows = min(max(self.document.line_count, 1), _INPUT_MAX_ROWS)
        self.styles.height = rows + 2  # + the rounded border


class _ConvCommands(Provider):
    """Command-palette entries for the conversation view's less-common actions, so
    they stay reachable while the steer bar has focus (which owns the letter keys)."""

    def _commands(self) -> list[tuple[str, Callable[[], None], str]]:
        conv = cast("ConversationScreen", self.screen)
        return [
            ("Cycle detail level", conv.action_cycle_detail, "none / collapsed / expanded"),
            ("Reload the log", conv.action_reload, "re-read from the start"),
            ("Copy via terminal", conv.action_suspend_copy, "drop to the shell to select + copy"),
            ("Copy via pager", conv.action_pager, "open the transcript in $PAGER"),
            ("Save transcript to a file", conv.action_write_file, "write to a temp file"),
        ]

    async def discover(self) -> Hits:
        for name, runnable, help_text in self._commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, runnable, help_text in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), runnable, help=help_text)


class ConversationScreen(Screen[None]):
    """Scrollable, live-following, selectable LLM conversation for a single run."""

    CSS = """
    ConversationScreen { background: $surface; }
    #conv-main { height: 1fr; }
    #conv-scroll { height: 1fr; }
    #conv-body { height: auto; padding: 0 1; }
    #conv-live {
        height: auto; max-height: 12; padding: 0 1;
        border-top: solid $border; background: $surface;
    }
    /* The steer bar: shown only while the run is live (see _sync_input). */
    #conv-input {
        display: none; height: auto; max-height: 8; margin: 0 1;
        border: round $primary; background: $surface;
    }
    #conv-input:focus { border: round $accent; }
    """

    MENUS: ClassVar = (
        Menu("File", (MenuItem("Back to dashboard", "close"),)),
        Menu(
            "View",
            (
                MenuItem("Detail: none / collapsed / expanded", "cycle_detail"),
                MenuItem("Scroll ↑ a page", "page_up"),
                MenuItem("Scroll ↓ a page", "page_down"),
                MenuItem("Scroll → top", "scroll_top"),
                MenuItem("Scroll → end", "scroll_bottom"),
                MenuItem("Reload the log", "reload"),
                MenuItem("Copy selection / all", "copy"),
                MenuItem("Copy via terminal", "suspend_copy"),
                MenuItem("Copy via pager", "pager"),
                MenuItem("Save transcript to file", "write_file"),
            ),
        ),
        Menu(
            "Help",
            (
                MenuItem("Keys & actions", "help"),
                MenuItem("Command palette", "command_palette"),
            ),
        ),
    )

    # The steer bar owns plain letters + Enter, so the transcript's nav actions are
    # priority bindings (fire before the bar). Everything else lives in the menu bar
    # (which shows the shortcuts from these bindings) and the command palette.
    BINDINGS: ClassVar = [
        Binding("escape", "close", "Back", key_display="Esc", priority=True),
        Binding("ctrl+c", "copy", "Copy", priority=True),
        Binding("pageup", "page_up", "Scroll up", priority=True, show=False),
        Binding("pagedown", "page_down", "Scroll down", priority=True, show=False),
        Binding("ctrl+home", "scroll_top", "Top", priority=True, show=False),
        Binding("ctrl+end", "scroll_bottom", "End", priority=True, show=False),
        *menu_bindings(MENUS),
    ]
    COMMANDS: ClassVar = {_ConvCommands}

    def __init__(self, logs_path: Path, *, title: str) -> None:
        super().__init__()
        self._logs_path = logs_path
        self._title = title
        self._detail: DetailLevel = "collapsed"  # one shortcut cycles none/collapsed/expanded
        self._tail = LogTail(logs_path)
        self._fold = TranscriptFold()
        self._content = Text()  # accumulated completed-turn lines (selectable)
        self._item_starts: list[int] = []  # logical start line of each rendered item (anchor)
        self._content_lines = 0  # total logical lines in _content
        self._live_think: list[str] = []
        self._live_text: list[str] = []
        self._live = False  # run.start seen and no run.end yet -> the steer bar shows
        self._prev_subtitle = ""  # app sub_title to restore when the view closes

    def compose(self) -> ComposeResult:
        yield MenuBar(self.MENUS)  # top row: menus + "agent6 — <run>", like every screen
        with Vertical(id="conv-main"):
            with VerticalScroll(id="conv-scroll"):
                yield Static(id="conv-body")  # renders as Content -> selectable
            yield _ChromeStatic("", id="conv-live")  # chrome: not part of a selection
        yield SteerInput(id="conv-input")  # steer bar (hidden unless the run is live)
        yield Footer()  # Footer is ALLOW_SELECT=False in textual already

    def on_mount(self) -> None:
        self._prev_subtitle = self.app.sub_title  # show the run in the menu bar's title
        self.app.sub_title = self._title
        self._reload()
        self.set_interval(0.3, self._poll)

    def on_unmount(self) -> None:
        self.app.sub_title = self._prev_subtitle

    def action_menu(self, mnemonic: str) -> None:
        self.query_one(MenuBar).open(mnemonic)

    async def on_menu_bar_selected(self, event: MenuBar.Selected) -> None:
        # Screen actions first, then app-level built-ins (command_palette), which are
        # coroutines -- await results. Mirrors the hub / config / machines screens.
        handler = getattr(self, f"action_{event.action}", None) or getattr(
            self.app, f"action_{event.action}", None
        )
        if handler is not None:
            result = handler()
            if inspect.isawaitable(result):
                await result

    def action_help(self) -> None:
        self.app.push_screen(
            HelpScreen(
                self.MENUS,
                self,
                title="agent6 — conversation",
                hints=(
                    "Steer bar: Enter sends the instruction",
                    "Ctrl-J or Shift+Enter inserts a newline",
                ),
            )
        )

    def _scroll(self) -> VerticalScroll:
        return self.query_one("#conv-scroll", VerticalScroll)

    def _append(self, item: TranscriptItem) -> bool:
        self._item_starts.append(self._content_lines)  # where this item begins (for the anchor)
        wrote = False
        for line in _item_renderables(item, detail=self._detail):
            self._content.append_text(line)
            self._content.append("\n")
            self._content_lines += 1
            wrote = True
        return wrote

    def _track_live(self, event: dict[str, object]) -> None:
        etype = event.get("type")
        if etype == "run.start":
            self._live = True
        elif etype == "run.end":
            self._live = False
        if etype in ("role.call", "role.result"):
            self._live_think.clear()
            self._live_text.clear()
        elif etype == "role.thinking_delta":
            self._live_think.append(str(event.get("text", "")))
        elif etype == "role.text_delta":
            self._live_text.append(str(event.get("text", "")))

    def _render_live(self) -> None:
        live = self.query_one("#conv-live", Static)
        think = "".join(self._live_think).strip()
        text = "".join(self._live_text).strip()
        if not think and not text:
            live.display = False
            return
        body = Text()
        if think:
            # Always show the live "thinking…" indicator (feedback that a turn is
            # working); stream the reasoning itself only when expanded (muted grey).
            body.append(f"{THINK} thinking… ", style="bold cyan")
            if self._detail == "expanded":
                body.append(_tail(think, _LIVE_TAIL), style="#6C7086")
        if text:
            if think:
                body.append("\n\n")
            body.append(_tail(text, _LIVE_TAIL))
        live.display = True
        live.update(body)

    def _reload(self) -> None:
        """Re-read the whole log from scratch (mount, reload, detail cycle)."""
        self._tail = LogTail(self._logs_path)
        self._fold = TranscriptFold()
        self._content = Text()
        self._item_starts = []
        self._content_lines = 0
        self._live_think.clear()
        self._live_text.clear()
        self._live = False
        wrote = False
        for event in self._tail.read():
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        empty = Text("(no conversation yet — it appears as the run streams)", style="dim italic")
        self.query_one("#conv-body", Static).update(self._content if wrote else empty)
        self._render_live()
        self._sync_input()
        self._scroll().scroll_end(animate=False)
        self._focus_default()

    def _at_bottom(self, scroll: VerticalScroll) -> bool:
        """Following the log: at the bottom within a small tolerance, so a one-line
        layout nudge (the live pane or steer bar resizing) keeps follow mode, while
        a deliberate scroll up of more than that drops it."""
        return scroll.max_scroll_y - scroll.scroll_y <= 2.0

    def _poll(self) -> None:
        """Append newly-completed turns (sticking to the bottom unless scrolled
        up) and refresh the live in-progress pane."""
        new_events = self._tail.read()
        if not new_events:
            return
        scroll = self._scroll()
        following = self._at_bottom(scroll)  # BEFORE this frame's layout changes
        wrote = False
        for event in new_events:
            self._track_live(event)
            for item in self._fold.feed(event):
                wrote = self._append(item) or wrote
        if wrote:
            self.query_one("#conv-body", Static).update(self._content)
        self._render_live()
        self._sync_input()
        # Re-pin AFTER the live pane / steer bar have (re)sized this frame: growing
        # them shrinks the scroll viewport and would otherwise nudge us off the exact
        # bottom, silently dropping follow mode even when nothing new was appended.
        if following:
            scroll.scroll_end(animate=False)

    def _sync_input(self) -> None:
        """Show the steer bar only while the run is live (a finished or historical
        run has nothing to steer)."""
        with contextlib.suppress(NoMatches):
            self.query_one("#conv-input", SteerInput).display = self._live

    def _focus_default(self) -> None:
        """A live run opens with the steer bar focused (type to steer immediately);
        a finished or historical one focuses the scrollback for keyboard nav."""
        if self._live:
            with contextlib.suppress(NoMatches):
                self.query_one("#conv-input", SteerInput).focus()
                return
        self._scroll().focus()

    def on_steer_input_submitted(self, message: SteerInput.Submitted) -> None:
        """A line typed into the steer bar: drop a steer request + the instruction
        over the file bridge (the same seam the dashboard's steer box uses). The run
        picks it up at its next safe boundary and injects it; it keeps running."""
        if not self._live:
            return
        run_dir = self._logs_path.parent
        clear_steer_answer(run_dir)  # discard any stale answer -> the run waits for this one
        request_steer(run_dir)
        write_steer_answer(run_dir, message.text)
        self.notify("steering the run…")

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

    def get_selected_text(self) -> str | None:
        """Copy the transcript body only. Textual's screen-wide gather -- used by
        the built-in Ctrl+C copy -- would otherwise include footer-key or live-pane
        text a drag strayed over; restrict every copy path to the body."""
        return self._body_selection()

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

    def action_cycle_detail(self) -> None:
        """Cycle the transcript's detail level (hidden -> collapsed -> expanded), keeping
        the block at the top of the viewport anchored across the re-render."""
        self._reload_keeping_place(lambda: setattr(self, "_detail", _DETAIL_CYCLE[self._detail]))

    def _item_visual_starts(self) -> list[int]:
        """The visual (wrapped) row where each rendered item begins, at the current
        body width. One pass over the content, so it is cheap enough for a
        user-initiated re-render even on a long transcript."""
        width = max(1, self.query_one("#conv-body", Static).content_size.width)
        starts: list[int] = []
        visual = 0
        nxt = 0
        for logical, line in enumerate(self._content.split("\n")):
            while nxt < len(self._item_starts) and self._item_starts[nxt] == logical:
                starts.append(visual)
                nxt += 1
            visual += max(1, -(-line.cell_len // width))  # ceil(cell_len / width)
        starts.extend([visual] * (len(self._item_starts) - nxt))
        return starts

    def _reload_keeping_place(self, flip: Callable[[], None]) -> None:
        """Apply *flip*, re-render, and restore the reading position: pinned to the
        bottom if we were following, else anchored to the block that was at the top of
        the viewport (kept at the same viewport offset across the re-render, so a block
        expanding above doesn't carry your place away)."""
        scroll = self._scroll()
        following = self._at_bottom(scroll)
        top = scroll.scroll_y
        old_visual = self._item_visual_starts()
        anchor = bisect.bisect_right(old_visual, top) - 1 if old_visual else -1
        offset = top - old_visual[anchor] if 0 <= anchor < len(old_visual) else 0.0
        flip()
        self._reload()  # rebuilds _content + _item_starts, ending scrolled to the bottom
        if following or not (0 <= anchor < len(self._item_starts)):
            return
        self._scroll().scroll_to(y=self._item_visual_starts()[anchor] + offset, animate=False)

    def action_scroll_top(self) -> None:
        self._scroll().scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self._scroll().scroll_end(animate=False)

    def action_page_up(self) -> None:
        self._scroll().scroll_page_up()

    def action_page_down(self) -> None:
        self._scroll().scroll_page_down()

    def action_close(self) -> None:
        self.dismiss()
