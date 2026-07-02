# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A full-screen, scrollable view of a run's LLM conversation (current or past).

The companion to ``LogScreen``: where that shows the terse ``logs.jsonl`` event
stream, this folds the run's lossless per-call transcripts (``transcripts/``)
into the actual conversation -- assistant text + thinking, and every tool call
with its full arguments and result. Read-only; ``t`` toggles thinking, ``r``
re-reads (a live run keeps appending a file per LLM call).

Deliberately lighter chrome than HomeScreen/ConfigScreen: a read-only pager
needs no File/View menus, so it skips the MenuBar/MENUS/palette convention and
keeps just a dock-top title and a terse binding set (Esc/q back, r refresh,
g/G scroll) -- the same minimal shape as its sibling LogScreen.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from agent6.transcript_render import Turn, fold_conversation, load_transcripts


def _clip(s: str, n: int = 4000) -> str:
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} chars)"


def _turn_renderables(tn: Turn, *, show_thinking: bool) -> list[Text]:
    """Render one folded Turn as styled Rich lines for the RichLog."""
    if tn.role == "marker":
        return [Text(f"── {tn.text} ──", style="dim italic"), Text("")]
    if tn.role == "tool":
        label = f" {tn.tool_name}" if tn.tool_name else ""
        return [Text(f"  ← {label.strip()}: ", style="green") + Text(_clip(tn.text)), Text("")]
    out: list[Text] = []
    role_style = {"assistant": "bold cyan", "user": "bold yellow", "system": "bold magenta"}.get(
        tn.role, "bold"
    )
    head = tn.role + (f"  (seq {tn.seq})" if tn.role == "assistant" else "")
    out.append(Text(head, style=role_style))
    if tn.thinking and show_thinking:
        out.append(Text(tn.thinking, style="dim italic"))
    if tn.text:
        out.append(Text(tn.text))
    for name, args in tn.tool_calls:
        out.append(Text("  → ", style="cyan") + Text(f"{name}({args})", style="dim"))
    out.append(Text(""))
    return out


class ConversationScreen(Screen[None]):
    """Scrollable, read-only LLM conversation for a single run (live or finished)."""

    CSS = """
    ConversationScreen { background: $surface; }
    #conv-title { dock: top; height: 1; padding: 0 1; background: $panel; text-style: bold; }
    #conv-body { height: 1fr; border: none; padding: 0 1; }
    """

    BINDINGS: ClassVar = [
        # q and Esc both close the pager (back out one level); shown as one "Esc/q
        # Back" footer entry. Only the root hub quits on q -- every other screen
        # backs out; Ctrl+Q is the app-wide hard quit.
        Binding("escape", "close", "Back", key_display="Esc/q"),
        Binding("q", "close", "Back", show=False),
        Binding("r", "reload", "Refresh"),
        Binding("t", "toggle_thinking", "Thinking"),
        Binding("g", "scroll_top", "Top"),
        Binding("G", "scroll_bottom", "End"),
    ]

    def __init__(self, transcripts_dir: Path, *, title: str) -> None:
        super().__init__()
        self._dir = transcripts_dir
        self._title = title
        self._show_thinking = True

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="conv-title")
        # wrap=True: prose reads better wrapped; markup off (tool args have brackets).
        yield RichLog(id="conv-body", highlight=False, markup=False, wrap=True, auto_scroll=False)
        yield Footer()

    def on_mount(self) -> None:
        self._load()

    def _load(self) -> None:
        log = self.query_one("#conv-body", RichLog)
        log.clear()
        turns = fold_conversation(load_transcripts(self._dir))
        if not turns:
            log.write("(no transcripts yet — press r to refresh)")
            log.focus()
            return
        for tn in turns:
            for line in _turn_renderables(tn, show_thinking=self._show_thinking):
                log.write(line)
        log.scroll_home(animate=False)  # a conversation reads top-down
        log.focus()

    def action_reload(self) -> None:
        self._load()

    def action_toggle_thinking(self) -> None:
        self._show_thinking = not self._show_thinking
        self._load()

    def action_scroll_top(self) -> None:
        self.query_one("#conv-body", RichLog).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#conv-body", RichLog).scroll_end(animate=False)

    def action_close(self) -> None:
        self.dismiss()
