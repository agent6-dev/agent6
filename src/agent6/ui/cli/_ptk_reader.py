# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""prompt_toolkit-backed inline input for the CLI (input-redesign exploration).

Rich input that stays INLINE (never the alternate screen), so the terminal's
native scrollback and copy/paste keep working:

- ``radio_select``: arrow-key single choice for ask_user options, by default with
  a "type your own answer" free-text escape. With ``allow_back`` it also reports
  series navigation (``RadioNav.BACK`` / ``RadioNav.SKIP``) so a caller can walk a
  question series one question at a time; that series flow lives in
  ``_interact._ask_series_tty``.
- ``ptk_prompt``: a line editor with slash-command completion + history
  auto-suggest, for the steer / command line.

Off a tty ``radio_select`` returns ``RadioNav.CANCEL`` and ``ptk_prompt`` returns
``None``, so every caller keeps its plain fallback. The radio widget is a
non-full-screen ``Application`` with ``erase_when_done`` (not ``radiolist_dialog``,
which takes over the whole screen and destroys scrollback); the caller prints the
chosen answer as an ordinary line so it lands in scrollback."""

from __future__ import annotations

import enum
import sys
from collections.abc import Sequence
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style


def on_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


class RadioNav(enum.Enum):
    """Non-answer outcomes of ``radio_select``."""

    BACK = enum.auto()  # left arrow: go to the previous question
    SKIP = enum.auto()  # esc in a series: leave this question unanswered, move on
    CANCEL = enum.auto()  # Ctrl-C (or esc outside a series, or no tty): abandon


_FREE_TEXT = "Type your own answer..."


def radio_select(
    question: str,
    options: Sequence[str],
    *,
    prefix: str = "",
    initial: int = 0,
    allow_back: bool = False,
    free_text: bool = True,
) -> str | RadioNav:
    """Inline arrow-key single choice over ``options``. With ``free_text`` (the
    default) a final "type your own answer" entry opens a line editor so the
    operator is never boxed into the listed choices; backing out of it (Ctrl-C /
    Ctrl-D) returns to the radio rather than cancelling. ``initial`` preselects an
    entry (revisiting an answered question). With ``allow_back`` the widget is part
    of a series: left returns ``RadioNav.BACK`` and esc ``RadioNav.SKIP``; otherwise
    esc cancels. Returns the chosen or typed answer, or a ``RadioNav``
    (``CANCEL`` when there is no tty, so the caller keeps its plain fallback)."""
    if not options or not on_tty():
        return RadioNav.CANCEL
    entries = [*options, _FREE_TEXT] if free_text else list(options)
    custom_idx = len(options) if free_text else None
    state = {"i": min(max(initial, 0), len(entries) - 1)}
    hint = (
        "  ↑/↓ move · enter select · ← back · esc skip"
        if allow_back
        else "  ↑/↓ move · enter select · esc cancel"
    )

    def render() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [("bold", f"{prefix}{question}\n")]
        for idx, opt in enumerate(entries):
            picked = idx == state["i"]
            bullet = "●" if picked else "○"
            cls = "class:sel" if picked else ("class:custom" if idx == custom_idx else "")
            lines.append((cls, f"  {bullet} {opt}\n"))
        lines.append(("class:hint", hint))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event: Any) -> None:
        state["i"] = (state["i"] - 1) % len(entries)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event: Any) -> None:
        state["i"] = (state["i"] + 1) % len(entries)

    @kb.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit(result=state["i"])

    @kb.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=RadioNav.CANCEL)

    @kb.add("escape")
    def _skip(event: Any) -> None:
        event.app.exit(result=RadioNav.SKIP if allow_back else RadioNav.CANCEL)

    if allow_back:

        @kb.add("left")
        def _back(event: Any) -> None:
            event.app.exit(result=RadioNav.BACK)

    style = Style.from_dict(
        {"sel": "reverse", "custom": "italic fg:#8888ff", "hint": "italic fg:#888888"}
    )
    while True:
        window = Window(FormattedTextControl(render, focusable=True), dont_extend_height=True)
        app: Application[int | RadioNav] = Application(
            layout=Layout(window),
            key_bindings=kb,
            style=style,
            full_screen=False,  # inline: scrollback preserved
            erase_when_done=True,  # drop the widget; the caller prints the chosen line
            mouse_support=False,
        )
        got = app.run()
        if isinstance(got, RadioNav):
            return got
        if got == custom_idx:  # "type your own" -> a free-text line editor
            typed = ptk_prompt(f"{prefix}{question}\n> ")
            if typed is None:  # backed out of the editor: re-show the radio
                continue
            return typed.strip()
        return options[got]


def ptk_prompt(
    text: str,
    *,
    options: Sequence[str] = (),
    history: Sequence[str] = (),
    interrupt_aborts: bool = False,
) -> str | None:
    """Inline line editor with tab-completion over ``options`` (slash-commands)
    and history auto-suggest. Returns the line, or ``None`` if there is no tty.

    Ctrl-D (EOF) returns ``None``. Ctrl-C returns ``None`` by default; with
    ``interrupt_aborts`` it re-raises KeyboardInterrupt instead -- the steer prompt
    uses that so Ctrl-C aborts the run rather than being swallowed as a no-op."""
    if not on_tty():
        return None
    hist = InMemoryHistory()
    for h in history:
        hist.append_string(h)
    session: PromptSession[str] = PromptSession(history=hist)
    completer = WordCompleter(list(options), sentence=True) if options else None
    try:
        return session.prompt(
            text,
            completer=completer,
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
        )
    except KeyboardInterrupt:
        if interrupt_aborts:
            raise
        return None
    except EOFError:
        return None
