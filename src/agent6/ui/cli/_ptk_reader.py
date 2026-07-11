# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""prompt_toolkit-backed inline input for the CLI (input-redesign exploration).

Rich input that stays INLINE (never the alternate screen), so the terminal's
native scrollback and copy/paste keep working:

- ``radio_select``: arrow-key single choice for ask_user options, always with a
  "type your own answer" free-text escape.
- ``ask_navigate``: arrow-key forward/back navigator over a multi-question series —
  answer in any order, go back to change one, then submit.
- ``ptk_prompt``: a line editor with slash-command completion + history
  auto-suggest, for the steer / command line.

Both return ``None`` when there is no controlling terminal, so every caller keeps
its plain fallback. The radio widget is a non-full-screen ``Application`` with
``erase_when_done`` (not ``radiolist_dialog``, which takes over the whole screen
and destroys scrollback); the caller prints the chosen answer as an ordinary line
so it lands in scrollback."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
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


def radio_select(question: str, options: Sequence[str], *, prefix: str = "") -> str | None:
    """Inline arrow-key single choice over ``options``, ALWAYS with a final "type
    your own answer" entry so the operator is never boxed into the listed choices
    (choose it to enter free text / a message to the agent). Returns the chosen
    option, the typed answer, or ``None`` (no tty / cancelled) to fall back."""
    if not options or not on_tty():
        return None
    custom_idx = len(options)
    entries = [*options, "Type your own answer..."]
    state = {"i": 0}

    def render() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [("bold", f"{prefix}{question}\n")]
        for idx, opt in enumerate(entries):
            picked = idx == state["i"]
            bullet = "●" if picked else "○"
            cls = "class:sel" if picked else ("class:custom" if idx == custom_idx else "")
            lines.append((cls, f"  {bullet} {opt}\n"))
        lines.append(("class:hint", "  ↑/↓ move · enter select · esc cancel"))
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
    @kb.add("escape")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    window = Window(FormattedTextControl(render, focusable=True), dont_extend_height=True)
    app: Application[int | None] = Application(
        layout=Layout(window),
        key_bindings=kb,
        style=Style.from_dict(
            {"sel": "reverse", "custom": "italic fg:#8888ff", "hint": "italic fg:#888888"}
        ),
        full_screen=False,  # inline: scrollback preserved
        erase_when_done=True,  # drop the widget; the caller prints the chosen line
        mouse_support=False,
    )
    idx = app.run()
    if idx is None:
        return None
    if idx == custom_idx:  # "type your own" -> a free-text line editor
        typed = ptk_prompt(f"{prefix}{question}\n> ")
        return typed.strip() if typed is not None else None
    return options[idx]


def ask_navigate(questions: Sequence[str], answer: Callable[[int], str | None]) -> list[str] | None:
    """Inline forward/back navigator over a question series: one "question -> answer"
    row per question plus a submit row. ↑/↓ move between questions FREELY (answer in
    any order, go back to change one); enter answers the selected question (via
    ``answer(i)``, which opens that question's radio / line editor) and auto-advances
    to the next unanswered; the ✓ submit row (or esc) submits. Returns the answers,
    or ``None`` with no tty so the caller keeps the plain numbered fallback."""
    if not questions or not on_tty():
        return None
    n = len(questions)
    answers = ["" for _ in questions]
    answered = [False] * n
    sel = {"i": 0}

    def render() -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [
            ("bold", "Answer these (↑/↓ move · enter to answer · esc/✓ submit):\n")
        ]
        for i, q in enumerate(questions):
            picked = i == sel["i"]
            mark = "▸" if picked else " "
            shown = answers[i] if answered[i] else "(not answered)"
            cls = "class:sel" if picked else ("" if answered[i] else "class:todo")
            lines.append((cls, f"  {mark} {q}  →  {shown}\n"))
        at_submit = sel["i"] == n
        lines.append(
            ("class:ok" if at_submit else "class:hint", f"  {'▸' if at_submit else ' '} ✓ submit\n")
        )
        lines.append(("class:hint", "  ↑/↓ move · enter answer/submit · esc submit as-is"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event: Any) -> None:
        sel["i"] = (sel["i"] - 1) % (n + 1)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event: Any) -> None:
        sel["i"] = (sel["i"] + 1) % (n + 1)

    @kb.add("enter")
    def _pick(event: Any) -> None:
        event.app.exit(result=sel["i"])

    @kb.add("c-c")
    @kb.add("c-d")
    @kb.add("escape")
    def _submit(event: Any) -> None:
        event.app.exit(result=n)

    style = Style.from_dict(
        {
            "sel": "reverse",
            "todo": "fg:#cc8800",
            "ok": "bold fg:#22aa22",
            "hint": "italic fg:#888888",
        }
    )
    while True:
        window = Window(FormattedTextControl(render, focusable=True), dont_extend_height=True)
        app: Application[int] = Application(
            layout=Layout(window),
            key_bindings=kb,
            style=style,
            full_screen=False,
            erase_when_done=True,
            mouse_support=False,
        )
        chosen = app.run()
        if chosen >= n:  # the submit row (or esc/ctrl-c/ctrl-d, which submit as-is)
            return answers
        new = answer(chosen)
        if new is not None:
            answers[chosen] = new
            answered[chosen] = True
            sel["i"] = next((j for j in range(n) if not answered[j]), n)  # next unanswered / submit


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
