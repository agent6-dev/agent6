# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The keys agent6 answers to, in one table.

A shortcut belongs to a surface, but the same key often means the same thing on
several: the approval letters are the CLI prompt's, the TUI row's and the TUI
modal's. Spelling them once keeps them from drifting apart, and makes a
collision visible by reading one file instead of thirty.

A leaf: no agent6 imports, no textual, so every front-end can read it. Textual
bindings are built from these tuples where they are used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Answer:
    """One answer to an approval: the letter, the value written to the answer
    bridge, what every surface calls it, and the words the CLI prompt also
    takes. `standing` marks the two an approval offers only when the operator
    may answer for the whole session."""

    key: str
    answer: str
    label: str
    aliases: tuple[str, ...] = ()
    standing: bool = False


# The order is the order they are offered in, everywhere: allow, allow all,
# deny, deny all. The CLI prompt's letters, so one keymap is learned once.
APPROVAL_ANSWERS: tuple[Answer, ...] = (
    Answer("y", "yes", "allow", ("yes",)),
    Answer("a", "session", "allow all (session)", ("all", "always", "session"), standing=True),
    Answer("n", "no", "deny", ("no",)),
    Answer("d", "session-deny", "deny all", ("deny", "never"), standing=True),
)


def answer_for(typed: str, *, standing: bool) -> str:
    """The answer a typed line means: the letter or any of its aliases, else
    "no". A session answer needs a standing prompt; on any other it is the
    letter it is, which denies, as `[y/N]` says."""
    word = typed.strip().lower()
    for entry in APPROVAL_ANSWERS:
        if word in (entry.key, *entry.aliases) and (standing or not entry.standing):
            return entry.answer
    return "no"


def approval_prompt_suffix(*, standing: bool) -> str:
    """The `[y/N/a/d]` line the CLI prompt ends with, built from the table so a
    new answer cannot appear in one place and not the other."""
    if not standing:
        return "[y/N]: "
    # The two plain answers first, then the scoped pair the line goes on to
    # explain: `[y/N/a/d]`, not the table's own allow/allow-all order.
    ordered = [e for e in APPROVAL_ANSWERS if not e.standing] + [
        e for e in APPROVAL_ANSWERS if e.standing
    ]
    letters = "/".join(e.key.upper() if e.answer == "no" else e.key for e in ordered)
    scoped = ", ".join(
        f"{e.key} = {e.label.removesuffix(' (session)')}" for e in APPROVAL_ANSWERS if e.standing
    )
    return f"[{letters}]  ({scoped}, this session): "


# The scroll keys every scrollable screen carries, as (key, action, label). The
# three screens that scroll a long body (the conversation, the dashboard, the
# event log) offer exactly these, so they are spelled once.
SCROLL_KEYS: tuple[tuple[str, str, str], ...] = (
    ("pageup", "page_up", "Scroll up"),
    ("pagedown", "page_down", "Scroll down"),
    ("ctrl+home", "scroll_top", "Top"),
    ("ctrl+end", "scroll_bottom", "End"),
)

# The control keys a live run's two views share. They differ only in where Esc
# goes and in the conversation's detail cycle, which each screen adds itself.
RUN_VIEW_KEYS: tuple[tuple[str, str, str], ...] = (
    ("ctrl+c", "copy", "Copy"),
    ("ctrl+r", "history_search", "History"),
)


# Every plain letter a TUI screen binds, by screen. Letters are scarce and
# actions are not, so one letter means different things on different screens:
# `d` deletes a run on the hub, unsets a setting on the config page and denies
# an approval for the session in a run view. This table is where that is
# visible; `tests/tui/test_keymap_screens.py` reads the screens' own BINDINGS
# and fails if it drifts, so a new letter is chosen with the others in sight.
#
# A destructive letter is confirmed before it acts (the hub's `d` asks; the
# config page's `d` is a config-file edit you undo by setting the value again).
SCREEN_LETTERS: dict[str, dict[str, str]] = {
    "hub": {
        "n": "new_work",
        "l": "view_logs",
        "m": "merge_selected",
        "d": "delete_selected",
        "r": "refresh",
        "c": "open_config",
        "M": "open_machines",
        "q": "quit",
    },
    # The two run views carry no letters of their own: the composer has the
    # keyboard, and these four answer an open approval from any non-text focus.
    "conversation": {
        "y": "answer('yes')",
        "a": "answer('session')",
        "n": "answer('no')",
        "d": "answer('session-deny')",
    },
    "dashboard": {
        "y": "answer('yes')",
        "a": "answer('session')",
        "n": "answer('no')",
        "d": "answer('session-deny')",
    },
    "event log": {"q": "close", "l": "close", "r": "reload"},
    "config": {
        "m": "toggle_modified",
        "e": "edit",
        "a": "add_provider",
        "d": "reset",
        "r": "reload",
        "q": "close",
    },
    "machines": {
        "v": "view",
        "r": "run",
        "w": "watch",
        "c": "create",
        "f": "refresh",
        "q": "close",
    },
    "machine": {"q": "close"},
    "machine watch": {"s": "steer", "m": "poke", "x": "stop", "q": "close"},
}
