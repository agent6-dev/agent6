# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The keys agent6 answers to, in one table.

A shortcut belongs to a surface, but the same key often means the same thing on
several: the approval letters are the CLI prompt's and the TUI row's.
Spelling them once keeps them from drifting apart, and makes a collision
visible by reading one file instead of thirty.

A leaf: no agent6 imports, no textual, so every front-end can read it. Textual
bindings are built from these tuples where they are used.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Answer:
    """One answer to an approval: the letter, the value written to the answer
    bridge, the word every surface shows for it, and the words the CLI prompt
    also takes. `standing` marks the two an approval offers only when the
    operator may answer for the whole session; `grants` the two that allow
    the call (the bridge's rule, pinned to this table by test_keymap)."""

    key: str
    answer: str
    label: str
    aliases: tuple[str, ...] = ()
    standing: bool = False
    grants: bool = False


# The order is the order they are offered in, everywhere: allow, allow all,
# deny, deny all. The CLI prompt's letters, so one keymap is learned once.
APPROVAL_ANSWERS: tuple[Answer, ...] = (
    Answer("y", "yes", "allow", ("yes",), grants=True),
    Answer("a", "session", "allow all", ("all", "always", "session"), standing=True, grants=True),
    Answer("n", "no", "deny", ("no",)),
    Answer("d", "session-deny", "deny all", ("deny", "never"), standing=True),
)


def answer_entry(answer: str) -> Answer:
    """The table's row for a bridge value; a value outside the table is a
    programming error, not an operator's."""
    return next(entry for entry in APPROVAL_ANSWERS if entry.answer == answer)


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
    scoped = ", ".join(f"{e.key} = {e.label}" for e in APPROVAL_ANSWERS if e.standing)
    return f"[{letters}]  ({scoped}, this session): "


# Every key a TUI screen binds to one of its menu actions, by screen: the
# one place a key is chosen, so a collision is read here before a key is
# taken (letters are scarce and actions are not: `d` deletes a run on the hub,
# unsets a setting on the config page and denies an approval for the session
# in a run view; a destructive letter is confirmed before it acts). A screen
# builds its bindings from its menus and this table (`menu_bindings`); an
# action absent here is reachable from the menu and the palette only. Commas
# join the aliases of one action: the first key carries the footer entry.
# The approval letters (`APPROVAL_ANSWERS`) are bound beside these on every
# view of a session.
SCREEN_KEYS: dict[str, dict[str, str]] = {
    "hub": {
        "new_work": "n",
        "open_selected": "enter",
        "merge_selected": "m",
        "delete_selected": "d",
        "refresh": "r",
        "quit": "q",
        "open_config": "c",
        "open_machines": "M",
        "view_logs": "l",
        "toggle_lanes": "space",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    # The two run views carry no letters of their own: the composer has the
    # keyboard, so their keys are modified keys and Esc.
    "conversation": {
        "close": "escape",
        "quit_hub": "ctrl+q",
        "history_search": "ctrl+r",
        "cycle_detail": "ctrl+t",
        "page_up": "pageup",
        "page_down": "pagedown",
        "scroll_top": "ctrl+home",
        "scroll_bottom": "ctrl+end",
        "copy": "ctrl+c",
        "toggle_dashboard": "ctrl+d",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "dashboard": {
        "to_hub": "escape",
        "quit_hub": "ctrl+q",
        "history_search": "ctrl+r",
        "focus_next_pane": "tab",
        "focus_prev_pane": "shift+tab",
        "page_up": "pageup",
        "page_down": "pagedown",
        "scroll_top": "ctrl+home",
        "scroll_bottom": "ctrl+end",
        "copy": "ctrl+c",
        "toggle_dashboard": "ctrl+d",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "new work": {
        "close": "escape",
        "quit_hub": "ctrl+q",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    # `l` closes the log too: the key that opened it (the hub's, the
    # dashboard's) toggles it shut.
    "event log": {
        "close": "escape,q,l",
        "page_up": "pageup",
        "page_down": "pagedown",
        "scroll_top": "ctrl+home",
        "scroll_bottom": "ctrl+end",
        "reload": "r",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "config": {
        "reload": "r",
        "close": "escape,q",
        "quit": "ctrl+q",
        "edit": "e",
        "add_provider": "a",
        "reset": "d",
        "search": "/",
        "toggle_modified": "m",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    # `r` refreshes on every screen that refreshes, so running a machine takes
    # the shifted letter, as `M` does for the machines screen itself.
    "machines": {
        "close": "escape,q",
        "quit": "ctrl+q",
        "view": "v",
        "run": "R",
        "watch": "w",
        "create": "c",
        "refresh": "r",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
    "machine watch": {
        "close": "escape,q",
        "steer": "s",
        "poke": "m",
        "stop": "x",
        "help": "question_mark",
        "command_palette": "ctrl+p",
    },
}
