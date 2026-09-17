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
