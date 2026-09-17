# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One table for the approval letters, read by every surface that offers them.

They lived in three places (the CLI prompt, the TUI row, the TUI modal), and a
fourth surface described them wrongly in a comment. These pin the shared table
and the strings built from it, so a change lands everywhere at once.
"""

from __future__ import annotations

from importlib import resources

from agent6.ui.keymap import APPROVAL_ANSWERS, answer_for, approval_prompt_suffix


def test_the_four_answers_and_their_order() -> None:
    assert [(e.key, e.answer) for e in APPROVAL_ANSWERS] == [
        ("y", "yes"),
        ("a", "session"),
        ("n", "no"),
        ("d", "session-deny"),
    ]
    assert [e.answer for e in APPROVAL_ANSWERS if e.standing] == ["session", "session-deny"]


def test_the_cli_prompt_line_is_unchanged() -> None:
    """The line an operator has learned: the plain answers, then the scoped
    pair the parenthesis explains."""
    assert approval_prompt_suffix(standing=True) == (
        "[y/N/a/d]  (a = allow all, d = deny all, this session): "
    )
    assert approval_prompt_suffix(standing=False) == "[y/N]: "


def test_a_letter_or_its_word_answers() -> None:
    for typed in ("y", "Y", " yes ", "YES"):
        assert answer_for(typed, standing=True) == "yes"
    for typed in ("a", "all", "always", "session"):
        assert answer_for(typed, standing=True) == "session"
    for typed in ("d", "deny", "never"):
        assert answer_for(typed, standing=True) == "session-deny"


def test_anything_else_denies() -> None:
    """`[y/N]` says so: the default answer is the safe one."""
    for typed in ("n", "", "   ", "maybe", "yolo"):
        assert answer_for(typed, standing=True) == "no"


def test_a_scoped_answer_needs_a_scoped_prompt() -> None:
    """A prompt nobody may answer for the session takes `a` as the letter it
    is, which denies."""
    assert answer_for("a", standing=False) == "no"
    assert answer_for("d", standing=False) == "no"
    assert answer_for("y", standing=False) == "yes"


def test_the_web_describes_the_shared_keys_correctly() -> None:
    """The web has no approval keys, but it names the CLI's in a comment; it
    said `x`, which is the machine-watch stop key."""
    js = resources.files("agent6.ui.web").joinpath("client_run.js").read_text(encoding="utf-8")
    assert "the CLI's `x`" not in js
    deny_all = next(e for e in APPROVAL_ANSWERS if e.answer == "session-deny")
    assert f"`{deny_all.key}` at the CLI prompt" in js
