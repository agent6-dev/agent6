# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `_summarise_assistant_text_for_commit`."""

from __future__ import annotations

from agent6.workflows.loop import (
    _summarise_assistant_text_for_commit,  # pyright: ignore[reportPrivateUsage]
)


def test_empty_text_falls_back() -> None:
    assert _summarise_assistant_text_for_commit("", 3) == "agent6 iter 3: verify passed"


def test_whitespace_only_text_falls_back() -> None:
    assert _summarise_assistant_text_for_commit("\n  \n\t\n", 1) == "agent6 iter 1: verify passed"


def test_takes_first_non_empty_line() -> None:
    text = "\n\nAdded a failing test for the parser bug.\nThen fixed the parser.\n"
    out = _summarise_assistant_text_for_commit(text, 7)
    assert out == "agent6 iter 7: Added a failing test for the parser bug."


def test_strips_markdown_heading_and_bullets() -> None:
    text = "# Plan\n- step one\n- step two"
    out = _summarise_assistant_text_for_commit(text, 2)
    assert out == "agent6 iter 2: Plan"


def test_strips_leading_thinking_block() -> None:
    text = "<thinking>internal monologue here</thinking>\nFix the off-by-one in foo()."
    out = _summarise_assistant_text_for_commit(text, 4)
    assert out == "agent6 iter 4: Fix the off-by-one in foo()."


def test_truncates_long_subject() -> None:
    long = "x" * 200
    out = _summarise_assistant_text_for_commit(long, 5)
    # "agent6 iter 5: " prefix + 72 chars of body.
    assert out.startswith("agent6 iter 5: ")
    assert len(out) - len("agent6 iter 5: ") == 72


def test_unclosed_thinking_block_falls_back() -> None:
    out = _summarise_assistant_text_for_commit("<thinking>oops never closed", 9)
    assert out == "agent6 iter 9: verify passed"
