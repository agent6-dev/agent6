# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The page's highlighter: spans per line from the tree-sitter language pack, bounded."""

from __future__ import annotations

import time

import pytest

from agent6.ui.web import highlight

_PY = 'def f(x):\n    return "hi" + str(x)  # c\n'


def test_a_python_sample_yields_spans_at_their_columns() -> None:
    """A keyword, a string, a comment and a call land on their columns.

    `f` is a function, the more specific capture, not the variable the generic pattern
    also names.
    """
    assert highlight.language_of("pkg/mod.py") == "python"
    rows = highlight.spans(_PY, "python")
    assert len(rows) == 2
    assert (0, 3, "keyword") in rows[0] and (4, 5, "function") in rows[0]
    assert (6, 7, "variable") in rows[0]
    assert (4, 10, "keyword") in rows[1] and (11, 15, "string") in rows[1]
    assert (18, 21, "function") in rows[1] and (26, 29, "comment") in rows[1]
    for row in rows:
        assert all(a < b for a, b, _ in row)
        assert all(row[i][1] <= row[i + 1][0] for i in range(len(row) - 1))


def test_columns_count_utf16_units_as_the_page_slices_strings() -> None:
    rows = highlight.spans('s = "é"  # ü\n', "python")
    assert (4, 7, "string") in rows[0] and (9, 12, "comment") in rows[0]


def test_nothing_is_highlighted_without_a_language_or_over_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert highlight.language_of("README") == ""
    assert highlight.language_of("notes.txt") == ""
    assert highlight.spans(_PY, "") == []
    assert highlight.spans(_PY, "no-such-language") == []
    monkeypatch.setattr(highlight, "HIGHLIGHT_MAX_BYTES", 10)
    assert highlight.spans(_PY, "python") == []


def test_a_parse_past_the_deadline_yields_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text never waits on the highlighter past `TIMEOUT_S`."""

    def slow(text: str, language: str) -> list[list[highlight.Span]]:
        time.sleep(0.3)
        return [[(0, 1, "keyword")]]

    monkeypatch.setattr(highlight, "_compute", slow)
    monkeypatch.setattr(highlight, "TIMEOUT_S", 0.05)
    assert highlight.spans(_PY, "python") == []
    monkeypatch.setattr(highlight, "TIMEOUT_S", 2.0)
    assert highlight.spans(_PY, "python") == [[(0, 1, "keyword")]]
