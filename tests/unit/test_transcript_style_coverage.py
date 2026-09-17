# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every transcript style name reaches every surface.

`StyleName` is the vocabulary `item_lines` emits; each surface renders it in
its own language (Rich markup, ANSI escapes, CSS classes), which is three
spellings of one list. Merging them would abstract three genuinely different
string vocabularies, so the list is pinned instead: a new name here fails until
all three know it.

A name with no colour anywhere carries no CSS rule either, which is why the
exemption is derived from the palettes rather than written down twice.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import get_args

from agent6.ui.cli._console_view import _STYLE_ANSI  # pyright: ignore[reportPrivateUsage]
from agent6.ui.tui.conversation import _STYLE_RICH  # pyright: ignore[reportPrivateUsage]
from agent6.viewmodel.transcript_style import StyleName

NAMES = frozenset(get_args(StyleName))
# The names every palette leaves blank: body text, painted by the surface.
UNSTYLED = frozenset(name for name, style in _STYLE_RICH.items() if not style)


def test_the_terminals_cover_the_vocabulary_exactly() -> None:
    """Both maps are `[]`-indexed while rendering, so a missing name is a
    KeyError mid-transcript and an extra one is a name nobody emits."""
    assert frozenset(_STYLE_RICH) == NAMES
    assert frozenset(_STYLE_ANSI) == NAMES


def test_the_web_styles_every_name_the_terminals_colour() -> None:
    """The client builds the class name (`'s-' + style`), so a missing rule is
    silent plain text rather than an error."""
    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    styled = frozenset(re.findall(r"\.s-([a-z-]+)", css))

    assert styled >= NAMES - UNSTYLED


def test_the_unstyled_names_are_the_same_two_everywhere() -> None:
    assert frozenset(name for name, style in _STYLE_ANSI.items() if not style) == UNSTYLED
    assert frozenset({"text", "body"}) == UNSTYLED
