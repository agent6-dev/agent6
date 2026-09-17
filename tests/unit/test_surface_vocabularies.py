# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every name a surface has to know reaches every surface.

Two vocabularies work this way: the transcript styles `item_lines` emits, and
the status levels `status_level` picks. Each surface renders them in its own
language (Rich markup, ANSI escapes, CSS classes), so the list is spelled three
times and merging them would abstract three different string vocabularies.
Coverage is pinned instead: a new name fails until all three know it.

A name that carries no colour anywhere carries no CSS rule either, so each
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


def test_every_status_level_reaches_every_surface() -> None:
    """The second vocabulary with three spellings: `status_level` picks a level,
    the CLI paints it with SGR, the TUI with a Rich style, the web with a pill
    class. `neutral` is plain by design, so it carries no pill."""
    from agent6.ui.cli._common import _LEVEL_SGR  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.tui.theme import STATUS_LEVEL_STYLE
    from agent6.viewmodel.format import STATUS_LEVEL, StatusLevel

    levels = frozenset(get_args(StatusLevel))
    assert frozenset(STATUS_LEVEL.values()) <= levels
    assert frozenset(_LEVEL_SGR) == levels
    assert frozenset(STATUS_LEVEL_STYLE) == levels

    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    assert frozenset(re.findall(r"\.pill\.([a-z]+)", css)) >= levels - {"neutral"}


def test_a_tool_description_quotes_the_cap_it_enforces() -> None:
    """The numbers the model reads and the numbers the handlers enforce are one
    value: the descriptions interpolate them, so a changed cap cannot leave a
    stale promise in every request's schema."""
    from agent6.tools.schema import LIST_DIR_CAP, ROSTER_MAX, ListDirInput, ReadSessionInput

    assert f"{LIST_DIR_CAP:,}" in ListDirInput.TOOL_DESCRIPTION
    assert f"{ROSTER_MAX} newest" in ReadSessionInput.TOOL_DESCRIPTION
