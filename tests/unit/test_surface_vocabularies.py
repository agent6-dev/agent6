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

from agent6.ui.cli import _console_view  # pyright: ignore[reportPrivateUsage]
from agent6.ui.tui import conversation  # pyright: ignore[reportPrivateUsage]
from agent6.viewmodel import transcript_style

NAMES = frozenset(get_args(transcript_style.StyleName))
# The names every palette leaves blank: body text, painted by the surface.
UNSTYLED = frozenset(name for name, style in conversation._STYLE_RICH.items() if not style)


def test_the_terminals_cover_the_vocabulary_exactly() -> None:
    """Both maps are indexed while rendering, so a missing name is a KeyError mid-transcript."""
    assert frozenset(conversation._STYLE_RICH) == NAMES
    assert frozenset(_console_view._STYLE_ANSI) == NAMES


def test_the_web_styles_every_name_the_terminals_colour() -> None:
    """The client builds the class name from the style, so a missing rule is silent plain text."""
    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    styled = frozenset(re.findall(r"\.s-([a-z-]+)", css))

    assert styled >= NAMES - UNSTYLED


def test_the_unstyled_names_are_the_same_two_everywhere() -> None:
    assert (
        frozenset(name for name, style in _console_view._STYLE_ANSI.items() if not style)
        == UNSTYLED
    )
    assert frozenset({"text", "body"}) == UNSTYLED


def test_every_status_level_reaches_every_surface() -> None:
    """The status level has three spellings: SGR, a Rich style and a pill class.

    `neutral` is plain by design, so it carries no pill.
    """
    from agent6.ui.cli import _common  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.tui import theme
    from agent6.viewmodel import format

    levels = frozenset(get_args(format.StatusLevel))
    assert frozenset(format.STATUS_LEVEL.values()) <= levels
    assert frozenset(_common._LEVEL_SGR) == levels
    assert frozenset(theme.STATUS_LEVEL_STYLE) == levels

    css = resources.files("agent6.ui.web").joinpath("styles.css").read_text(encoding="utf-8")
    assert frozenset(re.findall(r"\.pill\.([a-z]+)", css)) >= levels - {"neutral"}


def test_a_tool_description_quotes_the_cap_it_enforces() -> None:
    """The descriptions interpolate the caps the handlers enforce, so a cap cannot go stale."""
    from agent6.tools import schema

    assert f"{schema.LIST_DIR_CAP:,}" in schema.ListDirInput.TOOL_DESCRIPTION
    assert f"{schema.ROSTER_MAX} newest" in schema.ReadSessionInput.TOOL_DESCRIPTION
