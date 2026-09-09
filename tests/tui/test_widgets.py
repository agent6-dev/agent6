# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared form widgets keep their keyboard and mouse state visible."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from agent6.ui.tui.widgets import ChoiceField, TypeaheadField


class _ChoiceScrollHost(App[None]):
    CSS = "#scroll { height: 6; width: 40; } #before { height: 10; }"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroll"):
            yield Static("before", id="before")
            yield ChoiceField(tuple(f"option-{i}" for i in range(12)), "option-0")

    def on_mount(self) -> None:
        self.query_one(ChoiceField).focus()


def test_choice_cursor_stays_visible_below_prior_content() -> None:
    async def scenario() -> None:
        app = _ChoiceScrollHost()
        async with app.run_test(size=(50, 15)) as pilot:
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            field = app.query_one(ChoiceField)
            scroll = app.query_one("#scroll", VerticalScroll)
            cursor_y = field.content_region.y + field._cursor  # pyright: ignore[reportPrivateUsage]
            assert (
                scroll.scrollable_content_region.y
                <= cursor_y
                < scroll.scrollable_content_region.bottom
            )

    asyncio.run(scenario())


class _ChoiceWidthHost(App[None]):
    CSS = "ChoiceField { width: 12; }"

    def compose(self) -> ComposeResult:
        yield ChoiceField(("abcdefghijklmnopqrstuvwxyz", "second"), "abcdefghijklmnopqrstuvwxyz")


def test_choice_options_each_use_one_screen_row() -> None:
    async def scenario() -> None:
        app = _ChoiceWidthHost()
        async with app.run_test(size=(30, 10)) as pilot:
            await pilot.pause()
            field = app.query_one(ChoiceField)
            assert field.region.height == field._row_count  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


class _ChoiceChangeHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.changes = 0

    def compose(self) -> ComposeResult:
        yield ChoiceField(("first", "second"), "first")

    def on_mount(self) -> None:
        self.query_one(ChoiceField).focus()

    def on_choice_field_changed(self) -> None:
        self.changes += 1


def test_choice_posts_changed_only_when_the_selection_changes() -> None:
    async def scenario() -> None:
        app = _ChoiceChangeHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space")
            await pilot.pause()
            assert app.changes == 0
            await pilot.press("down", "space")
            await pilot.pause()
            assert app.query_one(ChoiceField).value == "second"
            assert app.changes == 1

    asyncio.run(scenario())


class _ChoiceCustomHost(App[None]):
    def compose(self) -> ComposeResult:
        yield ChoiceField(("fixed",), "abc", allow_custom=True)

    def on_mount(self) -> None:
        self.query_one(ChoiceField).focus()


def test_choice_custom_caret_follows_left_and_right() -> None:
    async def scenario() -> None:
        app = _ChoiceCustomHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            field = app.query_one(ChoiceField)
            await pilot.press("left")
            await pilot.pause()
            assert "[x] ab▌c" in field.render().plain
            await pilot.press("X")
            await pilot.pause()
            assert field.value == "abXc"
            assert "[x] abX▌c" in field.render().plain
            await pilot.press("right")
            await pilot.pause()
            assert "[x] abXc▌" in field.render().plain

    asyncio.run(scenario())


class _TypeaheadScrollHost(App[None]):
    CSS = "#scroll { height: 6; width: 40; } #before { height: 10; }"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroll"):
            yield Static("before", id="before")
            yield TypeaheadField("", [f"option-{i}" for i in range(10)])

    def on_mount(self) -> None:
        self.query_one(TypeaheadField).focus()


def test_typeahead_highlight_stays_visible_when_suggestions_expand() -> None:
    async def scenario() -> None:
        app = _TypeaheadScrollHost()
        async with app.run_test(size=(50, 15)) as pilot:
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            scroll = app.query_one("#scroll", VerticalScroll)
            highlight_y = (
                field.content_region.y + field._index + 1  # pyright: ignore[reportPrivateUsage]
            )
            assert (
                scroll.scrollable_content_region.y
                <= highlight_y
                < scroll.scrollable_content_region.bottom
            )

    asyncio.run(scenario())


class _TypeaheadWidthHost(App[None]):
    CSS = "TypeaheadField { width: 12; }"

    def compose(self) -> ComposeResult:
        yield TypeaheadField("abcdefghijklmnopqrstuvwxyz", ["one"])

    def on_mount(self) -> None:
        self.query_one(TypeaheadField).focus()


def test_typeahead_text_line_never_wraps() -> None:
    async def scenario() -> None:
        app = _TypeaheadWidthHost()
        async with app.run_test(size=(30, 10)) as pilot:
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            assert field.region.height == 2  # text line plus one suggestion

    asyncio.run(scenario())


def test_typeahead_caret_follows_left_and_right() -> None:
    async def scenario() -> None:
        app = _TypeaheadRefreshHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            await pilot.press("left")
            await pilot.pause()
            assert field.render().plain.startswith("curren▌t\n")
            await pilot.press("right")
            await pilot.pause()
            assert field.render().plain.startswith("current▌\n")

    asyncio.run(scenario())


class _TypeaheadRefreshHost(App[None]):
    def compose(self) -> ComposeResult:
        yield TypeaheadField("current", ["alpha", "beta"])

    def on_mount(self) -> None:
        self.query_one(TypeaheadField).focus()


def test_typeahead_live_refresh_preserves_highlighted_value() -> None:
    async def scenario() -> None:
        app = _TypeaheadRefreshHost()
        async with app.run_test() as pilot:
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            await pilot.press("down")
            assert field.value == "alpha"
            field.set_suggestions(["zeta", "alpha", "beta"])
            await pilot.pause()
            assert field.value == "alpha"

    asyncio.run(scenario())


def test_a_value_longer_than_the_row_scrolls_under_the_caret() -> None:
    """Clipping the row to one line (instead of wrapping) cut the caret away
    with the tail of a long value, so a focused field read as dead; the row
    shows the slice around the cursor."""
    import asyncio

    from textual.app import App
    from textual.geometry import Region

    value = "anthropic/claude-opus-4-1"

    class Host(App[None]):
        CSS = "TypeaheadField { width: 16; }"

        def compose(self) -> ComposeResult:
            yield TypeaheadField(value, ["one"])

        def on_mount(self) -> None:
            self.query_one(TypeaheadField).focus()

    async def scenario() -> list[str]:
        app = Host()
        async with app.run_test(size=(40, 12)) as pilot:
            await pilot.pause()
            field = app.query_one(TypeaheadField)
            await pilot.press("left")
            await pilot.pause()
            strips = field.render_lines(Region(0, 0, field.size.width, field.size.height))
            return ["".join(segment.text for segment in strip) for strip in strips]

    rows = asyncio.run(scenario())
    assert any("\u258c" in row for row in rows), rows
    assert all(len(row) <= 16 for row in rows), rows
