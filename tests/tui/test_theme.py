# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Theme setup (register + apply saved + persist) and the curated picker."""

from __future__ import annotations

import asyncio
import pathlib

import pytest
from textual import app as textual_app
from textual import color as textual_color
from textual import theme as textual_theme
from textual import widgets

from agent6.ui.tui import forms, settings, theme


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


class _Host(textual_app.App[None]):
    def compose(self) -> textual_app.ComposeResult:
        yield widgets.Static("host")

    def on_mount(self) -> None:
        theme.setup_theme(self)


def _luminance(rgb: tuple[int, int, int]) -> float:
    """Return the WCAG relative luminance of an sRGB colour."""
    unit = [v / 255 for v in rgb]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in unit)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Return the WCAG contrast ratio of two colours, from 1 to 21."""
    lighter, darker = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def test_a_light_theme_paints_every_ansi_colour_at_4_5_to_1(cfg: pathlib.Path) -> None:
    """An agent6 light theme's ANSI colours read at 4.5:1 or more on its background and surface.

    textual's light default painted yellow #cb9000, 2.36:1 on agent6-light's surface.
    """

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            light = [
                t
                for t in app.available_themes.values()
                if not t.dark and t.name not in textual_theme.BUILTIN_THEMES
            ]
            assert light
            low: list[str] = []
            for light_theme in light:
                app.theme = light_theme.name
                await pilot.pause()
                ansi = app.ansi_theme
                colours = {"foreground": ansi.foreground_color} | {
                    f"ANSI {n}": ansi.ansi_colors[n] for n in range(16)
                }
                variables = app.get_css_variables()
                for ground in ("background", "surface"):
                    rgb = textual_color.Color.parse(variables[ground]).rgb
                    low += [
                        f"{light_theme.name} {ground}: {name} at {_contrast(c, rgb):.2f}:1"
                        for name, c in colours.items()
                        if _contrast(c, rgb) < 4.5
                    ]
            assert not low, low

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["agent6-dark", "agent6-light"])
def test_an_agent6_theme_paints_its_text_at_4_5_to_1(cfg: pathlib.Path, name: str) -> None:
    """An agent6 theme's text colours read at 4.5:1 or more on its background and surface.

    The footer's colours are measured on the footer's own ground, the panel. agent6-light
    painted its warning at 3.86:1 on its surface; agent6-dark painted every red through
    textual's Monokai ANSI red at 4.12:1. A dark theme's ANSI black is a background (Rich
    paints code spans on it) and its bright black is dim by design, so those two are left out.
    """

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.theme = name
            await pilot.pause()
            current = app.current_theme
            variables = app.get_css_variables()
            ansi = app.ansi_theme
            parse = textual_color.Color.parse
            text: dict[str, tuple[int, int, int]] = {}
            for slot in (
                "foreground",
                "primary",
                "secondary",
                "accent",
                "success",
                "warning",
                "error",
            ):
                value = getattr(current, slot)
                assert isinstance(value, str), slot
                text[slot] = parse(value).rgb
            text["ANSI foreground"] = ansi.foreground_color
            skipped = {0, 8} if current.dark else set[int]()
            text |= {f"ANSI {n}": ansi.ansi_colors[n] for n in range(16) if n not in skipped}
            low: list[str] = []
            for ground in ("background", "surface"):
                rgb = parse(variables[ground]).rgb
                low += [
                    f"{name} {ground}: {label} at {_contrast(c, rgb):.2f}:1"
                    for label, c in text.items()
                    if _contrast(c, rgb) < 4.5
                ]
            footer_ground = parse(variables["footer-background"]).rgb
            for key in (
                "footer-foreground",
                "footer-key-foreground",
                "footer-description-foreground",
            ):
                ratio = _contrast(parse(variables[key]).rgb, footer_ground)
                if ratio < 4.5:
                    low.append(f"{name} footer: {key} at {ratio:.2f}:1")
            assert not low, low

    asyncio.run(scenario())


def test_setup_registers_applies_saved_and_persists(cfg: pathlib.Path) -> None:
    settings.save_theme("nord")

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "nord"  # the saved theme is applied on mount
            assert "agent6-dark" in app.available_themes
            assert "agent6-light" in app.available_themes
            # Any later change is persisted by the theme_changed_signal hook.
            app.theme = "gruvbox"
            await pilot.pause()
            assert settings.get_theme() == "gruvbox"

    asyncio.run(scenario())


def test_unknown_saved_theme_falls_back(cfg: pathlib.Path) -> None:
    settings.save_theme("no-such-theme")

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "agent6-dark"  # invalid name -> default, no crash

    asyncio.run(scenario())


def test_extra_builtin_themes_register_and_apply(cfg: pathlib.Path) -> None:
    """The four extra built-in themes are registered by setup_theme and each applies cleanly."""

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            for name in ("alice", "snow", "rose", "grimm"):
                assert name in app.available_themes
                app.theme = name
                await pilot.pause()
                assert app.current_theme.name == name
            assert app.available_themes["alice"].dark is False
            assert app.available_themes["snow"].dark is False
            assert app.available_themes["rose"].dark is True
            assert app.available_themes["grimm"].dark is True

    asyncio.run(scenario())


def test_saved_extra_theme_applies_on_mount(cfg: pathlib.Path) -> None:
    settings.save_theme("grimm")

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "grimm"  # parsed from ui.toml and applied

    asyncio.run(scenario())


async def _select_theme(pilot: object, field: forms.ChoiceField, target: str) -> None:
    """Highlight down to *target*, then Space to select it (applies live)."""
    for _ in range(len(field._options)):  # pyright: ignore[reportPrivateUsage]
        if field._options[field._cursor] == target:  # pyright: ignore[reportPrivateUsage]
            break
        await pilot.press("down")  # type: ignore[attr-defined]
        await pilot.pause()  # type: ignore[attr-defined]
    else:
        raise AssertionError(f"{target} not reachable in the picker")
    await pilot.press("space")  # type: ignore[attr-defined]
    await pilot.pause()  # type: ignore[attr-defined]


def test_picker_preview_on_select_and_esc_keeps(cfg: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "agent6-dark"
            app.push_screen(theme.ThemePicker())
            await pilot.pause()
            await _select_theme(pilot, app.screen.query_one(forms.ChoiceField), "nord")
            assert app.theme == "nord"  # Space applies the highlighted theme live
            await pilot.press("escape")
            await pilot.pause()
            assert app.theme == "nord"  # Esc keeps the previewed theme (no restore)

    asyncio.run(scenario())


def test_picker_enter_keeps_and_persists(cfg: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(theme.ThemePicker())
            await pilot.pause()
            await _select_theme(pilot, app.screen.query_one(forms.ChoiceField), "dracula")
            await pilot.press("enter")  # confirm + close
            await pilot.pause()
            assert app.theme == "dracula"
            assert settings.get_theme() == "dracula"  # persisted

    asyncio.run(scenario())


def test_picker_selects_extra_builtin_and_persists(cfg: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(theme.ThemePicker())
            await pilot.pause()
            await _select_theme(pilot, app.screen.query_one(forms.ChoiceField), "alice")
            assert app.theme == "alice"  # listed and applied live
            await pilot.press("enter")
            await pilot.pause()
            assert settings.get_theme() == "alice"  # persisted

    asyncio.run(scenario())


def test_picker_backdrop_click_closes(cfg: pathlib.Path) -> None:
    from textual import geometry

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            app.push_screen(theme.ThemePicker())
            await pilot.pause()
            assert isinstance(app.screen, theme.ThemePicker)
            await pilot.click(offset=geometry.Offset(2, 2))  # click the backdrop, outside the box
            await pilot.pause()
            assert not isinstance(app.screen, theme.ThemePicker)  # mouse-closed, no Esc needed

    asyncio.run(scenario())


def test_horizontal_scrollbar_thumb_is_half_height() -> None:
    """The horizontal thumb renders as a lower half-block band with quadrant end caps.

    A 1-cell horizontal bar then carries the vertical bar's visual weight; the track stays blank.
    """
    from rich.color import Color

    seg = theme.ThinScrollBarRender.render_bar(
        size=10,
        virtual_size=100,
        window_size=50,
        position=25,
        thickness=1,
        vertical=False,
        back_color=Color.parse("#111111"),
        bar_color=Color.parse("#aaaaaa"),
    )
    row = "".join(s.text for s in seg.segments if s.text != "\n")
    assert "▄" in row  # the half-height thumb body
    # No full-height cells: neither reverse-video blanks nor the default renderer's caps.
    assert not any(s.style and s.style.reverse for s in seg.segments)
    assert not any(ch in row for ch in "▉▊▋▌▍▎▏█")
    # Vertical bars keep textual's default full-cell rendering (reverse blanks).
    vseg = theme.ThinScrollBarRender.render_bar(
        size=10,
        virtual_size=100,
        window_size=50,
        position=25,
        thickness=1,
        vertical=True,
        back_color=Color.parse("#111111"),
        bar_color=Color.parse("#aaaaaa"),
    )
    assert any(s.style and s.style.reverse for s in vseg.segments)
