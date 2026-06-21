# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Theme setup (register + apply saved + persist) and the curated picker."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from agent6.ui.settings import get_theme, save_theme
from agent6.ui.theme import ThemePicker, setup_theme
from agent6.ui.widgets import ChoiceField


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path))
    return tmp_path


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        setup_theme(self)


def test_setup_registers_applies_saved_and_persists(cfg: Path) -> None:
    save_theme("nord")

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
            assert get_theme() == "gruvbox"

    asyncio.run(scenario())


def test_unknown_saved_theme_falls_back(cfg: Path) -> None:
    save_theme("no-such-theme")

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "agent6-dark"  # invalid name -> default, no crash

    asyncio.run(scenario())


async def _select_theme(pilot: object, field: ChoiceField, target: str) -> None:
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


def test_picker_preview_on_select_and_esc_keeps(cfg: Path) -> None:
    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "agent6-dark"
            app.push_screen(ThemePicker())
            await pilot.pause()
            await _select_theme(pilot, app.screen.query_one(ChoiceField), "nord")
            assert app.theme == "nord"  # Space applies the highlighted theme live
            await pilot.press("escape")
            await pilot.pause()
            assert app.theme == "nord"  # Esc keeps the previewed theme (no restore)

    asyncio.run(scenario())


def test_picker_enter_keeps_and_persists(cfg: Path) -> None:
    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(ThemePicker())
            await pilot.pause()
            await _select_theme(pilot, app.screen.query_one(ChoiceField), "dracula")
            await pilot.press("enter")  # confirm + close
            await pilot.pause()
            assert app.theme == "dracula"
            assert get_theme() == "dracula"  # persisted

    asyncio.run(scenario())


def test_picker_backdrop_click_closes(cfg: Path) -> None:
    from textual.geometry import Offset

    async def scenario() -> None:
        app = _Host()
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ThemePicker())
            await pilot.pause()
            assert isinstance(app.screen, ThemePicker)
            await pilot.click(offset=Offset(2, 2))  # click the backdrop, outside the box
            await pilot.pause()
            assert not isinstance(app.screen, ThemePicker)  # mouse-closed, no Esc needed

    asyncio.run(scenario())
