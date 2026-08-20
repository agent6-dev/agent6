# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What every screen with a menu bar shares: the palette source over its
menus, and the actions each menu bar offers (open a menu by mnemonic, the
help page, the theme and copy-method pickers)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.screen import Screen

from agent6.ui.tui.copy_method import open_copy_method_picker
from agent6.ui.tui.menubar import HelpScreen, Menu, MenuBar
from agent6.ui.tui.theme import open_theme_picker

PaletteCommand = tuple[str, Callable[[], Any], str]  # (label, runnable, help)


def menu_palette_commands(screen: Screen[Any], menus: tuple[Menu, ...]) -> Iterator[PaletteCommand]:
    """(label, runnable, help) per menu action of *screen*, for the Ctrl+P
    palette: the same registry as the menu bar and the key bindings, so the
    surfaces never drift. The handler is the screen's, else the app's (a run
    view's Run menu resolves on the Agent6TUI host); the palette opener and
    Quit are textual's own."""
    for menu in menus:
        for item in menu.items:
            if item.action in ("command_palette", "quit"):
                continue
            handler = getattr(screen, f"action_{item.action}", None) or getattr(
                screen.app, f"action_{item.action}", None
            )
            if handler is not None:
                yield (item.label, handler, menu.title)


class MenuCommands(Provider):
    """The one Ctrl+P palette provider: hits are the screen's
    `palette_commands()` (a `ScreenChrome` screen, or any screen defining it)."""

    def _commands(self) -> Iterator[PaletteCommand]:
        source = getattr(self.screen, "palette_commands", None)
        if not callable(source):
            return iter(())
        return iter(cast(Iterator[PaletteCommand], source()))

    async def discover(self) -> Hits:
        for name, runnable, help_text in self._commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, runnable, help_text in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), runnable, help=help_text)


class ScreenChrome:
    """Mix into a Screen (before Screen in the bases) that composes a MenuBar.
    The screen declares `MENUS` (or overrides `menus()` for per-instance
    menus); `HELP_TITLE` and `HELP_HINTS` feed its help page."""

    MENUS: ClassVar[tuple[Menu, ...]] = ()
    HELP_TITLE: ClassVar[str] = "agent6 — keys & actions"
    HELP_HINTS: ClassVar[tuple[str, ...]] = ()

    def menus(self) -> tuple[Menu, ...]:
        return self.MENUS

    def palette_commands(self) -> Iterator[PaletteCommand]:
        return menu_palette_commands(cast(Screen[Any], self), self.menus())

    def action_menu(self, mnemonic: str) -> None:
        cast(Screen[Any], self).query_one(MenuBar).open(mnemonic)

    def action_help(self) -> None:
        screen = cast(Screen[Any], self)
        screen.app.push_screen(
            HelpScreen(self.menus(), screen, title=self.HELP_TITLE, hints=self.HELP_HINTS)
        )

    def action_choose_theme(self) -> None:
        open_theme_picker(cast(Screen[Any], self).app)

    def action_choose_copy_method(self) -> None:
        open_copy_method_picker(cast(Screen[Any], self).app)
