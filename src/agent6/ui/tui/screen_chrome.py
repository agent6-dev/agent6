# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What every screen with a menu bar shares.

The palette source over its menus, and the actions each menu bar offers: open a
menu by mnemonic, the help page, the theme and copy-method pickers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.screen import Screen
from textual.widgets import Select

from agent6.ui.tui.copy_method import open_copy_method_picker
from agent6.ui.tui.menubar import HelpScreen, Menu, MenuBar
from agent6.ui.tui.theme import open_theme_picker

PaletteCommand = tuple[str, Callable[[], Any], str]  # (label, runnable, help)


def menu_palette_commands(screen: Screen[Any], menus: tuple[Menu, ...]) -> Iterator[PaletteCommand]:
    """Yield the Ctrl+P palette commands for a screen's menus.

    The same registry as the menu bar and the key bindings, so the surfaces never
    drift. The handler is the screen's, else the app's; the palette opener and
    Quit are textual's own.

    Args:
        screen: The screen whose actions the handlers resolve on.
        menus: The screen's menus.

    Yields:
        A label, runnable and help text per menu action.
    """
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
    """The one Ctrl+P palette provider; hits are the screen's `palette_commands()`."""

    def _commands(self) -> Iterator[PaletteCommand]:
        source = getattr(self.screen, "palette_commands", None)
        if not callable(source):
            return iter(())
        return iter(cast(Iterator[PaletteCommand], source()))

    async def discover(self) -> Hits:
        """Yield every command for the empty query."""
        for name, runnable, help_text in self._commands():
            yield DiscoveryHit(name, runnable, help=help_text)

    async def search(self, query: str) -> Hits:
        """Yield the commands whose label matches the query, scored."""
        matcher = self.matcher(query)
        for name, runnable, help_text in self._commands():
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), runnable, help=help_text)


class ScreenChrome:
    """The mixin for a screen that composes a `MenuBar`; listed before `Screen` in the bases.

    The screen declares `MENUS` or overrides `menus()` for per-instance menus;
    `HELP_TITLE` and `HELP_HINTS` feed its help page.
    """

    MENUS: ClassVar[tuple[Menu, ...]] = ()
    HELP_TITLE: ClassVar[str] = "agent6 — keys & actions"
    HELP_HINTS: ClassVar[tuple[str, ...]] = ()

    def menus(self) -> tuple[Menu, ...]:
        """Return the screen's menus."""
        return self.MENUS

    def palette_commands(self) -> Iterator[PaletteCommand]:
        """Return the palette commands over the screen's menus."""
        return menu_palette_commands(cast(Screen[Any], self), self.menus())

    def close_open_list(self) -> bool:
        """Close the open menu or dropdown list, if any.

        A screen's Esc binding fires before the list's own, so Back calls this
        first and leaves only when nothing closed.

        Returns:
            Whether one was open.
        """
        screen = cast(Screen[Any], self)
        bar = screen.query_one(MenuBar)
        if bar.opened:
            bar.close_menu()
            return True
        for select in screen.query(Select):
            if select.expanded:
                select.expanded = False
                select.focus()
                return True
        return False

    def action_menu(self, mnemonic: str) -> None:
        """Open the menu with the mnemonic."""
        cast(Screen[Any], self).query_one(MenuBar).open(mnemonic)

    def action_help(self) -> None:
        """Push the help page."""
        screen = cast(Screen[Any], self)
        screen.app.push_screen(
            HelpScreen(self.menus(), screen, title=self.HELP_TITLE, hints=self.HELP_HINTS)
        )

    def action_choose_theme(self) -> None:
        """Push the theme picker."""
        open_theme_picker(cast(Screen[Any], self).app)

    def action_choose_copy_method(self) -> None:
        """Push the copy-method picker."""
        open_copy_method_picker(cast(Screen[Any], self).app)
